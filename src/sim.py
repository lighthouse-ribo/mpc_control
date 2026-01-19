import threading
import time
from typing import List, Dict, Literal
import numpy as np

# [修改] 导入参数和新控制器
from src.vehicle import VehicleParams
from src.control.controller import MainController

# 导入物理模型相关
from src.vehicle.model import SimState, Control, TrackSettings
from src.vehicle.twodof import derivatives as deriv_2dof
from src.vehicle.dof_utils import body_to_world_2dof, curvature_4ws

# 3DOF 模块
from src.vehicle.threedof import (
    Vehicle3DOF,
    State3DOF,
    allocate_drive,
    derivatives_dfdr,
)

class SimEngine:
    """后端仿真引擎：维护状态、轨迹与控制，并在后台线程中积分。"""
    def __init__(self, params: VehicleParams, dt: float = 0.02):
        self.params = params
        self.dt = float(dt)
        
        # 状态初始化
        self.state2 = SimState()
        self.state3 = State3DOF(vx=params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
        self.ctrl = Control(U=params.U)
        
        # 动态扩展 Control 以支持前馈加速度
        if not hasattr(self.ctrl, 'ax_des'):
            self.ctrl.ax_des = 0.0

        self.track: List[Dict[str, float]] = [] 
        self.track_cfg = TrackSettings()
        self.mode: Literal['2dof', '3dof'] = '2dof'
        self._sim_t = 0.0

        self.running = False
        self._alive = True
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._loop, name="SimLoop", daemon=True)
        self._thread.start()

        # --- [核心修改] 控制器初始化 ---
        # 统一管理 PID 和 MPC，不再在此处保留散乱的控制参数
        self.controller = MainController(params, dt=self.dt)

        # 物理与限制参数
        self.delta_max = np.deg2rad(30.0)
        self.U_switch = 8.0 
        self._actual_speed_2dof = float(params.U)
        
        # 3DOF 专用参数 (保留用于非线性动力学模型)
        self.k_v = 0.8
        self.tau_ctrl = 0.15
        self.yaw_damp = 220.0
        self.yaw_sat_gain = 3.0
        self._U_cmd_filt = float(self.ctrl.U)
        self.drive_bias_front = 0.1
        self.drive_bias_rear = 0.9
        self.tau_low = 0.25
        self.tau_beta = 0.35

        # 遥测状态
        self._df_cur = 0.0
        self._dr_cur = 0.0
        self._df_dot = 0.0
        self._dr_dot = 0.0
        self._speed = float(self.params.U)
        self._radius: float | None = None
        self._beta_dot = 0.0
        self._r_dot = 0.0

        # 规划状态
        self.plan: List[Dict[str, float]] = []
        self.autop_enabled: bool = False
        self.autop_mode: Literal['mpc'] = 'mpc'
        self.goal_pose_end: Dict[str, float] | None = None
        
        # 内部状态：规划索引
        self._plan_idx = 0

    def _loop(self):
        """仿真主循环"""
        while self._alive:
            start = time.perf_counter()
            if self.running:
                with self._lock:
                    self._step(self.dt)
            # 保持实时性
            spent = time.perf_counter() - start
            sleep = max(0.0, self.dt - spent)
            time.sleep(sleep)

    def reset(self):
        """重置仿真状态"""
        with self._lock:
            self.state2 = SimState()
            self.state3 = State3DOF(vx=self.params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
            self.track.clear()
            self.running = False
            self._sim_t = 0.0
            
            # [修改] 调用控制器重置
            self.controller.reset()
            
            self._actual_speed_2dof = float(self.params.U)
            self.ctrl.ax_des = 0.0
            self._U_cmd_filt = float(self.params.U)

    def _step(self, dt: float):
        """单步仿真更新 (物理 + 控制)"""
        # 0. 获取当前状态快照
        current_state = self.get_state()
        current_ctrl = self.get_ctrl()

        # 1. 计算控制指令 (调用 MainController)
        ax_cmd = 0.0
        if self.autop_enabled and len(self.plan) > 0:
            # [核心修改] 统一获取横纵向指令
            control_out = self.controller.compute(current_state, self.plan, current_ctrl)
            
            # 应用指令
            self.ctrl.delta_f = control_out['df']
            self.ctrl.delta_r = control_out['dr']
            ax_cmd = control_out['ax']
            
            # 同步目标速度 (便于前端显示)
            if 'target_v' in control_out:
                self.ctrl.U = control_out['target_v']

        # 2. 物理积分
        if self.mode == '2dof':
            # --- [2DOF] 纵向动力学 ---
            v_curr = self._actual_speed_2dof
            
            if self.autop_enabled:
                # 自动模式：使用 Controller 计算的加速度 (PID输出)
                v_next = v_curr + ax_cmd * dt
            else:
                # 手动模式：简单的 P 控制模拟电机响应
                v_target = float(self.ctrl.U)
                ax_manual = 2.0 * (v_target - v_curr) + getattr(self.ctrl, 'ax_des', 0.0)
                v_next = v_curr + ax_manual * dt
            
            v_next = max(0.0, v_next)
            self._actual_speed_2dof = v_next
            self.params.U = v_next  # 同步参数

            # --- [2DOF] 横向动力学 ---
            x_vec = np.array([self.state2.beta, self.state2.r], dtype=float)
            d = deriv_2dof(x_vec, self.ctrl.delta_f, self.ctrl.delta_r, self.params)
            beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])

            U_signed = v_next
            psi_dot = self.state2.r
            x_dot, y_dot = body_to_world_2dof(U_signed, self.state2.beta, self.state2.psi)

            # 低速模型融合 (避免低速奇点)
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t_blend = max(0.0, min(1.0, U_signed / U_blend))
            w = t_blend * t_blend * (3.0 - 2.0 * t_blend)
            
            kappa = curvature_4ws(float(self.ctrl.delta_f), float(self.ctrl.delta_r), self.params.L)
            r_des = U_signed * kappa
            r_dot_kin = (r_des - self.state2.r) / max(1e-6, self.tau_low)
            beta_dot_kin = - self.state2.beta / max(1e-6, self.tau_beta)
            
            beta_dot = w * beta_dot + (1.0 - w) * beta_dot_kin
            r_dot = w * r_dot + (1.0 - w) * r_dot_kin

            # 更新状态
            self._beta_dot = float(beta_dot)
            self._r_dot = float(r_dot)
            self.state2.beta += beta_dot * dt
            self.state2.r += r_dot * dt
            self.state2.psi += psi_dot * dt
            self.state2.x += x_dot * dt
            self.state2.y += y_dot * dt

            # 遥测更新
            df_now = float(self.ctrl.delta_f)
            dr_now = float(self.ctrl.delta_r)
            self._df_dot = (df_now - self._df_cur) / dt
            self._dr_dot = (dr_now - self._dr_cur) / dt
            self._df_cur = df_now
            self._dr_cur = dr_now
            self._speed = v_next
            self._radius = (v_next / abs(self.state2.r)) if abs(self.state2.r) > 1e-6 else None
            self._sim_t += dt
            if self.track_cfg.enabled:
                self._push_track_point(self.state2.x, self.state2.y)

        else:
            # --- [3DOF] 动力学 ---
            vp3 = Vehicle3DOF(
                m=self.params.m, Iz=self.params.Iz, a=self.params.a, b=self.params.b, g=self.params.g,
                U_min=self.params.U_min, kf=self.params.kf, kr=self.params.kr, tire_model=self.params.tire_model,
            )
            vp3.yaw_damp = float(self.yaw_damp)
            vp3.yaw_sat_gain = float(self.yaw_sat_gain)
            
            # 轮胎参数同步
            try:
                mu_val = float(self.params.mu)
                vp3.tire_params_f.mu_y = mu_val
                vp3.tire_params_r.mu_y = mu_val
                vp3.tire_long_params_f.mu_x = mu_val
                vp3.tire_long_params_r.mu_x = mu_val
            except Exception: pass

            df_raw = float(self.ctrl.delta_f)
            dr_raw = float(self.ctrl.delta_r)

            # 纵向力计算
            if not self.autop_enabled:
                # 手动模式：计算 ax_cmd (简单P控制)
                alpha = 1.0 - np.exp(-dt / max(1e-6, self.tau_ctrl))
                self._U_cmd_filt += alpha * (float(self.ctrl.U) - self._U_cmd_filt)
                ax_cmd = self.k_v * (self._U_cmd_filt - self.state3.vx)
            # 自动模式下：ax_cmd 已经在前面由 Controller 算好了

            Fx_total = vp3.m * ax_cmd
            Fx_f_pure, Fx_r_pure = allocate_drive(Fx_total, df_raw, dr_raw, self.drive_bias_front, self.drive_bias_rear)

            # 3DOF 动力学混合积分
            speed_mag = float(np.hypot(self.state3.vx, self.state3.vy))
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t_blend = max(0.0, min(1.0, speed_mag / U_blend))
            w = t_blend * t_blend * (3.0 - 2.0 * t_blend)
            
            kappa = curvature_4ws(df_raw, dr_raw, vp3.L)
            r_des = self.state3.vx * kappa
            r_dot_kin = (r_des - self.state3.r) / max(1e-6, self.tau_low)
            vx_dot_kin = ax_cmd
            vy_dot_kin = - self.state3.vy / max(1e-6, self.tau_beta)
            xdot_kin, ydot_kin = body_to_world_2dof(self.state3.vx, 0.0, self.state3.psi)
            
            if w < 0.99:
                vx_dot_dyn, vy_dot_dyn, r_dot_dyn = 0.0, 0.0, 0.0
                x_dot_dyn, y_dot_dyn = 0.0, 0.0
            else:
                ds, _ = derivatives_dfdr(self.state3, df_raw, dr_raw, vp3, Fx_f_pure, Fx_r_pure)
                vx_dot_dyn, vy_dot_dyn, r_dot_dyn, x_dot_dyn, y_dot_dyn, _psi_dot_dyn = map(float, ds)
            
            vx_dot = w * vx_dot_dyn + (1.0 - w) * vx_dot_kin
            vy_dot = w * vy_dot_dyn + (1.0 - w) * vy_dot_kin
            r_dot  = w * r_dot_dyn  + (1.0 - w) * r_dot_kin
            x_dot  = w * x_dot_dyn  + (1.0 - w) * xdot_kin
            y_dot  = w * y_dot_dyn  + (1.0 - w) * ydot_kin
            psi_dot= self.state3.r

            self.state3.vx += vx_dot * dt
            self.state3.vy += vy_dot * dt
            self.state3.r  += r_dot  * dt
            self.state3.x  += x_dot  * dt
            self.state3.y  += y_dot  * dt
            self.state3.psi+= psi_dot * dt

            self._df_dot = (df_raw - self._df_cur) / dt
            self._dr_dot = (dr_raw - self._dr_cur) / dt
            self._df_cur = df_raw
            self._dr_cur = dr_raw
            self._speed = speed_mag
            self._radius = (self._speed / abs(self.state3.r)) if abs(self.state3.r) > 1e-6 else None
            
            denom = float(self.state3.vx**2 + self.state3.vy**2 + 1e-9)
            self._beta_dot = float((self.state3.vx * vy_dot - self.state3.vy * vx_dot) / denom)
            self._r_dot = float(r_dot)
            self._sim_t += dt

            if self.track_cfg.enabled:
                self._push_track_point(self.state3.x, self.state3.y)

    def _push_track_point(self, x: float, y: float):
        """记录轨迹点"""
        t = time.perf_counter()
        self.track.append({"x": float(x), "y": float(y), "t": float(t)})
        keep = self.track_cfg.retention_sec
        if keep is not None and keep > 0:
            tcut = t - keep
            i = 0
            while i < len(self.track) and self.track[i]["t"] < tcut:
                i += 1
            if i > 0:
                del self.track[:i]
        if len(self.track) > self.track_cfg.max_points:
            del self.track[:len(self.track) - self.track_cfg.max_points]

    def get_state(self) -> Dict[str, float]:
        """获取当前状态字典"""
        with self._lock:
            if self.mode == '2dof':
                return {
                    "x": self.state2.x,
                    "y": self.state2.y,
                    "psi": self.state2.psi,  # rad
                    "beta": self.state2.beta,
                    "r": self.state2.r,
                    "beta_dot": self._beta_dot,
                    "r_dot": self._r_dot,
                    "speed": self._speed,
                    "radius": self._radius if self._radius is not None else None,
                    "df": self._df_cur,
                    "dr": self._dr_cur,
                    "df_dot": self._df_dot,
                    "dr_dot": self._dr_dot,
                }
            else:
                beta = float(np.arctan2(self.state3.vy, max(1e-6, self.state3.vx)))
                return {
                    "x": self.state3.x,
                    "y": self.state3.y,
                    "psi": self.state3.psi,
                    "beta": beta,
                    "r": self.state3.r,
                    "beta_dot": self._beta_dot,
                    "r_dot": self._r_dot,
                    "speed": self._speed,
                    "radius": self._radius if self._radius is not None else None,
                    "df": self._df_cur,
                    "dr": self._dr_cur,
                    "df_dot": self._df_dot,
                    "dr_dot": self._dr_dot,
                }

    def get_track(self) -> List[Dict[str, float]]:
        with self._lock:
            return list(self.track)

    def get_ctrl(self) -> Dict[str, float]:
        with self._lock:
            return {
                "U": self.ctrl.U,
                "df": self.ctrl.delta_f,
                "dr": self.ctrl.delta_r,
                "running": self.running,
                "mode": self.mode,
            }

    def load_plan(self, points: List[Dict[str, float]]):
        """加载规划轨迹"""
        with self._lock:
            self.plan = [
                {
                    't': float(p.get('t', 0.0)),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'psi': float(p.get('psi', 0.0)),
                    'v': float(p.get('v', 0.0)), # 确保保留速度信息
                    'a': float(p.get('a', 0.0)), # 确保保留加速度信息
                }
                for p in points
            ]
            self._plan_idx = 0
            if len(self.plan) > 0:
                pend = self.plan[-1]
                self.goal_pose_end = {
                    'x': float(pend['x']),
                    'y': float(pend['y']),
                    'psi': float(pend.get('psi', 0.0)),
                }

    def set_autop(self, enabled: bool):
        with self._lock:
            self.autop_enabled = bool(enabled)
            # 简化：仅依赖布尔值开关，不需维护 control_type 字符串

    def set_autop_mode(self, mode: str):
        with self._lock:
            # 仅作为兼容接口保留
            m = str(mode or '').lower()
            if m == 'mpc':
                self.autop_mode = 'mpc'

    def get_controller_type(self) -> str:
        with self._lock:
            if not self.autop_enabled or len(self.plan) == 0:
                return 'manual'
            return 'mpc' # 统一返回 mpc

    def set_ctrl(self, **kw):
        """设置控制量 (前端手动输入)"""
        with self._lock:
            if "U" in kw and kw["U"] is not None:
                try: self.ctrl.U = float(kw["U"])
                except (TypeError, ValueError): pass
            if "df" in kw and kw["df"] is not None:
                try: self.ctrl.delta_f = float(kw["df"])
                except (TypeError, ValueError): pass
            if "dr" in kw and kw["dr"] is not None:
                try: self.ctrl.delta_r = float(kw["dr"])
                except (TypeError, ValueError): pass
            if "ax_des" in kw and kw["ax_des"] is not None:
                try: self.ctrl.ax_des = float(kw["ax_des"])
                except (TypeError, ValueError): pass

    def set_mode(self, mode: str):
        with self._lock:
            if mode in ('2dof', '3dof'):
                if mode != self.mode:
                    self.mode = mode
                    self.track.clear()
                    self._sim_t = 0.0

    def set_track_settings(self, enabled: bool | None = None, retention_sec: float | None = None, max_points: int | None = None):
        with self._lock:
            if enabled is not None:
                self.track_cfg.enabled = bool(enabled)
            if retention_sec is not None:
                try: self.track_cfg.retention_sec = max(0.0, float(retention_sec))
                except (TypeError, ValueError): pass
            if max_points is not None:
                try: self.track_cfg.max_points = max(100, int(max_points))
                except (TypeError, ValueError): pass

    def set_init_pose(self, x: float = 0.0, y: float = 0.0, psi_rad: float = 0.0):
        with self._lock:
            if self.mode == '2dof':
                self.state2.x = float(x)
                self.state2.y = float(y)
                self.state2.psi = float(psi_rad)
                self._actual_speed_2dof = 0.0
                self.params.U = 0.0
            else:
                self.state3.x = float(x)
                self.state3.y = float(y)
                self.state3.psi = float(psi_rad)
                self.state3.vx = 0.0
                self.state3.vy = 0.0
                self.state3.r = 0.0
            self.track.clear()

    def start(self):
        with self._lock: self.running = True

    def pause(self):
        with self._lock: self.running = False

    def toggle(self):
        with self._lock:
            self.running = not self.running
        return self.running

    def shutdown(self):
        self._alive = False
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass