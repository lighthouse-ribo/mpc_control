import threading
import time
from typing import List, Dict, Literal
import numpy as np
from .params import VehicleParams
from .model import SimState, Control, TrackSettings
from .twodof import derivatives as deriv_2dof
from .dof_utils import body_to_world_2dof, body_to_world_3dof, curvature_4ws
# from .mpc import solve_mpc_2dof, linearize_2dof
from .mpc import solve_mpc_kin_dyn_4dof
from .threedof import (
    Vehicle3DOF,
    State3DOF,
    allocate_drive,
    derivatives_dfdr,
)
from .planner import plan_quintic_xy
from .strategy import ideal_yaw_rate

class SimEngine:
    """后端仿真引擎：维护状态、轨迹与控制，并在后台线程中积分。"""
    def __init__(self, params: VehicleParams, dt: float = 0.02):
        self.params = params
        self.dt = float(dt)
        # 2DOF 与 3DOF 双态，按模式选择返回
        self.state2 = SimState()
        self.state3 = State3DOF(vx=params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
        self.ctrl = Control(U=params.U)
        
        # 动态扩展 Control 以支持前馈加速度（防止 model.py 未修改导致报错）
        if not hasattr(self.ctrl, 'ax_des'):
            self.ctrl.ax_des = 0.0

        self.track: List[Dict[str, float]] = []  # 每项为 {x, y, t}
        self.track_cfg = TrackSettings()

        # 模式
        self.mode: Literal['2dof', '3dof'] = '2dof'
        self._sim_t = 0.0

        self.running = False
        self._alive = True
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._loop, name="SimLoop", daemon=True)
        self._thread.start()

        self.delta_max = np.deg2rad(30.0)  # 轮角限幅（提升可达曲率）
        self.U_switch = 8.0                # 高速同相阈值（m/s）
        self.phase_auto = False            # 关闭高速同相覆盖，恢复手动后轮转向
        
        # --- [纵向控制参数: 串级PID + 前馈] ---
        # 速度环 PID 参数
        self.lon_kp = 1.5   # 比例
        self.lon_ki = 0.1   # 积分
        self.lon_kd = 0.05  # 微分
        
        # 纵向控制状态变量
        self._lon_integ = 0.0      # 积分累积
        self._lon_prev_err = 0.0   # 上一次误差（用于微分）
        self._actual_speed_2dof = float(params.U)  # 2DOF 模式下的物理速度状态
        # ------------------------------------

        self.k_v = 0.8                     # (3DOF 保留) 纵向速度跟踪增益
        self.tau_ctrl = 0.15               # 控制输入滤波时间常数（s）
        
        # MPPI 角速度占比（相对于 delta_max）
        self.delta_rate_frac = 0.5
        # 可配置的横摆阻尼与饱和控制参数
        self.yaw_damp = 220.0              # 横摆阻尼力矩系数
        self.yaw_sat_gain = 3.0            # 横摆率饱和额外阻尼增益
        self._df_filt = 0.0
        self._dr_filt = 0.0
        self._U_cmd_filt = float(self.ctrl.U)

        # 显示/诊断：轮角与角速度、车速与转弯半径
        self._df_cur = 0.0
        self._dr_cur = 0.0
        self._df_dot = 0.0
        self._dr_dot = 0.0
        self._speed = float(self.params.U)
        self._radius: float | None = None
        # 导数采样：beta_dot 与 r_dot（用于前端导出 CSV）
        self._beta_dot = 0.0
        self._r_dot = 0.0

        # 牵引分配
        self.drive_bias_front = 0.1        # 前轴基础牵引比例
        self.drive_bias_rear = 0.9         # 后轴基础牵引比例
        # 低速融合时间常数
        self.tau_low = 0.25
        self.tau_beta = 0.35
        # 规划与自动跟踪
        self.plan: List[Dict[str, float]] = []  # 每项 {t, x, y, psi}
        self.autop_enabled: bool = False
        self.autop_mode: Literal['simple', 'mpc', 'mppi'] = 'mpc'
        self._plan_idx: int = 0
        self.ctrl_plan_max_points = 4000
        self.ctrl_plan_pad_factor = 1.5
        # 控制器类型指示与几何回退标记
        self.controller_type: Literal['simple', 'mpc', 'mppi', 'geometric', 'manual'] = 'manual'
        self._geom_fallback_active: bool = False
        # 目标位姿与重规划开关
        self.goal_pose_end: Dict[str, float] | None = None
        self.replan_every_step: bool = False
        # 纯追踪/几何参考的预瞄距离参数
        self.Ld_k = 0.8                 # 预瞄距离线性系数：Ld = k*U + b
        self.Ld_b = 4.0                 # 预瞄距离偏置
        self.Ld_min = 3.0               # 预瞄下限
        self.Ld_max = 18.0              # 预瞄上限
        # MPC 车速控制参数
        self.U_cruise = float(params.U)
        self.U_max = float(params.U)           # 速度上限
        self.dU_max = 1.5                      # m/s 每秒加减速上限
        self.ay_limit_coeff = 0.85             # 横向加速度占比

    # 线程主循环
    def _loop(self):
        last = time.perf_counter()
        while self._alive:
            start = time.perf_counter()
            if self.running:
                # 固定步长积分
                with self._lock:
                    self._step(self.dt)
            # 控制节拍
            spent = time.perf_counter() - start
            sleep = max(0.0, self.dt - spent)
            time.sleep(sleep)

    # 重置仿真
    def reset(self):
        with self._lock:
            self.state2 = SimState()
            self.state3 = State3DOF(vx=self.params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
            self.track.clear()
            self.running = False
            self._sim_t = 0.0
            
            # [新增] 重置纵向 PID 状态
            self._lon_integ = 0.0
            self._lon_prev_err = 0.0
            self._actual_speed_2dof = float(self.params.U)
            self.ctrl.ax_des = 0.0
            self._U_cmd_filt = float(self.params.U)

    # [新增] 辅助函数：根据参考曲率计算目标速度与前馈加速度
    def _calc_curvature_speed_limit(self, kappa_ref: float, current_U: float) -> tuple[float, float]:
        """
        返回: (v_target, ax_ff)
        """
        # 1. 计算弯道限速 (ay_limit = U^2 * kappa)
        ay_limit = float(getattr(self.params, 'mu', 1.0) * getattr(self.params, 'g', 9.81)) * self.ay_limit_coeff
        kappa_mag = abs(kappa_ref)
        
        if kappa_mag > 1e-4:
            v_limit = np.sqrt(ay_limit / kappa_mag)
        else:
            v_limit = self.U_max  # 直线段使用最大巡航速度

        # 目标速度取 巡航设定 和 弯道限速 的最小值
        v_target = min(float(self.U_cruise), float(v_limit))

        # 2. 计算前馈加速度 (简单逻辑：前馈设为0，主要靠PID追踪v_target；
        #    如果需要更激进，可基于 v_target 的变化率计算)
        ax_ff = 0.0 

        return float(v_target), float(ax_ff)

    def _step(self, dt: float):
        # 自动跟踪：在积分前根据参考轨迹更新 df/dr/U
        if self.autop_enabled and len(self.plan) > 0:
            # 若启用每步重规划
            if self.replan_every_step:
                try:
                    self._replan_to_goal()
                except Exception:
                    pass
            if self.autop_mode == 'mpc' and self.mode == '2dof':
                self._autop_update_mpc()
            elif self.autop_mode == 'mppi':
                self._autop_update_mppi()
            elif self.mode == '2dof':
                self._autop_update_simple()

        # 模式分支：2DOF 线性 或 3DOF 非线性
        if self.mode == '2dof':
            # --- [修改核心] 2DOF 下的 纵向串级 PID + 前馈 ---
            
            # 1. 获取目标
            v_target = float(self.ctrl.U)
            ax_ff = getattr(self.ctrl, 'ax_des', 0.0)
            
            # 2. 计算误差
            v_curr = self._actual_speed_2dof
            error = v_target - v_curr
            
            # 3. PID 计算
            # 积分
            self._lon_integ += error * dt
            # 抗饱和
            self._lon_integ = np.clip(self._lon_integ, -5.0, 5.0) 
            
            # 微分
            d_error = (error - self._lon_prev_err) / max(1e-6, dt)
            self._lon_prev_err = error
            
            # PID 输出加速度
            ax_pid = (self.lon_kp * error) + (self.lon_ki * self._lon_integ) + (self.lon_kd * d_error)
            
            # 4. 总加速度 = 前馈 + 反馈
            ax_total = ax_ff + ax_pid
            
            # 5. 积分更新速度 (模拟物理响应)
            v_next = v_curr + ax_total * dt
            v_next = max(0.0, v_next) # 暂不考虑倒车
            
            self._actual_speed_2dof = v_next
            
            # 6. 同步给 params，以便模型计算 beta/r 时使用正确的速度
            self.params.U = v_next

            # --- [2DOF 横向动力学] ---
            x_vec = np.array([self.state2.beta, self.state2.r], dtype=float)
            d = deriv_2dof(x_vec, self.ctrl.delta_f, self.ctrl.delta_r, self.params)
            beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])

            # 姿态与位置积分（使用更新后的 v_next）
            U_signed = v_next
            psi_dot = self.state2.r
            x_dot, y_dot = body_to_world_2dof(U_signed, self.state2.beta, self.state2.psi)

            # 低速融合
            U_mag = v_next
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t = max(0.0, min(1.0, U_mag / U_blend))
            w = t * t * (3.0 - 2.0 * t)
            
            kappa = curvature_4ws(float(self.ctrl.delta_f), float(self.ctrl.delta_r), self.params.L)
            r_des = U_signed * kappa
            r_dot_kin = (r_des - self.state2.r) / max(1e-6, self.tau_low)
            beta_dot_kin = - self.state2.beta / max(1e-6, self.tau_beta)
            
            beta_dot = w * beta_dot + (1.0 - w) * beta_dot_kin
            r_dot = w * r_dot + (1.0 - w) * r_dot_kin

            # 记录导数
            self._beta_dot = float(beta_dot)
            self._r_dot = float(r_dot)

            self.state2.beta += beta_dot * dt
            self.state2.r += r_dot * dt
            self.state2.psi += psi_dot * dt
            self.state2.x += x_dot * dt
            self.state2.y += y_dot * dt

            # 遥测
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
            # 3DOF 非线性：直接使用 df/dr 控制
            vp3 = Vehicle3DOF(
                m=self.params.m,
                Iz=self.params.Iz,
                a=self.params.a,
                b=self.params.b,
                g=self.params.g,
                U_min=self.params.U_min,
                kf=self.params.kf,
                kr=self.params.kr,
                tire_model=self.params.tire_model,
            )
            vp3.yaw_damp = float(self.yaw_damp)
            vp3.yaw_sat_gain = float(self.yaw_sat_gain)
            try:
                mu_val = float(self.params.mu)
                vp3.tire_params_f.mu_y = mu_val
                vp3.tire_params_r.mu_y = mu_val
                vp3.tire_long_params_f.mu_x = mu_val
                vp3.tire_long_params_r.mu_x = mu_val
            except Exception:
                pass

            df_raw = float(self.ctrl.delta_f)
            dr_raw = float(self.ctrl.delta_r)

            alpha = 1.0 - np.exp(-dt / max(1e-6, self.tau_ctrl))
            self._U_cmd_filt += alpha * (float(self.ctrl.U) - self._U_cmd_filt)
            df = float(df_raw)
            dr = float(dr_raw)

            # 纵向驱动/制动：目标速度跟踪，体现 Fx-Fy 耦合
            ax_cmd = self.k_v * (self._U_cmd_filt - self.state3.vx)
            Fx_total = vp3.m * ax_cmd
            
            Fx_f_pure, Fx_r_pure = allocate_drive(Fx_total, df, dr, self.drive_bias_front, self.drive_bias_rear)

            U_signed = float(self.ctrl.U)
            speed_mag = float(abs(self.state3.vx))
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t = max(0.0, min(1.0, speed_mag / U_blend))
            w = t * t * (3.0 - 2.0 * t)
            
            kappa = curvature_4ws(df, dr, vp3.L)
            r_des = U_signed * kappa
            r_dot_kin = (r_des - self.state3.r) / max(1e-6, self.tau_low)
            vx_dot_kin = ax_cmd
            vy_dot_kin = - self.state3.vy / max(1e-6, self.tau_beta)
            xdot_kin, ydot_kin = body_to_world_2dof(U_signed, 0.0, self.state3.psi)
            
            if w < 0.99:
                vx_dot_dyn, vy_dot_dyn, r_dot_dyn = 0.0, 0.0, 0.0
                x_dot_dyn, y_dot_dyn = 0.0, 0.0
            else:
                ds, aux = derivatives_dfdr(self.state3, df, dr, vp3, Fx_f_pure, Fx_r_pure)
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

            self._df_dot = (df - self._df_cur) / dt
            self._dr_dot = (dr - self._dr_cur) / dt
            self._df_cur = df
            self._dr_cur = dr
            self._speed = float(np.hypot(self.state3.vx, self.state3.vy))
            self._radius = (self._speed / abs(self.state3.r)) if abs(self.state3.r) > 1e-6 else None

            denom = float(self.state3.vx * self.state3.vx + self.state3.vy * self.state3.vy + 1e-9)
            self._beta_dot = float((self.state3.vx * vy_dot - self.state3.vy * vx_dot) / denom)
            self._r_dot = float(r_dot)

            self._sim_t += dt

            if self.track_cfg.enabled:
                self._push_track_point(self.state3.x, self.state3.y)

    def _push_track_point(self, x: float, y: float):
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
        with self._lock:
            self.plan = [
                {
                    't': float(p.get('t', 0.0)),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'psi': float(p.get('psi', 0.0)),
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
            if not self.autop_enabled:
                self.controller_type = 'manual'

    def set_autop_mode(self, mode: str):
        with self._lock:
            m = str(mode or '').lower()
            if m in ('simple', 'mpc', 'mppi'):
                self.autop_mode = m
                if self.autop_enabled:
                    self.controller_type = m

    def get_controller_type(self) -> str:
        with self._lock:
            if not self.autop_enabled or len(self.plan) == 0:
                return 'manual'
            if self._geom_fallback_active:
                return 'geometric'
            return self.controller_type

    def _replan_to_goal(self):
        if self.goal_pose_end is None:
            if len(self.plan) == 0:
                return
            pend = self.plan[-1]
            self.goal_pose_end = {
                'x': float(pend['x']),
                'y': float(pend['y']),
                'psi': float(pend.get('psi', 0.0)),
            }

        start = {
            'x': float(self.state2.x),
            'y': float(self.state2.y),
            'psi': float(self.state2.psi),
        }
        end = dict(self.goal_pose_end)

        dist = float(np.hypot(end['x'] - start['x'], end['y'] - start['y']))
        U_eff = float(max(0.3, abs(self.params.U_eff())))
        T = float(np.clip(dist / U_eff if U_eff > 1e-6 else 1.0, 0.5, 30.0))
        N = int(np.clip(T / max(1e-6, self.dt), 60, 400))

        plan = plan_quintic_xy(start, end, T, N, U_start=float(self.params.U))
        self.plan = plan
        self._plan_idx = 0

    def _wrap_angle(self, a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _plan_ref_geometry(self, x: float, y: float, psi_cur: float, U: float) -> Dict[str, float]:
        n = len(self.plan)
        if n < 2:
            return {
                'base_i': 0, 'ref_i': 0, 'psi_ref': psi_cur,
                'e_lat': 0.0, 'psi_err': 0.0, 'kappa_ref': 0.0,
                'Ld': 5.0, 'ds_ref': 1.0,
            }
        start_hint = int(self._plan_idx)
        window = 200
        i0 = max(0, start_hint - window)
        i1 = min(n - 1, start_hint + window)
        best_i = i0
        best_d2 = float('inf')
        for i in range(i0, i1 + 1):
            px = float(self.plan[i]['x'])
            py = float(self.plan[i]['y'])
            d2 = (px - x) * (px - x) + (py - y) * (py - y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        base_i = int(best_i)
        self._plan_idx = base_i
        base_i = max(0, min(n - 2, base_i))
        U_mag = float(max(0.0, abs(U)))
        Ld = float(np.clip(self.Ld_k * U_mag + self.Ld_b, self.Ld_min, self.Ld_max))
        s_acc = 0.0
        ref_i = base_i
        while ref_i < n - 1 and s_acc < Ld:
            p = self.plan[ref_i]
            q = self.plan[ref_i + 1]
            ds_i = float(np.hypot(q['x'] - p['x'], q['y'] - p['y']))
            s_acc += ds_i
            if s_acc < Ld:
                ref_i += 1
        ref_i = min(ref_i, n - 2)

        p0 = self.plan[ref_i]
        p1 = self.plan[ref_i + 1]
        dx = float(p1['x'] - p0['x']); dy = float(p1['y'] - p0['y'])
        ds_ref = float(np.hypot(dx, dy))
        psi_ref = float(np.arctan2(dy, dx)) if ds_ref > 1e-6 else float(p0.get('psi', psi_cur))

        ex = float(x - p0['x']); ey = float(y - p0['y'])
        e_lat = float(-ex * np.sin(psi_ref) + ey * np.cos(psi_ref))
        psi_err = self._wrap_angle(psi_ref - float(psi_cur))

        j_prev = max(0, ref_i - 1)
        j_next = min(n - 2, ref_i + 1)
        def seg_psi(i: int) -> float:
            a = self.plan[i]; b = self.plan[i + 1]
            dx_i = float(b['x'] - a['x']); dy_i = float(b['y'] - a['y'])
            ds_i = float(np.hypot(dx_i, dy_i))
            return float(np.arctan2(dy_i, dx_i)) if ds_i > 1e-6 else float(a.get('psi', psi_ref))
        psi_a = seg_psi(j_prev)
        psi_b = seg_psi(j_next)
        dpsi = self._wrap_angle(psi_b - psi_a)
        ds_a = float(np.hypot(self.plan[j_prev + 1]['x'] - self.plan[j_prev]['x'], self.plan[j_prev + 1]['y'] - self.plan[j_prev]['y']))
        ds_b = float(np.hypot(self.plan[j_next + 1]['x'] - self.plan[j_next]['x'], self.plan[j_next + 1]['y'] - self.plan[j_next]['y']))
        ds_avg = max(1e-6, 0.5 * (ds_a + ds_b))
        kappa_ref = float(dpsi / ds_avg)

        return {
            'base_i': base_i,
            'ref_i': ref_i,
            'psi_ref': psi_ref,
            'e_lat': e_lat,
            'psi_err': psi_err,
            'kappa_ref': kappa_ref,
            'Ld': Ld,
            'ds_ref': ds_ref,
        }

    def _get_plan_for_controller(self, H_est: int) -> List[Dict[str, float]]:
        with self._lock:
            pts = self.plan
            n = len(pts)
            if n <= 2:
                return list(pts)
            x = float(self.state2.x)
            y = float(self.state2.y)
            psi_cur = float(self.state2.psi)
            U_mag = float(self.params.U_eff())
            ref = self._plan_ref_geometry(x, y, psi_cur, U_mag)
            base_i = int(ref.get('base_i', 0))
            base_i = max(0, min(n - 2, base_i))
            H_eff = max(1, int(H_est))
            Lwin = float(U_mag * self.dt * H_eff * float(self.ctrl_plan_pad_factor))
            s_acc = 0.0
            end_i = base_i
            while end_i < n - 1 and s_acc < Lwin:
                p = pts[end_i]
                q = pts[end_i + 1]
                s_acc += float(np.hypot(q['x'] - p['x'], q['y'] - p['y']))
                if s_acc < Lwin:
                    end_i += 1
            end_i = min(end_i + 1, n - 1)
            segment = pts[base_i:end_i + 1]
            m = len(segment)
            maxp = int(self.ctrl_plan_max_points)
            if m <= maxp or maxp <= 2:
                return list(segment)
            out = []
            for i in range(maxp):
                idx = int(round(i * (m - 1) / max(1, maxp - 1)))
                out.append(segment[idx])
            return out

    def _autop_update_simple(self):
        if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
            return
        self.controller_type = 'simple'
        x = float(self.state2.x)
        y = float(self.state2.y)
        psi_cur = float(self.state2.psi)
        U_mag = float(self.params.U_eff())
        ref = self._plan_ref_geometry(x, y, psi_cur, U_mag)
        i_goal = min(len(self.plan) - 1, int(ref['ref_i'] + 1))
        p_goal = self.plan[i_goal]
        alpha = self._wrap_angle(float(np.arctan2(p_goal['y'] - y, p_goal['x'] - x)) - psi_cur)
        L = float(self.params.L)
        Ld = float(max(1.0, ref['Ld']))
        df_cmd_raw = float(np.arctan2(2.0 * L * np.sin(alpha), Ld))
        df_cmd = float(np.clip(df_cmd_raw, -self.delta_max, self.delta_max))
        x_vec = np.array([self.state2.beta, self.state2.r], dtype=float)
        try:
            dr_cmd_raw, _diag = ideal_yaw_rate(df_cmd, x_vec, self.params)
        except Exception:
            dr_cmd_raw = 0.0
        dr_cmd = float(np.clip(dr_cmd_raw, -self.delta_max, self.delta_max))
        alpha_f = 1.0 - np.exp(-self.dt / max(1e-6, self.tau_ctrl))
        self._df_filt += alpha_f * (df_cmd - self._df_filt)
        self._dr_filt += alpha_f * (dr_cmd - self._dr_filt)
        self.ctrl.delta_f = float(self._df_filt)
        self.ctrl.delta_r = float(self._dr_filt)

    def _fallback_geom_3dof(self) -> None:
        if not (self.autop_enabled and self.mode == '3dof' and len(self.plan) > 0):
            return
        try:
            self._geom_fallback_active = True
            self.controller_type = 'geometric'
            x = float(getattr(self.state3, 'x', self.state2.x))
            y = float(getattr(self.state3, 'y', self.state2.y))
            psi_cur = float(getattr(self.state3, 'psi', self.state2.psi))
            U_mag = float(np.hypot(getattr(self.state3, 'vx', self.params.U_eff()), getattr(self.state3, 'vy', 0.0)))
            ref = self._plan_ref_geometry(x, y, psi_cur, U_mag)

            i_goal = min(len(self.plan) - 1, int(ref['ref_i'] + 1))
            p_goal = self.plan[i_goal]
            alpha = self._wrap_angle(float(np.arctan2(p_goal['y'] - y, p_goal['x'] - x)) - psi_cur)
            L = float(self.params.L)
            Ld = float(max(1.0, ref['Ld']))
            df_cmd_raw = float(np.arctan2(2.0 * L * np.sin(alpha), Ld))
            df_cmd = float(np.clip(df_cmd_raw, -self.delta_max, self.delta_max))

            kappa_mag = abs(float(ref['kappa_ref']))
            G_turn = max(0.0, min(1.0, (kappa_mag - 0.02) / (0.06 - 0.02 + 1e-9)))
            s_lin = max(0.0, min(1.0, (U_mag - 5.0) / (20.0 - 5.0)))
            k_t = (-0.8 * G_turn) + (0.10 * (1.0 - G_turn) * s_lin)
            dr_cmd = float(np.clip(k_t * df_cmd, -self.delta_max, self.delta_max))

            ay_limit = float(getattr(self.params, 'mu', 1.0) * getattr(self.params, 'g', 9.81)) * float(self.ay_limit_coeff)
            if kappa_mag > 1e-6:
                U_des = float(np.sqrt(max(0.0, ay_limit / max(1e-6, kappa_mag))))
            else:
                U_des = float(self.U_cruise)
            dU = float(np.clip(U_des - self.ctrl.U, -self.dU_max * self.dt, self.dU_max * self.dt))
            U_next = float(np.clip(self.ctrl.U + dU, 0.0, self.U_max))

            alpha_f = 1.0 - np.exp(-self.dt / max(1e-6, self.tau_ctrl))
            self._df_filt += alpha_f * (df_cmd - self._df_filt)
            self._dr_filt += alpha_f * (dr_cmd - self._dr_filt)
            self.ctrl.delta_f = float(self._df_filt)
            self.ctrl.delta_r = float(self._dr_filt)
            self.ctrl.U = U_next
        except Exception as e:
            print(f"[SimEngine] 几何型 3DOF 回退异常: {e}")

    def _autop_update_mppi(self):
        if not (self.autop_enabled and len(self.plan) > 0):
            return

        if not hasattr(self, '_mppi_ctrl') or self._mppi_ctrl is None:
            try:
                from .mppi_iface import MPPIController4WS
                self._mppi_ctrl = MPPIController4WS(
                    params=self.params,
                    dt=self.dt,
                    plan_provider=lambda: self._get_plan_for_controller(30),
                    delta_max=float(self.delta_max),
                    dU_max=float(self.dU_max),
                    U_max=float(self.U_max),
                    model_type=('3dof' if self.mode == '3dof' else '2dof'),
                    delta_rate_frac=float(self.delta_rate_frac),
                )
            except Exception:
                try:
                    self._mppi_ctrl = MPPIController4WS(
                        params=self.params,
                        dt=self.dt,
                        plan_provider=lambda: self._get_plan_for_controller(30),
                        delta_max=float(self.delta_max),
                        dU_max=float(self.dU_max),
                        U_max=float(self.U_max),
                        device='cpu',
                        model_type=('3dof' if self.mode == '3dof' else '2dof'),
                        delta_rate_frac=float(self.delta_rate_frac),
                    )
                except Exception as e:
                    print(f"[SimEngine] MPPI 初始化失败: {e}")
                    try:
                        self._fallback_geom_3dof()
                    except: pass
                    return
            try:
                self._mppi_ctrl.drive_bias_front = float(self.drive_bias_front)
                self._mppi_ctrl.drive_bias_rear = float(self.drive_bias_rear)
                self._mppi_ctrl.k_v = float(self.k_v)
            except Exception:
                pass

        if self.mode == '2dof':
            s_np = np.array([
                float(self.state2.x), float(self.state2.y), float(self.state2.psi),
                float(self.state2.beta), float(self.state2.r),
                float(self.ctrl.U),
                float(self.ctrl.delta_f), float(self.ctrl.delta_r),
            ], dtype=float)
        else:
            vx = float(getattr(self.state3, 'vx', self.params.U_eff()))
            vy = float(getattr(self.state3, 'vy', 0.0))
            s_np = np.array([
                vx, vy,
                float(getattr(self.state3, 'r', self.state2.r)),
                float(getattr(self.state3, 'x', self.state2.x)),
                float(getattr(self.state3, 'y', self.state2.y)),
                float(getattr(self.state3, 'psi', self.state2.psi)),
                float(self.ctrl.U),
                float(self.ctrl.delta_f), float(self.ctrl.delta_r),
            ], dtype=float)

        try:
            u_np = self._mppi_ctrl.command(s_np)
        except Exception as e:
            try:
                # 尝试重新初始化
                from .mppi_iface import MPPIController4WS
                self._mppi_ctrl = MPPIController4WS(
                    params=self.params,
                    dt=self.dt,
                    plan_provider=lambda: self._get_plan_for_controller(30),
                    delta_max=float(self.delta_max),
                    dU_max=float(self.dU_max),
                    U_max=float(self.U_max),
                    device='cpu',
                    model_type=('3dof' if self.mode == '3dof' else '2dof'),
                    delta_rate_frac=float(self.delta_rate_frac),
                )
                u_np = self._mppi_ctrl.command(s_np)
            except Exception:
                try:
                    self._fallback_geom_3dof()
                except: pass
                return
        
        d_df_cmd, d_dr_cmd, dU_cmd = float(u_np[0]), float(u_np[1]), float(u_np[2])
        self._geom_fallback_active = False
        self.controller_type = 'mppi'
        
        df_next = float(np.clip(self.ctrl.delta_f + d_df_cmd, -self.delta_max, self.delta_max))
        dr_next = float(np.clip(self.ctrl.delta_r + d_dr_cmd, -self.delta_max, self.delta_max))
        U_next = float(np.clip(self.ctrl.U + dU_cmd, 0.0, self.U_max))

        # 前馈加速度
        ax_feedforward = dU_cmd / max(1e-6, self.dt)
        self.ctrl.ax_des = ax_feedforward

        self.ctrl.delta_f = df_next
        self.ctrl.delta_r = dr_next
        self.ctrl.U = U_next

    def _linearize_2dof(self, x_vec: np.ndarray, df0: float, dr0: float) -> tuple[np.ndarray, np.ndarray]:
        base = deriv_2dof(x_vec, df0, dr0, self.params)
        xdot0 = np.array(base["xdot"], dtype=float)
        nx = 2
        nu = 2
        A = np.zeros((nx, nx), dtype=float)
        B = np.zeros((nx, nu), dtype=float)
        eps_x = 1e-4
        eps_u = 1e-3
        for j in range(nx):
            x_eps = np.array(x_vec, dtype=float)
            x_eps[j] += eps_x
            xdot_eps = np.array(deriv_2dof(x_eps, df0, dr0, self.params)["xdot"], dtype=float)
            A[:, j] = (xdot_eps - xdot0) / eps_x
        u0 = np.array([df0, dr0], dtype=float)
        for j in range(nu):
            u_eps = np.array(u0, dtype=float)
            u_eps[j] += eps_u
            xdot_eps = np.array(deriv_2dof(x_vec, float(u_eps[0]), float(u_eps[1]), self.params)["xdot"], dtype=float)
            B[:, j] = (xdot_eps - xdot0) / eps_u
        return A, B

    def _autop_update_mpc(self):
        """调用外部模块的 MPC 求解（2DOF 模式下使用）"""
        if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
            return
        self.controller_type = 'mpc'

        state_raw = {
            'x': float(self.state2.x),
            'y': float(self.state2.y),
            'psi': float(self.state2.psi),
            'beta': float(self.state2.beta),
            'r': float(self.state2.r),
        }
        # 使用 2DOF 下的“实际速度”
        U_mag = float(self._actual_speed_2dof)

        # 1. 获取参考信息
        ref_geom = self._plan_ref_geometry(
            state_raw['x'], state_raw['y'], state_raw['psi'], U_mag
        )
        
        # 2. [纵向] 计算目标速度和前馈
        kappa_ref = ref_geom.get('kappa_ref', 0.0)
        v_target, ax_ff = self._calc_curvature_speed_limit(kappa_ref, U_mag)
        self.ctrl.U = v_target
        self.ctrl.ax_des = ax_ff

        # 3. [横向] MPC 计算误差
        try:
            n_plan = len(self.plan)
            base_i = int(ref_geom.get('base_i', 0))
            i_seg = max(0, min(n_plan - 2, base_i))
            p0 = self.plan[i_seg]
            psi_base = float(p0.get('psi', state_raw['psi']))
            ex = float(state_raw['x'] - p0['x'])
            ey = float(state_raw['y'] - p0['y'])
            e_y = float(-ex * np.sin(psi_base) + ey * np.cos(psi_base))
            e_psi = float(self._wrap_angle(psi_base - float(state_raw['psi'])))
        except Exception:
            e_y = float(ref_geom['e_lat'])
            e_psi = float(ref_geom['psi_err'])

        state_for_mpc = {
            'x': state_raw['x'],
            'y': state_raw['y'],
            'psi': state_raw['psi'],
            'e_y': e_y,
            'e_psi': e_psi,
            'beta': state_raw['beta'],
            'r': state_raw['r'],
        }
        
        ctrl_raw = {
            'U': U_mag, # 告诉 MPC 当前实际速度
            'delta_f': float(self.ctrl.delta_f),
            'delta_r': float(self.ctrl.delta_r),
        }
        
        # 增加预测时域，使用阻尼参数防止画龙
        H_pred = 40 
        plan_ctrl = self._get_plan_for_controller(H_pred)
        
        df_cmd, dr_cmd = solve_mpc_kin_dyn_4dof(
            state_for_mpc,
            ctrl_raw,
            self.params,
            plan_ctrl,
            self.dt,
            H=H_pred,
            Q_ey=100,
            Q_epsi=1000,     # 降低航向误差权重 (从10000 -> 1000)
            Q_beta=20,
            Q_r=0.1,
            R_df=2,
            R_dr=2,
            R_delta_df=20.0, # 增加控制增量惩罚 (从0.2 -> 20.0)
            R_delta_dr=20.0,
            delta_max=self.delta_max,
        )
        
        self.ctrl.delta_f = float(df_cmd)
        self.ctrl.delta_r = float(dr_cmd)

    def set_ctrl(self, **kw):
        with self._lock:
            if "U" in kw and kw["U"] is not None:
                try:
                    self.ctrl.U = float(kw["U"])
                except (TypeError, ValueError):
                    pass
            if "df" in kw and kw["df"] is not None:
                try:
                    self.ctrl.delta_f = float(kw["df"])
                except (TypeError, ValueError):
                    pass
            if "dr" in kw and kw["dr"] is not None:
                try:
                    self.ctrl.delta_r = float(kw["dr"])
                except (TypeError, ValueError):
                    pass
            if "ax_des" in kw and kw["ax_des"] is not None:
                try:
                    self.ctrl.ax_des = float(kw["ax_des"])
                except (TypeError, ValueError):
                    pass

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
                try:
                    self.track_cfg.retention_sec = max(0.0, float(retention_sec))
                except (TypeError, ValueError):
                    pass
            if max_points is not None:
                try:
                    self.track_cfg.max_points = max(100, int(max_points))
                except (TypeError, ValueError):
                    pass

    def set_init_pose(self, x: float = 0.0, y: float = 0.0, psi_rad: float = 0.0):
        with self._lock:
            if self.mode == '2dof':
                self.state2.x = float(x)
                self.state2.y = float(y)
                self.state2.psi = float(psi_rad)
                # 复位内部速度
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
        with self._lock:
            self.running = True

    def pause(self):
        with self._lock:
            self.running = False

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