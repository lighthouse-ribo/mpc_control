import numpy as np
import torch
from typing import Callable, List, Dict, Optional

from .params import VehicleParams
from .twodof_torch import slip_angles_2dof_torch, lateral_forces_2dof_torch
from .threedof_torch import (
    slip_angles_3dof_torch,
    tire_forces_3dof_torch,
    derivatives_dfdr_torch,
    derivatives_speed_cmd_torch,
)
from .threedof import Vehicle3DOF
from .pytorch_mppi.mppi import MPPI, SMPPI, KMPPI, RBFKernel
from .strategy import ideal_yaw_rate


class MPPIController4WS:
    """4WS 车辆的 MPPI 控制封装。

    - 动力学：复用 twodof 的物理公式（Torch 版），在 MPPI 内进行批量滚动。
    - 状态：s = [x, y, psi, beta, r, U, delta_f, delta_r]
    - 动作：u = [d_delta_f, d_delta_r, dU]（转角与速度均使用微分形式）
    """

    def __init__(
        self,
        params: VehicleParams,
        dt: float,
        plan_provider: Callable[[], List[Dict[str, float]]],
        delta_max: float,
        dU_max: float,
        U_max: float,
        model_type: str = '2dof',
        device: Optional[str] = None,
        num_samples: int = 256,
        horizon: int = 30,
        lambda_: float = 20.0,
        delta_rate_frac: float = 0.5,
    ):
        self.params = params
        self.dt = float(dt)
        self.plan_provider = plan_provider
        self.delta_max = float(delta_max)
        self.dU_max = float(dU_max)
        self.U_max = float(U_max)
        # 每秒角度变化上限占比（相对于 delta_max），默认 0.5 更保守
        # 可调以加快转角速度，但不改变最终角度幅度上限
        self.delta_rate_frac = float(delta_rate_frac)
        self.model_type = str(model_type or '2dof').lower()
        # prefer MPS on macOS if available, else CUDA, else CPU
        try:
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        self.device = device or ('mps' if mps_available else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.w_lat = 20000
        self.w_head = 400000
        # 直线速度激励（机会成本），弯道弱化
        self.w_speed = 10.0
        self.w_u = 1.0
        self.w_du = 3.0
        # 速度相关的相位约束（低速更强反相，高速弱同相）
        self.w_phase_base = 300.0
        self.k_low = -0.30
        self.k_high = 0.10
        self.U1 = 5.0
        self.U2 = 20.0
        # 额外约束：直接惩罚同相（df、dr 同号）与加入理想横摆率前馈
        self.w_phase_sign = 180.0
        self.w_yaw_ff = 60.0

        # 曲率门控与蟹行/稳定性权重
        self.kappa_turn_k0 = 0.02   # 转弯门控起始阈值 [1/m]
        self.kappa_turn_k1 = 0.06   # 转弯门控满载阈值 [1/m]
        self.w_yaw_track = 240.0   # 强制 r ≈ U*kappa_ref
        self.w_crab = 22.0         # 弯道蟹行抑制 (df+dr)^2
        self.w_ay = 12.0           # 弯道横向加速度代价 ay^2
        self.w_beta = 10.0          # 弯道侧偏角代价 beta^2
        self.w_dU_turn = 1.2        # 弯道加速变化惩罚 dU^2
        self.k_max = 0.8            # 弯道相位比例的最大反相幅度

        # 控制噪声协方差（三维动作，使用微分形式）
        noise_sigma = torch.diag(torch.tensor([
            0.03,  # d_delta_f 每步角度增量
            0.03,  # d_delta_r 每步角度增量
            0.15   # dU 速度增量
        ], dtype=torch.float32))

        # 动作边界（微分形式：转角增量限制为每秒上限的 dt 倍）
        # 角速度上限：delta_rate_frac * delta_max / s（可由外部配置传入）
        # 例如：delta_rate_frac=0.8 表示 1 秒可接近 0.8*delta_max 的角度变化
        delta_rate_max = float(self.delta_rate_frac) * self.delta_max  # rad/s
        d_delta_bound = delta_rate_max * self.dt
        u_min = torch.tensor([
            -d_delta_bound,
            -d_delta_bound,
            -self.dU_max * self.dt
        ], dtype=torch.float32)
        u_max = torch.tensor([
            d_delta_bound,
            d_delta_bound,
            self.dU_max * self.dt
        ], dtype=torch.float32)

        # self.mppi = MPPI(
        #     dynamics=self._dynamics,
        #     running_cost=self._running_cost,
        #     nx=8,  # 状态维度: [x, y, psi, beta, r, U, delta_f, delta_r]
        #     noise_sigma=noise_sigma,
        #     num_samples=num_samples,
        #     horizon=horizon,
        #     lambda_=lambda_,
        #     u_min=u_min,
        #     u_max=u_max,
        #     device=self.device,
        # )

        # self.mppi = SMPPI(
        #     dynamics=self._dynamics,
        #     running_cost=self._running_cost,
        #     nx=8,  # 状态维度: [x, y, psi, beta, r, U, delta_f, delta_r]
        #     noise_sigma=noise_sigma,
        #     num_samples=num_samples,
        #     horizon=horizon,
        #     lambda_=lambda_,
        #     u_min=u_min,
        #     u_max=u_max,
        #     device=self.device,
        #     w_action_seq_cost=10,
        #     action_max=torch.tensor([1., 1.], dtype=torch.float32, device=self.device),
        # )
        # 状态维度：2DOF 为 8；3DOF 为 9
        nx = 8 if self.model_type == '2dof' else 9
        self.mppi = KMPPI(
            dynamics=self._dynamics,
            running_cost=self._running_cost,
            nx=nx,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            u_min=u_min,
            u_max=u_max,
            device=self.device,
            kernel=RBFKernel(sigma=2),
            num_support_pts=5,
            )

        # 缓存上一时刻动作用于平滑代价
        self._last_u = None

    # --- 动力学：f(s, u) -> s_next ---
    def _dynamics(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # s: [batch?, 6]; u: [batch?, 3]
        # 支持单样本和批量。内部全部使用张量广播。
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if u.ndim == 1:
            u = u.unsqueeze(0)

        # 2DOF: s=[x, y, psi, beta, r, U, df, dr]
        # 3DOF: s=[vx, vy, r, x, y, psi, U_cmd, df, dr]
        if self.model_type == '2dof':
            if s.shape[-1] >= 8:
                x, y, psi, beta, r, U, df_cur, dr_cur = s.unbind(-1)
            else:
                x, y, psi, beta, r, U = s.unbind(-1)
                df_cur = torch.zeros_like(U)
                dr_cur = torch.zeros_like(U)
        else:
            # 3DOF 状态
            # 保证至少 9 维
            if s.shape[-1] < 9:
                # 若调用方传入不足 9 维，则补零（容错）
                pad = torch.zeros((*s.shape[:-1], 9 - s.shape[-1]), dtype=s.dtype, device=s.device)
                s = torch.cat([s, pad], dim=-1)
            vx, vy, r, x, y, psi, U_cmd, df_cur, dr_cur = s.unbind(-1)

        # 解包动作：转角增量与速度增量
        d_df, d_dr, dU = u.unbind(-1)

        # 角度更新（限幅）
        df_next = torch.clamp(df_cur + d_df, -self.delta_max, self.delta_max)
        dr_next = torch.clamp(dr_cur + d_dr, -self.delta_max, self.delta_max)
        if self.model_type == '2dof':
            # 速度更新（带边界）
            U_next = torch.clamp(U + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))
        else:
            # 3DOF 使用速度指令（用于纵向驱动/制动），仍做边界
            U_next = torch.clamp(U_cmd + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))

        if self.model_type == '2dof':
            # 侧偏角与横向力（复用 twodof 公式）
            alpha_f, alpha_r = slip_angles_2dof_torch(beta, r, df_next, dr_next, float(self.params.a), float(self.params.b), U_next)
            Fy_f, Fy_r = lateral_forces_2dof_torch(alpha_f, alpha_r, self.params)

            # 2DOF 动力学方程
            m = float(self.params.m)
            Iz = float(self.params.Iz)
            beta_dot = (Fy_f + Fy_r) / (m * torch.clamp(U_next, min=1e-6)) - r
            r_dot = (float(self.params.a) * Fy_f - float(self.params.b) * Fy_r) / Iz

            # 几何部分（与 body_to_world_2dof 一致）
            x_dot = U_next * torch.cos(psi + beta)
            y_dot = U_next * torch.sin(psi + beta)
            psi_dot = r

            s_next = torch.stack([
                x + self.dt * x_dot,
                y + self.dt * y_dot,
                psi + self.dt * psi_dot,
                beta + self.dt * beta_dot,
                r + self.dt * r_dot,
                U_next,
                df_next,
                dr_next,
            ], dim=-1)
        else:
            # 3DOF 非线性动力学：使用速度指令（U_next）产生纵向驱动，耦合摩擦椭圆
            # 状态子向量为 [vx, vy, r, x, y, psi]
            s6 = torch.stack([vx, vy, r, x, y, psi], dim=-1)
            ds, _aux = derivatives_speed_cmd_torch(
                s6, df_next, dr_next, Vehicle3DOF(
                    m=float(self.params.m),
                    Iz=float(self.params.Iz),
                    a=float(self.params.a),
                    b=float(self.params.b),
                    g=float(self.params.g),
                    U_min=float(getattr(self.params, 'U_min', 0.5)),
                    kf=float(self.params.kf),
                    kr=float(self.params.kr),
                    tire_model=str(getattr(self.params, 'tire_model', 'linear')),
                ),
                U_next,
                k_v=float(getattr(self, 'k_v', 1.0)),
                front_bias=float(getattr(self, 'drive_bias_front', 0.5)),
                rear_bias=float(getattr(self, 'drive_bias_rear', 0.5)),
            )
            vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot = ds.unbind(-1)
            s_next = torch.stack([
                # 状态排布保持 3DOF 版本的 9 维
                vx + self.dt * vx_dot,
                vy + self.dt * vy_dot,
                r + self.dt * r_dot,
                x + self.dt * x_dot,
                y + self.dt * y_dot,
                psi + self.dt * psi_dot,
                U_next,
                df_next,
                dr_next,
            ], dim=-1)

        return s_next.squeeze(0)

    # --- 成本函数：l(s, u) ---
    def _running_cost(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        plan = self.plan_provider() or []
        if len(plan) < 2:
            return self.w_u * torch.sum(u * u, dim=-1)

        single = False
        if s.ndim == 1:
            s = s.unsqueeze(0)
            u = u.unsqueeze(0)
            single = True

        dtype = s.dtype
        device = s.device

        px_t = torch.tensor([p['x'] for p in plan], dtype=dtype, device=device)
        py_t = torch.tensor([p['y'] for p in plan], dtype=dtype, device=device)
        dx_seg = px_t[1:] - px_t[:-1]
        dy_seg = py_t[1:] - py_t[:-1]
        seg_psi_t = torch.atan2(dy_seg, torch.clamp(dx_seg, min=1e-12))

        if self.model_type == '2dof':
            x = s[..., 0]
            y = s[..., 1]
            psi_t = s[..., 2]
            U_mag_t = s[..., 5]
        else:
            vx = s[..., 0]
            vy = s[..., 1]
            x = s[..., 3]
            y = s[..., 4]
            psi_t = s[..., 5]
            U_mag_t = torch.sqrt(torch.clamp(vx * vx + vy * vy, min=1e-12))

        K = px_t.shape[0]
        dist2 = (x.unsqueeze(1) - px_t.unsqueeze(0)) ** 2 + (y.unsqueeze(1) - py_t.unsqueeze(0)) ** 2
        j = torch.argmin(dist2, dim=1)
        j = torch.clamp(j, min=0, max=K - 2)

        psi_base = seg_psi_t[j]
        ex = x - px_t[j]
        ey = y - py_t[j]
        e_y_t = -ex * torch.sin(psi_base) + ey * torch.cos(psi_base)
        dpsi = psi_base - psi_t
        e_psi_t = torch.atan2(torch.sin(dpsi), torch.cos(dpsi))

        j_prev = torch.clamp(j - 1, min=0, max=K - 2)
        j_next = torch.clamp(j + 1, min=0, max=K - 2)
        dpsi_seg = seg_psi_t[j_next] - seg_psi_t[j_prev]
        dpsi_seg_wrapped = torch.atan2(torch.sin(dpsi_seg), torch.cos(dpsi_seg))
        dx_prev = px_t[j_prev + 1] - px_t[j_prev]
        dy_prev = py_t[j_prev + 1] - py_t[j_prev]
        ds_a = torch.sqrt(dx_prev * dx_prev + dy_prev * dy_prev)
        dx_next = px_t[j_next + 1] - px_t[j_next]
        dy_next = py_t[j_next + 1] - py_t[j_next]
        ds_b = torch.sqrt(dx_next * dx_next + dy_next * dy_next)
        ds_avg = torch.clamp(0.5 * (ds_a + ds_b), min=1e-6)
        kappa_ref_t_signed = dpsi_seg_wrapped / ds_avg
        kappa_ref_mag = torch.abs(kappa_ref_t_signed)
        G_turn = torch.clamp((kappa_ref_mag - self.kappa_turn_k0) / (self.kappa_turn_k1 - self.kappa_turn_k0 + 1e-9), 0.0, 1.0)
        G_straight = 1.0 - 0.7 * G_turn

        cost = (
            self.w_lat * (e_y_t ** 2) +
            self.w_head * (e_psi_t ** 2) +
            self.w_u * torch.sum(u * u, dim=-1)
        )

        if self.model_type == '2dof':
            yaw_track_err = s[..., 4] - s[..., 5] * kappa_ref_t_signed
        else:
            yaw_track_err = s[..., 2] - U_mag_t * kappa_ref_t_signed
        cost = cost + self.w_yaw_track * (yaw_track_err ** 2)

        U_max_t = torch.tensor(self.U_max, dtype=dtype, device=device)
        if self.model_type == '2dof':
            speed_shortfall = torch.clamp(U_max_t - s[..., 5], min=0.0)
        else:
            speed_shortfall = torch.clamp(U_max_t - U_mag_t, min=0.0)
        cost = cost + self.w_speed * speed_shortfall * G_straight

        if self.model_type == '2dof':
            df_t = s[..., 6]
            dr_t = s[..., 7]
            kappa_cur_signed = s[..., 4] / torch.clamp(s[..., 5], min=1e-6)
            ay_approx = (s[..., 5] ** 2) * torch.abs(kappa_cur_signed)
            beta_t = s[..., 3]
        else:
            df_t = s[..., 7]
            dr_t = s[..., 8]
            kappa_cur_signed = s[..., 2] / torch.clamp(U_mag_t, min=0.3)
            ay_approx = (U_mag_t ** 2) * torch.abs(kappa_cur_signed)
            beta_t = torch.atan2(s[..., 1], torch.clamp(s[..., 0], min=1e-6))
        cost = cost + self.w_ay * (ay_approx ** 2) * G_turn
        cost = cost + self.w_beta * (beta_t ** 2) * (0.5 + 0.5 * G_turn)
        cost = cost + self.w_crab * ((df_t + dr_t) ** 2) * G_turn

        U_t_cur = s[..., 5] if self.model_type == '2dof' else U_mag_t
        s_lin = torch.clamp((U_t_cur - self.U1) / (self.U2 - self.U1 + 1e-12), 0.0, 1.0)
        k_t = (-self.k_max * G_turn) + (self.k_high * (1.0 - G_turn) * s_lin)
        w_phase = self.w_phase_base * (1.0 + 2.0 * G_turn)
        phase_err = dr_t - k_t * df_t
        cost = cost + w_phase * (phase_err ** 2)

        same_sign_pos = torch.clamp(df_t * dr_t, min=0.0)
        cost = cost + (self.w_phase_sign * G_turn) * (same_sign_pos ** 2)

        p = self.params
        U_eff = max(abs(p.U), p.U_min)
        U_eff_t = torch.tensor(U_eff, dtype=dtype, device=device)
        Lc = p.a + p.b
        Kc = (p.m / Lc) * (p.b / p.kr - p.a / p.kf)
        r_ref_coeff = U_eff / (Lc + Kc * U_eff * U_eff)
        r_ref = r_ref_coeff * df_t
        if U_eff > 0.3:
            r_max = p.mu * p.g / U_eff
            r_max_t = torch.tensor(r_max, dtype=dtype, device=device)
            r_cmd = torch.clamp(r_ref, -r_max_t, r_max_t)
        else:
            r_cmd = r_ref
        T1 = (-(p.a * p.kf - p.b * p.kr) / U_eff - p.m * U_eff) * r_cmd
        numerator = (
            p.a * p.kf * df_t * (p.kf + p.kr)
            - (p.a * p.kf - p.b * p.kr) * (p.kf * df_t + T1)
            - (p.a ** 2 * p.kf + p.b ** 2 * p.kr) * r_cmd * (p.kf + p.kr) / U_eff_t
        )
        denom = p.kf * p.kr * (p.a + p.b)
        delta_r_ff = numerator / denom
        beta_ref = (p.kf * df_t + p.kr * delta_r_ff + T1) / (p.kf + p.kr)
        Kr = 0.2
        Kbeta = 0.0
        r_cur = s[..., 4] if self.model_type == '2dof' else s[..., 2]
        beta_cur = s[..., 3] if self.model_type == '2dof' else torch.atan2(s[..., 1], torch.clamp(s[..., 0], min=1e-6))
        delta_r = delta_r_ff + Kr * (r_cmd - r_cur) + Kbeta * (beta_ref - beta_cur)
        delta_r = torch.clamp(delta_r, -self.delta_max, self.delta_max)
        dr_ff_t = delta_r
        cost = cost + self.w_yaw_ff * ((dr_t - dr_ff_t) ** 2)

        dU_t = u[..., 2]
        cost = cost + self.w_dU_turn * (dU_t ** 2) * G_turn

        if self._last_u is not None:
            du = u - self._last_u
            cost = cost + self.w_du * torch.sum(du * du, dim=-1)

        return cost.squeeze(0) if single else cost

    def command(self, s_np: np.ndarray) -> np.ndarray:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device)
        u = self.mppi.command(s)
        # 缓存上一动作
        self._last_u = u.detach()
        return u.detach().cpu().numpy()