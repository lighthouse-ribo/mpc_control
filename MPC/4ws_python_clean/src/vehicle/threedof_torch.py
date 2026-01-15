import torch
from typing import Tuple, Dict, Optional

# 直接复用 numpy 版本中的参数数据类，避免重复定义
from .threedof import Vehicle3DOF
from .tire import PacejkaParams, PacejkaLongParams
from .tire_torch import (
    lateral_force_dispatch_torch,
    pacejka_longitudinal_torch,
    combine_friction_ellipse_torch,
)


def static_loads_3dof_torch(
    a: float,
    b: float,
    m: float,
    g: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """前后轴静态法向载荷（忽略载荷转移，Torch 版）。"""
    L = a + b
    if abs(L) < 1e-9:
        L = 1e-9
    Fzf = m * g * (b / L)
    Fzr = m * g * (a / L)
    return (
        torch.tensor(Fzf, dtype=dtype, device=device),
        torch.tensor(Fzr, dtype=dtype, device=device),
    )


def slip_angles_3dof_torch(
    vx: torch.Tensor,
    vy: torch.Tensor,
    r: torch.Tensor,
    df: torch.Tensor,
    dr: torch.Tensor,
    a: float,
    b: float,
    U_min: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3DOF 自行车模型侧偏角（Torch 版）。

    alpha_f = atan2(vy + a*r, vx) - df
    alpha_r = atan2(vy - b*r, vx) - dr
    近零保护：使用 vx_eff = sign(vx) * max(|vx|, U_min) 避免低速时 |alpha| 过大。
    """
    # 低速近零保护（保持符号）
    U_min_t = torch.tensor(float(U_min), dtype=vx.dtype, device=vx.device)
    vx_mag = torch.abs(vx)
    vx_eff = torch.where(vx_mag < U_min_t, torch.sign(vx) * U_min_t, vx)
    alpha_f = torch.atan2(vy + (a * r), vx_eff) - df
    alpha_r = torch.atan2(vy - (b * r), vx_eff) - dr
    return alpha_f, alpha_r


def yaw_rate_limit_torch(mu_y: float, g: float, vx_eff: torch.Tensor) -> torch.Tensor:
    """附着-受限的横摆率边界 |r| ≤ mu_y * g / vx_eff（Torch 版）。"""
    vx_mag = torch.abs(vx_eff)
    tiny = torch.tensor(1e-9, dtype=vx_mag.dtype, device=vx_mag.device)
    return (float(mu_y) * float(g)) / torch.clamp(vx_mag, min=tiny)


def apply_yaw_saturation_torch(
    r: torch.Tensor,
    r_dot: torch.Tensor,
    mu_y: float,
    g: float,
    vx_eff: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    """当 |r| 超出边界时施加附加阻尼（Torch 版）。返回调整后的 r_dot。"""
    r_max = yaw_rate_limit_torch(mu_y, g, vx_eff)
    exceed = torch.abs(r) > r_max
    # 仅在超出时：r_dot -= gain * (|r|-r_max) * sign(r)
    correction = float(gain) * (torch.abs(r) - r_max) * torch.sign(r)
    r_dot_adj = torch.where(exceed, r_dot - correction, r_dot)
    return r_dot_adj


def allocate_drive_torch(
    Fx_total: torch.Tensor,
    df: torch.Tensor,
    dr: torch.Tensor,
    front_bias: float,
    rear_bias: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """角度感知的纵向力分配（Torch 版）。

    使用 cos^2 衰减与轴向比例权重，将 Fx_total 分配到前/后轴。
    """
    cosdf2 = torch.cos(df) ** 2
    cosdr2 = torch.cos(dr) ** 2
    front_share = float(front_bias) * cosdf2
    rear_share = float(rear_bias) * cosdr2
    share_sum = front_share + rear_share
    tiny = torch.tensor(1e-9, dtype=Fx_total.dtype, device=Fx_total.device)
    Fx_f_pure = Fx_total * front_share / torch.clamp(share_sum, min=tiny)
    Fx_r_pure = Fx_total * rear_share / torch.clamp(share_sum, min=tiny)
    return Fx_f_pure, Fx_r_pure


def tire_forces_3dof_torch(
    alpha_f: torch.Tensor,
    alpha_r: torch.Tensor,
    vp: Vehicle3DOF,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据选择的轮胎模型计算前/后轮横向力（Torch 版）。"""
    Fzf, Fzr = static_loads_3dof_torch(vp.a, vp.b, vp.m, vp.g, device=device, dtype=dtype)
    model_sel = (vp.tire_model or 'linear').lower()
    Fy_f = lateral_force_dispatch_torch(alpha_f, Fzf, model_sel, vp.kf, vp.tire_params_f)
    Fy_r = lateral_force_dispatch_torch(alpha_r, Fzr, model_sel, vp.kr, vp.tire_params_r)
    return Fy_f, Fy_r


def derivatives_dfdr_torch(
    s: torch.Tensor,
    df: torch.Tensor,
    dr: torch.Tensor,
    vp: Vehicle3DOF,
    Fx_f_pure: Optional[torch.Tensor] = None,
    Fx_r_pure: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """3DOF 非线性动力学的 Torch 版导数计算。

    输入
    - s: [..., 6] 张量，内容为 [vx, vy, r, x, y, psi]
    - df, dr: 轮角（广播兼容）
    - vp: 车辆参数（numpy dataclass，可直接读取标量）
    - Fx_f_pure, Fx_r_pure: 纯纵向力（可选；若未给出则视为 0）

    输出
    - ds: [..., 6] 张量导数
    - aux: 字典，包含中间量（alpha_f/r、Fy/Fx 等）便于诊断
    """
    if s.ndim == 1:
        s = s.unsqueeze(0)

    vx, vy, r, x, y, psi = s.unbind(-1)

    # 侧偏角与横向力
    alpha_f, alpha_r = slip_angles_3dof_torch(vx, vy, r, df, dr, float(vp.a), float(vp.b), float(vp.U_min))
    Fy_f_pure, Fy_r_pure = tire_forces_3dof_torch(alpha_f, alpha_r, vp, device=s.device, dtype=s.dtype)

    # 静载荷
    Fzf, Fzr = static_loads_3dof_torch(vp.a, vp.b, vp.m, vp.g, device=s.device, dtype=s.dtype)

    # 纵向纯力：若未提供则置零
    zero = torch.zeros((), dtype=s.dtype, device=s.device)
    Fx_f_pure_t = Fx_f_pure if Fx_f_pure is not None else zero
    Fx_r_pure_t = Fx_r_pure if Fx_r_pure is not None else zero

    # 摩擦椭圆合成
    Fx_f, Fy_f = combine_friction_ellipse_torch(
        Fx_f_pure_t, Fy_f_pure, Fzf, float(vp.tire_long_params_f.mu_x), float(vp.tire_params_f.mu_y)
    )
    Fx_r, Fy_r = combine_friction_ellipse_torch(
        Fx_r_pure_t, Fy_r_pure, Fzr, float(vp.tire_long_params_r.mu_x), float(vp.tire_params_r.mu_y)
    )

    # 动力学：vx_dot, vy_dot
    inv_m = torch.tensor(1.0 / float(vp.m), dtype=s.dtype, device=s.device)
    vx_dot = inv_m * (
        Fx_f * torch.cos(df) - Fy_f * torch.sin(df) + Fx_r * torch.cos(dr) - Fy_r * torch.sin(dr)
        - float(vp.Fwx) - float(vp.Fsx)
    ) + r * vy

    vy_dot = inv_m * (
        Fx_f * torch.sin(df) + Fy_f * torch.cos(df) + Fx_r * torch.sin(dr) + Fy_r * torch.cos(dr)
        + float(vp.Fwy) + float(vp.Fsy)
    ) - r * vx

    # 横摆力矩与 r_dot（含阻尼与饱和）
    Mz = (
        (Fx_f * torch.sin(df) + Fy_f * torch.cos(df)) * float(vp.a)
        - (Fx_r * torch.sin(dr) + Fy_r * torch.cos(dr)) * float(vp.b)
    )
    r_dot_raw = (Mz - float(vp.yaw_damp) * r) / float(vp.Iz)
    vx_eff = torch.clamp(torch.abs(vx), min=float(vp.U_min))
    mu_y = min(float(vp.tire_params_f.mu_y), float(vp.tire_params_r.mu_y))
    r_dot = apply_yaw_saturation_torch(r, r_dot_raw, mu_y, float(vp.g), vx_eff, float(vp.yaw_sat_gain))

    # 姿态与世界坐标速度
    x_dot = vx * torch.cos(psi) - vy * torch.sin(psi)
    y_dot = vx * torch.sin(psi) + vy * torch.cos(psi)
    psi_dot = r

    ds = torch.stack([vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot], dim=-1)

    ay = vy_dot + r * vx
    aux = {
        'df': df,
        'dr': dr,
        'alpha_f': alpha_f,
        'alpha_r': alpha_r,
        'Fy_f': Fy_f,
        'Fy_r': Fy_r,
        'Fx_f': Fx_f,
        'Fx_r': Fx_r,
        'ay': ay,
    }
    return ds.squeeze(0), aux


def derivatives_speed_cmd_torch(
    s: torch.Tensor,
    df: torch.Tensor,
    dr: torch.Tensor,
    vp: Vehicle3DOF,
    U_cmd: torch.Tensor,
    k_v: float = 1.0,
    front_bias: float = 0.5,
    rear_bias: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """基于速度指令的 3DOF 导数计算（为 MPPI 使用）。

    - 使用 ax_cmd = k_v * (U_cmd - vx) 生成总纵向力 Fx_total
    - 按角度感知与轴向比例分配到前后轴
    - 经摩擦椭圆与侧偏力组合后进入动力学
    """
    if s.ndim == 1:
        s = s.unsqueeze(0)

    vx, vy, r, x, y, psi = s.unbind(-1)

    # 纵向驱动/制动（纯力）
    ax_cmd = float(k_v) * (U_cmd - vx)
    Fx_total = ax_cmd * float(vp.m)
    Fx_f_pure, Fx_r_pure = allocate_drive_torch(Fx_total, df, dr, front_bias, rear_bias)

    # 调用主导数
    ds, aux = derivatives_dfdr_torch(s, df, dr, vp, Fx_f_pure, Fx_r_pure)
    aux.update({'U_cmd': U_cmd, 'ax_cmd': ax_cmd})
    return ds.squeeze(0), aux