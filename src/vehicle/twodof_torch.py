import torch
from typing import Tuple, Dict

from .params import VehicleParams


def matrices_torch(
    p: VehicleParams,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据论文模型生成状态空间矩阵 A、B（Torch 版）。
    状态 X=[beta, r]^T；输入 U_in=[delta_f, delta_r]^T
    与 src/twodof.py.matrices 保持公式一致。
    """
    U = float(p.U_eff())
    m, Iz, a, b, kf, kr = float(p.m), float(p.Iz), float(p.a), float(p.b), float(p.kf), float(p.kr)

    A11 = - (kf + kr) / (m * U)
    A12 = - (a * kf - b * kr) / (m * U * U) - 1.0
    A21 = - (a * kf - b * kr) / Iz
    A22 = - (a * a * kf + b * b * kr) / (Iz * U)

    B11 = kf / (m * U)
    B12 = kr / (m * U)
    B21 = a * kf / Iz
    B22 = - b * kr / Iz

    A = torch.tensor([[A11, A12], [A21, A22]], dtype=dtype, device=device)
    B = torch.tensor([[B11, B12], [B21, B22]], dtype=dtype, device=device)
    return A, B


def slip_angles_2dof_torch(
    beta: torch.Tensor,
    r: torch.Tensor,
    delta_f: torch.Tensor,
    delta_r: torch.Tensor,
    a: float,
    b: float,
    U: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2DOF 侧偏角（Torch 版）。
    alpha_f = beta + a*r/U - delta_f
    alpha_r = beta - b*r/U - delta_r
    支持标量或批量张量（会按广播规则计算）。
    """
    U_eff = torch.clamp(U, min=1e-6)
    alpha_f = beta + (a * r) / U_eff - delta_f
    alpha_r = beta - (b * r) / U_eff - delta_r
    return alpha_f, alpha_r


def static_loads_2dof_torch(
    a: float,
    b: float,
    m: float,
    g: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """前后轴静态法向载荷（忽略载荷转移，Torch 版）。"""
    Fzf = (b * m * g) / (a + b)
    Fzr = (a * m * g) / (a + b)
    return (
        torch.tensor(Fzf, dtype=dtype, device=device),
        torch.tensor(Fzr, dtype=dtype, device=device),
    )


def lateral_forces_2dof_torch(
    alpha_f: torch.Tensor,
    alpha_r: torch.Tensor,
    p: VehicleParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据选择的轮胎模型计算前/后轮横向力（Torch 版）。

    - 线性模型（推荐用于 MPPI 批量并行）：Fy = -k * alpha（与惯例一致）
    - Pacejka 模型：回退为逐样本 numpy 计算（性能较低），保留一致性选项。
    """
    model_sel = str(getattr(p, 'tire_model', 'linear') or 'linear').lower()
    if model_sel in ('linear', ''):
        kf = torch.tensor(float(p.kf), dtype=alpha_f.dtype, device=alpha_f.device)
        kr = torch.tensor(float(p.kr), dtype=alpha_r.dtype, device=alpha_r.device)
        # 方向约定与 Pacejka 保持一致：α>0（车体左向侧偏）产生向右的恢复力
        # 因此线性模型应为 Fy = -k * alpha（此前为 + 号，属符号错误）
        Fy_f = -kf * alpha_f
        Fy_r = -kr * alpha_r
        return Fy_f, Fy_r

    # 使用 Torch 版 Pacejka，实现张量广播与梯度可用
    from .tire_torch import lateral_force_dispatch_torch
    from .tire import PacejkaParams
    Fzf_t, Fzr_t = static_loads_2dof_torch(
        float(p.a), float(p.b), float(p.m), float(p.g),
        device=alpha_f.device, dtype=alpha_f.dtype,
    )
    tp_f = PacejkaParams(mu_y=float(p.mu))
    tp_r = PacejkaParams(mu_y=float(p.mu))

    Fy_f = lateral_force_dispatch_torch(alpha_f, Fzf_t, model_sel, float(p.kf), tp_f)
    Fy_r = lateral_force_dispatch_torch(alpha_r, Fzr_t, model_sel, float(p.kr), tp_r)
    return Fy_f, Fy_r


def derivatives_torch(
    x: torch.Tensor,
    delta_f: torch.Tensor,
    delta_r: torch.Tensor,
    p: VehicleParams,
) -> Dict[str, torch.Tensor]:
    """计算 2DOF 状态导数与观测量（Torch 版）。

    参数：
    - x: [..., 2]，分别为 beta, r；支持批量维度
    - delta_f, delta_r: 与 x 广播的张量；支持批量
    - p: VehicleParams（数值参数将被常量注入）

    返回：字典，包含 xdot（[..., 2]）、ay、Fy_f、Fy_r、alpha_f、alpha_r
    """
    beta = x[..., 0]
    r = x[..., 1]

    U = torch.tensor(float(p.U_eff()), dtype=x.dtype, device=x.device)
    a = float(p.a)
    b = float(p.b)
    m = float(p.m)
    Iz = float(p.Iz)

    alpha_f, alpha_r = slip_angles_2dof_torch(beta, r, delta_f, delta_r, a, b, U)
    Fy_f, Fy_r = lateral_forces_2dof_torch(alpha_f, alpha_r, p)

    beta_dot = (Fy_f + Fy_r) / (m * U) - r
    r_dot = (a * Fy_f - b * Fy_r) / Iz

    ay = U * (r + beta_dot)

    return {
        "xdot": torch.stack([beta_dot, r_dot], dim=-1),
        "ay": ay,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
    }