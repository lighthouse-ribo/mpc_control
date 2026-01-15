import torch
from typing import Optional, Tuple

# 复用 numpy 版本中的参数数据类，避免重复定义
from .tire import PacejkaParams, PacejkaLongParams


def pacejka_lateral_torch(
    alpha: torch.Tensor,
    Fz: torch.Tensor,
    p: Optional[PacejkaParams] = None,
) -> torch.Tensor:
    """Pacejka 魔术公式（纯侧偏）Torch 版。

    参数
    - alpha: 侧偏角 [rad]，支持标量或批量张量
    - Fz: 法向载荷 [N]，与 alpha 可广播
    - p: 轮胎参数；默认使用 PacejkaParams()

    返回
    - Fy: 横向力 [N]，方向约定与 numpy 版一致（alpha>0 产生负向恢复力）
    """
    if p is None:
        p = PacejkaParams()

    # 载荷非负
    Fz_pos = torch.clamp(Fz, min=0.0)
    D = float(p.mu_y) * Fz_pos

    Ba = float(p.B) * alpha
    atan_Ba = torch.atan(Ba)
    inner = Ba - float(p.E) * (Ba - atan_Ba)
    Fy = -D * torch.sin(float(p.C) * torch.atan(inner))
    return Fy


def pacejka_longitudinal_torch(
    lmbd: torch.Tensor,
    Fz: torch.Tensor,
    p: Optional[PacejkaLongParams] = None,
) -> torch.Tensor:
    """Pacejka 魔术公式（纯纵向滑移）Torch 版。

    参数
    - lmbd: 纵向滑移率 [-]
    - Fz: 法向载荷 [N]
    - p: 轮胎纵向参数；默认 PacejkaLongParams()

    返回
    - Fx: 纵向力 [N]
    """
    if p is None:
        p = PacejkaLongParams()

    Fz_pos = torch.clamp(Fz, min=0.0)
    D = float(p.mu_x) * Fz_pos
    Bl = float(p.B) * lmbd
    atan_Bl = torch.atan(Bl)
    inner = Bl - float(p.E) * (Bl - atan_Bl)
    Fx = D * torch.sin(float(p.C) * torch.atan(inner))
    return Fx


def combine_friction_ellipse_torch(
    Fx_pure: torch.Tensor,
    Fy_pure: torch.Tensor,
    Fz: torch.Tensor,
    mu_x: float,
    mu_y: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """摩擦椭圆合成（Torch 版）。

    约束 (Fx/(mu_x*Fz))^2 + (Fy/(mu_y*Fz))^2 <= 1，
    若超出，则按半径缩放至边界；支持广播。
    """
    Fz_pos = torch.clamp(Fz, min=0.0)
    tiny = torch.tensor(1e-12, dtype=Fz.dtype, device=Fz.device)
    denom_x = float(mu_x) * Fz_pos + tiny
    denom_y = float(mu_y) * Fz_pos + tiny
    nx = Fx_pure / denom_x
    ny = Fy_pure / denom_y
    radius2 = nx * nx + ny * ny
    scale = torch.where(radius2 <= 1.0, torch.ones_like(radius2), 1.0 / torch.sqrt(radius2))
    Fx = Fx_pure * scale
    Fy = Fy_pure * scale
    # Fz≈0 时，输出零力
    zero_mask = Fz_pos <= 1e-6
    Fx = torch.where(zero_mask, torch.zeros_like(Fx), Fx)
    Fy = torch.where(zero_mask, torch.zeros_like(Fy), Fy)
    return Fx, Fy


def lateral_force_dispatch_torch(
    alpha: torch.Tensor,
    Fz: torch.Tensor,
    model: str,
    linear_k: float | None,
    p_params: PacejkaParams | None,
) -> torch.Tensor:
    """横向力派发（Torch 版）。

    - model='linear' 时：Fy=-k*alpha（与约定一致）
    - 其他：使用 Pacejka 公式
    """
    m = (model or 'linear').lower().strip()
    if m == 'linear':
        k = float(linear_k or 0.0)
        return torch.tensor(k, dtype=alpha.dtype, device=alpha.device) * (-alpha)
    params = p_params or PacejkaParams()
    return pacejka_lateral_torch(alpha, Fz, params)