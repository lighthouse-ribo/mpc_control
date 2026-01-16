from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PacejkaParams:
    """
    Pacejka magic formula parameters for lateral force (pure slip).

    D ≈ mu_y * Fz, C ~ 1.9, B ~ 10 (shape), E ~ 0.97 (curvature).
    These are reasonable defaults for passenger car tires; tune as needed.
    """
    B: float = 10.0
    C: float = 1.9
    E: float = 0.97
    mu_y: float = 0.9  # peak lateral friction coefficient


def pacejka_lateral(alpha: float, Fz: float, p: Optional[PacejkaParams] = None) -> float:
    """
    Compute lateral tire force Fy using the simplified Pacejka magic formula for pure slip.

    Parameters
    - alpha: slip angle [rad]
    - Fz: vertical load [N]
    - p: parameters; if None, use defaults

    Returns
    - Fy [N]
    """
    if p is None:
        p = PacejkaParams()

    # Clamp Fz to non-negative
    Fz = max(0.0, Fz)
    D = p.mu_y * Fz

    # Magic formula core
    # Fy = D * sin(C * arctan(B*alpha - E*(B*alpha - arctan(B*alpha))))
    Ba = p.B * alpha
    atan_Ba = np.arctan(Ba)
    inner = Ba - p.E * (Ba - atan_Ba)
    # 方向约定：α>0（车体左向侧偏）产生向右的恢复力，因此取负号
    Fy = -D * np.sin(p.C * np.arctan(inner))

    return Fy


@dataclass
class PacejkaLongParams:
    """
    Pacejka magic formula parameters for longitudinal force (pure slip).
    """
    B: float = 12.0
    C: float = 1.9
    E: float = 1.0
    mu_x: float = 0.95  # peak longitudinal friction coefficient


def pacejka_longitudinal(lmbd: float, Fz: float, p: Optional[PacejkaLongParams] = None) -> float:
    """
    Compute longitudinal tire force Fx (pure longitudinal slip).

    Parameters
    - lmbd: longitudinal slip ratio [-]
    - Fz: vertical load [N]
    - p: parameters
    Returns Fx [N]
    """
    if p is None:
        p = PacejkaLongParams()
    Fz = max(0.0, Fz)
    D = p.mu_x * Fz
    Bl = p.B * lmbd
    atan_Bl = np.arctan(Bl)
    inner = Bl - p.E * (Bl - atan_Bl)
    Fx = D * np.sin(p.C * np.arctan(inner))
    return Fx


def combine_friction_ellipse(
    Fx_pure: float,
    Fy_pure: float,
    Fz: float,
    mu_x: float,
    mu_y: float,
) -> Tuple[float, float]:
    """
    Combine longitudinal and lateral forces via a simple friction ellipse.

    Scales (Fx_pure, Fy_pure) so that:
        (Fx / (mu_x*Fz))^2 + (Fy / (mu_y*Fz))^2 <= 1

    Returns scaled (Fx, Fy).
    """
    Fz = max(0.0, Fz)
    if Fz <= 1e-6:
        return 0.0, 0.0
    nx = Fx_pure / (mu_x * Fz + 1e-12)
    ny = Fy_pure / (mu_y * Fz + 1e-12)
    radius2 = nx * nx + ny * ny
    if radius2 <= 1.0:
        return Fx_pure, Fy_pure
    scale = 1.0 / np.sqrt(radius2)
    return Fx_pure * scale, Fy_pure * scale


def lateral_force_dispatch(
    alpha: float,
    Fz: float,
    model: str,
    linear_k: float | None,
    p_params: PacejkaParams | None,
) -> float:
    """派发横向力计算：线性或 Pacejka。

    - linear：返回 -k * alpha
    - 其他：使用 Pacejka 公式
    """
    m = (model or 'linear').lower().strip()
    if m == 'linear':
        k = float(linear_k or 0.0)
        return float(-k * float(alpha))
    params = p_params or PacejkaParams()
    return float(pacejka_lateral(float(alpha), float(Fz), params))