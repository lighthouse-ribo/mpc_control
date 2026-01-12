import math
from typing import Tuple


def static_loads(a: float, b: float, m: float, g: float) -> Tuple[float, float]:
    """Generic front/rear static vertical loads (ignore load transfer).

    Inputs match bicycle model geometry: CG distances `a` (front), `b` (rear), mass `m`, gravity `g`.
    Returns (Fzf, Fzr).
    """
    L = a + b
    if abs(L) < 1e-9:
        L = 1e-9
    Fzf = m * g * (b / L)
    Fzr = m * g * (a / L)
    return float(Fzf), float(Fzr)


def yaw_rate_limit(mu_y: float, g: float, vx_eff: float, eps: float = 1e-6) -> float:
    """Friction-limited yaw-rate bound: |r| ≤ mu_y * g / vx_eff.

    If vx_eff is near zero, return +inf to avoid false saturation.
    """
    vx_mag = abs(vx_eff)
    if vx_mag < eps:
        return float('inf')
    return float(mu_y * g / vx_mag)


def apply_yaw_saturation(r: float, r_dot: float, mu_y: float, g: float, vx_eff: float, gain: float) -> float:
    """Apply saturation damping when |r| exceeds friction-limited yaw-rate bound.

    Returns adjusted r_dot.
    """
    r_max = yaw_rate_limit(mu_y, g, vx_eff)
    if abs(r) > r_max:
        r_dot -= float(gain) * (abs(r) - r_max) * math.copysign(1.0, r)
    return float(r_dot)


def body_to_world_2dof(U: float, beta: float, psi: float) -> Tuple[float, float]:
    """2DOF 车体→世界坐标速度变换。

    x_dot = U * cos(psi + beta)
    y_dot = U * sin(psi + beta)
    """
    x_dot = float(U) * math.cos(float(psi) + float(beta))
    y_dot = float(U) * math.sin(float(psi) + float(beta))
    return x_dot, y_dot


def body_to_world_3dof(vx: float, vy: float, psi: float) -> Tuple[float, float]:
    """3DOF 车体→世界坐标速度变换。

    x_dot = vx * cos(psi) - vy * sin(psi)
    y_dot = vx * sin(psi) + vy * cos(psi)
    """
    c = math.cos(float(psi))
    s = math.sin(float(psi))
    x_dot = float(vx) * c - float(vy) * s
    y_dot = float(vx) * s + float(vy) * c
    return x_dot, y_dot


def slip_angles_2dof(beta: float, r: float, df: float, dr: float, a: float, b: float, U: float) -> Tuple[float, float]:
    """2DOF 自行车模型侧偏角：alpha_f/r 由 beta、r、df/dr 与几何参数计算。"""
    alpha_f = float(beta) + float(a) * float(r) / float(U) - float(df)
    alpha_r = float(beta) - float(b) * float(r) / float(U) - float(dr)
    return float(alpha_f), float(alpha_r)


def slip_angles_3dof(vx: float, vy: float, r: float, df: float, dr: float, a: float, b: float, U_min: float) -> Tuple[float, float]:
    """3DOF 自行车模型侧偏角：alpha = atan2(vy ± a/b * r, vx_eff) - d。

    说明
    - 低速/近零 vx 时，直接使用原始 vx 计算会导致 |alpha| 接近 π/2，产生不合理的横向力；
    - 使用 `vx_eff = sign(vx) * max(|vx|, U_min)` 进行近零保护，保持方向一致同时避免极端侧偏角；
    """
    vx_eff = math.copysign(max(float(U_min), abs(float(vx))), float(vx))
    alpha_f = math.atan2(float(vy) + float(a) * float(r), float(vx_eff)) - float(df)
    alpha_r = math.atan2(float(vy) - float(b) * float(r), float(vx_eff)) - float(dr)
    return float(alpha_f), float(alpha_r)


def curvature_4ws(df: float, dr: float, L: float) -> float:
    """4WS 等效自行车曲率 κ 近似：κ ≈ (tan(df) - tan(dr)) / L。

    - 小角度下与经典自行车模型一致，适用于低速几何融合。
    - 提供稳定的曲率估计用于 r 指令：r_des = U_signed * κ。
    """
    L_eff = max(1e-9, float(L))
    return (math.tan(float(df)) - math.tan(float(dr))) / L_eff