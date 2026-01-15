import numpy as np
from typing import Tuple, Dict

from .params import VehicleParams
from .tire import pacejka_lateral, PacejkaParams, lateral_force_dispatch
from .dof_utils import static_loads, slip_angles_2dof


def matrices(p: VehicleParams) -> Tuple[np.ndarray, np.ndarray]:
    """根据论文模型生成状态空间矩阵 A、B。
    状态 X=[beta, r]^T；输入 U_in=[delta_f, delta_r]^T
    """
    m, Iz, a, b, kf, kr, U = p.m, p.Iz, p.a, p.b, p.kf, p.kr, p.U_eff()

    A11 = - (kf + kr) / (m * U)
    A12 = - (a * kf - b * kr) / (m * U * U) - 1.0
    A21 = - (a * kf - b * kr) / Iz
    A22 = - (a * a * kf + b * b * kr) / (Iz * U)

    B11 = kf / (m * U)
    B12 = kr / (m * U)
    B21 = a * kf / Iz
    B22 = - b * kr / Iz

    A = np.array([[A11, A12], [A21, A22]], dtype=float)
    B = np.array([[B11, B12], [B21, B22]], dtype=float)
    return A, B


def derivatives(x: np.ndarray, delta_f: float, delta_r: float, p: VehicleParams) -> Dict[str, float | np.ndarray]:
    """计算状态导数与观测量（2DOF）。
    支持两种轮胎模型：
    - linear：线性侧偏刚度（与原 A/B 矩阵一致）
    - pacejka：Pacejka 魔术方程（非线性）

    返回：xdot, ay, Fy_f, Fy_r, alpha_f, alpha_r
    """
    beta, r = float(x[0]), float(x[1])
    U = p.U_eff()

    # 侧偏角与横向力
    alpha_f, alpha_r = slip_angles_2dof(beta, r, delta_f, delta_r, p.a, p.b, U)
    Fy_f, Fy_r = lateral_forces_2dof(alpha_f, alpha_r, p)

    # 动力学方程（2DOF）：m*U*(beta_dot + r) = Fy_f + Fy_r；Iz*r_dot = a*Fy_f - b*Fy_r
    beta_dot = (Fy_f + Fy_r) / (p.m * U) - r
    r_dot = (p.a * Fy_f - p.b * Fy_r) / p.Iz

    # 横向加速度（观测量）
    ay = U * (r + beta_dot)

    return {
        "xdot": np.array([float(beta_dot), float(r_dot)]),
        "ay": float(ay),
        "Fy_f": float(Fy_f),
        "Fy_r": float(Fy_r),
        "alpha_f": float(alpha_f),
        "alpha_r": float(alpha_r),
    }
    


def static_loads_2dof(p: VehicleParams) -> Tuple[float, float]:
    """前后轴静态法向载荷（忽略载荷转移）。"""
    Fzf, Fzr = static_loads(float(p.a), float(p.b), float(p.m), float(p.g))
    return float(Fzf), float(Fzr)


def lateral_forces_2dof(alpha_f: float, alpha_r: float, p: VehicleParams) -> Tuple[float, float]:
    """根据选择的轮胎模型计算前/后轮横向力（统一派发）。"""
    model_sel = (p.tire_model or 'linear').lower()
    Fzf, Fzr = static_loads_2dof(p)
    tp_f = PacejkaParams(mu_y=float(p.mu))
    tp_r = PacejkaParams(mu_y=float(p.mu))
    Fy_f = lateral_force_dispatch(float(alpha_f), Fzf, model_sel, p.kf, tp_f)
    Fy_r = lateral_force_dispatch(float(alpha_r), Fzr, model_sel, p.kr, tp_r)
    return float(Fy_f), float(Fy_r)