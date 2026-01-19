import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# [修复1] 修改导入路径，指向 src/vehicle/params.py
# 注意：假设 strategy.py 在 src/ 下，且 vehicle 包也在 src/ 下
try:
    from .vehicle.params import VehicleParams
except ImportError:
    # 兼容性导入，防止从不同层级调用时路径不一致
    from src.vehicle.params import VehicleParams


@dataclass
class RatioConfig:
    """速度相关比例律（同/反相）的参数。"""
    k_low: float = -0.3   # 低速反相比例
    k_high: float = 0.1   # 高速同相比例
    U1: float = 5.0       # 低速阈值（m/s）
    U2: float = 20.0      # 高速阈值（m/s）
    delta_max: float = 0.5  # 后轮转角限幅（rad）


def ratio_rear_steer(delta_f: float, p: VehicleParams, cfg: RatioConfig = RatioConfig()) -> float:
    """按速度插值的前后轮比例律：delta_r = k(U) * delta_f。"""
    U = p.U_eff()
    if U <= cfg.U1:
        k = cfg.k_low
    elif U >= cfg.U2:
        k = cfg.k_high
    else:
        s = (U - cfg.U1) / (cfg.U2 - cfg.U1)
        k = cfg.k_low * (1.0 - s) + cfg.k_high * s
    delta_r = k * float(delta_f)
    return float(np.clip(delta_r, -cfg.delta_max, cfg.delta_max))


@dataclass
class TrackingConfig:
    """理想横摆率跟踪（前馈+反馈）参数。"""
    Kr: float = 0.2       # r 误差反馈增益
    Kbeta: float = 0.0    # beta 误差反馈增益（可留 0）
    delta_max: float = 0.5


def ideal_yaw_rate(delta_f: float, x: np.ndarray, p: VehicleParams, cfg: TrackingConfig = TrackingConfig()) -> Tuple[float, Dict[str, float]]:
    """按论文口径构造横摆率跟踪的后轮转角。
    - 先计算 r_cmd（含摩擦极限），再由稳态条件（beta_dot=0, r_dot=0）给出前馈 delta_r_ff。
    - 叠加反馈项并限幅。
    返回：(delta_r, 诊断字典)
    """
    beta, r = float(x[0]), float(x[1])

    # 目标横摆率（含摩擦极限）
    r_cmd = p.yaw_rate_cmd(delta_f)
    U = p.U_eff()
    
    # [修复2] 增加极小值保护，防止除以零
    U_safe = max(1e-3, abs(U)) 

    # 来自 beta_dot=0 的中间项（论文式(3)展开中相应项）
    # 使用 U_safe 代替 U 进行除法运算
    T1 = (-(p.a * p.kf - p.b * p.kr) / U_safe - p.m * U) * r_cmd

    # 由两式联立解得 delta_r_ff（稳态解）
    numerator = (
        p.a * p.kf * delta_f * (p.kf + p.kr)
        - (p.a * p.kf - p.b * p.kr) * (p.kf * delta_f + T1)
        - (p.a ** 2 * p.kf + p.b ** 2 * p.kr) * r_cmd * (p.kf + p.kr) / U_safe
    )
    delta_r_ff = numerator / (p.kf * p.kr * (p.a + p.b))

    # beta 的稳态参考值，可用于反馈
    beta_ref = (p.kf * delta_f + p.kr * delta_r_ff + T1) / (p.kf + p.kr)

    # 反馈组合
    delta_r = delta_r_ff + cfg.Kr * (r_cmd - r) + cfg.Kbeta * (beta_ref - beta)

    # 限幅
    delta_r = float(np.clip(delta_r, -cfg.delta_max, cfg.delta_max))

    return delta_r, {"r_cmd": r_cmd, "delta_r_ff": delta_r_ff, "beta_ref": float(beta_ref)}