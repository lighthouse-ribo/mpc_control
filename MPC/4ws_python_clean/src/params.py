from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    """车辆参数与派生量，按论文口径。

    字段单位：SI（m, kg, s, rad, N）
    - m: 车辆质量
    - Iz: 横摆转动惯量
    - a: 质心到前轴距离
    - b: 质心到后轴距离
    - width: 车身宽度（用于前端绘制矩形）
    - kf: 前轴侧偏刚度（轴级聚合）
    - kr: 后轴侧偏刚度（轴级聚合）
    - U: 纵向速度（常值或缓变，仿真中可更新）
    - mu: 附着系数（用于摩擦极限约束）
    - g: 重力加速度
    - U_min: 速度下限，避免 r/U 奇异
    """

    m: float = 35000.0
    Iz: float = 500000.0
    a: float = 8.0
    b: float = 8.0
    width: float = 3.4
    track: float = 3.4
    kf: float = 450000.0
    kr: float = 450000.0
    U: float = 2.2
    mu: float = 0.85
    g: float = 9.81
    U_min: float = 0.3
    # 低速融合阈值：当 |U| < U_blend 时，逐渐转为几何模型以稳定数值
    U_blend: float = 0.3
    # 轮胎模型选择：'pacejka'（魔术方程）或 'linear'（线性侧偏刚度）
    tire_model: str = 'linear'

    @property
    def L(self) -> float:
        return self.a + self.b

    def U_eff(self) -> float:
        """数值保护下的有效速度幅值（允许 U 为负）。

        - 返回 `max(abs(U), U_min)`，解除 0.5 m/s 下限但保留近零保护。
        - 这样动力学中的 `r/U` 等项不会在 U→0 时奇异。
        """
        return max(abs(self.U), self.U_min)

    def understeer_gradient(self) -> float:
        """稳定性因数 K（Understeer Gradient）默认口径。
        K = (m/L) * (b/kr - a/kf)
        可根据具体论文口径在配置阶段替换。
        """
        return (self.m / self.L) * (self.b / self.kr - self.a / self.kf)

    def r_ref(self, delta_f: float) -> float:
        """理想稳态横摆率（未施加摩擦极限约束）。
        r_ref = U / (L + K U^2) * delta_f
        """
        U = self.U_eff()
        K = self.understeer_gradient()
        return (U / (self.L + K * U * U)) * float(delta_f)

    def yaw_rate_cmd(self, delta_f: float) -> float:
        """考虑摩擦极限的横摆率指令 r_cmd。
        |r| ≤ mu * g / U
        """
        r = self.r_ref(delta_f)
        U_mag = self.U_eff()
        # 近零速度时不做限幅（返回 +inf），避免过度抑制
        r_max = float('inf') if U_mag < 1e-6 else float(self.mu * self.g / U_mag)
        return float(np.clip(r, -r_max, r_max))

    def to_dict(self) -> dict:
        return {
            "m": self.m,
            "Iz": self.Iz,
            "a": self.a,
            "b": self.b,
            "width": self.width,
            "track": self.track,
            "kf": self.kf,
            "kr": self.kr,
            "U": self.U,
            "mu": self.mu,
            "g": self.g,
            "U_min": self.U_min,
            "U_blend": self.U_blend,
            "tire_model": self.tire_model,
            "L": self.L,
            "K": self.understeer_gradient(),
            # 兼容新增嵌套结构（前端当前不使用，作为附加信息返回）
            "tire": {
                "model": self.tire_model,
                "lateral": {"mu_y": self.mu},
                "longitudinal": {"mu_x": self.mu},
            },
        }