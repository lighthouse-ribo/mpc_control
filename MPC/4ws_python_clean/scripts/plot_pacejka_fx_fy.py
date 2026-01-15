#!/usr/bin/env python3
"""
Plot Fx–Fy relationship from Pacejka (Torch) for given steering angles.

Angles: 5°, 10°, 15°, 20°, 25°, 30°
Uses current default parameters from src.tire and src.params.
Saves figure to assets/pacejka_fx_fy_by_angle.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH so `src` imports work
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from src.tire_torch import (
    pacejka_lateral_torch,
    pacejka_longitudinal_torch,
    combine_friction_ellipse_torch,
)
from src.tire import PacejkaParams, PacejkaLongParams
from src.params import VehicleParams


def main() -> None:
    # Steering angles in degrees
    angles_deg = [5, 10, 15, 20, 25, 30]

    # Use current vehicle default params for mass and gravity
    vp = VehicleParams()
    # Approximate per-tire static vertical load (quarter-car)
    Fz_tire = float(vp.m * vp.g / 4.0)

    # Use global mu to override tire params for consistency
    p_lat = PacejkaParams(mu_y=float(vp.mu))
    p_long = PacejkaLongParams(mu_x=float(vp.mu))
    mu_x = float(p_long.mu_x)
    mu_y = float(p_lat.mu_y)

    # Torch setup
    dtype = torch.float64
    device = torch.device("cpu")
    Fz_t = torch.tensor(Fz_tire, dtype=dtype, device=device)

    # Longitudinal slip range (drive, positive)
    lmbd_vals = torch.linspace(0.0, 1.0, 200, dtype=dtype, device=device)

    # Use PingFang (苹方) font on macOS for proper CJK rendering
    # Try to register system PingFang if present; include robust fallbacks
    try:
        from matplotlib import font_manager as _fm
        _pf_candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/PingFang.ttc",
        ]
        for _p in _pf_candidates:
            if Path(_p).exists():
                try:
                    _fm.addfont(_p)
                except Exception:
                    pass
                break
    except Exception:
        pass
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(9, 6))

    for deg in angles_deg:
        alpha_rad = np.deg2rad(deg)
        alpha_t = torch.tensor(alpha_rad, dtype=dtype, device=device)

        # Pure forces
        Fy_pure = pacejka_lateral_torch(alpha_t, Fz_t, p_lat)  # scalar tensor
        Fx_pure = pacejka_longitudinal_torch(lmbd_vals, Fz_t, p_long)  # vector tensor

        # Combine via friction ellipse constraint (projection to boundary)
        Fx_c, Fy_c = combine_friction_ellipse_torch(Fx_pure, Fy_pure, Fz_t, mu_x, mu_y)


        # To numpy for plotting
        Fx_np = Fx_c.detach().cpu().numpy()
        Fy_np = Fy_c.detach().cpu().numpy()

        ax.plot(Fx_np, Fy_np, label=f"{deg}° 椭圆")

    # Axis helpers
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Fx 驱动力 [N]")
    ax.set_ylabel("Fy 侧偏力 [N]（负轴朝上）")
    ax.set_title("Pacejka 魔术公式：驱动力–侧偏力关系（摩擦椭圆）")
    # 保持几何比例，避免椭圆被拉伸成近似直线
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title="转角")

    # Invert Y so negative Fy appears upward
    ax.invert_yaxis()

    # Expand Fx range to where Fy→0 (up to theoretical μx·Fz)
    x_max = mu_x * Fz_tire * 1.05
    ax.set_xlim(0.0, x_max)

    fig.tight_layout()

    out_path = "assets/pacejka_fx_fy_by_angle.png"
    plt.show()
    plt.savefig(out_path, dpi=160)
    print(f"图已保存到: {out_path}")


if __name__ == "__main__":
    main()