from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple
import numpy as np
from .dof_utils import static_loads, yaw_rate_limit, apply_yaw_saturation, slip_angles_3dof

from .tire import (
    PacejkaParams,
    PacejkaLongParams,
    pacejka_lateral,
    pacejka_longitudinal,
    combine_friction_ellipse,
    lateral_force_dispatch,
)


@dataclass
class Vehicle3DOF:
    """
    3-DOF nonlinear lateral dynamics model parameters.

    Coordinates and parameters follow documents/3dof.md.
    """
    m: float = 35000.0            # mass [kg]
    Iz: float = 500000.0           # yaw inertia [kg*m^2]
    a: float = 8.0               # CG to front axle [m]
    b: float = 8.0               # CG to rear axle [m]
    g: float = 9.81              # gravity [m/s^2]
    U_min: float = 0.5           # min longitudinal speed to avoid singularities [m/s]
    # Linear cornering stiffness for optional linear tire model
    kf: float = 450000.0
    kr: float = 450000.0
    # Tire model selection: 'pacejka' or 'linear'
    tire_model: str = 'pacejka'

    # Tire & steering
    tire_params_f: PacejkaParams = field(default_factory=PacejkaParams)
    tire_params_r: PacejkaParams = field(default_factory=PacejkaParams)
    n_sw: float = 16.0           # steering ratio (steering wheel to front wheel)

    # 4WS control (example): delta_r = -c1 * delta_f + c2 * r
    c1: float = 0.3
    c2: float = 0.05

    # Simplified yaw damping torque coefficient [N·m·s/rad]
    # Approximates self-aligning torque aggregate effect to avoid unbounded yaw acceleration
    yaw_damp: float = 300.0
    # Yaw-rate saturation gain (extra damping when |r| exceeds friction-limited bound)
    yaw_sat_gain: float = 4.0

    # Resistances (simplified; set to 0 by default)
    Fwx: float = 0.0
    Fwy: float = 0.0
    Fsx: float = 0.0
    Fsy: float = 0.0

    # Longitudinal tire parameters (pure slip) for combined condition
    tire_long_params_f: PacejkaLongParams = field(default_factory=PacejkaLongParams)
    tire_long_params_r: PacejkaLongParams = field(default_factory=PacejkaLongParams)

    @property
    def L(self) -> float:
        return self.a + self.b

    def static_loads(self) -> Tuple[float, float]:
        """Front/rear static vertical loads (ignoring load transfer)."""
        return static_loads(float(self.a), float(self.b), float(self.m), float(self.g))


@dataclass
class State3DOF:
    vx: float = 20.0   # longitudinal speed [m/s]
    vy: float = 0.0    # lateral speed [m/s]
    r: float = 0.0     # yaw rate [rad/s]
    x: float = 0.0     # X position [m]
    y: float = 0.0     # Y position [m]
    psi: float = 0.0   # yaw angle [rad]


def control_4ws(delta_sw: float, r: float, vp: Vehicle3DOF) -> Tuple[float, float]:
    """Compute front and rear wheel angles from steering wheel angle and yaw rate."""
    df = delta_sw / vp.n_sw
    dr = -vp.c1 * df + vp.c2 * r
    return df, dr


    


def tire_forces(alpha_f: float, alpha_r: float, vp: Vehicle3DOF) -> Tuple[float, float]:
    """Compute lateral tire forces Fy front/rear based on selected model (dispatch)."""
    Fzf, Fzr = vp.static_loads()
    model_sel = (vp.tire_model or 'linear').lower()
    Fy_f = lateral_force_dispatch(float(alpha_f), Fzf, model_sel, vp.kf, vp.tire_params_f)
    Fy_r = lateral_force_dispatch(float(alpha_r), Fzr, model_sel, vp.kr, vp.tire_params_r)
    return Fy_f, Fy_r


def allocate_drive(Fx_total: float, df: float, dr: float, front_bias: float, rear_bias: float) -> Tuple[float, float]:
    """Distribute longitudinal force to front/rear axles with angle-aware attenuation."""
    cosdf2 = float(np.cos(df) ** 2)
    cosdr2 = float(np.cos(dr) ** 2)
    front_share = front_bias * cosdf2
    rear_share = rear_bias * cosdr2
    share_sum = front_share + rear_share
    if share_sum <= 1e-9:
        return 0.0, Fx_total
    Fx_f_pure = Fx_total * front_share / share_sum
    Fx_r_pure = Fx_total * rear_share / share_sum
    return float(Fx_f_pure), float(Fx_r_pure)


def derivatives_dfdr(
    s: State3DOF,
    df: float,
    dr: float,
    vp: Vehicle3DOF,
    Fx_f_pure: float = 0.0,
    Fx_r_pure: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute derivatives for 3DOF given wheel angles df/dr and external pure longitudinal forces.

    This mirrors `derivatives(...)` but uses df/dr directly and combines longitudinal forces via friction ellipse.
    Returns (ds, aux) like the original.
    """
    alpha_f, alpha_r = slip_angles_3dof(s.vx, s.vy, s.r, df, dr, vp.a, vp.b, vp.U_min)
    Fy_f_pure, Fy_r_pure = tire_forces(alpha_f, alpha_r, vp)
    Fzf, Fzr = vp.static_loads()

    Fx_f, Fy_f = combine_friction_ellipse(
        Fx_f_pure, Fy_f_pure, Fzf, vp.tire_long_params_f.mu_x, vp.tire_params_f.mu_y
    )
    Fx_r, Fy_r = combine_friction_ellipse(
        Fx_r_pure, Fy_r_pure, Fzr, vp.tire_long_params_r.mu_x, vp.tire_params_r.mu_y
    )

    # Dynamics (documents/3dof.md)
    vx_dot = (1.0 / vp.m) * (
        Fx_f * np.cos(df) - Fy_f * np.sin(df) + Fx_r * np.cos(dr) - Fy_r * np.sin(dr)
        - vp.Fwx - vp.Fsx
    ) + s.r * s.vy

    vy_dot = (1.0 / vp.m) * (
        Fx_f * np.sin(df) + Fy_f * np.cos(df) + Fx_r * np.sin(dr) + Fy_r * np.cos(dr)
        + vp.Fwy + vp.Fsy
    ) - s.r * s.vx

    Mz = (
        (Fx_f * np.sin(df) + Fy_f * np.cos(df)) * vp.a
        - (Fx_r * np.sin(dr) + Fy_r * np.cos(dr)) * vp.b
    )

    # Yaw damping and saturation
    r_dot = (Mz - vp.yaw_damp * s.r) / vp.Iz
    # 近零保护使用速度幅值，允许 vx 为负（倒车）
    vx_eff = max(vp.U_min, abs(s.vx))
    mu_y = min(vp.tire_params_f.mu_y, vp.tire_params_r.mu_y)
    r_dot = apply_yaw_saturation(s.r, r_dot, mu_y, vp.g, vx_eff, vp.yaw_sat_gain)

    psi_dot = s.r
    x_dot = s.vx * np.cos(s.psi) - s.vy * np.sin(s.psi)
    y_dot = s.vx * np.sin(s.psi) + s.vy * np.cos(s.psi)

    ds = np.array([vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot], dtype=float)
    ay = vy_dot + s.r * s.vx
    aux = {
        "df": float(df),
        "dr": float(dr),
        "alpha_f": float(alpha_f),
        "alpha_r": float(alpha_r),
        "Fy_f": float(Fy_f),
        "Fy_r": float(Fy_r),
        "Fx_f": float(Fx_f),
        "Fx_r": float(Fx_r),
        "ay": float(ay),
    }
    return ds, aux


def derivatives(s: State3DOF, delta_sw: float, vp: Vehicle3DOF) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute time derivatives for the 3-DOF nonlinear model using steering wheel input.

    Internally calls derivatives_dfdr(df, dr, ...) to avoid duplication.
    """
    df, dr = control_4ws(delta_sw, s.r, vp)
    # Longitudinal slip ratios provided via closures set by simulate()
    lmbd_f = derivatives._lambda_f(s)
    lmbd_r = derivatives._lambda_r(s)
    Fzf, Fzr = vp.static_loads()
    Fx_f_pure = pacejka_longitudinal(lmbd_f, Fzf, vp.tire_long_params_f)
    Fx_r_pure = pacejka_longitudinal(lmbd_r, Fzr, vp.tire_long_params_r)
    ds, aux = derivatives_dfdr(s, float(df), float(dr), vp, float(Fx_f_pure), float(Fx_r_pure))
    return ds, aux


def simulate(
    T: float,
    dt: float,
    vp: Vehicle3DOF,
    s0: State3DOF,
    delta_sw_fn: Callable[[float], float],
    lambda_f_fn: Callable[[float, State3DOF], float] | None = None,
    lambda_r_fn: Callable[[float, State3DOF], float] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run time-domain simulation for given steering wheel input function.

    Outputs arrays for t, vx, vy, r, x, y, psi, df, dr, Fy_f, Fy_r, alpha_f, alpha_r.
    """
    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, T, N)

    vx = np.empty(N)
    vy = np.empty(N)
    r = np.empty(N)
    x = np.empty(N)
    y = np.empty(N)
    psi = np.empty(N)

    df = np.empty(N)
    dr = np.empty(N)
    Fy_f = np.empty(N)
    Fy_r = np.empty(N)
    Fx_f = np.empty(N)
    Fx_r = np.empty(N)
    alpha_f = np.empty(N)
    alpha_r = np.empty(N)
    ay = np.empty(N)

    s = State3DOF(**s0.__dict__)

    # Provide longitudinal slip closure to derivatives
    def _lambda_f(s: State3DOF, ti: float) -> float:
        if lambda_f_fn is None:
            return 0.0
        return float(lambda_f_fn(ti, s))

    def _lambda_r(s: State3DOF, ti: float) -> float:
        if lambda_r_fn is None:
            return 0.0
        return float(lambda_r_fn(ti, s))

    derivatives._lambda_f = lambda s_obj: _lambda_f(s_obj, ti_current[0])
    derivatives._lambda_r = lambda s_obj: _lambda_r(s_obj, ti_current[0])

    ti_current = [0.0]

    for i, ti in enumerate(t):
        ti_current[0] = ti
        delta_sw = delta_sw_fn(ti)
        ds, aux = derivatives(s, delta_sw, vp)

        # Store
        vx[i], vy[i], r[i], x[i], y[i], psi[i] = s.vx, s.vy, s.r, s.x, s.y, s.psi
        df[i], dr[i] = aux["df"], aux["dr"]
        Fy_f[i], Fy_r[i] = aux["Fy_f"], aux["Fy_r"]
        Fx_f[i], Fx_r[i] = aux["Fx_f"], aux["Fx_r"]
        alpha_f[i], alpha_r[i] = aux["alpha_f"], aux["alpha_r"]
        ay[i] = aux["ay"]

        # Integrate: explicit Euler (small dt recommended)
        s.vx += ds[0] * dt
        s.vy += ds[1] * dt
        s.r += ds[2] * dt
        s.x += ds[3] * dt
        s.y += ds[4] * dt
        s.psi += ds[5] * dt

    return {
        "t": t,
        "vx": vx,
        "vy": vy,
        "r": r,
        "x": x,
        "y": y,
        "psi": psi,
        "df": df,
        "dr": dr,
        "Fx_f": Fx_f,
        "Fx_r": Fx_r,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "ay": ay,
    }