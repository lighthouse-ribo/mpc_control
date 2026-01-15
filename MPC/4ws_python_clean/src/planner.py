import numpy as np
from typing import List, Dict, Tuple


def solve_quintic(p0: float, v0: float, a0: float,
                  pT: float, vT: float, aT: float,
                  T: float) -> np.ndarray:
    """
    Solve a quintic polynomial coefficients for boundary conditions at t=0 and t=T.
    p(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    """
    T = float(T)
    M = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ], dtype=float)
    y = np.array([p0, v0, a0, pT, vT, aT], dtype=float)
    coeffs = np.linalg.solve(M, y)
    return coeffs


def sample_poly(coeffs: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample polynomial, first and second derivatives at t points."""
    a0, a1, a2, a3, a4, a5 = coeffs
    p = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    v = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    a = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return p, v, a


def plan_quintic_xy(start: Dict[str, float], end: Dict[str, float],
                    T: float, N: int,
                    U_start: float) -> List[Dict[str, float]]:
    """
    Plan a 2D path using independent quintic polynomials in x and y.
    - start: {x, y, psi} psi in radians
    - end: {x, y, psi} psi in radians
    - U_start: initial speed magnitude (m/s)
    Returns list of {t, x, y, psi}
    """
    x0, y0, psi0 = float(start['x']), float(start['y']), float(start['psi'])
    xT, yT, psiT = float(end['x']), float(end['y']), float(end['psi'])

    # Initial/terminal velocities projected onto x/y axes
    vx0 = U_start * np.cos(psi0)
    vy0 = U_start * np.sin(psi0)
    # Terminal velocity magnitude uses same U_start for simplicity here
    vxT = U_start * np.cos(psiT)
    vyT = U_start * np.sin(psiT)

    ax0 = 0.0
    ay0 = 0.0
    axT = 0.0
    ayT = 0.0

    cx = solve_quintic(x0, vx0, ax0, xT, vxT, axT, T)
    cy = solve_quintic(y0, vy0, ay0, yT, vyT, ayT, T)

    t = np.linspace(0.0, T, int(N))
    x, vx, _ = sample_poly(cx, t)
    y, vy, _ = sample_poly(cy, t)
    psi = np.arctan2(vy, vx)

    plan = [
        {
            't': float(tt),
            'x': float(xx),
            'y': float(yy),
            'psi': float(ppsi),
        }
        for tt, xx, yy, ppsi in zip(t, x, y, psi)
    ]
    return plan


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_2pi(a: float) -> float:
    a = float(a % (2.0 * np.pi))
    return a


def plan_circle_arc(start: Dict[str, float], end: Dict[str, float],
                    T: float, N: int,
                    U_start: float) -> List[Dict[str, float]]:
    """
    以起点/终点位置与航向角生成单圆弧参考轨迹。
    - start, end: {x, y, psi}，psi 为弧度（tangent 方位）
    - 选择与起点航向相切的圆心方向（左/右），在两者中挑选使终点航向更贴近的方向。
    - 若几何退化（半径无效或近似直线），退化为直线插值（psi 为切线方向）。

    返回 {t, x, y, psi} 列表，t 依据传入 T 均匀分布。
    """
    x0, y0, psi0 = float(start['x']), float(start['y']), float(start['psi'])
    x1, y1, psi1 = float(end['x']), float(end['y']), float(end['psi'])

    p0 = np.array([x0, y0], dtype=float)
    p1 = np.array([x1, y1], dtype=float)
    d = p1 - p0
    dnorm = float(np.hypot(d[0], d[1]))
    if dnorm < 1e-6:
        # 起终点重合：输出一个常值点
        t = np.linspace(0.0, max(1e-3, float(T)), int(N))
        return [{
            't': float(tt), 'x': x0, 'y': y0, 'psi': float(psi0)
        } for tt in t]

    def try_dir(dir_s: int) -> Tuple[bool, Dict[str, float]]:
        # 圆心方向：起点航向的法线（左: +1，对应 CCW；右: -1，对应 CW）
        u0 = np.array([np.cos(psi0 + dir_s * np.pi / 2.0),
                       np.sin(psi0 + dir_s * np.pi / 2.0)], dtype=float)
        denom = float(np.dot(d, u0))
        if abs(denom) < 1e-8:
            return False, {}
        r = float((dnorm * dnorm) / (2.0 * denom))
        if not np.isfinite(r) or r <= 1e-6:
            return False, {}
        C = p0 + r * u0
        # 起终点的径向角
        theta0 = float(np.arctan2(y0 - C[1], x0 - C[0]))
        theta1 = float(np.arctan2(y1 - C[1], x1 - C[0]))
        if dir_s > 0:
            # CCW：角度递增
            dtheta = _wrap_2pi(theta1 - theta0)
        else:
            # CW：角度递减
            dtheta = -_wrap_2pi(theta0 - theta1)
        psi1_pred = float(theta1 + dir_s * np.pi / 2.0)
        err_end = abs(_wrap_pi(psi1_pred - psi1))
        return True, {
            'dir': dir_s,
            'C': C,
            'r': r,
            'theta0': theta0,
            'theta1': theta1,
            'dtheta': dtheta,
            'err_end': err_end,
        }

    okL, candL = try_dir(+1)
    okR, candR = try_dir(-1)
    cand = None
    if okL and okR:
        cand = candL if candL['err_end'] <= candR['err_end'] else candR
    elif okL:
        cand = candL
    elif okR:
        cand = candR
    else:
        # 几何退化：退化为直线插值
        t = np.linspace(0.0, max(1e-3, float(T)), int(N))
        xs = np.linspace(x0, x1, int(N))
        ys = np.linspace(y0, y1, int(N))
        # 直线切线方向
        psi_line = float(np.arctan2(y1 - y0, x1 - x0))
        return [{
            't': float(tt), 'x': float(xx), 'y': float(yy), 'psi': float(psi_line)
        } for tt, xx, yy in zip(t, xs, ys)]

    C = np.array(cand['C'], dtype=float)
    r = float(cand['r'])
    dir_s = int(cand['dir'])
    theta0 = float(cand['theta0'])
    dtheta = float(cand['dtheta'])

    # 采样角度与时间
    t = np.linspace(0.0, max(1e-3, float(T)), int(N))
    s = np.linspace(0.0, 1.0, int(N))
    thetas = theta0 + s * dtheta
    xs = C[0] + r * np.cos(thetas)
    ys = C[1] + r * np.sin(thetas)
    psis = thetas + dir_s * np.pi / 2.0

    return [{
        't': float(tt), 'x': float(xx), 'y': float(yy), 'psi': float(pp)
    } for tt, xx, yy, pp in zip(t, xs, ys, psis)]