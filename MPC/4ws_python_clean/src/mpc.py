import numpy as np
from typing import List, Dict, Tuple, Optional
from .twodof import derivatives as deriv_2dof


def nearest_plan_index(plan: List[Dict[str, float]], x: float, y: float) -> int:
    """在参考轨迹中寻找与 (x,y) 最近的点索引。"""
    if not plan:
        return 0
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(plan):
        dx = float(p.get('x', 0.0)) - x
        dy = float(p.get('y', 0.0)) - y
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def linearize_2dof(params, x_vec: np.ndarray, df0: float, dr0: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """[旧] 数值线性化 2DOF 并离散化：返回 (A_d, B_d)。"""
    base = deriv_2dof(x_vec, df0, dr0, params)
    xdot0 = np.array(base["xdot"], dtype=float)
    nx = 2; nu = 2
    A = np.zeros((nx, nx), dtype=float); B = np.zeros((nx, nu), dtype=float)
    eps_x = 1e-4; eps_u = 1e-3
    for j in range(nx):
        x_eps = np.array(x_vec, dtype=float); x_eps[j] += eps_x
        xdot_eps = np.array(deriv_2dof(x_eps, df0, dr0, params)["xdot"], dtype=float)
        A[:, j] = (xdot_eps - xdot0) / eps_x
    u0 = np.array([df0, dr0], dtype=float)
    for j in range(nu):
        u_eps = np.array(u0, dtype=float); u_eps[j] += eps_u
        xdot_eps = np.array(deriv_2dof(x_vec, float(u_eps[0]), float(u_eps[1]), params)["xdot"], dtype=float)
        B[:, j] = (xdot_eps - xdot0) / eps_u
    A_d = np.eye(nx) + A * float(dt); B_d = B * float(dt)
    return A_d, B_d

# --- 新的 4-DOF 轨迹跟踪模型 ---

def get_kin_dyn_4dof_derivatives(
    x_aug: np.ndarray,
    u: np.ndarray,
    params,
    U: float,
    r_ref: float,
) -> np.ndarray:
    """
    计算 4-DOF 增广状态的导数: x_aug = [e_y, e_psi, beta, r]
    返回: x_dot_aug = [e_y_dot, e_psi_dot, beta_dot, r_dot]
    """
    e_y, e_psi, beta, r = float(x_aug[0]), float(x_aug[1]), float(x_aug[2]), float(x_aug[3])
    df, dr = float(u[0]), float(u[1])
    
    # 1. 运动学误差模型 (Kinematic Error Model)
    # 统一误差定义：e_psi = psi_ref - psi_cur
    # 线性近似下：e_y_dot ≈ U * (beta - e_psi)
    e_y_dot = U * (beta - e_psi)

    # e_psi 动态：e_psi_dot ≈ -r + r_ref（其中 r_ref ≈ U * kappa_ref）
    e_psi_dot = -r + r_ref

    # 2. 动力学模型 (Dynamic Model)
    # [beta_dot, r_dot] 来自 twodof
    x_dyn = np.array([beta, r])
    d = deriv_2dof(x_dyn, df, dr, params)
    beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])
    
    return np.array([e_y_dot, e_psi_dot, beta_dot, r_dot])

def linearize_kin_dyn_4dof(
    params,
    x0_aug: np.ndarray,
    u0: np.ndarray,
    dt: float,
    U: float,
    r_ref_0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值线性化 4-DOF 轨迹跟踪模型 (LTI)
    状态 x_aug = [e_y, e_psi, beta, r]
    输入 u = [df, dr]
    """
    base = get_kin_dyn_4dof_derivatives(x0_aug, u0, params, U, r_ref_0)
    xdot0 = np.array(base, dtype=float)
    
    nx = 4
    nu = 2
    A = np.zeros((nx, nx), dtype=float)
    B = np.zeros((nx, nu), dtype=float)
    eps_x = 1e-4
    eps_u = 1e-3

    # A: 对状态 x_aug 求导
    for j in range(nx):
        x_eps = np.array(x0_aug, dtype=float)
        x_eps[j] += eps_x
        xdot_eps = get_kin_dyn_4dof_derivatives(x_eps, u0, params, U, r_ref_0)
        A[:, j] = (xdot_eps - xdot0) / eps_x
        
    # B: 对控制 u 求导
    for j in range(nu):
        u_eps = np.array(u0, dtype=float)
        u_eps[j] += eps_u
        xdot_eps = get_kin_dyn_4dof_derivatives(x0_aug, u_eps, params, U, r_ref_0)
        B[:, j] = (xdot_eps - xdot0) / eps_u
        
    # 离散化（欧拉）
    A_d = np.eye(nx) + A * float(dt)
    B_d = B * float(dt)
    return A_d, B_d

def solve_mpc_kin_dyn_4dof(
    state_aug: Dict[str, float],
    ctrl: Dict[str, float],
    params,
    plan: List[Dict[str, float]],
    dt: float,
    H: int = 10,
    Q_ey: float = 10.0,
    Q_epsi: float = 5.0,
    Q_beta: float = 0.1,
    Q_r: float = 0.1,
    R_df: float = 0.5,
    R_dr: float = 0.5,
    R_delta_df: float = 1.0,
    R_delta_dr: float = 1.0,
    delta_max: Optional[float] = None,
) -> Tuple[float, float]:
    """
    基于 4-DOF [ey, epsi, beta, r] 模型的 MPC 求解器
    """
    if not plan:
        return float(ctrl.get('delta_f', 0.0)), float(ctrl.get('delta_r', 0.0))

    # --- 1. 初始状态 ---
    x0_raw = np.array([
        float(state_aug.get('e_y', 0.0)),
        float(state_aug.get('e_psi', 0.0)),
        float(state_aug.get('beta', 0.0)),
        float(state_aug.get('r', 0.0))
    ], dtype=float)
    df0 = float(ctrl.get('delta_f', 0.0))
    dr0 = float(ctrl.get('delta_r', 0.0))
    u0 = np.array([df0, dr0], dtype=float)

    # --- 2. 参考序列 ---
    def wrap(a: float) -> float:
        return float((a + np.pi) % (2*np.pi) - np.pi)
    
    x = float(state_aug.get('x', 0.0))
    y = float(state_aug.get('y', 0.0))
    psi0 = float(state_aug.get('psi', 0.0))
    U_signed = float(params.U)
    
    base_i = nearest_plan_index(plan, x, y)
    i_start = min(base_i + 1, len(plan) - 1)

    # 参考横摆率：基于曲率 kappa ≈ dpsi/ds，r_ref = U_signed * kappa
    r_ref_seq = np.zeros(H, dtype=float)
    n_plan = len(plan)
    def seg_psi(i: int) -> float:
        a = plan[i]
        b = plan[i + 1]
        dx_i = float(b['x'] - a['x']); dy_i = float(b['y'] - a['y'])
        ds_i = float(np.hypot(dx_i, dy_i))
        return float(np.arctan2(dy_i, dx_i)) if ds_i > 1e-6 else float(a.get('psi', psi0))
    for k in range(H):
        idx_center = min(n_plan - 2, i_start + k)
        j_prev = max(0, idx_center - 1)
        j_next = min(n_plan - 2, idx_center + 1)
        psi_a = seg_psi(j_prev)
        psi_b = seg_psi(j_next)
        dpsi = wrap(psi_b - psi_a)
        ds_a = float(np.hypot(plan[j_prev + 1]['x'] - plan[j_prev]['x'], plan[j_prev + 1]['y'] - plan[j_prev]['y']))
        ds_b = float(np.hypot(plan[j_next + 1]['x'] - plan[j_next]['x'], plan[j_next + 1]['y'] - plan[j_next]['y']))
        ds_avg = max(1e-6, 0.5 * (ds_a + ds_b))
        kappa_ref = float(dpsi / ds_avg)
        r_ref_seq[k] = float(U_signed * kappa_ref)

    # --- 3. 线性化 ---
    # 使用 LTI MPC：只在 k=0 处线性化一次
    r_ref_0 = float(r_ref_seq[0])
    A_d, B_d = linearize_kin_dyn_4dof(params, x0_raw, u0, dt, U_signed, r_ref_0)

    # --- 4. 预测矩阵 Φ 和 T ---
    nx, nu = 4, 2
    Phi = np.zeros((H * nx, nx), dtype=float)
    Tm = np.zeros((H * nx, H * nu), dtype=float)
    Ak = np.eye(nx)
    for k in range(H):
        Ak_plus_1 = Ak @ A_d
        Phi[k*nx:(k+1)*nx, :] = Ak_plus_1
        for j in range(k + 1):
            if k >= j:
                power = k - j
                Ad_pow = np.linalg.matrix_power(A_d, power)
                block = Ad_pow @ B_d
                Tm[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = block
        Ak = Ak_plus_1

    # --- 5. 成本矩阵 Q/R ---
    Qh = np.zeros((H * nx, H * nx), dtype=float)
    Rh = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        Qk = np.diag([Q_ey, Q_epsi, Q_beta, Q_r])
        Rk = np.diag([R_df, R_dr])
        Qh[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Qk
        Rh[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Rk

    # --- 6. 参考状态 Xref ---
    # 我们的目标是让 e_y -> 0, e_psi -> 0, beta -> 0, r -> r_ref
    Xref = np.zeros(H * nx, dtype=float)
    for k in range(H):
        Xref[k*nx + 0] = 0.0  # e_y_ref
        Xref[k*nx + 1] = 0.0  # e_psi_ref
        Xref[k*nx + 2] = 0.0  # beta_ref
        Xref[k*nx + 3] = r_ref_seq[k] # r_ref

    # --- 7. 控制变率惩罚 D, g, R_delta ---
    D = np.zeros((H * nu, H * nu), dtype=float)
    g = np.zeros(H * nu, dtype=float)
    Iu = np.eye(nu)
    u_prev = np.array([df0, dr0], dtype=float)
    for k in range(H):
        D[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Iu
        if k == 0:
            g[0:nu] = u_prev
        else:
            D[k*nu:(k+1)*nu, (k-1)*nu:k*nu] = -Iu
    R_delta = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        R_delta[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = np.diag([R_delta_df, R_delta_dr])

    # --- 8. QP 组装 (H U = -f) ---
    # 注意：我们不再需要 S 矩阵，因为 e_psi 已经是状态
    Hmat = (
        Tm.T @ Qh @ Tm
        + Rh
        + D.T @ R_delta @ D
    )
    fvec = (
        (Phi @ x0_raw - Xref).T @ Qh @ Tm
        - g.T @ R_delta @ D
    ).T
    
    # 求解: Hmat * U = -fvec
    try:
        Useq = np.linalg.solve(Hmat, -fvec)
    except np.linalg.LinAlgError:
        Useq = np.linalg.pinv(Hmat) @ -fvec

    df_cmd = float(Useq[0])
    dr_cmd = float(Useq[1])
    
    if delta_max is not None:
        df_cmd = float(np.clip(df_cmd, -delta_max, delta_max))
        dr_cmd = float(np.clip(dr_cmd, -delta_max, delta_max))
    return df_cmd, dr_cmd