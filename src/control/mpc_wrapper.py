import numpy as np
import math
from src.mpc import solve_mpc_kin_dyn_4dof

class LateralMPC:
    """
    横向路径跟踪控制器 (Intelligent Component)
    职责：
    1. 管理 MPC 权重与预瞄参数
    2. 处理参考路径切片 (Trajectory Slicing)
    3. 计算误差状态 (Error State Calculation)
    4. 调用底层 QP 求解器
    """
    def __init__(self, params, dt=0.02):
        self.params = params
        self.dt = dt
        
        # --- 控制参数 (从 SimEngine 迁移过来) ---
        self.delta_max = np.deg2rad(30.0)
        self.horizon = 40  # 预测步长
        
        # 预瞄距离参数 (用于提取局部路径)
        self.Ld_k = 0.8    # 速度增益
        self.Ld_b = 4.0    # 基础距离
        self.Ld_min = 3.0
        self.Ld_max = 18.0
        
        # 路径提取参数
        self.ctrl_plan_max_points = 4000
        self.ctrl_plan_pad_factor = 1.5

        # --- MPC 权重配置 ---
        self.weights = {
            'Q_ey': 100.0,
            'Q_epsi': 1000.0,
            'Q_beta': 20.0,
            'Q_r': 0.1,
            'R_df': 2.0,
            'R_dr': 2.0,
            'R_delta_df': 20.0,
            'R_delta_dr': 20.0
        }

    def compute(self, state: dict, full_plan: list, current_ctrl: dict) -> tuple:
        """
        核心计算接口
        :param state: 车辆状态 {x, y, psi, beta, r, ...}
        :param full_plan: 全局或局部规划轨迹 [{'x':, 'y':, 'psi':}, ...]
        :param current_ctrl: 当前控制量 {U, delta_f, delta_r}
        :return: (delta_f_cmd, delta_r_cmd)
        """
        if not full_plan:
            return 0.0, 0.0

        # 1. 提取局部参考路径 (包含预瞄逻辑)
        # 这一步包含了原来在 sim.py 里的复杂逻辑
        local_plan, ref_info = self._extract_local_segment(state, full_plan, current_ctrl['U'])
        
        if not local_plan:
            return 0.0, 0.0

        # 2. 计算误差状态 (Error States: e_y, e_psi)
        # 如果 plan 里没有预算的误差，我们需要自己算
        e_y, e_psi = self._calculate_errors(state, local_plan[0], ref_info)
        
        # 3. 组装 MPC 输入状态
        state_aug = {
            'x': state['x'],
            'y': state['y'],
            'psi': state['psi'],
            'e_y': e_y,
            'e_psi': e_psi,
            'beta': state['beta'],
            'r': state['r'],
        }

        # 4. 调用底层求解器
        df, dr = solve_mpc_kin_dyn_4dof(
            state_aug=state_aug,
            ctrl=current_ctrl,
            params=self.params,
            plan=local_plan,
            dt=self.dt,
            H=self.horizon,
            **self.weights, # 解包权重
            delta_max=self.delta_max
        )
        
        return df, dr

    def _extract_local_segment(self, state, full_plan, current_speed):
        """
        [内部逻辑] 从全局路径中切出一块适合 MPC 跟踪的局部路径
        移植自 src/sim.py 的 _get_plan_for_controller
        """
        n = len(full_plan)
        if n < 2:
            return full_plan, None

        x, y = state['x'], state['y']
        
        # 1. 寻找匹配点 (简单最近邻)
        # 注意：这里简化了逻辑，实战中可能需要利用上一帧索引加速
        best_i = 0
        best_d2 = float('inf')
        # 局部搜索优化：假设车不会离起点太远，或者根据位置搜索
        # 这里为了稳健先全搜，性能敏感可改为局部窗口
        for i, p in enumerate(full_plan):
            d2 = (p['x'] - x)**2 + (p['y'] - y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        
        # 2. 计算动态预瞄距离 Ld
        U_mag = abs(current_speed)
        Ld = np.clip(self.Ld_k * U_mag + self.Ld_b, self.Ld_min, self.Ld_max)
        
        # 3. 确定截取窗口
        # MPC 需要的窗口长度 L_window
        # 粗略估算：H * dt * U * buffer
        L_window = U_mag * self.dt * self.horizon * self.ctrl_plan_pad_factor
        L_window = max(L_window, Ld * 1.5) # 保证至少覆盖预瞄点
        
        # 向后截取
        current_len = 0.0
        end_i = best_i
        while end_i < n - 1 and current_len < L_window:
            p = full_plan[end_i]
            q = full_plan[end_i + 1]
            current_len += math.hypot(q['x'] - p['x'], q['y'] - p['y'])
            end_i += 1
        
        segment = full_plan[best_i : end_i + 1]
        
        # 4. 降采样 (防止点太密导致 H 覆盖距离过短)
        # 原 sim.py 里的逻辑：如果点太多就抽稀
        m = len(segment)
        if m > self.ctrl_plan_max_points:
             out = []
             for i in range(self.ctrl_plan_max_points):
                 idx = int(round(i * (m - 1) / (self.ctrl_plan_max_points - 1)))
                 out.append(segment[idx])
             segment = out

        # 返回片段和匹配点索引信息
        info = {'base_index': best_i}
        return segment, info

    def _calculate_errors(self, state, ref_point, info):
        """
        [内部逻辑] 计算 Frenet 误差 e_y, e_psi
        """
        # 参考点的航向
        psi_ref = ref_point.get('psi', 0.0)
        
        # 坐标转换到 Frenet
        dx = state['x'] - ref_point['x']
        dy = state['y'] - ref_point['y']
        
        # e_y: 横向误差 (根据参考航向投影)
        # e_y = -dx * sin(psi_ref) + dy * cos(psi_ref)
        e_y = -dx * math.sin(psi_ref) + dy * math.cos(psi_ref)
        
        # e_psi: 航向误差 (归一化到 -pi ~ pi)
        e_psi = state['psi'] - psi_ref
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi
        
        return e_y, -e_psi # 注意：通常定义的 e_psi 是 ref - state 还是 state - ref 需与 MPC 内部一致
                           # 检查 src/mpc.py: e_psi_dot = -r + r_ref
                           # 若 state[1] 是 e_psi = psi - psi_ref
                           # 则 d(psi - psi_ref) = r - r_ref = -(r_ref - r)
                           # 原 mpc 代码使用的是 e_psi = psi_ref - psi (sim.py line 467)
                           # 所以这里应该是 ref - state
                           
        # 修正：与 sim.py 保持一致 (sim.py line 467: wrap(psi_base - state_psi))
        e_psi_val = psi_ref - state['psi']
        e_psi_val = (e_psi_val + math.pi) % (2 * math.pi) - math.pi
        
        return e_y, e_psi_val