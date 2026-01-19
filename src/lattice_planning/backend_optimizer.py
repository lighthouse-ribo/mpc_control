import numpy as np
import cvxpy as cp
import math

class CfsOptimizer:
    def __init__(self, config):
        self.cfg = config
        
        # === 优化权重配置 ===
        self.w_smooth = 20.0      # 平滑性 (决定了曲线有多顺滑)
        self.w_ref = 2.0          # 参考线追踪 (决定了多听 Lattice 的话)
        
        # [已移除] self.w_consistency 
        # 原因：简单的索引对齐会导致车辆被“拉回”上一帧的位置，造成步数爆炸。
        
        self.w_terminal = 50.0    # 终点软锚定 (防止路径缩水)
        self.safety_margin = 0.3  # 障碍物安全缓冲 (米)

    def optimize(self, initial_path, dist_map, resolution, prev_path=None):
        """
        基于 CFS (凸可行域) 的几何路径优化
        (修复版：移除了导致车辆滞后的时空一致性约束)
        """
        N = len(initial_path)
        if N < 5: return initial_path

        # 1. 提取原始数据
        path_x = np.array([p['x'] for p in initial_path])
        path_y = np.array([p['y'] for p in initial_path])
        
        # 计算累计弧长 (用于速度插值)
        s_ref = np.zeros(N)
        for i in range(1, N):
            d = math.hypot(path_x[i] - path_x[i-1], path_y[i] - path_y[i-1])
            s_ref[i] = s_ref[i-1] + d
        total_len_ref = s_ref[-1]

        # 2. 定义变量
        x = cp.Variable(N)
        y = cp.Variable(N)
        
        cost = 0.0
        constraints = []

        # 3. 构建代价函数
        
        # (A) 平滑性 (Minimize Acceleration)
        acc_x = x[2:] - 2*x[1:-1] + x[:-2]
        acc_y = y[2:] - 2*y[1:-1] + y[:-2]
        cost += self.w_smooth * (cp.sum_squares(acc_x) + cp.sum_squares(acc_y))

        # (B) 参考线追踪
        cost += self.w_ref * (cp.sum_squares(x - path_x) + cp.sum_squares(y - path_y))

        # (C) 终点锚定
        cost += self.w_terminal * (cp.sum_squares(x[-1] - path_x[-1]) + cp.sum_squares(y[-1] - path_y[-1]))

        # (D) [关键修改] 仅保留起点强约束，移除“拖拽式”一致性
        # 强制起点必须是车辆当前位置 (Lattice 输入的第一个点)
        constraints += [x[0] == path_x[0], y[0] == path_y[0]]
        # 可选：约束起步的切线方向，防止车头乱摆
        # constraints += [x[1] == path_x[1], y[1] == path_y[1]]

        # 4. CFS 障碍物约束
        rows, cols = dist_map.shape
        origin_x, origin_y = 0.0, -10.0 # MapServer 原点
        
        for i in range(1, N): 
            cx, cy = path_x[i], path_y[i]
            r = int((cy - origin_y) / resolution)
            c = int((cx - origin_x) / resolution)
            
            if 0 <= r < rows and 0 <= c < cols:
                dist = dist_map[r, c]
                if dist < 2.0: # 只处理附近的障碍物
                    # 梯度计算
                    r_min, r_max = max(0, r-1), min(rows-1, r+1)
                    c_min, c_max = max(0, c-1), min(cols-1, c+1)
                    dr = (dist_map[r_max, c] - dist_map[r_min, c]) / 2.0
                    dc = (dist_map[r, c_max] - dist_map[r, c_min]) / 2.0
                    
                    grad_norm = math.hypot(dr, dc)
                    if grad_norm > 1e-4:
                        nx, ny = dc/grad_norm, dr/grad_norm
                        constraints.append(
                            nx * (x[i] - cx) + ny * (y[i] - cy) >= self.safety_margin - dist
                        )

        # 5. 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            # 减少最大迭代次数，提高实时性
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-2, eps_rel=1e-2, max_iter=500)
        except Exception:
            return initial_path

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return initial_path

        # 6. 结果重构
        new_x = x.value
        new_y = y.value
        optimized_path = []
        
        s_new = np.zeros(N)
        for i in range(1, N):
            s_new[i] = s_new[i-1] + math.hypot(new_x[i] - new_x[i-1], new_y[i] - new_y[i-1])
        total_len_new = s_new[-1]
        
        old_v = np.array([p['v'] for p in initial_path])
        current_t = 0.0

        for i in range(N):
            p = initial_path[i].copy()
            p['x'] = new_x[i]
            p['y'] = new_y[i]
            
            if i < N - 1:
                p['psi'] = math.atan2(new_y[i+1] - new_y[i], new_x[i+1] - new_x[i])
            else:
                p['psi'] = optimized_path[-1]['psi']
            
            # 速度插值
            if total_len_ref > 1e-3:
                s_mapped = (s_new[i] / total_len_new) * total_len_ref
                new_v = np.interp(s_mapped, s_ref, old_v)
            else:
                new_v = old_v[i]
            
            p['v'] = max(0.0, new_v)
            
            # 时间积分
            if i > 0:
                dist = s_new[i] - s_new[i-1]
                avg_v = (p['v'] + optimized_path[-1]['v']) / 2.0
                dt = dist / max(avg_v, 0.1)
                current_t += dt
            p['t'] = current_t
            
            optimized_path.append(p)
            
        return optimized_path