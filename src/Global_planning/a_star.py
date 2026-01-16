import numpy as np
import heapq
import math
from .utils import obs_dist_interp, is_strict_safe_move, is_direct_line_of_sight
# [新增] 导入我们刚才写的手写平滑器
from .smoother import b_spline_smooth, optimize_with_safety, fix_sharp_turns

class EnhancedAStar:
    def __init__(self, config):
        self.config = config

    def _heuristic(self, r1, c1, r2, c2, obs_dist):
        """ 启发式函数"""
        base_cost = math.hypot(r1 - r2, c1 - c2)
        
        safety_dist = obs_dist_interp(obs_dist, np.array([r1, c1]))
        safety_penalty = max(0.0, self.config.safety_margin - safety_dist)
        
        map_diag = math.hypot(obs_dist.shape[0], obs_dist.shape[1])
        safety_weight = 0.5 * min(1.0, base_cost / map_diag)
        
        return base_cost + safety_weight * safety_penalty

    def _calculate_turn_cost(self, direction_matrix, curr, neighbor):
        """ 转向代价计算"""
        cr, cc = int(curr[0]), int(curr[1])
        vec1 = direction_matrix[cr, cc, :].astype(float)
        vec2 = np.array([neighbor[0] - curr[0], neighbor[1] - curr[1]], dtype=float)
        
        if np.linalg.norm(vec1) == 0:
            return 0.0
            
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            return 0.0
            
        cos_theta = np.clip(np.dot(vec1, vec2) / denom, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_theta))
        
        if angle >= self.config.min_turn_angle:
            max_angle = self.config.max_turn_angle
            if angle > max_angle:
                return self.config.turn_cost_weight * 5 * (angle / max_angle) ** 3
            else:
                return self.config.turn_cost_weight * (angle / max_angle) ** 2
        return 0.0

    def plan(self, grid, start, goal, obs_dist, map_info=None):
        """
        执行 A* 规划并调用平滑后处理
        :param map_info: MapServer 实例，用于坐标转换 (Grid -> World)
        """
        rows, cols = grid.shape
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # 1. 基础越界检查
        if not (0 <= start[0] < rows and 0 <= start[1] < cols): return []
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols): return []

        # 2. A* 数据结构初始化
        g_cost = np.full((rows, cols), np.inf)
        parent = np.full((rows, cols, 2), -1, dtype=int)
        direction = np.zeros((rows, cols, 2), dtype=int)
        turn_cost_map = np.zeros((rows, cols), dtype=float)
        
        visited = np.zeros((rows, cols), dtype=bool)
        in_open = np.zeros((rows, cols), dtype=bool)
        
        # 起点初始化
        sr, sc = int(start[0]), int(start[1])
        g_cost[sr, sc] = 0.0
        h_val = self._heuristic(sr, sc, goal[0], goal[1], obs_dist)
        
        open_list = [(h_val, sr, sc)]
        in_open[sr, sc] = True
        
        # 8邻域定义
        neighbors = [
            (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        
        path_found = False

        # 3. A* 搜索主循环
        while open_list:
            f, r, c = heapq.heappop(open_list)
            in_open[r, c] = False
            visited[r, c] = True
            
            # 到达终点
            if r == int(goal[0]) and c == int(goal[1]):
                path_found = True
                break
            
            for dr, dc, move_len in neighbors:
                nr, nc = r + dr, c + dc
                
                # 越界与碰撞检查
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                if visited[nr, nc]: continue
                if grid[nr, nc] == 1: continue
                
                curr_pos = np.array([r, c])
                next_pos = np.array([nr, nc])
                dir_vec = np.array([dr, dc])
                
                # 严格安全检查 (包含穿墙检测)
                safe, _ = is_strict_safe_move(curr_pos, next_pos, grid, obs_dist, dir_vec, self.config)
                if not safe: continue
                
                # 计算 G 值
                new_g = g_cost[r, c] + move_len
                
                # 计算转向代价
                added_turn_cost = 0.0
                if parent[r, c, 0] != -1:
                    # 视线优化 (LOS)
                    if self.config.dynamic_turn_cost and \
                       is_direct_line_of_sight(next_pos, goal, grid, obs_dist, self.config.safety_margin):
                        added_turn_cost = turn_cost_map[r, c]
                    else:
                        added_turn_cost = self._calculate_turn_cost(direction, curr_pos, next_pos)
                
                total_new_cost = new_g + self.config.turn_cost_weight * added_turn_cost
                new_h = self._heuristic(nr, nc, goal[0], goal[1], obs_dist)
                
                # 更新节点
                if total_new_cost + new_h < g_cost[nr, nc] + new_h:
                    parent[nr, nc] = [r, c]
                    g_cost[nr, nc] = total_new_cost
                    turn_cost_map[nr, nc] = added_turn_cost
                    direction[nr, nc] = [dr, dc]
                    
                    if not in_open[nr, nc]:
                        heapq.heappush(open_list, (total_new_cost + new_h, nr, nc))
                        in_open[nr, nc] = True
                    else:
                        heapq.heappush(open_list, (total_new_cost + new_h, nr, nc))

        if not path_found:
            return []

        # 4. 原始路径重建 (Grid Frame)
        raw_path = self._reconstruct(parent, start, goal)
        final_path = raw_path

        # 5. 执行平滑后处理 (先做完所有平滑)
        if self.config.use_bspline_smoothing and len(final_path) > 3:
            
            # [Step A] 预处理：修复急转弯
            final_path = fix_sharp_turns(
                final_path, self.config.max_turn_angle
            )

            # [Step B] 预处理：安全优化
            final_path = optimize_with_safety(
                final_path, grid, obs_dist, self.config.safety_margin
            )
            
            # [Step C] 生成：B-Spline 平滑
            # 到这里，final_path 是一条完美的、曲率连续的曲线
            final_path = b_spline_smooth(final_path, self.config.smoothing_factor)

        # 6. [新增] 终点延长逻辑 (Extension AFTER Smoothing)
        # 此时基于平滑后最后一段的切线方向进行线性延伸
        if map_info is not None and len(final_path) >= 2:
            try:
                # 获取分辨率，默认为 0.5
                res = getattr(map_info, 'resolution', 0.5)
                extend_len_m = 3.0
                
                # 计算需要延伸的距离（换算成 grid 单位）
                # 注意：unit_vec 长度为 1 (grid unit)，代表 0.5m
                # 所以要延伸 3m，需要 3.0 / 0.5 = 6 个 grid unit
                # 我们以 1 个 grid unit 为步长添加点，这样密度与原路径一致
                num_steps = int(np.ceil(extend_len_m / res))
                
                p_end = final_path[-1]     # 平滑后的终点
                p_prev = final_path[-2]    # 平滑后的倒数第二点
                
                # 计算切线方向
                vec = p_end - p_prev
                norm = np.linalg.norm(vec)
                
                if norm > 1e-6:
                    unit_vec = vec / norm
                    extensions = []
                    
                    for i in range(1, num_steps + 1):
                        # 沿切线方向线性延伸
                        # i=1 -> 0.5m, i=2 -> 1.0m ...
                        new_pt = p_end + unit_vec * i
                        
                        # 边界检查 (Lattice 采样时可能并不希望点出界，安全起见还是检查一下)
                        if 0 <= new_pt[0] < rows and 0 <= new_pt[1] < cols:
                            extensions.append(new_pt)
                        else:
                            break
                    
                    if len(extensions) > 0:
                        # 将延伸点拼接到路径末尾
                        # final_path 是 numpy array, extensions 是 list of arrays
                        final_path = np.vstack((final_path, np.array(extensions)))

            except Exception as e:
                print(f"[A*] Path extension warning: {e}")

        # 7. 坐标转换 (Grid -> World)
        if map_info is not None:
            world_path = []
            for pt in final_path:
                # pt 是 [row, col]
                wx, wy = map_info.grid_to_world(pt[0], pt[1])
                world_path.append({'x': wx, 'y': wy})
            return world_path
            
        return final_path

    def _reconstruct(self, parent, start, goal):
        path = []
        curr = np.array([int(goal[0]), int(goal[1])])
        start_idx = np.array([int(start[0]), int(start[1])])
        
        while not np.array_equal(curr, start_idx):
            path.append(curr)
            pr, pc = parent[curr[0], curr[1]]
            if pr == -1: return []
            curr = np.array([pr, pc])
            
        path.append(start_idx)
        return np.array(path[::-1])