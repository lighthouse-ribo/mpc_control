import numpy as np
from casadi import SX, vertcat, cos, sin, tan, Function


class MPPIObstacleAvoider:
    def __init__(self, config):
        self.config = config
        self.dim_x = 4# 状态维度 [x, y, θ, v]
        self.dim_u = 2# 控制维度 [δ, a]
        
        self.u_prev = np.zeros((self.config.T, self.dim_u))  # 历史控制序列
        self.pre_waypoints_idx = 0# 上一时刻最近参考点索引
        
        self.vehicle_model = self._build_vehicle_kinematics()
        
    def _build_vehicle_kinematics(self):
        x, y, theta, v = SX.sym('x'), SX.sym('y'), SX.sym('theta'), SX.sym('v')
        steer, accel = SX.sym('steer'), SX.sym('accel')
        
        states = vertcat(x, y, theta, v)
        controls = vertcat(steer, accel)
        
        # 连续时间动力学
        states_dot = vertcat(
            v * cos(theta),                  # dx/dt
            v * sin(theta),                  # dy/dt
            v * tan(steer) / self.config.L,  # dθ/dt
            accel                            # dv/dt
        )
        
        next_states = states + states_dot * self.config.dt
        return Function('dynamics', [states, controls], [next_states])
    
    def compute_control(self, current_state, ref_path, obstacles):
        self.ref_path = ref_path
        self.ref_path_end_idx = len(ref_path) - 1
        
        # 查找当前位置最近的参考点
        nearest_idx, _, _, _, _ = self._find_nearest_waypoint(
            current_state[0], current_state[1], update_prev_idx=True
        )
        
        # 检查是否到达参考路径终点
        if nearest_idx >= self.ref_path_end_idx - 2:
            print("已到达参考路径终点")
            return None, None, None, None, True
        
        # 生成噪声序列
        epsilon = self._generate_noise_sequence(
            self.config.sigma, self.config.K, self.config.T, self.dim_u
        )
        
        # 采样控制序列并计算成本
        control_seqs = np.zeros((self.config.K, self.config.T, self.dim_u))
        traj_costs = np.zeros(self.config.K)  # 各轨迹成本
        
        for k in range(self.config.K):
            state = current_state.copy()
            obs = [o.copy() for o in obstacles]  # 障碍物状态深拷贝
            for t in range(self.config.T):
                # 生成带探索的控制序列
                if k < (1.0 - self.config.exploration) * self.config.K:
                    control_seqs[k, t] = self.u_prev[t] + epsilon[k, t]
                else:
                    control_seqs[k, t] = epsilon[k, t]
                
                # 控制输入限幅
                clamped_u = self._clamp_control(control_seqs[k, t])
                
                # 计算阶段成本
                stage_cost = self._compute_stage_cost(state, obs)
                control_smooth_cost = (self.config.w_a * clamped_u[1]**2 +
                                     self.config.w_delta_dot * (clamped_u[0] - self.u_prev[t, 0])**2)
                noise_cost = self.config.lambda_param * (1.0 - self.config.alpha) * \
                            self.u_prev[t].T @ np.linalg.inv(self.config.sigma) @ control_seqs[k, t]
                
                traj_costs[k] += stage_cost + control_smooth_cost + noise_cost

                # 更新车辆状态
                state = self.vehicle_model(state, clamped_u).full().flatten()
                # 更新障碍物状态
                obs = self._update_obstacle_states(obs)
            
            # 累加终端成本
            traj_costs[k] += self._compute_terminal_cost(state, obs)
        
        # 计算轨迹权重
        weights = self._compute_trajectory_weights(traj_costs)
        
        # 加权更新控制序列
        weighted_noise = np.zeros((self.config.T, self.dim_u))
        for t in range(self.config.T):
            for k in range(self.config.K):
                weighted_noise[t] += weights[k] * epsilon[k, t]
        
        # 优化控制序列并限幅
        optimal_u_seq = self.u_prev + weighted_noise
        for t in range(self.config.T):
            optimal_u_seq[t] = self._clamp_control(optimal_u_seq[t])
        
        # 计算最优轨迹
        optimal_traj = np.zeros((self.config.T, self.dim_x))
        state = current_state.copy()
        for t in range(self.config.T):
            state = self.vehicle_model(state, optimal_u_seq[t]).full().flatten()
            optimal_traj[t] = state
        
        # 计算采样轨迹（按成本排序）
        sampled_trajs = np.zeros((self.config.K, self.config.T, self.dim_x))
        sorted_indices = np.argsort(traj_costs)  # 按成本升序排序
        for idx, k in enumerate(sorted_indices):
            state = current_state.copy()
            for t in range(self.config.T):
                state = self.vehicle_model(state, control_seqs[k, t]).full().flatten()
                sampled_trajs[idx, t] = state
        
        # 控制序列滚动更新
        self.u_prev[:-1] = optimal_u_seq[1:]
        self.u_prev[-1] = optimal_u_seq[-1]
        
        return optimal_u_seq[0], optimal_u_seq, optimal_traj, sampled_trajs, False
    
    def _update_obstacle_states(self, obstacles):
        for obs in obstacles:
            x0, y0, heading, v, length, width, active = obs
            if active < 0.5:
                continue
            x0 += v * self.config.dt * cos(heading)
            y0 += v * self.config.dt * sin(heading)
            obs[0], obs[1] = x0, y0
        return obstacles
    
    def _compute_trajectory_weights(self, costs):
        min_cost = costs.min()
        exp_terms = np.exp((-1.0 / self.config.lambda_param) * (costs - min_cost))
        sum_exp = exp_terms.sum()
        return exp_terms / sum_exp
    
    def _compute_terminal_cost(self, final_state, obstacles):
        x, y, theta, v = final_state
        _, ref_x, ref_y, ref_theta, ref_v = self._find_nearest_waypoint(x, y)
        
        # 终端跟踪成本
        tracking_cost = (self.config.w_fx * (x - ref_x)**2 +
                       self.config.w_fy * (y - ref_y)** 2 +
                       self.config.w_ftheta * (np.sin(theta - ref_theta))**2 +
                       self.config.w_fv * (v - ref_v)** 2)
        
        # 边界惩罚
        boundary_cost = self._compute_boundary_cost(final_state)
        
        # 障碍物惩罚
        obstacle_cost = self._compute_obstacle_cost(x, y, theta, obstacles)
        
        return tracking_cost + boundary_cost + obstacle_cost
    
    def _compute_boundary_cost(self, state):
        y = state[1]
        if y < self.config.y_min:
            return self.config.w_bound * (self.config.y_min - y)**2
        elif y > self.config.y_max:
            return self.config.w_bound * (y - self.config.y_max)** 2
        return 0

    def _compute_stage_cost(self, state, obstacles):
        x, y, theta, v = state
        _, ref_x, ref_y, ref_theta, ref_v = self._find_nearest_waypoint(x, y)
        
        # 路径跟踪成本
        tracking_cost = (self.config.w_x * (x - ref_x)**2 +
                       self.config.w_y * (y - ref_y)** 2 +
                       self.config.w_theta * (np.sin(theta - ref_theta))**2 +
                       self.config.w_v * (v - ref_v)** 2)
        
        # 边界惩罚
        boundary_cost = self._compute_boundary_cost(state)
        
        # 障碍物惩罚
        obstacle_cost = self._compute_obstacle_cost(x, y, theta, obstacles)
        
        return tracking_cost + boundary_cost + obstacle_cost
    
    def _compute_obstacle_cost(self, x, y, theta, obstacles):
        total_cost = 0.0
        for obs in obstacles:
            if obs[6] < 0.5:  
                continue
                
            # 障碍物参数 [x, y, 航向, 速度, 长度, 宽度, 激活标志]
            obs_x, obs_y, _, _, obs_len, obs_width, _ = obs
            
            # 计算中心距离
            dx = x - obs_x
            dy = y - obs_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # 安全距离（车辆与障碍物外接圆半径之和 + 额外安全距离）
            vehicle_radius = np.sqrt((self.config.vehicle_length/2)**2 + (self.config.vehicle_width/2)** 2)
            obstacle_radius = np.sqrt((obs_len/2)**2 + (obs_width/2)** 2)
            safe_dist = vehicle_radius + obstacle_radius + self.config.safety_distance
            
            # 超出安全距离无惩罚
            if dist >= safe_dist + 0.1:
                continue

            # 近距离惩罚（指数+多项式结合）
            norm_dist = dist / (safe_dist + 0.1)
            penalty = (1.0 - norm_dist)**2 * np.exp(-self.config.obstacle_decay_rate * (norm_dist - 1.0))
            total_cost += self.config.w_obstacle * penalty

        return total_cost
    
    def _clamp_control(self, u):
        u_clamped = u.copy()
        u_clamped[0] = np.clip(u_clamped[0], self.config.delta_min, self.config.delta_max)  # 转向角
        u_clamped[1] = np.clip(u_clamped[1], self.config.a_min, self.config.a_max)          # 加速度
        return u_clamped
    
    def _generate_noise_sequence(self, sigma, K, T, dim_u):
        mu = np.zeros(dim_u)
        return np.random.multivariate_normal(mu, sigma, (K, T))
    
    def _find_nearest_waypoint(self, x, y, update_prev_idx=False):
        start_idx = self.pre_waypoints_idx
        end_idx = min(self.ref_path.shape[0]-1, start_idx + self.config.max_search_idx_len)
        
        dx = x - self.ref_path[start_idx:end_idx, 0]
        dy = y - self.ref_path[start_idx:end_idx, 1]
        dist_sq = dx**2 + dy**2
        min_idx = np.argmin(dist_sq) + start_idx  # 全局索引
      
        ref_x = self.ref_path[min_idx, 0]
        ref_y = self.ref_path[min_idx, 1]
        ref_theta = self.ref_path[min_idx, 2]
        ref_v = self.ref_path[min_idx, 3]
        
        if update_prev_idx:
            self.pre_waypoints_idx = min_idx

        return min_idx, ref_x, ref_y, ref_theta, ref_v