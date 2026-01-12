import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as plt_patches


class Config:
    def __init__(self):
        # MPPI核心参数
        self.T = 20                # 预测时域长度
        self.K = 600               # 采样轨迹数量
        self.dt = 0.2              # 时间步长
        self.max_obs = 5           # 最大障碍物数量
        self.lambda_param = 100.0# 温度参数
        self.alpha = 0.95          # 控制序列平滑系数
        self.exploration = 0.2     # 探索率（控制轨迹多样性）
        
        # 控制噪声协方差矩阵
        self.sigma = np.array([[0.075, 0.0], [0.0, 1.5]])  

        # 车辆与障碍物参数
        self.vehicle_length = 2.0# 车辆长度
        self.vehicle_width = 1.0   # 车辆宽度
        self.safety_distance = 0.1# 安全距离
        self.L = 1.5               # 轴距
        
        # 车辆运动约束
        self.v_max = 10.0                 # 最大速度
        self.v_min = 0.0                  # 最小速度
        self.a_max = 2.0                  # 最大加速度
        self.a_min = -3.0                 # 最大减速度
        self.delta_max = np.deg2rad(35)   # 最大转向角
        self.delta_min = -np.deg2rad(35)  # 最小转向角

        # 道路与场景参数
        self.road_length = 20.0    # 道路长度
        self.y_min = -3.0          # 道路下边界
        self.y_max = 3.0           # 道路上边界
        self.start_pos = np.array([1.0, 0.0, 0.0, 1.0])  # 起点 [x, y, θ, v]
        self.goal_pos = np.array([18.5, 0.0, 0.0, 0.0])  # 终点 [x, y, θ, v]

        # 可视化参数
        self.xlim = (-0.1, self.road_length + 0.1)
        self.ylim = (self.y_min - 0.1, self.y_max + 0.1)
        self.animation_interval = 100# 动画刷新间隔(ms)
        self.gif_filename = "mppi_obstacle_avoidance.gif"

        # 成本函数权重
        self.w_x = 30.0        # x方向跟踪权重
        self.w_y = 30.0        # y方向跟踪权重
        self.w_theta = 1.0     # 航向角跟踪权重
        self.w_v = 20.0        # 速度跟踪权重
        self.w_a = 0.1         # 加速度平滑权重
        self.w_delta_dot = 5.0# 转向角变化率权重
        self.w_bound = 500.0   # 道路边界惩罚权重
        self.w_obstacle = 300.0# 障碍物惩罚权重
        self.w_fx = 100.0      # 终端x跟踪权重
        self.w_fy = 100.0      # 终端y跟踪权重
        self.w_ftheta = 10.0   # 终端航向角跟踪权重
        self.w_fv = 20.0       # 终端速度跟踪权重

        # 障碍物惩罚参数
        self.obstacle_decay_rate = 3.0# 障碍物惩罚衰减率

        # 仿真参数
        self.sim_time = 20.0   # 最大仿真时间
        self.ref_speed = 2.0   # 参考速度
        self.goal_threshold = 0.2# 到达目标阈值
        self.max_search_idx_len = 50# 参考点搜索范围

def generate_reference_path(init_state, goal_state, num_points, dt, config):
    x_ref = np.linspace(init_state[0], goal_state[0], num_points)
    y_ref = np.zeros(num_points)
    theta_ref = np.zeros(num_points)
    
    # 速度规划（加速→匀速→减速）
    v_ref = np.zeros(num_points)
    total_dist = goal_state[0] - init_state[0]  # 总距离
    
    for k in range(num_points):
        remaining_dist = goal_state[0] - x_ref[k]  # 剩余距离
        
        # 远距段：加速到参考速度
        if remaining_dist > 0.7 * total_dist:
            v_ref[k] = min(config.ref_speed, init_state[3] + config.a_max * k * dt)
        # 中段：匀速
        elif remaining_dist > 0.3 * total_dist:
            v_ref[k] = config.ref_speed
        # 近距段：减速
        elif remaining_dist > 3.0:
            v_ref[k] = config.ref_speed * (remaining_dist / (0.3 * total_dist))
        # 终点附近：低速
        else:
            v_ref[k] = max(0.5, config.ref_speed * (remaining_dist / 3.0))
    
    return np.column_stack([x_ref, y_ref, theta_ref, v_ref])


def animate_simulation(state_history, obstacles, sampled_trajs_history, optimal_trajs_history, config, n_steps):
    """动画展示仿真过程"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.set_aspect('equal')
    ax.set_title("Mppi Obstacle Avoidance Simulation Bicycle Model")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    # 绘制道路元素
    ax.axhline(config.y_min, color='gray', linestyle='-', linewidth=2, label='Road Boundary')
    ax.axhline(config.y_max, color='gray', linestyle='-', linewidth=2)
    ax.axhline(0, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Road Centerline')

    # 起点和终点标记
    ax.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=8, label='Starting Point')
    ax.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='End')
    
    # 动画元素初始化
    path_line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7, label='Actual Trajectory')
    car_patch = plt_patches.Polygon([[0, 0]]*4, closed=True, color='blue', alpha=0.6)
    ax.add_patch(car_patch)
    
    # 障碍物可视化
    obstacle_patches = []
    for obs in obstacles:
        if obs[6] < 0.5:
            continue
        # 初始化障碍物多边形
        x0, y0, heading, _, length, width, _ = obs
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading), np.cos(heading)]])
        corners_local = np.array([
            [length/2, width/2], [length/2, -width/2],
            [-length/2, -width/2], [-length/2, width/2]
        ])
        corners_global = (R @ corners_local.T).T + np.array([x0, y0])
        obs_patch = plt_patches.Polygon(corners_global, closed=True, color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(obs_patch)
        obstacle_patches.append(obs_patch)

    # 预测轨迹可视化
    optimal_line, = ax.plot([], [], 'g-', lw=1.5, alpha=1.0, label='Optimal Predicted Trajectory')
    sampled_lines = [ax.plot([], [], '0.8', lw=0.5, alpha=0.2)[0] for _ in range(min(200, config.K))] 

    # 信息文本
    info_text = ax.text(0.02, 0.7, '', transform=ax.transAxes, fontsize=10)

    def _get_vehicle_corners(x, y, theta):
        corners_local = np.array([
            [config.vehicle_length/2,  config.vehicle_width/2],
            [config.vehicle_length/2, -config.vehicle_width/2],
            [-config.vehicle_length/2, -config.vehicle_width/2],
            [-config.vehicle_length/2,  config.vehicle_width/2]
        ])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return (R @ corners_local.T).T + np.array([x, y])

    def init_animation():
        path_line.set_data([], [])
        car_patch.set_xy(np.zeros((4, 2)))
        optimal_line.set_data([], [])
        for line in sampled_lines:
            line.set_data([], [])
        info_text.set_text('')
        return [path_line, car_patch, optimal_line] + sampled_lines + [info_text] + obstacle_patches 

    def update_animation(frame):
        idx = min(frame, len(state_history)-1)
        x, y, theta, v = state_history[idx]
        
        # 更新实际轨迹
        path_line.set_data(state_history[:idx+1, 0], state_history[:idx+1, 1])
        
        # 更新车辆位置
        car_corners = _get_vehicle_corners(x, y, theta)
        car_patch.set_xy(car_corners)
        
        # 更新信息文本
        info_text.set_text(
            f"Time: {idx*config.dt:.1f}s\n"
            f"Speed: {v:.2f}m/s\n"
            f"Heading Angle: {np.rad2deg(theta):.1f}°"
        )
        
        # 更新障碍物位置
        for obs_idx, obs in enumerate(obstacles):
            if obs[6] < 0.5:
                continue
            x0, y0, heading, v_obs, length, width, _ = obs
            t = config.dt * idx  # 当前时间
            x_obs = x0 + np.cos(heading) * v_obs * t
            y_obs = y0 + np.sin(heading) * v_obs * t
            
            # 更新障碍物多边形
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading), np.cos(heading)]])
            corners_local = np.array([
                [length/2, width/2], [length/2, -width/2],
                [-length/2, -width/2], [-length/2, width/2]
            ])
            corners_global = (R @ corners_local.T).T + np.array([x_obs, y_obs])
            obstacle_patches[obs_idx].set_xy(corners_global)
        
        # 更新预测轨迹
        if idx < len(optimal_trajs_history):
            optimal_line.set_data(optimal_trajs_history[idx][:, 0], optimal_trajs_history[idx][:, 1])
            # 显示部分采样轨迹
            for i, line in enumerate(sampled_lines):
                if i < min(len(sampled_trajs_history[idx]), config.K):
                    traj = sampled_trajs_history[idx][i]
                    line.set_data(traj[:, 0], traj[:, 1])
        
        return [path_line, car_patch, optimal_line] + sampled_lines + [info_text] + obstacle_patches

    # 创建并保存动画
    ani = animation.FuncAnimation(
        fig, update_animation, frames=n_steps+1, init_func=init_animation,
        interval=config.animation_interval, blit=True, repeat=False
    )
    
    writer = animation.PillowWriter(fps=10)
    ani.save(config.gif_filename, writer=writer)
    print(f"动画已保存为: {config.gif_filename}")
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重图例
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=4)
    
    plt.show()
    return ani


def plot_simulation_results(state_history, config):
    plt.figure(figsize=(10, 40))
    
    # 1. 轨迹图
    plt.subplot(4, 1, 1)
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', label='Actual Trajectory')
    plt.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Starting Point')
    plt.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Vehicle Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.xlim(config.xlim)
    plt.ylim(config.ylim)
    
    # 2. 航向角和速度变化
    plt.subplot(4, 1, 2)
    time = np.arange(len(state_history)) * config.dt
    plt.plot(time, np.rad2deg(state_history[:, 2]), 'g-', label='Heading Angle')
    plt.plot(time, state_history[:, 3], 'b-', label='Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Heading Angle and Speed Change')
    plt.grid(True)
    plt.legend()
    
    # 3. 控制输入变化
    plt.subplot(4, 1, 3)
    if len(state_history) > 1:
        steering_angle = np.rad2deg(state_history[:-1, 4])  # 转向角
        acceleration = np.diff(state_history[:, 3]) / config.dt  # 加速度
        time_ctrl = np.arange(len(steering_angle)) * config.dt
        
        plt.plot(time_ctrl, steering_angle, 'r-', label='Steering Angle (°)')
        plt.plot(time_ctrl, acceleration, 'c-', label='Acceleration (m/s²)')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Signal Change')
        plt.grid(True)
        plt.legend()
    
    # 4. 横向误差
    plt.subplot(4, 1, 4)
    lateral_error = state_history[:, 1]  # 相对于中心线的横向误差
    plt.plot(time, lateral_error, 'm-', label='Lateral Error')
    plt.axhline(0, color='g', linestyle='--', alpha=0.5)  # 中心线
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Lateral Error from Centerline')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.subplots_adjust(hspace=0.4)         
    plt.show()
