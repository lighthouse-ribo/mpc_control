import numpy as np
from util import Config, generate_reference_path, plot_simulation_results, animate_simulation
from mppi_obstacle_avoider import MPPIObstacleAvoider


def main():
    # 初始化配置与控制器
    config = Config()
    
    # 初始状态 [x, y, θ, v]
    initial_state = np.array([
        config.start_pos[0],
        config.start_pos[1],
        config.start_pos[2],
        config.start_pos[3]
    ])
    
    # 障碍物定义: [x0, y0, 航向角(rad), 速度(m/s), 长度(m), 宽度(m), 激活标志]
    obstacles = [
        [6.0, 1.0, np.deg2rad(0), 0.4, 2.0, 1.0, 1.0],   # 右侧障碍物（低速前进）
        [14.0, -1.3, np.deg2rad(180), 1.8, 2.0, 1.0, 1.0] # 左侧障碍物（反向行驶）
    ]
    
    # 生成全局参考路径
    global_ref_path = generate_reference_path(
        initial_state, config.goal_pos, 100, config.dt, config
    )
    
    # 初始化MPPI控制器
    mppi_controller = MPPIObstacleAvoider(config)
    
    # 仿真初始化
    n_steps = int(config.sim_time / config.dt)
    state_history = np.zeros((n_steps + 1, 5))  # [x, y, θ, v, δ]
    state_history[0, :4] = initial_state
    current_state = initial_state.copy()
    
    # 存储预测轨迹历史
    sampled_trajs_history = []
    optimal_trajs_history = []
    
    # 运行仿真循环
    real_steps = n_steps
    for i in range(n_steps):
        # 更新障碍物当前状态
        current_time = i * config.dt
        updated_obstacles = []
        for obs in obstacles:
            x0, y0, heading, v_obs, length, width, flag = obs
            x_obs = x0 + np.cos(heading) * v_obs * current_time
            y_obs = y0 + np.sin(heading) * v_obs * current_time
            updated_obstacles.append([x_obs, y_obs, heading, v_obs, length, width, flag])
        
        # 计算最优控制
        u_opt, _, optimal_traj, sampled_trajs, arrived = mppi_controller.compute_control(
            current_state, global_ref_path, updated_obstacles
        )
        
        if arrived:
            real_steps = i
            break
        
        # 记录预测轨迹
        sampled_trajs_history.append(sampled_trajs)
        optimal_trajs_history.append(optimal_traj)
        
        # 更新车辆状态
        next_state = mppi_controller.vehicle_model(current_state, u_opt).full().flatten()
        current_state = next_state
        
        # 记录状态（含转向角）
        state_history[i+1, :4] = current_state
        state_history[i+1, 4] = u_opt[0]
        
        # 打印状态信息
        print(f"步骤 {i+1}/{n_steps}: "
              f"位置=({current_state[0]:.2f}, {current_state[1]:.2f})m, "
              f"航向={np.rad2deg(current_state[2]):.1f}°, "
              f"速度={current_state[3]:.2f}m/s, "
              f"转向角={np.rad2deg(u_opt[0]):.1f}°")
        
        # 检查是否到达目标
        pos_error = np.linalg.norm(current_state[:2] - config.goal_pos[:2])
        if pos_error < config.goal_threshold:
            print(f"\n到达目标,位置误差: {pos_error:.2f}m")
            real_steps = i + 1
            break

    # 可视化结果
    animate_simulation(
        state_history[:real_steps+1, :4], 
        obstacles,
        sampled_trajs_history,
        optimal_trajs_history,
        config, 
        real_steps
    )
    plot_simulation_results(state_history[:real_steps+1], config)


if __name__ == "__main__":
    main()
