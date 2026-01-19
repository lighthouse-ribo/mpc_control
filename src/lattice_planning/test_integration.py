import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# 路径补丁：确保能导入 src 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.map_server import MapServer
from src.config import PlanningConfig
from src.Global_planning import EnhancedAStar
from src.lattice_planning.lattice_planner import LatticePlanner

def plot_integration_result(map_server, start, goal, ref_path, history_path, filename="integration_result.png"):
    """
    绘制集成规划结果：
    Subplot 1: 地图 + A*参考线 + 实际轨迹
    Subplot 2: 速度与曲率剖面
    """
    # [修改] 恢复创建两个子图 (2行1列)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- 子图1: 轨迹视图 ---
    # 1. 画地图
    ax1.imshow(1 - map_server.grid, cmap='gray', origin='lower', extent=[
        map_server.origin_x, 
        map_server.origin_x + map_server.width,
        map_server.origin_y,
        map_server.origin_y + map_server.height
    ], alpha=0.5)
    
    # 2. 画起终点
    ax1.plot(start[0], start[1], 'go', markersize=12, label='Start')
    ax1.plot(goal[0], goal[1], 'rx', markersize=12, label='Goal')
    
    # 3. 画 A* 全局参考线
    if ref_path is not None:
        rx = [p['x'] for p in ref_path]
        ry = [p['y'] for p in ref_path]
        ax1.plot(rx, ry, 'g--', linewidth=1.5, alpha=0.8, label='Global A* Reference')
    
    # 4. 画车辆实际行驶轨迹
    if history_path:
        hx = [p['x'] for p in history_path]
        hy = [p['y'] for p in history_path]
        ax1.plot(hx, hy, 'r-', linewidth=2.5, label='Actual Driven Path')
        # 标注卡住的位置
        ax1.plot(hx[-1], hy[-1], 'ko', markersize=8, label='End/Stuck Pos')
        
    ax1.set_title(f"Integration Test: A* + Lattice Loop (Steps: {len(history_path)})")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # --- 子图2: 动力学剖面 ---
    if history_path:
        # 构造时间轴 (假设每步 0.1s，这只是近似，用于可视化)
        t = [i * 0.1 for i in range(len(history_path))]
        v = [p['v'] for p in history_path]
        k = [p['k'] for p in history_path]
        
        # 左轴：速度
        ln1 = ax2.plot(t, v, 'r-', linewidth=2, label='Velocity (m/s)')
        ax2.set_ylabel("Velocity [m/s]", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True)
        ax2.set_xlabel("Simulation Steps (approx time)")
        
        # 右轴：曲率
        ax2b = ax2.twinx()
        ln2 = ax2b.plot(t, k, 'b:', linewidth=1, label='Curvature (1/m)')
        ax2b.set_ylabel("Curvature [1/m]", color='b')
        ax2b.tick_params(axis='y', labelcolor='b')
        
        # 合并图例
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='upper right')
        ax2.set_title("Dynamics Profile")

    print(f"Saving visualization to {filename}...")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("=== Integration Test: Simulation Loop ===")
    
    # 1. 初始化
    cfg = PlanningConfig()
    ms = MapServer()
    astar = EnhancedAStar(cfg)
    lattice = LatticePlanner()
    
    # 2. 定义场景
    # 场景：从左侧绕过中间障碍物去右侧
    start_pos = (30.0, 0.0) 
    end_pos = (80.0, 0.0)
    
    # 3. [Step 1] 全局规划 A* (只跑一次!)
    print("[1/2] Running Global Planner (A*)...")
    sr, sc = ms.world_to_grid(*start_pos)
    er, ec = ms.world_to_grid(*end_pos)
    
    if not (ms.is_valid(sr, sc) and ms.is_valid(er, ec)):
        print("Error: Start/Goal out of bounds!")
        return

    # A* 算出一个静态参考路径
    ref_path = astar.plan(ms.grid, (sr, sc), (er, ec), ms.obs_dist, map_info=ms)
    
    if not ref_path:
        print("Global Planning Failed!")
        return
    print(f"Global Path Generated: {len(ref_path)} points.")

    # 4. [Step 2] 局部规划循环 (Simulation Loop)
    print("[2/2] Starting Lattice Loop Simulation...")
    
    # 初始状态
    current_state = {
        'x': start_pos[0],
        'y': start_pos[1],
        'yaw': 0.0, # 初始朝向
        'v': 0.0,   # 静止起步
        'a': 0.0,
        'k': 0.0
    }
    
    # 自动推断初始朝向 (对齐 A* 切线)
    if len(ref_path) > 1:
        dx = ref_path[1]['x'] - ref_path[0]['x']
        dy = ref_path[1]['y'] - ref_path[0]['y']
        current_state['yaw'] = math.atan2(dy, dx)

    history_path = [] # 记录车跑过的路
    max_steps = 500   # 防止死循环
    goal_tolerance = 1.0 # 到达阈值 (米)
    
    for step in range(max_steps):
        # 4.1 检查是否到达终点
        dist_to_goal = math.hypot(current_state['x'] - end_pos[0], current_state['y'] - end_pos[1])
        if dist_to_goal < goal_tolerance:
            print(f"Goal Reached! (Final dist: {dist_to_goal:.2f}m)")
            break
            
        # 4.2 调用 Lattice 规划器 (基于当前位置)
        # Lattice 会根据当前车的位置，在 A* 参考线附近生成一段局部最优轨迹
        local_traj = lattice.plan(
            grid_map=ms.grid,
            obs_dist=ms.obs_dist,
            start_state=current_state,
            ref_path_points=ref_path,
            resolution=ms.resolution
        )
        
        if not local_traj:
            print(f"Lattice Planning Failed at step {step}! Car stuck.")
            break
            
        # 4.3 模拟车辆运动 (Move Car)
        # 在真实自动驾驶中，我们会把整条 local_traj 发给控制模块执行
        # 在这个测试脚本中，我们简单地"瞬移"到轨迹的一小段之后，模拟车已经开过去了
        
        # 选取轨迹中向前的一点作为新的当前位置 (例如第 3 个点，模拟走了 0.3~0.5秒)
        # Lattice 输出的 t 分辨率通常是 0.1s
        move_idx = min(len(local_traj) - 1, 3) # 每次向前推进 3 个时间步
        next_pt = local_traj[move_idx]
        
        # 更新车辆状态
        current_state = {
            'x': next_pt['x'],
            'y': next_pt['y'],
            'yaw': next_pt['psi'], # Lattice 输出的是 psi
            'v': next_pt['v'],
            'a': next_pt['a'],
            'k': next_pt['k']
        }
        
        # 记录历史
        history_path.append(current_state)
        
        if step % 10 == 0:
            print(f"Step {step}: x={current_state['x']:.1f}, y={current_state['y']:.1f}, v={current_state['v']:.2f}")

    # 5. 画图验证
    plot_integration_result(ms, start_pos, end_pos, ref_path, history_path)
    print("Done. Check 'integration_result.png'.")

if __name__ == "__main__":
    main()