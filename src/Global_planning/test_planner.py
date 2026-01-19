import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录加入 python path，确保能导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map_server import MapServer
from src.config import PlanningConfig
from src.Global_planning.a_star import EnhancedAStar

def plot_planning_result(map_server, start_world, end_world, raw_path_grid, smooth_path_world, filename="verification_result.png"):
    """
    可视化规划结果：显示地图、障碍物、A*原始路径、平滑路径
    """
    plt.figure(figsize=(12, 6))
    
    # 1. 绘制地图障碍物
    # grid 是 0/1 矩阵，1为障碍
    # 为了显示方便，我们用 imshow，注意原点处理
    # grid 索引 (r, c) -> image (y, x)
    # 我们直接画 grid 矩阵，然后把轨迹点转换回 grid 坐标来叠加显示会更直观
    
    plt.imshow(1 - map_server.grid, cmap='gray', origin='lower', extent=[
        map_server.origin_x, 
        map_server.origin_x + map_server.width,
        map_server.origin_y,
        map_server.origin_y + map_server.height
    ])
    
    # 2. 绘制起终点
    plt.plot(start_world[0], start_world[1], 'go', markersize=10, label='Start')
    plt.plot(end_world[0], end_world[1], 'rx', markersize=10, label='Goal')
    
    # 3. 绘制原始 A* 路径 (折线)
    # raw_path_grid 是 [(r, c), ...]
    if raw_path_grid is not None and len(raw_path_grid) > 0:
        rx, ry = [], []
        for p in raw_path_grid:
            wx, wy = map_server.grid_to_world(p[0], p[1])
            rx.append(wx)
            ry.append(wy)
        plt.plot(rx, ry, 'b--', linewidth=1, alpha=0.7, label='Raw A* (Grid)')
        
    # 4. 绘制平滑后的路径 (曲线)
    # smooth_path_world 是 [{'x':x, 'y':y}, ...]
    if smooth_path_world is not None and len(smooth_path_world) > 0:
        sx = [p['x'] for p in smooth_path_world]
        sy = [p['y'] for p in smooth_path_world]
        plt.plot(sx, sy, 'g-', linewidth=2, label='Smoothed (B-Spline)')
        
    plt.title(f"A* Planning Verification\nStart:{start_world}, Goal:{end_world}")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    print(f"Saving visualization to {filename}...")
    plt.savefig(filename)
    plt.close()

def main():
    print("=== Initializing Planning System ===")
    
    # 1. 初始化配置与地图
    cfg = PlanningConfig()
    print(f"Config Loaded: SafetyMargin={cfg.safety_margin}m, Smoothing={cfg.smoothing_factor}")
    
    ms = MapServer() # 会打印地图信息
    
    # 2. 初始化规划器
    planner = EnhancedAStar(cfg)
    
    # 3. 定义测试场景 (世界坐标)
    # 场景：从左侧穿过中间的障碍物群到达右侧
    start_pos = (30.0, 0.0) 
    end_pos = (80.0, 0.0)
    
    print(f"\nPlanning from {start_pos} to {end_pos}...")
    
    # 4. 坐标转换
    sr, sc = ms.world_to_grid(*start_pos)
    er, ec = ms.world_to_grid(*end_pos)
    
    if not (ms.is_valid(sr, sc) and ms.is_valid(er, ec)):
        print("Error: Start or Goal is out of map bounds!")
        return

    # 5. 为了对比，我们需要稍微“破解”一下 plan 函数，分别获取原始路径和平滑路径
    # 正常调用 plan 只能拿到最终结果
    
    # A) 获取原始 A* 路径 (Hack for visualization)
    # 临时关闭平滑开关
    original_smooth_setting = cfg.use_bspline_smoothing
    cfg.use_bspline_smoothing = False 
    
    raw_path_grid = planner.plan(ms.grid, (sr, sc), (er, ec), ms.obs_dist, map_info=None)
    print(f"Raw A* Path Nodes: {len(raw_path_grid)}")
    
    # B) 获取最终平滑路径
    # 恢复开关
    cfg.use_bspline_smoothing = original_smooth_setting
    
    final_path_world = planner.plan(ms.grid, (sr, sc), (er, ec), ms.obs_dist, map_info=ms)
    print(f"Smoothed Path Points: {len(final_path_world)}")
    
    if len(final_path_world) == 0:
        print("Planning Failed!")
    else:
        print("Planning Success!")
        # 6. 画图
        plot_planning_result(ms, start_pos, end_pos, raw_path_grid, final_path_world)
        print("Done. Please check 'verification_result.png'.")

if __name__ == "__main__":
    main()