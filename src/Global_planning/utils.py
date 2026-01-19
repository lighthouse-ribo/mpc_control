import numpy as np
import math

def bresenham(r1, c1, r2, c2):
    """
    Bresenham 画线算法 (0-based)
    返回: (list_rows, list_cols)
    """
    r1, c1, r2, c2 = int(round(r1)), int(round(c1)), int(round(r2)), int(round(c2))
    
    if r1 == r2 and c1 == c2:
        return [r1], [c1]

    line_r = []
    line_c = []

    dx = abs(c2 - c1)
    dy = abs(r2 - r1)
    steep = dy > dx

    if steep:
        r1, c1 = c1, r1
        r2, c2 = c2, r2
    
    if r1 > r2:
        r1, r2 = r2, r1
        c1, c2 = c2, c1

    dx = r2 - r1
    dy = abs(c2 - c1)
    error = dx // 2
    ystep = 1 if c1 < c2 else -1
    y = c1

    for x in range(r1, r2 + 1):
        if steep:
            line_r.append(y)
            line_c.append(x)
        else:
            line_r.append(x)
            line_c.append(y)
            
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return line_r, line_c

def obs_dist_interp(obs_dist: np.ndarray, pos: np.ndarray) -> float:
    """
    双线性插值获取障碍物距离 (0-based, 解决精细化问题)
    :param obs_dist: 距离场矩阵
    :param pos: [row, col] 浮点坐标
    """
    rows, cols = obs_dist.shape
    r, c = pos[0], pos[1]

    # 边界钳制 (Clamping) 到 [0, rows-1]
    r_clamped = min(max(r, 0.0), rows - 1.001)
    c_clamped = min(max(c, 0.0), cols - 1.001)

    r_floor = int(math.floor(r_clamped))
    r_ceil = r_floor + 1
    c_floor = int(math.floor(c_clamped))
    c_ceil = c_floor + 1

    # 归一化偏移量
    dr = r_clamped - r_floor
    dc = c_clamped - c_floor

    # 获取四个邻域点的值
    # 这里的 min 是防止 ceil 越界
    v00 = obs_dist[r_floor, c_floor]
    v01 = obs_dist[r_floor, min(c_ceil, cols-1)]
    v10 = obs_dist[min(r_ceil, rows-1), c_floor]
    v11 = obs_dist[min(r_ceil, rows-1), min(c_ceil, cols-1)]

    # 双线性插值公式
    return (
        (1 - dr) * (1 - dc) * v00 +
        (1 - dr) * dc * v01 +
        dr * (1 - dc) * v10 +
        dr * dc * v11
    )

def is_path_segment_safe(p1, p2, grid, obs_dist, safety_margin):
    """
    检查路径段安全性 (包含碰撞检测 + 距离场检测)
    """
    # 采样点数：至少包含起终点，且密度足够
    dist = np.linalg.norm(p2 - p1)
    num_points = max(int(math.ceil(dist * 2)), 2) 
    
    # 生成线性插值点
    r_space = np.linspace(p1[0], p2[0], num_points)
    c_space = np.linspace(p1[1], p2[1], num_points)
    
    min_dist_found = float("inf")
    
    for r, c in zip(r_space, c_space):
        # 1. 越界检查
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False, 0.0
            
        # 2. 距离场检查 (使用插值获取高精度距离)
        d = obs_dist_interp(obs_dist, np.array([r, c]))
        min_dist_found = min(min_dist_found, d)
        
        if d < safety_margin:
            return False, min_dist_found
            
        # 3. 栅格硬碰撞检查 (辅助)
        if grid[int(round(r)), int(round(c))] == 1:
            return False, 0.0
            
    return True, min_dist_found

def is_strict_safe_move(curr, neighbor, grid, obs_dist, dir_vec, config):
    """
    严格的单步移动检查 (考虑对角线穿墙)
    """
    # 1. 目标点是否安全
    dist_neighbor = obs_dist_interp(obs_dist, neighbor)
    if dist_neighbor < config.safety_margin:
        return False, dist_neighbor
        
    # 2. 连线是否安全
    safe, min_d = is_path_segment_safe(curr, neighbor, grid, obs_dist, config.safety_margin)
    if not safe:
        return False, min_d
        
    # 3. 对角线穿墙检查 (防止从两个障碍物夹缝穿过)
    # 如果是斜向移动 (dx!=0 and dy!=0)
    if abs(dir_vec[0]) == 1 and abs(dir_vec[1]) == 1:
        # 检查两个“胳膊肘”位置
        corner1 = obs_dist_interp(obs_dist, np.array([curr[0], neighbor[1]]))
        corner2 = obs_dist_interp(obs_dist, np.array([neighbor[0], curr[1]]))
        
        if corner1 < config.safety_margin or corner2 < config.safety_margin:
            return False, min(min_d, corner1, corner2)
            
    return True, min_d

def is_direct_line_of_sight(p1, p2, grid, obs_dist, safety_margin):
    """视线检查 (使用 Bresenham)"""
    line_r, line_c = bresenham(p1[0], p1[1], p2[0], p2[1])
    for r, c in zip(line_r, line_c):
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if grid[r, c] == 1:
            return False
        # 使用中心点插值距离
        d = obs_dist_interp(obs_dist, np.array([r, c]))
        if d < safety_margin:
            return False
    return True