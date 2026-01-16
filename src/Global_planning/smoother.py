import numpy as np
import math
from .utils import obs_dist_interp, is_path_segment_safe

def b_spline_smooth(path, alpha):
    """
   
    使用均匀三次 B 样条对路径进行平滑 (手写实现)
    :param path: numpy array [[r, c], ...]
    :param alpha: 平滑因子 (config.smoothing_factor)
    """
    path = np.array(path)
    if len(path) < 4:
        return path

    n = len(path)
    # 根据 alpha 计算插值步数，保持与原代码一致的密度控制
    # 原代码: num_steps = int(math.ceil(n / (alpha / 2)))
    num_steps = int(math.ceil(n / (alpha / 2.0))) 
    smooth_path = []

    # 三次 B 样条基函数 (Basis Functions)
    B0 = lambda u: (1 - u) ** 3 / 6
    B1 = lambda u: (3 * u**3 - 6 * u**2 + 4) / 6
    B2 = lambda u: (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6
    B3 = lambda u: u**3 / 6

    for i in range(num_steps):
        u_global = i / (num_steps - 1)
        # 映射参数 u 到具体的控制点段
        u_val = u_global * (n - 3)
        segment_index = min(int(math.floor(u_val)), n - 4)
        local_u = u_val - segment_index

        # 边界保护
        if segment_index < 0 or segment_index > n - 4:
            continue

        p0 = path[segment_index]
        p1 = path[segment_index + 1]
        p2 = path[segment_index + 2]
        p3 = path[segment_index + 3]

        # 计算插值坐标
        new_r = (B0(local_u)*p0[0] + B1(local_u)*p1[0] + B2(local_u)*p2[0] + B3(local_u)*p3[0])
        new_c = (B0(local_u)*p0[1] + B1(local_u)*p1[1] + B2(local_u)*p2[1] + B3(local_u)*p3[1])
        
        smooth_path.append([new_r, new_c])

    smooth_path = np.array(smooth_path)

    # 二次平滑 (Secondary Smoothing)
    if len(smooth_path) > 4:
        smooth_path = secondary_smoothing(smooth_path)

    # 强制对齐起终点 (Anchor Start/End)
    if len(smooth_path) > 0:
        smooth_path[0] = path[0]
        smooth_path[-1] = path[-1]
        
    return smooth_path

def secondary_smoothing(path):
    """
   
    简单的滑动平均滤波 (权重: 0.2, 0.6, 0.2)
    """
    smoothed = path.copy()
    # 迭代 3 次
    for _ in range(3):
        new_path = smoothed.copy()
        # 原代码使用循环迭代更新
        for i in range(1, len(path) - 1):
            new_path[i] = 0.6 * smoothed[i] + 0.2 * smoothed[i - 1] + 0.2 * smoothed[i + 1]
        smoothed = new_path
    return smoothed

def find_safe_point(prev, nxt, grid_map, obs_dist, safety_margin):
    """
   
    径向搜索：当某点不安全时，在周围寻找安全的替代点
    """
    map_rows, map_cols = grid_map.shape
    mid_point = (prev + nxt) / 2.0
    candidate = mid_point

    # 搜索范围: 距离 0~2, 角度 0~360 (步长45度)
    for dist in range(0, 3):
        for angle in range(0, 360, 45):
            theta = math.radians(angle)
            offset = dist * np.array([math.cos(theta), math.sin(theta)])
            test_point = mid_point + offset
            
            # 越界检查
            if not (0 <= test_point[0] < map_rows and 0 <= test_point[1] < map_cols):
                continue

            # 获取该点的精确距离
            point_dist = obs_dist_interp(obs_dist, test_point)
            r_int = int(round(test_point[0]))
            c_int = int(round(test_point[1]))
            
            # 1. 检查点本身是否安全
            # 注意: grid_map 检查需要防止索引越界
            r_idx = min(max(r_int, 0), map_rows - 1)
            c_idx = min(max(c_int, 0), map_cols - 1)
            
            if point_dist >= safety_margin and grid_map[r_idx, c_idx] == 0:
                # 2. 检查与前后点的连线是否也安全
                safe_prev, _ = is_path_segment_safe(prev, test_point, grid_map, obs_dist, safety_margin)
                safe_next, _ = is_path_segment_safe(test_point, nxt, grid_map, obs_dist, safety_margin)
                
                if safe_prev and safe_next:
                    return test_point

    # 如果没找到，退化返回中点
    return candidate

def optimize_with_safety(path, grid_map, obs_dist, safety_margin):
    """
   
    安全性修正：检查平滑后的路径点，如果离障碍物太近，尝试将其推离
    """
    if len(path) < 2:
        return path

    optimized = [path[0]]
    map_rows, map_cols = grid_map.shape

    for i in range(1, len(path) - 1):
        curr = path[i]
        
        # 越界检查
        if not (0 <= curr[0] < map_rows and 0 <= curr[1] < map_cols):
            # 越界视为不安全，寻找替代点
            candidate = find_safe_point(np.array(optimized[-1]), path[i+1], grid_map, obs_dist, safety_margin)
            optimized.append(candidate)
            continue
            
        dist = obs_dist_interp(obs_dist, curr)
        r_int = int(round(curr[0]))
        c_int = int(round(curr[1]))
        r_idx = min(max(r_int, 0), map_rows - 1)
        c_idx = min(max(c_int, 0), map_cols - 1)

        # 碰撞或距离过近
        if grid_map[r_idx, c_idx] == 1 or dist < safety_margin:
            # 寻找更安全的点
            candidate = find_safe_point(np.array(optimized[-1]), path[i+1], grid_map, obs_dist, safety_margin)
            optimized.append(candidate)
        else:
            optimized.append(curr)

    optimized.append(path[-1])
    return np.array(optimized)

def fix_sharp_turns(path, max_angle_deg):
    """
   
    修复急转弯：检测向量夹角，如果过大则插入中间控制点
    """
    if len(path) < 3:
        return path

    # 使用 list 以方便动态插入
    optimized = list(path)
    i = 1
    
    while i < len(optimized) - 1:
        p_prev = np.array(optimized[i-1])
        p_curr = np.array(optimized[i])
        p_next = np.array(optimized[i+1])
        
        vec1 = p_curr - p_prev
        vec2 = p_next - p_curr
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 防止除零
        if norm1 < 1e-6 or norm2 < 1e-6:
            i += 1
            continue
            
        cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_theta))
        
        if angle > max_angle_deg:
            # 在急转弯处插入中间点 (Midpoint insertion)
            mid1 = (p_prev + p_curr) / 2.0
            mid2 = (p_curr + p_next) / 2.0
            control = (mid1 + mid2) / 2.0
            
            optimized.insert(i+1, control)
            # 原始逻辑是 i+=1，跳过新插入的点检查后续
            i += 1 
        i += 1
        
    return np.array(optimized)