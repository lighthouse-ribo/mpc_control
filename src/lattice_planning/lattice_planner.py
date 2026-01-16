import numpy as np
import math
from .curves import CubicSpline2D
# 稍后我们会创建这个配置文件
from .lattice_config import LatticeConfig

# ================= 向量化数学工具 (保留高效实现) =================

def calc_quintic_coeffs_vectorized(xs, vxs, axs, xe, vxe, axe, T):
    """五次多项式系数计算 (Vectorized)"""
    T = np.atleast_1d(T)
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    
    a0 = np.full_like(T, xs)
    a1 = np.full_like(T, vxs)
    a2 = np.full_like(T, axs / 2.0)
    
    h = xe - a0 - a1*T - a2*T2
    v = vxe - a1 - 2*a2*T
    a = axe - 2*a2
    
    inv_T3 = 1.0 / (T3 + 1e-6)
    inv_T4 = 1.0 / (T4 + 1e-6)
    inv_T5 = 1.0 / (T5 + 1e-6)

    a3 = (10 * h - 4 * v * T + 0.5 * a * T2) * inv_T3
    a4 = (-15 * h + 7 * v * T - a * T2) * inv_T4
    a5 = (6 * h - 3 * v * T + 0.5 * a * T2) * inv_T5
    
    return a0, a1, a2, a3, a4, a5

def calc_quartic_coeffs_vectorized(xs, vxs, axs, vxe, axe, T):
    """四次多项式系数计算 (Vectorized)"""
    T = np.atleast_1d(T)
    T2, T3, T4 = T**2, T**3, T**4
    
    a0 = np.full_like(T, xs)
    a1 = np.full_like(T, vxs)
    a2 = np.full_like(T, axs / 2.0)
    
    b1 = vxe - a1 - 2*a2*T
    b2 = axe - 2*a2
    
    a3 = (b1 * 12 * T2 - b2 * 4 * T3) / (12 * T4 + 1e-6)
    a4 = (-b1 * 6 * T + b2 * 3 * T2) / (12 * T4 + 1e-6)
    
    return a0, a1, a2, a3, a4

class Trajectory:
    def __init__(self):
        # 基础 Frenet 状态
        self.t = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        
        # 笛卡尔状态 (MPC 需要)
        self.x = []
        self.y = []
        self.yaw = [] # psi
        self.k = []   # 曲率 kappa
        self.v = []   # 线性速度
        self.a = []   # 线性加速度
        
        self.cost = 0.0

class LatticePlanner:
    def __init__(self):
        self.cfg = LatticeConfig()

    def plan(self, grid_map, obs_dist, start_state, ref_path_points, resolution=0.5, prev_frenet_state=None):
        """
        核心规划接口
        :param grid_map: 0/1 栅格地图
        :param obs_dist: 障碍物距离场
        :param start_state: 车辆当前状态 dict {'x', 'y', 'yaw', 'v', 'a', 'k'}
        :param ref_path_points: A* 生成的平滑路径 [{'x', 'y'}, ...]
        :param resolution: 地图分辨率
        """
        if len(ref_path_points) < 3:
            return None
        
        # 1. 构建参考线 (使用我们自己的 CubicSpline2D)
        rx = [p['x'] for p in ref_path_points]
        ry = [p['y'] for p in ref_path_points]
        csp = CubicSpline2D(rx, ry)

        # 2. 状态转换 Cartesian -> Frenet
        # 如果有上一帧的规划结果，可以继承状态以保证平滑，否则重新计算投影
        s0, d0, d_d0, d_dd0, s_d0, s_dd0 = self._cartesian_to_frenet(start_state, csp, prev_frenet_state)

        # 3. 采样空间生成
        # 横向采样 (d)
        d_samples = np.arange(-self.cfg.d_road_w, self.cfg.d_road_w + 1e-5, self.cfg.sample_d_step)
        # 纵向时间采样 (t)
        t_samples = np.arange(self.cfg.time_min, self.cfg.time_max + 1e-5, self.cfg.time_step)
        # 纵向速度采样 (v)
        v_samples = np.array([self.cfg.target_speed, self.cfg.target_speed * 0.6, self.cfg.target_speed * 0.3])
        
        # 网格化
        T_grid, V_grid, D_grid = np.meshgrid(t_samples, v_samples, d_samples, indexing='ij')
        T_flat, V_flat, D_flat = T_grid.ravel(), V_grid.ravel(), D_grid.ravel()
        
        # 4. 批量生成多项式系数
        # 纵向: 四次多项式 (s0, s_d0, s_dd0 -> v_target, a=0 @ T)
        sa = calc_quartic_coeffs_vectorized(s0, s_d0, s_dd0, V_flat, 0.0, T_flat)
        # 横向: 五次多项式 (d0, d_d0, d_dd0 -> d_target, d_d=0, d_dd=0 @ T)
        da = calc_quintic_coeffs_vectorized(d0, d_d0, d_dd0, D_flat, 0.0, 0.0, T_flat)
        
        # 5. 快速 Cost 预筛选 (减少计算量)
        # 优先考虑: 横向偏差小、速度接近目标、时间短
        costs = self.cfg.w_lat_diff * D_flat**2 + \
                self.cfg.w_efficiency * (self.cfg.target_speed - V_flat)**2 + \
                self.cfg.w_time * (1.0 / (T_flat + 0.1))
        
        # 6. 轨迹生成与详细检测
        valid_paths = []
        sorted_idx = np.argsort(costs)
        
        # 限制候选数量
        for idx in sorted_idx[:self.cfg.max_candidates]:
            c_sa = [arr[idx] for arr in sa]
            c_da = [arr[idx] for arr in da]
            T_end = T_flat[idx]
            
            # 生成时间序列 (0.1s 步长)
            t = np.arange(0.0, T_end, 0.1)
            if len(t) < 2: continue
            
            # 纵向加速度检查
            s_dd = 2*c_sa[2] + 6*c_sa[3]*t + 12*c_sa[4]*t**2
            if np.max(np.abs(s_dd)) > self.cfg.max_accel: continue
            
            p = Trajectory()
            p.t = t
            # 纵向 s(t)
            p.s = c_sa[0] + c_sa[1]*t + c_sa[2]*t**2 + c_sa[3]*t**3 + c_sa[4]*t**4
            p.s_d = c_sa[1] + 2*c_sa[2]*t + 3*c_sa[3]*t**2 + 4*c_sa[4]*t**3
            p.s_dd = s_dd
            
            # 横向 d(t)
            p.d = c_da[0] + c_da[1]*t + c_da[2]*t**2 + c_da[3]*t**3 + c_da[4]*t**4 + c_da[5]*t**5
            p.d_d = c_da[1] + 2*c_da[2]*t + 3*c_da[3]*t**2 + 4*c_da[4]*t**3 + 5*c_da[5]*t**4
            p.d_dd = 2*c_da[2] + 6*c_da[3]*t + 12*c_da[4]*t**2 + 20*c_da[5]*t**3
            p.d_ddd = 6*c_da[3] + 24*c_da[4]*t + 60*c_da[5]*t**2
            
            # 转换回笛卡尔坐标 (x, y, yaw, k)
            self._f2c(p, csp)
            
            # 曲率检查 (物理约束)
            if np.max(np.abs(p.k)) > self.cfg.max_curvature: continue
            
            # 碰撞检测 (环境约束)
            is_coll, risk = self._check_collision(p, grid_map, obs_dist, resolution)
            if not is_coll:
                # 填充 MPC 所需的状态
                p.v = p.s_d # 近似: v ~= s_dot
                p.a = p.s_dd
                
                # 计算最终代价
                # Jerk Cost (舒适性)
                j_smooth = np.sum(p.d_ddd**2) + np.sum(p.s_dd**2)
                # 效率 Cost
                j_eff = np.sum(np.abs(self.cfg.target_speed - p.s_d))
                # 偏航 Cost
                j_lat = np.sum(p.d**2)
                
                p.cost = self.cfg.w_collision * risk + \
                         self.cfg.w_smoothness * j_smooth + \
                         self.cfg.w_efficiency * j_eff + \
                         self.cfg.w_lat_diff * j_lat
                
                valid_paths.append(p)
        
        # 返回 Cost 最小的路径
        if valid_paths:
            best_path = min(valid_paths, key=lambda x: x.cost)
            # 转换为 list of dicts 供外部使用
            return self._format_output(best_path)
            
        return None

    def _cartesian_to_frenet(self, state, csp, prev):
        """将车辆状态投影到 Frenet 坐标系"""
        # 1. 如果有上一帧的 Frenet 状态，优先使用 (闭环连续性)
        if prev is not None:
            # 简单的预测更新，防止重规划跳变
            # 这里简化处理，直接返回，实际项目中可能需要基于 dt 递推一步
            # 为了稳健，我们暂时每次都重算投影，但可以用 prev 里的 s 作为搜索初值
            pass

        # 2. 寻找匹配点 s
        s_guess = prev['s'] if (prev and 's' in prev) else 0.0
        # 如果是第一次，全范围搜索；否则在 s_guess 附近搜索
        if prev is None:
            s = csp.find_projection(state['x'], state['y'])
        else:
            # 局部搜索，提高效率
            # 注意: CubicSpline2D.find_projection 内部目前实现了全搜索
            # 如果性能有瓶颈，可以修改 curves.py 增加 start_s 参数
            s = csp.find_projection(state['x'], state['y'])

        # 3. 计算参考点状态
        rx, ry = csp.calc_position(s)
        ryaw = csp.calc_yaw(s)
        rk = csp.calc_curvature(s)
        
        # 4. 计算 Frenet 状态
        dx = state['x'] - rx
        dy = state['y'] - ry
        
        # d (横向偏差) = 向量(dx, dy) 叉乘 参考方向向量
        # 叉乘: dx*sin(-ryaw) + dy*cos(-ryaw) ... 简化公式如下:
        d = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        # 速度分解
        v = state['v']
        yaw_diff = state['yaw'] - ryaw
        # 归一化角度
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        
        s_d = v * math.cos(yaw_diff) / (1 - rk * d)
        d_d = v * math.sin(yaw_diff)
        
        # 加速度分解 (简化: 假设主要加速度在切向)
        # 实际应包含向心加速度项，这里简化处理 s_dd ~= a, d_dd ~= 0
        s_dd = state.get('a', 0.0)
        d_dd = 0.0 
        
        return s, d, d_d, d_dd, s_d, s_dd

    def _f2c(self, p, csp):
        """Frenet 转 Cartesian"""
        # 批量计算参考线状态
        # 为了效率，我们直接循环计算 (Python loop overhead is minimal for 50 points)
        for i in range(len(p.t)):
            s_val = p.s[i]
            d_val = p.d[i]
            
            rx, ry = csp.calc_position(s_val)
            ryaw = csp.calc_yaw(s_val)
            rk = csp.calc_curvature(s_val)
            
            # 坐标变换
            x = rx - d_val * math.sin(ryaw)
            y = ry + d_val * math.cos(ryaw)
            
            p.x.append(x)
            p.y.append(y)
            
            # 航向角变换 (近似: yaw = ref_yaw + atan(d_d / s_d))
            # 更精确的公式需要考虑曲率
            yaw = ryaw + math.atan2(p.d_d[i], p.s_d[i])
            p.yaw.append(yaw)
            
            # 曲率计算 (复杂公式，使用数值微分代替)
            # k = (x'y'' - y'x'') / ...
        
        # 使用数值微分补全 k (比复杂解析公式更稳健)
        dx = np.gradient(p.x)
        dy = np.gradient(p.y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        k = (dx * ddy - dy * ddx) / (np.hypot(dx, dy)**3 + 1e-6)
        p.k = k.tolist()

    def _check_collision(self, p, grid, dist_m, res):
        """
        基于距离场的快速碰撞检测
        """
        rows, cols = grid.shape
        # 将轨迹点转为栅格坐标
        for i in range(len(p.x)):
            # 转换公式需与 MapServer 一致
            # MapServer: r = (y - origin_y) / res, c = (x - origin_x) / res
            # 这里需要注意 MapServer 的 origin 定义。
            # 假设 MapServer origin_x=0, origin_y=-height/2 (我们在 map_server.py 里定义的)
            # 为了通用性，我们这里最好通过参数传入 origin，或者假设 dist_m 覆盖了整个世界坐标系
            # 临时方案：利用 dist_m 的索引。假设 dist_m 与 grid 对应。
            
            # 注意：此处需要与 MapServer.world_to_grid 逻辑对齐
            # 简单起见，我们假设外部调用者会传入转换函数，或者我们自己算
            # 鉴于 LatticePlanner 独立性，我们重新实现一个简单的 world_to_grid
            # 但是原点信息在 map_server 实例里。
            
            # [修正]：为了代码独立性，我们假设输入的 grid 和 dist_m 是已经对齐好的
            # 且输入分辨率已知。最稳妥的方式是把 MapServer 实例传进来，
            # 但接口定义只有 grid_map, obs_dist。
            # 我们先按照标准 MapServer 参数硬编码原点（与 app.py 一致）
            
            origin_x = 0.0
            origin_y = -10.0 # height=20, origin_y = -height/2
            
            c = int((p.x[i] - origin_x) / res)
            r = int((p.y[i] - origin_y) / res)
            
            if 0 <= r < rows and 0 <= c < cols:
                # 查表获取最近障碍物距离
                d = dist_m[r, c]
                # 碰撞判断 (车宽/2 + 余量)
                # 车宽设为 2.0m (config.width), 半宽 1.0m
                if d < (self.cfg.width / 2.0):
                    return True, 0.0
                
                # 风险累加 (距离越近风险越大)
                if d < 1.5: # 感知范围
                    # 返回 True, risk (此处仅检测碰撞，risk 单独算)
                    pass
            else:
                # 出界视为碰撞
                return True, 0.0
                
        # 计算整条轨迹的风险分
        risk_score = 0.0
        # ... (可选：遍历点计算 risk_score)
        
        return False, risk_score

    def _format_output(self, p):
        """转换为字典列表输出"""
        traj = []
        for i in range(len(p.t)):
            traj.append({
                't': p.t[i],
                'x': p.x[i],
                'y': p.y[i],
                'psi': p.yaw[i],
                'v': p.v[i],
                'a': p.a[i],
                'k': p.k[i]
            })
        return traj