class LatticeConfig:
    def __init__(self):
        # --- 车辆参数 ---
        self.width = 1.8           # 车宽 (米)
        self.length = 4.0          # 车长
        self.max_speed = 10.0      # 最大速度
        self.max_accel = 2.0       # 最大加速度
        self.max_curvature = 0.5   # 最大曲率 (约 30度前轮转角)
        
        # --- 采样参数 ---
        self.target_speed = 3.5    # 目标巡航速度 (m/s)
        self.time_min = 2.0        # 最小预测时间
        self.time_max = 4.0        # 最大预测时间
        self.time_step = 0.5       # 时间步长
        self.d_road_w = 3.5        # 道路半宽 (采样范围 +/-)
        self.sample_d_step = 0.5   # 横向采样步长
        
        # --- 代价权重 (Weights) ---
        self.w_collision = 1000.0  # 碰撞惩罚 (极大)
        self.w_smoothness = 10.0   # 平滑性 (Jerk)
        self.w_lat_diff = 2.0      # 横向偏差 (尽量走中间)
        self.w_efficiency = 5.0    # 效率 (接近目标速度)
        self.w_time = 1.0          # 时间成本
        
        # --- 候选数量 ---
        self.max_candidates = 20   # 性能优化：只精细计算前 N 个候选





        