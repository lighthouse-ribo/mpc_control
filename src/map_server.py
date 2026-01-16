import numpy as np
from scipy.ndimage import distance_transform_edt

class MapServer:
    def __init__(self, config=None):
        """
        初始化地图服务
        :param config: 规划配置对象 (可选)
        """
        # 地图参数 (单位: 米)
        self.width = 180.0
        self.height = 20.0
        self.resolution = 0.5
        
        # 计算栅格尺寸
        self.cols = int(self.width / self.resolution)
        self.rows = int(self.height / self.resolution)
        
        # 坐标原点偏移 (定义 (0,0) 在地图左侧中间)
        # grid[0] 对应 y = -10m, grid[rows] 对应 y = +10m
        self.origin_x = 0.0
        self.origin_y = -self.height / 2.0 
        
        self.grid = None      # 0=空闲, 1=占用
        self.obs_dist = None  # 欧几里得距离场 (EDT)
        
        # 初始化构建
        self._build_static_map()

    def _build_static_map(self):
        """
        生成静态障碍物地图
        移植自 create_map 函数
        """
        self.grid = np.zeros((self.rows, self.cols))
        
        # 1. 边界障碍物 (上下边界各 1m 厚)
        # 2格 * 0.5m = 1.0m
        self.grid[0:2, :] = 1.0
        self.grid[-2:, :] = 1.0
        
        # 2. 自定义障碍物列表 (x, y)
        obstacles = [
            (35, 4), (35, -4), (38, 3), (38, -3),
            (60, 0), (60, 1.5), (60, -1.5), (55, 0), (55, -1),
            (65, 0), (65, 2), (65, -2)
        ]
        
        for ox, oy in obstacles:
            # 将物理坐标转为栅格坐标
            r, c = self.world_to_grid(ox, oy)
            if self.is_valid(r, c):
                # 简单膨胀：以中心点向外扩展 
                # (r-1 ~ r+1) -> 3格 (1.5m)
                # (c-2 ~ c+2) -> 4格 (2.0m)
                r_min, r_max = max(0, r-1), min(self.rows, r+1)
                c_min, c_max = max(0, c-2), min(self.cols, c+2)
                self.grid[r_min:r_max, c_min:c_max] = 1.0
        
        # 3. 计算障碍物距离场 (Euclidean Distance Transform)
        # distance_transform_edt 计算的是到“非零背景”的距离，所以我们输入 1-grid
        # 结果乘以 resolution 转换为米
        self.obs_dist = distance_transform_edt(1.0 - self.grid) * self.resolution
        
        print(f"[MapServer] Map generated: {self.width}x{self.height}m ({self.cols}x{self.rows} grids)")

    def world_to_grid(self, x, y):
        """世界坐标 (m) -> 栅格索引 (r, c)"""
        c = int((x - self.origin_x) / self.resolution)
        r = int((y - self.origin_y) / self.resolution)
        return r, c

    def grid_to_world(self, r, c):
        """栅格索引 (r, c) -> 世界坐标 (m)"""
        x = c * self.resolution + self.origin_x
        y = r * self.resolution + self.origin_y
        return x, y
        
    def is_valid(self, r, c):
        """检查索引是否在地图范围内"""
        return 0 <= r < self.rows and 0 <= c < self.cols
        
    def check_collision(self, x, y):
        """检查世界坐标 (x,y) 是否碰撞"""
        r, c = self.world_to_grid(x, y)
        if not self.is_valid(r, c):
            return True # 出界视为碰撞
        return self.grid[r, c] == 1.0