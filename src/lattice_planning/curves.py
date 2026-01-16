import math
import numpy as np
import bisect

class CubicSpline1D:
    """
    1D 三次样条插值
    用于计算 x(s) 和 y(s)
    """
    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted")

        self.a = np.array(y, dtype=float)
        self.x = np.array(x, dtype=float)
        self.nx = len(x)
        
        # 求解线性方程组 Ax = B 计算二阶导数系数 c
        # 使用自然样条边界条件: c[0] = 0, c[n-1] = 0
        A = np.zeros((self.nx, self.nx))
        B = np.zeros(self.nx)
        
        A[0, 0] = 1.0
        
        # [Fix] 循环范围修正为 range(self.nx - 2)
        # 仅遍历内部节点 1 到 nx-2 构建方程
        for i in range(self.nx - 2):
            # 对应行索引为 i + 1
            # 涉及节点: i, i+1, i+2
            
            # 对角线 A[i+1, i+1] 对应 c[i+1]
            A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            
            # 左侧 A[i+1, i] 对应 c[i]
            A[i + 1, i] = h[i]
            
            # 右侧 A[i+1, i+2] 对应 c[i+2]
            A[i + 1, i + 2] = h[i + 1]
            
            # 右侧向量 B[i+1]
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
            
        A[self.nx - 1, self.nx - 1] = 1.0
        
        # 求解 c
        self.c = np.linalg.solve(A, B)
        
        # 计算 b 和 d
        self.b = np.zeros(self.nx - 1)
        self.d = np.zeros(self.nx - 1)
        for i in range(self.nx - 1):
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            self.b[i] = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0

    def calc_position(self, x):
        """计算位置"""
        if x < self.x[0]: return self.a[0]
        if x > self.x[-1]: return self.a[-1]

        i = self._search_index(x)
        dx = x - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3

    def calc_first_derivative(self, x):
        """计算一阶导数 (速度/斜率)"""
        if x < self.x[0] or x > self.x[-1]: return 0.0

        i = self._search_index(x)
        dx = x - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2

    def calc_second_derivative(self, x):
        """计算二阶导数 (加速度)"""
        if x < self.x[0] or x > self.x[-1]: return 0.0

        i = self._search_index(x)
        dx = x - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    def _search_index(self, x):
        """二分查找 x 所在的区间"""
        # bisect 可能会返回 len(x)，需要钳制到 len(x)-1
        idx = bisect.bisect(self.x, x) - 1
        return max(0, min(idx, self.nx - 2))


class CubicSpline2D:
    """
    2D 三次样条类
    输入一组 (x, y) 点，自动计算弧长 s，并提供 s -> (x, y, yaw, k) 的映射
    这是 Reference Line 的核心
    """
    def __init__(self, x, y):
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def _calc_s(self, x, y):
        """计算累计弧长 s"""
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(ds))
        return s

    def calc_position(self, s):
        """输入 s，返回 x, y"""
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        return x, y

    def calc_curvature(self, s):
        """输入 s，返回曲率 k"""
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        # k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
        den = (dx ** 2 + dy ** 2)**1.5
        if den < 1e-9:
             return 0.0
        return (ddy * dx - ddx * dy) / den

    def calc_yaw(self, s):
        """输入 s，返回航向 yaw"""
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        return math.atan2(dy, dx)

    def find_projection(self, x, y):
        """
        寻找外部点 (x,y) 在样条曲线上的投影点 s (Frenet 坐标中的 s)
        粗略搜索 + 梯度下降精修
        """
        # 1. 粗略搜索：找到最近的离散点
        min_dist = float('inf')
        min_idx = 0
        
        # 采样密度不需要太高，因为后面会精修
        ds_step = 1.0 
        s_range = np.arange(self.s[0], self.s[-1], ds_step)
        if len(s_range) == 0: return 0.0

        for i, s_val in enumerate(s_range):
            px, py = self.calc_position(s_val)
            dist = (x - px)**2 + (y - py)**2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        # 2. 初始猜测 s
        s_guess = s_range[min_idx]
        
        # 3. 简单的局部搜索精修
        # 在 s_guess 周围 +/- 1.0 范围内搜索
        s_search = np.linspace(max(self.s[0], s_guess - 1.0), min(self.s[-1], s_guess + 1.0), 20)
        best_s = s_guess
        min_d = min_dist
        
        for s_val in s_search:
            px, py = self.calc_position(s_val)
            dist = (x - px)**2 + (y - py)**2
            if dist < min_d:
                min_d = dist
                best_s = s_val
                
        return best_s


class QuinticPolynomial:
    """
    五次多项式
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    用于生成 Jerk 最小化的轨迹 (横向/纵向)
    """
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # xs, vxs, axs: 起始位置、速度、加速度
        # xe, vxe, axe: 终止位置、速度、加速度
        # T: 时间长度
        
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        except np.linalg.LinAlgError:
            self.a3 = 0
            self.a4 = 0
            self.a5 = 0

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2