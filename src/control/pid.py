import numpy as np

class CascadePID:
    """
    纵向串级 PID 控制器 (Cascade PID + Feedforward)
    结构：
    1. 前馈：直接利用规划的加速度 a_ref
    2. 外环(速度环)：PI 控制，输入速度误差，输出加速度补偿量
    3. 内环(加速度环)：P 控制，输入加速度误差，输出最终控制指令
    """
    def __init__(self, 
                 kp_v=2.0, ki_v=0.1, kd_v=0.0,  # 速度环参数 (主导)
                 kp_a=0.5,                      # 加速度环参数 (辅助)
                 dt=0.02):
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.kd_v = kd_v
        self.kp_a = kp_a
        self.dt = dt
        
        # 速度环内部状态
        self._integ_v = 0.0
        self._prev_err_v = 0.0
        
        # 积分限幅 (防止由于长时间偏差导致的超调)
        self.integ_limit = 3.0

    def reset(self):
        """重置控制器状态"""
        self._integ_v = 0.0
        self._prev_err_v = 0.0

    def compute(self, target_v: float, current_v: float, target_a: float, current_a: float) -> float:
        """
        计算双环控制输出
        :param target_v: 目标速度 (Lattice 规划)
        :param current_v: 当前实车速度
        :param target_a: 目标加速度 (Lattice 前馈)
        :param current_a: 当前实车加速度 (需估算)
        :return: 最终纵向控制指令 (ax_cmd)
        """
        # --- 1. 外环：速度控制 (Velocity Loop) ---
        err_v = target_v - current_v
        
        # 积分项
        self._integ_v += err_v * self.dt
        self._integ_v = np.clip(self._integ_v, -self.integ_limit, self.integ_limit)
        
        # 微分项 (速度误差的微分其实就是加速度误差的一部分，但在PID公式中仍保留)
        d_err_v = (err_v - self._prev_err_v) / max(1e-6, self.dt)
        self._prev_err_v = err_v
        
        # 速度环输出：期望加速度的【修正量】
        a_fb_v = (self.kp_v * err_v) + (self.ki_v * self._integ_v) + (self.kd_v * d_err_v)
        
        # --- 2. 合成期望加速度 (Desired Acceleration) ---
        # 期望加速度 = 规划前馈 + 速度环修正
        a_des_total = target_a + a_fb_v
        
        # --- 3. 内环：加速度控制 (Acceleration Loop) ---
        # 简单 P 控制，用于修正执行器的响应延迟
        err_a = a_des_total - current_a
        a_fb_a = self.kp_a * err_a
        
        # 最终输出
        # 如果内环 kp_a = 0，则退化为单级 PID + 前馈
        # 如果内环 kp_a > 0，则相当于增强了对加速度误差的抑制
        # 在理想仿真中(a_cmd直接积分)，内环作用有限；但在实车或复杂动力学中非常重要
        final_cmd = a_des_total + a_fb_a
        
        return final_cmd