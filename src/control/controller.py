import math
import numpy as np
# 确保 pid 和 mpc_wrapper 在同一目录下，或者根据你的包结构调整 import
from .pid import CascadePID
from .mpc_wrapper import LateralMPC

class MainController:
    """
    自动驾驶主控制器 (稳定性增强版)
    改进点：
    1. 增加了纵向加速度的低通滤波 (LPF)，消除微分噪声。
    2. 优化了状态重置逻辑。
    """
    def __init__(self, params, dt=0.02):
        self.params = params
        self.dt = dt
        
        # --- 纵向控制配置 ---
        # 建议参数：如果仍然震荡，可以尝试减小 kp_v (例如改到 1.0 - 1.5)
        self.lon_ctl = CascadePID(kp_v=2.0, ki_v=0.1, kp_a=0.5, dt=dt)
        
        # --- 横向控制配置 ---
        self.lat_ctl = LateralMPC(params, dt=dt)
        
        # --- 内部状态 ---
        self._last_match_idx = 0
        
        # 加速度估算相关
        self._last_v = 0.0
        self._current_a_filtered = 0.0 # 存储滤波后的加速度
        self._first_step = True
        
        # [关键参数] 加速度滤波系数 (0.0 ~ 1.0)
        # alpha 越大，平滑效果越强，但延迟越大。推荐 0.6 ~ 0.8
        self.acc_filter_alpha = 0.7 

    def reset(self):
        """重置所有子控制器状态"""
        self.lon_ctl.reset()
        self._last_match_idx = 0
        self._last_v = 0.0
        self._current_a_filtered = 0.0
        self._first_step = True

    def compute(self, state: dict, plan: list, current_ctrl: dict) -> dict:
        """
        计算下一时刻的控制指令
        """
        # 1. 安全检查与异常处理
        if not plan:
            return {
                'ax': -1.0,  # 默认刹车
                'df': 0.0,
                'dr': 0.0,
                'target_v': 0.0
            }

        # 2. 空间速度对齐 (Spatial Alignment)
        match_idx = self._find_nearest_index(state, plan)
        self._last_match_idx = match_idx
        
        ref_point = plan[match_idx]
        
        # 3. 提取纵向目标 (Target States)
        # 优先使用规划中的速度，如果没有则回退到当前指令
        target_v = ref_point.get('v', current_ctrl.get('U', 0.0))
        target_a = ref_point.get('a', 0.0) # 前馈加速度 (Feedforward)
        
        # 4. 获取当前状态并估算加速度 (稳定性核心)
        current_v = state.get('speed', current_ctrl.get('U', 0.0))
        
        if self._first_step:
            current_a_raw = 0.0
            self._current_a_filtered = 0.0
            self._first_step = False
        else:
            # 原始微分加速度 (噪声大)
            dt_safe = max(1e-6, self.dt)
            current_a_raw = (current_v - self._last_v) / dt_safe
            
            # [关键改进] 执行低通滤波
            # Formula: a_filt = alpha * a_filt_prev + (1 - alpha) * a_raw
            self._current_a_filtered = (self.acc_filter_alpha * self._current_a_filtered) + \
                                       ((1.0 - self.acc_filter_alpha) * current_a_raw)
            
        self._last_v = current_v
        
        # 使用平滑后的加速度作为反馈
        current_a_feedback = self._current_a_filtered
        
        # 5. 执行纵向控制 (Cascade PID)
        ax_cmd = self.lon_ctl.compute(
            target_v=target_v, 
            current_v=current_v, 
            target_a=target_a,          # 前馈输入
            current_a=current_a_feedback # 内环反馈 (已滤波)
        )

        # 6. 执行横向控制 (MPC)
        df_cmd, dr_cmd = self.lat_ctl.compute(state, plan, current_ctrl)

        return {
            'ax': ax_cmd,
            'df': df_cmd,
            'dr': dr_cmd,
            'target_v': target_v,
            'match_idx': match_idx
        }

    def _find_nearest_index(self, state, plan):
        """寻找最近点索引 (带搜索窗口优化)"""
        x, y = state['x'], state['y']
        
        if self._last_match_idx == 0:
            search_start = 0
            search_end = len(plan)
        else:
            # 仅在上一帧索引附近搜索，提高效率并防止跳变
            search_start = self._last_match_idx
            search_end = min(len(plan), self._last_match_idx + 50)
            
        best_idx = search_start
        min_dist_sq = float('inf')
        
        for i in range(search_start, search_end):
            dx = plan[i]['x'] - x
            dy = plan[i]['y'] - y
            d2 = dx*dx + dy*dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_idx = i
                
        return best_idx