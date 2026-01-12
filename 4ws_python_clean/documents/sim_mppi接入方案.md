# 在 sim.py 接入 MPPI 算法与成本函数设计方案

本文面向 `/Users/gjc/src/4ws/src/sim.py` 的后端仿真引擎，结合项目内 MPPI 相关代码（`/Users/gjc/src/4ws/scripts/mppi/`、`/Users/gjc/src/4ws/src/pytorch_mppi/`、`/Users/gjc/src/4ws/tests/tests/`）给出可落地的接入方案与成本设计。目标是：在 `SimEngine` 的自动控制链路中新增 `mppi` 模式，基于 4WS（前后轮转角）与纵向速度命令，完成道路跟踪与障碍规避。

## 总览
- 控制器：使用 `src/pytorch_mppi/mppi.py` 的 `MPPI`（或平滑版 `SMPPI`/核插值版 `KMPPI`）。
- 动力学：以当前仿真状态为真值，MPPI 内部滚动采用近似可微的几何+简化动力学混合（与 `sim.py` 的低速融合思想一致）。
- 成本：分为阶段成本（tracking/约束/控制代价/障碍）与终端成本（到达目标）。
- 接入点：`SimEngine._step()` 中 autop 分支，新增 `elif self.autop_mode == 'mppi': self._autop_update_mppi()`；并扩展 `set_autop_mode('mppi')`。
 - 实现位置：控制器封装位于 `src/mppi_iface.py`，`sim.py` 中延迟初始化并调用。
 - 动力学实现：Torch 版 2DOF 模块位于 `src/twodof_torch.py`，供 MPPI 并行滚动复用。

## 状态与动作选择
为保证与现有 4WS 仿真契合，建议采用下列状态/动作维度：
- 状态向量 `s`（建议 6 维）：`[x, y, psi, beta, r, U]`。
  - 与 `sim.py` 保持一致：`x/y/psi` 来自 2DOF 姿态；`beta/r` 为 2DOF 状态；`U` 来自 `params.U`（或 `ctrl.U`）。
- 动作向量 `u`（建议 3 维）：`[delta_f, delta_r, dU]`。
  - `delta_f`/`delta_r` 为前后轮角（受 `delta_max` 限制）。
  - `dU` 为每步速度增量（受 `dU_max * dt` 限制），与现有 `U_cruise/U_max` 逻辑兼容。

若初期希望更快落地，可采用 2 维动作 `[delta_f, a]`（前轮角+加速度），并用 `strategy.ideal_yaw_rate` 推导后轮角（等效同相/反相策略），但最终推荐直接采样 `[delta_f, delta_r, dU]` 以充分发挥 4WS 优势。

## 动力学（MPPI 内部滚动）
MPPI 要求一个步进函数 `f(s, u) -> s_next`，建议采取“几何-动力学混合”的轻量近似，与 `sim.py` 3DOF 低速融合的思想一致（保持数值稳定与采样效率）。

伪代码（Torch 版，与项目函数保持符号一致）：

```python
import torch

def curvature_4ws_torch(df, dr, L):
    # 与 src/dof_utils.py 中 curvature_4ws 一致的近似（小角近似）：
    return (torch.tan(df) + torch.tan(dr)) / L

def body_to_world_2dof_torch(U, beta, psi):
    x_dot = U * torch.cos(psi + beta)
    y_dot = U * torch.sin(psi + beta)
    return x_dot, y_dot

def dynamics_step_torch(state, action, params, dt):
    # state: [x, y, psi, beta, r, U]
    # action: [df, dr, dU]
    x, y, psi, beta, r, U = state.unbind(-1)
    df, dr, dU = action.unbind(-1)

    # 速度命令 & 低速融合目标横摆率
    U_next = torch.clamp(U + dU, 0.0, torch.tensor(params.U_max))
    kappa = curvature_4ws_torch(df, dr, torch.tensor(params.L))
    r_des = U_next * kappa
    r_dot_kin = (r_des - r) / torch.tensor(params.tau_low)
    beta_dot_kin = -beta / torch.tensor(params.tau_beta)

    # 姿态与位置（几何部分）
    x_dot, y_dot = body_to_world_2dof_torch(U_next, beta, psi)
    psi_dot = r

    # 欧拉步进（可替换为半隐式/中点法以提升稳定性）
    x_next   = x   + dt * x_dot
    y_next   = y   + dt * y_dot
    psi_next = psi + dt * psi_dot
    beta_next= beta+ dt * beta_dot_kin
    r_next   = r   + dt * r_dot_kin

    return torch.stack([x_next, y_next, psi_next, beta_next, r_next, U_next], dim=-1)
```

说明：
- 上述近似不涉及轮胎非线性与牵引分配，因此采样开销低、数值稳定，适合 MPPI 大量滚动。
- 真值积分仍由 `sim.py` 的 2DOF/3DOF 实现，不冲突；MPPI 仅用于求最优动作。

## 成本函数设计
将成本拆分为阶段成本 `l_t` 与终端成本 `φ_T`：

阶段成本（每步）：
```text
l_t = w_lat * e_y^2
    + w_head * e_psi^2
    + w_kappa * (kappa - kappa_ref)^2
    + w_speed * (U - U_des)^2
    + w_yaw * r^2 + w_beta * beta^2
    + w_u * ||u||^2 + w_du * ||u_t - u_{t-1}||^2
    + w_ay * ReLU(ay - ay_limit)^2
    + w_obs * obstacle_cost
```
- 误差项：
  - `e_y/e_psi` 使用 `sim.py._plan_ref_geometry(...)` 最近段锚点计算（代码已有），与当前 `MPC` 保持一致。
  - `kappa_ref` 来自参考段的 `psi` 变化率与弧长（`sim.py` 已有计算）。
- 速度项：
  - `U_des` 可设为 `U_cruise` 或随 `kappa_ref` 调整（急弯减速），保持与 `ay_limit_coeff` 一致。
- 动作项：
  - `||u||^2` 控制幅值；`||u_t - u_{t-1}||^2` 抑制抖动（SMPPI/KMPPI 已内建平滑也可减少该项权重）。
- 约束项：
  - `ay = U^2 * |kappa|` 近似横向加速度；超过 `ay_limit_coeff * mu*g` 用 ReLU 二次惩罚。
- 障碍项：
  - 参考 `scripts/mppi/mppi_obstacle_avoider.py`，对每个障碍采用矩形安全距离的软约束：
    ```text
    obstacle_cost = sum_i softplus(alpha * (d_safe - d_i))
    ```
    其中 `d_i` 为最近点到障碍包络的距离，`d_safe` 由 `vehicle_width/length` 与 `safety_distance` 决定。

终端成本（T 步）：
```text
φ_T = W_lat * e_y(T)^2 + W_head * e_psi(T)^2 + W_pos * ||p(T) - p_goal||^2
```
- `p_goal` 为计划终点（`SimEngine.goal_pose_end`），亦可在分段任务中换成下一航点。
- 终端权重 `W_*` 通常大于阶段权重，促使到达目标。

权重初值建议（与现有 MPC 直觉一致）：
- `w_lat = 60 ~ 150`，`w_head = 300 ~ 800`
- `w_kappa = 8 ~ 25`，`w_speed = 2 ~ 6`
- `w_yaw = 0.05 ~ 0.3`，`w_beta = 5 ~ 30`
- `w_u = 0.5 ~ 1.5`，`w_du = 0.2 ~ 0.6`
- `w_ay = 40 ~ 120`，`w_obs = 200 ~ 800`
- 终端权重 `W_lat/W_head/W_pos` 可设为阶段的 2~4 倍。

## MPPI 初始化与调用
示例（以 `MPPI` 为例，Torch CPU/GPU 均可）：

```python
import torch
from src.pytorch_mppi.mppi import MPPI

class MPPIController4WS:
    def __init__(self, params, dt, plan_provider, device=None):
        self.params = params
        self.dt = float(dt)
        self.plan_provider = plan_provider  # 提供参考路径与最近段误差计算
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 噪声协方差：前后轮角与速度增量
        noise_sigma = torch.diag(torch.tensor([
            0.08,  # delta_f
            0.08,  # delta_r
            0.5    # dU
        ], dtype=torch.float32))

        self.mppi = MPPI(
            dynamics=lambda s, u: dynamics_step_torch(s, u, self.params, torch.tensor(self.dt)),
            running_cost=lambda s, u: self.running_cost(s, u),
            noise_sigma=noise_sigma,
            num_samples=400,
            horizon=20,
            lambda_=150.0,
            u_min=torch.tensor([-self.params.delta_max, -self.params.delta_max, -self.params.dU_max*self.dt], dtype=torch.float32),
            u_max=torch.tensor([ self.params.delta_max,  self.params.delta_max,  self.params.dU_max*self.dt], dtype=torch.float32),
            device=self.device,
        )

    def running_cost(self, s, u):
        # s: [x, y, psi, beta, r, U]
        # 取参考几何、误差与参考曲率/速度
        # 这里可将 sim.py 的误差计算逻辑移植为 Torch 版本；
        # 初期实现可以用近似（例如将最近段误差通过 numpy 计算后喂给 Torch 张量）。
        # 返回 shape [batch] 的成本值。
        # 伪代码略（见上文成本文式），实际实现将各项计算合并为张量表达式。
        pass

    def command(self, s_np):
        # s_np: numpy array shape [6]
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device)
        u = self.mppi.command(s)
        return u.detach().cpu().numpy()
```

说明：
- 初期可以把参考误差计算（`_plan_ref_geometry` 与最近段锚点法）保留在 numpy，结果转成 Torch 张量参与成本计算；后续再完全迁移为 Torch。
- 若希望抑制抖动，直接替换为 `SMPPI` 并将成本中的 `w_du` 降低；或使用 `KMPPI` 在控制点上采样以获得更平滑的轨迹。

## 接入 sim.py 的改动
1) 接入模式开关：
```python
# sim.py
def set_autop_mode(self, mode: str):
    with self._lock:
        m = str(mode or '').lower()
        if m in ('simple', 'mpc', 'mppi'):
            self.autop_mode = m
        else:
            pass
```

2) 在 `_step()` 的 autop 分支中增加 mppi：
```python
# sim.py
if self.autop_enabled and len(self.plan) > 0:
    if self.replan_every_step:
        try:
            self._replan_to_goal()
        except Exception:
            pass
    if self.autop_mode == 'mpc' and self.mode == '2dof':
        self._autop_update_mpc()
    elif self.autop_mode == 'mppi' and self.mode == '2dof':
        self._autop_update_mppi()
    elif self.mode == '2dof':
        self._autop_update_simple()
```

3) 新增 `_autop_update_mppi()`（示例骨架）：
```python
# sim.py
def _autop_update_mppi(self):
    if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
        return

    # 延迟初始化控制器（缓存到 self._mppi_ctrl）
    if not hasattr(self, '_mppi_ctrl') or self._mppi_ctrl is None:
        from .pytorch_mppi.mppi import MPPI  # 可放到模块顶层
        self._mppi_ctrl = MPPIController4WS(self.params, self.dt, lambda: self.plan)

    # 当前状态（与上文 s 对齐）
    s_np = np.array([
        float(self.state2.x),
        float(self.state2.y),
        float(self.state2.psi),
        float(self.state2.beta),
        float(self.state2.r),
        float(self.ctrl.U),
    ], dtype=float)

    # 求最优动作
    u_np = self._mppi_ctrl.command(s_np)
    df_cmd, dr_cmd, dU_cmd = float(u_np[0]), float(u_np[1]), float(u_np[2])

    # 应用控制（限幅与速度边界）
    df_cmd = float(np.clip(df_cmd, -self.delta_max, self.delta_max))
    dr_cmd = float(np.clip(dr_cmd, -self.delta_max, self.delta_max))
    U_next = float(np.clip(self.ctrl.U + dU_cmd, 0.0, self.U_max))

    self.ctrl.delta_f = df_cmd
    self.ctrl.delta_r = dr_cmd
    self.ctrl.U = U_next
```

注意：
- 若使用 `[delta_f, a]` 两维动作，请将 `dU_cmd` 替换为 `a_cmd * dt`，并用 `ideal_yaw_rate` 或简单比例生成 `delta_r`。
- 3DOF 模式暂不在 autop 集成中调用 MPPI（保持现有 3DOF 真值积分与可视化），但可在后续拓展。

## 参数与约束建议
- MPPI 参数：
  - `num_samples = 300 ~ 800`（CPU 取小，GPU 可取大）。
  - `horizon = 15 ~ 25`，`dt` 与 `sim.dt` 对齐（如 `0.02`）。
  - `lambda_ = 100 ~ 300`（温度；越大越平滑，越小越激进）。
  - `noise_sigma = diag([0.06~0.12, 0.06~0.12, 0.3~0.7])`。
- 动作边界：
  - `delta_f/delta_r ∈ [-30°, 30°]`（与 `sim.py.delta_max` 一致）。
  - `dU ∈ [-dU_max*dt, dU_max*dt]`（`dU_max=1.5 m/s^2` 见 `sim.py`）。
- 横向加速度限制：
  - `ay_limit = ay_limit_coeff * mu * g`（`ay_limit_coeff` 已在 `sim.py`）。
- 速度策略：
  - 急弯时降低 `U_des`（如 `U_des = clip(U_cruise, 0, sqrt(ay_limit/|kappa_ref|))`）。

## 障碍物集成要点
- 参考 `scripts/mppi/mppi_obstacle_avoider.py`，障碍物以矩形包络与安全距离建模；在 `SimEngine` 中可新增 `self.obstacles` 并暴露更新接口。
- 成本中对每个障碍计算最近距离，并采用 softplus/平方惩罚以避免梯度消失。

## 与项目现有示例的关系
- `tests/tests/smooth_mppi.py` 展示了 `MPPI/SMPPI/KMPPI` 的用法与成本示例（LQRCost/HillCost），可作为 API 参考。
- `scripts/mppi/main.py` 与 `mppi_obstacle_avoider.py` 展示了经典 2D 车模与障碍规避策略（CasADi 版），本方案做了 Torch 化与 4WS 对齐。
- `src/pytorch_mppi/mppi.py` 为核心采样优化器；上文 `MPPIController4WS` 与之直接对接。

## 运行与调试
- 启动仿真时使用：
  - `python3 your_app_entry.py` 或项目已有运行脚本（保持使用 `python3`）。
- 切换自动控制模式：
  - `engine.set_autop(True)`；`engine.set_autop_mode('mppi')`。
- 可视化/诊断：
  - 观察 `track` 与 `ctrl.delta_f/delta_r/U` 变化；横摆率/侧偏等变量在 UI 中已有显示。
- 性能建议：
  - CPU 下适当降低 `num_samples/horizon`；GPU 可提升。

## 实施步骤清单（落地顺序）
1. 在 `sim.py` 扩展 `set_autop_mode` 支持 `'mppi'`。
2. 在 `sim.py._step()` 的 autop 分支新增 `self._autop_update_mppi()` 调用。
3. 新建 `MPPIController4WS` 封装类（可放 `src/` 或 `scripts/mppi/`），实现 `dynamics_step_torch` 与 `running_cost`。
4. 在 `_autop_update_mppi()` 中完成状态打包与动作应用（限幅、速度更新）。
5. 根据路测结果调整权重与噪声（先跟踪为主，再逐步增强障碍/约束）。

---

如需我把以上骨架代码直接落到仓库指定位置，并对 `sim.py` 做最小改动来跑通首个版本，请告知我具体存放路径（例如 `src/controllers/mppi4ws.py`），我可以直接提交补丁以便你 `python3` 运行验证。