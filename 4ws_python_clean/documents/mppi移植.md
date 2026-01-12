我将先创建一个示例脚本，演示接入新动力学。

toolName: todo_write

status: success

Todos updated: 2 items

toolName: apply_patch

status: failed

toolName: write_to_file

status: success

filePath: /Users/gjc/src/pytorch_mppi/tests/custom_dynamics_example.py

content:

toolName: todo_write

status: success

Todos updated: 2 items

**操作步骤**

- 定义你的动力学函数：`next_state = dynamics(state, action)`，支持批量输入（`K x nx` 与 `K x nu`）并返回 `K x nx`。
- 定义运行代价：`cost = running_cost(state, action)`，返回每条采样轨迹的代价向量（通常形状 `K`）。
- 选择并初始化控制器：`MPPI`（基础）、`SMPPI`（动作导数平滑）、`KMPPI`（控制点插值平滑）。
- 设置关键参数：`num_samples(K)`, `horizon(T)`, `lambda_`, `noise_sigma`（建议对角矩阵）、`u_min/u_max`（动作边界）。
- 准备环境：需要有 `reset()` 和 `step(action)`；若无 Gym，可用轻量环境类（示例已提供）。
- 运行仿真：循环调用 `command(state)` 获得动作并 `env.step(action)`；或直接用库内 `run_mppi(...)`。

**示例脚本**

- 我已添加示例：`tests/custom_dynamics_example.py`，内容包含：
  - 一个不依赖 Gym 的 `ToyEnv`，状态 `[x,y,vx,vy]`，动作 `[ax, ay]`（双积分器动力学）。
  - 自定义 `dynamics(state, action)` 与 `running_cost(state, action)`，可直接替换为你的模型。
  - `MPPI` 初始化与调用 `run_mppi` 执行 300 步仿真（不渲染）。
- 你可以据此把你的动力学模型替换到 `dynamics`，把你的任务目标替换到 `running_cost`。

**运行命令**

- 进入项目根目录后执行：
  - `python3 tests/custom_dynamics_example.py`
- 如需在 GPU 上运行，确保 `torch.cuda.is_available()` 为真，脚本会自动选择 `cuda`。

**关键参数建议**

- `noise_sigma`: 对角矩阵，数值越大探索越强；例如 `torch.diag(torch.tensor([1., 1.]))`。
- `lambda_`: 温度参数，影响加权与探索范围，通常从 `1e-2 ~ 1` 调。
- `u_min/u_max`: 动作边界，务必与实际控制范围一致；不居中的范围时考虑设定 `noise_mu`。
- `horizon/num_samples`: `num_samples` 增大对效果提升更明显（尤其 GPU）；`horizon` 影响滚动长度与计算量。

**变体与进阶**

- 时间相关动力学：若你的动力学需要步索引 `t`，用 `MPPI(..., step_dependent_dynamics=True)`，并定义 `dynamics(state, action, t)`。
- 随机动力学：模型输出分布时，设 `rollout_samples=M` 增加状态滚动采样数，并可用 `rollout_var_cost`/`rollout_var_discount`惩罚方差。
- 平滑控制：
  - `SMPPI`: 在动作导数空间采样并积分为动作，适合抑制抖动。
  - `KMPPI`: 通过时间核插值控制点（如 `RBFKernel`），设置 `num_support_pts` 控制采样点数量，兼顾平滑与最优性。

**把你的模型接进去**

- 用你的模型替换 `tests/custom_dynamics_example.py` 里的 `dynamics`，保持输入输出批维度。
- 按你的任务定义 `running_cost`，通常由目标偏差、速度项、控制能量等组成。
- 若你已有 Gym 环境，直接传入 `env = gym.make(..., render_mode=\"human\")`，然后用现有 `run_mppi(ctrl, env, retrain_fn, ...)` 运行。
- 如果你要边运行边重训近似动力学，把 `do_nothing` 换成你的训练函数，`run_mppi` 每 `retrain_after_iter` 步会调用它。

需要的话，我可以把你的具体动力学接口（函数签名、状态/动作维度、是否随机）套进示例脚本里，直接给你一版可运行的定制脚本。
