# Dobot HG-DAgger 在线后训练

本文介绍如何在 RLinf 中使用 Dobot CR5AF、OpenPI PI0.5 和键盘人工接管进行
HG-DAgger 在线后训练。训练数据只来自**完整人工接管且完整执行**的 action
chunk；模型动作、部分接管 chunk、episode 提前结束后的 padding 动作不会进入
expert replay buffer。

## 数据与控制语义

- MODEL：执行模型输出。
- ENGAGE：人工键盘动作替换完整 action chunk。
- `Enter`：成功结束当前 episode。
- `Backspace`：中止当前 episode，清空残留 action queue，再 reset。
- `Esc`：按安全终止语义结束当前 episode，并请求整个训练任务退出。
- reset 默认等待操作员按 `y`，确认工作区安全后才开始下一条 episode。

每个可训练 expert chunk 必须同时满足：

1. chunk 内每个 action 都是人工动作；
2. chunk 内每个 action 都已真实执行，没有 padding；
3. `episode_id` 和连续的 `episode_step_ids` 均有效；
4. action 和模型输入不含 NaN/Inf；
5. Dobot pose 输入包含 `observation/prev_state`。

任一条件不满足时采用 fail-closed：抛出错误并停止入库，不能静默污染训练数据。
Replay 的内存采样与 `.pt` 持久化均保留上述审计字段。

## 配置

- 单节点：`examples/embodiment/config/dobot_hg_dagger_openpi.yaml`
- 双节点：`examples/embodiment/config/dobot_hg_dagger_openpi_2node.yaml`

两份配置默认 `is_dummy: true`，不会连接真机。模型路径、归一化统计和学习率是
必填环境变量：

```bash
export DOBOT_HG_DAGGER_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_train_1200_torch
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_MODEL_PATH/assets/dobot_cf5af_t265_pose/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-5
```

首次运行前检查：

```bash
test -f "$DOBOT_HG_DAGGER_MODEL_PATH/model.safetensors"
test -f "$DOBOT_HG_DAGGER_MODEL_PATH/config.json"
test -f "$DOBOT_HG_DAGGER_NORM_STATS_PATH"
```

## 无硬件检查

先执行测试：

```bash
cd /home/zylab/project/RLinf
.venv/bin/pytest -q \
  tests/unit_tests/test_dobot_hg_dagger_config.py \
  tests/unit_tests/test_dobot_hg_dagger_contract.py \
  tests/unit_tests/test_dobot_hg_dagger_replay.py \
  tests/unit_tests/test_dobot_hg_dagger_runner.py \
  tests/unit_tests/test_dobot_hg_dagger_smoke.py
```

仅验证真实 checkpoint 的配置和 worker 初始化时，必须保持 dummy 环境并禁用键盘：

```bash
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /home/zylab/project/RLinf/examples/embodiment/config \
  --config-name dobot_hg_dagger_openpi \
  runner.max_steps=0 \
  env.train.override_cfg.is_dummy=true \
  env.eval.override_cfg.is_dummy=true \
  env.train.use_keyboard_intervention=false \
  env.eval.use_keyboard_intervention=false
```

## 单节点真机运行

先核对配置中的机械臂 IP、夹爪串口、相机路径、初始关节角和工作空间。键盘设备
变量必须在启动 Ray 前设置：

```bash
export RLINF_NODE_RANK=0
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>
ray stop
ray start --head --port=6379 --node-ip-address=<本机IP>
```

然后显式关闭 dummy：

```bash
cd /home/zylab/project/RLinf
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name dobot_hg_dagger_openpi \
  env.train.override_cfg.is_dummy=false
```

程序在 reset 阶段等待 `y`。操作员应先确认急停可用、机器人处于安全初始位姿且
工作区无人，再按 `y`。

## 双节点运行

双节点拓扑为：

- node 0（GPU 节点）：actor，负责训练和权重更新；
- node 1（robot 节点）：rollout + env，负责模型推理，并连接 Dobot、夹爪、
  相机和键盘。

rollout 会先从基础 checkpoint 创建一份推理模型，再持续接收 actor 同步的最新
权重。因此，当前双节点配置要求**两台机器都能以相同路径读取基础模型和
norm stats**。两台机器也必须使用相同版本的代码和 Python 环境。

Ray worker 只继承 `ray start` 时的环境，不能看到之后在其他终端执行的
`export`。所以必须先设置全部环境变量，再启动 Ray；修改变量后必须在对应节点
执行 `ray stop` 并重新加入集群。

两台机器还必须显式设置各自用于节点间通信的网卡。先分别查询到对端的路由：

```bash
# 在 GPU 节点执行
ip route get 192.168.3.224

# 在 robot 节点执行
ip route get 192.168.3.223
```

取输出中 `dev` 后面的网卡名，分别赋给 `RLINF_COMM_NET_DEVICES`。RLinf 会用它
设置 Gloo/NCCL 的 socket 网卡。不要假设两台机器的网卡名称相同，也不要同时
手动设置不一致的 `GLOO_SOCKET_IFNAME` 或 `NCCL_SOCKET_IFNAME`。

以下示例假设：

- GPU 节点：`192.168.3.223`，`RLINF_NODE_RANK=0`
- robot 节点：`192.168.3.224`，`RLINF_NODE_RANK=1`
- robot 节点通往 GPU 节点的网卡：`enp130s0`
- GPU 节点网卡：运行 `ip route get 192.168.3.224` 后用实际名称替换
  `<GPU_COMM_IFACE>`
- 两台机器上的基础 checkpoint 路径均为
  `/data/checkpoints/pi05_dobot_t265_pose_train_1200_torch`

GPU 节点：

```bash
cd /home/tyz/project/RLinf
source .venv/bin/activate

ray stop

export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<GPU_COMM_IFACE>
# 两台机器没有可用于 PyTorch 通信的 InfiniBand 时保留此项。
export NCCL_IB_DISABLE=1
export DOBOT_HG_DAGGER_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_train_1200_torch
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_MODEL_PATH/assets/dobot_cf5af_t265_pose/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-5

test -f "$DOBOT_HG_DAGGER_MODEL_PATH/model.safetensors"
test -f "$DOBOT_HG_DAGGER_NORM_STATS_PATH"

ray start --head --port=6379 --node-ip-address=192.168.3.223
```

机器人节点：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

ray stop

export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0
export NCCL_IB_DISABLE=1
export DOBOT_HG_DAGGER_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_train_1200_torch
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_MODEL_PATH/assets/dobot_cf5af_t265_pose/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-5
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>

test -f "$DOBOT_HG_DAGGER_MODEL_PATH/model.safetensors"
test -f "$DOBOT_HG_DAGGER_NORM_STATS_PATH"
test -r "$RLINF_KEYBOARD_DEVICE"

ray start --address='192.168.3.223:6379'
```

确认 `ray status` 同时看到两个节点后，在 GPU 节点（node 0）启动：

```bash
cd /home/tyz/project/RLinf
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name dobot_hg_dagger_openpi_2node \
  env.train.override_cfg.is_dummy=false
```

如果两台机器的仓库路径不同，`--config-path` 必须使用**启动命令所在的 GPU
节点**路径。

若日志显示 `DobotController` 出现在 GPU 节点，或 actor 出现在 robot 节点，
说明 `RLINF_NODE_RANK` 设置反了或 Ray 仍在使用旧环境。此时必须停止两边 Ray，
按上述顺序重新设置变量并启动集群。

若初始化权重同步时出现 `Gloo connectFullMesh failed`，先检查两边
`RLINF_COMM_NET_DEVICES` 是否与 `ip route get <对端IP>` 的 `dev` 一致，并确认
这些变量是在 `ray start` 前设置的。若网卡正确仍然失败，再检查两台机器的防火墙
是否阻止了 Gloo 使用的节点间动态 TCP 连接。

## 运行监控与停止

重点监控：

- `env/episode_end/operator_success`
- `env/episode_end/operator_abort`
- `env/episode_end/operator_quit`
- `env/episode_end/keyboard_disconnected`
- `env/episode_end/keyboard_listener_error`
- `env/episode_end/executed_action_fraction`
- `env/episode_end/skipped_action_steps`

键盘断开或监听线程异常会触发安全终止。`Esc` 的 quit 会在当前通信轮次收束后、
下一次 actor 更新前停止非流水线训练。DAGGER 配置不支持训练流水线，因此不会
出现“退出后又执行一次并发更新”的情况。

Replay 数据默认写入：

```text
logs/dobot_hg_dagger/replay_buffer_h50/rank_<actor_rank>/
```

不要把 replay 目录当作未经检查即可发布的数据集；训练前仍应核对 episode 数量、
介入比例、动作分布、终止原因和相机质量。

## 独立自主评估

自主评估必须作为独立进程运行，不能与在线训练同时占用同一台 Dobot。评估保留
`y` 开始门、`Enter` 成功、`Backspace` 失败和 `Esc` 退出，但
`allow_motion_intervention: false` 会硬性禁止 `h` 进入 ENGAGE；即使按住
`w/a/...`，模型动作也不会被替换。

先设置基础模型、归一化统计、待评估权重及人工可读的 checkpoint 标识：

```bash
export DOBOT_HG_DAGGER_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_train_1200_torch
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_MODEL_PATH/assets/dobot_cf5af_t265_pose/norm_stats.json"
export DOBOT_HG_DAGGER_EVAL_CHECKPOINT=/path/to/checkpoint.pt
export DOBOT_HG_DAGGER_EVAL_CHECKPOINT_ID=hgdagger-step-0040
```

保持 dummy 的无硬件配置检查：

```bash
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /home/zylab/project/RLinf/examples/embodiment/config \
  --config-name dobot_hg_dagger_eval \
  env.eval.keyboard_intervention.wait_for_start_on_reset=false
```

真机评估时只显式关闭 eval dummy：

```bash
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /home/zylab/project/RLinf/examples/embodiment/config \
  --config-name dobot_hg_dagger_eval \
  env.eval.override_cfg.is_dummy=false \
  env.eval.rollout_epoch=20
```

输出包含 `autonomous_success`、`episode_duration_s`、`success_once`、
`success_no_intervened` 和 `episode_end/<termination_reason>`；启动日志同时打印
checkpoint id 与路径。评估配置固定为一个物理环境。

## 混合 50 步训练窗口

PI0.5 的训练 horizon 为 50 步（`action_horizon=50`），但环境每轮执行 10 步
（`num_action_chunks=10`）。当前 HG-DAgger 不再只保存 10 步专家 chunk，而是
构建 50 步混合窗口：每个全人工 10 步 chunk 成为一个窗口锚点，后续重新推理且
实际执行的模型 chunk 填充至 50 步。

### 窗口构建规则

- 每个全人工 10 步 chunk 开启一个候选窗口。
- 一个窗口只能包含同一模型版本生成的 chunk；权重版本变化会丢弃未完成的旧
  anchor，避免版本 provenance 与实际 suffix 不一致。
- 后续 chunk（人工或模型）按时间顺序追加到所有兼容的候选窗口。
- 当候选窗口积累 5 个 chunk（50 步）时，生成一个训练样本。
- 模型后缀动作是每次 10 步重新推理的结果，不是干预前预测的残留动作。
- Dobot safety guard 拒绝指令或检测到 NaN/Inf 动作时立即停止当前 chunk；
  rejected step 和剩余 padding 都不会标记为 executed。该 episode 记录
  `controller_rejection` 并要求显式 reset，不会自动 reset 后继续运动。
- `only_save_expert: true` 是兼容既有 rollout 代码的旧配置名，表示关闭经典
  DAgger 的 model-only chunk 专家重标注；它不表示 replay 窗口只有专家动作。
  H50 样本仍由人工前缀和真机实际执行的模型后缀组成。
- 窗口内只要有人工动作就优先使用人工动作。例如 30 步连续人工干预会产生
  三个窗口：
  - `h0-h29 + m30-m49`（obs@s0，60% 人工）
  - `h10-h29 + m30-m59`（obs@s10，40% 人工）
  - `h20-h29 + m30-m69`（obs@s20，20% 人工）

### Loss 范围

默认 `loss_scope: human_only`，仅对 `human_action_mask=True` 的动作步计算
8 维环境动作 MSE；模型 suffix 用于补齐 PI0.5 的 50 步输入目标，但不会产生
监督梯度。实验性 `loss_scope: full_window` 会恢复对全部 50 步计算 loss 的
自蒸馏行为。未知 scope 或缺失 human mask 会立即失败。

Replay 同时保存 step-level `[1,1,50]` 的 `human_action_mask`，并把
`Trajectory.intervene_flags` 按 8 维环境动作展开为 `[1,1,400]`。窗口级
`human_steps/model_steps/human_fraction` 保持 `[1,1]`，三类字段的语义和维度
有意不同。

### 窗口可观测指标

Actor 指标包含累计计数 `dagger/window_emitted`、
`dagger/window_dropped`，当前 gauge `dagger/pending_anchor`，以及
`dagger/window_drop_reason/<reason>`。drop reason 区分终止、未执行动作、
episode 切换、模型版本变化、step 不连续和人工步数不足；控制器拒绝同时记录
`episode_end/controller_rejection`。即使 replay 尚未达到
最小训练大小、该轮跳过 optimizer update，这些指标也会返回，便于发现“训练
进程存活但窗口一直没有落盘”的情况。

### Replay 目录

50 步窗口使用独立 schema（`dobot_hg_dagger_hybrid_h50_v2`）。v2 增加
masked-loss 输入、规范化版本和 action-aligned intervention mask；旧 replay
无法加载到新 buffer。目录为：

```text
logs/dobot_hg_dagger/replay_buffer_h50/rank_<actor_rank>/
```

Hybrid replay 只保存 SFT 所需的 observation、tokenized prompt、action 和
human mask；`chains`、`denoise_inds`、`model_action` 等 rollout-only 字段会
在写入前删除。空目录允许初始化新的 metadata；非空目录缺少
`metadata.json`、schema 不匹配，或 checkpoint schema 不匹配时都会拒绝启动
或加载。

### Preflight 检查

硬件启动前运行真实 checkpoint 的 forward 验证：

```bash
.venv/bin/python tests/integration_tests/test_dobot_hg_dagger_openpi_h50.py
```

脚本从自身位置解析仓库根目录，并从
`DOBOT_HG_DAGGER_MODEL_PATH`、`DOBOT_HG_DAGGER_NORM_STATS_PATH` 读取
checkpoint，不依赖 `/home/zylab` 或 `/home/tyz` 等机器特定路径。

验证 `prepare_dagger_sft_batch` 正确 reshape 为 `[B, 50, 32]`，`sft_forward`
产生有限 masked loss，严格验证全部 PaliGemma 参数冻结且 action expert 至少
存在一个可训练参数。脚本必须完成
backward、optimizer step、冻结参数不变、可训练参数变化和更新后有限 loss 的
全部断言才返回成功；CUDA OOM 会使 gate 失败。完整验证需要 32 GB 以上显存。

## 机器人独占所有权

每个 Dobot controller IP 在机器人主机上对应一个进程锁。锁在 SDK 构造、
连接和使能之前获取。因此，如果训练已经占用同一机械臂，评估会立即报
`DobotOwnershipError`，不会向机械臂发送任何命令。正常 `close()` 会释放锁；
进程崩溃或被终止时，操作系统也会自动释放底层文件锁，不需要人工删除 lock
文件。锁仅限单机，所以双节点部署必须确保所有连接该机械臂的 controller 都在
同一机器人节点上运行。

硬件测试前运行新增的无硬件契约：

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_dobot_hg_dagger_eval_config.py \
  tests/unit_tests/test_dobot_exclusive_ownership.py
```
