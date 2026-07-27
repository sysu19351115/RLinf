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
  --config-path /home/zylab/project/RLinf/examples/embodiment/config \
  --config-name dobot_hg_dagger_openpi \
  env.train.override_cfg.is_dummy=false
```

程序在 reset 阶段等待 `y`。操作员应先确认急停可用、机器人处于安全初始位姿且
工作区无人，再按 `y`。

## 双节点运行

双节点拓扑为：

- node 0：actor + rollout，负责 GPU 推理、训练和权重同步；
- node 1：env，连接 Dobot、夹爪、相机和键盘。

两台机器必须使用同一版本代码和 Python 环境。在启动 Ray 前分别设置：

GPU 节点：

```bash
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<GPU_NODE_IP>
```

机器人节点：

```bash
export RLINF_NODE_RANK=1
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>
ray start --address='<GPU_NODE_IP>:6379'
```

在 node 0 启动：

```bash
cd /home/zylab/project/RLinf
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /home/zylab/project/RLinf/examples/embodiment/config \
  --config-name dobot_hg_dagger_openpi_2node \
  env.train.override_cfg.is_dummy=false
```

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
logs/dobot_hg_dagger/replay_buffer/rank_<actor_rank>/
```

不要把 replay 目录当作未经检查即可发布的数据集；训练前仍应核对 episode 数量、
介入比例、动作分布、终止原因和相机质量。
