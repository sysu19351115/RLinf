# Dobot HG-DAgger 在线后训练

本文只保留 Dobot CR5AF 使用 HG-DAgger 在线训练所需的操作步骤。配置默认使用
dummy 环境；连接真机时必须显式设置 `env.train.override_cfg.is_dummy=false`。

## 1. 操作与安全约定

- `y`：reset 后确认工作区安全并开始 episode。
- `h`：进入人工接管。
- `m`：在当前 action chunk 结束后恢复模型控制。
- `Enter`：成功结束当前 episode。
- `Backspace`：中止当前 episode 并 reset。
- `Esc`：安全结束当前 episode，并退出训练任务。

人工切回模型时，系统会丢弃当前 chunk 中尚未执行的旧模型动作，使用机器人反馈
位姿保持，随后基于最新观测重新推理。新模型动作必须通过运行时安全检查：

```yaml
safe_model_handoff: true
handoff_max_position_jump_m: 0.005
handoff_max_rotation_jump_deg: 2.0
```

超出阈值时不会执行模型目标，episode 会以 `unsafe_model_handoff` 结束并要求
reset。真机运行前必须确认急停可用、机械臂处于安全初始位姿、工作区无人，并核对
机械臂 IP、夹爪串口、相机设备和初始关节角。

## 2. 选择配置

- 单节点：`dobot_hg_dagger_openpi_pytorch`
- 双节点：`dobot_hg_dagger_openpi_pytorch_2node`

模型训练 horizon 为 H50，环境每次执行 H10。默认只对人工接管步骤计算监督
损失。

## 3. 准备模型

Checkpoint 目录必须包含：

```text
model.safetensors
dobot_lerobot_pose_data/norm_stats.json
```

如果现有 checkpoint 是旧格式，先转换：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

OLD_MODEL=/data/checkpoints/pi05_dobot_pose_torch
NEW_MODEL=/data/checkpoints/pi05_dobot_pose_newtorch

python -m rlinf.utils.ckpt_convertor.openpi.convert old2new \
  --input-model "$OLD_MODEL" \
  --input-norm-stats \
    "$OLD_MODEL/assets/dobot_cf5af_t265_pose/norm_stats.json" \
  --output-model "$NEW_MODEL" \
  --output-norm-stats \
    "$NEW_MODEL/dobot_lerobot_pose_data/norm_stats.json"

export DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH="$NEW_MODEL"
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$NEW_MODEL/dobot_lerobot_pose_data/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-6

```

如果 norm stats 位于 checkpoint 内的其他 asset 目录，变量必须直接指向实际
文件。例如当前多物体 checkpoint：

```bash
export DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json"
```

## 4. OpenPI PyTorch 上线前门禁

门禁只加载模型并使用合成观测，不会创建 Dobot controller，也不会向机器人发送
动作。

在 rollout 节点验证 checkpoint 推理和动作变换：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

python tests/hardware_tests/openpi_pytorch_dobot_hg_dagger_gate.py \
  --checkpoint "$DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH" \
  --norm-stats "$DOBOT_HG_DAGGER_NORM_STATS_PATH" \
  --mode rollout \
  --device cuda
```

在 actor 节点验证 backward、梯度、参数冻结和 optimizer step：

```bash
python tests/hardware_tests/openpi_pytorch_dobot_hg_dagger_gate.py \
  --checkpoint "$DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH" \
  --norm-stats "$DOBOT_HG_DAGGER_NORM_STATS_PATH" \
  --mode actor \
  --device cuda
```

两项门禁通过且显存余量满足要求后，再连接真机训练。

## 5. 单节点真机训练

环境变量必须在 `ray start` 前设置：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

export RLINF_NODE_RANK=0
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>
# 设置第 3 节中的模型路径和 DOBOT_HG_DAGGER_LR。

test -r "$RLINF_KEYBOARD_DEVICE"

ray stop
ray start --head --port=6379 --node-ip-address=<本机IP>
```

```bash
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name dobot_hg_dagger_openpi_pytorch \
  env.train.override_cfg.is_dummy=false
```

程序在 reset 阶段等待 `y`。确认现场安全后再开始 episode。

## 6. 双节点真机训练

默认拓扑：

- node 0：actor，负责训练和权重更新；
- node 1：rollout + env，负责推理并连接机器人、夹爪、相机和键盘。

机械臂参数统一在
`dobot_hg_dagger_openpi_pytorch_2node.yaml` 顶层修改：

```yaml
dobot:
  ip: "192.168.5.2"
  tool_index: 2
  initial_joint_pos: [-5.87, -0.799, -2.175, 1.89, 1.424, 0.621, 0.9]
```

`initial_joint_pos` 前六项为弧度制关节角；reset 固定使用关节空间 ServoJ。

两台机器必须使用相同代码版本和兼容的 Python 环境，并能读取内容一致的基础
checkpoint。先在两台机器分别执行 `ip route get <对端IP>`，将输出中的网卡名
设置为 `RLINF_COMM_NET_DEVICES`。

### GPU 节点（node 0）

```bash
cd /home/tyz/project/RLinf
source .venv/bin/activate

export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<GPU_COMM_IFACE>
export DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-6

ray stop
ray start --head --port=6379 --node-ip-address=192.168.3.223
```

### Robot 节点（node 1）

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=<ROBOT_COMM_IFACE>
export NCCL_IB_DISABLE=1
export DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH=/data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new
export DOBOT_HG_DAGGER_NORM_STATS_PATH="$DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json"
export DOBOT_HG_DAGGER_LR=1e-6
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>

test -r "$RLINF_KEYBOARD_DEVICE"

ray stop
ray start --address=192.168.3.223:6379
```

所有环境变量都必须在对应节点执行 `ray start` 前设置；修改后应重启该节点的
Ray。执行 `ray status` 确认两个节点都在线，然后在 GPU 节点启动：

```bash
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name dobot_hg_dagger_openpi_pytorch_2node \
  env.train.override_cfg.is_dummy=false
```

## 7. 运行检查与停止

运行时至少关注：

- `env/episode_end/keyboard_disconnected`
- `env/episode_end/keyboard_listener_error`
- `env/episode_end/unsafe_model_handoff`
- `env/episode_end/controller_rejection`
- `dagger/actor_loss`、`actor/grad_norm`
- replay buffer size 和 actor/rollout weight version

出现键盘断开、controller rejection、权重版本不更新或 loss/梯度非有限值时，应
立即停止训练并排查。按 `Esc` 请求安全退出；不要直接杀死仍在向机器人发送动作的
worker，除非需要紧急停止。

Replay 默认写入：

```text
logs/dobot_hg_dagger/replay_buffer_h50/rank_<actor_rank>/
```

发布或复用 replay 前，应检查 episode、人工介入比例、终止原因、动作分布和相机
质量。

## 8. 常见故障

- checkpoint 找不到：确认环境变量指向模型目录，而不是 `assets/` 目录。
- PyTorch norm stats 找不到：检查
  `<checkpoint>/dobot_lerobot_pose_data/norm_stats.json`。
- Ray worker 看不到环境变量：在对应节点重新 export，然后重启 Ray。
- `Gloo connectFullMesh failed`：检查两边 `RLINF_COMM_NET_DEVICES` 是否与
  `ip route get <对端IP>` 的网卡一致，并检查防火墙。
- actor OOM：保持 `micro_batch_size=1` 和 gradient checkpointing，降低 batch
  或增加梯度累积。
- `observation/prev_state` 缺失或 action expert 无梯度：停止训练，检查模型配置、
  replay 数据和 actor 门禁。
- controller 出现在 GPU 节点：检查 `RLINF_NODE_RANK` 和 Ray 节点环境，停止
  两边 Ray 后按顺序重新启动。

同一机械臂只能由一个训练或评估进程占用。不要同时启动另一个连接相同 Dobot IP
的任务。
