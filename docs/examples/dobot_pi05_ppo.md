# Dobot CR5AF + PI0.5 + PPO 真机复现

本文只包含当前双节点真机训练的必要步骤。拓扑如下：

```text
cloud  192.168.3.223（RLINF_NODE_RANK=0）：Actor 训练
robot  192.168.3.224（RLINF_NODE_RANK=1）：PI0.5 rollout + Dobot 环境
```

使用配置：

```text
examples/embodiment/config/dobot_async_ppo_pi05_newtorch.yaml
```

> 真机运行必须有人全程监护，保证工作空间无人员和障碍物、硬件急停可用。训练和 reset 只允许 ServoJ/ServoP；不要运行任何 MoveJ/MovJ 归位命令，也不要给 `verify_env.py` 传 `--home-joints`。

## 1. 两节点代码与模型

两节点分别执行：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate
git rev-parse HEAD
git submodule status third_party/dobot_zhiyu
```

两端必须使用相同的 RLinf 提交和 submodule 版本。当前训练使用的模型目录也必须在两端存在：

```bash
test -f /data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/config.json
test -f /data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/model.safetensors
test -f /data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json
```

配置中的下面两个路径必须与上述目录一致：

```yaml
actor:
  model:
    model_path: /data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new
    openpi_data:
      norm_stats_path: /data/checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json
```

## 2. 修改唯一的 Dobot 配置入口

只修改 `dobot_async_ppo_pi05_newtorch.yaml` 顶部的 `dobot:` 段，不要再同时修改公共 `env/realworld_dobot.yaml`：

```yaml
dobot:
  ip: "192.168.5.2"
  speed: 5
  gripper_port: "/dev/ttyACM0"
  camera_serials:
    - "/dev/video0"
  camera_type: "opencv"
  camera_resolution: [640, 480]
  camera_fps: 30
  camera_fourcc: "MJPG"

  action_mode: "cartesian"
  state_mode: "pose"
  gripper_relative_threshold: 0.2

  # [j1..j6 rad, gripper norm]；必须是现场确认过的安全 reset 位姿
  initial_joint_pos: [-5.87, -0.799, -2.175, 1.89, 1.424, 0.621, 0.9]

  max_num_steps: 1000
  max_episode_steps: 1000
  max_steps_per_rollout_epoch: 1000
  total_num_envs: 1
  ignore_terminations: false
```

本项目不绑定相机序列号，直接使用当前枚举出的 `/dev/videoN`。每次插拔或更换
USB 口后先确认编号；当前为 `/dev/video0`，如果编号变化，要同时更新上面的
`dobot.camera_serials`：

```bash
ls -l /dev/video*
```

必须保持以下训练契约：

```text
action_mode=cartesian，state_mode=pose，action_dim=8
num_action_chunks=10，joint_logprob=false
total_num_envs=1，auto_reset=false，terminal_padding.enabled=true
adv_type=gae，group_size=1，reward_label_validity.enabled=true
gripper_relative_threshold=0.2
```

### FSDP 冻结参数契约

`actor.fsdp_config` 必须保持以下取值，不得改回 `use_orig_params=False`：

```yaml
strategy: "fsdp"
sharding_strategy: "no_shard"
use_orig_params: True
ignore_frozen_params: True
```

该组合下 FSDP 只管理可训练参数：`train_expert_only` 冻结的 VLM expert 与
大 embedding 通过 `ignored_states` 排除出 FlatParameter，但它们仍保留在原始
模型、完整 state dict 和首次权重同步中；增量 PatchWeightSyncer 同步只更新
action expert、投影层和 value head。

出现以下两类错误时，先核对日志中的 `[FSDP] Ignoring ... frozen parameter
tensors (... GiB); managing ... trainable elements`，再按根因处理：

```text
Must flatten tensors with uniform requires_grad
embedding writeback shape [257152, 2048]
Expected [2097152] but got [2048, 1024]
```

- `Must flatten tensors with uniform requires_grad`：冻结与可训练参数混入了同一
  FlatParameter，通常是 `use_orig_params=False` 导致。禁止把
  `use_orig_params` 改回 `False`；应确认 `ignore_frozen_params: True` 生效。
- `embedding writeback shape [257152, 2048]`：冻结大 embedding 仍被 FSDP 管理，
  original-parameter writeback 失败。该修复只作用于 FSDP1
  `sharding_strategy=no_shard`，不要只靠切换 `strategy: fsdp2` 规避。
- `Expected [2097152] but got [2048, 1024]`：`[2048, 1024]` 是 action expert
  `llm.layers.*.attn.q_proj.1.weight` 的二维原始权重，`2097152 == 2048 * 1024`
  元素数量一致，属于 FSDP storage/view 错误而非数据维度错误。critic warmup
  阶段 loss 只有 value 项且 actor 项被常数 0 切断计算图时，根 FSDP handle 在
  backward 中没有梯度，参数视图不会恢复，下一次 forward 的 writeback 即崩溃。
  FSDP 包装完成后禁止切换 `requires_grad`；临时 `critic_warmup_steps=0` 只用于
  诊断，不能作为长期修复。

当前一个 episode 为 `1000 / 10 = 100` 个 chunk，`rollout_epoch=2`，所以每轮产生 200 个 chunk；它能被 `global_batch_size=100` 整除。

### 人工确认门（HIL start gate）

`env.train/eval.keyboard_intervention` 已开启 `wait_for_start_on_reset: True`
（`start_key: "y"`，超时 600s），与 HIL-HG-Dagger 一致：每次 episode 结束后，
机械臂不会自动 reset，训练会等待操作员按 `y` 才开始下一轮 rollout。等待期间按
`Esc` 会取消并终止训练；600 秒无按键会超时终止。dummy 模式自动放行，不阻塞。

## 3. Robot 节点运行前检查

在 robot `192.168.3.224` 执行：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

ls -l /dev/ttyACM0
ls -l /dev/video*
ls -l /dev/input/by-id/*-event-kbd /dev/input/event*
test -r /dev/ttyACM0 && test -w /dev/ttyACM0
test -r /dev/video0 && test -w /dev/video0
nc -vz 192.168.5.2 29999
nc -vz 192.168.5.2 30004
df -h / /home
```

键盘奖励监听使用 Linux `evdev`，不依赖桌面终端，所以不需要设置 `DISPLAY`、
`XAUTHORITY` 或 `PYNPUT_BACKEND`。找到物理键盘对应的 event 设备后进行验证：

```bash
# 将 eventX 替换为上一步识别出的物理键盘；也可以填写实际的 by-id 路径
export RLINF_KEYBOARD_DEVICE=/dev/input/eventX
test -r "$RLINF_KEYBOARD_DEVICE"

RLINF_KEYBOARD_DEVICE="$RLINF_KEYBOARD_DEVICE" .venv/bin/python - <<'PY'
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener

listener = KeyboardListener()
print(f"keyboard ready: path={listener.device.path}, name={listener.device.name!r}")
PY
```

如果没有 `/dev/input/by-id/*-event-kbd`，可从 `/dev/input/event*` 中选择实际键盘，
但插拔后编号可能变化。`zylab` 必须属于 `input` 组，并对目标设备有读权限。
如果有多个 keyboard-capable 设备而没有设置 `RLINF_KEYBOARD_DEVICE`，Env worker
会拒绝启动，防止监听到错误的键盘。

相机只读验证，不连接机械臂：

```bash
.venv/bin/python rlinf/envs/realworld/dobot/verify_env.py \
  --camera-only \
  --camera-serial /dev/video0 \
  --camera-resolution 640 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG
```

机械臂只连接并读取反馈，不运动、不控制夹爪：

```bash
.venv/bin/python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.2 \
  --skip-motion \
  --skip-gripper \
  --skip-camera
```

现场再次确认 `initial_joint_pos` 的 reset 路径不会碰撞。不要使用验证脚本的 `--home-joints`。

根分区和结果目录必须有足够空间。当前配置每 5 个 global step 保存一次大模型 checkpoint；空间不足时应先清理磁盘或提高 `runner.save_interval`，不能带着满盘状态启动。

### Checkpoint 路径与保留策略

- `runner.logger.log_path` 固定为 `${project_path:results}/${now:%Y%m%d-%H%M%S}`：
  绝对路径 + 每次启动的时间戳，不同运行写入独立目录，互不覆盖。路径必须在
  配置初始化阶段转换为绝对路径（`validate_cfg`），因为相对路径在 Driver 与
  Ray Actor 进程中会按各自 cwd 解析，导致 checkpoint 写到意外目录。
- 恢复训练时 `runner.resume_dir` 必须是绝对路径，且目标目录必须包含
  `actor/COMPLETED` 标记；缺失标记会拒绝加载，避免读到半成品 checkpoint。
- `runner.checkpoint_keep_last: 3`：每次新 checkpoint 验证成功后只保留最近
  3 个 `global_step_*` 目录，更早的自动删除。
- 每个完整 checkpoint 约 15 GB（DCP + full weights）。真机调试建议
  `save_interval` 保持 5~10；正式长训建议提高到 100 以上并配合保留策略。

## 4. 运行回归门禁

在 cloud 节点执行；这些测试不连接真机：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

.venv/bin/python -m pytest -q \
  tests/unit_tests/test_versions_shape.py \
  tests/unit_tests/test_dobot_async_ppo_pi05_pytorch_config.py \
  tests/unit_tests/test_dobot_hg_dagger_envworker.py \
  tests/unit_tests/test_dobot_reward_and_dummy.py
```

预期结果为全部通过。以下任一错误都不能进入真机训练：

```text
versions reshape RuntimeError
terminal padding / staleness 测试失败
Hydra 配置无法 resolve
相机、夹爪串口或 checkpoint 不存在
磁盘空间不足
```

## 5. 启动双节点 Ray

如果 IP 或网卡已改变，先用 `ip -br -4 addr` 确认。当前两端通信网卡为 `enp130s0`。

Cloud 节点：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0

ray stop
ray start --head \
  --port=6379 \
  --node-ip-address=192.168.3.223 \
  --disable-usage-stats
```

Robot 节点：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0
# 必须替换为第 3 节已经验证通过的同一设备路径
export RLINF_KEYBOARD_DEVICE=/dev/input/eventX
test -r "$RLINF_KEYBOARD_DEVICE"

ray stop
ray start \
  --address=192.168.3.223:6379 \
  --node-ip-address=192.168.3.224 \
  --disable-usage-stats
```

回到 cloud 确认两个节点均为 alive：

```bash
ray status
```

## 6. 启动真机 PPO

确保 robot 上的物理键盘监听可用，操作员手边有硬件急停。在 cloud 节点执行：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

.venv/bin/python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05_newtorch
```

键盘语义：

| 按键 | 含义 | 行为 |
|---|---|---|
| Enter | 成功，reward=1 | 当前 10-action chunk 执行完后结束 episode |
| Backspace | 正常失败，reward=0 | 当前 chunk 执行完后结束 episode |
| Esc | 安全停止，不是失败标签 | 立即 hold/truncate |
| 硬件急停 | 紧急停止 | 任何危险情况优先使用 |

Enter/Backspace 不是急停。当前 chunk 最多继续约 `10 / 30 ≈ 0.33` 秒；之后不应再出现新的机器人运动或 rollout 推理。剩余位置只进行逻辑 padding，随后下一次 bootstrap 使用 ServoJ minimum-jerk reset。

## 7. 首轮必须确认的结果

第一次运行不要无人值守。至少确认：

1. Actor 在 cloud、Rollout 和 Env 在 robot 启动，模型与 norm stats 加载成功。
2. 相机画面、pose state 和 8 维动作没有 shape/NaN 错误。
3. 按 Enter 或 Backspace 后只完成当前 chunk，之后保持静止并进入 reset。
4. 提前终止的数据没有因为 padding version `-1` 被整体丢弃。
5. Actor 完成首个 PPO update，`global_step` 持续增长，没有 `versions.reshape` 错误。
6. policy loss、value loss、grad norm 均为有限值。

重点观察：

```text
rollout/valid_chunks
rollout/padded_chunks
rollout/padding_fraction
rollout/terminal_to_reset_latency_s
rollout/reward_label_valid
train/actor/policy_loss
train/critic/value_loss
train/actor/grad_norm
```

TensorBoard：

```bash
cd /home/zylab/project/RLinf
tensorboard --logdir ../results
```

如果评分后机器人继续执行新的 chunk、出现 MoveJ、loss/grad 为 NaN、Actor 长时间等待 rollout，立即按 Esc/硬件急停并停止任务，保留两节点日志后排查。

## 8. 正常停止

先按 Esc 让机器人安全 hold，再在 cloud 训练终端按 `Ctrl+C`。最后两节点分别执行：

```bash
ray stop
```
