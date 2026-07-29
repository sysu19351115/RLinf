# 示例6：Dobot CR5AF + PI0.5 + PPO Pose 异步训练

双节点真机训练流程：

```text
cloud（rank 0，GPU）：actor 训练
robot（rank 1，GPU）：rollout 推理 + env worker + Dobot
```

配置文件：`examples/embodiment/config/dobot_async_ppo_pi05_newtorch.yaml`

> 真机运行前必须清空工作空间、确认急停可用，并由操作人员全程监护。

## 1. 环境准备

两台机器都需要 GPU 环境、相同版本的代码和 PyTorch。

**两节点分别执行：**

```bash
cd /home/zylab/project/RLinf
git submodule update --init --recursive

sudo bash requirements/embodied/sys_deps.sh nvidia
bash requirements/install_local.sh embodied \
  --model openpi \
  --env dobot \
  --force \
  --no-root

source .venv/bin/activate
uv pip install -e .
```

确认两端版本一致：

```bash
git rev-parse HEAD
git submodule status third_party/dobot_zhiyu
```

## 2. 模型准备

两节点都需要 PI0.5 pose checkpoint（8 维 `[x,y,z,qw,qx,qy,qz,gripper]`）：

```bash
test -f checkpoints/pi05_dobot_t265_pose_train_800_torch/model.safetensors
test -f checkpoints/pi05_dobot_t265_pose_train_800_torch/assets/dobot_cf5af_t265_pose/norm_stats.json
```

从 cloud 同步到 robot：

```bash
rsync -avP checkpoints/pi05_dobot_t265_pose_train_800_torch/ \
  zylab@<robot-ip>:/home/zylab/project/RLinf/checkpoints/pi05_dobot_t265_pose_train_800_torch/
```

## 3. Robot 节点硬件配置

检查设备连通性：

```bash
ls -l /dev/ttyACM*                          # 夹爪串口
ls -l /dev/v4l/by-id/                        # USB 相机（用 index0，不用 index1）
nc -vz 192.168.5.1 29999 && nc -vz 192.168.5.1 30004  # Dobot 控制器
```

若串口/相机无权限：

```bash
sudo usermod -aG dialout,video "$USER"
# 重新登录生效
```

填写 `examples/embodiment/config/dobot_async_ppo_pi05_newtorch.yaml` 中
`cluster.node_groups` 的 robot hardware 配置：

```yaml
hardware:
  type: Dobot
  configs:
    - ip: "192.168.5.1"
      speed: 5
      action_mode: "cartesian"
      state_mode: "pose"
      gripper_port: "/dev/ttyACM0"
      camera_serials:
        - "/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0"
      camera_type: "opencv"
      camera_resolution: [1920, 1080]
      camera_fps: 30
      camera_fourcc: "MJPG"
      node_rank: 1
```

填写 `examples/embodiment/config/env/realworld_dobot.yaml` 中的 `init_params`：

```yaml
init_params:
  ip: "192.168.5.1"
  gripper_port: "/dev/ttyACM0"
  camera_serials:
    - "/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0"
  camera_type: "opencv"
  camera_resolution: [1920, 1080]
  camera_fps: 30
  camera_fourcc: "MJPG"
  enable_high_camera: false
  task_description: "pick up the plug and plug it into the socket"
  initial_joint_pos: [<j1_rad>, <j2_rad>, <j3_rad>, <j4_rad>, <j5_rad>, <j6_rad>, <gripper_0_to_1>]
```

> `enable_high_camera: false` 时唯一的相机命名为 `cam_left_wrist`（策略输入槽）。
>
> `camera_fourcc: "MJPG"` 是 1080p 采集所必需的（YUYV 在 1080p 下仅约 5 FPS）。
>
> `initial_joint_pos` 前 6 维是弧度，最后一维是夹爪归一化位置，不要保留占位值。

## 4. 键盘人工稀疏奖励

评分键由 robot 节点上的物理键盘监听，不再启动 HTTP 人工奖励服务：

| 按键 | 含义 | 生效时机 |
|---|---|---|
| Enter | 成功，reward=1 | 当前 10-action chunk 完整执行后 |
| Backspace | 失败，reward=0 | 当前 10-action chunk 完整执行后 |
| Esc | 安全停止，不是失败标签 | 立即 |

Enter/Backspace 是评分键，不是急停键。按下后，当前 chunk 最多还会运动约
0.33 秒（30 Hz、10 actions）；其后的 chunk 不再调用机器人，也不再执行
OpenPI rollout 推理。系统只发送固定形状的轻量 padding 消息，待本轮 rollout
协议结束后，在下一次 bootstrap 开始 ServoJ reset。

Esc、键盘断连、监听器异常和控制器拒绝会立即 hold/truncate，并把整条
trajectory 标为不可训练。合法的 Backspace 与 episode timeout 虽然 reward
同为 0，仍是有效训练标签。

配置必须保持：

```yaml
env:
  train:
    auto_reset: false
    ignore_terminations: false
    terminal_padding:
      enabled: true
    override_cfg:
      use_reward_model: false
      reward_mode: none
    use_keyboard_intervention: true
    keyboard_intervention:
      allow_motion_intervention: false
      episode_control_mode: online_chunk_boundary
      safe_model_handoff: false
      done_key: Key.enter
      abort_key: Key.backspace
      quit_keys: [Key.esc]

algorithm:
  adv_type: gae
  group_size: 1
  reward_label_validity:
    enabled: true
```

reset 只使用现有 ServoJ minimum-jerk 路径，禁止 MoveJ。

当前 terminal padding 是显式启用的单环境 Dobot 协议。每个 env worker 的每个
pipeline stage 必须恰好只有一个环境；多环境配置会在启动时直接拒绝，而不会用
`dones.any()` 提前停止同 stage 的其他环境。其他 embodied 算法默认不启用该协议。

`reward_label_validity` 当前只支持 GAE 且 `group_size=1`。这是为了确保无效安全
终止不会进入 GRPO 等算法的分组 reward 均值和标准差；不满足条件的配置会在
worker 初始化时直接报错。

## 5. 启动 Ray 集群

假设 cloud IP 为 `192.168.3.223`、robot IP 为 `192.168.3.224`、网卡为 `enp130s0`（用 `ip addr` 确认实际值）。

**Cloud 节点（rank 0）：**

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

**Robot 节点（rank 1）：**

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0

ray stop
ray start \
  --address='192.168.3.223:6379' \
  --node-ip-address=192.168.3.224 \
  --disable-usage-stats
```

在 cloud 节点验证集群：

```bash
ray status
```

应看到两个节点均为 alive。

## 6. Dummy 验证

按第 5 节启动两个 Ray 节点后在 cloud 节点执行：

```bash
python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05_newtorch \
  env.train.override_cfg.is_dummy=True \
  env.train.use_intervention_in_dummy=True \
  env.eval.override_cfg.is_dummy=True \
  env.eval.use_intervention_in_dummy=True
```

Dummy 模式不连接 Dobot、不需要评分服务。内置 dummy keyboard listener 不会自动
产生 Enter/Backspace；自动化评分测试需要注入 fake listener。通过标准：actor
在 cloud 启动、rollout 和 dummy env 在 robot 启动、checkpoint 成功加载、完成
至少一次训练交互。完成后 `Ctrl+C` 停止。

## 7. 硬件验证

在 robot 节点执行以下验证，确认设备可用后再进入真机训练。详见
`rlinf/envs/realworld/dobot/verify_env.py --help` 了解完整参数。

```bash
source .venv/bin/activate

# 1. 相机（不连接机械臂）
python rlinf/envs/realworld/dobot/verify_env.py --camera-only

# 2. Dobot 连接 + 读取反馈（不发送运动命令）
python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.1 --speed 5 --skip-gripper --skip-motion

# 3. 小幅 Servo 运动（确认工作空间和急停后由现场人员执行）
python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.1 --speed 5 --skip-gripper --frames 5 --step-s 0.5
```

第 2 步会输出 6 维关节弧度值，可作为 `initial_joint_pos` 填入配置。

## 8. 真机训练

确认：两个 Ray 节点 alive、robot 键盘可用、硬件验证通过、`is_dummy: False`、
`action_mode: cartesian`、`state_mode: pose`。

在 cloud 节点启动：

```bash
python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05_newtorch
```

任务成功时按 Enter，任务失败时按 Backspace；紧急情况按 Esc 或硬件急停。评分
生效后机械臂不会执行新的 action chunk。逻辑 padding 不需要等待机器人运动或
模型推理，但固定数量的进程间消息仍需完成，因此 reset 不是评分后立即发生。

查看 TensorBoard：

```bash
tensorboard --logdir ../results
```

停止集群（两节点分别执行）：`ray stop`

## 9. 常见问题

### Robot 节点没有加入 Ray

```bash
echo "$RLINF_NODE_RANK"    # cloud=0, robot=1
echo "$RLINF_COMM_NET_DEVICES"
ray status
```

### 按 Enter/Backspace 后没有立刻 reset

评分会在当前 chunk 边界生效，之后系统以 padding fast path 补齐本 rollout 的
固定通信轮数，再在下一 bootstrap reset。可查看：

```text
rollout/valid_chunks
rollout/padded_chunks
rollout/padding_fraction
rollout/padding_fast_path_count
rollout/reward_label_valid
rollout/terminal_to_reset_latency_s
```

如果评分后仍听到或看到新一段机械臂运动，应立即按 Esc/急停并停止训练；这是
异常行为，不应解释为正常 padding。

### 相机打开失败 / device busy

```bash
fuser /dev/video6
```

确认无其他进程占用，`camera_type: opencv`、`camera_fourcc: MJPG`。

### Pose 观测单元测试

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_dobot_reward_and_dummy.py
```
