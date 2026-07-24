# 示例6：Dobot CR5AF + PI0.5 + PPO Pose 异步训练

双节点真机训练流程：

```text
cloud（rank 0，GPU）：actor 训练 + 人工奖励服务
robot（rank 1，GPU）：rollout 推理 + env worker + Dobot
```

配置文件：`examples/embodiment/config/dobot_async_ppo_pi05.yaml`

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

填写 `examples/embodiment/config/dobot_async_ppo_pi05.yaml` 中 `cluster.node_groups` 的 robot hardware 配置：

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

> `enable_high_camera: false` 时唯一的相机命名为 `cam_left_wrist`（策略输入槽）。`reward_image_key` 也必须用 `cam_left_wrist`。
>
> `camera_fourcc: "MJPG"` 是 1080p 采集所必需的（YUYV 在 1080p 下仅约 5 FPS）。
>
> `initial_joint_pos` 前 6 维是弧度，最后一维是夹爪归一化位置，不要保留占位值。

## 4. 人工稀疏奖励

每个 episode 结束时，cloud 节点把最终图像显示在网页上，人工点击 Success（`1.0`）或 Failure（`0.0`）。

在 cloud 节点启动评分服务：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

python examples/embodiment/human_reward_server.py \
  --port 12345 \
  --log_dir logs/human_rewards
```

浏览器访问 `http://<cloud-ip>:12345`（若无法直连，用 `ssh -L 12345:localhost:12345 <cloud-user>@<cloud-ip>` 转发）。

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
  --config-name dobot_async_ppo_pi05 \
  env.train.override_cfg.is_dummy=True \
  env.train.override_cfg.use_reward_model=False \
  env.eval.override_cfg.is_dummy=True \
  env.eval.override_cfg.use_reward_model=False
```

Dummy 模式不连接 Dobot、不需要评分服务。通过标准：actor 在 cloud 启动、rollout 和 dummy env 在 robot 启动、checkpoint 成功加载、完成至少一次训练交互。完成后 `Ctrl+C` 停止。

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

确认：两个 Ray 节点 alive、评分服务运行中、硬件验证通过、`is_dummy: False`、`action_mode: cartesian`、`state_mode: pose`。

在 cloud 节点启动：

```bash
python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05
```

每个 episode 结束后在网页点击 Success 或 Failure。查看 TensorBoard：

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

### 评分页面一直没有 episode

```bash
curl http://127.0.0.1:12345/current_episode
```

确认服务运行在 cloud 节点，配置中 `human_reward_url` 为 `http://127.0.0.1:12345`。

### 人工评分图像不正确

确认 `reward_image_key: cam_left_wrist`（单相机部署的帧名）。

### 相机打开失败 / device busy

```bash
fuser /dev/video6
```

确认无其他进程占用，`camera_type: opencv`、`camera_fourcc: MJPG`。

### Pose 观测单元测试

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_dobot_reward_and_dummy.py
```
