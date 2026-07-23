# 示例6：Dobot CR5AF + PI0.5 + PPO Pose 异步训练

本文介绍 Dobot CR5AF 的双节点真机训练流程：

```text
cloud（rank 0，GPU）：actor 训练 + 人工奖励服务
robot（rank 1，GPU）：rollout 推理 + env worker + Dobot
```

默认使用 pose 控制和人工稀疏奖励，对应配置：

```text
examples/embodiment/config/dobot_async_ppo_pi05.yaml
```

两台机器需要使用可互访的 IP。若不在同一内网，先参考
`docs/wireguard_build.md` 或 `docs/ssh_reverse_tunnel_build.md`。

> 真机运行前必须清空工作空间、确认急停可用，并由操作人员全程监护。

## 1. 环境准备

两台机器都运行 OpenPI，因此都需要 GPU 环境、相同版本的代码和 PyTorch。

### 1.1 Cloud 节点

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

### 1.2 Robot 节点

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

### 1.3 检查两端版本

分别执行：

```bash
git rev-parse HEAD
git submodule status third_party/dobot_zhiyu

python - <<'PY'
import numpy
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("numpy:", numpy.__version__)
PY
```

两端的 Git commit、子模块 commit、Torch 和 CUDA 构建应一致。

## 2. 模型准备

### 2.1 PI0.5 pose checkpoint

在两个节点都准备：

```text
checkpoints/pi05_dobot_pose/
├── model.safetensors
├── config.json
└── <dataset_assets>/
    └── norm_stats.json
```

检查文件：

```bash
test -f checkpoints/pi05_dobot_pose/model.safetensors
find checkpoints/pi05_dobot_pose -name norm_stats.json -print
```

该 checkpoint 必须对应 8 维 pose state/action：

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

不要使用 joint checkpoint 或 7 维 joint 数据的归一化统计量。

如需从 cloud 同步到 robot：

```bash
rsync -avP checkpoints/pi05_dobot_pose/ \
  zylab@192.168.3.224:/home/zylab/project/RLinf/checkpoints/pi05_dobot_pose/
```

### 2.2 检查模型配置

`examples/embodiment/config/dobot_async_ppo_pi05.yaml` 默认应包含：

```yaml
actor:
  model:
    model_path: "checkpoints/pi05_dobot_pose"
    action_dim: 8
    openpi:
      config_name: "pi05_dobot_pose"
```

## 3. Robot 节点硬件配置

### 3.1 检查设备

```bash
ls -l /dev/ttyACM*
nc -vz 192.168.5.1 29999
nc -vz 192.168.5.1 30004
```

若夹爪串口无权限：

```bash
sudo usermod -aG dialout "$USER"
```

执行后重新登录。

USB 相机走 V4L2，使用稳定的 by-id 路径（跨重启不变）：

```bash
ls -l /dev/v4l/by-id/
```

应能看到形如 `usb-RYS_CAMERA071101_...-video-index0` 的符号链接（指向
`/dev/video6`）。注意 `index1`（`/dev/video7`）是同一物理相机的第二个接口，
**不要**把它配成第二台相机。检查访问权限：

```bash
ls -l /dev/video6
id -nG
```

运行用户必须通过 `video` 组（或等效 ACL）拥有访问权限。若不在该组：

```bash
sudo usermod -aG video "$USER"
```

执行后重新登录。不要使用 `chmod 777`。

相机采集 **1920×1080 MJPG**，env 会自动 center-crop 成 1080×1080 正方形再
resize 到 224×224 作为观测（`min(h, w)` 机制）。必须用 `camera_fourcc: "MJPG"`
压缩格式：默认的未压缩 YUYV 在 1080p 下受 USB 带宽限制仅约 5 FPS，且会被后端
的帧率校验拒绝；MJPG 可达请求帧率。只验证相机（不连机械臂）：

```bash
python rlinf/envs/realworld/dobot/verify_env.py --camera-only
```

预期看到 `帧 shape=(1080, 1920, 3)`。

### 3.2 验证 Dobot

只检查 import，不连接硬件：

```bash
source .venv/bin/activate
python rlinf/envs/realworld/dobot/verify_env.py --check-import
```

连接、使能并读取反馈，但不发送运动命令：

```bash
python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.1 \
  --speed 5 \
  --skip-gripper \
  --skip-motion
```

记录输出中的 6 维关节弧度值，作为安全初始位姿。

在确认工作空间和急停后，可由现场人员执行小幅 Servo 验证：

```bash
python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.1 \
  --speed 5 \
  --skip-gripper \
  --frames 5 \
  --step-s 0.5
```

### 3.3 填写配置

编辑：

```text
examples/embodiment/config/dobot_async_ppo_pi05.yaml
```

在 `cluster.node_groups` 的 robot hardware 配置中填写：

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

只配置一台相机（`index0`，对应 `/dev/video6`）。`index1`（`/dev/video7`）
是同一物理相机的第二个接口，不要写入。`camera_fourcc: "MJPG"` 是 1080p 采集
所必需的（见 3.1 节说明）。

再编辑：

```text
examples/embodiment/config/env/realworld_dobot.yaml
```

填写：

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

> 相机映射说明：本部署只有一台 USB 相机。`enable_high_camera: false` 时，
> `DobotEnv` 把唯一的相机命名为 `cam_left_wrist`（策略输入槽）。这是必需的，
> 因为 checkpoint 用 `observation.images.cam_left_wrist` 训练。`main_image_key`
> 也保持为 `cam_left_wrist`。

`initial_joint_pos` 前 6 维是弧度，最后一维是夹爪归一化位置。不要保留示例中的全零占位值。

## 4. 人工稀疏奖励

每个 episode 结束时，cloud 节点把最终 `cam_high` 图像显示在网页上，由人工点击：

- `Success`：奖励 `1.0`
- `Failure`：奖励 `0.0`

奖励只写入 episode 最后一步，之前所有步骤奖励均为 `0`。

### 4.1 启动评分服务

在 cloud 节点单独打开一个终端：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

python examples/embodiment/human_reward_server.py \
  --port 12345 \
  --log_dir logs/human_rewards
```

浏览器访问：

```text
http://<cloud-ip>:12345
```

如果浏览器无法直接访问 cloud：

```bash
ssh -L 12345:localhost:12345 <cloud-user>@<cloud-ip>
```

然后访问 `http://localhost:12345`。

### 4.2 默认配置

`dobot_async_ppo_pi05.yaml` 的 train/eval 默认使用：

```yaml
override_cfg:
  use_dense_reward: False
  use_reward_model: True
  reward_mode: terminal
  reward_image_key: "cam_left_wrist"
  reward_worker_node_rank: 0
  reward_worker_node_group: "cloud"
  reward_worker_cfg:
    use_reward_model: True
    model:
      model_type: "human"
      human_reward_url: "http://127.0.0.1:12345"
      timeout: 600.0
      server_wait_timeout: 60.0
      default_reward: 0.0
      task_description: "pick up the plug and plug it into the socket"
```

这里的 `use_reward_model` 是 RLinf 奖励 worker 的接口名称；
`model_type: human` 表示等待人工评分。

`reward_image_key` 必须指向实际存在的帧。本部署只有一台相机，帧名为
`cam_left_wrist`，因此 reward 也取 `cam_left_wrist`。若仍写 `cam_high`，
episode 结束时会因找不到帧而报 missing-frame-key 错误。

### 4.3 测试评分服务

先生成测试图片：

```bash
python - <<'PY'
from PIL import Image

Image.new("RGB", (224, 224), color=(80, 120, 160)).save(
    "/tmp/human_reward_test.jpg"
)
PY
```

提交测试：

```bash
python examples/embodiment/test_human_reward_server.py \
  --image /tmp/human_reward_test.jpg \
  --url http://127.0.0.1:12345 \
  --task "Dobot human reward test"
```

在网页点击 Success 或 Failure 后，终端应打印收到的奖励。

## 5. 启动 Ray 集群

假设：

```text
cloud IP：192.168.3.223
robot IP：192.168.3.224
通信网卡：enp130s0
```

使用下面命令确认实际 IP 和网卡：

```bash
ip addr
ip route | grep default
```

### 5.1 Cloud 节点

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

### 5.2 Robot 节点

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

### 5.3 验证集群

在 cloud 节点执行：

```bash
ray status
python tests/unit_tests/diag_cloud_to_local.py
```

应看到两个节点均为 alive，且 cloud 到 robot 的测试为 SUCCESS。

## 6. Dummy 验证

### 6.1 Pose 观测和 terminal reward 单元测试

任一节点执行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_dobot_reward_and_dummy.py
```

该测试验证：

- pose dummy 观测包含 `prev_state`
- 人工奖励只在 episode 最后一步计算一次
- 非法 `reward_mode` 会被拒绝

### 6.2 双节点训练编排

先按第 5 节启动两个 Ray 节点。然后在 cloud 节点执行：

```bash
python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05 \
  env.train.override_cfg.is_dummy=True \
  env.train.override_cfg.use_reward_model=False \
  env.eval.override_cfg.is_dummy=True \
  env.eval.override_cfg.use_reward_model=False
```

Dummy 模式不会连接或移动 Dobot，也不需要启动人工评分服务。通过标准：

- actor 在 cloud 节点启动
- rollout 和 dummy env 在 robot 节点启动
- PI0.5 pose checkpoint 成功加载
- 完成至少一次 rollout/训练交互

完成后按 `Ctrl+C` 停止。

## 7. 真机训练

开始前确认：

1. 两个 Ray 节点均 alive。
2. 人工评分服务正在 cloud 节点运行。
3. 操作者已打开评分页面。
4. RobotMode、相机、夹爪和初始位姿检查通过。
5. `is_dummy: False`、`action_mode: cartesian`、`state_mode: pose`。

在 cloud 节点启动：

```bash
python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05
```

每个 episode 结束后，在网页根据最终图像点击 Success 或 Failure。评分记录和最终帧保存在：

```text
logs/human_rewards/
```

训练日志默认写入：

```text
../results/
```

查看 TensorBoard：

```bash
tensorboard --logdir ../results
```

## 8. 常见问题

### Robot 节点没有加入 Ray

检查：

```bash
echo "$RLINF_NODE_RANK"
echo "$RLINF_COMM_NET_DEVICES"
ray status
```

确保 cloud 使用 rank 0，robot 使用 rank 1，并且两端 IP 可以互相访问。

### 评分页面一直没有 episode

检查服务：

```bash
curl http://127.0.0.1:12345/current_episode
```

确认评分服务运行在 cloud 节点，且配置中的
`human_reward_url` 为 `http://127.0.0.1:12345`。

### 人工评分图像不正确

确认 `reward_image_key: cam_left_wrist`。本部署只有一台 USB 相机，帧名为
`cam_left_wrist`。检查 `camera_serials` 里的 by-id 路径是否正确指向该相机。

### 相机打开失败 / device busy

```bash
fuser /dev/video6
```

若有进程占用，先确认归属再释放；不要 kill 未知进程。确认 `camera_type` 为
`opencv`、`camera_resolution` / `camera_fps` 与相机实际支持的模式一致
（本配置 `640x480@30`）。纯相机验证（不连接机械臂）：

```bash
cd /home/zylab/project/RLinf
PYTHONPATH=. .venv/bin/python - <<'PY'
from rlinf.envs.realworld.common.camera import CameraInfo, create_camera

info = CameraInfo(
    name="cam_left_wrist",
    serial_number="/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0",
    camera_type="opencv",
    resolution=(640, 480),
    fps=30,
)
camera = create_camera(info)
try:
    camera.open()
    frame = camera.get_frame(timeout=5)
    print("shape:", frame.shape)
    print("dtype:", frame.dtype)
finally:
    camera.close()
PY
```

预期输出 `shape: (480, 640, 3)`、`dtype: uint8`。该命令不会初始化机械臂。

### 停止集群

在两个节点分别执行：

```bash
ray stop
```
