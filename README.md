# RLinf 复现

## 示例1: LIBERO+PI0+PPO

### 1. 环境安装

```bash
bash requirements/install_local.sh embodied --model openpi --env maniskill_libero
```

### 2. 训练

训练前需要下载模型，放在"checkpoints/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"中，下载方式：

```bash
source .venv/bin/activate
hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
```

启动训练，相关的参数放在"examples/embodiment/configs/libero_spatial_ppo_openpi_quickstart.yaml"
```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_quickstart
```

## 示例2: GYM_ALOHA+PI0+PPO

### 1. 环境安装

```bash
bash requirements/install_local.sh embodied --model openpi --env gym_aloha
```

### 2. 模型准备

从 OpenPI 官方获取 pi0 预训练模型，放到 `checkpoints/pi0_aloha_sim_pytorch` 目录下：

```bash
source .venv/bin/activate

# 从 OpenPI 官方下载 pi0 模型（具体下载命令参考 OpenPI 文档）
# 模型文件需包含 model.safetensors、config.json 以及
# assets/lerobot/aloha_sim_transfer_cube_human/ 目录下的 norm_stats.json
```

创建 norm_stats 符号链接：
```bash
mkdir -p checkpoints/pi0_aloha_sim_pytorch/lerobot
ln -s ../assets/lerobot/aloha_sim_transfer_cube_human \
      checkpoints/pi0_aloha_sim_pytorch/lerobot/aloha_sim_transfer_cube_human
```

最终目录结构：
```
checkpoints/pi0_aloha_sim_pytorch/
├── model.safetensors          # 模型权重
├── config.json                # 模型配置
├── assets/
│   └── lerobot/aloha_sim_transfer_cube_human/
│       └── norm_stats.json    # 归一化统计量
└── lerobot/                   # symlink
    └── aloha_sim_transfer_cube_human -> ../assets/lerobot/aloha_sim_transfer_cube_human
```

### 3. 训练

```bash
bash examples/embodiment/run_embodiment.sh gym_aloha_ppo_openpi_pi0 ALOHA
```

训练配置位于 `examples/embodiment/config/gym_aloha_ppo_openpi_pi0.yaml`。

### 4. 评估

评估预训练模型：
```bash
bash evaluations/run_eval.sh gym_aloha gym_aloha_grpo_openpi_pi0_eval
```

评估 RL 训练后的 checkpoint：
```bash
bash evaluations/run_eval.sh gym_aloha gym_aloha_grpo_openpi_pi0_eval \
  runner.ckpt_path=<path/to/full_weights.pt>
```

例如：
```bash
bash evaluations/run_eval.sh gym_aloha gym_aloha_grpo_openpi_pi0_eval \
  runner.ckpt_path=logs/.../checkpoints/global_step_240/actor/model_state_dict/full_weights.pt
```

## 示例3: ReBot+PI0.5+PPO

多机真机训练：云端 GPU 服务器（actor 训练 + rollout 推理）+ 本地 ReBot Arm 真机（env worker）。

### 1. 环境准备

分别在云端和本地两套环境中执行。

#### 1.1 云端服务器（GPU 训练 + 推理）

```bash
bash requirements/install_local.sh embodied --model openpi
```

云端只需要 RLinf 核心 + openpi 模型 + 训练依赖，不需要真机控制 SDK。

#### 1.2 本地真机节点（CPU-only，env worker）

```bash
bash requirements/install_local.sh --cpu-only --env rebot
```

CPU-only 模式只安装 env worker 所需的最小依赖（gymnasium、opencv、pyrealsense2 等），加上 rebot 机械臂 SDK 的 Python 依赖（motorbridge、pinocchio 等），**不安装 CUDA/torch GPU 版本**。

reBotArm 控制 SDK 已内置在仓库中（`rlinf/envs/realworld/rebot/reBotArm_control_py/`），控制器启动时会自动添加到 `sys.path`。

#### 1.3 拉起 CAN 总线（一次性，本地节点）

```bash
sudo modprobe peak_usb                    # PCAN-USB 适配器
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0                # 验证: state UP, bitrate 1000000
```

#### 1.4 真机环境验证

```bash
source .venv/bin/activate
python rlinf/envs/realworld/rebot/verify_env.py
```

全部 5 步通过后即可进入训练阶段。如未连接相机：

```bash
python rlinf/envs/realworld/rebot/verify_env.py --skip-camera
```

#### 1.5 代码同步

云端和本地需要相同的 RLinf 代码版本。在云端启动训练前执行：

```bash
export RLINF_CODE_WORKING_DIR=auto
```

或手动 `git pull` 保持两端代码一致。

### 2. 模型准备

将 pi0.5 SFT checkpoint 放到 `checkpoints/pi05_rebot_insertion_pytorch` 目录下：

```
checkpoints/pi05_rebot_insertion_pytorch/
├── model.safetensors          # 模型权重
├── config.json                # 模型配置
└── rebot_lerobot_data/
    └── norm_stats.json        # 归一化统计量
```

### 3. 启动 Ray 集群

在**云端**（head，rank 0）先启动：

```bash
source .venv/bin/activate
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<飞连网卡名>

ray start --head --port=12345 --include-dashboard=false \
  --node-ip-address=<cloud_feilian_ip>
```

在**本地**（worker，rank 1）拉起 CAN 总线后启动：

```bash
# 拉起 CAN
sudo ip link set can0 up type can bitrate 1000000 restart-ms 100

source .venv/bin/activate
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=<飞连网卡名>

ray start --address='<cloud_feilian_ip>:12345'
```

验证：`ray status` 应显示 2 个节点。

> 飞连网卡名可通过以下命令获取，常见格式为 `tun0`（Linux）或 `utun0`（macOS）：
> ```bash
> ip addr show | grep -B2 "<cloud_feilian_ip>"   # 找到该 IP 所在网卡名
> ```

### 4. 训练

训练配置位于 `examples/embodiment/config/rebot_async_ppo_pi05.yaml`。

先在**云端 head** 执行 dummy 验证（检查 Ray 通信 + pi0.5 推理是否正常）：

```bash
python examples/embodiment/train_async.py --config-name rebot_async_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

验证通过后，关闭 dummy 开始真机训练：

```bash
python examples/embodiment/train_async.py --config-name rebot_async_ppo_pi05
```

训练开始前需要将配置文件中的占位符替换为实际值：

| 配置项 | 说明 |
|---|---|
| `TARGET_EE_POSE` | 目标末端位姿 `[x, y, z, rx, ry, rz]`（米/弧度） |
| `camera_serials` | Realsense D435 序列号，如 `["12345678"]` |

### 5. 评估

评估 RL 训练后的 checkpoint：

```bash
bash evaluations/run_eval.sh realworld realworld_rebot_eval \
  rollout.model.model_path=<path/to/checkpoint> \
  runner.ckpt_path=<path/to/full_weights.pt>
```

