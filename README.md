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

## 示例3: ReBot+PI0.5+PPO异步训练

多机真机训练：云端 GPU 服务器（actor 训练 + rollout 推理）+ 本地 ReBot Arm 真机（env worker）。
两端通过 WireGuard 组成 Layer 3 网络（云端 `10.200.200.2`，本地 `10.200.200.3`），Ray 集群直接互通。

> 如果网络环境不支持 WireGuard 直连（如单向 NAT），可以使用 SSH 双向隧道方案替代：
> 详见 `docs/ssh_reverse_tunnel.md`与`docs/wireguard_build.md`。WireGuard 方案更简单，推荐优先采用。

### 1. 环境准备

分别在云端和本地两套环境中执行。两端 PyTorch 版本必须一致（`pyproject.toml` 已锁定 `torch==2.7.0`），确保 Gloo 跨节点通信兼容。

#### 1.1 云端服务器（GPU 训练 + 推理，H100）

```bash
bash requirements/install_local.sh --force embodied --model openpi --env rebot
uv pip install -e . #装RLinf
```

`--force` 跳过缓存检查，确保拿到最新的 `torch 2.7.0+cu128`。

#### 1.2 本地真机节点（两种选择）

**选择 A：使用本地 GPU（RTX 5090/5070ti）**

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia # 先用sudo权限安装系统依赖
bash requirements/install_local.sh embodied --model openpi --env rebot --force --no-root # 再安装环境
uv pip install -e . #装RLinf
```

脚本自动检测 RTX 5090（Blackwell），安装 `torch 2.7.0+cu128`，与云端完全一致。验证torch可正常使用gpu：

```bash
# 在云端执行
python tests/unit_tests/pytorch_test.py
```

**选择 B：纯 CPU（不依赖 GPU，仅 env worker）**

```bash
bash requirements/install_local.sh --cpu-only --env rebot --force --no-root 
uv pip install -e . #装RLinf
```

CPU-only 模式只安装 env worker 所需的最小依赖（gymnasium、opencv 等），加上 rebot 机械臂 SDK 的 Python 依赖。

reBotArm 控制 SDK 已内置在仓库中（`rlinf/envs/realworld/rebot/reBotArm_control_py/`）。

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

云端和本地需要相同的 RLinf 代码版本。可在云端启动训练前执行：

```bash
export RLINF_CODE_WORKING_DIR=auto
```

或手动 `git pull` 保持两端代码一致。优先手动同步，因为自动同步依赖于网络，将主节点的代码强制传输到本地节点。

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

如果没有可相互访问随意端口的双向通信ip地址，请先参考 ‘docs/wireguard_build.md‘ 搭建虚拟网卡，但虚拟网卡用UDP协议，极度依赖良好的网络条件。

若有本地局域网ip，可查看节点的ip与网卡后运行下述命令。

```bash
# 云端（head）
source .venv/bin/activate
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0 # 通过ip addr 查看该机器绑定ip所在的网口
ray start --head --port=6389 --node-ip-address=192.168.3.223 \
  --disable-usage-stats

# 本地（worker）— 放宽心跳容忍，防止 WireGuard 延迟触发 GCS 误判节点 dead
source .venv/bin/activate
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0 # 通过ip addr 查看该机器绑定ip所在的网口
# export RAY_health_check_initial_delay_ms=30000 #（optional）
# export RAY_health_check_period_ms=10000 #（optional）
# export RAY_num_heartbeats_timeout=300 #（optional）
ray start --address='192.168.3.223:6389' \
  --node-ip-address=192.168.3.224 \
  --disable-usage-stats
```

验证集群和跨节点通信：

```bash
# 在云端执行
python tests/unit_tests/diag_cloud_to_local.py
```

测试通过的条件是输出 `Test 1: cloud driver -> LOCAL node ... SUCCESS`，且两个节点alive=True

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

## 示例4: ReBot+PI0.5+PPO单节点真机训练

本示例适用于**训练机器与真机在同一台电脑上**的场景（例如本地 RTX 5090 + ReBot Arm B601），无需搭建 WireGuard/SSH 隧道，直接在本机启动 Ray 并完成训练。

核心思路：为同一个 `node_rank: 0` 注册两个 node group：
- `local_gpu`：自动检测 GPU，运行 actor/rollout
- `rebot`：声明 `RebotArm` 硬件，运行 env

对应配置：`examples/embodiment/config/rebot_single_node_ppo_pi05.yaml`

### 1. 环境安装

在本地单机上安装完整环境（包含 openpi 与 rebot）：

```bash
sudo bash requirements/install_local.sh embodied --model openpi --env rebot --force
uv pip install -e .   # 安装 RLinf 本体
```

脚本会自动检测 RTX 5090（Blackwell）并安装 `torch 2.7.0+cu128`。

### 2. 拉起 CAN 总线

```bash
sudo modprobe peak_usb                    # PCAN-USB 适配器
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0                # 验证: state UP, bitrate 1000000
```

### 3. 真机环境验证

```bash
source .venv/bin/activate
python rlinf/envs/realworld/rebot/verify_env.py
```

如未连接相机：

```bash
python rlinf/envs/realworld/rebot/verify_env.py --skip-camera
```

全部 5 步通过后再进入训练阶段。

### 4. 模型准备

将 pi0.5 SFT checkpoint 放到 `checkpoints/pi05_rebot_insertion_pytorch` 目录下：

```
checkpoints/pi05_rebot_insertion_pytorch/
├── model.safetensors          # 模型权重
├── config.json                # 模型配置
└── rebot_lerobot_data/
    └── norm_stats.json        # 归一化统计量
```

### 5. 修改配置

编辑 `examples/embodiment/config/rebot_single_node_ppo_pi05.yaml`，填写：

| 配置项 | 说明 |
|---|---|
| `env.train.override_cfg.target_ee_pose` | 目标末端位姿 `[x, y, z, rx, ry, rz]`（米/弧度） |
| `env.eval.override_cfg.target_ee_pose` | 同上 |
| `cluster.node_groups[1].hardware.configs[0].camera_serials` | Realsense D435 序列号，如 `["12345678"]`；无相机填 `[]` |

> 若本地只有 1 张 GPU，当前默认的 actor/rollout placement（`local_gpu` 的 GPU 0）可直接使用；若有多张 GPU，可在 placement 中把 actor 与 rollout 分到不同卡。

### 6. Dummy 验证

先在无硬件模式下跑通 Ray + pi0.5 链路：

```bash
source .venv/bin/activate
python examples/embodiment/train_async.py --config-name rebot_single_node_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

通过标准：三个 worker group（actor/rollout/env）都能启动，并完成至少 1 个训练 step。

### 7. 真机训练

Dummy 验证通过后，关闭 dummy 并启动真机训练：

```bash
python examples/embodiment/train_async.py --config-name rebot_single_node_ppo_pi05 \
  env.train.override_cfg.target_ee_pose=[x,y,z,rx,ry,rz] \
  env.eval.override_cfg.target_ee_pose=[x,y,z,rx,ry,rz]
```

> 首次上真机前请确认急停按钮可达；建议初期降低 `max_num_steps` 并在旁监护。

### 8. 监控

```bash
tensorboard --logdir results/rebot-pi05-single-node-ppo
```

关注指标：`env/success_once`、`train/loss`、`rollout/...`。

### 9. 评估

评估 RL 训练后的 checkpoint：

```bash
bash evaluations/run_eval.sh realworld realworld_rebot_eval \
  rollout.model.model_path=<path/to/checkpoint> \
  runner.ckpt_path=<path/to/full_weights.pt>
```

