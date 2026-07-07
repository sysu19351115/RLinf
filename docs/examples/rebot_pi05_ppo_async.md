# 示例3: ReBot + PI0.5 + PPO 异步训练

多机真机训练：云端 GPU 服务器（actor 训练）+ 本地 ReBot Arm 真机（rollout 推理 + env worker）。

两端通过局域网的可互访 ip 组成集群（云端 `192.168.3.223`，本地 `192.168.3.224`），Ray 集群直接互通。

> 如果网络环境不支持互相访问的 ip（如单向 NAT），可使用 WireGuard 或 SSH 双向隧道方案替代，但难度极大，且依赖良好的网络条件：详见 `docs/ssh_reverse_tunnel_build.md` 与 `docs/wireguard_build.md`。

## 1. 环境准备

分别在云端和本地两套环境中执行。两端 PyTorch 版本必须一致（`pyproject.toml` 已锁定 `torch==2.7.0`），确保 Gloo 跨节点通信兼容。

### 1.1 云端服务器（5090 节点）

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia # 先用 sudo 权限安装系统依赖
bash requirements/install_local.sh --force embodied --model openpi --env rebot --no-root
uv pip install -e . # 安装 RLinf
```

`--force` 跳过缓存检查，确保拿到最新的 `torch 2.7.0+cu128`。脚本自动检测 RTX 5090（Blackwell），安装 `torch 2.7.0+cu128`。验证 torch 可正常使用 gpu：

```bash
python tests/unit_tests/pytorch_test.py
```

### 1.2 本地真机节点（两种选择）

**选择 A：使用本地 GPU（5070ti 节点）**

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia # 先用 sudo 权限安装系统依赖
bash requirements/install_local.sh embodied --model openpi --env rebot --force --no-root
uv pip install -e . # 安装 RLinf
```

脚本自动检测 RTX 5090（Blackwell），安装 `torch 2.7.0+cu128`，与云端完全一致。验证 torch 可正常使用 gpu：

```bash
python tests/unit_tests/pytorch_test.py
```

**选择 B：纯 CPU（不依赖 GPU，仅 env worker）**

```bash
bash requirements/install_local.sh --cpu-only --env rebot --force --no-root
uv pip install -e . # 安装 RLinf
```

CPU-only 模式只安装 env worker 所需的最小依赖（gymnasium、opencv 等），加上 rebot 机械臂 SDK 的 Python 依赖。但需要修改训练配置，让 rollout 跑在云端节点。

### 1.3 本地节点拉起 CAN 总线

```bash
sudo modprobe peak_usb                    # PCAN-USB 适配器
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0                # 验证: state UP, bitrate 1000000
```

### 1.4 真机环境验证

```bash
source .venv/bin/activate
python rlinf/envs/realworld/rebot/verify_env.py
```

全部 5 步通过后即可进入训练阶段。如未连接相机：

```bash
python rlinf/envs/realworld/rebot/verify_env.py --skip-camera
```

#### 1.4.1 查看 RealSense D435 序列号

验证脚本通过后会输出相机信息，例如：

```text
[ OK ] Camera detection (RealSense D435)
       Intel RealSense D435  serial=148522073709
```

将该序列号填入 `examples/embodiment/config/rebot_async_ppo_pi05.yaml`：

```yaml
node_groups:
  - label: robot
    node_ranks: 1
    hardware:
      type: RebotArm
      configs:
        - can_interface: can0
          camera_serials:
            - "148522073709"
          camera_type: "realsense"
          node_rank: 1
```

### 1.5 代码同步

云端和本地需要相同的 RLinf 代码版本。可手动 `git pull` 保持两端代码一致。也可云端启动训练前执行：

```bash
export RLINF_CODE_WORKING_DIR=auto
```

优先手动同步，因为自动同步依赖于网络，将主节点的代码强制传输到本地节点。

## 2. 模型准备

### 2.1 pi0.5 SFT checkpoint

将 pi0.5 SFT checkpoint 放到 `checkpoints/pi05_rebot_insertion_pytorch` 目录下：

```
checkpoints/pi05_rebot_insertion_pytorch/
├── model.safetensors          # 模型权重
├── config.json                # 模型配置
└── rebot_lerobot_data/
    └── norm_stats.json        # 归一化统计量
```

### 2.2 Reward Model（推荐）

本示例默认使用基于视觉的 reward model。如果你已有训练好的 checkpoint，可跳过本小节，直接到步骤 2.2.5 配置路径。

#### 2.2.1 准备 LeRobot 格式数据集

将人工演示数据整理为 LeRobot 格式，放在 `datasets/rebot_lerobot_data/` 目录下：

```
datasets/rebot_lerobot_data/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    ├── info.json
    ├── tasks.jsonl
    └── ...
```

#### 2.2.2 预处理

```bash
cd /home/tyz/project/RLinf
source .venv/bin/activate

python examples/reward/preprocess_rebot_lerobot.py \
    --dataset-path datasets/rebot_lerobot_data \
    --output-dir logs/rebot_reward_data/processed \
    --image-key observation.images.cam_left_wrist \
    --success-ratio 0.2 \
    --val-split 0.2 \
    --val-balance-ratio 1.0 \
    --seed 42
```

输出：
- `logs/rebot_reward_data/processed/train.pt`
- `logs/rebot_reward_data/processed/val.pt`

#### 2.2.3 训练

```bash
python examples/reward/train_reward_model.py --config-name rebot_reward_training
```

训练配置在 `examples/reward/config/rebot_reward_training.yaml`。关键参数：
- `data.train_data_paths` / `data.val_data_paths`：指向步骤 2.2.2 的输出
- `actor.model.arch`：默认 `resnet18`
- `actor.model.hidden_dim`：默认 `256`
- `actor.micro_batch_size` / `actor.global_batch_size`：根据 GPU 显存调整

checkpoint 默认保存路径：
```
logs/rebot_reward_model/rebot_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt
```

#### 2.2.4 Dummy 验证

```bash
python examples/reward/verify_reward_model_dummy.py
```

#### 2.2.5 配置 RL YAML

编辑 `examples/embodiment/config/rebot_async_ppo_pi05.yaml`，在 `env.train.override_cfg` 和 `env.eval.override_cfg` 中启用 reward model：

```yaml
env:
  train:
    override_cfg:
      is_dummy: False
      use_reward_model: True
      reward_image_key: "wrist_1"
      reward_worker_cfg:
        use_reward_model: True
        model:
          model_type: "resnet"
          model_path: "/absolute/path/to/logs/rebot_reward_model/rebot_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt"
          arch: "resnet18"
          hidden_dim: 256
          dropout: 0.1
          image_size: [3, 224, 224]
          normalize: true
          precision: "fp32"
      max_num_steps: 240
```

> 注意：
> - `model_path` 必须使用**绝对路径**，并确保本地（robot）节点可以访问。
> - `hidden_dim` 必须与训练时保持一致。
> - 同步代码时确保 checkpoint 文件也同步到本地节点。

## 3. 启动 Ray 集群

如果没有可相互访问随意端口的双向通信 ip 地址，请先参考 `docs/wireguard_build.md` 搭建虚拟网卡。但虚拟网卡用 UDP 协议，极度依赖良好的网络条件。

若有本地局域网 ip，可查看节点的 ip 与网卡后运行下述命令。

```bash
# 云端（head）
source .venv/bin/activate
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0 # 通过 ip addr 查看该机器绑定 ip 所在的网口
ray start --head --port=6389 --node-ip-address=192.168.3.223 \
  --disable-usage-stats

# 本地（worker）— 放宽心跳容忍，防止 WireGuard 延迟触发 GCS 误判节点 dead
source .venv/bin/activate
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0 # 通过 ip addr 查看该机器绑定 ip 所在的网口
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

测试通过的条件是输出 `Test 1: cloud driver -> LOCAL node ... SUCCESS`，且两个节点 alive=True。

## 4. Dummy 验证

在启动真机训练前，先在**云端 head** 执行 dummy 验证，检查 Ray 通信 + pi0.5 推理是否正常：

```bash
python examples/embodiment/train_async.py --config-name rebot_async_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

## 5. 真机训练

验证通过后，关闭 dummy 开始真机训练：

```bash
python examples/embodiment/train_async.py --config-name rebot_async_ppo_pi05
```

## 6. 评估

评估 RL 训练后的 checkpoint：

```bash
bash evaluations/run_eval.sh realworld realworld_rebot_eval \
  rollout.model.model_path=<path/to/checkpoint> \
  runner.ckpt_path=<path/to/full_weights.pt>
```
