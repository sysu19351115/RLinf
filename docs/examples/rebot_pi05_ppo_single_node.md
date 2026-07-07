# 示例4: ReBot + PI0.5 + PPO 单节点真机训练

本示例适用于**训练机器与真机在同一台电脑上**的场景（例如本地 RTX 5090 + ReBot Arm B601），无需搭建 WireGuard/SSH 隧道，直接在本机启动 Ray 并完成训练。

核心思路：为同一个 `node_rank: 0` 注册两个 node group：

- `local_gpu`：自动检测 GPU，运行 actor/rollout
- `rebot`：声明 `RebotArm` 硬件，运行 env

对应配置：`examples/embodiment/config/rebot_single_node_ppo_pi05.yaml`

## 1. 环境安装

在本地单机上安装完整环境（包含 openpi 与 rebot）：

```bash
sudo bash requirements/install_local.sh embodied --model openpi --env rebot --force
uv pip install -e .   # 安装 RLinf 本体
```

脚本会自动检测 RTX 5090（Blackwell）并安装 `torch 2.7.0+cu128`。

## 2. 拉起 CAN 总线

```bash
sudo modprobe peak_usb                    # PCAN-USB 适配器
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0                # 验证: state UP, bitrate 1000000
```

## 3. 真机环境验证

```bash
source .venv/bin/activate
python rlinf/envs/realworld/rebot/verify_env.py
```

如未连接相机：

```bash
python rlinf/envs/realworld/rebot/verify_env.py --skip-camera
```

全部 5 步通过后再进入训练阶段。

### 3.1 查看 RealSense D435 序列号

验证脚本通过后会输出相机信息，例如：

```text
[ OK ] Camera detection (RealSense D435)
       Intel RealSense D435  serial=148522073709
```

将该序列号填入 `examples/embodiment/config/rebot_single_node_ppo_pi05.yaml`：

```yaml
cluster:
  node_groups:
    - label: rebot
      node_ranks: 0
      hardware:
        type: RebotArm
        configs:
          - can_interface: can0
            camera_serials:
              - "148522073709"
            camera_type: "realsense"
            node_rank: 0
```

## 4. 模型准备

### 4.1 pi0.5 SFT checkpoint

将 pi0.5 SFT checkpoint 放到 `checkpoints/pi05_rebot_insertion_pytorch` 目录下：

```
checkpoints/pi05_rebot_insertion_pytorch/
├── model.safetensors          # 模型权重
├── config.json                # 模型配置
└── rebot_lerobot_data/
    └── norm_stats.json        # 归一化统计量
```

### 4.2 Reward Model

本示例默认使用基于视觉的 reward model。如果你已有训练好的 checkpoint，可跳过本小节，直接到步骤 4.2.5 配置路径。

#### 4.2.1 准备 LeRobot 格式数据集

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

#### 4.2.2 预处理

```bash
cd /home/tyz/project/RLinf
source .venv/bin/activate

python examples/reward/preprocess_rebot_lerobot.py \
    --dataset-path datasets/rebot_lerobot_data \
    --output-dir logs/rebot_reward_data/processed \
    --image-key observation.images.cam_left_wrist \
    --success-ratio 0.1 \
    --val-split 0.2 \
    --seed 42
```

输出：
- `logs/rebot_reward_data/processed/train.pt`
- `logs/rebot_reward_data/processed/val.pt`

#### 4.2.3 训练

```bash
python examples/reward/train_reward_model.py --config-name rebot_reward_training
```

训练配置在 `examples/reward/config/rebot_reward_training.yaml`。关键参数：
- `data.train_data_paths` / `data.val_data_paths`：指向步骤 4.2.2 的输出
- `actor.model.arch`：默认 `resnet18`
- `actor.model.hidden_dim`：默认 `256`
- `actor.micro_batch_size` / `actor.global_batch_size`：根据 GPU 显存调整

checkpoint 默认保存路径：
```
logs/rebot_reward_model/rebot_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt
```

#### 4.2.4 Dummy 验证

```bash
python examples/reward/verify_reward_model_dummy.py
```

#### 4.2.5 配置 RL YAML

编辑 `examples/embodiment/config/rebot_single_node_ppo_pi05.yaml`，在 `env.train.override_cfg` 和 `env.eval.override_cfg` 中启用 reward model：

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
> - `model_path` 必须使用**绝对路径**。
> - `hidden_dim` 必须与训练时保持一致。

> 若本地只有 1 张 GPU，当前默认的 actor/rollout placement（`local_gpu` 的 GPU 0）可直接使用；若有多张 GPU，可在 placement 中把 actor 与 rollout 分到不同卡。

## 5. Dummy 验证

先在无硬件模式下跑通 Ray + pi0.5 链路：

```bash
source .venv/bin/activate
python examples/embodiment/train_async.py --config-name rebot_single_node_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

通过标准：三个 worker group（actor/rollout/env）都能启动，并完成至少 1 个训练 step。

## 6. 真机训练

Dummy 验证通过后，关闭 dummy 并启动真机训练：

```bash
python examples/embodiment/train_async.py --config-name rebot_single_node_ppo_pi05
```

> 首次上真机前请确认急停按钮可达；建议初期降低 `max_num_steps` 并在旁监护。

## 7. 监控

```bash
tensorboard --logdir results/rebot-pi05-single-node-ppo
```

关注指标：`env/success_once`、`train/loss`、`rollout/...`。

## 8. 评估

评估 RL 训练后的 checkpoint：

```bash
bash evaluations/run_eval.sh realworld realworld_rebot_eval \
  rollout.model.model_path=<path/to/checkpoint> \
  runner.ckpt_path=<path/to/full_weights.pt>
```
