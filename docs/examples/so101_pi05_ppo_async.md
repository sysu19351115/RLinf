# 示例5: SO101 + PI0.5 + PPO 异步训练

多机真机训练：云端 GPU 服务器（actor 训练）+ 本地 SO101 真机（rollout 推理 + env worker）。

两端通过局域网的可互访 ip 组成集群。若网络环境不支持互相访问的 ip，可参考 `docs/wireguard_build.md` 与 `docs/ssh_reverse_tunnel_build.md`。

## 1. 环境准备

分别在云端和本地两套环境中执行。两端 PyTorch 版本必须一致。

### 1.1 云端服务器

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia
bash requirements/install_local.sh --force embodied --model openpi --env so101 --no-root
uv pip install -e .
```

### 1.2 本地真机节点

**选择 A：使用本地 GPU**

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia
bash requirements/install_local.sh embodied --model openpi --env so101 --force --no-root
uv pip install -e .
```

**选择 B：纯 CPU（仅 env worker）**

```bash
bash requirements/install_local.sh --cpu-only --env so101 --force --no-root
uv pip install -e .
```

### 1.3 检查串口与相机

```bash
ls /dev/ttyACM*       # 应能看到左右臂串口
ls -l /dev/v4l/by-path/  # 记录三相机稳定路径
```

如果无法直接判断哪个路径对应哪个设备，可逐个插拔 USB 并观察 `/dev/ttyACM*` 和 `/dev/v4l/by-path/` 的变化，从而确定左右臂串口与三个相机（全局、左腕、右腕）的稳定路径。YAML 默认使用 `/dev/videoN` 设备号，只要保持固定的 USB 插入顺序即可复用。当前建议插入顺序：**全局相机 → 左腕相机 → 右腕相机**（左/右臂串口独立识别，不受此顺序影响）。

### 1.4 真机环境验证

**快速检查（不连接相机/机械臂）：**

```bash
source .venv/bin/activate
python rlinf/envs/realworld/so101/verify_env.py --skip-hardware
```

**完整真机验证（使用你实际的串口和相机路径）：**

```bash
source .venv/bin/activate
python rlinf/envs/realworld/so101/verify_env.py \
  --left-follower-port /dev/ttyACM0 \
  --right-follower-port /dev/ttyACM1 \
  --left-wrist-camera /dev/video2 \
  --right-wrist-camera /dev/video4 \
  --left-global-camera /dev/video0 \
  --test-control
```

如果只想验证串口和机械臂连接、跳过相机，可加 `--skip-camera`。

全部通过后，将串口/相机路径填入 `examples/embodiment/config/so101_async_ppo_pi05.yaml` 的 `cluster.node_groups[1].hardware.configs`。

## 2. 模型准备

### 2.1 pi0.5 SFT checkpoint

将 pi0.5 SO101 checkpoint 放到 `checkpoints/pi05_so101_cache_torch/`：

```
checkpoints/pi05_so101_cache_torch/
├── so101_xlerobot_openpi_224_lerobot_data
├── model.safetensors
├── config.json
└── ...
```

### 2.2 Reward Model（推荐）

SO101 基础环境没有可靠的几何成功判断，真实任务建议训练一个基于视觉的 reward model。整体流程为：**拍摄成功/失败案例图片 → 转换为二分类数据集 → 训练 ResNet 奖励模型 → 在 RL YAML 中启用**。

#### 2.2.1 拍摄数据集

使用 `examples/reward/collect_reward_images.py` 调用全局相机拍摄图片。脚本会自动创建 `success/` 和 `failure/` 两个文件夹：

```bash
cd /home/zylab/project/RLinf
source .venv/bin/activate

python examples/reward/collect_reward_images.py \
    --camera /dev/video0 \
    --output-dir datasets/so101_reward_images \
    --width 640 \
    --height 480 \
    --fps 25 \
    --headless
```

操作说明（预览窗口需处于焦点状态）：
- **p**：拍一张照并保存到当前目标文件夹
- **q**：切换当前目标文件夹（`success` ↔ `failure`）
- **ESC / e**：退出采集

如果是 SSH/无显示器环境，加上 `--headless` 即可在终端中采集（同样按 `p`/`q`，`e` 退出）：


> 提示：
> - 请把双臂和场景摆到任务完成状态后拍入 `success`，摆到明显失败/未完成任务状态后拍入 `failure`。
> - 文件名包含时间戳，重复采集不会互相覆盖。
> - 若相机不是 `/dev/video0`，请替换为实际路径（如 `/dev/v4l/by-path/...`）。

#### 2.2.2 转换数据集

拍摄完成后，使用 `examples/reward/convert_reward_images_to_pt.py` 把图片转成 reward model 训练所需的 `train.pt` 和 `test.pt`：

```bash
python examples/reward/convert_reward_images_to_pt.py \
    --input-dir datasets/so101_reward_images \
    --output-dir logs/so101_reward_data/processed \
    --test-ratio 0.2 \
    --seed 42
```

输出：
- `logs/so101_reward_data/processed/train.pt`
- `logs/so101_reward_data/processed/test.pt`

> 说明：`test.pt` 在 `so101_reward_training.yaml` 中作为验证集（`val_data_paths`）用于 early stopping。如需调整为其他拆分比例，可修改 `--test-ratio`。

#### 2.2.3 训练

推荐使用单卡 standalone 训练脚本，避免 Ray/FSDP 在单 rank 下可能卡死的问题：

```bash
python examples/reward/train_reward_model_simple.py --config-name so101_reward_training
```

训练配置在 `examples/reward/config/so101_reward_training.yaml`。关键参数：
- `data.train_data_paths` / `data.val_data_paths`：指向步骤 2.2.2 的输出
- `actor.model.arch`：默认 `resnet18`
- `actor.model.hidden_dim`：默认 `256`
- `actor.micro_batch_size`：根据 GPU 显存调整
- `actor.optim.lr`：学习率

checkpoint 默认保存路径：
```
logs/so101_reward_model/so101_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt
```

> 如果你有多卡并且想用 FSDP 训练，也可以运行：
> ```bash
> python examples/reward/train_reward_model.py --config-name so101_reward_training
> ```

#### 2.2.4 Dummy 验证

```bash
python examples/reward/verify_reward_model_dummy.py \
    --env-type so101 \
    --checkpoint-path logs/so101_reward_model/so101_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt \
    --data-pt logs/so101_reward_data/processed/test.pt \
    --reward-image-key cam_high
```

#### 2.2.5 配置 RL YAML（已默认启用）

`examples/embodiment/config/so101_async_ppo_pi05.yaml` 中 `env.train.override_cfg` 和 `env.eval.override_cfg` 已默认启用 reward model：

```yaml
env:
  train:
    override_cfg:
      use_reward_model: True
      reward_image_key: "cam_high"
      reward_worker_cfg:
        use_reward_model: True
        model:
          model_type: "resnet"
          model_path: "/home/tyz/project/RLinf/logs/so101_reward_model/so101_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt"
          arch: "resnet18"
          hidden_dim: 256
          dropout: 0.1
          image_size: [3, 224, 224]
          normalize: true
          precision: "fp32"
```

> 注意：
> - 当前配置走 **选择 A**：reward worker 默认跑在 head 节点，因此 `model_path` 必须是 **head 节点上的绝对路径**。
> - 训练完成后，请把 robot 节点上的 checkpoint 同步到 head 节点：
>   ```bash
>   rsync -avP /home/zylab/project/RLinf/logs/so101_reward_model tyz@192.168.3.223:/home/tyz/project/RLinf/logs/
>   ```
> - 如果 head 用户名或项目路径不同，请相应修改 `model_path` 中的 `/home/tyz/project/RLinf`。
> - `hidden_dim` 必须与训练时保持一致。
> - `reward_image_key` 可改为 `cam_left_wrist`、`cam_right_wrist` 等，需与采集时使用的相机一致。若使用全局相机采集，这里应填 `cam_high`。

## 3. 记录初始位姿

将双臂移动到起始位姿：

```bash
python rlinf/envs/realworld/so101/record_joints.py \
    --pose-kind=initial \
    --left-follower-port=/dev/ttyACM0 \
    --right-follower-port=/dev/ttyACM1
```

将生成的 `initial_joints.json` 内容填入 YAML 的 `env.train.override_cfg.initial_joints`。

## 4. 启动 Ray 集群

SO101 异步训练需要两台机器组成一个 Ray 集群：
- **云端 head**（rank 0）：运行 actor 训练，有 GPU。
- **本地 robot 节点**（rank 1）：运行 rollout + env worker，连接机械臂和相机。

两台机器之间需要可互相访问的 IP 地址。如果当前网络环境不满足（例如单向 NAT），可先参考 `docs/wireguard_build.md` 或 `docs/ssh_reverse_tunnel_build.md` 搭建虚拟网卡/隧道。

### 4.1 查看 IP 与网卡

在每台机器上执行：

```bash
ip addr                 # 找到本机用于内网通信的 IP
ip route | grep default # 找到默认路由对应的网口，例如 enp130s0
```

假设：
- 云端 head IP：`192.168.3.223`
- 本地 robot IP：`192.168.3.224`
- 两端通信网口均为 `enp130s0`（按你实际网口名填写）

### 4.2 云端 head 启动 Ray

```bash
source .venv/bin/activate

export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0

ray start --head --port=6379 \
  --node-ip-address=192.168.3.223 \
  --disable-usage-stats
```

### 4.3 本地 robot 节点加入 Ray

```bash
source .venv/bin/activate

export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0

# 如果走 VPN 或网络延迟较大，可适当放宽心跳容忍，避免 GCS 误判节点 dead：
# export RAY_health_check_initial_delay_ms=30000
# export RAY_health_check_period_ms=10000
# export RAY_num_heartbeats_timeout=300

ray start --address='192.168.3.223:6379' \
  --node-ip-address=192.168.3.224 \
  --disable-usage-stats
```

### 4.4 验证跨节点通信

在云端 head 执行：

```bash
python tests/unit_tests/diag_cloud_to_local.py
```

通过的条件是输出包含 `Test 1: cloud driver -> LOCAL node ... SUCCESS`，且两个节点 `alive=True`。

## 5. Dummy 验证

在云端 head 执行：

```bash
python examples/embodiment/train_async.py --config-name so101_async_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

## 6. 真机训练

首次真机运行务必设置 `tele_mode: true`，确认策略输出、相机图像、工作空间安全后再改为 `false`：

```bash
python examples/embodiment/train_async.py --config-name so101_async_ppo_pi05
```

## 7. Reward Model（可选）

详见本文第 2.2 节训练 vision-based reward model，并在 YAML 中启用 `use_reward_model`。
