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

如果无法直接判断哪个路径对应哪个设备，可逐个插拔 USB 并观察 `/dev/ttyACM*` 和 `/dev/v4l/by-path/` 的变化，从而确定左右臂串口与三个相机（全局、左腕、右腕）的稳定路径。当前yaml中设备的插入顺序：左机械臂、右机械臂、全局相机、左腕相机、右腕相机

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
  --left-wrist-camera /dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.2:1.0-video-index0 \
  --right-wrist-camera /dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.1:1.0-video-index0 \
  --left-global-camera /dev/v4l/by-path/pci-0000:80:14.0-usb-0:2:1.0-video-index0
```

如果只想验证串口和机械臂连接、跳过相机，可加 `--skip-camera`。

全部通过后，将串口/相机路径填入 `examples/embodiment/config/so101_async_ppo_pi05.yaml` 的 `cluster.node_groups[1].hardware.configs`。

## 2. 模型准备

将 pi0.5 SO101 checkpoint 放到 `checkpoints/pi05_so101_cache_torch/`：

```
checkpoints/pi05_so101_cache_torch/
├── so101_xlerobot_openpi_224_lerobot_data
├── model.safetensors
├── config.json
└── ...
```

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

参考 `docs/examples/rebot_pi05_ppo_async.md` 第 3 节，设置 `RLINF_NODE_RANK` 与 `RLINF_COMM_NET_DEVICES` 后启动 Ray。

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

参考 `docs/examples/rebot_pi05_ppo_async.md` 第 2.2 节训练 vision-based reward model，并在 YAML 中启用 `use_reward_model`。
