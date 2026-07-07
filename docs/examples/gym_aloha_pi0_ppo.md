# 示例2: GYM_ALOHA + PI0 + PPO

本示例介绍如何在 Gym Aloha 机器人操作环境上使用 OpenPI 策略和 PPO 算法进行训练与评估。

## 1. 环境安装

```bash
bash requirements/install_local.sh embodied --model openpi --env gym_aloha
```

## 2. 模型准备

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

## 3. 训练

```bash
bash examples/embodiment/run_embodiment.sh gym_aloha_ppo_openpi_pi0 ALOHA
```

训练配置位于 `examples/embodiment/config/gym_aloha_ppo_openpi_pi0.yaml`。

## 4. 评估

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
