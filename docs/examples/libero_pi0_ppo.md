# 示例1: LIBERO + PI0 + PPO

本示例介绍如何在 LIBERO 机器人操作基准上使用 OpenPI 策略和 PPO 算法进行训练。

## 1. 环境安装

```bash
bash requirements/install_local.sh embodied --model openpi --env maniskill_libero
```

## 2. 模型准备

训练前需要下载模型，放在 `checkpoints/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` 中：

```bash
source .venv/bin/activate
hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
  --local-dir RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
```

## 3. 训练

启动训练，相关参数位于 `examples/embodiment/configs/libero_spatial_ppo_openpi_quickstart.yaml`：

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_quickstart
```
