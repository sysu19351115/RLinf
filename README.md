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

