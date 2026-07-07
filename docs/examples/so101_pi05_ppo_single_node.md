# 示例6: SO101 + PI0.5 + PPO 单节点训练

单台机器同时运行 actor + rollout + SO101 真机 env。

## 环境准备

```bash
sudo bash requirements/embodied/sys_deps.sh nvidia
bash requirements/install_local.sh embodied --model openpi --env so101 --force --no-root
uv pip install -e .
```

## 检查硬件

```bash
ls /dev/ttyACM*
ls -l /dev/v4l/by-path/
```

## 记录初始位姿

```bash
python rlinf/envs/realworld/so101/record_joints.py \
    --pose-kind=initial \
    --left-follower-port=/dev/ttyACM2 \
    --right-follower-port=/dev/ttyACM3
```

将结果填入 `examples/embodiment/config/so101_single_node_ppo_pi05.yaml`。

## Dummy 验证

```bash
python examples/embodiment/train_async.py --config-name so101_single_node_ppo_pi05 \
  env.train.override_cfg.is_dummy=True
```

## 真机训练

```bash
python examples/embodiment/train_async.py --config-name so101_single_node_ppo_pi05
```

首次运行保持 `tele_mode: true`，安全确认后改为 `false`。

## 更多细节

完整文档参见 `docs/source-zh/rst_source/examples/embodied/so101.rst`。
