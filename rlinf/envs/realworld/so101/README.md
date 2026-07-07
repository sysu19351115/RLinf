# SO101 Bimanual Real-World Environment

SO101 双机械臂在 RLinf `realworld` 环境下的集成。

## 硬件要求

- 2x SO101 follower arms，通过 USB 串口连接（默认 `/dev/ttyACM2`、`/dev/ttyACM3`）。
- 3x USB 相机：左全局相机、左手腕相机、右手腕相机。
- Ubuntu 22.04+，Python 3.10+。

## 安装

```bash
bash requirements/install.sh embodied --model openpi --env so101
uv pip install -e .
```

如果只安装 env worker 依赖（无 GPU 训练节点）：

```bash
bash requirements/install.sh --cpu-only --env so101
uv pip install -e .
```

## 验证

```bash
# 不连接机械臂，只检查 RLinf env 与 dummy 模式
python rlinf/envs/realworld/so101/verify_env.py --skip-hardware --skip-camera

# 连接机械臂与相机完整验证
python rlinf/envs/realworld/so101/verify_env.py
```

## 记录初始/结束位姿

移动双臂到目标位姿后运行：

```bash
python rlinf/envs/realworld/so101/record_joints.py --pose-kind=initial
python rlinf/envs/realworld/so101/record_joints.py --pose-kind=end
```

生成的 `initial_joints.json` / `end_joints.json` 可填入 YAML 的 `initial_joints` / `end_joints`。

## 安全调试

首次真机运行务必设置：

```yaml
override_cfg:
  tele_mode: true
```

此时 env 会读取观测、查询策略，但**不会**下发动作。确认以下事项后再改为 `false`：

- 策略返回 12-dim action。
- 三相机图像都存在且左右手腕相机未 swap。
- 机械臂位于训练初始位姿附近。
- 工作空间无障碍物。

## 单节点训练

```bash
python examples/embodiment/train_async.py --config-name so101_single_node_ppo_pi05
```

## 多节点（云端 GPU + 本地机器人）

参考 `docs/examples/rebot_pi05_ppo_async.md` 拉起 Ray 集群，然后在 head 节点执行：

```bash
python examples/embodiment/train_async.py --config-name so101_async_ppo_pi05
```

## 观测与动作

- `state`: 12-dim motor positions
  - left: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  - right: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
- `frames`: `cam_high`（左全局）、`cam_left_wrist`、`cam_right_wrist`
- `action`: 12-dim 绝对位置目标（模型输出经 output_transform 后的绝对值）
