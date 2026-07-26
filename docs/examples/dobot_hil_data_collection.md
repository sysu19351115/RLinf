# Dobot 键盘人在环数据采集

本文档介绍如何使用键盘在 PI0.5 策略运行过程中安全接管 Dobot CR5AF 6D 末端位姿和夹爪，并将实际执行动作、模型动作、接管标记、pose `prev_state` 与单路 USB 图像保存为可继续训练的 LeRobot 数据集。

## 架构

Dobot 已使用 8 维绝对 Cartesian action：

```
[x, y, z, qw, qx, qy, qz, gripper]
```

- 平移：基坐标系 X/Y/Z
- 旋转：工具局部坐标系 roll/pitch/yaw
- 夹爪：归一化 `[0, 1]`

键盘 wrapper 直接维护 8 维目标位姿，不使用 IK。人工动作与策略动作完全同构，导出的 `action` 可直接用于 OpenPI pose 数据管线。

### MODEL/ENGAGE 状态机

- **MODEL**：策略输出 action chunk，逐帧执行。
- **ENGAGE**：操作者用键盘控制目标位姿，策略 action chunk 被冻结（不消费队列）。
- 切换控制权时清空旧 action chunk、重置 Servo 平滑器。
- ENGAGE→MODEL 后先执行一帧当前 pose hold，再用新观测重新推理。

## 键盘约定

| 按键 | 行为 |
|------|------|
| `w` / `s` | 前方 / 后方（基坐标 ±X，经 `base_frame_euler_deg` 旋转） |
| `a` / `d` | 左方 / 右方（基坐标 ±Y） |
| `q` / `e` | 上方 / 下方（基坐标 ±Z） |
| `i` / `k` | 工具坐标 roll ± |
| `l` / `j` | 工具坐标 pitch ± |
| `o` / `u` | 工具坐标 yaw ± |
| `,` / `.` | 夹爪闭 / 开 |
| `h` | MODEL ↔ ENGAGE |
| `m` | 返回 MODEL |
| `Enter` | 成功结束并保存 episode |
| `Backspace` | 丢弃当前 episode 并复位 |
| `ESC` | 丢弃当前 episode、回到初始位姿并退出 |

### 基坐标系旋转适配

机械臂非标准正装时，操作者的物理方向直觉（前后左右上下）与基坐标系方向不一致。通过 `base_frame_euler_deg` 参数设置欧拉角（xyz 内旋顺序，单位度，`[rx, ry, rz]`），wrapper 自动把平移增量从物理坐标系转到基坐标系：

- 标准正装：`base_frame_euler_deg: [0, 0, 0]`（默认，无需设置）
- 侧装使物理"前方"对齐基坐标系 +Y：`base_frame_euler_deg: [0, 0, 90]`
- 倒挂安装（物理"上方"对齐基坐标系 -Z）：`base_frame_euler_deg: [180, 0, 0]`

旋转键（`i/k/j/l/u/o`）使用工具坐标系，不受此参数影响。

推荐增量：`position_delta=0.002` (2mm)、`rotation_delta=0.02` (约1.15°)、`gripper_delta=0.05`。

## 前置准备

### 键盘权限

```bash
ls -l /dev/input/by-id/
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/<keyboard-event-kbd>
```

生产环境使用 `input` 组或 udev rule，不要长期使用 `chmod a+r`。

### USB 相机

确认相机设备路径未被其他进程占用：

```bash
fuser /dev/video6
```

当前相机：`/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0`，1920x1080 @ 30 FPS，MJPG。

### 工作空间配置（可选）

软件工作空间 clamp 是**可选的**。Dobot 控制器的 SDK 已内置硬件安全墙（`max_jump_m` 跳变护栏 + slew 限速器），会拒绝越限命令。

软件 workspace clamp 是额外的绝对边界层，防止持续按键导致缓慢漂移（每帧增量很小，能通过硬件的相对限幅检查）。如需启用，在配置中同时设置 `workspace_low` 和 `workspace_high`：

```yaml
keyboard_intervention:
  workspace_low: [0.2, -0.2, 0.1]
  workspace_high: [0.6, 0.2, 0.4]
```

不设置则完全依赖硬件安全墙 + 操作员急停。

## 三种启动模式

### 1. Dummy 模式（无硬件）

测试软件链路，不连接机械臂：

```bash
python examples/embodiment/collect_dobot_hil_data.py \
    --config-name dobot_hil_collect \
    policy_mode=dummy \
    env.eval.override_cfg.is_dummy=true
```

### 2. Hold 模式（真机安全门）

连接真机但不加载模型，始终建议当前 pose：

```bash
python examples/embodiment/collect_dobot_hil_data.py \
    --config-name dobot_hil_collect \
    policy_mode=hold \
    env.eval.override_cfg.is_dummy=false \
    env.eval.keyboard_intervention.start_in_engage=true
```

**安全门流程：**
1. 启动后不按方向键观察 5 秒，机械臂应保持当前 pose
2. 分轴小步测试 `w/s/a/d/q/e`，核对基坐标方向
3. 分轴旋转测试 `i/k/j/l/u/o`，确认工具局部轴
4. 夹爪测试 `,/.`，验证 [0,1] 限制
5. 切换测试：`m` 后先 hold，`h` 重新从当前反馈初始化
6. `Backspace` 安全回初始位姿，`ESC` 释放设备

### 3. Model 模式（PI0.5 + HIL）

加载 PI0.5 checkpoint，完整的 HIL 数据采集：

```bash
python examples/embodiment/collect_dobot_hil_data.py \
    --config-name dobot_hil_collect \
    policy_mode=model \
    env.eval.override_cfg.is_dummy=false \
    env.eval.keyboard_intervention.start_in_engage=false
```

启动前检查 checkpoint 和 norm_stats：

```bash
test -f checkpoints/pi05_dobot_t265_pose_train_800_torch/model.safetensors
test -f checkpoints/pi05_dobot_t265_pose_train_800_torch/assets/dobot_cf5af_t265_pose/norm_stats.json
```

## 数据保存语义

- **Enter**：`reward=1.0 + terminated=True + success_once=True`，episode 写入当前会话数据集。
- **Backspace**：`hil_event=abort`，collector 调用 `reset()` 丢弃 buffer，不增加 episode 计数。
- **ESC**：`quit_program=True`，collector 先 `reset()` 回初始位姿再退出。

每次启动采集器都会创建一个独立的时间戳目录，例如：

```text
logs/dobot_hil_collect/collected_data/
└── 20260724_203015/
    └── rank_0/
        └── id_0/
            ├── data/
            ├── meta/
            └── videos/
```

同一秒内重复启动会依次使用 `_01`、`_02` 后缀。写入器不会删除已有数据集；如果目标 shard 已存在，会拒绝启动并报告 `FileExistsError`。

主字段在保存时直接使用 OpenPI Dobot pose 格式：

| 语义 | 数据集字段 |
|------|------------|
| 当前 pose 状态 | `observation.state` |
| 真实执行动作 | `action` |
| 上一 pose 状态 | `observation.prev_state` |
| 单路 USB 图像 | `observation.images.cam_left_wrist` |

额外保留 `model_action`、`model_action_valid`、`intervene_flag`、`segment_id`、`is_success` 和 `done`。`model_action_valid` 仅 MODEL 推理帧为 True，ENGAGE 帧和 ENGAGE→MODEL 过渡 hold 帧为 False。当前未采集真实六维力传感器数据，因此不会伪造 `observation.wrench`。

当 `env.eval.override_cfg.gripper_relative_threshold` 配置为 `(0, 1)` 内的值（HIL 示例默认 `0.2`）时，环境在最终下发前使用带状态保持的相对阈值控制夹爪：

- 模型或人工目标比真实夹爪反馈高出阈值时，全开（`1.0`）；
- 真实夹爪反馈比目标高出阈值时，全闭（`0.0`）；
- 差值未超过阈值时，保持上一开闭状态；
- 每次 reset 清除保持状态，并由 reset 后的真实夹爪反馈初始化。

数据集中的 `action` 和人工帧的 `intervene_action` 保存最终下发动作，因此夹爪维度为二值 `0.0/1.0`；`model_action` 保留模型原始连续输出用于诊断；观测 `observation.state` 中的夹爪位置始终是真实连续反馈。

## 数据集验收

采集后用验收脚本检查：

```bash
python examples/embodiment/inspect_dobot_hil_dataset.py \
    logs/dobot_hil_collect/collected_data/<SESSION>/rank_0/id_0
```

Dummy 模式允许常量图像：

```bash
python examples/embodiment/inspect_dobot_hil_dataset.py \
    logs/dobot_hil_collect/collected_data/<SESSION>/rank_0/id_0 \
    --allow-dummy-images
```

检查内容包括：
- Schema：`observation.state` / `observation.prev_state` / `action` 各 8 维 float32
- 数值安全：quaternion norm、gripper [0,1]、XYZ 在 workspace 内
- prev_state 连续性
- 每条 episode 最后一帧 `done=True`
- 至少一帧 `intervene_flag=True`

## 数据恢复

恢复中断会话时必须显式指定已有会话目录，不能自动选择“最新”目录：

```bash
python examples/embodiment/collect_dobot_hil_data.py \
    --config-name dobot_hil_collect \
    policy_mode=model \
    env.eval.data_collection.resume=true \
    env.eval.data_collection.create_session_dir=false \
    env.eval.data_collection.save_dir=/absolute/path/to/20260724_203015
```

新 episode 会写入该会话下的新 `id_N` shard，不覆盖已 finalize 数据。

## 常见故障

| 故障 | 排查 |
|------|------|
| 键盘无响应 | 检查 `RLINF_KEYBOARD_DEVICE`，确认设备在 `input` 组 |
| 颜色错误 | 检查相机 fourcc 和分辨率 |
| pose 跳变 | 确认 quaternion 同半球归一化；检查 ENGAGE 初始化是否从真实 TCP 读取 |
| workspace clamp | 检查 `workspace_low/high` 配置是否覆盖实际工作范围 |
| action queue 未清 | 确认 HIL 状态切换时 collector 清空了 `_action_queue` |
| `prev_state` 缺失 | 确认 `required_observation_fields=("prev_states",)` 已传入 CollectEpisode |
| dataset metadata 未 finalize | 异常退出时可能丢失最后几条 episode；降低 `finalize_interval` |

## 急停与安全

- 真机操作时负责人必须握住急停，工作空间清空
- 任何非有限值、越界 workspace、异常 quaternion 都不会发送给机械臂
- `DobotEnv.step()` 会在 NaN/Inf 时硬拒绝（不发送动作）
- 异常路径始终通过 `try/finally` 关闭 env 和硬件

## 相关文档

- [Dobot PI0.5 PPO 训练](dobot_pi05_ppo.md)
- [SO101 HIL 数据采集](so101_hil_data_collection.md)（参考实现）
