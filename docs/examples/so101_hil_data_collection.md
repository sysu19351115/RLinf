# 示例：SO101 人在环（HIL）数据采集

在真实机器人上运行 pi0.5 checkpoint，并在模型犯错时随时通过**键盘**接管，采集的每条数据同时保存**模型动作**与**人类动作**，导出为 LeRobot 格式。

## 前置条件

- 2 台 SO101 follower 臂（从臂），每臂 6 轴。
- 3 台 USB 相机（左全局、左手腕、右手腕）。
- 运行 RLinf 的 PC（可与机器人同机，也可通过 Ray 多节点分离）。
- 已准备 pi0.5 SO101 checkpoint，目录结构例如：

```text
checkpoints/pi05_so101_cache_torch/
├── model.safetensors
├── config.json
└── ...
```

## 依赖安装

```bash
bash requirements/install.sh embodied --model openpi --env so101
source .venv/bin/activate
uv pip install -e .
```

## 检查硬件串口

```bash
ls /dev/ttyACM*
ls -l /dev/v4l/by-path/
```

确认并记录：

- 左从臂串口（如 `/dev/ttyACM0`）
- 右从臂串口（如 `/dev/ttyACM1`）

## 记录初始位姿

将双臂移动到期望起始位姿后执行：

```bash
python rlinf/envs/realworld/so101/record_joints.py \
    --pose-kind=initial \
    --left-follower-port=/dev/ttyACM0 \
    --right-follower-port=/dev/ttyACM1
```

生成的 `initial_joints.json` 路径可填入 YAML，或直接把 12 维数组写入配置。

## 配置文件说明

主要配置文件：`examples/embodiment/config/so101_hil_collect.yaml`

需要按实际硬件修改的字段：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env:
      node_group: so101
      placement: 0
  node_groups:
    - label: so101
      node_ranks: 0
      hardware:
        type: SO101Arm
        configs:
          - node_rank: 0

runner:
  num_data_episodes: 5

use_dummy_policy: False

env:
  eval:
    # SO101 串口和相机在 init_params 中配置（会覆盖 env/realworld_so101.yaml 的默认值）
    init_params:
      left_follower_port: "/dev/ttyACM0"
      right_follower_port: "/dev/ttyACM1"
      left_wrist_camera:
        index_or_path: "/dev/video2"
        width: 640
        height: 480
        fps: 30
        fourcc: "MJPG"
      right_wrist_camera:
        index_or_path: "/dev/video4"
        width: 640
        height: 480
        fps: 30
        fourcc: "MJPG"
      left_global_camera:
        index_or_path: "/dev/video0"
        width: 640
        height: 480
        fps: 25
        fourcc: "MJPG"

    use_keyboard_intervention: True
    use_intervention_in_dummy: True
    keyboard_intervention:
      active_arm: "left"
      position_delta: 0.005
      rotation_delta: 0.05
      gripper_delta: 5.0
      done_key: "Key.enter"
      quit_keys: ["Key.esc"]
    override_cfg:
      is_dummy: False
      tele_mode: false
      # 必须开启，否则 RealWorldEnv 会用 max_episode_steps 覆盖键盘 wrapper 设置的 truncated，
      # 导致 Enter/ESC 无法结束 episode 并保存数据。
      manual_episode_control_only: True
      initial_joints: [...]          # 来自 record_joints.py
      task_description: "拿起黑色胶带并放入白色盒子"

actor:
  model:
    model_path: "checkpoints/pi05_so101_cache_torch"
```

**注意**：`cluster.node_groups.hardware.configs` 仅用于 Ray 调度时展示硬件信息，**不会覆盖** `env.eval.init_params` 中的串口、相机等实际运行配置。实际连接机械臂和相机时请务必修改 `env.eval.init_params`。

首次真机运行建议保持 `tele_mode: true`，确认图像、动作维度、工作空间安全后再改为 `false`。

## Dummy 模式验证

不连接真实硬件，验证配置、OpenPI 推理、wrapper 与 LeRobot 导出链路：

```bash
cd examples/embodiment
sudo chmod a+r /dev/input/event17
export EMBODIED_PATH="$(pwd)"
export RLINF_KEYBOARD_DEVICE=/dev/input/event17
python collect_so101_hil_data.py --config-name so101_hil_collect use_dummy_policy=True
```

`use_dummy_policy=True` 会同时启用 dummy 策略和不连接硬件，生成零动作并保存 LeRobot 数据集。

**dummy 模式下按 `w`/`a`/`s`/`d` 等位姿键时，真实机械臂不会运动**，但日志会每秒打印一次当前激活臂、按键和目标位姿，例如：

```text
[SO101KeyboardIntervention] active=left key=w target_pos=[0.1508, -0.0089, -0.0037] target_euler=[-178.91, 24.60, 178.27] gripper=2.7
```

看到这类日志就说明键盘控制链路正常。按 `Enter` 会保存当前 episode 并开始下一个；按 `ESC` 会保存当前 episode 并退出。

## 真机数据采集

### 单节点

```bash
cd examples/embodiment
export EMBODIED_PATH="$(pwd)"
python collect_so101_hil_data.py --config-name so101_hil_collect
```

### 多节点（机器人节点与 GPU 节点分离）

1. 在每个节点 source 虚拟环境，并在 `ray start` 前导出 `RLINF_NODE_RANK`。
2. 启动 Ray：

```bash
# GPU / 主控节点（node rank 0）
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0
ray start --head --port=6379 --node-ip-address=<HEAD_IP> --disable-usage-stats

# 机器人节点（node rank 1）
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0
ray start --address=<HEAD_IP>:6379 --node-ip-address=<ROBOT_IP> --disable-usage-stats
```

3. 修改 YAML 中的 `cluster.num_nodes` 与 `node_groups`，使 `env` 落在机器人节点。
4. 在主控节点启动采集：

```bash
python collect_so101_hil_data.py --config-name so101_hil_collect
```

## 键盘控制（6D 末端位姿）

当 `use_keyboard_intervention: True` 时使用。每次按键更新当前激活臂的**目标末端位姿**，wrapper 通过 pinocchio IK 解算关节角后下发。

| 按键 | 含义 |
|------|------|
| `w` / `s` | 当前臂 EE 沿 X 轴 ± 移动 |
| `a` / `d` | 当前臂 EE 沿 Y 轴 ± 移动 |
| `q` / `e` | 当前臂 EE 沿 Z 轴 ± 移动 |
| `i` / `k` | 当前臂 EE 绕 X 轴（roll）± 旋转 |
| `j` / `l` | 当前臂 EE 绕 Y 轴（pitch）± 旋转 |
| `u` / `o` | 当前臂 EE 绕 Z 轴（yaw）± 旋转 |
| `,` / `.` | 当前臂爪闭 / 开（evdev 键名为 `Key.comma` / `Key.dot`） |
| `Tab` | 切换激活臂：`left` ↔ `right` |
| `h` | 切换 MODEL / ENGAGE |
| `m` | 立即回到 MODEL（策略自动控制） |
| `Enter` | 结束当前 episode，保存到同一个数据集并开始下一个 |
| `ESC` | 直接退出采集程序，不保存当前 episode |

**两态说明**：

- **MODEL**：双臂由 pi0.5 策略控制。
- **ENGAGE**：当前激活臂由键盘 6D 位姿控制；另一臂保持当前关节位姿不变。

**episode 控制**：

- 每个 episode 没有固定步数上限（YAML 中设置了很大的安全兜底值）。任务做完或需要结束本条时，按 `Enter` 保存当前 episode 到同一个 LeRobot 数据集，日志会显示 `Episode N saved; continuing to next episode.`，并立即开始下一条。
- 想结束采集时按 `ESC`，**不保存**当前 episode，日志显示 `Quit requested by operator (ESC); exiting without saving the current episode.`，随后程序退出。
- `m` 仅用于温和地切回 MODEL，不会结束 episode。

**关于数据集**：所有 episode 都保存在同一个 LeRobot repo（例如 `logs/so101_hil_collect/collected_data/rank_0/id_0/`）下，每条 episode 对应一个 `episode_*.parquet` 文件。这不是"每条一个数据集"，而是标准的多 episode 数据集结构。

## 输出数据

数据保存在 `runner.logger.log_path/collected_data/` 下，LeRobot 格式。

每条帧包含：

- `actions`：实际执行的动作（MODEL 时为策略动作，ENGAGE 时为人类动作）。
- `model_action`：策略原本输出的动作，用于从错误中恢复的学习。
- `intervene_flag`：该帧是否处于人工接管。
- `state`、相机图像、`task` 等常规字段。

可用 LeRobot 读取：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.from_preloaded("/path/to/collected_data/rank_0/id_0")
print(ds[0]["actions"])
print(ds[0]["model_action"])
print(ds[0]["intervene_flag"])
```

## 安全与排错

1. **首次真机务必 `tele_mode: true`**，确认策略输出合理后再关闭。
2. **ENGAGE 时先小步验证**：首次进入 ENGAGE 后按一次 `w`/`s` 等，确认机械臂实际运动方向与键盘指令一致；如果方向反了，检查 URDF 与实际机器人的关节零点是否一致。
3. **键盘无响应**：
   - 运行用户需要能读取 `/dev/input/event*`（在 `input` 组，或设备权限为 `a+r`）。
   - 如果系统检测到多个键盘设备，`KeyboardListener` 会报错，此时需设置 `RLINF_KEYBOARD_DEVICE=/dev/input/eventX`，指向实际使用的键盘设备。
4. **Enter/ESC 不结束 episode / 数据未保存**：确认 YAML 的 `override_cfg` 中设置了 `manual_episode_control_only: True`。否则 `RealWorldEnv` 会用 `max_episode_steps` 覆盖键盘 wrapper 设置的 `truncated`。
5. **找不到串口**：执行 `ls /dev/ttyACM*` 并更新 YAML 中的 `left_follower_port` / `right_follower_port`。
6. **IK 不收敛/抖动**：调大 `keyboard_intervention.rotation_delta` 或 `position_delta`，或调整 IK 参数 `tol`/`damping`（目前代码内置默认值）。
7. **数据未写入**：`runner.logger.log_path` 建议使用绝对路径，或确保 worker 节点上的相对路径可写。

## 更多细节

完整 SO101 真机 RL 文档参见 `docs/source-zh/rst_source/examples/embodied/so101.rst`。
