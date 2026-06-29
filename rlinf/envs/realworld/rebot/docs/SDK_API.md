# reBotArm SDK 使用手册（用户接口）

本文档只覆盖**用户实际会接触到的高层接口** —— 即 `reBotArm` 这个门面类。
运动学/动力学/轨迹/电机驱动等底层模块属于内部实现，正常使用无需直接调用。

```python
from reBotArm_control_py import reBotArm
```

底层架构：`reBotArm` 封装在官方 `RebotArm`（统一分组，持 `arm` / `gripper` 两个
`JointGroup`）+ `RebotArmEndPose`（末端控制器）之上。上电后跑一个 500Hz 控制环，
**每周期按组同步发送**：机械臂走 `POS_VEL`，夹爪走 `MIT`。所有运动接口最终都落到
两个目标上——`_q_target`（6 臂关节）与 `_gripper_target`（夹爪），由控制环实发。

---

## 快速开始

### 方式一：上下文管理器（推荐）

自动完成 连接 → 上电 → 运动 → 安全下电断开。

```python
from reBotArm_control_py import reBotArm

with reBotArm() as arm:
    arm.move_pose(0.3, 0.0, 0.2, duration=2.0)             # 末端移动到 (x,y,z)
    arm.move_joints([0.5, -0.3, 0, 0, 0, 0], duration=1.5) # 关节运动
    arm.move_open_gripper()                                # 张开夹爪
# 退出 with 自动下电、断开
```

### 方式二：分步控制（可在不上电时先读角度）

```python
arm = reBotArm()
arm.connect()                       # 仅建立通信
print(arm.get_joint_positions())    # 不上电也能读关节角
arm.power_on()                      # 上电：清故障 → 切模式 → 使能 → 启动控制循环
arm.move_pose(0.3, 0.0, 0.2)
arm.disconnect()                    # 安全关闭
```

> ⚠️ **重力风险**：`power_off()` / `disconnect()` 后机械臂失去力矩，会因自重下坠。下电前请确保机械臂处于安全姿态或有支撑。

> **DM / RS 切换**：改 `config/rebotarm.yaml` 的 `hardware_yaml`（`rebotarm_rs.yaml` / `rebotarm_dm.yaml`）即可，电机驱动与运动学 URDF 一起切换。

---

## 1. 生命周期管理

| 方法 / 属性 | 说明 |
|------|------|
| `connect()` | 连总线、注册电机。此时可读关节角，但电机未使能，不能运动 |
| `power_on()` | 上电：清故障 → 臂切 POS_VEL、夹爪切 MIT → 使能 → 目标初始化为当前位置（防猛动）→ 启动 500Hz 控制循环 |
| `power_off()` | 下电：停控制循环 → 失能全部电机 ⚠️ 会掉力下坠 |
| `disconnect()` | 安全关闭：停运动 → 回零（safe_home）→ 下电 → 关总线 |
| `clear_error(retries=3)` | 清除电机错误保护状态（达妙错误码 8~E 使能前需先清，RS 兼容此流程） |
| `estop()` | 紧急停止，立即失能所有电机 ⚠️ 会掉力 |
| `is_connected` / `is_powered` | 属性：是否已连接 / 是否已上电使能 |
| `with reBotArm() as arm:` | 上下文管理器，自动 `connect()`+`power_on()` / `disconnect()` |

---

## 2. 状态读取

所有读取均为实时值，可在运动过程中调用。

| 方法 | 返回 | 单位 |
|------|------|------|
| `get_joint_positions()` | 6 关节角度，shape=(6,) | 弧度 |
| `get_joint_velocities()` | 6 关节角速度，shape=(6,) | rad/s |
| `get_joint_torques()` | 6 关节力矩，shape=(6,) | N·m |
| `get_state()` | `(pos, vel, torq)`，各 shape=(6,) | 弧度 / rad·s⁻¹ / N·m |
| `get_end_effector_pose()` | `(position, rpy)` 末端位姿 | 米 / 弧度 |
| `get_end_effector_position()` | `[x, y, z]` 末端位置 | 米 |
| `get_end_effector_orientation()` | `[roll, pitch, yaw]` 末端姿态 | 弧度 |
| `get_gripper_position()` | 夹爪当前位置 | 电机角度 |
| `is_moving()` | `bool`，是否正在执行轨迹运动 | — |

```python
pos, rpy = arm.get_end_effector_pose()
print(f"末端在 {pos}, 姿态 {rpy}")
```

> FK 内部会把 6 维臂关节角用 `pad_q_for_model` 补齐到模型维度（RS 臂为 8 DOF，含 2 个夹爪平动）。

---

## 3. 运动控制

运动接口分两类：

- **规划类**（`move_*`）：默认**阻塞**直到到位；传 `wait=False` 立即返回，配 `is_moving()` 自行轮询。
- **伺服类**（`servo_*`）：即时写目标、**无轨迹平滑**，由调用方自行控频（高频流控 / 遥操作 / 视觉跟踪）。

### 关节空间

| 方法 | 说明 |
|------|------|
| `move_joints(q_target, duration=2.0, wait=True)` | 运动到 6 个目标关节角（关节空间线性插值后逐点送 `_q_target`） |
| `move_home(duration=3.0)` | 所有关节回零位 |
| `servo_joints(q_target)` | 即时把 6 关节目标写入控制环，无规划 |

### 末端笛卡尔空间

| 方法 | 说明 |
|------|------|
| `move_pose(x, y, z, roll=0, pitch=0, yaw=0, duration=2.0, wait=True)` | 末端运动到目标位姿（IK → SE(3) 测地线规划 → CLIK 跟踪），返回 `bool` |
| `move_position(x, y, z, duration=2.0, wait=True)` | 只控位置，保持当前姿态 |
| `move_relative(dx=0, dy=0, dz=0, duration=2.0, wait=True)` | 相对当前末端位置做平移 |
| `servo_pose(x, y, z, roll=0, pitch=0, yaw=0)` | 即时 IK 一步求解直接写 `_q_target`，无平滑，返回 IK 是否收敛 |

```python
# 规划：阻塞到位
arm.move_pose(0.3, 0.0, 0.3, pitch=0.4, duration=2.0)

# 规划：非阻塞 + 监控
arm.move_pose(0.3, 0.0, 0.2, wait=False)
while arm.is_moving():
    print(arm.get_end_effector_position())
    time.sleep(0.1)

# 伺服：高频流控（调用方自己 sleep 控频）
while streaming:
    arm.servo_pose(x, y, z, roll, pitch, yaw)
    time.sleep(0.02)
```

> `servo_*` 只更新机械臂的 6 个目标；夹爪由控制环独立保持在最近一次设定的目标，**无需也不应**在伺服里传夹爪。

---

## 4. 夹爪控制

夹爪随控制环以 **MIT** 持续驱动，下列接口只更新目标 / 刚度，实发由控制环完成。

| 方法 | 说明 |
|------|------|
| `move_gripper(pos, vlim=None)` | 设夹爪目标位置，配置默认刚度保持。`vlim` 已忽略（MIT 无速度限制参数） |
| `move_open_gripper()` | 完全张开（电机位取硬件配置 `gripper_open`；RS=4.7，DM=-5.7） |
| `move_close_gripper()` | 完全闭合（电机位取硬件配置 `gripper_close`，默认 0.0） |
| `move_gripper_force(pos, ratio=0.3, vlim=None)` | **限力夹持**：把夹爪 MIT 的 kp 缩放到基准的 `ratio` 倍（软弹簧=柔顺），夹到物体不硬怼。DM / RS 通用 |

```python
arm.move_open_gripper()
arm.move_gripper_force(0.0, ratio=0.2)   # 柔顺夹持，ratio 越小越软
```

> **限力机制**：`move_gripper_force` 限的是 MIT“刚度”（kp），不是绝对力矩。它对所有厂商有效，与达妙固件 `FORCE_POS` 的“绝对电流上限”机制不同。本封装不使用 `FORCE_POS`（与持续 MIT 控制环冲突）。

---

## 5. 拖动示教 + 录制回放

进入示教模式后机械臂临时切到 **MIT 力控**并做重力补偿，**可徒手拖动**，同时自动以 30Hz 记录关节角轨迹（夹爪在示教期间也由 MIT 顶住，不会松脱）。

| 方法 | 说明 |
|------|------|
| `teach_mode(True, kd=1.0)` | 进入示教：重力补偿浮停 + 自动开始录制，返回 `None` |
| `teach_mode(False)` | 退出示教：停录制 → 自动存 `.npy` → 臂切回 POS_VEL 锁定当前位置，返回保存路径 |
| `replay_trajectory(traj=None, path=None, rate=30.0, wait=True, record=False)` | 回放轨迹（传数组 / 从 `.npy` 加载 / 默认用最近一次录制）。`record=True` 同时记录实际角并返回 |
| `save_teach_trajectory(path=None)` | 先做轨迹优化再保存（默认带时间戳，存到 `records/`） |
| `optimize_trajectory(traj, rate=30.0)` | 贪心覆盖法重参数化，保证相邻帧位移不超电机可达范围 |
| `get_teach_trajectory()` / `clear_teach_trajectory()` | 取 / 清当前录制轨迹，shape=(N, 6) |
| `start_teach_recording()` / `stop_teach_recording(auto_save=True)` | 手动控制录制开关 |

```python
arm.teach_mode(True)
input("拖动机械臂走一遍，完成后回车...")
path = arm.teach_mode(False)     # 自动保存到 records/，返回路径
arm.replay_trajectory(path=path) # 回放
```

---

## 6. 停止

| 方法 | 说明 |
|------|------|
| `stop()` | 立即停止当前运动，**保持当前位置**（不掉力）；示教中则退出示教 |
| `estop()` | 紧急停止，失能所有电机 ⚠️ **会掉力下坠** |

---

## 7. 控制模式说明（重要）

控制不是“全部走 MIT”，而是**按组分模式**，由控制环每周期同步发送：

| 部件 | 正常运行模式 | “刚度/响应”由什么决定 |
|------|------|------|
| 机械臂 6 关节 | **POS_VEL**（电机内部位置+速度 PID 闭环） | 配置里的 PI 增益 `vel_kp/vel_ki/pos_kp/pos_ki` + `vlim`，在切入 POS_VEL 时写入电机寄存器（RS 用 `robstride_write_param_f32`，DM 用 `write_register_f32`），**不是每帧下发 kp/kd** |
| 夹爪 | **MIT**（主机下发 pos/vel/kp/kd/tau 阻抗控制） | 每周期下发的 `kp/kd`；`move_gripper_force` 即通过缩放 kp 实现柔顺限力 |
| 机械臂（示教时） | 临时 **MIT** | 重力补偿 `tau` + 低 kp/kd（kp=0、kd≈1），实现可拖动浮停 |

所以你说的“发 PV 指令就把刚度调一下”——**机械臂本来就一直在跑 PV（POS_VEL）**。PV 下没有“每帧 kp/kd”这个概念，它的软硬由那几个 PI 增益决定（在 `rebotarm_rs.yaml`/`rebotarm_dm.yaml` 里配，切模式时写一次寄存器）。真正“每帧调 kp/kd 刚度”的是 MIT，目前只有夹爪和示教用。

> 如果将来想让**机械臂**也走 MIT（带重力前馈的阻抗控制、可调每帧刚度），把封装内构造改成
> `RebotArmEndPose(self._arm, arm_control_mode="mit")` 即可——控制器两种模式都支持，
> 届时臂的刚度就由 MIT 的 kp/kd 决定。当前默认用 POS_VEL（定位更稳、参数在固件侧闭环）。

---

## 8. 配置文件

| 文件 | 内容 |
|------|------|
| `config/rebotarm.yaml` | 全局入口：`hardware_yaml` 指向实际硬件配置（默认 `rebotarm_rs.yaml`），切换 DM/RS 只改这一行 |
| `config/rebotarm_rs.yaml` | RobStride 电机 + CAN 总线（`can0`）：关节分组（arm + gripper）、电机 id/型号/增益、URDF 路径与末端帧 |
| `config/rebotarm_dm.yaml` | 达妙电机 + 串口桥（`/dev/ttyACM0`）：同上字段，对应 DM 机械臂 |

机械臂与夹爪在同一份硬件配置中以 `groups` 分组（`arm` / `gripper`），由统一的
`RebotArm`（JointGroup 架构）加载；`urdf_path` 与 `end_effector_frame` 也在硬件配置中指定，
DM / RS 自动切换对应的 URDF 与运动学模型。

---

## 典型应用模板

### 点位搬运 / 抓取

```python
with reBotArm() as arm:
    arm.move_pose(0.30, 0.10, 0.20)   # 移到取料点上方
    arm.move_gripper_force(0.0, ratio=0.3)  # 限力抓取
    arm.move_pose(0.30, -0.10, 0.20)  # 移到放料点
    arm.move_open_gripper()            # 释放
    arm.move_home()
```

### 示教编程（Teach & Repeat）

```python
with reBotArm() as arm:
    arm.teach_mode(True)
    input("手把手示教，完成回车...")
    path = arm.teach_mode(False)
    for _ in range(3):                 # 重复执行 3 遍
        arm.replay_trajectory(path=path)
```

### 末端轨迹任务（画线 / 巡检）

```python
with reBotArm() as arm:
    waypoints = [(0.3, 0.0, 0.3), (0.3, 0.1, 0.3), (0.3, 0.1, 0.2)]
    for x, y, z in waypoints:
        arm.move_pose(x, y, z, duration=1.5)
```

---

## 注意事项

- **下电会掉力**：`power_off` / `disconnect` / `estop` 后机械臂失去力矩，注意重力下坠。`stop()` 则保持位置不掉力。
- **按组分模式**：机械臂用 POS_VEL、夹爪用 MIT、示教临时用 MIT；模式切换由门面自动管理，正常使用无需关心（详见第 7 节）。
- **运动阻塞性**：`move_*` 默认阻塞到位，非阻塞用 `wait=False` + `is_moving()`；`servo_*` 永远即时返回。
- **更多底层示例**：见 `example/` 目录（单电机调试、正逆运动学、重力补偿、仿真等）。
