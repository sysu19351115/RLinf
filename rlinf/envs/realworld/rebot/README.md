# reBot Arm B601-RS 的 Pinocchio 与 MeshCat 入门指南

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Ubuntu-orange.svg" alt="Platform">
    <img src="https://img.shields.io/badge/Motor-RobStride-green.svg" alt="RobStride">
    <img src="https://img.shields.io/badge/Bus-CAN%201Mbps-red.svg" alt="CAN 1Mbps">
</p>

<p align="center">
  <strong>6 自由度机械臂 · RobStride 电机 · CAN 总线 · 运动学求解 · 轨迹规划 · 完全开源</strong>
</p>

---

## 📖 项目简介

**reBotArm Control** 是一个面向 reBot Arm B601 系列机械臂的 Python 控制库，提供从底层电机控制到上层运动学解算的完整解决方案。本指南以 **B601-RS（RobStride / 灵足电机 + CAN 总线）** 为准；DM（达妙）版只需改一行配置即可切换（见文末）。

### ✨ 核心特性

- 🦾 **双型号支持** — B601-RS（RobStride 电机，**默认**）和 B601-DM（达妙电机）
- 🚌 **CAN 总线通信** — RS 版经 USB-CAN 适配器走标准 SocketCAN，1 Mbps
- 🧮 **运动学求解** — 基于 Pinocchio 的正/逆运动学计算
- 🛤️ **轨迹规划** — SE(3) 测地线轨迹 + CLIK 跟踪
- 🔧 **灵活配置** — YAML 配置文件，快速适配不同硬件

---

## ⚙️ 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| **Python** | 3.10+ |
| **操作系统** | Ubuntu 22.04+ |
| **通信接口** | USB-CAN 适配器（PCAN-USB / CANable 等），SocketCAN |
| **总线速率** | **1 Mbps**（RobStride 固定） |

### 安装步骤

#### 步骤 1. 安装 uv（如未安装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 步骤 2. 克隆并同步环境（安装所有依赖）

```bash
git clone https://cnb.cool/THU-HiGroup/rebot.git
cd rebot
uv sync
```

:::tip
`uv sync` 会自动创建虚拟环境（如不存在）并根据 `pyproject.toml` 和 `uv.lock` 安装所有依赖（含 `motorbridge` 电机 SDK、`pinocchio` 等）。
:::

---

## 🔌 硬件配置（RobStride + CAN，重点）

reBot Arm B601-RS 用 RobStride（灵足）电机，经 USB-CAN 适配器（PCAN-USB / CANable 等）走标准 **SocketCAN**，总线速率 **1 Mbps**。下面这套 `can0` 的搭建是 RS 版能否通信的关键。

### 步骤 1. 接好硬件

USB-CAN 适配器一端 USB 接电脑，另一端 CAN_H / CAN_L 接机械臂的 CAN 总线，确认总线两端有 **120 Ω 终端电阻**，再给机械臂上电。

### 步骤 2. 构建 `can0` 接口（核心）

```bash
# (a) 加载内核模块
#     - PCAN-USB 适配器：
sudo modprobe peak_usb
#     - CANable / gs_usb 系：内核一般自带 gs_usb，可跳过；slcan 类适配器另见其说明

# (b) 把 can0 配成 1 Mbps 并启用（RobStride 总线速率固定 1 Mbps）
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up

# (c) 验证：应显示 state UP、bitrate 1000000
ip -details link show can0
```

:::tip
- **每次重启电脑或重插适配器后都要重新执行 (b)**。可把这几条写进开机脚本 / `systemd` 服务自动拉起。
- `bitrate` 必须是 **1000000（1 Mbps）**；写成 500000 等其它值会导致收不到电机反馈、表现为“扫描不到电机 / 状态全 0”。
- `restart-ms 100` 让总线 bus-off 后自动恢复，建议保留。
:::

### 步骤 3. 扫描电机（确认 7 个电机在线）

```bash
motorbridge-cli scan --vendor robstride --channel can0 --start-id 1 --end-id 127
```

应能扫到 **id 0x01 ~ 0x07**（6 个关节 + 1 个夹爪）。RobStride 的 **host / feedback id 固定为 0xFD**。

### 步骤 4. 单电机连通性测试

```bash
uv run python example/0x01rs06_test.py
# 进入交互后输入：ping（探测电机）/ state（读状态）/ enable / disable / set_zero ...
```

### RS 电机布局（来自 `config/rebotarm_rs.yaml`）

| 关节 | motor id | 型号 | feedback / host id |
|------|----------|------|--------------------|
| joint1 | `0x01` | `rs-06` | `0xFD` |
| joint2 | `0x02` | `rs-06` | `0xFD` |
| joint3 | `0x03` | `rs-06` | `0xFD` |
| joint4 | `0x04` | `rs-00` | `0xFD` |
| joint5 | `0x05` | `rs-00` | `0xFD` |
| joint6 | `0x06` | `rs-00` | `0xFD` |
| gripper | `0x07` | `rs-00` | `0xFD` |

> 底层连接等价于：`ctrl = Controller("can0")` → `ctrl.add_robstride_motor(motor_id, 0xFD, model)`，由 `reBotArm` 按 `config/rebotarm_rs.yaml` 自动完成。

### 电机品牌 / 传输对照

| 电机品牌 | 传输方式 | `transport` | 总线速率 | feedback id 规则 |
|----------|---------|-------------|---------|------------------|
| **RobStride（默认）** | CAN 接口（`can0`） | `socketcan` | **1 Mbps** | **固定 `0xFD`** |
| 达妙 (Damiao) | USB2CAN 串口桥 | `dm-serial` | 921600 | `motor_id + 0x10` |
| 达妙 (Damiao) | CAN 接口 | `socketcan` | 1 Mbps | `motor_id + 0x10` |

:::tip
- RobStride **不走串口**，必须先用上面的 `ip link` 把 `can0` 拉起来。
- RobStride 的 `feedback_id` 是固定的 `0xFD`（host id），**不是** `motor_id + 0x10`（那是达妙的规则）。
:::

### 切换到 DM（达妙）版

只改 `config/rebotarm.yaml` 一行：

```yaml
hardware_yaml: "rebotarm_dm.yaml"   # 默认是 rebotarm_rs.yaml
```

DM 版走达妙 USB2CAN 串口桥，系统识别为 `/dev/ttyACM0`，`dm-serial` @ 921600，**不需要**上面的 `ip link` 步骤；验证用：

```bash
ls /dev/ttyACM0
motorbridge-cli scan --vendor damiao --transport dm-serial \
    --serial-port /dev/ttyACM0 --serial-baud 921600
```

---

## 📁 项目结构

```
reBotArm_control_py/
├── config/                     # 配置文件
│   ├── rebotarm.yaml           # 全局入口：hardware_yaml 指向实际硬件（默认 rebotarm_rs.yaml）
│   ├── rebotarm_rs.yaml        # RobStride + CAN（can0）：电机 id/型号/增益、URDF、末端帧
│   └── rebotarm_dm.yaml        # 达妙 + 串口桥（/dev/ttyACM0）
├── example/                    # 示例程序
│   ├── 0x01rs06_test.py        # RobStride 单电机控制台（CAN）
│   ├── 0x01damiao_test.py      # 达妙单电机控制台（串口）
│   ├── 2_zero_and_read.py      # 零点校准 + 角度监控
│   ├── 3_mit_control.py        # MIT 控制
│   ├── 4_pos_vel_control.py    # POS_VEL 控制
│   ├── 5_fk_test.py            # 正运动学
│   ├── 6_ik_test.py            # 逆运动学
│   ├── 7_arm_ik_control.py     # IK 实时控制
│   ├── 8_arm_traj_control.py   # 轨迹规划
│   ├── 9_gravity_compensation.py  # 重力补偿
│   ├── replay_and_record.py    # 示教录制 + 回放
│   └── sim/                    # 仿真工具
├── reBotArm_control_py/        # 核心库
│   ├── actuator/               # 执行器模块
│   ├── kinematics/             # 运动学模块
│   ├── controllers/            # 控制器模块
│   └── trajectory/             # 轨迹规划模块
├── urdf/                       # URDF 模型（RS / DM 各自）
└── README.md
```

---

## 🎮 示例程序

> 运行前请先确认 `can0` 已按上文拉起（`ip -details link show can0` 显示 UP、1 Mbps）。

### 调试工具

#### 1️⃣ RobStride 单电机控制台 (`0x01rs06_test.py`)

直接使用 motorbridge SDK 在 `can0` 上测试单个 RobStride 电机（默认 id `0x01`、型号 `rs-06`、host id `0xFD`）。

**运行方式**：
```bash
uv run python example/0x01rs06_test.py
```

**交互命令**：
| 命令 | 说明 |
|------|------|
| `ping` | 探测电机是否在线 |
| `enable` / `disable` | 使能 / 失能 |
| `mit <pos_deg> [vel kp kd tau]` | MIT 阻抗控制 |
| `posvel <pos_deg> [vlim [loc_kp]]` | POS_VEL 位置控制 |
| `set_zero` | 设置零位 |
| `state` | 查看状态 |
| `clear_error` | 清除错误 |

---

#### 2️⃣ 零点校准与角度监控 (`2_zero_and_read.py`)

自动设置所有关节零点，实时显示关节角度。

**运行方式**：
```bash
uv run python example/2_zero_and_read.py
```

---

### 运动学测试

#### 5️⃣ 正运动学测试 (`5_fk_test.py`)

根据关节角度计算末端位姿。

**输入**：6 个关节角度（度）

**输出**：
- 末端位置 (X, Y, Z) — 单位：米
- 旋转矩阵 (3×3)
- 欧拉角 (横滚/俯仰/偏航) — 单位：度

**示例**：
```bash
uv run python example/5_fk_test.py
> 0 0 0 0 0 0
> 45 -30 15 -60 90 180
```

---

#### 6️⃣ 逆运动学测试 (`6_ik_test.py`)

根据期望末端位姿求解关节角度。

**输入格式**：
- 仅位置：`<x> <y> <z>`（米）
- 位置 + 姿态：`<x> <y> <z> <roll> <pitch> <yaw>`（度）

**示例**：
```bash
uv run python example/6_ik_test.py
> 0.25 0.0 0.15              # 仅位置
> 0.25 0.0 0.15 0 0 0        # 位置 + 姿态
```

---

### 实机控制

:::tip 设备权限
- **RobStride（CAN）**：`can0` 是网络接口，按上文 `sudo ip link set can0 up` 拉起后即可使用，**无需 `chmod`**。
- **达妙（串口）**：才需要给串口设备授权 `sudo chmod 666 /dev/ttyACM0`。
:::

#### 7️⃣ IK 实时控制 (`7_arm_ik_control.py`)

基于 IK 解算的机械臂实时末端控制。

**交互命令**：
| 命令 | 说明 |
|------|------|
| `x y z [roll pitch yaw]` | 目标末端位姿 |
| `state` | 查看状态 |
| `pos` | 当前末端位置 |
| `q/quit/exit` | 退出 |

**运行方式**：
```bash
uv run python example/7_arm_ik_control.py
> 0.3 0.0 0.2
> 0.3 0.1 0.25 0 0.5 0
```

---

#### 8️⃣ 轨迹规划控制 (`8_arm_traj_control.py`)

SE(3) 测地线轨迹规划 + CLIK 跟踪。

**输入格式**：
```
x y z [roll pitch yaw] [duration]
```

**参数说明**：
- `x, y, z`: 目标位置（米）
- `roll, pitch, yaw`: 目标姿态（弧度）
- `duration`: 运动时长（秒），默认 2.0s

**运行方式**：
```bash
uv run python example/8_arm_traj_control.py
> 0.3 0.0 0.3 0 0.4 0 2.0
```

---

#### 9️⃣ 重力补偿控制 (`9_gravity_compensation.py`)

使用 Pinocchio 动力学模型补偿关节重力。

**控制律**：
```
tau = g(q)          — 重力前馈
pos = 当前电机位置   — 关节位置跟随当前位置
kp = 2,  kd = 1     — 所有关节统一刚度/阻尼
```

**预期行为**：
- 机械臂可以在任意姿态下"漂浮"
- 松开后不会因自重坠落
- 可以手动掰动到任意位置

**运行方式**：
```bash
uv run python example/9_gravity_compensation.py
```

**输出**：
- 实时显示各关节期望力矩（N·m）
- 按 `Ctrl+C` 停止并断开连接

---

## 🛠️ 常见问题（RS / CAN）

| 现象 | 排查 |
|------|------|
| `scan` 扫不到电机 / 状态全 0 | `can0` 没拉起或速率不对：`ip -details link show can0` 确认 UP 且 `bitrate 1000000`；检查 120 Ω 终端电阻与接线 |
| 重启后通信失败 | `can0` 重启会丢，需重新执行步骤 2(b)；建议写进开机脚本 |
| `RTNETLINK answers: Device or resource busy` | 先 `sudo ip link set can0 down` 再重配 |
| `Cannot find device "can0"` | 内核模块没加载（PCAN 需 `sudo modprobe peak_usb`）或适配器没插好 |
| 夹爪 / 某关节不响应 | 用 `0x01rs06_test.py` 改 `MOTOR_ID` 单独 `ping` 该电机；确认 host id 为 `0xFD` |

---

## 📄 License

本项目采用 **MIT 许可证** 开源。

---

## ☎ 联系我们

- **项目仓库**: [cnb.cool/THU-HiGroup/rebot](https://cnb.cool/THU-HiGroup/rebot)

---

<p align="center">
  <strong>🌟 如果本项目对你有帮助，请给个 Star 支持一下！</strong>
</p>
