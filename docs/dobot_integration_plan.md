# Dobot 机器人集成到 RLinf 的实施规划

> **文档目的**：固化 Dobot CR5AF（6-DOF 机械臂 + 达妙夹爪）在 RLinf 中的集成方案，供后续实施参考。
>
> **核心结论**：以 `rlinf/envs/realworld/rebot/` 为 1:1 结构模板，三处定制化扩展——(1) 双控制模式（关节角 `ServoJ` + 位姿 `ServoP`），(2) SDK 以 git submodule 形式引入，(3) 单臂 HIL 键盘接管。

---

## 一、需求与约束

| # | 需求 | 来源 |
|---|---|---|
| 1 | **关节角控制 + 位姿控制**（两种模式都要支持，已在 openpi 验证） | openpi `examples/dobot_clean` |
| 2 | **SDK 用干净维护版**：`https://cnb.cool/THU-HiGroup/dobot_zhiyu`，以 git submodule 引入 | 用户指定 |
| 3 | **HIL 键盘接管**（人在环数据采集） | 参考 so101_keyboard_intervention |

**硬性约束**（移植中绝不可破坏，否则真实机器人危险或训练/推理不一致）：

1. **单位边界**：控制器对外统一用 **弧度**（关节）、**米 + 四元数 `[x,y,z,qw,qx,qy,qz]`**（位姿，w-first）、**归一化 `[0,1]` 0=闭/1=开**（夹爪）。SDK 内部是 度/mm/欧拉角/电机角度，转换只在控制器内做一次。
2. **四元数 w-first**：在所有层（env / 策略 / DeltaPose / AbsolutePose）保持 `[qw,qx,qy,qz]`，仅在 scipy 调用内部临时转 `[x,y,z,w]`。
3. **夹爪方向反转**：电机角度 `closed=0°, open=-320°`（open 更负），归一化空间 `0=闭, 1=开`——与电机角度符号相反。
4. **`user_index` / `tool_index` / 夹爪校准 (`closed_deg/open_deg`)** 采集与推理必须一致，否则位姿参考系漂移。
5. **位姿 delta 是 SE(3)**：训练用 `DeltaPose`、推理用 `AbsolutePose`，走 4×4 齐次矩阵，**不是**朴素加减；缺 `prev_state` 时变 no-op（会把相对位姿直接送机器人，非物理）。
6. **首帧 `prev_state = state` 自身**（identity delta），env 的 `PoseStateTracker` 与数据集转换器都要遵守。
7. **夹爪永远 absolute**：joint 模式 mask `make_bool_mask(6, -1)`，pose 模式 mask `make_bool_mask(7, -1)`——只有前 6（关节）或 7（位姿）维变 delta。
8. **ServoJ/ServoP 安全**：限速（pose 平移 5mm/周期、pose 旋转 2°/周期、关节每轴 2°/周期）+ 跳变拒绝（pose 平移 10cm、pose 旋转 30°、关节单轴 30°）+ RobotMode 健康检查 ∈ `{5,7,8}` + `_servo_rejected_count` 告警。
9. **单次反馈读取** `read_state_from_feedback` 同时取关节+位姿+wrench（30Hz 下避免三次 socket 读）；wrench 不可信时保持上一帧，绝不回退 `ActualTCPForce`。
10. **夹爪 FORCE_POS 模式**（不是 MIT，已验证损坏），每次命令前 `ensure_mode` 重发。
11. **位姿模式必走 ServoP**：`state_mode="pose"` ⟹ `action_mode="cartesian"`（env __init__ 强制校验）；但归位/保持等管理动作始终走关节空间（`drive_arm_joints`）。
12. **动作维度 fail-fast**：`split_follower_action` 维度不匹配直接 `ValueError`，绝不静默切片。
13. **图像处理一致性**：center crop → 保比例 resize 到长边 224 → 补零到 224×224 → HWC uint8，训练/推理完全一致。

---

## 二、整体架构对照

| 层 | openpi（dobot 现状） | RLinf rebot（结构模板） | dobot 在 RLinf 的对应 |
|---|---|---|---|
| 入口 | `main.py` + tyro CLI | `train_embodied_agent.py` + Hydra YAML | 复用 rebot 入口 |
| Env ABC | `openpi_client.Environment`（4 方法） | `gym.Env`（reset/step） | 适配为 `gym.Env` |
| 控制器 | `DobotController`（普通类） | `RebotArmController(Worker)`（Ray actor） | `DobotController(Worker)` |
| SDK 引入 | `third_party/dobot_zhiyu`（submodule）+ `sys.path` 注入 | 控制器 `__init__` 内 `sys.path.insert` | 同 rebot + submodule |
| 硬件枚举 | 无（手填 IP） | `RebotArmRobot(Hardware)` 装饰器 | `DobotRobot(Hardware)` |
| 任务注册 | 无 | `gym.register("RebotArmPickAndPlaceEnv-v1")` | `gym.register("DobotXxxEnv-v1")` |
| 策略变换 | `dobot_policy.py`（已存在） | `rebot_policy.py` | 直接复用 openpi 的 `DobotInputs/Outputs` + 适配键名 |
| HIL | 无 | so101 键盘 wrapper（双臂 + pinocchio IK） | 单臂 + Dobot 原生 IK |

**最大改造点**：(a) 控制器从普通类→`Worker`（分布式 Ray actor，所有调用 `.wait()`）；(b) 双模式（joint/pose）贯穿 env-action_space-obs-policy-dataconfig；(c) HIL 单臂化 + 去 pinocchio。

---

## 三、文件清单（新建/修改）

### 3.1 新建（按 rebot 目录结构 1:1 复刻 + 双模式 + HIL）

```
third_party/dobot_zhiyu/                          # git submodule（见 §四.1）
└── (由 cnb.cool/THU-HiGroup/dobot_zhiyu 提供)

rlinf/envs/realworld/dobot/
├── __init__.py                    # 导出 DobotEnv, DobotRobotConfig, DobotRobotState
├── dobot_env.py                   # DobotEnv(gym.Env) + DobotRobotConfig
├── dobot_robot_state.py           # DobotRobotState 数据类
├── dobot_controller.py            # DobotController(Worker) — 核心改造
├── kinematics.py                  # DobotKinematics（封装 SDK 原生 IK，HIL 用）
├── verify_env.py                  # 硬件就绪检查（可选，调试用）
├── tasks/
│   ├── __init__.py                # gym.register("DobotXxxEnv-v1")
│   └── pick_and_place.py          # 任务子类
└── (URDF 不需要：Dobot 用 SDK 原生 IK，不像 so101 要 pinocchio)

rlinf/envs/realworld/common/wrappers/
└── dobot_keyboard_intervention.py # DobotKeyboardIntervention(gym.ActionWrapper)

rlinf/scheduler/hardware/robots/
└── dobot.py                       # DobotRobot(Hardware) + DobotConfig + DobotHWInfo

rlinf/models/embodiment/openpi/
├── policies/dobot_policy.py       # 复用 openpi 的 DobotInputs/Outputs，适配 RLinf 键名
└── dataconfig/
    └── dobot_dataconfig.py        # DobotDataConfig（支持 joint/pose 两种 mask）

examples/embodiment/
├── config/
│   ├── env/realworld_dobot.yaml             # 基础 env 配置（含 action_mode/state_mode）
│   ├── dobot_single_node_ppo_pi05.yaml      # 单节点 RL
│   ├── dobot_async_ppo_pi05.yaml            # 双节点（云 GPU + 本地机器人）
│   └── dobot_hil_collect.yaml               # HIL 数据采集
└── collect_dobot_hil_data.py                # DobotHILCollector(Worker)

examples/reward/config/
└── dobot_reward_training.yaml               # ResNet18 reward model SFT（可选）

docs/examples/
├── dobot_pi05_ppo_single_node.md
├── dobot_pi05_ppo_async.md
└── dobot_hil_data_collection.md
```

### 3.2 修改（注册/导出接线）

| 文件 | 改动 |
|---|---|
| `.gitmodules` | 加 `third_party/dobot_zhiyu` submodule 条目 |
| `rlinf/envs/realworld/__init__.py` | `from .dobot import DobotEnv, DobotRobotConfig, DobotRobotState`；`from .dobot import tasks as dobot_tasks`；加入 `__all__` |
| `rlinf/scheduler/hardware/robots/__init__.py` | 导出 `DobotConfig, DobotHWInfo` |
| `rlinf/scheduler/hardware/__init__.py` | 导出 dobot 硬件类 |
| `rlinf/envs/realworld/common/wrappers/apply.py` | 加 `apply_dobot_wrappers(env, cfg)` |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 注册 `pi05_dobot_joint` 和 `pi05_dobot_pose` 两个 `TrainConfig` |
| `requirements/install.sh` 或 `requirements/embodied/` | 加 `motorbridge`（达妙夹爪依赖） |
| `rlinf/config.py` | **无需改动**（dobot 走 `REALWORLD` enum + gym id 派发，参考 rebot） |

---

## 四、SDK 引入策略（git submodule）

### 4.1 添加 submodule

```bash
cd /home/zylab/project/RLinf
git submodule add https://cnb.cool/THU-HiGroup/dobot_zhiyu third_party/dobot_zhiyu
git commit -s -m "feat(dobot): add dobot_zhiyu SDK as submodule"
```

> **注意**：openpi 中已是同一仓库的 submodule（已确认 `git remote -v` 指向 `cnb.cool/THU-HiGroup/dobot_zhiyu`），结构与本文假设一致。RLinf 直接引用同一上游，保证版本同步。

### 4.2 SDK 包结构（来自仓库）

```
third_party/dobot_zhiyu/
├── src/dobot_control/
│   ├── __init__.py
│   ├── dobot_api.py         # TCP 协议层（端口 29999 控制 + 30004 反馈）
│   ├── dobot_robot.py       # DobotRobot 高层封装（slew/jump 安全）
│   └── dobot_geometry.py    # 单位转换（mm+欧拉 ↔ m+四元数）
├── examples/                # 官方/测试脚本（不打包，仅供调试参考）
└── (纯 numpy + scipy 依赖，自包含)
```

### 4.3 sys.path 注入（在控制器 `__init__` 内，延迟导入）

```python
# rlinf/envs/realworld/dobot/dobot_controller.py
import os, sys

_SDK_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "third_party", "dobot_zhiyu", "src",
)
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)
from dobot_control import DobotRobot  # 延迟导入，GPU-only 节点也能 import 本模块
```

> 参考 rebot：路径从本文件位置反推到仓库根 + `third_party/dobot_zhiyu/src`。**所有 SDK/scipy/motorbridge 导入必须延迟到 `__init__`**，否则 GPU-only 节点 import 失败。

### 4.4 依赖追加

- `numpy`, `scipy`：RLinf 已有。
- `motorbridge`：达妙夹爪，加到 `requirements/embodied/` 或 `install.sh` 的 `SUPPORTED_MODELS` 分支。
- `evdev`：HIL 键盘（RLinf 已有，so101 用）。

---

## 五、核心模块实现要点

### 5.1 `dobot_controller.py` — Worker 化（核心改造）

**职责**：把 openpi 的普通类 `DobotController` 包装成 RLinf `Worker`（Ray actor）。完全模仿 `RebotArmController`。

```python
class DobotController(Worker):
    @staticmethod
    def launch_controller(ip="192.168.5.1", env_idx=0, node_rank=0,
                          worker_rank=0, **dobot_kwargs):
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return (DobotController.create_group(ip, **dobot_kwargs)
                .launch(cluster=cluster, placement_strategy=placement,
                        name=f"DobotController-{worker_rank}-{env_idx}"))

    def __init__(self, ip="192.168.5.1", speed=100, action_mode="joint",
                 state_mode="joint", gripper_port="/dev/ttyACM0",
                 enable_gripper=True, user_index=0, tool_index=0,
                 gripper_closed_deg=0.0, gripper_open_deg=-320.0, **kwargs):
        super().__init__()
        # sys.path 注入 + 延迟导入（见 §四.3）
        # 构造内部 DobotRobot + DamiaoGripper，enable + engage
```

**对外 RPC API**（全部返回 Future，env 侧 `.wait()`）：

| 方法 | 返回 | 说明 |
|---|---|---|
| `is_robot_up() -> bool` | Future[bool] | `RobotMode ∈ {5,7,8}` |
| `get_state() -> DobotRobotState` | Future[State] | **单次** `read_state_from_feedback` 取关节+位姿（+可选 wrench） |
| `get_state_and_pose() -> (state, pose, prev_state_or_None)` | Future[tuple] | pose 模式下额外返回 8 维 prev_state（由控制器内 `PoseStateTracker` 维护） |
| `send_action(action, action_mode=None) -> bool` | Future[bool] | `drive_follower`：ServoJ/ServoP + 夹爪；返回 False=护栏拒绝 |
| `move_joints(q_rad, duration=3.0)` | Future | MovJ 归位（仅 reset/管理用，不走 servo） |
| `reset_to_pose(joints, init_steps=60, init_fps=30)` | Future | 关节空间插值归位 |
| `engage()` / `release()` | Future | `reset_smoothing()`（清除 slew 参考）+ 置 `_follower_engaged` |
| `open_gripper()` / `close_gripper()` | Future | 归一化 1.0 / 0.0 |
| `move_gripper(norm_01)` | Future | 归一化 [0,1] |
| `tare_ft_sensor()` | Future | ForceVLA 才用，`six_force_home()` |
| `hold_current_position()` | Future | 两段 Ctrl+C 的一段保持 |
| `return_home_then_close(joints)` | Future | 两段 Ctrl+C 的归位关闭（min-jerk） |
| `close()` | Future | 幂等关闭 |

**内部单位边界**（控制器内一次性转换）：
- `drive_arm_joints(q_rad)`：`q_deg = q_rad * 180/π` → `robot.servo_joints(q_deg)`
- `drive_arm_pose(pose_quat_m)`：直接 `robot.servo_pose(pose_quat_m)`（SDK 内做 m→mm、quat→euler）
- `get_state`：关节度→弧度（`× π/180`）；位姿已是 m+quat（SDK `dobot_pose_to_quat_m`）

**关键复用**：`split_follower_action`、`binarize_gripper_action`、`RelativeGripperBinarizer`、`DamiaoGripper` 全部从 openpi 原样拷贝（纯函数 + 自包含硬件类）。

### 5.2 `dobot_robot_state.py`

```python
@dataclass
class DobotRobotState:
    arm_joint_position: np.ndarray          # (6,) 弧度
    arm_joint_velocity: Optional[np.ndarray] = None   # (6,)
    tcp_pose: np.ndarray                    # (7,) [x,y,z m, qw,qx,qy,qz] w-first
    gripper_position: float                 # 归一化 [0,1]
    gripper_open: bool
    action_mode: str = "joint"              # "joint" | "cartesian"，标记本帧来源
    wrench: Optional[np.ndarray] = None     # (6,) [Fx,Fy,Fz,Mx,My,Mz]，ForceVLA 才用
    servo_accepted: bool = True             # 上一帧 servo 是否被护栏接受
```

### 5.3 `dobot_env.py` — 适配 gym.Env（双模式）

照搬 `rebot_env.py` 结构，关键改动是 **`action_mode` / `state_mode` 双模式贯穿**。

#### 配置数据类

```python
@dataclass
class DobotRobotConfig:
    # 连接
    ip: Optional[str] = None
    speed: int = 100
    user_index: int = 0
    tool_index: int = 0
    gripper_port: str = "/dev/ttyACM0"
    enable_gripper: bool = True
    gripper_closed_deg: float = 0.0
    gripper_open_deg: float = -320.0

    # 双模式（核心）
    action_mode: str = "joint"      # "joint"(ServoJ) | "cartesian"(ServoP)
    state_mode: str = "joint"       # "joint"(7维) | "pose"(8维 + prev_state)

    # 相机
    camera_serials: Optional[list[str]] = None
    camera_type: str = "realsense"
    enable_high_camera: bool = True     # cam_high 可选

    # 其他（同 rebot）
    is_dummy: bool = False
    task_description: str = ""
    use_dense_reward: bool = False
    use_reward_model: bool = False
    reward_worker_cfg: Optional[dict] = None
    reward_worker_node_rank: Optional[int] = None
    reward_worker_node_group: Optional[str] = None
    reward_worker_hardware_rank: int = 0
    reward_image_key: Optional[str] = None
    step_frequency: float = 30.0        # dobot 30Hz（远高于 rebot 10Hz）
    initial_joint_pos: list = field(default_factory=lambda: [0.0]*7)  # [j1..j6 rad, gripper norm]
    joint_limit_low: np.ndarray = field(default_factory=lambda: np.array([-π]*6))
    joint_limit_high: np.ndarray = field(default_factory=lambda: np.array([ π]*6))
    max_num_steps: int = 500             # openpi dobot 默认 500
    reward_threshold: np.ndarray = ...
    target_ee_pose: np.ndarray = ...

    def __post_init__(self):
        # 硬约束校验（同 openpi env.py）
        if self.state_mode not in ("joint", "pose"):
            raise ValueError(...)
        if self.state_mode == "pose" and self.action_mode != "cartesian":
            raise ValueError("state_mode='pose' requires action_mode='cartesian'")
```

#### 动作空间（按模式切换）

```python
def _init_action_obs_spaces(self):
    if self.config.action_mode == "cartesian":
        # 8 维 [x,y,z m, qw,qx,qy,qz, gripper 0..1]
        low  = np.array([-1,-1,-1, -1,-1,-1,-1, 0.0])
        high = np.array([ 1, 1, 1,  1, 1, 1, 1, 1.0])
        # 实际限位由 SDK 内部 slew/jump 护栏 + config 限位决定
    else:
        # 7 维 [j1..j6 rad, gripper 0..1]
        low  = np.append(self.config.joint_limit_low,  0.0)
        high = np.append(self.config.joint_limit_high, 1.0)
    self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
```

#### 观测空间（按模式切换）

```python
if self.config.state_mode == "pose":
    state_dim = 8   # [x,y,z m, qw,qx,qy,qz, gripper]
    state_keys = ("ee_pose_state",)   # 8 维
else:
    state_dim = 7   # [j1..j6 rad, gripper]
    state_keys = ("arm_joint_position", "gripper_position")

observation_space = gym.spaces.Dict({
    "state": gym.spaces.Dict({k: gym.spaces.Box(-inf, inf, (d,)) for k, d in ...}),
    "frames": gym.spaces.Dict({
        "cam_high":       Box(0,255,(224,224,3),uint8),   # enable_high_camera 时
        "cam_left_wrist": Box(0,255,(224,224,3),uint8),   # 必备
    }),
})
# 注意：图像 HWC uint8（与 rebot 一致），尺寸 224×224（pi0.5 训练尺寸，非 rebot 的 128）
```

#### `step()`（双模式分支）

```python
def step(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    if not self.config.is_dummy and not self.config.tele_mode:
        # send_action 内部按 action_mode 派发 ServoJ/ServoP
        accepted = self._controller.send_action(
            action, action_mode=self.config.action_mode).wait()
        if not accepted:
            self._servo_rejected_count += 1
            if self._servo_rejected_count % 30 == 0:
                self._logger.warning("Dobot servo rejected 30 consecutive frames")
        else:
            self._servo_rejected_count = 0
    # 限频 30Hz
    # 读状态（pose 模式下控制器返回 prev_state）
    # 算 reward（复用 rebot 三模式：几何/dense/reward_model）
    # terminated / truncated
```

#### `reset()`

```python
def reset(self, **kwargs):
    self._controller.reset_to_pose(self.config.initial_joint_pos).wait()  # 关节空间归位
    if self.config.state_mode == "pose":
        self._controller.reset_pose_tracker().wait()    # 清 prev_state（首帧=自身）
    # 读初始观测
```

#### 安全要点（务必保留 openpi 逻辑）

- `_assert_robot_ready()`：节流 1Hz 调 `controller.is_robot_up()`——E-stop/告警下 ServoJ/ServoP **静默失败**但反馈读取正常，只有模式检查能发现。
- `_servo_rejected_count`：累计连续被拒 servo，每 30 帧告警。

#### 相机

复用 `rlinf/envs/realworld/common/camera/`（`create_camera` + `BaseCamera` 线程化读取），**不用** openpi 自己的 camera 实现。`_get_camera_frames()` center-crop 到正方形 + resize 到 224×224 + BGR→RGB。

#### `get_joint_positions()`（HIL 契约，必须暴露）

```python
def get_joint_positions(self) -> np.ndarray:
    """返回当前关节向量（HIL wrapper 与 collector 都要调用）。
    单位与 action_space 一致：joint 模式返回 7 维 [j1..j6 rad, gripper]，
    cartesian 模式仍返回关节（IK seed 用）。"""
    state = self._controller.get_state().wait()
    return np.append(state.arm_joint_position, state.gripper_position)
```

### 5.4 `tasks/__init__.py` + `pick_and_place.py`

```python
# tasks/__init__.py
from gymnasium.envs.registration import register
from rlinf.envs.realworld.dobot.tasks.pick_and_place import DobotPickAndPlaceEnv
register(id="DobotPickAndPlaceEnv-v1",
         entry_point="rlinf.envs.realworld.dobot.tasks:DobotPickAndPlaceEnv")

# tasks/pick_and_place.py
@dataclass
class DobotPickAndPlaceConfig(DobotRobotConfig):
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.array([0.3,-0.2,0.15, 1,0,0,0]))
    reward_threshold: np.ndarray = field(default_factory=lambda: np.array([0.02,0.02,0.02, 0.2,0.2,0.2]))
    use_dense_reward: bool = True
    max_num_steps: int = 500

class DobotPickAndPlaceEnv(DobotEnv):
    def __init__(self, override_cfg=None, worker_info=None, hardware_info=None,
                 env_idx=0, env_cfg=None, **kwargs):
        super().__init__(config=DobotPickAndPlaceConfig(), ...)
```

### 5.5 硬件注册 `rlinf/scheduler/hardware/robots/dobot.py`

照 `rebot.py` 1:1 写，连接参数换成 dobot：

```python
@Hardware.register()
class DobotRobot(Hardware):
    HW_TYPE = "Dobot"

    @classmethod
    def enumerate(cls, node_rank, configs=None):
        # 过滤 node_rank；RobotAutoConfig.resolve(count_fields=("ip",))
        # 验证 IP 可达（TCP 29999 端口探测，warn-only，disable_validate 可关）
        # 返回 HardwareResource(infos=[DobotHWInfo(...)])

@NodeHardwareConfig.register_hardware_config("Dobot")
@dataclass
class DobotConfig(HardwareConfig):
    ip: str = "192.168.5.1"
    speed: int = 100
    action_mode: str = "joint"
    state_mode: str = "joint"
    gripper_port: str = "/dev/ttyACM0"
    enable_gripper: bool = True
    user_index: int = 0
    tool_index: int = 0
    gripper_closed_deg: float = 0.0
    gripper_open_deg: float = -320.0
    camera_serials: Optional[list[str]] = None
    camera_type: str = "realsense"
    controller_node_rank: Optional[int] = None
    disable_validate: bool = False
```

### 5.6 策略与数据配置（复用 openpi 现有代码）

#### `policies/dobot_policy.py`

直接复用 openpi 的 `DobotInputs/DobotOutputs`（joint 模式）与 `DobotT265Inputs/DobotT265Outputs`（支持 joint + pose）。**唯一调整**：把输入键名从 openpi 的 `"image"/"wrist_image"/"state"` 对齐到 RLinf 观测管线产出的键（参考 `rebot_policy.py` 用 `observation/image`、`observation/state`）。**关键**：pose 模式下必须透传 `prev_state` 和 `rtc_obs`（详见 §六）。

**图像槽映射**（来自 `DobotInputs`）：
- `cam_left_wrist` → `left_wrist_0_rgb`（mask True）
- `cam_high`（可选）→ `base_0_rgb`（mask True；缺失则零填充 mask False）
- `right_wrist_0_rgb` → 零填充 mask False（dobot 无右腕）

#### `dataconfig/dobot_dataconfig.py`

照 `rebot_dataconfig.py` 写，**支持 joint/pose 两套 mask**：

```python
@dataclasses.dataclass(frozen=True)
class DobotDataConfig(DataConfigFactory):
    action_mode: str = "joint"      # "joint" | "cartesian"
    use_pose: bool = False          # 等价：action_mode == "cartesian"
    default_prompt: str | None = None

    def create(self, assets_dirs, model_config) -> DataConfig:
        state_dim = 8 if self.use_pose else 7
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
            # pose 模式额外映射 prev_state
            **({"observation.prev_state": "prev_state"} if self.use_pose else {}),
        })])

        data_transforms = _transforms.Group(
            inputs=[dobot_policy.DobotInputs(state_dim=state_dim)],
            outputs=[dobot_policy.DobotOutputs(action_dim=state_dim)],
        )

        if self.use_pose:
            # pose 模式：DeltaPose / AbsolutePose（SE(3)，需要 prev_state）
            mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaPose(mask)],
                outputs=[_transforms.AbsolutePose(mask)])
        else:
            # joint 模式：DeltaActions / AbsoluteActions（朴素加减）
            mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)])
        ...
```

#### 注册两个 TrainConfig（`dataconfig/__init__.py`）

```python
TrainConfig(
    name="pi05_dobot_joint",
    model=Pi0Config(pi05=True),
    data=DobotDataConfig(use_pose=False,
        assets=AssetsConfig(assets_dir="checkpoints/pi05_dobot_joint",
                            asset_id="dobot_lerobot_joint_data")),
    pytorch_weight_path="checkpoints/pi05_dobot_joint",
),
TrainConfig(
    name="pi05_dobot_pose",
    model=Pi0Config(pi05=True),
    data=DobotDataConfig(use_pose=True,
        assets=AssetsConfig(assets_dir="checkpoints/pi05_dobot_pose",
                            asset_id="dobot_lerobot_pose_data")),
    pytorch_weight_path="checkpoints/pi05_dobot_pose",
),
```

---

## 六、双模式（joint / pose）的关键细节

### 6.1 动作/状态布局

| 模式 | action 维度 | action 布局 | state 维度 | state 布局 |
|---|---|---|---|---|
| joint | 7 | `[j1..j6 rad, gripper 0..1]` | 7 | 同 action |
| pose | 8 | `[x,y,z m, qw,qx,qy,qz, gripper 0..1]` | 8 | 同 action |

夹爪永远是最后一维、永远归一化 `[0,1]`、0=闭/1=开。

### 6.2 `split_follower_action`（控制器内，fail-fast）

```python
def split_follower_action(action, action_mode):
    action = np.asarray(action, dtype=float).reshape(-1)
    if action_mode == "cartesian":
        if action.shape[0] != 8:
            raise ValueError(f"cartesian action must be 8-dim [x,y,z,qw,qx,qy,qz, gripper], got {action.shape[0]}")
        return action[:7], float(action[7])     # arm=7 pose, gripper=scalar
    if action.shape[0] < 7:
        raise ValueError(f"joint action must be 7-dim [j1..j6, gripper], got {action.shape[0]}")
    return action[:6], float(action[6])          # arm=6 joints, gripper=scalar
```

### 6.3 ServoJ vs ServoP（SDK 层）

| 项 | ServoJ（关节） | ServoP（位姿） |
|---|---|---|
| 输入 | 6 关节 **度** | 7 维 `[x,y,z m, qw,qx,qy,qz]` |
| SDK 内转换 | 控制器 rad→deg | SDK 内 m→mm、quat→extrinsic-xyz 欧拉度 |
| slew 限速 | 每轴 clip ±2°/周期 | 平移欧式 clamp 5mm/周期；旋转 rotvec clamp 2°/周期 |
| 跳变拒绝 | max 单轴 30° | 平移 10cm + 旋转 30°（双检查） |
| 调用 | `ServoJ(J1..J6, t=0.04, aheadtime=50, gain=500)` | `ServoP(X,Y,Z,RX,RY,RZ, t=0.04, aheadtime=50, gain=500)`（mm+度） |

**`reset_smoothing()` 必须在 engage 时调用**，清除 `_last_cmd_pose/_last_cmd_joints`，让 slew 从当前位姿重启。

### 6.4 `prev_state` / `PoseStateTracker`（pose 模式必需）

```python
class PoseStateTracker:
    def __init__(self): self._prev = None
    def update(self, state):
        prev = self._prev if self._prev is not None else state   # 首帧=自身
        self._prev = state
        return prev
    def reset(self): self._prev = None
```

- pose 模式下，env 每帧 `obs["state"]` 是 8 维 pose state，`obs["prev_state"]` 是上一帧的 state（首帧=自身）。
- `env.reset()` 必须 `tracker.reset()`。
- `prev_state` 是 `DeltaPose`（训练）和 `AbsolutePose`（推理）的必需输入。缺它则这两个变换变 no-op，模型的相对位姿输出会直接送机器人（非物理）。

### 6.5 DeltaPose / AbsolutePose（SE(3)，不是朴素加减）

- **训练 `DeltaPose`**：`T_delta = T_ref⁻¹ @ T_curr`，其中 state 的 ref=`prev_state`，actions 的 ref=当前 `state`。位置 delta 在**旋转后的参考系**内表达。带半球连续性（`dot(result, reference) < 0` 时取反四元数）。
- **推理 `AbsolutePose`**：`T_curr = T_prev @ T_rel`。
- 辅助：`_pose_to_homogeneous` / `_homogeneous_to_pose`（w-first）、`_invert_homogeneous`。

**移植警示**：绝不能用 `actions + state` 朴素加减实现 pose 模式，方向和旋转坐标系下的平移都会错。

### 6.6 双模式选择

由 YAML `init_params.action_mode` / `init_params.state_mode` 控制，env `__init__` 校验组合合法性。两种模式各自对应一个 `TrainConfig`（`pi05_dobot_joint` / `pi05_dobot_pose`），训练与推理必须用同一种。

---

## 七、HIL 键盘接管（单臂定制）

### 7.1 设计原则（来自 so101 分析，单臂化）

so101 的 HIL 是双臂 + pinocchio IK。Dobot 单臂 + SDK 原生 IK，做三处裁剪：
1. 删除双臂切换（`_ARMS`、`_LEFT_SLICE`、`_RIGHT_SLICE`、`switch_arm_key`、`_active_arm`、per-arm 字典）。
2. `SO101ArmKinematics`（pinocchio）替换为 `DobotKinematics`（封装 SDK 原生 IK），保持 `forward(q)->(pos,quat)` / `inverse(pos,quat,q_init)->q` 同签名。
3. **冻结目标规则仍需保留**：MODEL→ENGAGE 时从当前 FK 种子目标，之后只用键盘 delta 累加，**不每帧重新种子**（否则重力下坠会累积漂移）。

### 7.2 `kinematics.py` — DobotKinematics

```python
class DobotKinematics:
    """封装 Dobot SDK 原生 IK/FK，对 HIL wrapper 暴露与 SO101ArmKinematics 同签名。"""
    def __init__(self, controller_or_sdk, user_index=0, tool_index=0):
        self._robot = controller_or_sdk   # DobotRobot 或 DobotController 句柄

    def forward(self, q_rad) -> tuple[np.ndarray, np.ndarray]:
        """q_rad (6,) -> (pos (3,) m, quat (4,) xyzw)。"""
        # 1. ServoJ 到 q_rad（或读当前位姿）；2. 读 get_tcp_pose -> m+quat wxyz
        # 3. 转 scipy 约定 xyzw 返回

    def inverse(self, pos, quat_xyzw, q_init_rad) -> np.ndarray:
        """(pos (3,) m, quat (4,) xyzw, q_init (6,) rad) -> q (6,) rad。"""
        # 1. quat_xyzw -> wxyz；2. SDK InverseKin(pose_m+quat, seed_deg) -> (reachable, joints_deg)
        # 3. joints_deg -> rad 返回
```

> SDK 已有 `inverse_kin(pose_quat_m, seed_joints_deg) -> (reachable, joints|None)`，直接包装即可。**无需 URDF/pinocchio**。

### 7.3 `dobot_keyboard_intervention.py`

照 `so101_keyboard_intervention.py` 改造，单臂化：

```python
class DobotKeyboardIntervention(gym.ActionWrapper):
    def __init__(self, env,
                 position_delta=0.005,     # m/步
                 rotation_delta=0.05,      # rad/步
                 gripper_delta=0.1,        # 归一化 [0,1]/步（dobot 夹爪是连续值，非 so101 的 [0,100]）
                 toggle_key="h",           # MODEL <-> ENGAGE
                 model_key="m",            # 强制 MODEL
                 done_key="Key.enter",     # 结束本 episode，开下一个
                 quit_keys=("Key.esc",)):  # 保存并退出
        self.listener = KeyboardListener()  # 复用 rlinf/envs/realworld/common/keyboard/
        self._state = "model"
        self._kinematics = DobotKinematics(env.controller)
        self._target_pos = None; self._target_quat = None; self._target_gripper = None
        ...
```

**键位映射**（单臂，删掉 Tab 切换臂）：

| 功能 | 键 | 效果 |
|---|---|---|
| 平移 +X/−X | `w`/`s` | `_target_pos[0] ±= position_delta` |
| 平移 +Y/−Y | `a`/`d` | `_target_pos[1] ±=` |
| 平移 +Z/−Z | `q`/`e` | `_target_pos[2] ±=` |
| 旋转 +X/−X | `i`/`k` | `_rotate_active_target("x", ±rotation_delta)` |
| 旋转 +Y/−Y | `j`/`l` | 同上 y |
| 旋转 +Z/−Z | `u`/`o` | 同上 z |
| 夹爪闭/开 | `,`/`.` | `_target_gripper ∓= gripper_delta`（clamp [0,1]） |
| MODEL↔ENGAGE | `h` | 离散事件 |
| 强制 MODEL | `m` | 离散事件 |
| 结束 episode | `Enter` | `truncated=True` |
| 退出程序 | `Esc` | `info["quit_program"]=True` |

**旋转在末端工具坐标系**（右乘 delta）：`new_rot = current_rot * Rotation.from_euler(axis, delta)`。

**状态机 `action()`**（返回 `(action_out, replaced)`）：
- 纯 MODEL 快速路径：直接返回 `(model_action, False)`。
- ENGAGE（或离开 ENGAGE 的过渡帧）：读当前关节 → `_update_active_target` 应用 held-key delta → IK 求 q → `[q, target_gripper]` → clip → 返回 `(action_out, True)`。
- MODEL→ENGAGE 首帧：从当前 FK 种子 `_target_pos/_target_quat/_target_gripper`。
- `was_engage` 快照：即使在 ENGAGE 帧按了 `h`/`m`，本帧仍执行人 IK 命令，下一帧才切 MODEL。

**`step()` 注入 info**（与 so101 完全一致，`CollectEpisode` 据此记录）：
```python
info["model_action"] = model_action              # 始终
if replaced:
    info["intervene_action"] = new_action
    info["intervene_flag"] = np.ones(1, dtype=bool)
info["hil_state"] = self._state                  # 始终（collector 读）
```

### 7.4 `apply_dobot_wrappers`

在 `apply.py` 加：
```python
def apply_dobot_wrappers(env, cfg):
    config = env.get_wrapper_attr("config")
    use_keyboard = cfg.get("use_keyboard_intervention", False)
    active_in_dummy = not config.is_dummy or cfg.get("use_intervention_in_dummy", False)
    if use_keyboard and active_in_dummy:
        kcfg = cfg.get("keyboard_intervention", {})
        env = DobotKeyboardIntervention(
            env,
            position_delta=float(kcfg.get("position_delta", 0.005)),
            rotation_delta=float(kcfg.get("rotation_delta", 0.05)),
            gripper_delta=float(kcfg.get("gripper_delta", 0.1)),
            toggle_key=kcfg.get("toggle_key", "h"),
            model_key=kcfg.get("model_key", "m"),
            done_key=kcfg.get("done_key", "Key.enter"),
            quit_keys=tuple(kcfg.get("quit_keys", ("Key.esc",))),
        )
    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))
    return env
```

> 修复了 so101 的一个小瑕疵：`done_key` 在此版本可通过 YAML 覆盖。

### 7.5 `DobotHILCollector`（`examples/embodiment/collect_dobot_hil_data.py）

照 `collect_so101_hil_data.py` 改类名 + env 配置即可，**核心队列/hold-step/purge 逻辑 verbatim 转移**：

- `self._action_queue`（chunk, action_dim）：策略预测 num_action_chunks 步，每步弹一个，空则重新推理。
- `self._pending_model_hold_step`：ENGAGE→MODEL 过渡时，发一帧 `current_q`（保持），让 `env.step()` 产出反映介入后位姿的干净观测，下一帧再调策略。
- ENGAGE 期间**不动队列**（传零向量，wrapper 会覆盖），保留策略先前的计划不被立即重放。
- HIL 状态变化时清空 `_action_queue`，下个 MODEL 帧从干净观测重新推理。

`CollectEpisode` 完全 robot-agnostic，无需改动，只需传 `robot_type="dobot"` 与真实 `fps`。记录字段：`actions`（实际执行）+ `intervene_flag`（人帧标记）+ `model_action`（策略原预测，custom feature）。

### 7.6 `KeyboardListener` 复用

直接用 `rlinf/envs/realworld/common/keyboard/keyboard_listener.py`（Linux evdev，守护线程，`get_key()` 返回 held key、`pop_pressed_keys()` 返回 edge presses）。跨平台开发可写 pynput 后备，但生产用 evdev 即可。

---

## 八、YAML 配置

### 8.1 `env/realworld_dobot.yaml`（基础 env）

```yaml
env_type: realworld
total_num_envs: 1
auto_reset: False
ignore_terminations: False
main_image_key: cam_left_wrist          # 不是 rebot 的 wrist_1
max_episode_steps: 500
video_cfg: {save_video: False, info_on_video: True, video_base_dir: ${runner.logger.log_path}/video/eval}

init_params:
  id: "DobotPickAndPlaceEnv-v1"
  ip: "192.168.5.1"
  speed: 100
  action_mode: "joint"                  # "joint" | "cartesian"
  state_mode: "joint"                   # "joint" | "pose"（pose ⟹ action_mode="cartesian"）
  user_index: 0
  tool_index: 0
  gripper_port: "/dev/ttyACM0"
  enable_gripper: true
  gripper_closed_deg: 0.0
  gripper_open_deg: -320.0
  camera_serials: ["<wrist_cam>", "<high_cam>"]
  camera_type: "realsense"
  enable_high_camera: true
  is_dummy: false
  task_description: "pick up the plug and insert into the socket"
  max_num_steps: 500
  step_frequency: 30.0                  # dobot 30Hz
  initial_joint_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]   # [j1..j6 rad, gripper norm]
```

### 8.2 训练 YAML（关键覆盖）

```yaml
defaults:
  - env/realworld_dobot@env.train
  - env/realworld_dobot@env.eval
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer

actor.model:
  model_path: checkpoints/pi05_dobot_joint        # 或 pi05_dobot_pose
  model_type: "openpi"
  add_value_head: True
  action_dim: 7                                    # joint: 7；pose: 8
  num_action_chunks: 10                            # dobot action_horizon=15，rebot 用 5
  num_steps: 4
  openpi:
    config_name: "pi05_dobot_joint"                # 或 pi05_dobot_pose
    num_images_in_input: 2                          # cam_left_wrist + cam_high
    noise_method: "flow_noise"
    action_horizon: 15

cluster:
  num_nodes: 1
  component_placement:
    actor:   {node_group: local_gpu, placement: 0}
    rollout: {node_group: local_gpu, placement: 0}
    env:     {node_group: dobot, placement: 0}
  node_groups:
    - label: local_gpu
      node_ranks: 0
    - label: dobot
      node_ranks: 0
      hardware:
        type: Dobot                                # 匹配 HW_TYPE
        configs:
          - ip: "192.168.5.1"
            action_mode: "joint"
            state_mode: "joint"
            gripper_port: "/dev/ttyACM0"
            camera_serials: ["..."]
            node_rank: 0
```

### 8.3 HIL 采集 YAML（`dobot_hil_collect.yaml`，仿 `so101_hil_collect.yaml`）

```yaml
defaults:
  - env/realworld_dobot@env.eval
  - model/pi0_5@actor.model

runner:
  num_data_episodes: 10000                       # 操作员用 Esc 停止
use_dummy_policy: False

env.eval:
  ignore_terminations: False
  auto_reset: False
  total_num_envs: 1
  max_episode_steps: 10000
  main_image_key: cam_left_wrist
  override_cfg:
    is_dummy: ${use_dummy_policy}
    tele_mode: false
    use_dense_reward: False
    use_reward_model: False
    max_num_steps: 10000
    manual_episode_control_only: True            # wrapper 拥有 episode 结束
    step_frequency: ${env.eval.data_collection.fps}

  use_keyboard_intervention: True
  use_intervention_in_dummy: True
  keyboard_intervention:
    position_delta: 0.005
    rotation_delta: 0.05
    gripper_delta: 0.1                            # 归一化 [0,1]/步
    toggle_key: "h"
    model_key: "m"
    done_key: "Key.enter"
    quit_keys: ["Key.esc"]

  data_collection:
    enabled: True
    save_dir: ${project_path:logs/dobot_hil_collect/collected_data}
    export_format: "lerobot"
    only_success: False
    robot_type: "dobot"
    fps: 30                                       # 也驱动 step_frequency
    finalize_interval: 100
    resume: False

actor.model:
  model_path: "checkpoints/pi05_dobot_joint"
  model_type: "openpi"
  num_action_chunks: 10
  action_dim: 7                                   # joint: 7；pose: 8
  openpi: {config_name: "pi05_dobot_joint", train_expert_only: True}
```

---

## 九、可直接复用 vs 需改写

| 组件 | 处理方式 |
|---|---|
| `dobot_control/` SDK（TCP+几何） | **git submodule**（cnb.cool/THU-HiGroup/dobot_zhiyu） |
| `DamiaoGripper`（达妙夹爪） | 从 openpi 拷贝（motorbridge 依赖），放 `dobot/robot/gripper.py` |
| `action_split.py`（纯函数） | 从 openpi 原样拷贝 |
| `pose_obs.py`（`pose_state`/`PoseStateTracker`） | 从 openpi 原样拷贝 |
| `dobot_policy.py`（openpi `DobotInputs/Outputs`） | 拷贝 + 改输入键名对齐 RLinf |
| `LeRobotDobotT265DataConfig`（openpi） | 改写为 RLinf 的 `DobotDataConfig(DataConfigFactory)` |
| `DobotEnvironment`（openpi `openpi_client.Environment`） | **重写**为 `DobotEnv(gym.Env)`，照 `rebot_env.py` + 双模式 |
| `DobotController`（openpi 普通类） | **包装**为 `DobotController(Worker)`，加 `launch_controller` + `.wait()` |
| 相机 | 弃用 openpi 的，**改用** `rlinf/envs/realworld/common/camera/` |
| reward | **复用** rebot 三模式 + `EmbodiedRewardWorker.launch_for_realworld()` |
| 数据采集 `CollectEpisode` | **完全复用**（robot-agnostic），传 `robot_type="dobot"` |
| HIL collector | 照 `collect_so101_hil_data.py` 改名 + env 配置，队列逻辑 verbatim |
| `KeyboardListener` | **直接复用** `common/keyboard/`（Linux evdev） |
| HIL keyboard wrapper | 照 `so101_keyboard_intervention.py` **单臂化** + 去 pinocchio + 用 `DobotKinematics` |
| IK | **不引入 pinocchio**，用 Dobot SDK 原生 IK 封装 `DobotKinematics` |

---

## 十、实施顺序（建议）

### 阶段 0：SDK 落位（半天）
1. `git submodule add`，确认 `third_party/dobot_zhiyu/src/dobot_control/` 可 import。
2. `verify_env.py`：独立脚本跑通"连接→使能→engage→读关节+位姿→ServoJ 一个关节→ServoP 一个位姿→归位→close"。

> **前提（已确认）**：openpi 的 pose 模式（`pi05_dobot_t265_pose_train`）已在 Dobot 真机跑通且姿态准确，即 `DOBOT_EULER_SEQ="xyz"`（SDK 内部 mm+欧拉角 ↔ 上层 m+四元数 的旋转顺序约定）已被端到端验证正确。RLinf 上层位姿统一用 m + 四元数 `[qw,qx,qy,qz]`（w-first），欧拉角只存在于 SDK 内部翻译层，无需在 RLinf 侧重新验证。`dobot_geometry.py` 文件头"待真机验证"注释为历史遗留，可视为已知正确常量。

### 阶段 1：Controller Worker 化（1-2 天）
1. 写 `dobot_controller.py`，包装 SDK + 夹爪为 `Worker`。
2. 实现全部 RPC API（`is_robot_up` / `get_state` / `send_action` 双模式 / `move_joints` / `reset_to_pose` / `engage` / `open/close_gripper` / `tare_ft_sensor` / `hold_current_position` / `return_home_then_close` / `close`）。
3. 单测每个方法的 Future 语义；真机验证双模式 servo 与护栏拒绝计数。

### 阶段 2：Env + 任务注册（1-2 天）
1. 写 `dobot_env.py`（双模式 action/obs space、step、reset、reward、相机、`get_joint_positions`）。
2. 写 `dobot_robot_state.py`、`tasks/`。
3. `is_dummy=True` 在无硬件时跑通 `gym.make("DobotPickAndPlaceEnv-v1")` 的 reset/step。
4. 接线：`realworld/__init__.py` 导出 + gym register。

### 阶段 3：硬件注册（半天）
1. 写 `dobot.py`（`DobotRobot(Hardware)` + `DobotConfig`）。
2. 导出接线；YAML `type: Dobot` 能解析，`enumerate` 正确返回 `HardwareResource`。

### 阶段 4：策略/数据配置（1 天）
1. 搬 `dobot_policy.py`，改键名。
2. 写 `DobotDataConfig`（joint/pose 两套 mask）。
3. 注册 `pi05_dobot_joint` / `pi05_dobot_pose`。

### 阶段 5：端到端训练（1 天）
1. 写训练 YAML；先 `is_dummy=True` 跑通训练 loop。
2. 上真机：先 joint 模式（简单），再 pose 模式（验证 `prev_state`/DeltaPose 链路）。

### 阶段 6：HIL 数据采集（1-2 天）
1. 写 `DobotKinematics`（SDK 原生 IK 封装）。
2. 写 `dobot_keyboard_intervention.py`（单臂化）+ `apply_dobot_wrappers`。
3. 写 `collect_dobot_hil_data.py` + `dobot_hil_collect.yaml`。
4. 真机验证：MODEL/ENGAGE 切换、键盘 EE 控制、ENGAGE→MODEL hold step、`intervene_flag`/`model_action` 入库。

### 阶段 7：reward model（可选，0.5 天）
1. 仿 `rebot_reward_training.yaml` 写 `dobot_reward_training.yaml`。
2. 仿 `preprocess_rebot_lerobot.py` 写 dobot 版预处理。

---

## 十一、验证清单

实施完成后逐项核验：

- [ ] `git submodule status` 显示 `dobot_zhiyu` 指向 cnb.cool 上游。
- [ ] `verify_env.py` 真机跑通连接/使能/双模式 servo/归位。
- [ ] `is_dummy=True` 下 `gym.make("DobotPickAndPlaceEnv-v1")` reset/step 无错。
- [ ] joint 模式：训练 loop 跑通，policy 输出 7 维，`AbsoluteActions` 正确还原。
- [ ] pose 模式：训练 loop 跑通，policy 输出 8 维，`prev_state`/`DeltaPose`/`AbsolutePose` 链路正确（首帧 `prev_state==state`）。
- [ ] `type: Dobot` YAML 解析，`enumerate` 返回正确 `HardwareResource`。
- [ ] 单节点 + 双节点（云 GPU + 本地机器人）配置均可启动。
- [ ] HIL：`h` 切换 MODEL/ENGAGE，键盘 EE 控制（平移/旋转/夹爪），`Enter` 结束 episode，`Esc` 保存退出。
- [ ] HIL：ENGAGE→MODEL 有 hold step；`intervene_flag`/`model_action` 正确入库。
- [ ] 安全：E-stop 下 `_assert_robot_ready` 报错；servo 连续被拒 30 帧告警；限速/跳变护栏生效。
- [ ] 单位边界：控制器对外=弧度/m/w-first quat/归一化夹爪，SDK 内转换正确。
- [ ] 双模式各自端到端跑通（joint 与 pose 都验证）。

---

## 十二、关键风险与对策

| 风险 | 对策 |
|---|---|
| pose 模式 `prev_state` 链路复杂，易漏 | 严格测试首帧=自身、reset 清 tracker、DeltaPose/AbsolutePose 不 no-op |
| SDK 原生 IK 精度/连续性不足，HIL 卡顿 | `DobotKinematics.inverse` 用当前关节作 seed（delta 小，收敛快）；必要时加阻尼最小二乘 |
| 夹爪 FORCE_POS 模式偶发不响应 | 每次命令前 `ensure_mode`（3 次重试），复用 openpi `DamiaoGripper` 逻辑 |
| 双模式 action_dim 切换易训练/推理不一致 | 用独立 `TrainConfig`（`pi05_dobot_joint` / `pi05_dobot_pose`），YAML `config_name` 与 `action_mode` 强绑定 |
| 控制器 Worker 化后延迟影响 30Hz | 单次反馈读取（`read_state_from_feedback`）；必要时把控制器与 env 同节点（`controller_node_rank` 同 env） |
| submodule 在 CI/新环境未初始化 | 文档强调 `git clone --recursive`；`requirements/install.sh` 加 `git submodule update --init` |

---

## 附录 A：关键文件路径（openpi 参考）

- 策略变换：`/home/zylab/project/openpi/src/openpi/policies/dobot_policy.py`
- 训练配置：`/home/zylab/project/openpi/src/openpi/training/config.py`（582-710 数据配置，1453-1560 训练配置）
- Env：`/home/zylab/project/openpi/examples/dobot_clean/env.py`
- 控制器：`/home/zylab/project/openpi/examples/dobot_clean/robot/controller.py`
- 夹爪：`/home/zylab/project/openpi/examples/dobot_clean/robot/gripper.py`
- 动作切分：`/home/zylab/project/openpi/examples/dobot_clean/robot/action_split.py`
- SDK 路径注入：`/home/zylab/project/openpi/examples/dobot_clean/robot/__init__.py`
- 位姿辅助：`/home/zylab/project/openpi/examples/dobot_clean/inference/pose_obs.py`
- SDK 机器人层：`/home/zylab/project/openpi/third_party/dobot_zhiyu/src/dobot_control/dobot_robot.py`
- SDK 几何：`/home/zylab/project/openpi/third_party/dobot_zhiyu/src/dobot_control/dobot_geometry.py`

## 附录 B：关键文件路径（RLinf 参考）

- rebot env：`/home/zylab/project/RLinf/rlinf/envs/realworld/rebot/rebot_env.py`
- rebot 控制器：`/home/zylab/project/RLinf/rlinf/envs/realworld/rebot/rebot_controller.py`
- rebot 硬件：`/home/zylab/project/RLinf/rlinf/scheduler/hardware/robots/rebot.py`
- rebot 策略：`/home/zylab/project/RLinf/rlinf/models/embodiment/openpi/policies/rebot_policy.py`
- rebot 数据配置：`/home/zylab/project/RLinf/rlinf/models/embodiment/openpi/dataconfig/rebot_dataconfig.py`
- rebot env YAML：`/home/zylab/project/RLinf/examples/embodiment/config/env/realworld_rebot.yaml`
- so101 HIL wrapper：`/home/zylab/project/RLinf/rlinf/envs/realworld/common/wrappers/so101_keyboard_intervention.py`
- so101 运动学：`/home/zylab/project/RLinf/rlinf/envs/realworld/so101/kinematics.py`
- so101 HIL collector：`/home/zylab/project/RLinf/examples/embodiment/collect_so101_hil_data.py`
- so101 HIL YAML：`/home/zylab/project/RLinf/examples/embodiment/config/so101_hil_collect.yaml`
- 键盘监听：`/home/zylab/project/RLinf/rlinf/envs/realworld/common/keyboard/keyboard_listener.py`
- 数据集录制：`/home/zylab/project/RLinf/rlinf/envs/wrappers/collect_episode.py`
- wrapper 接线：`/home/zylab/project/RLinf/rlinf/envs/realworld/common/wrappers/apply.py`
