# 云端训练 + 本地真机在线 RL 实施规划（pi0.5）

## 约束与前提

| 项目 | 现状 |
|------|------|
| 训练节点 | 云端 GPU 服务器（无公网 IP） |
| 真机节点 | 本地，RTX 5090（RLinf 当前依赖不支持 Blackwell） |
| 网络 | 飞连 VPN，本地→云端可达（单向），云端→本地不可达 |
| 防火墙 | 云端仅对本地开放端口 **12345**（自定义） |
| 模型 | 已有 SFT 后的 pi0.5 checkpoint |
| 任务 | 自定义真机任务，无目标位姿，由奖励模型判断成功/失败 |
| 真机控制 | Python 实现的自定义控制接口 |

### 端口需求：1 个端口

| 方向 | 端口 | 用途 |
|------|------|------|
| 本地 → 云端 | **12345** | Ray GCS（集群控制服务）——本地 worker 连接、任务调度、心跳、Channel 通信全部走此端口的 gRPC 长连接 |

不需要开放其他端口的原因：
- Dashboard（8265）关闭：`--include-dashboard=false`
- RLinf Channel 通信：走已建立的 gRPC 连接，不额外占用端口
- actor ↔ rollout 的 PyTorch 分布式 weight sync：两者都在云端同一节点，走 **localhost**，不穿越防火墙
- 本地 env worker 只通过 Ray 内部 gRPC 与云端 channel actor 通信，不参与任何 `torch.distributed` 进程组

### Blackwell 约束说明

RLinf 当前依赖的 PyTorch 版本不支持 Blackwell 架构（RTX 5090），因此本地 GPU 无法用于训练或推理。**所有 GPU 计算（actor 训练 + pi0.5 推理）放在云端**，本地只运行 CPU 型 env worker（机器人控制 + 相机采集）。

---

## 1. 通信拓扑分析

### 1.1 为什么 "单向 VPN + 所有计算放云端" 可以工作

RLinf 的 Channel 通信默认使用 `distributed=False, node_rank=0`，即所有 channel actor 都部署在 **云端 head 节点（rank 0）** 上：

```
┌──────────────────────────────────────────────────┐
│  云端 (rank 0)                                    │
│  ┌────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │ Actor  │  │ Rollout │  │ Channel Actors  │   │
│  │ (训练) │  │ (推理)   │  │ Env/Rollout/    │   │
│  │        │  │         │  │ Actor/Reward    │   │
│  └────────┘  └─────────┘  └────────┬────────┘   │
│       │            │               │             │
│       └──weight───┘               │             │
│       sync (本地)                  │             │
└──────────────────────────────┬────┴─────────────┘
                               │
                     飞连 VPN (单向: local→cloud:12345)
                               │
┌──────────────────────────────┴──────────────────┐
│  本地 (rank 1)                                    │
│  ┌──────────────────────────────────────────┐   │
│  │  Env Worker (CPU)                        │   │
│  │  - 真机 Python 控制接口                   │   │
│  │  - 相机采集                               │   │
│  │  - 奖励模型推理 (可选, 放 cloud 或 local)  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

各通道的通信方向：

| 通信路径 | 方向 | 通过 | 可达性 |
|---------|------|------|--------|
| rollout → channel (actions) | cloud local | Ray local rpc | ✓ |
| channel → env worker (actions) | cloud → local | Ray: remote actor call 到 channel, local 从 channel pull | ✓ (local→cloud pull) |
| env worker → channel (obs/trajectories) | local → cloud | Ray: local 写 channel 对象 | ✓ |
| channel → rollout (obs) | cloud local | Ray local rpc | ✓ |
| channel → actor (trajectories) | cloud local | Ray local rpc | ✓ |
| actor → rollout (weight sync) | cloud local | Gloo broadcast 同节点 | ✓ |

**结论：单向 VPN 可行。无需反向隧道或全互联 overlay。**

---

## 2. 网络与 Ray 集群搭建

### 2.1 确认飞连 VPN 连通性

```bash
# 在本地节点上测试能否访问云端
ping <cloud_feilian_ip>
```

本地必须能访问云端的飞连 VPN IP。

### 2.2 启动 Ray

在云端（rank 0，head）：

```bash
# 1. Source RLinf 环境
source <rlinf_venv>/bin/activate

# 2. 设置环境变量
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<feilian_interface>   # feilian网卡名，如 utun 或 et0
# 可选：如果飞连分配的是IPv6地址
# export RAY_USE_IPV6=1

# 3. 启动 Ray head（端口 12345，关闭 Dashboard）
ray start --head --port=12345 --include-dashboard=false --node-ip-address=<cloud_feilian_ip>

# 4. 验证
ray status
```

在本地（rank 1，worker）：

```bash
# 1. Source RLinf 环境
source <rlinf_venv>/bin/activate

# 2. 确保 ray stop 先停止旧实例
ray stop

# 3. 设置环境变量
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=<feilian_interface>

# 4. Source 机器人 setup（如有 ROS、自定义控制库等）
# 参考 ray_utils/realworld/setup_before_ray.sh 修改并 source
source ray_utils/realworld/setup_before_ray.sh

# 5. 启动 Ray worker（连接到云端端口 12345）
ray start --address='<cloud_feilian_ip>:12345'

# 6. 验证：应显示 2 nodes
ray status
```

### 2.3 本地节点安装（最小依赖）

本地真机节点只需要 env worker，无需 GPU 依赖、模型包、训练库。使用 `install_local.sh --cpu-only`：

```bash
# 在仓库根目录执行
bash requirements/install_local.sh --cpu-only --env franka

# 安装完成后激活环境
source .venv/bin/activate
```

该模式会：
- 强制安装 CPU 版 torch（`UV_TORCH_BACKEND=cpu`），避免 CUDA wheel
- 跳过 flash-attn、apex、openpi/vla 模型包等训练组件
- 只安装 `dependencies` 核心 + `realworld-env` extra（gymnasium、opencv、psutil 等）
- 通过 `--env franka` 额外安装机器人控制包（pyrealsense2、pyspacemouse 等）

### 2.4 代码同步

云端和本地需要相同的 RLinf 代码副本（包括自定义环境代码）：

```bash
# 方式1：在 head 节点运行训练脚本前执行
export RLINF_CODE_WORKING_DIR=auto

# 方式2：手动在两台机器上 git pull 保持同步
```

详细说明见 `docs/source-zh/rst_source/tutorials/usage/multi_node.rst`。

---

## 3. 组件放置配置

```yaml
cluster:
  num_nodes: 2
  component_placement:
    actor:
      node_group: cloud
      placement: 0-3          # 云端 GPU 数量，根据实际调整
    rollout:
      node_group: cloud
      placement: 0            # rollout 放云端，与 actor 同节点
    env:
      node_group: robot
      placement: 0            # env 放本地
    reward:
      node_group: cloud       # 奖励模型放云端（如本地无 GPU）
      placement: 0
  node_groups:
    - label: cloud
      node_ranks: 0
    - label: robot
      node_ranks: 1
      # 方案A：自定义硬件类型（推荐）
      # hardware:
      #   type: MyRobot
      #   configs:
      #     - node_rank: 1
      # 方案B：暂时忽略硬件检测（快速原型）
      ignore_hardware: True
```

---

## 4. 自定义真机环境实现

### 4.1 环境结构

创建以下文件：

```text
rlinf/envs/realworld/my_robot/
├── __init__.py
├── my_robot_env.py          # 主环境类
├── my_robot_config.py       # 配置 dataclass
└── tasks/
    ├── __init__.py          # 任务注册（gymnasium register）
    └── my_task_env.py       # 具体任务子类（可选）
```

### 4.2 my_robot_env.py 接口规范

```python
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from rlinf.scheduler import WorkerInfo


@dataclass
class MyRobotConfig:
    """真机配置——所有 override_cfg 中的 key-value 会被注入"""

    task_description: str = "default task description"
    is_dummy: bool = False
    max_num_steps: int = 100
    # 添加你的自定义参数...

    def __post_init__(self):
        # 配置校验与后处理
        pass


class MyRobotEnv(gym.Env):
    """自定义真机环境，必须满足以下接口。

    构造函数签名：
        __init__(
            self,
            override_cfg: dict,          # 来自 YAML env.train.override_cfg
            worker_info: Optional[WorkerInfo] = None,
            hardware_info: Optional[Any] = None,    # 自定义 MyRobotHWInfo
            env_idx: int = 0,
            env_cfg: Any = None,
        )
    """

    def __init__(self, override_cfg, worker_info=None,
                 hardware_info=None, env_idx=0, env_cfg=None):
        super().__init__()
        self.config = MyRobotConfig(**override_cfg)
        self.hardware_info = hardware_info
        self.env_idx = env_idx

        # ---- 初始化真机控制接口 ----
        # self.robot = YourRobotController(...)
        # self.cameras = YourCameraCapture(...)
        # ---- 初始化奖励模型（可选，或使用独立 reward worker） ----
        # self.reward_model = YourRewardModel(...)

        # ---- 定义 observation/action space ----
        # state 必须是 gym.spaces.Dict，包含各状态分量的 Box
        # frame keys 中必须包含 main_image_key
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "gripper": gym.spaces.Box(0.0, 1.0, shape=(1,)),
                # ... 其他状态量
            }),
            "frames": gym.spaces.Dict({
                "wrist_1": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "wrist_2": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                # ... 其他相机
            }),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))

    def reset(self, *, seed=None, options=None):
        """返回 (obs, info)。
        obs 必须是 {"state": {...}, "frames": {...}} 的字典。
        """
        # 复位机器人到初始位姿
        # self.robot.go_to_rest()
        # 等待状态稳定
        # ...
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """执行动作，返回 (obs, reward, terminated, truncated, info)。

        action 形状: (action_dim,) 的一维 numpy 数组。
        """
        # 1. 执行动作
        # self.robot.send_action(action)

        # 2. 获取新观测
        obs = self._get_obs()

        # 3. 计算奖励
        reward = self._compute_reward(obs)
        # 或由独立 reward worker 计算（在 _compute_reward 中调用模型推理）

        # 4. 判断终止
        terminated = self._check_termination()
        truncated = self._check_truncation()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """组装 observation dict"""
        # state = self.robot.get_state()        # 自定义接口
        # frames = self.cameras.get_frames()    # 自定义接口
        return {
            "state": {
                "tcp_pose": state["tcp_pose"],
                # ...
            },
            "frames": {
                "wrist_1": frames["wrist_camera"],
                # ...
            },
        }

    def _compute_reward(self, obs):
        """奖励逻辑"""
        # 方式1：调用独立奖励模型
        # reward = self.reward_model.predict(obs)

        # 方式2：环境内简单启发式
        # ...

        # 方式3：返回 dummy reward（训练初期或调试用）
        return 0.0

    def _check_termination(self):
        """成功/失败检测"""
        # 方式1：奖励模型输出成功标志
        # 方式2：环境内条件判断
        return False

    def _check_truncation(self):
        """超时截断"""
        return self.step_count >= self.config.max_num_steps

    @property
    def task_description(self) -> str:
        """必须与 SFT 训练时的 prompt 一致"""
        return self.config.task_description
```

### 4.3 注册环境

```python
# rlinf/envs/realworld/my_robot/tasks/__init__.py
from gymnasium.envs.registration import register
from rlinf.envs.realworld.my_robot.my_robot_env import MyRobotEnv

# 方式1：直接注册类
register(
    id="MyRobotEnv-v1",
    entry_point="rlinf.envs.realworld.my_robot.my_robot_env:MyRobotEnv",
)

# 方式2：注册工厂函数（如需加 wrapper）
# def make_my_robot_env(override_cfg, worker_info, hardware_info, env_idx, env_cfg):
#     env = MyRobotEnv(override_cfg, worker_info, hardware_info, env_idx, env_cfg)
#     return env
#
# register(
#     id="MyRobotEnv-v1",
#     entry_point="rlinf.envs.realworld.my_robot.tasks:make_my_robot_env",
# )
```

### 4.4 在 realworld/__init__.py 中导入

```python
# rlinf/envs/realworld/__init__.py
# 在现有 import 之后添加：
from .my_robot import tasks as my_robot_tasks
```

### 4.5 真机控制代码位置

将你的自定义 Python 控制代码放在任意 Python 可 import 的位置，例如：

- `rlinf/envs/realworld/my_robot/controller.py`（推荐，与 env 一起管理）
- 本地系统的某个 pip 包（需要在 `setup_before_ray.sh` 中确保 PYTHONPATH 包含）

---

## 5. pi0.5 SFT Checkpoint 接入

### 5.1 Checkpoint 目录结构

确保以下路径存在：

```text
<SFT_CKPT_PATH>/
├── actor/
│   └── model_state_dict/
│       └── full_weights.pt       # 权重文件
├── <repo_id>/                     # SFT 时使用的 LeRobot repo id
│   └── norm_stats.json           # 状态/动作归一化统计
```

如果缺少 `norm_stats.json`：

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_root
python toolkits/lerobot/calculate_norm_stats.py \
    --config-name <你的_pi05_config_name> \
    --repo-id <repo_id>
```

然后将生成的 `<repo_id>/norm_stats.json` 复制到 checkpoint 目录。

### 5.2 模型 YAML 配置项

```yaml
actor:
  model:
    model_type: "openpi"
    model_path: "<SFT_CKPT_PATH>"           # 云端路径
    precision: null
    add_value_head: True                     # PPO 必须开启
    num_action_chunks: 8                     # 一次推理生成的动作帧数
    action_dim: 7                            # 单步动作维度
    num_steps: 4                             # 扩散去噪步数
    openpi:
      config_name: "<SFT_config_name>"       # 若与 pi05_franka 匹配则复用
      value_after_vlm: True                  # pi0.5 必须 True
      noise_method: "flow_noise"
      joint_logprob: True
      action_chunk: ${actor.model.num_action_chunks}
      action_env_dim: ${actor.model.action_dim}
      num_steps: ${actor.model.num_steps}
      train_expert_only: False               # 在线 RL 必须 False
      detach_critic_input: True
    openpi_data:
      repo_id: "<SFT_repo_id>"               # 与 norm_stats 路径匹配

### 5.3 新增自定义 TrainConfig（如需要）

如果你的真机动作/状态布局与现有 config（pi05_franka, pi05_libero 等）不匹配，需要新增配置。

1. 创建 `rlinf/models/embodiment/openpi/dataconfig/my_robot_dataconfig.py`

   参考现有示例（如 `dual_franka_tcp_rot6d_dataconfig.py`），定义：
   - `RepackTransform`（observation → model input 的映射）
   - `MyRobotInputs` / `MyRobotOutputs`（input/output transform）

2. 在 `rlinf/models/embodiment/openpi/dataconfig/__init__.py` 注册：

   ```python
   TrainConfig(
       name="pi05_my_robot",
       model=pi0_config.Pi0Config(
           pi05=True,
           action_horizon=20,          # 总动作预测长度
           discrete_state_input=False,
       ),
       data=MyRobotDataConfig(
           repo_id="",
           base_config=DataConfig(prompt_from_task=True),
           assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_base/assets"),
       ),
       pytorch_weight_path="checkpoints/torch/pi05_base",
   )
   ```

3. 在 `rlinf/models/embodiment/openpi/policies/` 创建对应的 policy transform

4. YAML 中设置 `config_name: "pi05_my_robot"`

---

## 6. 奖励模型

### 6.1 部署方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **环境内嵌** | 在 `MyRobotEnv._compute_reward()` 中直接加载奖励模型推理 | 奖励模型较小（可 CPU 运行） |
| **独立 reward worker** | 在 YAML 中配置 `reward.use_reward_model: True`，训练脚本自动创建 `EmbodiedRewardWorker` | 奖励模型需要 GPU |

### 6.2 独立 reward worker 配置

```yaml
reward:
  use_reward_model: True
  standalone_realworld: False
  group_name: "RewardGroup"
  # reward_model 相关配置...
  model_path: "<reward_model_path>"
```

`train_async.py` 会在 `use_reward_model=True` 且 `standalone_realworld=False` 时创建 reward worker 组。

---

## 7. 完整 YAML 配置

基于 `examples/embodiment/config/realworld_peginsertion_async_ppo_pi05.yaml` 修改：

```yaml
defaults:
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

cluster:
  num_nodes: 2
  component_placement:
    actor:
      node_group: cloud
      placement: 0            # 云端 GPU 索引，多 GPU 时写 0-3
    rollout:
      node_group: cloud
      placement: 0
    env:
      node_group: robot
      placement: 0
    reward:
      node_group: cloud       # 奖励模型放云端
      placement: 0
  node_groups:
    - label: cloud
      node_ranks: 0
    - label: robot
      node_ranks: 1
      ignore_hardware: True   # 暂时跳过硬件检测

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: my-robot-pi05
    experiment_name: "async_ppo"
    logger_backends: ["tensorboard"]

  max_epochs: 8000
  max_steps: -1
  only_eval: False
  val_check_interval: -1
  save_interval: 50

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0
  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level
  update_epoch: 2
  adv_type: gae
  loss_type: decoupled_actor_critic
  loss_agg_func: "token-mean"
  kl_beta: 0.0
  entropy_bonus: 0.005
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  staleness_threshold: 1

env:
  group_name: "EnvGroup"
  train:
    env_type: realworld
    rollout_epoch: 20
    ignore_terminations: True
    total_num_envs: 1
    group_size: 1
    auto_reset: True
    main_image_key: wrist_1
    max_episode_steps: 100
    max_steps_per_rollout_epoch: 300          # 必须是 num_action_chunks 的整数倍
    init_params:
      id: "MyRobotEnv-v1"                     # 你注册的环境 ID
    override_cfg:
      task_description: "<与 SFT 一致的 prompt>"
      is_dummy: False
      max_num_steps: 100
      # 你的自定义参数...
    video_cfg:
      save_video: False
  eval:
    env_type: realworld
    rollout_epoch: 1
    total_num_envs: 1
    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400          # 必须是 num_action_chunks 的整数倍
    init_params:
      id: "MyRobotEnv-v1"
    override_cfg:
      task_description: "<与 SFT 一致的 prompt>"
      is_dummy: False
      max_num_steps: 400

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 80
  global_batch_size: 400
  seed: 0
  enable_offload: False
  model:
    model_path: "<SFT_CKPT_PATH>"
    model_type: "openpi"
    precision: null
    add_value_head: True
    num_action_chunks: 8
    action_dim: 7
    num_steps: 4
    openpi:
      config_name: "<pi05_config_name>"
      value_after_vlm: True
      noise_method: "flow_noise"
      joint_logprob: True
      action_chunk: ${actor.model.num_action_chunks}
      action_env_dim: ${actor.model.action_dim}
      num_steps: ${actor.model.num_steps}
      train_expert_only: False
      detach_critic_input: True
      num_images_in_input: 2               # 相机数量，按实际
    openpi_data:
      repo_id: "<SFT_repo_id>"
  optim:
    lr: 7.91e-6
    value_lr: 1.55e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-05
    clip_grad: 1.0
    critic_warmup_steps: 2
  fsdp_config:
    strategy: "fsdp"
    sharding_strategy: "no_shard"
    gradient_checkpointing: False
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

critic:
  use_critic_model: False

reward:
  use_reward_model: False                   # 改为 True 如果使用独立 reward worker
  # group_name: "RewardGroup"
  # model_path: "<reward_model_path>"
```

---

## 8. 执行步骤 Checklist

### Phase 0：验证环境可达性

- [ ] 确认本地能 ping 通云端的飞连 VPN IP
- [ ] 确认云端能 ping 通本地的飞连 VPN IP（可选；如不行，下文方案仍可行）
- [ ] 记录飞连 VPN 网卡名（`ifconfig` / `ip addr`）

### Phase 1：依赖安装

- [ ] 云端：安装 RLinf + pi0.5 依赖
  ```bash
  bash requirements/install.sh embodied --model openpi --env franka
  ```
- [ ] 本地：安装 RLinf + 真机控制依赖
  ```bash
  bash requirements/install.sh embodied --env franka   # 或最小安装
  # 单独安装真机控制相关 Python 包
  ```
- [ ] 确认两端的 RLinf 代码版本一致

### Phase 2：自定义环境开发

- [ ] 实现 `rlinf/envs/realworld/my_robot/my_robot_env.py`（参考第 4 节）
- [ ] 注册环境 ID（`gymnasium.register`）
- [ ] 在 `rlinf/envs/realworld/__init__.py` 中 import 任务模块
- [ ] 本地测试环境是否工作：
  ```python
  import gymnasium as gym
  env = gym.make("MyRobotEnv-v1", override_cfg={"is_dummy": True}, ...)
  obs, info = env.reset()
  obs, reward, t, tr, info = env.step(env.action_space.sample())
  print(obs.keys(), obs["frames"].keys())
  ```

### Phase 3：pi0.5 Checkpoint 准备

- [ ] 确认 checkpoint 目录结构（权重 + norm_stats）
- [ ] 如缺少 `norm_stats.json`，运行 `calculate_norm_stats.py` 生成
- [ ] 如果动作/状态布局不匹配现有 config，新增 `TrainConfig` 和 data config
- [ ] 确认 `task_description` 与 SFT 时的 prompt 一致

### Phase 4：Ray 集群启动

- [ ] 在云端：设置 `RLINF_NODE_RANK=0`，`RLINF_COMM_NET_DEVICES=<飞连网卡>`，启动 `ray start --head`
- [ ] 在本地：设置 `RLINF_NODE_RANK=1`，source `setup_before_ray.sh`，启动 `ray start --address=<cloud_feilian_ip>:6379`
- [ ] 执行 `ray status` 确认 2 个节点在线

### Phase 5：Dummy 验证

- [ ] 创建 dummy 配置：`env.*.override_cfg.is_dummy: True`
- [ ] 在云端 head 执行：
  ```bash
  bash examples/embodiment/run_realworld_async.sh my_robot_pi05_ppo
  ```
- [ ] 验证：Ray placement 正确、channel 通信正常、pi0.5 推理无报错

### Phase 6：真机训练

- [ ] 关闭 dummy：`is_dummy: False`
- [ ] 配置安全参数：关节限位、工作空间限位、最大动作幅度
- [ ] 启动训练（脚本同上），小批量短 episode 试跑
- [ ] 监控 TensorBoard：`env/return`、`env/success_once`、`train/actor/loss`
- [ ] 逐步增大 `max_steps_per_rollout_epoch`、`rollout_epoch`

### Phase 7：调优

- [ ] 带宽优化（如果 VPN 带宽不够）：
  - 降低图像分辨率（如 64×64）
  - 减少相机数量
- [ ] 权重同步优化：
  - 使用 LoRA（`actor.model.is_lora: True`）减小传输量
  - 调大 `weight_sync_interval`
- [ ] 训练稳定性：
  - 调整 `lr`、`value_lr`、`clip_ratio`
  - 打开 `filter_rewards`

---

## 9. 安全注意事项

1. **硬件急停**：确保真机有物理急停按钮
2. **软件限位**：在 env 的 `step()` 中做关节/工作空间限位裁剪
3. **人工干预**：可在 `override_cfg` 中启用 SpaceMouse/GELLO 干预
4. **键盘奖励**：训练早期可用 `keyboard_reward_wrapper` 人工标 reward
5. **Episodic 控制**：设置 `manual_episode_control_only: True`，由人工控制 episode 何时结束
6. **渐进扩大**：从短 episode、低动作幅度开始，逐步释放

---

## 10. 风险与备选

| 风险 | 备选方案 |
|------|---------|
| 飞连 VPN 不提供双向公网 IP | 无需双向。本地通过端口 12345 连接云端 Ray head，所有 Channel 通信走同一 gRPC 长连接 |
| Ray 跨机器通信不稳定 | 让云端防火墙确认放行端口 12345（TCP），`--node-ip-address` 显式绑定飞连 IP |
| 飞连 VPN 带宽不足，图像上传卡顿 | 降低图像分辨率；本地压缩 JPEG 后上传；减少相机数量 |
| pi0.5 全参数 weight sync 量大 | 开启 LoRA；使用 `weight_syncer/compressed_patch_syncer` |
| 奖励模型推理延迟高 | 将 reward model 部署到本地 CPU 上做本地推理 |
| 真机环境与仿真差异大 | 先用离线 demo 数据 + 在线 RLPD 混合训练 |
| SFT checkpoint 与自定义任务不匹配（分布外） | 先在真机上采集少量数据做领域微调 SFT，再启动 RL |

---

## 11. 相关文件与文档索引

| 资源 | 路径 |
|------|------|
| 云边协同教程 | `docs/source-zh/rst_source/tutorials/embodied/cloud_edge.rst` |
| 真机机器人训练启动 | `docs/source-zh/rst_source/tutorials/embodied/realworld_robot.rst` |
| 多节点配置 | `docs/source-zh/rst_source/tutorials/usage/multi_node.rst` |
| 异质集群配置 | `docs/source-zh/rst_source/tutorials/configuration/hetero.rst` |
| 自定义环境接入 pi0.5 | `docs/ADD_CUSTOM_ENV_WITH_PI0.md` |
| RealWorldEnv 包装器 | `rlinf/envs/realworld/realworld_env.py` |
| 现有真机环境示例 | `rlinf/envs/realworld/franka/franka_env.py` |
| Env worker | `rlinf/workers/env/env_worker.py` |
| pi0.5 model config | `examples/embodiment/config/model/pi0_5.yaml` |
| Async PPO 配置模板 | `examples/embodiment/config/realworld_peginsertion_async_ppo_pi05.yaml` |
| 训练入口 | `examples/embodiment/train_async.py` |
