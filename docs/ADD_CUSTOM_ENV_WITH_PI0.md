# 在 RLinf 中添加自定义环境并集成 pi0 模型 — 完整流程

本文档假定你的环境是完全自定义的（新的仿真器或真机环境，observation 和 action 空间均与现有环境不同），模型使用 pi0 / pi05 (OpenPI)。每个步骤均标注了需要修改的文件和关键代码行号。

---

## 目录

1. [整体架构回顾](#1-整体架构回顾)
2. [步骤 1：环境注册](#2-步骤-1环境注册)
3. [步骤 2：环境实现](#3-步骤-2环境实现)
4. [步骤 3：Action 工具函数](#4-步骤-3action-工具函数)
5. [步骤 4：Pi0 Policy 适配层](#5-步骤-4pi0-policy-适配层)
6. [步骤 5：Pi0 DataConfig 适配层](#6-步骤-5pi0-dataconfig-适配层)
7. [步骤 6：注册 Pi0 TrainConfig](#7-步骤-6注册-pi0-trainconfig)
8. [步骤 7：Config YAML 文件](#8-步骤-7config-yaml-文件)
9. [步骤 8：Config 校验（可选）](#9-步骤-8config-校验可选)
10. [步骤 9：Install / Docker / CI（可选）](#10-步骤-9install--docker--ci可选)
11. [步骤 10：文档（可选）](#11-步骤-10文档可选)
12. [验证清单](#12-验证清单)
13. [附录 A：关键数据结构](#附录-a关键数据结构)
14. [附录 B：Pi0 模型支持的图像输入](#附录-bpi0-模型支持的图像输入)

---

## 1. 整体架构回顾

一条 RLinf 训练链的完整数据流：

```
Config YAML (Hydra)
  │
  ├── env_type: "my_custom_env"
  │     └── get_env_cls() → MyCustomEnv  ──── 产生 obs dict
  │
  ├── model_type: "openpi" (已存在，无需修改)
  │     └── get_model()  → OpenPi0ForRLActionPrediction
  │           │
  │           └── setup_wrappers(transforms=[...])
  │                 ├── LiberoInputs  ── 将 env obs → pi0 期望的输入格式
  │                 ├── Normalize(norm_stats) ── 归一化
  │                 ├── ModelTransforms ── tokenize prompt / actions
  │                 ├── Unnormalize(norm_stats) ── 反归一化
  │                 └── LiberoOutputs ── pi0 输出 → env action 格式
  │
  ├── action_utils.prepare_actions(env_type=..., model_type=...)
  │     └── prepare_actions_for_libero() ── 动作后处理（夹爪变换等）
  │
  └── cluster.component_placement → 调度 env/actor/rollout workers
```

**你需要修改的 5 个核心环节**（至少）：

| # | 环节 | 文件 | 工作量 |
|---|------|------|--------|
| 1 | 环境注册 | `rlinf/envs/__init__.py` | 小 |
| 2 | 环境实现 | `rlinf/envs/<name>/<name>_env.py` (新建) | 中 |
| 3 | Action 工具 | `rlinf/envs/action_utils.py` | 小 |
| 4 | Policy 适配 | `rlinf/models/embodiment/openpi/policies/<name>_policy.py` (新建) | 中 |
| 5 | DataConfig | `rlinf/models/embodiment/openpi/dataconfig/<name>_dataconfig.py` (新建) | 中 |
| 6 | 注册 TrainConfig | `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 小 |
| 7 | Config YAML | `examples/embodiment/config/env/<name>.yaml` (新建) + 主训练 config | 小 |

---

## 2. 步骤 1：环境注册

**文件：** `rlinf/envs/__init__.py`

### 2a. 添加 `SupportedEnvType` 枚举成员（~第 18 行）

```python
class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    # ... 现有成员 ...
    POLARIS = "polaris"
    # ↓↓↓ 新增 ↓↓↓
    MY_CUSTOM_ENV = "my_custom_env"    # value 必须全部小写, 使用下划线分隔
```

### 2b. 在 `get_env_cls()` 中添加 lazy import 分支（~第 62 行附近）

```python
def get_env_cls(env_type: str, env_cfg=None):
    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.LIBERO:
        from rlinf.envs.libero.libero_env import LiberoEnv
        return LiberoEnv
    # ↓↓↓ 新增 ↓↓↓
    elif env_type == SupportedEnvType.MY_CUSTOM_ENV:
        from rlinf.envs.my_custom_env.my_custom_env import MyCustomEnv
        return MyCustomEnv
    # ...
```

**要点：**
- 使用 lazy import，避免在 `__init__.py` 加载时引入重型依赖
- 如果你的环境像 IsaacLab 那样需要在运行时根据 `env_cfg` 选择不同子类，在此处实现分支逻辑（参考 `isaaclab` 的 `REGISTER_ISAACLAB_ENVS` 模式）

---

## 3. 步骤 2：环境实现

**文件（新建）：** `rlinf/envs/my_custom_env/my_custom_env.py`

### 3a. 目录结构

```
rlinf/envs/my_custom_env/
├── __init__.py        # 可空或导出 MyCustomEnv
├── my_custom_env.py   # 主环境实现
└── venv.py            # (可选) 自定义向量化包装器
```

### 3b. 环境类契约

你的环境类**不必须**继承 `gym.Env`，但必须实现以下接口。所有已有环境（LIBERO、MetaWorld 等）都遵循此约定。

构造函数签名必须与所有 Worker 调用的参数一致：

```python
class MyCustomEnv:
    def __init__(
        self,
        cfg,                    # DictConfig: 环境的 Hydra 配置片段
        num_envs: int,          # 并行环境数量
        seed_offset: int,       # 种子偏移量（确保不同 worker 独立的随机性）
        total_num_processes: int,  # 总进程数
        worker_info: dict,      # Worker 元信息（node_id, group_rank 等）
    ):
        self._num_envs = num_envs
        self.cfg = cfg
        # 在此处初始化你的仿真器、真机连接等
```

**必须实现的公开属性：**

```python
@property
def elapsed_steps(self) -> int:
    """当前 episode 已执行的步数."""

@property
def is_start(self) -> bool:
    """第一次 reset 后为 True，用于标记 episode 开始."""

@property
def info_logging_keys(self) -> list[str]:
    """哪些 info dict 中的 key 需要被记录到日志."""
    return ["success", "reward", ...]
```

**必须实现的方法：**

#### `reset(env_idx, reset_state_ids) → (obs, infos)`

```python
def reset(self, env_idx: list[int], reset_state_ids: list[int] | None):
    """
    重置指定索引的环境。

    Args:
        env_idx: 需要重置的环境索引列表
        reset_state_ids: 每个重置环境对应的初始状态 ID，
                         如果为 None 则随机采样

    Returns:
        obs: 观测 dict，格式见附录 A
        infos: dict，包含额外信息（如 success, reward 等）
    """
    # 对每个 env_idx[i] 执行 reset
    # 构造 observation dict
    obs = self._wrap_obs(raw_obs)
    return obs, infos
```

#### `step(actions, auto_reset=True) → (obs, rewards, terminations, truncations, infos)`

```python
def step(self, actions, auto_reset: bool = True):
    """
    执行一步动作。

    Args:
        actions: numpy 或 torch tensor, shape [num_envs, action_dim]
        auto_reset: 如果为 True，terminated 的环境自动 reset

    Returns:
        obs: dict
        rewards: [num_envs], float
        terminations: [num_envs], bool
        truncations: [num_envs], bool
        infos: dict
    """
```

#### `chunk_step(chunk_actions) → (obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list)`

```python
def chunk_step(self, chunk_actions):
    """
    执行一段动作序列。

    Args:
        chunk_actions: [num_envs, num_action_chunks, action_dim]

    Returns:
        obs_list: list of obs dicts, 长度为 num_action_chunks
        chunk_rewards: [num_envs, num_action_chunks]
        chunk_terminations: [num_envs, num_action_chunks]
        chunk_truncations: [num_envs, num_action_chunks]
        infos_list: list of infos dicts, 长度为 num_action_chunks
    """
```

### 3c. Observation 格式规范

Observation dict 的可选字段取决于你的环境配置。可复用的字段见附录 A。

**对于 pi0 模型的通用要求：**
- 必须提供 `"main_images"` 或等价的图像字段
- 必须提供 `"states"`（proprioceptive state）
- 必须提供 `"task_descriptions"`（语言指令列表）

### 3d. 向量化（可选）

如果使用 `SubprocVectorEnv` 进行进程级并行，可参考 `rlinf/envs/libero/venv.py` 中的 `ReconfigureSubprocEnv` 实现。如果你的仿真器本身支持批量运行，可以直接在环境类内部实现。

---

## 4. 步骤 3：Action 工具函数

**文件：** `rlinf/envs/action_utils.py`

pi0 模型输出的原始动作（经过 de-normalize）可能需要按环境的实际动作接口做后处理。

### 4a. 创建专用函数

参考 `prepare_actions_for_libero()`（第 66 行，`action_utils.py:66`），创建你的版本：

```python
def prepare_actions_for_my_custom_env(
    raw_chunk_actions,
    model_type: str,
) -> np.ndarray:
    """
    将 pi0 原始输出动作转换为自定义环境需要的格式。

    Args:
        raw_chunk_actions: [num_envs, num_chunks, model_action_dim]
        model_type: 模型类型字符串 (例如 "openpi")

    Returns:
        chunk_actions: [num_envs, num_chunks, env_action_dim]
    """
    chunk_actions = raw_chunk_actions

    # 对于 pi0 (OpenPI) 模型，通常不需要特殊变换
    # 如果你的环境动作语义与 pi0 输出有差异，在此处转换
    # 例如：夹爪阈值处理、坐标变换、维度裁剪/填充等

    return chunk_actions
```

### 4b. 在 `prepare_actions()` 主函数中添加入口（~第 282 行，`action_utils.py:282`）

```python
def prepare_actions(raw_chunk_actions, env_type, model_type, ...):
    # ...
    elif env_type == SupportedEnvType.ROBOTWIN:
        chunk_actions = raw_chunk_actions
    # ↓↓↓ 新增 ↓↓↓
    elif env_type == SupportedEnvType.MY_CUSTOM_ENV:
        chunk_actions = prepare_actions_for_my_custom_env(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    # ...
    return chunk_actions
```

**注意：** 如果导入了新函数，需要在文件顶部添加 `from` 导入或使用 lazy import（参考 `prepare_actions_for_robocasa` 的实现，`action_utils.py:176`）。

---

## 5. 步骤 4：Pi0 Policy 适配层

**这是最关键的一步。** Pi0 模型通过 `LiberoInputs` / `LiberoOutputs` 变换类将环境观测映射到模型内部格式。你需要为自己的环境创建对应的变换类。

**文件（新建）：** `rlinf/models/embodiment/openpi/policies/my_custom_env_policy.py`

参考文件：`rlinf/models/embodiment/openpi/policies/libero_policy.py`（118 行）

### 5a. `make_*_example()` 函数（用于测试）

```python
def make_my_custom_env_example() -> dict:
    """创建一个随机输入样本用于策略测试."""
    return {
        "observation/state": np.random.rand(STATE_DIM),        # 你的 state 维度
        "observation/image": np.random.randint(                # 主图
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(          # 腕部图（可选）
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }
```

### 5b. `MyCustomEnvInputs` 类（关键）

将环境观测转换为 pi0 模型的标准输入格式：

```python
@dataclasses.dataclass(frozen=True)
class MyCustomEnvInputs(transforms.DataTransformFn):
    """
    将环境 observation 转换为 pi0 模型的标准输入格式。

    这个类在训练和推理时都会被调用。
    """

    model_type: _model.ModelType   # pi0 或 pi0-FAST

    def __call__(self, data: dict) -> dict:
        # 解析图像 — to uint8, (H, W, C) 格式
        base_image = _parse_image(data["observation/image"])

        # 如果有腕部图像
        wrist_image = _parse_image(data.get("observation/wrist_image", np.zeros_like(base_image)))

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                # ↓ 根据你的摄像头配置，可启用以下之一或使用 zeros
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if wrist_image is not zeros else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # 训练时才有 actions
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 语言指令（task description）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

**pi0 模型支持的图像输入**（详见附录 B）：
| 键名 | 含义 | 必须？ |
|------|------|--------|
| `base_0_rgb` | 第三人称视角 | 必须 |
| `left_wrist_0_rgb` | 左手腕相机 | 可选 |
| `right_wrist_0_rgb` | 右手腕相机 | 可选 |

若你的环境只有一张主图，将 `left_wrist_0_rgb` 和 `right_wrist_0_rgb` 都设为 `np.zeros_like(base_image)`，并在 `image_mask` 中将它们标记为 `np.False_`（pi0）或 `np.True_`（pi0-FAST）。

### 5c. `MyCustomEnvOutputs` 类

将 pi0 模型输出转换回环境需要的 action 格式：

```python
@dataclasses.dataclass(frozen=True)
class MyCustomEnvOutputs(transforms.DataTransformFn):
    """
    从 pi0 模型输出中提取环境需要的 action 维度。

    pi0 模型的 action 维度是固定的（由 model_config 定义），
    如果你的环境 action 维度较小，需要在此处裁剪。
    """

    def __call__(self, data: dict) -> dict:
        # pi0 输出 "actions" 的维度 = model_config.max_action_dim
        # 只取前 ACTION_DIM 维作为环境的实际动作
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
```

**关键说明：**

- **`ACTION_DIM`** 是你的环境的动作维度。Libero 是 7（3D 位置 + 3D 旋转 + 1D 夹爪）。你的环境可能是不同的维度。
- pi0 模型的 `max_action_dim`（内部 action token 数）在 `Pi0Config` 中定义，模型输出的 action 维度可能比你的环境动作维度大。`MyCustomEnvOutputs` 负责裁剪到环境实际需要的维度。
- 如果你的环境动作维度 > pi0 的 `max_action_dim`，需要同时修改 `Pi0Config` 或在 `Inputs` 中做 padding。

---

## 6. 步骤 5：Pi0 DataConfig 适配层

**文件（新建）：** `rlinf/models/embodiment/openpi/dataconfig/my_custom_env_dataconfig.py`

参考文件：`rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py`（102 行）

```python
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import my_custom_env_policy


@dataclasses.dataclass(frozen=True)
class LeRobotMyCustomEnvDataConfig(DataConfigFactory):
    """自定义环境的 DataConfig。"""

    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # 1. RepackTransform: 将数据集的旧 key 重映射为环境使用的 key
        #    如果数据集 key 与环境 key 一致，可以省略
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 2. DataTransforms: 使用上一步创建的 Policy 变换类
        data_transforms = _transforms.Group(
            inputs=[
                my_custom_env_policy.MyCustomEnvInputs(
                    model_type=model_config.model_type
                )
            ],
            outputs=[my_custom_env_policy.MyCustomEnvOutputs()],
        )

        # 3. 如果你的数据是 absolute actions 需要转换为 delta actions
        #    如果你的数据已经是 delta actions，跳过此段
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(
                ACTION_DIM - 1, -1  # 前 N-1 维做 delta，最后一维（夹爪）保持绝对
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 4. ModelTransforms: 标准 pi0 的 tokenize 和 action 处理，不需要修改
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

---

## 7. 步骤 6：注册 Pi0 TrainConfig

**文件：** `rlinf/models/embodiment/openpi/dataconfig/__init__.py`

在 `_CONFIGS` 列表（~第 72 行）中添加你的 TrainConfig 条目，并在文件顶部的 import 中添加对应的 DataConfig 导入。

```python
# 文件顶部 import 区域添加：
from rlinf.models.embodiment.openpi.dataconfig.my_custom_env_dataconfig import (
    LeRobotMyCustomEnvDataConfig,
)

# 在 _CONFIGS = [...] 列表中添加（注意 config name 必须唯一）：
    TrainConfig(
        name="pi05_my_custom_env",           # 必须唯一，对应 YAML 中的 config_name
        model=pi0_config.Pi0Config(
            pi05=True,                        # 使用 pi05 架构
            action_horizon=10,                # 动作预测长度
            discrete_state_input=False,       # 连续状态输入
        ),
        data=LeRobotMyCustomEnvDataConfig(
            repo_id="your-org/your-dataset",  # HuggingFace 数据集 ID 或本地路径
            base_config=DataConfig(
                prompt_from_task=True          # 从 task 生成自然语言指令
            ),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_my_custom_env/assets"
            ),
            extra_delta_transform=False,       # 如果你的数据是 absolute actions，设为 True
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"        # 从 pi05 基础模型加载权重
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=256,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
    ),
```

**配置字段说明：**

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `name` | 全局唯一名称，对应 YAML 的 `openpi.config_name` | `"pi05_my_custom_env"` |
| `model.pi05` | 使用 pi05 还是 pi0 架构 | `True` |
| `model.action_horizon` | 单次预测的动作帧数 | `10` |
| `model.discrete_state_input` | 是否使用离散化状态 | `False`（连续状态） |
| `data.repo_id` | LeRobot 格式数据集 ID | `"your-org/your-dataset"` |
| `data.extra_delta_transform` | 是否在 data pipeline 中额外做 absolute→delta 转换 | `False` |
| `data.assets_dirs` / `AssetsConfig` | 归一化统计量存放位置 | `"checkpoints/torch/..."` |
| `weight_loader` | 预训练权重加载器 | `CheckpointWeightLoader(...)` |
| `pytorch_weight_path` | PyTorch 格式权重本地路径 | `"checkpoints/torch/pi05_base"` |

---

## 8. 步骤 7：Config YAML 文件

### 8a. 创建环境默认配置

**文件（新建）：** `examples/embodiment/config/env/my_custom_env.yaml`

```yaml
# 必须字段
env_type: my_custom_env       # 必须与 SupportedEnvType 的 value 一致

# 通用默认值（在具体训练 config 中可覆盖）
total_num_envs: null

auto_reset: False
ignore_terminations: False
max_steps_per_rollout_epoch: 240
max_episode_steps: 240

use_rel_reward: True
use_step_penalty: False
reward_coef: 1.0

seed: 0
group_size: 1

# 自定义环境特定参数
init_params:
  camera_heights: 256
  camera_widths: 256
  # 你的环境特定参数 ...
```

### 8b. 创建主训练配置

**文件（新建）：** `examples/embodiment/config/my_custom_env_ppo_openpi_pi05.yaml`

```yaml
defaults:
  - env/my_custom_env@env.train           # 训练环境
  - env/my_custom_env@env.eval            # 评估环境
  - model/pi0_5@actor.model               # 复用 pi05 模型模板
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all                 # 单机部署

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "my_custom_env_ppo_openpi_pi05"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1
  val_check_interval: -1
  save_interval: 40

algorithm:
  adv_type: gae
  loss_type: actor_critic
  normalize_advantages: True
  gamma: 0.99
  gae_lambda: 0.95
  group_size: 1
  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level
  update_epoch: 1
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  kl_beta: 0.0

env:
  group_name: "EnvGroup"
  train:
    rollout_epoch: 8
    total_num_envs: 64
    max_episode_steps: 240
    max_steps_per_rollout_epoch: 240
  eval:
    rollout_epoch: 1
    total_num_envs: 500
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 240

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  recompute_logprobs: False
  pipeline_stage_num: 1
  model:
    model_path: "/path/to/model/RLinf-Pi05-SFT"  # 你的 SFT 模型路径
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 128
  global_batch_size: 2048
  seed: 42
  enable_offload: False

  # 覆盖 model/pi0_5.yaml 中的默认值
  model:
    model_path: "/path/to/model/RLinf-Pi05-SFT"
    model_type: "openpi"              # 已注册，无需修改
    num_action_chunks: 5               # 环境一次 consume 的动作帧数
    action_dim: 7                      # 你的环境的动作维度
    num_steps: 3
    add_value_head: True
    openpi:
      config_name: "pi05_my_custom_env"   # ← 必须与 TrainConfig.name 一致
      num_images_in_input: 2             # 你的环境有多少张图
      train_expert_only: True
      noise_level: 0.5
      noise_method: "flow_sde"
      value_after_vlm: True

  optim:
    lr: 5.0e-6
    value_lr: 1.0e-4
    weight_decay: 0.01
    clip_grad: 1.0

  fsdp_config:
    strategy: "fsdp"
    sharding_strategy: "no_shard"
    gradient_checkpointing: False

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

### 8c. 启动训练

```bash
python examples/embodiment/train_embodied_agent.py \
    --config-name my_custom_env_ppo_openpi_pi05
```

---

## 9. 步骤 8：Config 校验（可选）

**文件：** `rlinf/config.py`，函数 `validate_embodied_cfg`（~第 814 行）

如果你的环境有特殊的参数约束，在验证逻辑中添加对应的校验分支。参考现有环境的例子：

- ManiSkill：自动选择 `control_mode` 和 `policy` 类型
- Behavior：验证 `base_config_name` 互斥字段

通用校验（`total_num_envs`、`max_steps_per_rollout_epoch` 与 `num_action_chunks` 的整除关系等）已在现有逻辑中处理，无需额外添加。

---

## 10. 步骤 9：Install / Docker / CI（可选）

如果你的自定义环境引入了新的 Python 依赖，需要修改以下文件。所有步骤可使用 `.claude/skills/` 下的自动化工具协助完成：

### 10a. Install 脚本

**文件：** `requirements/install.sh`

1. 在 `SUPPORTED_ENVS` 数组中添加 `MY_CUSTOM_ENV="my_custom_env"`
2. 实现 `install_my_custom_env()` 函数，安装你的环境依赖
3. 在 `main()` 中 `case "$ENV"` 添加分支
4. 验证：运行 `bash .claude/skills/install-check/check.sh`

### 10b. Dockerfile

**文件：** `docker/Dockerfile`

添加 `embodied-my-custom-env-image` 构建阶段。**关键：多个 `install.sh` 调用必须在同一个 `RUN` 中（用 `&&` 连接）**，因为 `uv` hardlink 模式不支持跨层。

### 10c. CI

在 `.github/workflows/` 中添加对应的 e2e test job 和 docker build job。

---

## 11. 步骤 10：文档（可选）

使用 `.claude/skills/add-example-doc-model-env/SKILL.md` 中的工作流，创建英文和中文的示例文档 RST 页面。

---

## 12. 验证清单

完成所有步骤后，逐项检查：

- [ ] `SupportedEnvType` 枚举添加了新成员
- [ ] `get_env_cls()` 添加了 lazy import 分支
- [ ] 环境类实现了 `__init__`, `reset`, `step`, `chunk_step` 完整接口
- [ ] Observation dict 包含 pi0 需要的字段（image, state, task_descriptions）
- [ ] `prepare_actions()` 添加了环境分支
- [ ] `*Inputs` 类正确将 env obs 映射到 pi0 内部格式
- [ ] `*Outputs` 类正确从 pi0 输出中提取环境 action 维度
- [ ] DataConfig 正确注册了 repack + data + model transforms
- [ ] `TrainConfig` 已添加到 `_CONFIGS` 列表，`name` 唯一
- [ ] env YAML 配置中 `env_type` 与 `SupportedEnvType.value` 一致
- [ ] 训练 YAML 中 `openpi.config_name` 与 `TrainConfig.name` 一致
- [ ] `action_dim` 与 `MyCustomEnvOutputs` 中的 `ACTION_DIM` 一致
- [ ] `num_images_in_input` 与 `MyCustomEnvInputs` 中的图像数量一致
- [ ] 运行 `python examples/embodiment/train_embodied_agent.py --config-name <name>` 启动训练

---

## 附录 A：关键数据结构

### Observation Dict (`reset` / `step` 的返回值)

```python
obs = {
    # ---- 图像 ----
    "main_images": torch.Tensor,      # [B, H, W, C], uint8, 第三人称视角
    "wrist_images": torch.Tensor,     # [B, H, W, C] or None, 腕部相机

    # ---- 本体状态 ----
    "states": torch.Tensor,           # [B, state_dim], float

    # ---- 任务描述 ----
    "task_descriptions": list[str],   # 长度为 B 的字符串列表
}
```

### Action 格式

```python
# pi0 模型输出的原始 action（经过 Unnormalize 后）
raw_actions: np.ndarray  # [B, num_chunks, model_max_action_dim]

# 经过 Outputs 提取后 → prepare_actions() 输入
chunk_actions: np.ndarray  # [B, num_chunks, env_action_dim]

# 传给 step() 的单个动作
actions: np.ndarray  # [num_envs, env_action_dim]
```

### 环境配置 (cfg) 的常用字段

```python
cfg.total_num_envs          # 总环境数
cfg.max_episode_steps       # episode 最大步数（truncation）
cfg.max_steps_per_rollout_epoch  # 每次 rollout 的最大步数
cfg.auto_reset              # 是否自动 reset
cfg.ignore_terminations     # 是否忽略 termination（不 reset）
cfg.seed                    # 种子
cfg.group_size              # 环境分组大小
cfg.init_params             # 环境初始化参数（如 camera 分辨率）
```

---

## 附录 B：Pi0 模型支持的图像输入

Pi0 支持最多 3 张图像同时输入模型：

| 内部键名 | 含义 | 是否必须 |
|----------|------|----------|
| `base_0_rgb` | 第三人称视图（base view） | **必须** |
| `left_wrist_0_rgb` | 左腕部相机 | 可选 |
| `right_wrist_0_rgb` | 右腕部相机 | 可选 |

**`image_mask` 控制哪些图像实际参与模型计算：**

- **Pi0 (原始版):** 只对实际存在的图像传入 `np.True_`，padding 图像填入 `np.False_`
- **Pi0-FAST:** 所有图像 slot 一定要给 `np.True_`（因为 FAST variant 依赖固定的图像结构）

**如果你的环境只有 1 张图：**

```python
base_image = _parse_image(data["observation/image"])
inputs = {
    "image": {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": np.zeros_like(base_image),
        "right_wrist_0_rgb": np.zeros_like(base_image),
    },
    "image_mask": {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    },
}
```

**State 输入：** `"state"` 字段直接传入模型的 proprioceptive tokenizer，通常不需要特殊的维度约束（但建议 ≤~64 维以获得合理的 token 使用效率）。
