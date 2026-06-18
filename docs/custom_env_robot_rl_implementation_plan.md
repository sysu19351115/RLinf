# RLinf 自定义环境 + 自定义机器人策略 强化学习实施计划

> 目标：将一个全新的仿真环境（或真实机器人）以及一个全新的机器人策略接入 RLinf，完成端到端的强化学习训练。
>
> 本计划假设使用 **同步 embodied RL 入口** `examples/embodiment/train_embodied_agent.py`，训练后端为 **FSDP**，算法为 **PPO actor-critic**。

---

## 前置准备

1. 已安装 RLinf 及依赖：
   ```bash
   bash requirements/install.sh embodied --model <base_model> --env <base_env>
   ```
2. 已启动 Ray（单机可自动启动，多机需先在各节点执行 `ray start` 并设置 `RLINF_NODE_RANK`）。
3. 已通读关键文件：
   - `examples/embodiment/train_embodied_agent.py`
   - `rlinf/runners/embodied_runner.py`
   - `rlinf/envs/__init__.py`
   - `rlinf/envs/action_utils.py`
   - `rlinf/models/embodiment/base_policy.py`
   - `rlinf/models/embodiment/mlp_policy/mlp_policy.py`
   - `rlinf/config.py`

---

## 第一步：实现自定义环境

### 1.1 创建环境包

创建目录：
```bash
mkdir -p rlinf/envs/my_env
```

新建文件 `rlinf/envs/my_env/__init__.py`：
```python
from rlinf.envs.my_env.my_env import MyEnv

__all__ = ["MyEnv"]
```

### 1.2 实现环境类

新建文件 `rlinf/envs/my_env/my_env.py`，参考 `rlinf/envs/embodichain/embodichain_env.py` 的最小结构：

```python
import gymnasium as gym
import numpy as np
import torch
from typing import Any, Optional, Union


class MyEnv(gym.Env):
    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.group_size = int(getattr(cfg, "group_size", 1))
        self.num_group = self.num_envs // self.group_size
        self.seed = int(getattr(cfg, "seed", 0)) + int(seed_offset)
        self.max_episode_steps = int(getattr(cfg, "max_episode_steps", 500))
        self.auto_reset = bool(getattr(cfg, "auto_reset", True))
        self.ignore_terminations = bool(getattr(cfg, "ignore_terminations", False))
        self.video_cfg = getattr(cfg, "video_cfg", None)

        self._device = torch.device("cuda:0")  # RLinf 会为每个 worker 单独设置 CUDA_VISIBLE_DEVICES
        self._is_start = True
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self._device)

        # 在此处构建/加载你的底层环境
        self.env = self._build_env()

        # 设置 action_space（gymnasium.spaces.Box 或 Discrete）
        action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        if action_low.ndim > 1:
            action_low = action_low[0]
            action_high = action_high[0]
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=tuple(action_low.shape), dtype=np.float32
        )

        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._init_metrics()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    def _build_env(self):
        """构造或导入底层环境。可以是 gym.make、自定义仿真器、IsaacLab 等。"""
        raise NotImplementedError("请在这里实例化你的底层环境")

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is None:
            self.prev_step_reward.zero_()
            self.success_once.zero_()
            self.fail_once.zero_()
            self.returns.zero_()
            self._elapsed_steps.zero_()
            return
        env_idx = torch.as_tensor(env_idx, dtype=torch.long, device=self._device)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.fail_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(self, step_reward: torch.Tensor, infos: dict[str, Any]) -> dict[str, Any]:
        episode_info: dict[str, Any] = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"].bool()
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"].bool()
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_len = self.elapsed_steps
        episode_info["episode_len"] = episode_len.clone()
        denom = torch.clamp(episode_len.float(), min=1.0)
        episode_info["reward"] = episode_info["return"] / denom
        infos["episode"] = episode_info
        return infos

    def _wrap_obs(self, raw_obs: Any) -> dict[str, torch.Tensor]:
        """把底层观测转换成模型输入字典。"""
        # 示例：状态输入
        if not isinstance(raw_obs, torch.Tensor):
            raw_obs = torch.as_tensor(raw_obs, dtype=torch.float32, device=self._device)
        return {"states": raw_obs.reshape(self.num_envs, -1)}

    def _wrap_info(self, infos: Any) -> dict[str, Any]:
        if infos is None:
            return {}
        if isinstance(infos, dict):
            return infos
        return dict(infos)

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        options = {} if options is None else dict(options)
        env_idx = options.pop("env_idx", None)
        if env_idx is None:
            raw_obs, infos = self.env.reset(seed=seed)
            self._reset_metrics()
        else:
            reset_ids = torch.as_tensor(env_idx, dtype=torch.int32, device=self._device)
            raw_obs, infos = self.env.reset(options={"reset_ids": reset_ids})
            self._reset_metrics(reset_ids)
        self._is_start = True
        return self._wrap_obs(raw_obs), self._wrap_info(infos)

    def step(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_tensor = (
            actions.to(self._device, dtype=torch.float32)
            if isinstance(actions, torch.Tensor)
            else torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        )
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        action_tensor = action_tensor.reshape(self.num_envs, -1)

        raw_obs, rewards, terminations, truncations, infos = self.env.step(action_tensor)
        infos = self._wrap_info(infos)
        self._elapsed_steps += 1
        infos = self._record_metrics(rewards, infos)

        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations, dtype=torch.bool, device=self._device)

        dones = torch.logical_or(terminations, truncations)
        if dones.any() and self.auto_reset:
            done_ids = torch.nonzero(dones, as_tuple=False).flatten().to(torch.int32)
            self._reset_metrics(done_ids)

        self._is_start = False
        return self._wrap_obs(raw_obs), rewards, terminations, truncations, infos

    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        """执行动作块 [num_envs, chunk_steps, action_dim]。"""
        chunk_actions = (
            chunk_actions.to(self._device, dtype=torch.float32)
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(chunk_actions, dtype=torch.float32, device=self._device)
        )
        if chunk_actions.ndim != 3:
            raise ValueError(
                f"chunk_actions must have shape [num_envs, chunk_steps, action_dim], got {chunk_actions.shape}"
            )

        obs_list, infos_list, chunk_rewards = [], [], []
        raw_terms, raw_truncs = [], []
        for step_idx in range(int(chunk_actions.shape[1])):
            obs, rewards, terminations, truncations, infos = self.step(chunk_actions[:, step_idx])
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(rewards)
            raw_terms.append(terminations)
            raw_truncs.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_terms_t = torch.stack(raw_terms, dim=1)
        raw_truncs_t = torch.stack(raw_truncs, dim=1)

        past_terminations = raw_terms_t.any(dim=1)
        past_truncations = raw_truncs_t.any(dim=1)

        chunk_terminations = torch.zeros_like(raw_terms_t)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_truncs_t)
        chunk_truncations[:, -1] = past_truncations

        return obs_list, chunk_rewards_t, chunk_terminations, chunk_truncations, infos_list

    def update_reset_state_ids(self):
        return None

    def sample_action_space(self) -> torch.Tensor:
        return torch.as_tensor(self.action_space.sample(), dtype=torch.float32, device=self._device)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
```

### 1.3 注册环境

编辑 `rlinf/envs/__init__.py`：

```python
class SupportedEnvType(Enum):
    ...
    MY_ENV = "my_env"


def get_env_cls(env_type: str, env_cfg=None):
    ...
    elif env_type == SupportedEnvType.MY_ENV:
        from rlinf.envs.my_env.my_env import MyEnv
        return MyEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
```

---

## 第二步：添加动作后处理（如需要）

如果你的策略输出 gripper 维度需要二值化、维度裁剪或其他转换，编辑 `rlinf/envs/action_utils.py`：

```python
def prepare_actions_for_my_env(raw_chunk_actions, model_type, action_dim):
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.MY_POLICY:
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] > 0.5, 1.0, -1.0)
    return chunk_actions
```

然后在 `prepare_actions()` 的分发逻辑中加入：

```python
elif env_type == SupportedEnvType.MY_ENV:
    chunk_actions = prepare_actions_for_my_env(
        raw_chunk_actions=raw_chunk_actions,
        model_type=model_type,
        action_dim=action_dim,
    )
```

如果不需要特殊转换，直接返回 `raw_chunk_actions` 即可。

---

## 第三步：实现自定义机器人/策略

### 3.1 创建策略包

```bash
mkdir -p rlinf/models/embodiment/my_policy
```

新建 `rlinf/models/embodiment/my_policy/__init__.py`：
```python
from rlinf.models.embodiment.my_policy.my_policy import MyPolicy


def get_model(cfg, torch_dtype=None):
    return MyPolicy(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.get("num_action_chunks", 1),
        hidden_dim=cfg.get("hidden_dim", 256),
        add_value_head=cfg.get("add_value_head", True),
    )
```

### 3.2 实现策略类

新建 `rlinf/models/embodiment/my_policy/my_policy.py`，继承 `BasePolicy`：

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class MyPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_action_chunks: int = 1,
        hidden_dim: int = 256,
        add_value_head: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim * num_action_chunks)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim * num_action_chunks))

        if add_value_head:
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            self.value_head = None

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"forward_type {forward_type} not supported")

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=True,
        **kwargs,
    ):
        states = forward_inputs["states"]
        actions = forward_inputs["action"]
        feat = self.backbone(states)
        action_mean = self.actor_mean(feat)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        output = {}
        if compute_logprobs:
            output["logprobs"] = dist.log_prob(actions.reshape_as(action_mean)).sum(dim=-1)
        if compute_entropy:
            output["entropy"] = dist.entropy().sum(dim=-1)
        if compute_values and self.value_head is not None:
            output["values"] = self.value_head(feat).squeeze(-1)
        return output

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, **kwargs):
        states = env_obs["states"].float()
        feat = self.backbone(states)
        action_mean = self.actor_mean(feat)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        actions = dist.sample()
        logprobs = dist.log_prob(actions).sum(dim=-1)

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        values = self.value_head(feat).squeeze(-1) if self.value_head is not None else None

        forward_inputs = {
            "states": states,
            "action": actions,
        }

        return chunk_actions, {
            "prev_logprobs": logprobs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
```

> 如果你的策略是 VLA、Diffusion、Flow 等更复杂的模型，请参照 `rlinf/models/embodiment/openvla/`、`openpi/`、`gr00t/` 的实现，并在 `predict_action_batch` 中返回与训练时 `default_forward` 兼容的 `forward_inputs`。

---

## 第四步：注册策略

### 4.1 运行时注册（推荐用于自定义项目）

在入口脚本或一个前置模块中加入：

```python
from rlinf.models import register_model


def build_my_policy(cfg, torch_dtype):
    from rlinf.models.embodiment.my_policy import get_model
    return get_model(cfg, torch_dtype)


register_model("my_policy", build_my_policy, category="embodiment")
```

如果 Ray worker 在独立进程中运行，确保通过 `RLINF_EXT_MODULE` 让 worker 也能加载该注册逻辑（参见 `docs/source-en/rst_source/tutorials/extend/new_model_fsdp.rst`）。

### 4.2 合入 RLinf 主仓库时

编辑 `rlinf/config.py`：
```python
SupportedModel.MY_POLICY = SupportedModel.register("my_policy", force=True)
```

编辑 `rlinf/models/__init__.py`，在 `_register_builtin_models()` 中加入：
```python
def _build_my_policy(cfg: DictConfig, torch_dtype):
    from rlinf.models.embodiment.my_policy import get_model
    return get_model(cfg, torch_dtype)


register_model(
    SupportedModel.MY_POLICY.value, _build_my_policy, category="embodied", force=True
)
```

---

## 第五步：添加 YAML 配置

### 5.1 环境默认配置

新建 `examples/embodiment/config/env/my_env.yaml`：

```yaml
env_type: my_env

# 你的环境自定义参数
some_env_param: "value"
headless: true
sim_device: cuda

seed: 0
total_num_envs: null
group_size: 1

auto_reset: True
ignore_terminations: False
max_steps_per_rollout_epoch: 128
max_episode_steps: 500

use_rel_reward: False
reward_coef: 1.0
is_eval: False

video_cfg:
  save_video: False
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train
```

### 5.2 策略默认配置

新建 `examples/embodiment/config/model/my_policy.yaml`：

```yaml
model_type: "my_policy"
model_path: ""

obs_dim: 42
action_dim: 8
num_action_chunks: 1
hidden_dim: 256

precision: "32"
add_value_head: True

is_lora: False
lora_rank: 32
```

### 5.3 完整实验配置

新建 `examples/embodiment/config/my_env_ppo_my_policy.yaml`：

```yaml
defaults:
  - env/my_env@env.train
  - env/my_env@env.eval
  - model/my_policy@actor.model
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
    actor,env,rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "my_env_ppo_my_policy"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1
  val_check_interval: 50
  save_interval: 100

algorithm:
  adv_type: gae
  loss_type: actor_critic
  normalize_advantages: True
  group_size: 1
  update_epoch: 4

  reward_type: action_level
  logprob_type: action_level
  entropy_type: action_level

  kl_beta: 0.0
  entropy_bonus: 0.0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  value_clip: 1.0

  gamma: 0.99
  gae_lambda: 0.95

env:
  group_name: "EnvGroup"
  train:
    rollout_epoch: 1
    total_num_envs: 64
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
  eval:
    is_eval: True
    rollout_epoch: 1
    total_num_envs: 16
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: 1

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ""
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 64
  global_batch_size: 256
  seed: 1234
  enable_offload: False

  model:
    model_type: "my_policy"

  optim:
    lr: 3.0e-4
    value_lr: 3.0e-4
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 0.5

  fsdp_config:
    strategy: "fsdp"
    sharding_strategy: "no_shard"
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

---

## 第六步：运行训练

### 6.1 设置环境变量

```bash
export REPO_PATH=/workspace/pjk/ELM/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$REPO_PATH:${PYTHONPATH}
# 如有需要
export MUJOCO_GL=egl
```

### 6.2 启动训练

推荐方式：
```bash
bash examples/embodiment/run_embodiment.sh my_env_ppo_my_policy
```

或直接调用 Hydra：
```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path $EMBODIED_PATH/config \
  --config-name my_env_ppo_my_policy
```

### 6.3 多机运行

1. 各节点设置唯一 rank：
   ```bash
   export RLINF_NODE_RANK=<0,1,2,...>
   ```
2. Head 节点：
   ```bash
   ray start --head --port=6379 --node-ip-address=<head_ip>
   ```
3. Worker 节点：
   ```bash
   ray start --address=<head_ip>:6379
   ```
4. 仅在 head 节点运行入口脚本，配置中设置 `cluster.num_nodes: <总数>`。

---

## 第七步：测试与调试

### 7.1 最小化验证

先创建最小测试脚本 `tests/unit_tests/test_my_env.py`：

```python
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from omegaconf import OmegaConf


def test_env_registration():
    cls = get_env_cls("my_env")
    print("Registered env:", cls)


def test_model_build():
    cfg = OmegaConf.create({
        "model_type": "my_policy",
        "obs_dim": 42,
        "action_dim": 8,
        "num_action_chunks": 1,
        "hidden_dim": 256,
        "add_value_head": True,
        "precision": "32",
    })
    model = get_model(cfg)
    print("Built model:", model)
```

### 7.2 常见调试点

| 问题 | 检查点 |
|---|---|
| 环境未注册 | `rlinf/envs/__init__.py` 中 `SupportedEnvType` 和 `get_env_cls` 分支 |
| 模型未注册 | `register_model` 是否被 worker 进程导入；检查 `rlinf/models/__init__.py` |
| 动作维度错误 | `rlinf/envs/action_utils.py` 中对应 `env_type` 的处理 |
| CUDA OOM | 减小 `total_num_envs`、`micro_batch_size`、`global_batch_size`；开启 `enable_offload` |
| 观测 key 不匹配 | 确保 `_wrap_obs` 返回的 key（如 `states`）与模型 `forward_inputs` 一致 |
| FSDP 包装问题 | 在 YAML 的 `fsdp_config` 中配置 `auto_wrap_policy` 或 `sharding_strategy: no_shard` |

---

## 第八步：合入主线时的附加工作（可选）

如果最终要把自定义环境/策略合入 RLinf 主仓库，还需要：

1. **安装脚本**
   - `requirements/install.sh`：把 env/model 加入 `SUPPORTED_ENVS`/`SUPPORTED_MODELS`，并添加 `install_my_env_env()` / `install_my_policy_model()`。
   - `pyproject.toml`：如有独立依赖，添加 optional dependency。
   - `requirements/embodied/sys_deps.sh`：如需系统级依赖。

2. **Docker**
   - `docker/Dockerfile`：新增 `embodied-my_env-my_policy-image` stage，一个 `RUN` 内完成所有 `install.sh` 调用。

3. **CI**
   - `.github/workflows/docker-build.yml`：新增 `build-embodied-my_env-my_policy` job。
   - `.github/workflows/embodied-e2e-tests.yml`：新增 e2e test job。
   - `tests/e2e_tests/embodied/my_env_ppo_my_policy.yaml`：最小 e2e 配置（`max_epochs: 1`，`total_num_envs: 2`）。

4. **文档**
   - 新增 `docs/source-en/rst_source/examples/embodied/my_env_my_policy.rst`。
   - 在对应分类 index（如 `simulators_index.rst` 或 `vla_wam_index.rst`）中加入隐藏 toctree。
   - 同步中文文档 `docs/source-zh/...`。
   - 更新 `README.md` / `README.zh-CN.md` 的 What's NEW。

---

## 文件创建/修改清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `rlinf/envs/my_env/__init__.py` | 新建 | 导出 `MyEnv` |
| `rlinf/envs/my_env/my_env.py` | 新建 | 自定义环境实现 |
| `rlinf/envs/__init__.py` | 修改 | 注册 `MY_ENV` |
| `rlinf/envs/action_utils.py` | 修改（可选） | 动作后处理 |
| `rlinf/models/embodiment/my_policy/__init__.py` | 新建 | 模型工厂 |
| `rlinf/models/embodiment/my_policy/my_policy.py` | 新建 | 自定义策略 |
| `rlinf/models/__init__.py` | 修改（运行时注册则可选） | 注册 builder |
| `rlinf/config.py` | 修改（运行时注册则可选） | 注册 `SupportedModel` |
| `examples/embodiment/config/env/my_env.yaml` | 新建 | 环境默认配置 |
| `examples/embodiment/config/model/my_policy.yaml` | 新建 | 策略默认配置 |
| `examples/embodiment/config/my_env_ppo_my_policy.yaml` | 新建 | 完整实验配置 |
| `tests/unit_tests/test_my_env.py` | 新建（可选） | 单元测试 |
| `tests/e2e_tests/embodied/my_env_ppo_my_policy.yaml` | 新建（合入主线） | e2e 测试配置 |
| `requirements/install.sh` | 修改（合入主线） | 安装逻辑 |
| `docker/Dockerfile` | 修改（合入主线） | Docker stage |
| `.github/workflows/*` | 修改（合入主线） | CI |
| `docs/source-{en,zh}/...` | 修改（合入主线） | 文档 |

---

## 关键参考文件

- 入口脚本：`examples/embodiment/train_embodied_agent.py`
- 同步 runner：`rlinf/runners/embodied_runner.py`
- 环境基线：`rlinf/envs/embodichain/embodichain_env.py`
- 策略基线：`rlinf/models/embodiment/mlp_policy/mlp_policy.py`
- 策略接口：`rlinf/models/embodiment/base_policy.py`
- 配置校验：`rlinf/config.py`（`validate_embodied_cfg`）
- 环境注册：`rlinf/envs/__init__.py`
- 动作处理：`rlinf/envs/action_utils.py`
- 模型注册：`rlinf/models/__init__.py`
- 新环境教程：`docs/source-en/rst_source/tutorials/extend/new_env.rst`
- 新模型教程：`docs/source-en/rst_source/tutorials/extend/new_model_fsdp.rst`
