# ReBot 环境 Reward Model 集成规划

> 基于 RLinf 框架，为 ReBot 真实机器人环境引入 ResNet-based Reward Model，替代当前基于固定 `target_ee_pose` 的几何奖励函数。

---

## 1. 背景与动机

### 1.1 当前 ReBot Reward 的局限性

当前 `rlinf/envs/realworld/rebot/rebot_env.py` 中的奖励函数为：

```python
target_delta = np.abs(position - self.config.target_ee_pose)
is_in_target_zone = np.all(target_delta[:3] <= self.config.reward_threshold[:3])
```

该设计假设：
- 目标 TCP 位姿 `target_ee_pose` 是固定的；
- 成功标准只与机器人末端执行器（TCP）和固定目标点的距离有关。

**问题**：
- 当任务中物体位置变化时（例如插销需要插入当前孔位、抓取当前位置的物体），固定 `target_ee_pose` 无法表达动态目标；
- 无法处理需要视觉反馈才能判断成功与否的任务（如"插紧"、"对准"、"稳定抓取"）；
- 姿态阈值 `reward_threshold[3:]` 在代码中实际未被使用。

### 1.2 Reward Model 的优势

引入基于视觉的 Reward Model 后：
- 奖励由当前观测图像决定，不再依赖固定几何目标；
- 可以学习复杂的成功判据（视觉对齐、插入深度、接触状态等）；
- 对物体位置变化、光照变化等具有更好的泛化能力（前提是训练数据覆盖充分）。

---

## 2. RLinf Reward Model 架构

RLinf 在 `rlinf/models/embodiment/reward/` 中内置了三种图像奖励模型：

| 类型 | 类 | 适用场景 | 速度 | 训练数据需求 |
|---|---|---|---|---|
| `resnet` | `ResNetRewardModel` | 二分类成功/失败判断 | 快 | 中等（几百~几千张图像） |
| `vlm` | `VLMRewardModel` | 复杂语义任务（VLM 判断） | 慢 | 低（零样本或少量提示） |
| `history_vlm` | `HistoryVLMRewardModel` | 需历史帧的语义判断 | 慢 | 低 |

本规划选用 **`resnet`**，原因：
- 推理速度快，适合在线 RL 的实时 reward 计算；
- 训练成本可控，适合收集真实机器人数据；
- 与 HIL-SERL 工作方式一致，已有成熟实践。

### 2.1 ResNet Reward Model 输入输出

- **训练输入**：`(B, C, H, W)` 图像 + 二元标签 `1=success, 0=fail`
- **推理输入**：`observation["main_images"]`
- **输出**：`sigmoid(logits)`，即成功概率，作为 reward 值
- **默认图像尺寸**：`[3, 224, 224]`
- **预处理**：`NHWC -> NCHW`、归一化到 `[0,1]`、ImageNet 标准化

### 2.2 与 ReBot 环境的集成方式

当前 `rebot_env.py` 只支持内置 reward 计算。需要参考 `franka_env.py`，增加：
- `use_reward_model` 配置开关；
- `reward_worker_cfg` 配置；
- `_setup_reward_worker()` 初始化方法；
- `_calc_step_reward()` 中调用 reward worker 的分支。

---

## 3. 实现步骤

### Phase 1：收集成功/失败图像数据

#### 3.1.1 数据来源

**来源一：已有的人工演示数据集（LeRobot 格式）**

如果你已经用 LeRobot 格式保存了人工演示数据，数据目录结构通常为：

```bash
your_dataset/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── videos/
│       └── observation.images.main/
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ...
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   └── stats.json
```

每个 `episode_xxxxxx.parquet` 包含：
- `timestamp`：时间戳
- `observation.images.main`：视频帧索引
- `action`：动作
- `state`：机器人状态
- `episode_index`：episode 编号
- `frame_index`：帧在 episode 内的编号

**来源二：实时人工标注**

如果没有现成数据集，可以使用 RLinf 的键盘标注工具实时收集。


#### 3.1.2 标签策略

将每条演示轨迹视为一个成功 episode：

```python
def label_episode(frames, success_ratio=0.1):
    """
    frames: list of image frames in one episode
    success_ratio: 最后多少比例标为成功
    """
    n = len(frames)
    k = max(1, int(n * success_ratio))
    labels = [0] * (n - k) + [1] * k
    return labels
```

**推荐参数**：
- `success_ratio=0.1`：即每个 episode 最后 10% 的帧标为 1；
- 对于 `max_num_steps=240` 的 episode，约最后 24 帧为 1，前 216 帧为 0；
- 正负样本比例约 1:9，比只标最后一帧（1:239）更平衡。

#### 3.1.3 失败样本补充

仅有成功样本会导致模型无法区分失败状态。建议补充失败数据：
- 从成功 episode 的前半段采样作为"失败"状态（标签 0）；
- 或运行随机/次优策略收集真实失败 episode；
- 失败:成功比例建议控制在 2:1 到 5:1 之间。

#### 3.1.4 数据量建议

| 类别 | 最小数量 | 推荐数量 |
|---|---|---|
| 成功帧 | 200 | 500+ |
| 失败帧 | 400 | 1000+ |

#### 3.1.5 使用 RLinf 现成工具（可选）

RLinf 提供了 `examples/reward/realworld_collect_process_dataset.py`，支持通过键盘实时标注：
- `c`：当前帧 success
- `a`：当前帧 fail

如果已有 LeRobot 格式演示数据集，推荐直接使用 **3.2.3 节的 LeRobot 预处理脚本** 自动打标签。

### Phase 2：数据预处理

#### 3.2.1 输出格式

最终需要生成两个 PyTorch 文件：
```
logs/xxx/processed_reward_data/
├── train.pt
└── val.pt
```

每个 `.pt` 文件包含 `RewardDatasetPayload` 对象，其中有图像和标签。

#### 3.2.2 使用 RLinf 预处理脚本

如果原始数据是 `.pkl` episode 格式，使用：

```bash
python examples/reward/preprocess_reward_dataset.py \
    --raw-data-path logs/xxx/collected_data \
    --output-dir logs/xxx/processed_reward_data
```

该脚本支持：
- 每个 episode 均匀采样若干帧；
- 自动保留最后一帧；
- 训练/验证划分；
- 失败:成功比例重采样。

#### 3.2.3 LeRobot 数据集预处理脚本

针对 LeRobot 格式数据，推荐使用 `lerobot` 库直接加载并转换：

```python
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.data.datasets.reward_model import RewardDatasetPayload


def preprocess_lerobot_for_reward(
    dataset_path: str,
    output_dir: str,
    image_key: str = "observation.images.main",
    success_ratio: float = 0.1,
    val_split: float = 0.2,
    fail_episodes: list[int] | None = None,
):
    """
    将 LeRobot 数据集转换为 ResNet reward model 训练格式。

    Args:
        dataset_path: LeRobot 数据集根目录或 repo_id
        output_dir: 输出 train.pt / val.pt 的目录
        image_key: 图像 observation 的 key
        success_ratio: 每个成功 episode 最后多少比例标为 1
        val_split: 验证集比例
        fail_episodes: 指定哪些 episode 为失败（全部标 0），None 表示没有
    """
    dataset = LeRobotDataset(dataset_path)

    # 按 episode 分组
    episodes = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        ep_idx = int(item["episode_index"])
        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append((idx, item))

    images = []
    labels = []

    for ep_idx, frames in sorted(episodes.items()):
        frames = sorted(frames, key=lambda x: x[0])
        n = len(frames)
        k = max(1, int(n * success_ratio))
        is_fail = fail_episodes is not None and ep_idx in fail_episodes

        for i, (idx, item) in enumerate(frames):
            img = item[image_key]
            if isinstance(img, torch.Tensor):
                img = img.clone()
            else:
                img = torch.from_numpy(img)

            # 确保是 CHW
            if img.dim() == 3 and img.shape[-1] in [1, 3]:
                img = img.permute(2, 0, 1)

            if is_fail:
                label = 0
            else:
                label = 1 if i >= n - k else 0

            images.append(img)
            labels.append(label)

    # 训练/验证划分
    total = len(images)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    val_size = int(total * val_split)
    val_idx = set(perm[:val_size].tolist())
    train_idx = set(perm[val_size:].tolist())

    train_payload = RewardDatasetPayload(
        images=[images[i] for i in range(total) if i in train_idx],
        labels=[labels[i] for i in range(total) if i in train_idx],
        metadata={"num_train": len(train_idx), "num_val": val_size},
    )
    val_payload = RewardDatasetPayload(
        images=[images[i] for i in range(total) if i in val_idx],
        labels=[labels[i] for i in range(total) if i in val_idx],
        metadata={"num_train": len(train_idx), "num_val": val_size},
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_payload.save(str(output_dir / "train.pt"))
    val_payload.save(str(output_dir / "val.pt"))

    print(f"Saved {len(train_idx)} train, {val_size} val samples to {output_dir}")
    print(f"Positive ratio: {sum(labels)/len(labels):.3f}")


# 使用示例
preprocess_lerobot_for_reward(
    dataset_path="/path/to/your/lerobot/dataset",
    output_dir="logs/rebot_reward_data/processed",
    image_key="observation.images.main",
    success_ratio=0.1,
    val_split=0.2,
)
```

**注意事项**：
- `image_key` 必须和 LeRobot 数据集中的图像 key 一致，常见有 `observation.images.main`、`observation.images.wrist` 等；
- `ResNetRewardModel.preprocess_images()` 会自动 resize 到 `image_size`，所以预处理时不需要手动 resize；
- 如果 LeRobot 图像是 uint8，`preprocess_images()` 会自动除以 255；如果是 float 且最大值大于 1，也会自动归一化。

#### 3.2.4 检查预处理结果

```python
from rlinf.data.datasets.reward_model import RewardDatasetPayload

payload = RewardDatasetPayload.load("logs/rebot_reward_data/processed/train.pt")
print(f"Total samples: {len(payload.labels)}")
print(f"Positive samples: {sum(payload.labels)}")
print(f"Image shape: {payload.images[0].shape}")
print(f"Positive ratio: {sum(payload.labels)/len(payload.labels):.3f}")
```

### Phase 3：训练 ResNet Reward Model

#### 3.3.1 训练脚本

使用 RLinf 现成脚本：

```bash
python examples/reward/train_reward_model.py
```

#### 3.3.2 配置文件

修改 `examples/reward/config/reward_training.yaml`：

```yaml
data:
  train_data_paths: "logs/xxx/processed_reward_data/train.pt"
  val_data_paths: "logs/xxx/processed_reward_data/val.pt"
  num_workers: 4

actor:
  group_name: "RewardActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 64
  seed: 42
  enable_offload: false

  model:
    model_type: "resnet"
    model_path: null
    arch: "resnet18"          # 可选 resnet18/34/50/101/152
    pretrained: True
    hidden_dim: 256
    dropout: 0.1
    image_size: [3, 224, 224]  # ResNet 输入尺寸，预处理脚本无需手动 resize
    normalize: true
    precision: "fp32"

  optim:
    lr: 1.0e-4
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-8
    weight_decay: 1.0e-5
    clip_grad: 1.0
```

#### 3.3.3 训练输出

训练完成后 checkpoint 路径类似：
```
logs/reward_model/checkpoints/global_step_xxx/
└── model_state_dict/
    └── full_weights.pt
```

### Phase 4：改造 `rebot_env.py` 支持 Reward Model

参考 `rlinf/envs/realworld/franka/franka_env.py` 的实现。

#### 3.4.1 增加配置字段

在 `RebotArmRobotConfig` 中添加：

```python
use_reward_model: bool = False
reward_worker_cfg: Optional[dict] = None
reward_worker_node_rank: Optional[int] = None
reward_worker_node_group: Optional[str] = None
reward_worker_hardware_rank: int = 0
```

#### 3.4.2 初始化 Reward Worker

在 `RebotArmEnv.__init__` 中，硬件初始化之后添加：

```python
if not self.config.is_dummy:
    self._setup_reward_worker()
```

实现 `_setup_reward_worker`：

```python
def _setup_reward_worker(self):
    if not self.config.use_reward_model:
        return
    if self.config.reward_worker_cfg is None:
        raise ValueError(
            "use_reward_model=True but reward_worker_cfg is not provided"
        )

    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

    self._reward_worker = EmbodiedRewardWorker.launch_for_realworld(
        reward_cfg=self.config.reward_worker_cfg,
        node_rank=self.config.reward_worker_node_rank or self.node_rank,
        node_group_label=self.config.reward_worker_node_group,
        hardware_rank=self.config.reward_worker_hardware_rank,
        env_idx=self.env_idx,
        worker_rank=self.env_worker_rank,
    )
    self._reward_worker.init_worker().wait()
```

#### 3.4.3 修改 Reward 计算逻辑

在 `_calc_step_reward` 中增加 reward model 分支：

```python
def _calc_step_reward(self, observation, is_gripper_action_effective=False):
    if self.config.use_reward_model and not self.config.is_dummy:
        reward = self._reward_worker.compute_reward(observation)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        if reward >= 1.0:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        if self.config.enable_gripper_penalty and is_gripper_action_effective:
            reward -= self.config.gripper_penalty

        return max(0.0, min(1.0, float(reward)))

    # 保留原有逻辑
    ...
```

#### 3.4.4 确保 Observation 包含 `main_images`

`ResNetRewardModel.compute_reward` 要求 observation 字典中有 `main_images` key。需要检查 `RebotArmEnv._get_observation()` 返回的结构，可能需要将 `frames` 中的图像重命名为 `main_images`，或在 reward worker 调用前做映射。

### Phase 5：配置 ReBot RL 训练 YAML

在 `examples/embodiment/config/rebot_async_ppo_pi05.yaml` 中：

```yaml
env:
  train:
    override_cfg:
      is_dummy: False
      use_dense_reward: False       # 关闭旧几何奖励
      use_reward_model: True
      reward_worker_cfg:
        use_reward_model: True
        model:
          model_type: "resnet"
          model_path: "logs/reward_model/checkpoints/global_step_xxx/model_state_dict/full_weights.pt"
          arch: "resnet18"
          image_size: [3, 224, 224]
          normalize: true
          precision: "fp32"
```

注意区分两个层级的 `use_reward_model`：
- `env.train.override_cfg.use_reward_model`：控制 env 是否调用 reward worker；
- `reward_worker_cfg.use_reward_model`：控制 reward worker 内部是否启用 reward model。

### Phase 6：验证

#### 3.6.1 独立验证 Reward Model

写一个独立脚本加载训练好的模型，输入几张成功/失败图像：

```python
from rlinf.models.embodiment.reward import ResNetRewardModel
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.create({
    "model_type": "resnet",
    "arch": "resnet18",
    "model_path": ".../full_weights.pt",
    "image_size": [3, 224, 224],
    "normalize": True,
    "precision": "fp32",
})
model = ResNetRewardModel(cfg)
model.eval()

# 输入图像 (B, C, H, W)
with torch.no_grad():
    reward = model.compute_reward({"main_images": images})
print(reward)
```

期望：
- 成功图像 → probability 接近 1.0
- 失败图像 → probability 接近 0.0

#### 3.6.2 验证 Env Reward

在 `is_dummy=False` 模式下运行几个 env step，打印 reward，确认：
- 接近成功状态 reward 升高；
- 远离成功状态 reward 降低。

#### 3.6.3 接入 RL 训练

启动完整 RL 训练，观察：
- metric table 中 `env/rewards` 是否非零；
- 策略是否能根据 reward model 反馈收敛；
- 训练是否稳定（reward model 输出不会突变）。

---

## 4. 关键注意事项

| 问题 | 说明 | 建议 |
|---|---|---|
| 类别不平衡 | 成功帧远少于失败帧 | 最后 10% 帧标 1；使用 `pos_weight`；采样时控制正负比例 |
| 图像一致性 | 训练、推理时图像尺寸/预处理必须一致 | 统一 `image_size=[3,224,224]`，使用 ImageNet normalize |
| 观测 key | ResNet 只认 `main_images` | 在 rebot observation 中映射或重命名图像 key |
| dummy 模式 | dummy 下不应调用 reward worker | 保留原有 `_calc_step_reward` 的 dummy 分支 |
| 多环境 | `total_num_envs > 1` 时每个 env 一个 reward worker | 注意 GPU 显存和启动开销 |
| reward threshold | 可在配置中设置 `reward_threshold: 0.5` | 低于阈值时 reward 置 0，高于时保留概率值 |
| rollout 显存 | reward worker 会占用额外 GPU | 建议把 rollout 放到独立 GPU |

---

## 5. 与当前训练优化的关系

本 reward model 集成计划独立于之前的显存优化：

- 之前的优化（关闭 LoRA、`train_expert_only=True`、 rollout 拆卡）解决的是**训练能不能跑起来**；
- 本计划解决的是**奖励信号是否合理**，从而让策略学到正确的行为。

建议先完成显存优化确保训练稳定运行，再并行或后续进行 reward model 改造。

---

## 6. 后续扩展

| 方向 | 说明 |
|---|---|
| 渐进式标签 | 不只用 0/1，而是根据到成功状态的距离给 0~1 之间的连续标签 |
| 多视角图像 | 如果机器人有多个相机，可以训练多输入 ResNet 或 history_resnet |
| 在线更新 | 在 RL 训练过程中持续收集新数据，定期 finetune reward model |
| VLM fallback | 对难以学习的 corner case，用 VLMRewardModel 作为补充 |

---

*文档版本：2026-07-04*
*基于 RLinf commit 当前工作区状态*
