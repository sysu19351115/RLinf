# GYM_ALOHA + GRPO 参数约束文档

本文档记录了将 `gym_aloha` 仿真环境集成到 RLinf GRPO 训练过程中，通过多轮调试总结出的**强制性参数约束**和**推荐配置**。

---

## 目录

1. [强制性约束](#1-强制性约束)
2. [推荐配置](#2-推荐配置)
3. [约束推导（以 max_steps=300 为例）](#3-约束推导以-max_steps300-为例)
4. [奖励设计原则](#4-奖励设计原则)
5. [已知问题](#5-已知问题)
6. [启动命令](#6-启动命令)

---

## 1. 强制性约束

| # | 约束 | 来源 | 说明 |
|---|------|------|------|
| C1 | `max_steps % num_action_chunks == 0` | `rlinf/config.py:980` | rollout epoch 步数必须被 action chunk 大小整除 |
| C2 | `total_envs % env_world_size == 0` | `rlinf/config.py:961` | 总环境数必须能被 env worker 数量整除 |
| C3 | `total_envs / world_size / stage_num % group_size == 0` | `rlinf/config.py:974` | 每个 worker 的环境数必须能被 GRPO group_size 整除 |
| C4 | `global_batch % (micro_batch × actor_world_size) == 0` | `rlinf/config.py:1368` | 全局 batch 大小必须是 micro_batch × actor rank 数的整数倍 |
| C5 | `rollout_per_rank % (global_batch / actor_world_size) == 0` | `rlinf/workers/actor/fsdp_actor_worker.py:1319` | 每个 rank 的 rollout 数据量必须能被其 batch 大小整除 |
| C6 | `micro_batch ≤ global_batch` | 显式 | micro batch 不能超过 global batch |

### 约束 C5 的展开

```
rollout_per_rank = total_envs × rollout_epoch × (max_steps / num_action_chunks) / actor_ranks
batch_per_rank   = global_batch / actor_ranks

要求: rollout_per_rank % batch_per_rank == 0
```

---

## 2. 推荐配置

以 `max_steps=300`, `num_action_chunks=50`, 8 GPU (`actor,env,rollout: all`) 为例：

| 参数 | 值 | 验证 |
|------|-----|------|
| `max_steps_per_rollout_epoch` | 300 | C1: 300%50=0 ✓ |
| `max_episode_steps` | 300 | 任务通常在 300 步内完成 |
| `total_num_envs` | 64 | C2: 64%8=0 ✓, C3: 8/1%8=0 ✓ |
| `group_size` | 8 | C3: 8%8=0 ✓ |
| `rollout_epoch` | **8** | C5: 48×8=384, 384%128=0 ✓ |
| `micro_batch_size` | 128 | FSDP 基础 micro batch |
| `global_batch_size` | 1024 | C4: 1024%(128×8)=0 ✓, C6 ✓ |
| `use_rel_reward` | False | raw 分层奖励（0-4），给 GRPO 最丰富的信号 |
| `use_step_penalty` | False | 不需要差分惩罚 |
| `reward_coef` | 0.25 | 每步 reward ~0-1 |
| `auto_reset` | False | 当前 GRPO 训练已验证可行 |

### GRPO 特有参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `adv_type` | `grpo` | 组内归一化，不需 critic |
| `loss_type` | `actor` | PPO actor loss，无需 value head |
| `kl_beta` | 0.0 | GRPO 不需要 KL 正则 |
| `entropy_bonus` | 0 | GRPO 不需要额外探索激励 |
| `filter_rewards` | False | raw 奖励不做滤波 |
| `update_epoch` | 1 | 每批数据用 1 次 |

---

## 3. 约束推导（以 max_steps=300 为例）

### 已知量

```
num_action_chunks = 50
actor_world_size  = 8 (8 GPUs, "all" placement)
env_world_size    = 8
stage_num         = 1
max_steps         = 300
total_envs        = 64
group_size        = 8
micro_batch       = 128
```

### C1 → chunks 数

```
chunks_per_epoch = max_steps // num_action_chunks = 300 // 50 = 6
```

### C2 → 环境数可分配

```
64 % 8 = 0 ✓
envs_per_worker = 64 / 8 / 1 = 8
```

### C3 → GRPO group 可分组

```
envs_per_worker / stage_num % group_size = 8 / 1 % 8 = 0 ✓
```

### C4 → global_batch 可选值

```
global_batch % (128 × 8) == 0  →  global_batch ∈ {1024, 2048, ...}
取最小: global_batch = 1024
batch_per_rank = 1024 / 8 = 128
```

### C5 → rollout_epoch 求解

```
rollout_per_rank = 64 × rollout_epoch × 6 / 8 = 48 × rollout_epoch
request: 48 × rollout_epoch % 128 == 0
LCM(48, 128) = 384
rollout_epoch = 384 / 48 = 8
```

### 验证

```
C5: rollout_per_rank = 48 × 8 = 384,  384 % 128 = 0 ✓
C6: micro_batch(128) ≤ global_batch(1024) ✓
```

---

## 4. 奖励设计原则

### gym_aloha 的原始奖励特性

gym_aloha 的奖励是**分层状态值**（每步 0-4），不是累积增量：

| reward | 含义 |
|--------|------|
| 0 | 无接触 |
| 1 | 右夹爪碰到方块 |
| 2 | 右夹爪抓起方块（离桌） |
| 3 | 左夹爪也碰到方块 |
| 4 | 左夹爪抓起，转移成功 |

### 为什么用 `use_rel_reward=False`

1. **GRPO 不需要稀疏信号。** GRPO 在组内做归一化（`adv = (score - group_mean) / group_std`），不需要绝对 reward 值。每步都有信号的 raw reward 比差分 reward 更稠密，对 GRPO 更友好。
2. **差分 reward 对 PPO+critic 有益（防止 value model 需预测累积 reward），但 GRPO 没有 value model。**
3. **raw reward 不会导致"摸鱼"。** GRPO 在 8 个 episode 的组内比较，progressing episode 天然获得高于 stagnant episode 的 advantage。

### 为什么需要 `reward_coef=0.25`

虽然 GRPO 不依赖绝对 reward 量级（组内归一化），但大幅异常的 reward 会产生极端 advantage 值。`coef=0.25` 使每步 reward ≈0-1，episode score ≈0-300，与 Libero 量级接近。

---

## 5. 已知问题

### 5.1 `auto_reset=True` 不兼容

`auto_reset=True` 会导致 GRPO 的 `loss_mask` 计算出错（成功 episode 提前终止，生成多个 episode 边界，数据形状与固定长度 episode 的假设不匹配）。当前安全做法：`auto_reset=False`。

### 5.2 仿真时间瓶颈

每 global step 约 400-680s，其中 env 交互占 94%。优化方向：
- 缩短 `max_steps`（已从 500 优化到 300）
- `auto_reset=True` 后（待修复兼容性）成功 episode 不再空跑
- 子进程并行 env（需较大工程改动）

### 5.3 SFT 在 RLinf 中的表现退化

独立测试 90% 成功率 → RLinf 训练中 ~35%。可能原因：ActionChunkBroker_RTC（实时修正）在 RLinf 中不可用，50 步的固定 action chunk 无法中途修正。

---

## 6. 启动命令

### 安装

```bash
bash requirements/install_local.sh embodied --model openpi --env gym_aloha
```

### 训练（每次重启前先 `ray stop`）

```bash
ray stop
sleep 5
bash examples/embodiment/run_embodiment.sh gym_aloha_grpo_openpi_pi0 ALOHA
```

### 评估（纯推理，不训练，验证 SFT checkpoint 或训练后的模型）

```bash
bash examples/embodiment/run_eval_gym_aloha.sh [model_path] [num_action_chunks]

# 示例（使用默认参数）：
bash examples/embodiment/run_eval_gym_aloha.sh
```
评估完成后查看 `eval_results/<timestamp>/metrics.log` 中的 `success_once` 均值，视频在 `eval_results/<timestamp>/video/eval`。

评估完成后查看 `eval_results/metrics.log` 中的 `success_once` 均值。

### checkpoint 路径要求

```
checkpoints/pi0_aloha_sim_pytorch/
├── model.safetensors
├── config.json
├── assets/
│   └── lerobot/aloha_sim_transfer_cube_human/
│       └── norm_stats.json
└── lerobot/                                    ← symlink
    └── aloha_sim_transfer_cube_human → ../assets/lerobot/aloha_sim_transfer_cube_human
```

创建 symlink：
```bash
mkdir -p checkpoints/pi0_aloha_sim_pytorch/lerobot
ln -s ../assets/lerobot/aloha_sim_transfer_cube_human \
      checkpoints/pi0_aloha_sim_pytorch/lerobot/aloha_sim_transfer_cube_human
```

---

## 附录：约束速查卡

| 参数 | 合法值/条件 | 验证方法 |
|------|-----------|---------|
| `max_steps` | 50 的倍数 | `X % 50 == 0` |
| `total_envs` | 8 的倍数 | `X % 8 == 0` |
| `envs_per_worker` | group_size 的倍数 | `(total/8) % g == 0` |
| `global_batch` | micro_batch × 8 的倍数 | `B % (M×8) == 0` |
| `rollout_epoch` × `chunks` × 8 | batch_per_rank 的倍数 | `48×E % (B/8) == 0` |
