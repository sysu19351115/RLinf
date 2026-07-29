# FSDP 忽略冻结参数实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让单卡 `NO_SHARD` OpenPI PyTorch Actor 的 FSDP FlatParameter 只管理可训练参数，避免冻结 VLM 产生整模型梯度缓冲区。

**Architecture:** 为 FSDP1 增加显式、默认关闭的 `ignore_frozen_params` 配置。模型完成冻结后、FSDP 包装前收集 `requires_grad=False` 参数，并通过 PyTorch FSDP 的 `ignored_states` 排除；冻结参数仍属于原始模型、保留在 state dict 和首次权重同步中。首版仅支持 `NO_SHARD`，避免未经验证的多 rank 一致性风险。

**Tech Stack:** PyTorch 2.7 FSDP1、OmegaConf/Hydra、pytest、RLinf PatchWeightSyncer。

---

## 设计决策

推荐 `ignored_states`，不推荐以下两个替代方案：

- `use_orig_params=False`：当前模型缺少有效 auto-wrap，根 FlatParameter 会混合冻结与可训练参数，容易触发 FSDP 的统一 `requires_grad` 限制。
- 按模块手工 auto-wrap：OpenPI 双 expert 参数交织在同一层结构中，按类包装容易把冻结 expert-0 和可训练 expert-1 再次放入同一 FSDP 单元，维护成本高。
- 直接迁移 FSDP2：现有 FSDP2 已有 `ignored_params` 基础能力，但迁移会同时改变 state dict、checkpoint 和运行时行为，超出本次显存修复范围。

## 安全边界

- 新配置默认 `false`，不改变其他算法。
- 首版只允许 `sharding_strategy: no_shard`；其他策略启用时应明确报错。
- 仅忽略 `requires_grad=False` 的参数，buffer 不在忽略集合中。
- 如果模型没有可训练参数，初始化必须失败，避免任务看似运行但完全不更新。
- 不自动在 223 启动训练；双节点硬件门禁由用户执行。

### Task 1: 增加配置契约和参数收集单元测试

**Files:**
- Modify: `examples/embodiment/config/hybrid_engines/fsdp.yaml`
- Modify: `tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py`
- Create: `tests/unit_tests/test_fsdp_ignore_frozen_params.py`

**Step 1: 写失败的默认配置测试**

断言公共 FSDP 配置默认包含：

```python
assert cfg.actor.fsdp_config.ignore_frozen_params is False
```

**Step 2: 写失败的参数收集测试**

构造包含共享参数、冻结层和可训练层的 tiny model，断言收集结果去重且只包含冻结 `nn.Parameter`。

**Step 3: 运行测试确认失败**

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_fsdp_ignore_frozen_params.py \
  tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py
```

预期：缺少配置项和收集函数导致失败。

**Step 4: 添加最小默认配置和收集函数**

在 `fsdp.yaml` 添加：

```yaml
ignore_frozen_params: False
```

在 FSDP1 strategy 中添加私有 helper，使用参数对象 identity 去重：

```python
def _collect_frozen_parameters(module: nn.Module) -> set[nn.Parameter]:
    return {parameter for parameter in module.parameters() if not parameter.requires_grad}
```

**Step 5: 运行测试确认通过**

预期：新增测试全部 PASS。

**Step 6: 提交**

```bash
git add examples/embodiment/config/hybrid_engines/fsdp.yaml \
  tests/unit_tests/test_fsdp_ignore_frozen_params.py \
  tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py
git commit -m "增加FSDP冻结参数忽略配置"
```

### Task 2: FSDP1 使用 ignored_states 排除冻结参数

**Files:**
- Modify: `rlinf/hybrid_engines/fsdp/strategy/fsdp.py`
- Modify: `tests/unit_tests/test_fsdp_ignore_frozen_params.py`

**Step 1: 写失败的包装前校验测试**

覆盖以下行为：

- `ignore_frozen_params=False` 返回 `None`，保持现状。
- `True + no_shard` 返回冻结参数集合。
- `True + full_shard` 抛出带修复建议的 `ValueError`。
- 全部参数冻结时抛出 `ValueError`。
- 当前 PyTorch FSDP 构造函数不支持 `ignored_states` 时快速失败。

**Step 2: 运行新测试确认失败**

预期：FSDP strategy 尚未构造 `ignored_states`。

**Step 3: 实现 guarded ignored_states**

在 `wrap_model()` 调用 FSDP 前：

```python
ignored_states = None
if self.cfg.fsdp_config.get("ignore_frozen_params", False):
    if sharding_strategy != ShardingStrategy.NO_SHARD:
        raise ValueError("ignore_frozen_params initially supports NO_SHARD only")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("FSDP model has no trainable parameters")
    ignored_states = _collect_frozen_parameters(model)
```

然后传入：

```python
FSDP(..., ignored_states=ignored_states)
```

记录冻结参数数量、元素数量和按 dtype 估算的字节数，便于硬件门禁确认。

**Step 4: 运行单元测试确认通过**

预期：所有 guard 和收集测试 PASS。

**Step 5: 提交**

```bash
git add rlinf/hybrid_engines/fsdp/strategy/fsdp.py \
  tests/unit_tests/test_fsdp_ignore_frozen_params.py
git commit -m "让FSDP忽略冻结模型参数"
```

### Task 3: 验证反向传播、state dict 和 checkpoint 契约

**Files:**
- Modify: `tests/unit_tests/test_fsdp_ignore_frozen_params.py`
- Modify if required: `rlinf/hybrid_engines/fsdp/strategy/fsdp.py`

**Step 1: 写单 rank CUDA 集成测试**

使用 tiny frozen tower + trainable expert，初始化单 rank process group 并包装 FSDP，验证：

```python
assert frozen_weight.grad is None
assert trainable_weight.grad is not None
assert managed_flat_numel == trainable_numel
```

测试无 CUDA 时显式 skip，不伪造 FSDP 行为。

**Step 2: 验证完整 state dict**

断言 FSDP state dict 同时包含冻结参数和可训练参数；保存后加载到新实例，逐 key 比较完全一致。

**Step 3: 验证 optimizer state**

执行一个 optimizer step，断言 optimizer 只为可训练参数创建状态，冻结参数值保持不变。

**Step 4: 运行测试**

```bash
.venv/bin/python -m pytest -q tests/unit_tests/test_fsdp_ignore_frozen_params.py
```

预期：CUDA 环境全部 PASS；CPU 环境仅 CUDA 集成用例 SKIP。

**Step 5: 提交**

```bash
git add tests/unit_tests/test_fsdp_ignore_frozen_params.py \
  rlinf/hybrid_engines/fsdp/strategy/fsdp.py
git commit -m "验证FSDP冻结参数状态兼容性"
```

### Task 4: 验证 PatchWeightSyncer 初始同步与增量同步

**Files:**
- Modify: `tests/unit_tests/test_openpi_pytorch_dagger_weight_sync.py`

**Step 1: 添加首次同步测试**

Actor state dict 必须包含冻结 VLM 和可训练 expert；receiver 初始化后逐 key 相等。

**Step 2: 添加增量同步测试**

只修改 action expert，验证 delta payload 不包含冻结 VLM，但 rollout 最终版本和 expert 权重正确更新。

**Step 3: 运行权重同步测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_openpi_pytorch_dagger_weight_sync.py \
  tests/unit_tests/test_weight_syncer.py
```

预期：首次全量同步和后续选择性同步全部 PASS。

**Step 4: 提交**

```bash
git add tests/unit_tests/test_openpi_pytorch_dagger_weight_sync.py
git commit -m "验证冻结参数权重同步契约"
```

### Task 5: 仅在 Dobot 两节点配置启用并完成门禁

**Files:**
- Modify: `examples/embodiment/config/dobot_hg_dagger_openpi_pytorch_2node.yaml`
- Modify: `tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py`

**Step 1: 写失败的两节点配置测试**

断言两节点为 `true`，单节点和其他配置仍为 `false`。

**Step 2: 启用配置**

```yaml
actor:
  fsdp_config:
    ignore_frozen_params: True
```

保留当前 BF16 `param/reduce/buffer_dtype` 作为低峰值基线；单 rank `NO_SHARD` 没有 FP32 reduce 的收益。

**Step 3: 运行静态门禁**

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_fsdp_ignore_frozen_params.py \
  tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py \
  tests/unit_tests/test_openpi_pytorch_dagger_weight_sync.py
.venv/bin/ruff check rlinf/hybrid_engines/fsdp/strategy/fsdp.py \
  tests/unit_tests/test_fsdp_ignore_frozen_params.py
git diff --check
```

**Step 4: 用户执行双节点硬件门禁**

由于 223 禁止写入、启动或停止进程，工程实现不得自动执行该步骤。用户运行后应确认：

- 日志报告被忽略的冻结参数约 2.5B，而 managed FlatParameter 仅覆盖 trainable expert。
- 模型加载、首次权重同步和至少一次 `optimizer.step()` 成功。
- backward 不再出现 6.25 GiB `_cast_grad_to_param_dtype` 分配。
- 保存 checkpoint 后重新启动能够恢复。
- rollout 版本递增且 action expert 权重发生预期变化。

**Step 5: 提交**

```bash
git add examples/embodiment/config/dobot_hg_dagger_openpi_pytorch_2node.yaml \
  tests/unit_tests/test_dobot_hg_dagger_openpi_pytorch_config.py
git commit -m "为Dobot训练忽略冻结参数"
```
