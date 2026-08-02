# Dobot Residual HIL-RLPD

在冻结 Pi0.5 之上训练小型 residual 策略，支持键盘实时纠偏的在线强化学习
（chunk 级 RLPD）。V1 只支持 Dobot `cartesian/pose` 单真机环境、
`num_action_chunks=10`、`algorithm.adv_type=embodied_sac`。

## 核心不变量

- `Trajectory.actions` 只保存 residual action；绝对 8D 动作只出现在
  nominal/commanded/executed 独立字段。
- critic 训练动作必须由真实 `executed_actions` 相对 `nominal_actions` 反解。
- 人工接管 transition 双写 online + demo；非人工只进 online；采样严格 50/50。
- Pi0.5 全程冻结：rollout 中 `eval()` + `requires_grad_(False)`，learner 不
  实例化、不同步、不保存 base 权重。
- 越界动作不静默裁剪：超出 `data_limit` 的 transition 标记无效并拒绝入缓冲。

## 运行

```bash
.venv/bin/python examples/embodiment/train_async.py \
  --config-name dobot_async_residual_hil_rlpd_pi05
```

真机分级验收（Gate A-F）见 `docs/plans/2026-08-02-dobot-residual-hil-rlpd.md`。
dummy 验证：

```bash
.venv/bin/python -m pytest -q \
  tests/integration_tests/test_dobot_residual_hil_rlpd_dummy.py
```

## 离线演示转换

```bash
.venv/bin/python toolkits/embodiment/convert_dobot_hil_to_residual_replay.py \
  --episodes episodes.npz --nominals nominals.npz --output ./replay --dry-run
```

`--nominals` 由冻结 Pi0.5 checkpoint 对每个 chunk 起始观测重新推理生成。
