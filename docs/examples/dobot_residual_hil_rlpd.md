# Dobot CR5AF + 冻结 Pi0.5 + Residual HIL-RLPD 真机复现

在冻结 Pi0.5 名义动作先验之上训练小型 residual SAC/RLPD 策略，支持键盘实时
纠偏（HIL），训练动作严格等于真实执行动作。不改动 `dobot_zhiyu` SDK，不训练、
不同步、不保存 Pi0.5 权重。

拓扑：cloud（GPU：Actor 训练）＋ robot（GPU：Pi0.5 rollout + Dobot env）。
两端必须同提交、同 torch 版本、互相可达（局域网直连；单向 NAT 用
WireGuard/SSH 隧道，见 `docs/wireguard_build.md`、`docs/ssh_reverse_tunnel_build.md`）。

> 真机运行必须有人全程监护，工作空间无人员与障碍物、硬件急停可用。全程只允许
> ServoJ/ServoP，禁止 MoveJ/`--home-joints`。

## V1 契约（违反即拒绝启动）

- Dobot `action_mode=cartesian`、`state_mode=pose`、`action_dim=8`。
- `num_action_chunks=10`、`total_num_envs=1`、`auto_reset=false`、
  `terminal_padding.enabled=true`、`algorithm.adv_type=embodied_sac`。
- 单环境、单 pipeline stage、非 decoupled 模式。
- `Trajectory.actions` 只保存 residual action；nominal/sampled/commanded/
  executed 四类绝对动作字段严格分离。
- critic 训练动作必须由真实 `executed_actions` 相对 `nominal_actions` 反解；
  越界不静默裁剪（`out_of_support` 拒绝入缓冲）。

## 1. 环境安装（两节点分别执行）

```bash
cd /home/zylab/project/RLinf
sudo bash requirements/embodied/sys_deps.sh nvidia
bash requirements/install_local.sh --force embodied --model openpi --env dobot --no-root
uv pip install -e .
python tests/unit_tests/pytorch_test.py   # 验证 torch/GPU
```

robot 节点必须配 GPU（Pi0.5 rollout 在本机）；两端 `torch` 版本必须一致
（`pyproject.toml` 已锁定）。

## 2. 代码与模型准备（两节点）

```bash
git rev-parse HEAD              # 两端一致
git submodule status third_party/dobot_zhiyu
```

checkpoint 与 norm stats **两端都必须存在**（cloud 端 Actor 会 hash 这些文件
计算 base fingerprint；缺失会导致所有 transition 被拒）：

```bash
test -f checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/model.safetensors
test -f checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/config.json
test -f checkpoints/pi05_dobot_t265_pose_multiobject_800/40000_new/dobot_cf5af_t265_pose_multiobject_800_trimmed/norm_stats.json
```

## 3. 修改唯一配置入口

只改 `examples/embodiment/config/dobot_async_residual_hil_rlpd_pi05.yaml`：

- `dobot.ip`（当前 `192.168.5.2`）、`gripper_port`、`camera_serials`、
  `initial_joint_pos`（现场确认安全）。
- `algorithm.residual_hil_rlpd.safety_workspace_min_m/max_m`：按现场工作空间
  填写，这是执行前最后一道独立安全屏障。
- `run_id`：两端必须一致（默认 `dobot-residual-hil-rlpd`；新 run 换新值，防
  消息串用）。
- 相机编号每次插拔后确认：`ls -l /dev/video*`。

## 4. Robot 节点预检

```bash
ls -l /dev/ttyACM0 /dev/video* /dev/input/event*
test -r /dev/ttyACM0 && test -w /dev/ttyACM0
test -r /dev/video0 && test -w /dev/video0
nc -vz 192.168.5.2 29999 && nc -vz 192.168.5.2 30004
df -h / /home
```

确认物理键盘 event 设备（Linux evdev，无需 DISPLAY）：

```bash
export RLINF_KEYBOARD_DEVICE=/dev/input/eventX   # 替换为验证过的设备
test -r "$RLINF_KEYBOARD_DEVICE"
```

相机/机械臂只读验证（不运动、不控制夹爪）：

```bash
.venv/bin/python rlinf/envs/realworld/dobot/verify_env.py --camera-only \
  --camera-serial /dev/video0 --camera-resolution 640 480 --camera-fps 30 --camera-fourcc MJPG

.venv/bin/python rlinf/envs/realworld/dobot/verify_env.py \
  --ip 192.168.5.2 --skip-motion --skip-gripper --skip-camera
```

## 5. 离线演示转换（可选，先 dry-run）

```bash
.venv/bin/python toolkits/embodiment/convert_dobot_hil_to_residual_replay.py \
  --episodes episodes.npz --nominals nominals.npz --output ./replay --dry-run
```

`episodes.npz` 默认禁止 object array（pickle）；确属可信数据才可加
`--allow-pickle`。`nominals.npz` 必须由同一冻结 Pi0.5 checkpoint 对每个 chunk
起始观测重算。dry-run 通过后去掉 `--dry-run` 生成正式 replay。

## 6. 回归测试（cloud 节点，不连真机）

```bash
.venv/bin/pytest -q tests/unit_tests/test_dobot_residual_*.py \
  tests/unit_tests/test_residual_*.py \
  tests/unit_tests/test_convert_dobot_hil_residual_replay.py \
  tests/unit_tests/test_checkpoint_utils.py
.venv/bin/pytest -q tests/integration_tests/test_dobot_residual_hil_rlpd_dummy.py
.venv/bin/pytest -q tests/unit_tests/test_dobot_keyboard_intervention.py \
  tests/unit_tests/test_dobot_async_ppo_pi05_pytorch_config.py \
  tests/unit_tests/test_dobot_hg_dagger_contract.py
```

任一失败不得上真机。

## 7. 启动双节点 Ray（环境变量必须先于 ray start）

cloud（`192.168.3.223`）：

```bash
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp130s0        # 以 ip -br -4 addr 实测为准
export RLINF_RUN_ID=dobot-residual-hil-rlpd
ray stop
ray start --head --port=6379 --node-ip-address=192.168.3.223 --disable-usage-stats
```

robot（`192.168.3.224`）：

```bash
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp130s0
export RLINF_KEYBOARD_DEVICE=/dev/input/eventX
export RLINF_RUN_ID=dobot-residual-hil-rlpd
ray stop
ray start --address=192.168.3.223:6379 --node-ip-address=192.168.3.224 --disable-usage-stats
```

回 cloud 执行 `ray status` 确认两节点 alive。同一机械臂只能被一个训练/评估
进程占用。

## 8. 真机分级验收（Gate A-F，每级通过后才能进下一级）

- **Gate A 只读基线**：不启用 residual，测 Pi0.5 推理/控制周期；目标 residual
  推理 p99 < 20ms、总 chunk 周期退化 < 10%。
- **Gate B shadow**：真机只执行 nominal，后台计算 residual/commanded 并审计；
  ≥20 episode 无 NaN/shape/fingerprint 错误，zero-residual round-trip 全过。
- **Gate C base-only HIL 收数**：`residual_scale=0`，允许键盘接管；抽查 ≥100
  个 transition 的 executed/反解/双缓冲/reward 标签。
- **Gate D 小范围 deterministic**：只开 actor mean，policy limit 从
  1mm/0.5° 起步；确认 workspace、controller rejection、夹爪锁存、
  MODEL→ENGAGE→MODEL handoff 安全（评估链路已实现 deterministic 组合）。
- **Gate E 小范围随机训练**：critic-only → 按 ramp 开 actor；Q/TD 非有限、
  residual 饱和、out-of-support、controller rejection 或 autonomous success
  下降时自动 pause。
- **Gate F 自主评估**：`allow_motion_intervention=false` + deterministic
  actor；assisted 不计入 autonomous success；报告 base-only 与 base+residual
  对照。

## 9. 启动训练（cloud 节点执行）

```bash
.venv/bin/python examples/embodiment/train_async.py \
  --config-name dobot_async_residual_hil_rlpd_pi05
```

键盘语义：Enter=成功（当前 chunk 执行完后结束 episode）、Backspace=失败、
Esc=安全停止、硬件急停优先。Enter/Backspace 不是急停；当前 chunk 最多继续约
`10 / 30 ≈ 0.33s`。

首轮必须有人确认：

1. Actor 在 cloud、Rollout 和 Env 在 robot 启动，模型与 norm stats 加载成功。
2. 相机画面、pose state、8D 动作无 shape/NaN 错误。
3. 按 Enter/Backspace 后只完成当前 chunk，随后保持静止并进入 reset。
4. `online_size/demo_size` 增长，`malformed_message_dropped` 为 0，
   `buffer_rejected` 原因可解释。
5. Q/TD/alpha/两个 grad norm 均有限；`residual_scale` 按 ramp 上升；
   `safety_violations` 为 0；Pi0.5 权重逐字节不变。

## 10. 评估

评估在训练中按 `runner.val_check_interval` 触发（如 5，需整除
`save_interval=10`），使用 `env.eval`（`allow_motion_intervention=false`），
rollout 侧以 **deterministic** 模式组合 residual 并透传夹爪 bypass：

```bash
.venv/bin/python examples/embodiment/train_async.py \
  --config-name dobot_async_residual_hil_rlpd_pi05 \
  runner.val_check_interval=5
```

观察 `eval/*` 指标与 `autonomous_success`（assisted 不计入）。Gate F 对照：
分别记录 base-only（residual 未启用）与 base+residual 的自主成功率。

## 11. 恢复训练

```bash
... --config-name dobot_async_residual_hil_rlpd_pi05 \
  runner.resume_dir=/绝对路径/.../global_step_<N>
```

resume 目录必须含 `actor/COMPLETED`；manifest 缺文件、hash/大小不符或存在
未知文件都会拒绝加载。

## 12. 紧急停止与回滚

- 紧急：硬件急停 > Esc > 终止进程（Ctrl-C + `ray stop`）。
- 自动 fail-closed：NaN/Inf、safety violation、连续 controller rejection、
  消息 schema/run_id 错误、demo 采样比例偏离、键盘断连、Q 发散——一律回退
  base-only（不静默裁剪，不继续训练）。
- 回滚：不删数据；异常 run 保留日志/checkpoint；新 run 使用新时间戳目录；
  需要纯 base-only 时保持 `residual_scale=0`（不进入 actor ramp）。

## 13. 关键指标

`residual_transition` 计数、`online/demo size` 与 50/50 采样审计、policy lag、
`residual_scale`/`gripper_enabled`、Q ensemble mean/min/max/disagreement、
TD error、buffer bytes/watermark、`safety_violations`、`autonomous_success`
与 `operator_*` 终止原因。
