# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EnvWorker residual chunk finalization tests (deferred next-nominal flow)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.transition import (
    CHUNK_LEN,
    ResidualChunkTransition,
)
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.realworld.realworld_env import _stack_residual_feedback
from rlinf.workers.env.env_worker import EnvWorker


def _cfg():
    return OmegaConf.create(
        {
            "algorithm": {
                "gamma": 0.99,
                "residual_hil_rlpd": {
                    "translation_data_limit_m": [0.1, 0.1, 0.1],
                    "rotation_data_limit_deg": [30.0, 30.0, 30.0],
                    "gripper_debounce_chunks": 2,
                    "safety_hold_chunks": 3,
                    "safety_workspace_min_m": [-0.5, -0.5, 0.0],
                    "safety_workspace_max_m": [0.5, 0.5, 0.5],
                    "safety_max_translation_delta_m": 0.02,
                    "safety_max_rotation_delta_deg": 10.0,
                },
            },
            "actor": {
                "model": {
                    "base_policy": {"model_path": "base-fp"},
                    "openpi_data": {"norm_stats_path": "norm.json"},
                }
            },
        }
    )


def _make_worker() -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = _cfg()
    worker._logger = logging.getLogger("test_residual_envworker_finalize")
    worker._residual_pending_ctx = None
    worker._residual_chunk_counter = 0
    worker._pending_residual_transitions = []
    worker._residual_gripper_inhibit_remaining = 0
    worker._residual_gripper_suppressed_mask = None
    worker._residual_safety_hold_remaining = 0
    worker._residual_safety_violations = 0
    worker._residual_safety_barrier = None
    worker._residual_safety_violated_chunk = False
    return worker


def _chunk_audit_and_feedback(
    rng: np.random.Generator,
    codec: ResidualCodec,
    k: int,
    *,
    terminate: bool = False,
):
    """Build a valid audit dict + per-step feedback for one chunk."""
    h = CHUNK_LEN
    nominal = np.stack(
        [
            np.concatenate(
                [
                    rng.uniform(-0.2, 0.2, size=2),
                    [rng.uniform(0.05, 0.3)],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5],
                ]
            )
            for _ in range(h)
        ]
    ).astype(np.float32)
    # Keep residuals inside the *policy* envelope so the independent safety
    # barrier (0.02 m / 10 deg) never trips on nominal test data.
    u = rng.uniform(-0.1, 0.1, size=(h, 6)).astype(np.float32)
    commanded = np.zeros((h, 8), dtype=np.float32)
    for i in range(h):
        pose = codec.compose(
            torch.as_tensor(nominal[i, :7], dtype=torch.float32),
            torch.as_tensor(u[i], dtype=torch.float32),
        ).numpy()
        commanded[i] = np.concatenate([pose, [nominal[i, 7]]])
    executed = commanded.copy()
    accepted = np.zeros(h, dtype=bool)
    accepted[:k] = True
    terminations = np.zeros(h, dtype=bool)
    if terminate:
        terminations[k - 1] = True
    audit = {
        "nominal_actions": nominal,
        "sampled_arm_residual": u,
        "sampled_gripper_mode": np.zeros(h, dtype=np.int64),
        "commanded_actions": commanded,
        "gripper_bypass_mask": np.zeros(h, dtype=bool),
        "policy_version": 3,
    }
    steps = [
        {
            "executed_action": executed[i],
            "action_command_accepted": bool(accepted[i]),
            "human_intervention": False,
            "handoff_hold": False,
            "gripper_bypass": False,
            "reward": 0.0,
            "termination": bool(terminations[i]),
            "truncation": False,
            "reward_label_valid": True,
        }
        for i in range(h)
    ]
    return audit, _stack_residual_feedback(steps, chunk_size=h, action_dim=8)


def _obs(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "main_images": np.full((8, 8, 3), value, dtype=np.uint8),
        "prev_states": np.full(8, value, dtype=np.float32),
    }


def _dummy_transition() -> ResidualChunkTransition:
    h = CHUNK_LEN
    zero8 = np.zeros((h, 8), dtype=np.float32)
    return ResidualChunkTransition(
        curr_obs=_obs(0.0),
        next_obs=_obs(1.0),
        nominal_actions=zero8.copy(),
        next_nominal_actions=zero8.copy(),
        sampled_arm_residual=np.zeros((h, 6), dtype=np.float32),
        sampled_gripper_mode=np.zeros(h, dtype=np.int64),
        commanded_actions=zero8.copy(),
        gripper_bypass_mask=np.zeros(h, dtype=bool),
        executed_actions=zero8.copy(),
        actions_arm=np.zeros((h, 6), dtype=np.float32),
        actions_gripper=np.zeros((h, 3), dtype=np.float32),
        rewards=np.zeros(h, dtype=np.float32),
        executed_action_mask=np.ones(h, dtype=bool),
        human_intervention_mask=np.zeros(h, dtype=bool),
        handoff_hold_mask=np.zeros(h, dtype=bool),
        terminations=np.zeros(h, dtype=bool),
        truncations=np.zeros(h, dtype=bool),
        bootstrap_mask=np.array([True]),
        discount_steps=np.array([h]),
        discounted_return=0.0,
        transition_valid=np.array([True]),
        reward_label_valid=np.array([True]),
        source=0,
        policy_version=np.array([3]),
        base_fingerprint="base-fp",
        codec_fingerprint="codec-fp",
        episode_id=7,
        chunk_id=120,
    )


class _AuditedRolloutResult:
    def __init__(self, audit: dict, nominal: np.ndarray):
        self.audit_info = audit
        self._nominal = nominal


def test_deferred_finalize_two_chunks_then_flush(monkeypatch):
    """Chunk k is finalized only when chunk k+1's nominal is known; the final
    chunk is flushed with ``next_nominal=None`` and stage_id is propagated."""
    worker = _make_worker()
    calls = []

    def fake_finalize(
        rollout_result,
        env_output,
        stage_id,
        *,
        chunk_start_obs,
        next_nominal=None,
        safety_violated=False,
    ):
        calls.append(
            (
                rollout_result._nominal.tolist(),
                stage_id,
                chunk_start_obs["tag"],
                None if next_nominal is None else next_nominal.tolist(),
            )
        )
        return "T" + str(len(calls))

    def fake_audit(rollout_result):
        return np.asarray(rollout_result._nominal, dtype=np.float32)

    monkeypatch.setattr(worker, "_finalize_residual_transition", fake_finalize)
    monkeypatch.setattr(worker, "_audit_nominal_actions", fake_audit)

    nominal0 = np.full((CHUNK_LEN, 8), 0.0, dtype=np.float32)
    nominal1 = np.full((CHUNK_LEN, 8), 1.0, dtype=np.float32)
    rr0 = _AuditedRolloutResult({"dummy": True}, nominal0)
    rr1 = _AuditedRolloutResult({"dummy": True}, nominal1)
    feedback = _stack_residual_feedback(
        [
            {
                "executed_action": np.zeros(8, dtype=np.float32),
                "action_command_accepted": True,
                "human_intervention": False,
                "handoff_hold": False,
                "gripper_bypass": False,
                "reward": 0.0,
                "termination": False,
                "truncation": False,
                "reward_label_valid": True,
            }
        ],
        chunk_size=CHUNK_LEN,
        action_dim=8,
    )
    env0 = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback, "episode_id": [3]},
    )
    env1 = EnvOutput(
        obs=_obs(1.0),
        env_infos={"residual_feedback": feedback, "episode_id": [3]},
    )
    obs0 = {**_obs(0.0), "tag": "chunk0-start"}
    obs1 = {**_obs(1.0), "tag": "chunk1-start"}

    # First chunk: deferred, nothing finalized yet.
    worker._residual_observe_chunk(rr0, env0, obs0, stage_id=0)
    assert worker._residual_pending_ctx is not None
    assert worker._residual_chunk_counter == 0
    assert calls == []

    # Second chunk: previous chunk finalizes with chunk1's nominal, and the
    # real stage_id flows through.
    worker._residual_observe_chunk(rr1, env1, obs1, stage_id=1)
    assert worker._residual_chunk_counter == 1
    assert len(calls) == 1
    finalized_nominal, chunk0_stage, start_tag, next_nominal = calls[0]
    assert np.allclose(finalized_nominal, nominal0.tolist())
    # The deferred chunk was observed on stage 0; its own stage is preserved.
    assert chunk0_stage == 0
    assert start_tag == "chunk0-start"
    assert np.allclose(next_nominal, nominal1.tolist())

    # Flush: the last (potentially terminal) chunk is still finalized.
    worker._flush_residual_pending()
    assert worker._residual_pending_ctx is None
    assert worker._residual_chunk_counter == 2
    assert len(calls) == 2
    assert np.allclose(calls[1][0], nominal1.tolist())
    assert calls[1][1] == 1  # flushed chunk was observed on stage 1
    assert calls[1][3] is None
    assert worker._pending_residual_transitions == ["T1", "T2"]


def test_real_finalize_keeps_terminal_chunk_and_uses_next_nominal():
    """End-to-end EnvWorker plumbing: audit + feedback become valid chunk
    transitions, the terminal chunk is not dropped, and chunk k's transition
    carries chunk k+1's nominal as next_nominal."""
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(0)
    audit0, feedback0 = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    audit1, feedback1 = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN, terminate=True)

    env0 = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback0, "episode_id": [7]},
    )
    env1 = EnvOutput(
        obs=_obs(1.0),
        env_infos={"residual_feedback": feedback1, "episode_id": [7]},
    )
    worker._residual_observe_chunk(
        _AuditedRolloutResult(audit0, np.zeros((CHUNK_LEN, 8))),
        env0,
        _obs(0.0),
        stage_id=0,
    )
    worker._residual_observe_chunk(
        _AuditedRolloutResult(audit1, np.zeros((CHUNK_LEN, 8))),
        env1,
        _obs(1.0),
        stage_id=0,
    )
    worker._flush_residual_pending()

    assert worker._residual_chunk_counter == 2
    assert len(worker._pending_residual_transitions) == 2
    t0, t1 = worker._pending_residual_transitions
    assert t0.chunk_id == 0
    assert t1.chunk_id == 1
    assert t0.episode_id == 7
    assert t1.episode_id == 7
    # Chunk 0 bootstraps onto chunk 1's nominal.
    assert bool(t0.bootstrap_mask[0])
    np.testing.assert_allclose(
        np.asarray(t0.next_nominal_actions),
        np.asarray(audit1["nominal_actions"]),
    )
    # Terminal chunk is kept and carries its termination.
    assert not bool(t1.bootstrap_mask[0])
    assert bool(np.asarray(t1.terminations)[-1])


def test_intervention_sets_gripper_inhibit():
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(1)
    _, feedback = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    # Human intervention on one step.
    feedback["human_intervention_mask"][3] = True
    env = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback, "episode_id": [7]},
    )
    worker._residual_observe_chunk(
        _AuditedRolloutResult({"dummy": True}, np.zeros((CHUNK_LEN, 8))),
        env,
        _obs(0.0),
        stage_id=0,
    )
    assert worker._residual_gripper_inhibit_remaining == 2  # debounce chunks


def test_controller_rejection_sets_gripper_inhibit():
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(2)
    _, feedback = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    feedback["executed_action_mask"][5] = False  # controller rejected a step
    env = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback, "episode_id": [7]},
    )
    worker._residual_observe_chunk(
        _AuditedRolloutResult({"dummy": True}, np.zeros((CHUNK_LEN, 8))),
        env,
        _obs(0.0),
        stage_id=0,
    )
    assert worker._residual_gripper_inhibit_remaining == 2


def test_pre_execution_safety_violation_holds_nominal():
    """P2-5: the barrier runs before the servo; a violating chunk is replaced
    with nominal in-place, a hold is entered, and the gripper label is marked
    suppressed (KEEP)."""
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(3)
    audit, feedback = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    # Command a pose far outside the workspace to trip the barrier.
    audit["commanded_actions"][:, 0] += 0.9
    audit["nominal_actions"][:, 0] += 0.9
    nominal = np.asarray(audit["nominal_actions"], dtype=np.float32)
    commanded = np.asarray(audit["commanded_actions"], dtype=np.float32)
    actions = torch.as_tensor(commanded, dtype=torch.float32).unsqueeze(0)
    rollout_result = _AuditedRolloutResult(audit, np.zeros((CHUNK_LEN, 8)))
    rollout_result.actions = actions
    bypass = np.ones(CHUNK_LEN, dtype=bool)

    new_bypass, suppressed = worker._residual_safety_check(
        rollout_result, nominal, bypass
    )

    assert worker._residual_safety_violations == 1
    assert worker._residual_safety_violated_chunk is True
    assert worker._residual_safety_hold_remaining == 3
    assert suppressed is not None and suppressed.all()
    assert not new_bypass.any()
    # The commanded actions were replaced by nominal *before execution*.
    np.testing.assert_allclose(actions[0, :, :7].numpy(), nominal[:, :7], atol=1e-6)
    np.testing.assert_allclose(actions[0, :, 7].numpy(), nominal[:, 7], atol=1e-6)


def test_safety_violated_chunk_rejected_at_finalize():
    """P2-5: a chunk flagged by the pre-execution barrier never enters replay."""
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(5)
    audit, feedback = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    env = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback, "episode_id": [7]},
    )
    transition = worker._finalize_residual_transition(
        _AuditedRolloutResult(audit, np.zeros((CHUNK_LEN, 8))),
        env,
        0,
        chunk_start_obs=_obs(0.0),
        next_nominal=None,
        safety_violated=True,
    )
    assert transition is None


def test_suppressed_gripper_label_matches_executed_keep():
    """P2-4: when the env worker suppresses the model gripper override, the
    transition's training label must be KEEP (matches the executed command)."""
    worker = _make_worker()
    codec = ResidualCodec()
    rng = np.random.default_rng(4)
    audit, feedback = _chunk_audit_and_feedback(rng, codec, CHUNK_LEN)
    audit["sampled_gripper_mode"] = np.full(CHUNK_LEN, 2, dtype=np.int64)
    feedback["executed_actions"][:, 7] = audit["nominal_actions"][:, 7]
    worker._residual_gripper_suppressed_mask = np.ones(CHUNK_LEN, dtype=bool)
    env = EnvOutput(
        obs=_obs(0.0),
        env_infos={"residual_feedback": feedback, "episode_id": [7]},
    )
    transition = worker._finalize_residual_transition(
        _AuditedRolloutResult(audit, np.zeros((CHUNK_LEN, 8))),
        env,
        0,
        chunk_start_obs=_obs(0.0),
        next_nominal=audit["nominal_actions"],
    )
    assert transition is not None
    assert np.all(np.asarray(transition.actions_gripper)[:, 0] == 1.0)  # KEEP


def test_env_evaluate_step_forwards_gripper_bypass_mask(monkeypatch):
    """Evaluation must forward the model gripper bypass mask to the env, so
    FORCE_CLOSE/FORCE_OPEN execute correctly during autonomous eval."""
    import rlinf.workers.env.env_worker as env_worker_module

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "realworld_dobot",
                    "auto_reset": False,
                    "override_cfg": {"step_frequency": 1.0},
                }
            }
        }
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "residual_dobot_policy",
            "num_action_chunks": CHUNK_LEN,
            "action_dim": 8,
        }
    )
    worker.use_external_reward_model = False
    worker.eval_prev_done = [torch.zeros(1, dtype=torch.bool)]

    captured: dict[str, object] = {}

    class _FakeEvalEnv:
        def chunk_step(self, chunk_actions, gripper_bypass_mask=None):
            captured["mask"] = gripper_bypass_mask
            obs = np.zeros((8, 8, 3), dtype=np.uint8)
            return (
                [{"main_images": obs, "prev_states": np.zeros(8, dtype=np.float32)}],
                torch.zeros(1, CHUNK_LEN, dtype=torch.float32),
                torch.zeros(1, CHUNK_LEN, dtype=torch.bool),
                torch.zeros(1, CHUNK_LEN, dtype=torch.bool),
                [{}],
            )

    worker.eval_env_list = [_FakeEvalEnv()]
    monkeypatch.setattr(
        env_worker_module,
        "prepare_actions",
        lambda **kwargs: torch.zeros(1, CHUNK_LEN, 8),
    )
    mask = np.zeros(CHUNK_LEN, dtype=bool)
    mask[2] = True
    worker.env_evaluate_step(
        torch.zeros(1, CHUNK_LEN, 8),
        0,
        gripper_bypass_mask=mask,
    )
    assert captured["mask"] is mask


def test_pending_residual_transitions_ship_via_channel_put():
    """Real-machine regression: the strict transition envelope is an atomic
    payload; ``send_to``'s batch inference rejects it with
    'Unsupported payload type for batch-size inference: str'. It must be
    shipped with a direct ``channel.put`` like the trajectory path."""
    worker = _make_worker()
    worker.residual_hil_rlpd_mode = True
    worker._run_id = "run-1"
    worker._rank = 0
    worker._pending_residual_transitions = [_dummy_transition()]

    class _RecordingChannel:
        def __init__(self):
            self.items = []

        def put(self, item, **kwargs):
            self.items.append((item, kwargs))

    channel = _RecordingChannel()
    worker._send_pending_residual_transitions(channel)

    assert len(channel.items) == 1
    message, kwargs = channel.items[0]
    assert kwargs.get("async_op") is True
    assert message["type"] == "residual_transition"
    assert int(message["schema_version"]) == 1
    assert message["run_id"] == "run-1"
    assert message["sender_rank"] == 0
    assert message["transition"] is not None
    assert worker._pending_residual_transitions == []

    # Non-residual workers are no-ops.
    worker2 = _make_worker()
    worker2.residual_hil_rlpd_mode = False
    worker2._pending_residual_transitions = [_dummy_transition()]
    worker2._send_pending_residual_transitions(_RecordingChannel())
    assert len(worker2._pending_residual_transitions) == 1
