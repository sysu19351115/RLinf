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

"""Rollout composition and finalizer tests."""

from __future__ import annotations

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.action_codec import (
    FORCE_CLOSE,
    FORCE_OPEN,
    KEEP_NOMINAL,
    ResidualCodec,
)
from rlinf.algorithms.residual_hil_rlpd.finalizer import (
    ChunkEnvFeedback,
    finalize_chunk_transition,
)
from rlinf.algorithms.residual_hil_rlpd.rollout import (
    Pi05ResidualComposer,
    ResidualRolloutPolicy,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    SOURCE_ONLINE,
    SOURCE_ONLINE_INTERVENTION,
)
from rlinf.models.embodiment.residual_dobot_policy import ResidualDobotActor


def _nominal_chunk() -> torch.Tensor:
    nominal = torch.zeros(10, 8)
    for i in range(10):
        nominal[i, 0] = 0.1 + 0.01 * i
        nominal[i, 1] = -0.05
        nominal[i, 2] = 0.30
        nominal[i, 3] = 1.0  # identity quaternion
        nominal[i, 7] = 0.5
    return nominal


class _FakeBasePolicy:
    def __init__(self):
        self.nominal = _nominal_chunk()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.nominal.clone().unsqueeze(0)


def _make_policy():
    return ResidualRolloutPolicy(
        base_policy=_FakeBasePolicy(),
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
    )


def _inputs():
    return (
        torch.zeros(1, 3, 16, 16),
        torch.zeros(1, 8),
    )


def test_zero_residual_scale_equals_base():
    policy = _make_policy()
    images, proprio = _inputs()

    commanded, audit = policy.rollout_chunk(
        images, proprio, residual_scale=0.0, deterministic=True
    )

    np.testing.assert_allclose(commanded, policy.base_policy.nominal.numpy(), atol=1e-6)
    np.testing.assert_allclose(audit.sampled_arm_residual, np.zeros((10, 6)), atol=1e-6)


def test_rollout_audit_fields_and_composition():
    policy = _make_policy()
    images, proprio = _inputs()

    commanded, audit = policy.rollout_chunk(
        images, proprio, residual_scale=1.0, deterministic=False
    )

    assert audit.nominal_actions.shape == (10, 8)
    assert audit.sampled_arm_residual.shape == (10, 6)
    assert audit.sampled_gripper_mode.shape == (10,)
    assert audit.commanded_actions.shape == (10, 8)
    assert audit.gripper_bypass_mask.shape == (10,)
    # commanded pose == compose(nominal, sampled residual)
    codec = policy.codec
    for i in range(10):
        composed = codec.compose(
            torch.as_tensor(audit.nominal_actions[i, :7], dtype=torch.float32),
            torch.as_tensor(audit.sampled_arm_residual[i], dtype=torch.float32),
        ).numpy()
        np.testing.assert_allclose(commanded[i, :7], composed, atol=1e-6)


def test_gripper_modes_produce_bypass():
    policy = _make_policy()
    images, proprio = _inputs()
    commanded, audit = policy.rollout_chunk(
        images, proprio, residual_scale=0.0, deterministic=True
    )
    nominal = audit.nominal_actions

    # KEEP: no bypass, nominal gripper retained.
    assert not audit.gripper_bypass_mask[0]
    np.testing.assert_allclose(commanded[0, 7], nominal[0, 7])

    # Manually verify FORCE semantics through the codec helpers.
    from rlinf.algorithms.residual_hil_rlpd.action_codec import (
        combine_gripper_mode,
    )

    assert combine_gripper_mode(0.5, FORCE_CLOSE) == (0.0, True)
    assert combine_gripper_mode(0.5, FORCE_OPEN) == (1.0, True)
    assert combine_gripper_mode(0.5, KEEP_NOMINAL) == (0.5, False)


def test_finalizer_produces_valid_transition_from_executed_feedback():
    policy = _make_policy()
    images, proprio = _inputs()
    _, audit = policy.rollout_chunk(images, proprio, residual_scale=1.0)

    feedback = ChunkEnvFeedback(
        executed_actions=audit.commanded_actions.copy(),
        executed_action_mask=np.ones(10, dtype=bool),
        gripper_bypass_mask=audit.gripper_bypass_mask,
        human_intervention_mask=np.zeros(10, dtype=bool),
        handoff_hold_mask=np.zeros(10, dtype=bool),
        rewards=np.full(10, 0.05, dtype=np.float32),
        terminations=np.zeros(10, dtype=bool),
        truncations=np.zeros(10, dtype=bool),
        reward_label_valid=True,
    )

    transition, valid, reason = finalize_chunk_transition(
        curr_obs={"main_images": images, "prev_states": proprio},
        next_obs={"main_images": images, "prev_states": proprio},
        audit=audit,
        feedback=feedback,
        codec=policy.codec,
        source=SOURCE_ONLINE,
        base_fingerprint="base-fp",
        episode_id=1,
        chunk_id=0,
        next_nominal_actions=audit.nominal_actions,
    )

    assert valid, reason
    assert transition is not None
    np.testing.assert_allclose(
        transition.actions_arm, audit.sampled_arm_residual, atol=2e-5
    )
    assert transition.source == SOURCE_ONLINE


def test_finalizer_marks_human_intervention_source():
    policy = _make_policy()
    images, proprio = _inputs()
    _, audit = policy.rollout_chunk(images, proprio, residual_scale=1.0)
    feedback = ChunkEnvFeedback(
        executed_actions=audit.commanded_actions.copy(),
        executed_action_mask=np.ones(10, dtype=bool),
        gripper_bypass_mask=audit.gripper_bypass_mask,
        human_intervention_mask=np.array(
            [False] * 4 + [True] + [False] * 5, dtype=bool
        ),
        handoff_hold_mask=np.zeros(10, dtype=bool),
        rewards=np.full(10, 0.05, dtype=np.float32),
        terminations=np.zeros(10, dtype=bool),
        truncations=np.zeros(10, dtype=bool),
        reward_label_valid=True,
    )
    feedback.executed_actions[4, 7] = 0.0

    transition, valid, _ = finalize_chunk_transition(
        curr_obs={"main_images": images, "prev_states": proprio},
        next_obs={"main_images": images, "prev_states": proprio},
        audit=audit,
        feedback=feedback,
        codec=policy.codec,
        source=SOURCE_ONLINE_INTERVENTION,
        base_fingerprint="base-fp",
        episode_id=1,
        chunk_id=0,
        next_nominal_actions=audit.nominal_actions,
    )

    assert valid
    assert transition is not None
    assert transition.human_intervention_mask[4]
    np.testing.assert_allclose(transition.actions_gripper[4], [0, 1, 0])


def test_pi05_composer_uses_openpi_shaped_obs():
    composer = Pi05ResidualComposer(
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
    )
    nominal = torch.stack([_nominal_chunk() for _ in range(2)])  # [2, 10, 8]
    env_obs = {
        "main_images": torch.zeros(2, 3, 16, 16),
        "prev_states": torch.zeros(2, 8),
    }

    commanded, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=0.0,
        deterministic=True,
        policy_version=7,
    )

    assert commanded.shape == (2, 10, 8)
    np.testing.assert_allclose(audit["nominal_actions"], nominal.numpy(), atol=1e-6)
    assert audit["policy_version"] == 7
    np.testing.assert_allclose(
        audit["sampled_arm_residual"], np.zeros((2, 10, 6)), atol=1e-6
    )
    np.testing.assert_allclose(commanded.numpy(), audit["commanded_actions"], atol=1e-6)


def test_pi05_composer_accepts_real_camera_hwc_uint8():
    composer = Pi05ResidualComposer(
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
    )
    nominal = _nominal_chunk().unsqueeze(0)
    env_obs = {
        "main_images": torch.randint(0, 255, (1, 16, 16, 3), dtype=torch.uint8),
        "prev_states": torch.zeros(1, 8),
    }

    commanded, audit = composer.compose_chunk(
        nominal, env_obs, residual_scale=0.0, deterministic=True
    )

    assert commanded.shape == (10, 8)
    np.testing.assert_allclose(
        commanded[:, :7], audit["nominal_actions"][:, :7], atol=1e-6
    )


def test_pi05_composer_enforces_policy_rotation_limit(monkeypatch):
    composer = Pi05ResidualComposer(
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
        policy_translation_scale_m=(0.01, 0.01, 0.01),
        policy_rotation_scale_deg=(5.0, 5.0, 5.0),
    )
    nominal = _nominal_chunk().unsqueeze(0)
    env_obs = {
        "main_images": torch.zeros(1, 3, 16, 16),
        "prev_states": torch.zeros(1, 8),
    }

    def _forced_sample(images, proprio, nominal_t, deterministic=False):
        u = torch.zeros(1, 10, 6)
        u[..., 3] = 1.0  # max normalized rotation on x axis
        return u, torch.zeros(1, 10, dtype=torch.int64), torch.zeros(1, 10), None

    monkeypatch.setattr(composer.residual_actor, "sample", _forced_sample)

    commanded, _ = composer.compose_chunk(
        nominal, env_obs, residual_scale=1.0, deterministic=True
    )

    for i in range(10):
        delta = commanded[i, :3] - nominal[0, i, :3]
        assert np.abs(delta).max() <= 0.0101
        # Rotation angle must stay within the 5 deg policy limit.
        from scipy.spatial.transform import Rotation

        q_nom = nominal[0, i, 3:7]
        q_exec = commanded[i, 3:7]
        if np.dot(q_nom, q_exec) < 0:
            q_exec = -q_exec
        angle = (
            Rotation.from_quat([q_nom[1], q_nom[2], q_nom[3], q_nom[0]]).inv()
            * Rotation.from_quat([q_exec[1], q_exec[2], q_exec[3], q_exec[0]])
        ).magnitude()
        assert np.degrees(angle) <= 5.2


def test_pi05_composer_base_only_records_effective_keep(monkeypatch):
    composer = Pi05ResidualComposer(
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
    )
    nominal = _nominal_chunk().unsqueeze(0)
    env_obs = {
        "main_images": torch.zeros(1, 3, 16, 16),
        "prev_states": torch.zeros(1, 8),
    }

    def _forced_force_open(images, proprio, nominal_t, deterministic=False):
        u = torch.zeros(1, 10, 6)
        return (
            u,
            torch.full((1, 10), 2, dtype=torch.int64),
            torch.zeros(1, 10),
            None,
        )

    monkeypatch.setattr(composer.residual_actor, "sample", _forced_force_open)

    commanded, audit = composer.compose_chunk(
        nominal, env_obs, residual_scale=0.0, deterministic=True
    )

    assert (audit["sampled_gripper_mode"] == 0).all()
    assert (audit["raw_sampled_gripper_mode"] == 2).all()
    np.testing.assert_allclose(commanded[:, 7], np.full(10, float(nominal[0, 0, 7])))


def test_pi05_composer_gripper_enable_and_rate_limits(monkeypatch):
    """P2-4: gripper override is disabled by default and rate-limited when on."""
    composer = Pi05ResidualComposer(
        residual_actor=ResidualDobotActor(hidden=32, image_channels=3),
        codec=ResidualCodec(),
        device="cpu",
        gripper_max_switches_per_chunk=1,
        gripper_min_hold_steps=15,
        gripper_debounce_chunks=2,
    )
    nominal = _nominal_chunk().unsqueeze(0)
    env_obs = {
        "main_images": torch.zeros(1, 3, 16, 16),
        "prev_states": torch.zeros(1, 8),
    }

    def _alternating(images, proprio, nominal_t, deterministic=False):
        modes = torch.tensor(
            [FORCE_OPEN, KEEP_NOMINAL] * 5, dtype=torch.int64
        ).unsqueeze(0)
        return torch.zeros(1, 10, 6), modes, torch.zeros(1, 10), None

    monkeypatch.setattr(composer.residual_actor, "sample", _alternating)

    # Disabled: raw samples recorded, effective mode always KEEP.
    _, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=1.0,
        gripper_enabled=False,
        deterministic=True,
    )
    assert (audit["sampled_gripper_mode"] == 0).all()
    assert int(audit["raw_sampled_gripper_mode"][0]) == FORCE_OPEN

    # Enabled, chunk 1: first switch accepted, then min-hold keeps it.
    _, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=1.0,
        gripper_enabled=True,
        deterministic=True,
    )
    assert int(audit["sampled_gripper_mode"][0]) == FORCE_OPEN
    assert (audit["sampled_gripper_mode"] == FORCE_OPEN).all()

    # Chunk 2: min-hold + debounce keep FORCE_OPEN.
    _, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=1.0,
        gripper_enabled=True,
        deterministic=True,
    )
    assert (audit["sampled_gripper_mode"] == FORCE_OPEN).all()

    # Chunk 3: cooldown consumed; a switch back to KEEP is allowed.
    _, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=1.0,
        gripper_enabled=True,
        deterministic=True,
    )
    assert int(audit["sampled_gripper_mode"][0]) == FORCE_OPEN
    assert int(audit["sampled_gripper_mode"][1]) == KEEP_NOMINAL
    assert (audit["sampled_gripper_mode"][1:] == KEEP_NOMINAL).all()

    # Chunk 4: new cooldown + hold keep KEEP.
    _, audit = composer.compose_chunk(
        nominal,
        env_obs,
        residual_scale=1.0,
        gripper_enabled=True,
        deterministic=True,
    )
    assert (audit["sampled_gripper_mode"] == KEEP_NOMINAL).all()


def test_worker_eval_mode_uses_deterministic_composition(monkeypatch):
    """Eval mode (periodic evaluate or only_eval) must compose deterministically;
    training must keep stochastic sampling."""
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from rlinf.workers.rollout.hf.residual_hil_rollout_worker import (
        ResidualHILRolloutWorker,
    )

    worker = object.__new__(ResidualHILRolloutWorker)
    worker._eval_mode = False
    worker.only_eval = False
    worker._residual_scale = 0.0
    worker._gripper_enabled = False
    worker.version = 1
    worker._last_env_obs = {
        "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "prev_states": torch.zeros(1, 8),
    }
    calls: list[bool] = []

    class _FakeComposer:
        def compose_chunk(
            self,
            actions,
            env_obs,
            *,
            residual_scale,
            gripper_enabled,
            deterministic,
            policy_version,
        ):
            calls.append(deterministic)
            commanded = torch.zeros(10, 8)
            audit = {
                "nominal_actions": np.zeros((10, 8), dtype=np.float32),
                "sampled_arm_residual": np.zeros((10, 6), dtype=np.float32),
                "sampled_gripper_mode": np.zeros(10, dtype=np.int64),
                "commanded_actions": commanded.numpy(),
                "gripper_bypass_mask": np.zeros(10, dtype=bool),
                "policy_version": 1,
            }
            return commanded, audit

    worker.residual_composer = _FakeComposer()

    class _FakeRolloutResult:
        def __init__(self):
            self.actions = None
            self.audit_info = None

    monkeypatch.setattr(
        MultiStepRolloutWorker,
        "_build_rollout_result",
        lambda self, actions, result, **kwargs: _FakeRolloutResult(),
    )

    worker._build_rollout_result(torch.zeros(10, 8), {})
    assert calls[-1] is False

    worker._eval_mode = True
    worker._build_rollout_result(torch.zeros(10, 8), {})
    assert calls[-1] is True

    worker._eval_mode = False
    worker.only_eval = True
    worker._build_rollout_result(torch.zeros(10, 8), {})
    assert calls[-1] is True
