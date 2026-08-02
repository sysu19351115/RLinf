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

"""Chunk-SMDP transition builder tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from rlinf.algorithms.residual_hil_rlpd.action_codec import (
    ResidualCodec,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    CHUNK_LEN,
    SOURCE_ONLINE,
    SOURCE_ONLINE_INTERVENTION,
    STATUS_K_ZERO,
    STATUS_MISSING_NEXT_NOMINAL,
    STATUS_NON_CONTIGUOUS_EXECUTED_MASK,
    STATUS_OUT_OF_SUPPORT,
    STATUS_REWARD_LABEL_INVALID,
    build_residual_chunk_transition,
)


def _random_pose(rng: np.random.Generator, gripper: float = 0.5) -> np.ndarray:
    position = rng.uniform(-0.2, 0.2, size=3)
    quat = Rotation.random(random_state=rng).as_quat()
    return np.concatenate([position, [quat[3], quat[0], quat[1], quat[2]], [gripper]])


def _chunk_inputs(
    rng: np.random.Generator,
    codec: ResidualCodec,
    k: int,
    human_steps: tuple[int, ...] = (),
    terminations: np.ndarray | None = None,
    truncations: np.ndarray | None = None,
    out_of_support_step: int | None = None,
):
    h = CHUNK_LEN
    nominal = np.stack([_random_pose(rng) for _ in range(h)]).astype(np.float32)
    u = rng.uniform(-0.4, 0.4, size=(h, 6)).astype(np.float32)
    gripper_modes = rng.integers(0, 3, size=h)
    commanded = np.zeros((h, 8), dtype=np.float32)
    for i in range(h):
        composed_pose = codec.compose(
            torch.as_tensor(nominal[i, :7], dtype=torch.float32),
            torch.as_tensor(u[i], dtype=torch.float32),
        ).numpy()
        commanded[i] = np.concatenate([composed_pose, [nominal[i, 7]]])
    executed = commanded.copy()
    for i in range(h):
        if i in human_steps:
            executed[i, 7] = 0.0  # human forced-close command
        else:
            executed[i, 7] = nominal[i, 7]
    if out_of_support_step is not None:
        executed[out_of_support_step, :3] += np.array([0.5, 0.0, 0.0])
    executed_mask = np.zeros(h, dtype=bool)
    executed_mask[:k] = True
    human_mask = np.zeros(h, dtype=bool)
    human_mask[list(human_steps)] = True
    handoff_mask = np.zeros(h, dtype=bool)
    rewards = rng.uniform(0.0, 0.1, size=h).astype(np.float32)
    return {
        "curr_obs": {"main_images": np.zeros((1, 8, 8), dtype=np.uint8)},
        "next_obs": {"main_images": np.zeros((1, 8, 8), dtype=np.uint8)},
        "nominal_actions": nominal,
        "next_nominal_actions": nominal,
        "sampled_arm_residual": u,
        "sampled_gripper_mode": gripper_modes,
        "commanded_actions": commanded,
        "executed_actions": executed,
        "gripper_bypass_mask": np.zeros(h, dtype=bool),
        "rewards": rewards,
        "executed_action_mask": executed_mask,
        "human_intervention_mask": human_mask,
        "handoff_hold_mask": handoff_mask,
        "terminations": (
            np.zeros(h, dtype=bool) if terminations is None else terminations
        ),
        "truncations": (
            np.zeros(h, dtype=bool) if truncations is None else truncations
        ),
        "reward_label_valid": True,
        "source": SOURCE_ONLINE,
        "policy_version": 3,
        "base_fingerprint": "base123",
    }


@pytest.fixture()
def codec():
    return ResidualCodec()


def test_full_chunk_is_valid_and_recovers_residuals(codec):
    rng = np.random.default_rng(0)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN)

    transition, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, episode_id=1, chunk_id=2, **inputs
    )

    assert valid, reason
    assert transition.discount_steps[0] == CHUNK_LEN
    assert transition.bootstrap_mask[0]
    expected_return = float(np.sum(inputs["rewards"] * (0.99 ** np.arange(CHUNK_LEN))))
    np.testing.assert_allclose(transition.discounted_return, expected_return, atol=1e-6)
    np.testing.assert_allclose(
        transition.actions_arm, inputs["sampled_arm_residual"], atol=2e-5
    )
    transition.validate(codec)


def test_early_enter_keeps_executed_prefix_and_disables_bootstrap(codec):
    rng = np.random.default_rng(1)
    truncations = np.zeros(CHUNK_LEN, dtype=bool)
    truncations[2] = True
    inputs = _chunk_inputs(rng, codec, k=3, truncations=truncations)

    transition, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert valid, reason
    assert transition.discount_steps[0] == 3
    assert not transition.bootstrap_mask[0]
    assert np.count_nonzero(transition.actions_arm[3:]) == 0
    assert np.count_nonzero(transition.actions_gripper[3:]) == 0
    expected_return = float(np.sum(inputs["rewards"][:3] * (0.99 ** np.arange(3))))
    np.testing.assert_allclose(transition.discounted_return, expected_return, atol=1e-6)


def test_backspace_termination_disables_bootstrap(codec):
    rng = np.random.default_rng(2)
    terminations = np.zeros(CHUNK_LEN, dtype=bool)
    terminations[1] = True
    inputs = _chunk_inputs(rng, codec, k=2, terminations=terminations)

    transition, valid, _ = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert valid
    assert not transition.bootstrap_mask[0]


def test_human_intervention_inverts_gripper_from_executed(codec):
    rng = np.random.default_rng(3)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN, human_steps=(4, 5))
    inputs["source"] = SOURCE_ONLINE_INTERVENTION

    transition, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert valid, reason
    # Human steps executed gripper 0.0 -> FORCE_CLOSE one-hot.
    np.testing.assert_allclose(transition.actions_gripper[4], [0, 1, 0])
    np.testing.assert_allclose(transition.actions_gripper[5], [0, 1, 0])
    assert np.count_nonzero(transition.human_intervention_mask) == 2
    assert transition.source == SOURCE_ONLINE_INTERVENTION


def test_invalid_reward_label_rejects(codec):
    rng = np.random.default_rng(4)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN)
    inputs["reward_label_valid"] = False

    _, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert not valid
    assert reason == STATUS_REWARD_LABEL_INVALID


def test_non_contiguous_executed_mask_rejects(codec):
    rng = np.random.default_rng(5)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN)
    inputs["executed_action_mask"][5] = False
    inputs["executed_action_mask"][7] = True

    _, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert not valid
    assert reason == STATUS_NON_CONTIGUOUS_EXECUTED_MASK


def test_k_zero_rejects(codec):
    rng = np.random.default_rng(6)
    inputs = _chunk_inputs(rng, codec, k=0)

    _, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert not valid
    assert reason == STATUS_K_ZERO


def test_out_of_support_rejects(codec):
    rng = np.random.default_rng(7)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN, out_of_support_step=3)

    _, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert not valid
    assert reason == STATUS_OUT_OF_SUPPORT


def test_handoff_hold_is_not_human_intervention(codec):
    rng = np.random.default_rng(8)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN, human_steps=(2,))
    inputs["handoff_hold_mask"][3] = True

    transition, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert valid, reason
    assert not transition.human_intervention_mask[3]


def test_missing_next_nominal_rejects_bootstrap(codec):
    rng = np.random.default_rng(10)
    inputs = _chunk_inputs(rng, codec, k=CHUNK_LEN)
    inputs["next_nominal_actions"] = np.zeros((CHUNK_LEN, 8), dtype=np.float32)

    _, valid, reason = build_residual_chunk_transition(
        codec=codec, gamma=0.99, **inputs
    )

    assert not valid
    assert reason == STATUS_MISSING_NEXT_NOMINAL
