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

"""Offline HIL -> residual replay conversion tests."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from toolkits.embodiment.convert_dobot_hil_to_residual_replay import (
    convert_episodes_to_residual_replay,
)


def _episode(episode_id: int, time_len: int = 23, human_at: int | None = None):
    actions = np.zeros((time_len, 8), dtype=np.float32)
    human = np.zeros(time_len, dtype=bool)
    for i in range(time_len):
        actions[i, 0] = 0.1 + 0.001 * i
        actions[i, 2] = 0.3
        actions[i, 3] = 1.0
        actions[i, 7] = 0.5
    if human_at is not None:
        actions[human_at, 7] = 0.0
        human[human_at] = True
    return {
        "episode_id": episode_id,
        "observations": {
            "main_images": np.zeros((time_len, 3, 16, 16), dtype=np.uint8),
            "prev_states": np.zeros((time_len, 8), dtype=np.float32),
        },
        "actions": actions,
        "human_intervention_mask": human,
        "handoff_hold_mask": np.zeros(time_len, dtype=bool),
        "rewards": np.full(time_len, 0.05, dtype=np.float32),
        "terminated": True,
    }


def _nominal_provider(episode_id: int, chunk_id: int) -> np.ndarray:
    nominal = np.zeros((10, 8), dtype=np.float32)
    nominal[:, 0] = 0.1
    nominal[:, 2] = 0.3
    nominal[:, 3] = 1.0
    nominal[:, 7] = 0.5
    return nominal


def _next_nominal_provider(episode_id: int, chunk_id: int) -> np.ndarray:
    return _nominal_provider(episode_id, chunk_id + 1)


def test_conversion_chunks_episodes_and_tails():
    episodes = [_episode(0, time_len=23, human_at=11)]

    transitions, report = convert_episodes_to_residual_replay(
        episodes,
        nominal_provider=_nominal_provider,
        next_nominal_provider=_next_nominal_provider,
        codec=ResidualCodec(),
        base_fingerprint="base-fp",
        episode_id_offset=0,
    )

    # 23 steps -> 3 chunks: 10 + 10 + 3.
    assert report["transitions"] == 3
    assert report["rejected"] == 0
    assert transitions[0].discount_steps[0] == 10
    assert transitions[1].discount_steps[0] == 10
    assert transitions[2].discount_steps[0] == 3
    assert transitions[2].executed_action_mask[3:].sum() == 0
    assert transitions[1].human_intervention_mask[1]
    assert transitions[1].source == 2  # SOURCE_OFFLINE_DEMO
    # Middle chunks must keep bootstrap (no artificial truncation).
    assert transitions[0].bootstrap_mask[0]
    assert transitions[1].bootstrap_mask[0]
    assert not transitions[0].truncations.any()
    assert not transitions[1].truncations.any()
    assert not transitions[2].bootstrap_mask[0]


def test_conversion_rejects_missing_nominal():
    episodes = [_episode(0, time_len=10)]

    def missing_provider(episode_id, chunk_id):
        raise KeyError(episode_id, chunk_id)

    _, report = convert_episodes_to_residual_replay(
        episodes,
        nominal_provider=missing_provider,
        next_nominal_provider=_next_nominal_provider,
        codec=ResidualCodec(),
        base_fingerprint="base-fp",
        episode_id_offset=0,
    )

    assert report["rejected"] == 1
    assert report["rejected_reasons"]["missing_nominal"] == 1


def test_conversion_recovers_residual_within_data_limit():
    episodes = [_episode(0, time_len=10)]
    # Move one executed action by a small offset so the residual is non-zero.
    episodes[0]["actions"][4, :3] += np.array([0.01, 0.0, 0.0])

    transitions, _ = convert_episodes_to_residual_replay(
        episodes,
        nominal_provider=_nominal_provider,
        next_nominal_provider=_next_nominal_provider,
        codec=ResidualCodec(),
        base_fingerprint="base-fp",
        episode_id_offset=0,
    )

    assert np.count_nonzero(transitions[0].actions_arm[4]) > 0
