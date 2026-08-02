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

"""Dual-buffer 50/50 sampling tests."""

from __future__ import annotations

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.transition import (
    CHUNK_LEN,
    SOURCE_OFFLINE_DEMO,
    SOURCE_ONLINE,
    SOURCE_ONLINE_INTERVENTION,
    build_residual_chunk_transition,
)
from rlinf.data.residual_hil_replay import (
    ResidualHilReplayBuffer,
    transition_resident_bytes,
    transitions_to_torch_batch,
)


def _make_transition(
    episode_id: int,
    chunk_id: int,
    source: int = SOURCE_ONLINE,
    human_steps: tuple[int, ...] = (),
) -> object:
    codec = ResidualCodec()
    rng = np.random.default_rng(episode_id * 100 + chunk_id)
    h = CHUNK_LEN
    nominal = np.stack(
        [
            np.concatenate(
                [rng.uniform(-0.2, 0.2, size=3), [1.0, 0.0, 0.0, 0.0], [0.5]]
            )
            for _ in range(h)
        ]
    ).astype(np.float32)
    u = rng.uniform(-0.3, 0.3, size=(h, 6)).astype(np.float32)
    commanded = np.zeros((h, 8), dtype=np.float32)
    for i in range(h):
        pose = codec.compose(
            torch.as_tensor(nominal[i, :7], dtype=torch.float32),
            torch.as_tensor(u[i], dtype=torch.float32),
        ).numpy()
        commanded[i] = np.concatenate([pose, [nominal[i, 7]]])
    executed = commanded.copy()
    human_mask = np.zeros(h, dtype=bool)
    for i in human_steps:
        executed[i, 7] = 0.0
        human_mask[i] = True
    executed_mask = np.ones(h, dtype=bool)
    transition, valid, reason = build_residual_chunk_transition(
        curr_obs={
            "main_images": torch.zeros(1, 3, 16, 16, dtype=torch.float32),
            "prev_states": torch.zeros(1, 8, dtype=torch.float32),
        },
        next_obs={
            "main_images": torch.zeros(1, 3, 16, 16, dtype=torch.float32),
            "prev_states": torch.zeros(1, 8, dtype=torch.float32),
        },
        nominal_actions=nominal,
        next_nominal_actions=nominal,
        sampled_arm_residual=u,
        sampled_gripper_mode=np.zeros(h, dtype=np.int64),
        commanded_actions=commanded,
        executed_actions=executed,
        gripper_bypass_mask=np.zeros(h, dtype=bool),
        rewards=rng.uniform(0, 0.1, size=h).astype(np.float32),
        executed_action_mask=executed_mask,
        human_intervention_mask=human_mask,
        handoff_hold_mask=np.zeros(h, dtype=bool),
        terminations=np.zeros(h, dtype=bool),
        truncations=np.zeros(h, dtype=bool),
        reward_label_valid=True,
        codec=codec,
        source=source,
        policy_version=1,
        base_fingerprint="base",
        episode_id=episode_id,
        chunk_id=chunk_id,
    )
    assert valid, reason
    return transition


def test_waiting_for_demo_until_intervention():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    buffer.add(_make_transition(0, 0))
    assert buffer.waiting_for_demo()

    buffer.add(_make_transition(0, 1, human_steps=(2,)))
    assert not buffer.waiting_for_demo()


def test_intervention_duplicated_to_demo():
    buffer = ResidualHilReplayBuffer()
    buffer.add(_make_transition(0, 0))
    buffer.add(_make_transition(0, 1, human_steps=(3,)))
    buffer.add(
        _make_transition(0, 2, source=SOURCE_ONLINE_INTERVENTION, human_steps=(1,))
    )

    sizes = buffer.sizes()
    assert sizes["online"] == 3
    assert sizes["demo"] == 2


def test_duplicate_transition_id_ignored():
    buffer = ResidualHilReplayBuffer()
    transition = _make_transition(0, 0, human_steps=(2,))
    assert buffer.add(transition)
    assert not buffer.add(transition)
    assert buffer.sizes() == {"online": 1, "demo": 1}


def test_source_namespace_keeps_offline_and_online_ids_distinct():
    """Offline demos and online data may reuse (episode_id, chunk_id); the
    source dimension must keep both instead of deduplicating one away."""
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    online = _make_transition(0, 0, human_steps=(1,), source=SOURCE_ONLINE)
    offline = _make_transition(
        0, 0, human_steps=(2,), source=SOURCE_OFFLINE_DEMO
    )
    assert buffer.add(online)
    assert buffer.add(offline)
    assert buffer.sizes() == {"online": 2, "demo": 2}
    assert buffer.add(online) is False
    assert buffer.add(offline) is False
    assert buffer.sizes() == {"online": 2, "demo": 2}
    counts = buffer.source_counts()
    assert counts[str(SOURCE_ONLINE)] == 1
    assert counts[str(SOURCE_OFFLINE_DEMO)] == 1


def test_byte_budget_evicts_oldest():
    buffer = ResidualHilReplayBuffer(
        max_online_transitions=100,
        max_demo_transitions=100,
        max_online_bytes=transition_resident_bytes(_make_transition(0, 0)) * 2,
        max_demo_bytes=1e12,
    )
    assert buffer.add(_make_transition(0, 0, human_steps=(1,)))
    assert buffer.add(_make_transition(1, 0))
    assert buffer.add(_make_transition(2, 0, human_steps=(2,)))
    assert buffer.sizes()["online"] == 2
    assert buffer.evicted_count == 1
    assert buffer.source_counts()[str(SOURCE_ONLINE)] == 2


def test_transition_count_cap_evicts_oldest():
    buffer = ResidualHilReplayBuffer(
        max_online_transitions=2,
        max_demo_transitions=2,
        max_online_bytes=1e12,
        max_demo_bytes=1e12,
    )
    for episode in range(4):
        buffer.add(_make_transition(episode, 0))
    assert buffer.sizes()["online"] == 2
    assert buffer.evicted_count == 2
    remaining = buffer.source_counts()
    assert remaining[str(SOURCE_ONLINE)] == 2
    assert buffer.duplicate_count == 0


def test_memory_watermark_blocks_training():
    one = transition_resident_bytes(_make_transition(0, 0))
    buffer = ResidualHilReplayBuffer(
        min_demo_size=1,
        max_online_transitions=100,
        max_demo_transitions=100,
        max_online_bytes=one * 4,
        max_demo_bytes=1e12,
        memory_high_watermark=0.5,
    )
    buffer.add(_make_transition(0, 0, human_steps=(1,)))
    assert buffer.can_train()
    buffer.add(_make_transition(1, 0))  # 2 * one == 0.5 * (4 * one)
    assert buffer.over_memory_watermark()
    assert not buffer.can_train()


def test_demo_ratio_sampling():
    buffer = ResidualHilReplayBuffer(
        min_demo_size=1,
        demo_ratio=0.25,
        max_online_transitions=100,
        max_demo_transitions=100,
        max_online_bytes=1e12,
        max_demo_bytes=1e12,
    )
    for episode in range(8):
        buffer.add(_make_transition(episode, 0, human_steps=(episode,)))
    transitions, source_mask = buffer.sample(16)
    assert int(source_mask.sum()) == 4  # 25% demo
    assert abs(float(np.mean(source_mask)) - 0.25) < 1e-6
    assert len(transitions) == 16


def test_stats_report_counters():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    buffer.add(_make_transition(0, 0, human_steps=(1,)))
    buffer.add(_make_transition(0, 0, human_steps=(1,)))  # duplicate
    stats = buffer.stats()
    assert stats["accepted_count"] == 1
    assert stats["duplicate_count"] == 1
    assert stats["rejected_count"] == 0
    assert "online_bytes" in stats
    assert "demo_bytes" in stats


def test_state_dict_round_trip_keeps_bytes_and_stats():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    buffer.add(_make_transition(0, 0, human_steps=(1,)))
    buffer.add(_make_transition(1, 0))
    restored = ResidualHilReplayBuffer(min_demo_size=1)
    restored.load_state_dict(buffer.state_dict())
    assert restored.sizes() == buffer.sizes()
    assert abs(
        restored.memory_usage()["online_bytes"]
        - buffer.memory_usage()["online_bytes"]
    ) < 1e-6
    assert restored.accepted_count == buffer.accepted_count


def test_strict_50_50_sampling():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    for episode in range(10):
        buffer.add(_make_transition(episode, 0))
        buffer.add(_make_transition(episode, 1, human_steps=(1,)))

    transitions, source_mask = buffer.sample(64)

    assert len(transitions) == 64
    assert float(source_mask.mean()) == 0.5
    assert int(source_mask[:32].sum()) == 0
    assert int(source_mask[32:].sum()) == 32
    assert buffer.audit_ratio(transitions, source_mask) == 0.5


def test_save_load_round_trip_reproduces_sampling():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    for episode in range(5):
        buffer.add(_make_transition(episode, 0))
        buffer.add(_make_transition(episode, 1, human_steps=(2,)))

    saved_state = buffer.state_dict()
    first_ids = [(t.episode_id, t.chunk_id) for t, _ in zip(*buffer.sample(16))]

    restored = ResidualHilReplayBuffer()
    restored.load_state_dict(saved_state)
    second_ids = [(t.episode_id, t.chunk_id) for t, _ in zip(*restored.sample(16))]

    assert first_ids == second_ids


def test_torch_batch_conversion_shapes():
    buffer = ResidualHilReplayBuffer(min_demo_size=1)
    buffer.add(_make_transition(0, 0))
    buffer.add(_make_transition(0, 1, human_steps=(1,)))
    transitions, _ = buffer.sample(2)

    batch = transitions_to_torch_batch(transitions)

    assert batch["actions_arm"].shape == (2, CHUNK_LEN, 6)
    assert batch["actions_gripper"].shape == (2, CHUNK_LEN, 3)
    assert batch["bootstrap_mask"].shape == (2,)
    assert batch["discount_steps"].shape == (2,)
    assert batch["discounted_return"].shape == (2,)


def test_torch_batch_preprocesses_real_bhwc_uint8():
    codec = ResidualCodec()
    rng = np.random.default_rng(9)
    h = CHUNK_LEN
    nominal = np.zeros((h, 8), dtype=np.float32)
    nominal[:, 3] = 1.0
    u = np.zeros((h, 6), dtype=np.float32)
    executed = nominal.copy()
    obs = {
        "main_images": torch.randint(0, 255, (1, 16, 16, 3), dtype=torch.uint8),
        "prev_states": torch.zeros(1, 8, dtype=torch.float32),
    }
    transition, valid, reason = build_residual_chunk_transition(
        curr_obs=obs,
        next_obs=obs,
        nominal_actions=nominal,
        next_nominal_actions=nominal,
        sampled_arm_residual=u,
        sampled_gripper_mode=np.zeros(h, dtype=np.int64),
        commanded_actions=executed,
        executed_actions=executed,
        gripper_bypass_mask=np.zeros(h, dtype=bool),
        rewards=rng.uniform(0, 0.1, size=h).astype(np.float32),
        executed_action_mask=np.ones(h, dtype=bool),
        human_intervention_mask=np.zeros(h, dtype=bool),
        handoff_hold_mask=np.zeros(h, dtype=bool),
        terminations=np.zeros(h, dtype=bool),
        truncations=np.zeros(h, dtype=bool),
        reward_label_valid=True,
        codec=codec,
        source=SOURCE_ONLINE,
        policy_version=1,
        base_fingerprint="base",
        episode_id=0,
        chunk_id=0,
    )
    assert valid, reason

    batch = transitions_to_torch_batch([transition])

    assert batch["curr_images"].shape == (1, 3, 16, 16)
    assert batch["curr_images"].max() <= 1.0
