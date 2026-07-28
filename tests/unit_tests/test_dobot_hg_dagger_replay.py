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

"""Replay persistence tests for the Dobot HG-DAgger audit contract."""

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _expert_trajectory() -> Trajectory:
    return Trajectory(
        max_episode_length=100,
        model_weights_id="weights-7",
        actions=torch.ones((1, 1, 32)),
        intervene_flags=torch.ones((1, 1, 32), dtype=torch.bool),
        rewards=torch.zeros((1, 1, 4)),
        versions=torch.tensor([[[7]]]),
        forward_inputs={
            "action": torch.ones((1, 1, 32)),
            "observation/prev_state": torch.zeros((1, 1, 8)),
        },
        audit_info={
            "executed_action_mask": torch.ones((1, 1, 4), dtype=torch.bool),
            "termination_reason_code": torch.zeros((1, 1), dtype=torch.int64),
            "episode_id": torch.tensor([[12]], dtype=torch.int64),
            "episode_step_ids": torch.arange(4).reshape(1, 1, 4),
        },
    )


def test_replay_sample_preserves_audit_info(tmp_path):
    replay = TrajectoryReplayBuffer(
        seed=1,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    replay.add_trajectories([_expert_trajectory()])

    sampled = replay.sample(1)
    assert sampled["audit_info"]["executed_action_mask"].all()
    torch.testing.assert_close(sampled["audit_info"]["episode_id"], torch.tensor([12]))

    replay.close(wait=True)
    loaded = replay._load_trajectory(0, "weights-7")
    torch.testing.assert_close(
        loaded.audit_info["episode_step_ids"], torch.arange(4).reshape(1, 1, 4)
    )


def test_schema_mismatch_fails_closed(tmp_path):
    """A replay directory written with one schema must reject a different one."""
    import json
    import os

    # Write a metadata.json with an old schema version
    meta_path = os.path.join(str(tmp_path), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"schema_version": "dobot_hg_dagger_10step_v1"}, f)

    import pytest

    with pytest.raises(ValueError, match="schema mismatch"):
        TrajectoryReplayBuffer(
            seed=1,
            auto_save=True,
            auto_save_path=str(tmp_path),
            schema_version="dobot_hg_dagger_hybrid_h50_v1",
        )


def test_corrupt_metadata_fails_closed(tmp_path):
    """A replay directory with unreadable metadata must never be accepted."""
    import os

    meta_path = os.path.join(str(tmp_path), "metadata.json")
    with open(meta_path, "w") as f:
        f.write("{not-valid-json")

    import pytest

    with pytest.raises(ValueError, match="Cannot read replay metadata"):
        TrajectoryReplayBuffer(
            seed=1,
            auto_save=True,
            auto_save_path=str(tmp_path),
            schema_version="dobot_hg_dagger_hybrid_h50_v1",
        )


def test_schema_match_succeeds(tmp_path):
    """Same schema version should not raise."""
    import json
    import os

    meta_path = os.path.join(str(tmp_path), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"schema_version": "dobot_hg_dagger_hybrid_h50_v1"}, f)

    TrajectoryReplayBuffer(
        seed=1,
        auto_save=True,
        auto_save_path=str(tmp_path),
        schema_version="dobot_hg_dagger_hybrid_h50_v1",
    )


def test_unversioned_directory_does_not_block(tmp_path):
    """A new empty directory without metadata.json should allow creation."""
    TrajectoryReplayBuffer(
        seed=1,
        auto_save=True,
        auto_save_path=str(tmp_path),
        schema_version="dobot_hg_dagger_hybrid_h50_v1",
    )


def test_nonempty_directory_without_metadata_fails_closed(tmp_path):
    """Existing replay payload without metadata must never be guessed."""
    import pytest

    (tmp_path / "trajectory_0_weights-7.pt").write_bytes(b"legacy replay")

    with pytest.raises(ValueError, match="metadata.json is missing"):
        TrajectoryReplayBuffer(
            seed=1,
            auto_save=True,
            auto_save_path=str(tmp_path),
            schema_version="dobot_hg_dagger_hybrid_h50_v2",
        )


def test_checkpoint_schema_mismatch_fails_closed(tmp_path):
    """load_checkpoint must enforce the same schema check as construction."""
    import json

    import pytest

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    with (checkpoint_path / "metadata.json").open("w") as f:
        json.dump({"schema_version": "dobot_hg_dagger_10step_v1"}, f)

    replay = TrajectoryReplayBuffer(
        seed=1,
        auto_save=False,
        schema_version="dobot_hg_dagger_hybrid_h50_v2",
    )
    try:
        with pytest.raises(ValueError, match="schema mismatch"):
            replay.load_checkpoint(str(checkpoint_path))
    finally:
        replay.close(wait=True)
