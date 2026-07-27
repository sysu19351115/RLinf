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
