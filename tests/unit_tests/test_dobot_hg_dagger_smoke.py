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

"""CPU-only HG-DAgger rollout-to-weight-sync smoke test."""

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_dagger_policy_worker import (
    EmbodiedDAGGERFSDPPolicy,
)


def _expert_rollout() -> Trajectory:
    return Trajectory(
        max_episode_length=100,
        model_weights_id="rollout-v7",
        actions=torch.full((1, 1, 32), 0.5),
        intervene_flags=torch.ones((1, 1, 32), dtype=torch.bool),
        rewards=torch.zeros((1, 1, 4)),
        versions=torch.tensor([[[7]]]),
        forward_inputs={
            "action": torch.full((1, 1, 32), 0.5),
            "observation/prev_state": torch.full((1, 1, 8), 0.25),
        },
        audit_info={
            "executed_action_mask": torch.ones((1, 1, 4), dtype=torch.bool),
            "termination_reason_code": torch.zeros((1, 1), dtype=torch.int64),
            "episode_id": torch.tensor([[3]], dtype=torch.int64),
            "episode_step_ids": torch.arange(4).reshape(1, 1, 4),
        },
    )


def test_rollout_replay_update_and_weight_sync_smoke():
    actor = EmbodiedDAGGERFSDPPolicy.__new__(EmbodiedDAGGERFSDPPolicy)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "dagger": {"required_forward_input_keys": ["observation/prev_state"]}
            }
        }
    )
    actor.replay_buffer = TrajectoryReplayBuffer(
        seed=1,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
    )

    actor.recv_buffer_rollout_trajectories([_expert_rollout()])
    batch = actor.replay_buffer.sample(1)
    assert batch["audit_info"]["executed_action_mask"].all()

    training_model = torch.nn.Linear(8, 32)
    rollout_model = torch.nn.Linear(8, 32)
    optimizer = torch.optim.SGD(training_model.parameters(), lr=0.1)
    before = training_model.weight.detach().clone()
    prediction = training_model(batch["forward_inputs"]["observation/prev_state"])
    loss = torch.nn.functional.mse_loss(prediction, batch["actions"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(training_model.weight, before)

    rollout_model.load_state_dict(training_model.state_dict())
    for actor_param, rollout_param in zip(
        training_model.parameters(), rollout_model.parameters()
    ):
        torch.testing.assert_close(actor_param, rollout_param)
