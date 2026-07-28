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

"""CPU-only HG-DAgger hybrid-window smoke test.

Feeds a 5-chunk trajectory (1 human + 4 model) through the assembler,
verifies the emitted 50-step window enters the replay buffer, and checks
a forward/backward/weight-sync cycle.
"""

import torch
from omegaconf import OmegaConf

import rlinf.workers.actor.fsdp_dagger_policy_worker as dagger_worker
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_dagger_policy_worker import (
    EmbodiedDAGGERFSDPPolicy,
)

_ACTION_DIM = 8
_CHUNK = 10
_WINDOW = 50
_T = 5  # 5 chunks = 50 steps


def _make_trajectory() -> Trajectory:
    actions = torch.zeros((_T, 1, _CHUNK * _ACTION_DIM))
    step_intervene = torch.zeros((_T, 1, _CHUNK), dtype=torch.bool)
    # First chunk is fully human
    step_intervene[0] = True
    intervene = (
        step_intervene.unsqueeze(-1)
        .expand(-1, -1, -1, _ACTION_DIM)
        .reshape(_T, 1, _CHUNK * _ACTION_DIM)
    )
    actions[0] = 0.5
    # Remaining chunks are model actions
    for t in range(1, _T):
        actions[t] = 0.3

    step_ids = torch.stack(
        [torch.arange(t * _CHUNK, (t + 1) * _CHUNK) for t in range(_T)]
    ).unsqueeze(1)  # [T, 1, chunk]

    return Trajectory(
        max_episode_length=100,
        model_weights_id="rollout-v7",
        actions=actions,
        intervene_flags=intervene,
        rewards=torch.zeros((_T, 1, _CHUNK)),
        versions=torch.tensor([[[7]]] * _T),
        forward_inputs={
            "action": actions.clone(),
            "model_action": torch.randn((_T, 1, _WINDOW * 32)),
            "chains": torch.randn((_T, 1, 4, _WINDOW, 32)),
            "denoise_inds": torch.arange(4).reshape(1, 1, 4).expand(_T, 1, 4),
            "observation/prev_state": torch.full((_T, 1, _ACTION_DIM), 0.25),
            "observation/image": torch.randn((_T, 1, 3, 8, 8)),
        },
        audit_info={
            "executed_action_mask": torch.ones((_T, 1, _CHUNK), dtype=torch.bool),
            "termination_reason_code": torch.zeros((_T, 1), dtype=torch.int64),
            "episode_id": torch.tensor([[3]] * _T, dtype=torch.int64),
            "episode_step_ids": step_ids,
        },
    )


def test_process_train_metrics_exposes_window_assembly_state(monkeypatch):
    actor = EmbodiedDAGGERFSDPPolicy.__new__(EmbodiedDAGGERFSDPPolicy)
    actor.enable_online_lerobot = False
    actor.replay_buffer = type(
        "ReplayStats",
        (),
        {"get_stats": lambda self: {"size": 0}},
    )()
    actor._window_assembler = type(
        "WindowStats",
        (),
        {
            "get_metrics": lambda self: {
                "window_emitted": 2,
                "window_dropped": 3,
                "pending_anchor": 1,
                "window_drop_reason/unexecuted_action": 3,
            }
        },
    )()
    monkeypatch.setattr(
        dagger_worker,
        "all_reduce_dict",
        lambda metrics, op: metrics,
    )

    metrics = actor.process_train_metrics({})

    assert metrics["dagger/window_emitted"] == 2
    assert metrics["dagger/window_dropped"] == 3
    assert metrics["dagger/pending_anchor"] == 1
    assert metrics["dagger/window_drop_reason/unexecuted_action"] == 3


def test_hybrid_window_replay_update_and_weight_sync_smoke():
    actor = EmbodiedDAGGERFSDPPolicy.__new__(EmbodiedDAGGERFSDPPolicy)
    actor.cfg = OmegaConf.create(
        {
            "actor": {"model": {"action_dim": _ACTION_DIM}},
            "algorithm": {
                "dagger": {
                    "required_forward_input_keys": [
                        "observation/prev_state",
                        "observation/image",
                    ],
                    "min_human_steps_per_window": 10,
                    "execution_chunk_steps": _CHUNK,
                    "training_window_steps": _WINDOW,
                    "window_stride_steps": _CHUNK,
                },
            },
        }
    )
    actor.replay_buffer = TrajectoryReplayBuffer(
        seed=1,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
    )
    from rlinf.data.embodied_io_struct import ExpertAnchoredWindowAssembler

    actor._window_assembler = ExpertAnchoredWindowAssembler(
        execution_chunk_steps=_CHUNK,
        training_window_steps=_WINDOW,
        window_stride_steps=_CHUNK,
        min_human_steps_per_window=10,
        action_dim=_ACTION_DIM,
    )

    actor.recv_buffer_rollout_trajectories([_make_trajectory()])
    batch = actor.replay_buffer.sample(1)

    # The window has 50 steps, 10 human + 40 model
    assert batch["actions"].shape[-1] == _WINDOW * _ACTION_DIM
    human_mask = batch["audit_info"]["human_action_mask"]
    assert human_mask.shape[-1] == _WINDOW
    assert int(human_mask.sum()) == _CHUNK
    assert batch["audit_info"]["executed_action_mask"].all()
    assert batch["versions"].shape == (1,)
    assert batch["versions"].item() == 7
    assert batch["intervene_flags"].shape[-1] == _WINDOW * _ACTION_DIM
    assert batch["forward_inputs"]["observation/image"].shape == (1, 3, 8, 8)
    assert batch["forward_inputs"]["human_action_mask"].shape == (1, _WINDOW)
    assert "model_action" not in batch["forward_inputs"]
    assert "chains" not in batch["forward_inputs"]
    assert "denoise_inds" not in batch["forward_inputs"]

    # Forward / backward / weight-sync
    training_model = torch.nn.Linear(_ACTION_DIM, _WINDOW * _ACTION_DIM)
    rollout_model = torch.nn.Linear(_ACTION_DIM, _WINDOW * _ACTION_DIM)
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
