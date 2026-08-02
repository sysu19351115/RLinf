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

"""In-process dummy end-to-end run of the residual HIL-RLPD stack."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.finalizer import (
    ChunkEnvFeedback,
    finalize_chunk_transition,
)
from rlinf.algorithms.residual_hil_rlpd.learner import ResidualHilRLPDLearner
from rlinf.algorithms.residual_hil_rlpd.rollout import ResidualRolloutPolicy
from rlinf.algorithms.residual_hil_rlpd.transition import (
    SOURCE_ONLINE,
    SOURCE_ONLINE_INTERVENTION,
)
from rlinf.models.embodiment.residual_dobot_policy import (
    ResidualDobotActor,
    ResidualDobotCritic,
)


class _FakeBasePolicy(nn.Module):
    """Tiny trainable-parameter base used to prove the base stays frozen."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.nominal = torch.zeros(10, 8)
        for i in range(10):
            self.nominal[i, 0] = 0.15
            self.nominal[i, 2] = 0.30
            self.nominal[i, 3] = 1.0
            self.nominal[i, 7] = 0.5

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.nominal.clone().unsqueeze(0)


def _obs(value: float):
    return {
        "main_images": torch.full((1, 3, 16, 16), value, dtype=torch.float32),
        "prev_states": torch.full((1, 8), value, dtype=torch.float32),
    }


def _run_dummy_training(num_episodes: int = 4, chunks_per_episode: int = 2):
    codec = ResidualCodec()
    base = _FakeBasePolicy()
    actor = ResidualDobotActor(hidden=32, image_channels=3)
    critic = ResidualDobotCritic(hidden=32, image_channels=3, num_q_heads=4)
    policy = ResidualRolloutPolicy(base, actor, codec, device="cpu")
    learner = ResidualHilRLPDLearner(
        actor=actor,
        critic=critic,
        codec=codec,
        batch_size=4,
        utd_ratio=1.0,
        base_only_collect_steps=0,
        critic_only_updates=0,
        residual_scale_ramp_updates=50,
        min_demo_size=1,
        num_q_sample=2,
        device="cpu",
    )

    base_before = {
        name: param.detach().clone() for name, param in base.named_parameters()
    }
    episode_id = 0
    for episode in range(num_episodes):
        for chunk_id in range(chunks_per_episode):
            images = torch.full((1, 3, 16, 16), float(episode), dtype=torch.float32)
            proprio = torch.full((1, 8), float(chunk_id), dtype=torch.float32)
            commanded, audit = policy.rollout_chunk(
                images,
                proprio,
                residual_scale=learner.residual_scale(),
            )

            executed = commanded.copy()
            human_mask = np.zeros(10, dtype=bool)
            source = SOURCE_ONLINE
            if episode == 1 and chunk_id == 1:
                # Scripted human intervention: force-close gripper mid-chunk.
                executed[5, 7] = 0.0
                human_mask[5] = True
                source = SOURCE_ONLINE_INTERVENTION

            feedback = ChunkEnvFeedback(
                executed_actions=executed,
                executed_action_mask=np.ones(10, dtype=bool),
                gripper_bypass_mask=audit.gripper_bypass_mask,
                human_intervention_mask=human_mask,
                handoff_hold_mask=np.zeros(10, dtype=bool),
                rewards=np.full(10, 0.05, dtype=np.float32),
                terminations=np.zeros(10, dtype=bool),
                truncations=np.zeros(10, dtype=bool),
                reward_label_valid=True,
            )
            transition, valid, reason = finalize_chunk_transition(
                curr_obs=_obs(float(episode)),
                next_obs=_obs(float(episode)),
                audit=audit,
                feedback=feedback,
                codec=codec,
                source=source,
                base_fingerprint="dummy-base-fp",
                episode_id=episode_id,
                chunk_id=chunk_id,
                next_nominal_actions=audit.nominal_actions,
            )
            assert valid, reason
            assert transition is not None
            learner.add_transition(transition)
        episode_id += 1

    metrics_list = []
    for _ in range(20):
        if learner.can_update():
            metrics_list.append(learner.update())
    return base, base_before, learner, metrics_list


def test_dummy_e2e_trains_and_keeps_base_frozen():
    base, base_before, learner, metrics_list = _run_dummy_training()

    # Base must remain byte-for-byte unchanged.
    for name, param in base.named_parameters():
        torch.testing.assert_close(base_before[name], param.detach())

    # Learner actually updated.
    assert len(metrics_list) > 0
    assert learner.update_counter == len(metrics_list)
    assert metrics_list[-1]["can_update"] == 1.0
    assert learner.residual_scale() > 0.0

    # Dual-buffer semantics: the intervention chunk lives in both buffers.
    sizes = learner.buffer.sizes()
    assert sizes["online"] >= 8
    assert sizes["demo"] >= 1
    assert not learner.buffer.waiting_for_demo()

    # One more update after the run still works (no writeback/NaN).
    metrics = learner.update()
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


def test_dummy_e2e_checkpoint_round_trip():
    _, _, learner, _ = _run_dummy_training()
    state = learner.state_dict()

    rebuilt = ResidualHilRLPDLearner(
        actor=ResidualDobotActor(hidden=32, image_channels=3),
        critic=ResidualDobotCritic(hidden=32, image_channels=3, num_q_heads=4),
        codec=ResidualCodec(),
        batch_size=4,
        utd_ratio=1.0,
        base_only_collect_steps=0,
        critic_only_updates=0,
        residual_scale_ramp_updates=50,
        min_demo_size=1,
        num_q_sample=2,
        device="cpu",
    )
    rebuilt.load_state_dict(state)

    assert rebuilt.update_counter == learner.update_counter
    assert rebuilt.buffer.sizes() == learner.buffer.sizes()
    for (n1, p1), (n2, p2) in zip(
        rebuilt.actor.named_parameters(), learner.actor.named_parameters()
    ):
        torch.testing.assert_close(p1, p2)
    # Both learners can keep updating after the restore.
    assert rebuilt.can_update() == learner.can_update()
