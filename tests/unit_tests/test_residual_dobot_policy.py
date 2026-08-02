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

"""Tests for the hybrid residual actor and Q ensemble."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.residual_dobot_policy import (
    ResidualDobotActor,
    ResidualDobotCritic,
)


def _inputs(batch: int = 4):
    images = torch.randn(batch, 3, 64, 64)
    proprio = torch.randn(batch, 8)
    nominal = torch.randn(batch, 10, 8)
    return images, proprio, nominal


def test_actor_output_shapes_and_initial_near_zero():
    actor = ResidualDobotActor()
    images, proprio, nominal = _inputs()

    mean, log_std, gripper_logits = actor(images, proprio, nominal)

    assert mean.shape == (4, 10, 6)
    assert log_std.shape == (4, 10, 6)
    assert gripper_logits.shape == (4, 10, 3)
    # Zero-initialized mean -> first policy is "keep nominal".
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert float(log_std.min()) < -3.0


def test_actor_sample_bounds_and_log_prob():
    actor = ResidualDobotActor()
    images, proprio, nominal = _inputs()

    u, gripper_mode, log_prob, _ = actor.sample(images, proprio, nominal)

    assert u.shape == (4, 10, 6)
    assert (u.abs() <= 1.0 + 1e-5).all()
    assert gripper_mode.shape == (4, 10)
    assert (gripper_mode >= 0).all() and (gripper_mode <= 2).all()
    assert torch.isfinite(log_prob).all()
    assert log_prob.shape == (4, 10)


def test_actor_deterministic_eval_is_stable():
    actor = ResidualDobotActor().eval()
    images, proprio, nominal = _inputs()

    u1, m1, _, _ = actor.sample(images, proprio, nominal, deterministic=True)
    u2, m2, _, _ = actor.sample(images, proprio, nominal, deterministic=True)

    assert torch.equal(u1, u2)
    assert torch.equal(m1, m2)
    assert torch.allclose(u1, torch.zeros_like(u1), atol=1e-6)


def test_critic_ensemble_shapes_and_layernorm():
    critic = ResidualDobotCritic(num_q_heads=5)
    images, proprio, nominal = _inputs()
    arm = torch.randn(4, 10, 6)
    gripper = torch.zeros(4, 10, 3)
    gripper[..., 0] = 1.0
    mask = torch.ones(4, 10, dtype=torch.bool)

    q = critic(images, proprio, nominal, arm, gripper, mask)

    assert q.shape == (4, 5)
    assert torch.isfinite(q).all()
    assert any(
        isinstance(layer, torch.nn.LayerNorm) for head in critic.heads for layer in head
    )


def test_actor_critic_parameters_are_disjoint():
    actor = ResidualDobotActor()
    critic = ResidualDobotCritic()
    actor_ids = {id(p) for p in actor.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_state_dict_round_trip():
    actor = ResidualDobotActor()
    critic = ResidualDobotCritic(num_q_heads=5)
    actor_sd = actor.state_dict()
    critic_sd = critic.state_dict()

    actor2 = ResidualDobotActor()
    critic2 = ResidualDobotCritic(num_q_heads=5)
    actor2.load_state_dict(actor_sd)
    critic2.load_state_dict(critic_sd)

    images, proprio, nominal = _inputs()
    mean1, _, _ = actor(images, proprio, nominal)
    mean2, _, _ = actor2(images, proprio, nominal)
    assert torch.equal(mean1, mean2)
    q1 = critic(
        images,
        proprio,
        nominal,
        torch.randn(4, 10, 6),
        torch.zeros(4, 10, 3),
        torch.ones(4, 10, dtype=torch.bool),
    )
    q2 = critic2(
        images,
        proprio,
        nominal,
        torch.randn(4, 10, 6),
        torch.zeros(4, 10, 3),
        torch.ones(4, 10, dtype=torch.bool),
    )
    assert q1.shape == q2.shape
