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

"""SAC/RLPD loss and learner tests."""

from __future__ import annotations

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.learner import ResidualHilRLPDLearner
from rlinf.algorithms.residual_hil_rlpd.losses import (
    compute_alpha_loss,
    compute_critic_loss,
    compute_gripper_reinforce_loss,
    compute_q_target,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    CHUNK_LEN,
    build_residual_chunk_transition,
)
from rlinf.models.embodiment.residual_dobot_policy import (
    ResidualDobotActor,
    ResidualDobotCritic,
)


def _obs(value: float = 0.0):
    return {
        "main_images": torch.full((1, 3, 16, 16), value, dtype=torch.float32),
        "prev_states": torch.full((1, 8), value, dtype=torch.float32),
    }


def _make_transition(
    episode_id: int,
    chunk_id: int,
    human: bool = False,
    next_nominal: np.ndarray | None = None,
    policy_version: int = 0,
):
    codec = ResidualCodec()
    rng = np.random.default_rng(episode_id * 31 + chunk_id)
    h = CHUNK_LEN
    nominal = np.stack(
        [
            np.concatenate(
                [rng.uniform(-0.1, 0.1, size=3), [1.0, 0.0, 0.0, 0.0], [0.5]]
            )
            for _ in range(h)
        ]
    ).astype(np.float32)
    u = rng.uniform(-0.3, 0.3, size=(h, 6)).astype(np.float32)
    if next_nominal is None:
        next_nominal = nominal
    commanded = np.zeros((h, 8), dtype=np.float32)
    for i in range(h):
        pose = codec.compose(
            torch.as_tensor(nominal[i, :7], dtype=torch.float32),
            torch.as_tensor(u[i], dtype=torch.float32),
        ).numpy()
        commanded[i] = np.concatenate([pose, [nominal[i, 7]]])
    executed = commanded.copy()
    human_mask = np.zeros(h, dtype=bool)
    if human:
        executed[5, 7] = 0.0
        human_mask[5] = True
    transition, valid, reason = build_residual_chunk_transition(
        curr_obs=_obs(0.1),
        next_obs=_obs(0.2),
        nominal_actions=nominal,
        next_nominal_actions=next_nominal,
        sampled_arm_residual=u,
        sampled_gripper_mode=np.zeros(h, dtype=np.int64),
        commanded_actions=commanded,
        executed_actions=executed,
        gripper_bypass_mask=np.zeros(h, dtype=bool),
        rewards=rng.uniform(0.0, 0.1, size=h).astype(np.float32),
        executed_action_mask=np.ones(h, dtype=bool),
        human_intervention_mask=human_mask,
        handoff_hold_mask=np.zeros(h, dtype=bool),
        terminations=np.zeros(h, dtype=bool),
        truncations=np.zeros(h, dtype=bool),
        reward_label_valid=True,
        codec=codec,
        source=1 if human else 0,
        policy_version=policy_version,
        base_fingerprint="base",
        episode_id=episode_id,
        chunk_id=chunk_id,
    )
    assert valid, reason
    return transition


def _learner(**overrides):
    kwargs = {
        "actor": ResidualDobotActor(hidden=32, image_channels=3),
        "critic": ResidualDobotCritic(hidden=32, image_channels=3, num_q_heads=4),
        "codec": ResidualCodec(),
        "batch_size": 4,
        "utd_ratio": 1.0,
        "base_only_collect_steps": 0,
        "critic_only_updates": 0,
        "residual_scale_ramp_updates": 10,
        "min_demo_size": 1,
        "num_q_sample": 2,
    }
    kwargs.update(overrides)
    return ResidualHilRLPDLearner(**kwargs)


def test_q_target_hand_calculation():
    q_next = torch.tensor([1.0, 2.0])
    returns = torch.tensor([0.5, 0.5])
    bootstrap = torch.tensor([True, False])
    steps = torch.tensor([3.0, 3.0])

    y = compute_q_target(q_next, returns, bootstrap, steps, gamma=0.99)

    torch.testing.assert_close(y[0], torch.tensor(0.5 + 0.99**3 * 1.0))
    torch.testing.assert_close(y[1], torch.tensor(0.5))


def test_critic_loss_is_mse():
    q_pred = torch.tensor([[1.0, 2.0, 3.0]])
    q_target = torch.tensor([2.0])
    loss = compute_critic_loss(q_pred, q_target)
    torch.testing.assert_close(loss, torch.tensor((1.0 + 0.0 + 1.0) / 3.0))


def test_alpha_loss_sign():
    log_alpha = torch.tensor(1.0, requires_grad=True)
    # log_prob is more negative than the target -> alpha should increase.
    loss = compute_alpha_loss(log_alpha, torch.tensor(-10.0), target_entropy=-6.0)
    assert loss.item() > 0
    loss.backward()
    assert log_alpha.grad is not None and log_alpha.grad.item() > 0


def test_learner_waits_for_demo_and_then_updates():
    learner = _learner()
    assert learner.can_update() is False
    learner.add_transition(_make_transition(0, 0))  # online only
    assert learner.can_update() is False  # waiting for demo

    learner.add_transition(_make_transition(0, 1, human=True))
    assert learner.can_update() is True

    metrics = learner.update()

    assert metrics["can_update"] == 1.0
    assert metrics["demo_size"] == 1
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    assert metrics["residual_scale"] > 0.0


def test_learner_updates_change_parameters():
    learner = _learner()
    for episode in range(3):
        learner.add_transition(_make_transition(episode, 0))
        learner.add_transition(_make_transition(episode, 1, human=True))
    before = {
        name: param.detach().clone() for name, param in learner.actor.named_parameters()
    }

    for _ in range(3):
        learner.update()

    changed = any(
        not torch.equal(before[name], param.detach())
        for name, param in learner.actor.named_parameters()
    )
    assert changed


def test_learner_state_dict_round_trip():
    learner = _learner()
    for episode in range(2):
        learner.add_transition(_make_transition(episode, 0))
        learner.add_transition(_make_transition(episode, 1, human=True))
    learner.update()

    state = learner.state_dict()
    restored = _learner()
    restored.load_state_dict(state)

    assert restored.update_counter == learner.update_counter
    assert restored.executed_step_counter == learner.executed_step_counter
    torch.testing.assert_close(restored.log_alpha_arm, learner.log_alpha_arm)
    for (n1, p1), (n2, p2) in zip(
        restored.actor.named_parameters(), learner.actor.named_parameters()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


def test_learner_critic_only_phase_disables_actor():
    learner = _learner(critic_only_updates=10)
    for episode in range(2):
        learner.add_transition(_make_transition(episode, 0))
        learner.add_transition(_make_transition(episode, 1, human=True))
    before = {
        name: param.detach().clone() for name, param in learner.actor.named_parameters()
    }

    learner.update()

    assert learner.actor_enabled() is False
    assert learner.residual_scale() == 0.0
    for name, param in learner.actor.named_parameters():
        torch.testing.assert_close(before[name], param.detach())


def _score_gradient_mc(logits, q, alpha, old_formula, n_samples, seed):
    """Estimate the score-function gradient for a categorical policy via MC.

    ``logits`` is a leaf tensor; ``q`` is a fixed reward per outcome.  The
    coefficient is detached from the logits, exactly like the production loss.
    The old (broken) formula uses ``-(q + alpha) * log pi`` which makes
    ``alpha`` a zero-mean baseline; the new formula uses
    ``(-q + alpha * log pi) * log pi``.
    """
    torch.manual_seed(seed)
    probs = torch.softmax(logits.detach(), dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)  # differentiable term
    samples = torch.multinomial(probs, n_samples, replacement=True)
    lp = log_probs[samples]
    qs = q[samples]
    if old_formula:
        loss = torch.mean(-(qs.detach() + alpha) * lp)
    else:
        loss = compute_gripper_reinforce_loss(lp, qs, torch.tensor(alpha))
    loss.backward()
    grad = logits.grad.clone()
    logits.grad = None
    return grad


def test_gripper_reinforce_loss_matches_analytic_score_gradient():
    """Alpha must change the gripper policy gradient exactly as
    ``E[(alpha * log pi - Q) grad log pi]`` predicts (non-uniform policy)."""
    logits = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    probs = torch.softmax(logits.detach(), dim=-1)
    log_probs = torch.log_softmax(logits.detach(), dim=-1)
    q = torch.tensor([0.3, -0.2, 0.1])

    n = 200_000
    grad_new_0 = _score_gradient_mc(logits, q, 0.0, False, n, seed=7)
    grad_new_1 = _score_gradient_mc(logits, q, 1.0, False, n, seed=7)
    expected = probs * log_probs - probs * (probs * log_probs).sum()
    torch.testing.assert_close(
        grad_new_1 - grad_new_0,
        expected,
        atol=0.01,
        rtol=0.0,
    )
    assert float((grad_new_1 - grad_new_0).abs().max()) > 0.1


def test_alpha_baseline_formula_does_not_change_gradients():
    """Regression guard: the old ``-(Q + alpha) * log pi`` formula makes alpha
    a zero-mean baseline, so its alpha-difference is pure MC noise (~0)."""
    logits = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    q = torch.tensor([0.3, -0.2, 0.1])
    n = 200_000
    grad_old_0 = _score_gradient_mc(logits, q, 0.0, True, n, seed=7)
    grad_old_1 = _score_gradient_mc(logits, q, 1.0, True, n, seed=7)
    assert float((grad_old_1 - grad_old_0).abs().max()) < 0.01


def test_alpha_gripper_changes_gripper_gradients():
    """End-to-end wiring: through compute_actor_loss, alpha_gripper must change
    the gripper gradient by more than sampling noise."""
    from rlinf.algorithms.residual_hil_rlpd.losses import compute_actor_loss

    torch.manual_seed(0)
    actor = ResidualDobotActor(hidden=32, image_channels=3)
    critic = ResidualDobotCritic(hidden=32, image_channels=3, num_q_heads=2)
    images = torch.randn(256, 3, 16, 16)
    proprio = torch.randn(256, 8)
    nominal = torch.randn(256, 10, 8)
    mask = torch.ones(256, 10, dtype=torch.bool)

    def _gripper_grad(alpha):
        actor.zero_grad()
        loss, _, _, _, _ = compute_actor_loss(
            actor,
            critic,
            images,
            proprio,
            nominal,
            mask,
            torch.tensor(0.1),
            torch.tensor(alpha),
        )
        loss.backward()
        return sum(
            p.grad.abs().sum().item()
            for p in actor.gripper_head.parameters()
            if p.grad is not None
        )

    grad_0 = _gripper_grad(0.0)
    grad_1 = _gripper_grad(1.0)
    # KEEP-dominant init makes the gripper distribution clearly non-uniform,
    # so the entropy term is strong; the alpha-driven difference must be far
    # above any 256-sample sampling noise (~1e-2).
    assert abs(grad_1 - grad_0) > 0.05


def test_policy_lag_rejects_future_and_unknown_versions():
    learner = _learner()
    # Future version (transition from a rolled-back/faster node) -> rejected.
    assert not learner.add_transition(
        _make_transition(0, 0, policy_version=learner.update_counter + 1)
    )
    # Unknown negative version -> rejected.
    assert not learner.add_transition(
        _make_transition(0, 1, policy_version=-1)
    )
    assert learner.policy_lag_rejected == 2
    assert learner.buffer.sizes()["online"] == 0


def test_policy_lag_rejects_beyond_threshold():
    learner = _learner(policy_lag_reject_threshold=3)
    # Version 0 is within threshold of update_counter=0.
    assert learner.add_transition(
        _make_transition(0, 0, policy_version=0)
    )
    learner.update_counter = 10  # simulate training progress
    # lag = 10 - 0 = 10 > 3 -> rejected.
    assert not learner.add_transition(
        _make_transition(0, 1, policy_version=0)
    )
    assert learner.policy_lag_rejected == 1


def test_sync_failure_freezes_residual_scale():
    learner = _learner(
        critic_only_updates=0,
        residual_scale_ramp_updates=10,
    )
    learner.update_counter = 5
    assert learner.residual_scale() > 0.0
    learner.last_sync_ok = False
    assert learner.residual_scale() == 0.0
    assert not learner.gripper_enabled()


def test_residual_scale_ramp_starts_at_zero_and_caps():
    learner = _learner(
        critic_only_updates=10,
        residual_scale_ramp_updates=100,
        residual_scale_cap=1.0,
    )
    assert learner.residual_scale() == 0.0
    learner.update_counter = 10
    assert learner.residual_scale() == 0.0
    learner.update_counter = 60
    assert abs(learner.residual_scale() - 0.5) < 1e-6
    learner.update_counter = 200
    assert learner.residual_scale() == 1.0

    capped = _learner(
        critic_only_updates=10,
        residual_scale_ramp_updates=100,
        residual_scale_cap=0.3,
    )
    capped.update_counter = 200
    assert capped.residual_scale() == 0.3


def test_gripper_enabled_threshold():
    learner = _learner(gripper_enable_after_updates=100)
    learner.update_counter = 50
    assert not learner.gripper_enabled()
    learner.update_counter = 100
    assert learner.gripper_enabled()
    learner.last_sync_ok = False
    assert not learner.gripper_enabled()
