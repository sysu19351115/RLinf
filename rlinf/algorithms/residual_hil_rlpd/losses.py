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

"""SAC/RLPD loss math for the hybrid residual policy."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.residual_dobot_policy import (
    GRIPPER_NUM_MODES,
    ResidualDobotActor,
    ResidualDobotCritic,
)


def compute_gripper_reinforce_loss(
    gripper_log_prob_total: torch.Tensor,
    q_mean: torch.Tensor,
    alpha_gripper: torch.Tensor,
) -> torch.Tensor:
    """Entropy-regularized REINFORCE objective for the discrete gripper policy.

    ``q_mean`` is detached (treated as reward) and ``alpha_gripper`` multiplies
    ``log pi`` inside the coefficient, so the score-function gradient equals

        E[(alpha_g * log pi - Q) * grad log pi]

    i.e. changing ``alpha_gripper`` really changes the policy gradient.  A plain
    additive ``-alpha * log pi`` term would only be a zero-mean baseline.

    Args:
        gripper_log_prob_total: Total (mask-summed) log-prob of the sampled
            gripper chunk, shape ``[B]``.  This is the differentiable term.
        q_mean: Mean critic value per sample, shape ``[B]``.
        alpha_gripper: Scalar entropy coefficient.
    """
    return torch.mean(
        (-q_mean.detach() + alpha_gripper * gripper_log_prob_total.detach())
        * gripper_log_prob_total
    )


def compute_q_target(
    next_q_min: torch.Tensor,
    discounted_return: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    discount_steps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute ``y = R + d * gamma^K * min_Q(s', z')`` (no entropy backup)."""
    discount = gamma ** discount_steps.to(dtype=next_q_min.dtype)
    return discounted_return + bootstrap_mask.to(dtype=next_q_min.dtype) * (
        discount * next_q_min
    )


def compute_critic_loss(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between all ensemble heads and the shared target."""
    return torch.mean((q_pred - q_target.unsqueeze(-1)) ** 2)


def sample_target_actions(
    target_actor: ResidualDobotActor,
    images: torch.Tensor,
    proprio: torch.Tensor,
    nominal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample next-chunk actions from the target actor (for bootstrap)."""
    u, mode, log_prob, _ = target_actor.sample(images, proprio, nominal)
    one_hot = torch.zeros(
        *mode.shape, GRIPPER_NUM_MODES, dtype=images.dtype, device=images.device
    )
    one_hot.scatter_(-1, mode.unsqueeze(-1), 1.0)
    return u, one_hot, log_prob


def compute_actor_loss(
    actor: ResidualDobotActor,
    critic: ResidualDobotCritic,
    images: torch.Tensor,
    proprio: torch.Tensor,
    nominal: torch.Tensor,
    executed_mask: torch.Tensor,
    alpha_arm: torch.Tensor,
    alpha_gripper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the hybrid actor objective.

    Returns:
        ``(actor_loss, arm_log_prob_mean, gripper_log_prob_mean, u_sample, gripper_logits)``.
    """
    mean, log_std, gripper_logits = actor(images, proprio, nominal)
    std = log_std.exp().clamp_min(1e-6)
    noise = torch.randn_like(mean)
    pre_activation = mean + std * noise
    u_sample = torch.tanh(pre_activation)

    from rlinf.models.embodiment.residual_dobot_policy.policy import (
        _tanh_gaussian_log_prob,
    )

    arm_log_prob = _tanh_gaussian_log_prob(pre_activation, mean, log_std, u_sample)
    mask = executed_mask.to(arm_log_prob.dtype)
    arm_log_prob_total = (arm_log_prob * mask).sum(dim=-1)  # [B, H] -> [B, H]

    # Sample the full joint gripper chunk (one mode per step) and use
    # REINFORCE with the entropy bonus for the discrete part; the arm part is
    # reparameterized. Un-executed tail steps are excluded from log-probs.
    gripper_probs = torch.softmax(gripper_logits, dim=-1)
    gripper_dist = torch.distributions.Categorical(probs=gripper_probs)
    gripper_mode = gripper_dist.sample()
    gripper_log_probs = gripper_dist.log_prob(gripper_mode)
    gripper_log_prob_total = (gripper_log_probs * mask).sum(dim=-1)
    one_hot = torch.zeros(
        *gripper_mode.shape,
        GRIPPER_NUM_MODES,
        dtype=images.dtype,
        device=images.device,
    )
    one_hot.scatter_(-1, gripper_mode.unsqueeze(-1), 1.0)
    q = critic(
        images,
        proprio,
        nominal,
        u_sample,
        one_hot,
        executed_mask,
    )  # [B, heads]
    q_mean = q.mean(dim=-1)  # [B]

    arm_loss = torch.mean(alpha_arm * arm_log_prob_total - q_mean)
    gripper_loss = compute_gripper_reinforce_loss(
        gripper_log_prob_total,
        q_mean,
        alpha_gripper,
    )
    actor_loss = arm_loss + gripper_loss
    return (
        actor_loss,
        torch.mean(arm_log_prob_total),
        torch.mean(gripper_log_prob_total),
        u_sample,
        gripper_logits,
    )


def compute_alpha_loss(
    log_alpha: torch.Tensor,
    log_prob_mean: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Standard SAC alpha objective."""
    return -(log_alpha * (log_prob_mean + target_entropy).detach())
