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

"""Async-friendly RLPD learner for the hybrid residual policy."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.losses import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_loss,
    compute_q_target,
    sample_target_actions,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    ResidualChunkTransition,
)
from rlinf.data.residual_hil_replay import (
    ResidualHilReplayBuffer,
    transitions_to_torch_batch,
)
from rlinf.models.embodiment.residual_dobot_policy import (
    GRIPPER_NUM_MODES,
    ResidualDobotActor,
    ResidualDobotCritic,
)


class ResidualHilRLPDLearner:
    """Chunk-level RLPD learner; no Pi0.5 state is ever instantiated here."""

    def __init__(
        self,
        actor: ResidualDobotActor,
        critic: ResidualDobotCritic,
        codec: ResidualCodec,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 64,
        utd_ratio: float = 0.1,
        max_update_backlog: float = 500.0,
        base_only_collect_steps: int = 100,
        critic_only_updates: int = 500,
        residual_scale_ramp_updates: int = 2000,
        min_demo_size: int = 1,
        base_fingerprint: str | None = None,
        max_online_transitions: int = 20_000,
        max_demo_transitions: int = 5_000,
        max_online_bytes: float = 40e9,
        max_demo_bytes: float = 10e9,
        memory_high_watermark: float = 0.9,
        demo_ratio: float = 0.5,
        policy_lag_warn_threshold: int = 50,
        policy_lag_reject_threshold: int = 500,
        gripper_enable_after_updates: int = 1500,
        gripper_residual_enabled: bool = True,
        num_q_sample: int = 2,
        alpha_arm_init: float = 0.1,
        alpha_gripper_init: float = 0.05,
        target_entropy_arm: float = -6.0,
        device: str = "cpu",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)
        self.codec = codec
        self.device = device

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.utd_ratio = float(utd_ratio)
        self.min_steps_per_update = max(1.0 / self.utd_ratio, 0.5)
        self.max_update_backlog = float(max_update_backlog)
        self.base_only_collect_steps = int(base_only_collect_steps)
        self.critic_only_updates = int(critic_only_updates)
        self.residual_scale_ramp_updates = int(residual_scale_ramp_updates)
        self.num_q_sample = int(num_q_sample)
        self.target_entropy_arm = float(target_entropy_arm)
        self.target_entropy_gripper = -math.log(GRIPPER_NUM_MODES)

        self.log_alpha_arm = nn.Parameter(
            torch.tensor(math.log(alpha_arm_init), dtype=torch.float32, device=device)
        )
        self.log_alpha_gripper = nn.Parameter(
            torch.tensor(
                math.log(alpha_gripper_init), dtype=torch.float32, device=device
            )
        )
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.AdamW(
            [self.log_alpha_arm, self.log_alpha_gripper], lr=lr
        )

        self.buffer = ResidualHilReplayBuffer(
            min_demo_size=min_demo_size,
            codec_fingerprint=codec.fingerprint(),
            base_fingerprint=base_fingerprint,
            max_online_transitions=max_online_transitions,
            max_demo_transitions=max_demo_transitions,
            max_online_bytes=max_online_bytes,
            max_demo_bytes=max_demo_bytes,
            memory_high_watermark=memory_high_watermark,
            demo_ratio=demo_ratio,
        )
        self.base_fingerprint = base_fingerprint
        self.policy_lag_warn_threshold = int(policy_lag_warn_threshold)
        self.policy_lag_reject_threshold = int(policy_lag_reject_threshold)
        self.gripper_enable_after_updates = int(gripper_enable_after_updates)
        # Master switch: when False, the gripper residual is never applied on
        # the robot (rollout always receives gripper_enabled=False -> KEEP,
        # i.e. the frozen VLA's nominal gripper command is used as-is).
        self.gripper_residual_enabled = bool(gripper_residual_enabled)
        self.policy_lag_max = 0
        self.policy_lag_warned = 0
        self.policy_lag_rejected = 0
        self.last_sync_ok = True
        self.executed_step_counter = 0
        self.update_counter = 0
        self.utd_token_pool = 0.0

    # ── Data intake ─────────────────────────────────────────────────────────

    def add_transition(self, transition: ResidualChunkTransition) -> bool:
        """Ingest one transition; enforce policy-staleness bounds (P2-3)."""
        transition_version = int(
            np.asarray(transition.policy_version).reshape(-1)[0]
        )
        if transition_version < 0 or transition_version > self.update_counter:
            # Unknown (never trained) or future/rolled-back version: reject.
            self.policy_lag_rejected += 1
            return False
        lag = self.update_counter - transition_version
        if lag > self.policy_lag_reject_threshold:
            self.policy_lag_rejected += 1
            return False
        if lag > self.policy_lag_warn_threshold:
            self.policy_lag_warned += 1
        self.policy_lag_max = max(self.policy_lag_max, lag)
        added = self.buffer.add(transition)
        if added:
            self.executed_step_counter += int(transition.discount_steps[0])
            self.utd_token_pool = min(
                self.utd_token_pool + int(transition.discount_steps[0]),
                self.max_update_backlog,
            )
        return added

    # ── Scheduling ──────────────────────────────────────────────────────────

    def can_update(self) -> bool:
        if not self.buffer.can_train():
            return False
        if self.executed_step_counter < self.base_only_collect_steps:
            return False
        return self.utd_token_pool >= self.min_steps_per_update

    def actor_enabled(self) -> bool:
        # ``critic_only_updates`` counts gradient updates, not collected steps.
        return self.update_counter >= self.critic_only_updates

    def residual_scale(self) -> float:
        """Ramp the rollout residual scale from 0 to 1 over the ramp window."""
        if not self.last_sync_ok:
            # Weight sync failed: never ramp residual on top of stale weights.
            return 0.0
        if not self.actor_enabled():
            return 0.0
        if self.residual_scale_ramp_updates <= 0:
            return 1.0
        return min(1.0, self.update_counter / self.residual_scale_ramp_updates)

    def gripper_enabled(self) -> bool:
        """Discrete gripper override is separate from arm residual scale: it
        stays KEEP until an explicit update threshold is crossed (P2-4)."""
        if not self.gripper_residual_enabled:
            return False
        if not self.last_sync_ok:
            return False
        return self.update_counter >= self.gripper_enable_after_updates

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self) -> dict[str, float]:
        """Run one gradient update; consumes ``min_steps_per_update`` tokens."""
        if not self.can_update():
            return {
                "waiting_for_demo": float(self.buffer.waiting_for_demo()),
                "over_memory_watermark": float(self.buffer.over_memory_watermark()),
                "can_update": 0.0,
            }
        transitions, _ = self.buffer.sample(self.batch_size)
        batch = transitions_to_torch_batch(transitions)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        mask = batch["executed_action_mask"]

        def _grad_norm(params) -> float:
            norms = [
                p.grad.detach().float().norm().item()
                for p in params
                if p.grad is not None
            ]
            return float(np.sqrt(sum(n * n for n in norms))) if norms else 0.0

        def _param_delta(params, snapshot: dict[str, torch.Tensor]) -> float:
            total = 0.0
            for name, p in params:
                if p.grad is None:
                    continue
                delta = (p.detach() - snapshot[name]).float().norm().item()
                total += delta * delta
            return float(np.sqrt(total))

        # ── Target Q ──
        with torch.no_grad():
            u_next, gripper_next, _ = sample_target_actions(
                self.actor,
                batch["next_images"],
                batch["next_proprio"],
                batch["next_nominal"],
            )
            next_mask = torch.ones_like(mask)
            q_next = self.target_critic(
                batch["next_images"],
                batch["next_proprio"],
                batch["next_nominal"],
                u_next,
                gripper_next,
                next_mask,
            )  # [B, heads]
            head_idx = torch.randint(
                0,
                q_next.shape[-1],
                (self.batch_size, self.num_q_sample),
                device=self.device,
            )
            q_next_min = q_next.gather(1, head_idx).min(dim=1).values
            q_target = compute_q_target(
                q_next_min,
                batch["discounted_return"],
                batch["bootstrap_mask"],
                batch["discount_steps"],
                self.gamma,
            )

        # ── Critic ──
        q_pred = self.critic(
            batch["curr_images"],
            batch["curr_proprio"],
            batch["nominal"],
            batch["actions_arm"],
            batch["actions_gripper"],
            mask,
        )
        critic_loss = compute_critic_loss(q_pred, q_target)
        critic_snapshot = {
            name: p.detach().clone()
            for name, p in self.critic.named_parameters()
        }
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _grad_norm(self.critic.parameters())
        self.opt_critic.step()

        metrics: dict[str, float] = {
            "critic_loss": float(critic_loss.detach()),
            "critic_grad_norm": critic_grad_norm,
            "critic_param_delta": _param_delta(
                self.critic.named_parameters(), critic_snapshot
            ),
            "q_mean": float(q_pred.mean().detach()),
            "q_std": float(q_pred.std().detach()) if q_pred.numel() > 1 else 0.0,
            "q_ensemble_min": float(q_pred.min(dim=-1).values.mean().detach()),
            "q_ensemble_max": float(q_pred.max(dim=-1).values.mean().detach()),
            "q_ensemble_disagreement": float(
                (q_pred.max(dim=-1).values - q_pred.min(dim=-1).values)
                .mean()
                .detach()
            ),
            "q_target_mean": float(q_target.mean().detach()),
            "td_error": float((q_pred.mean(dim=-1) - q_target).abs().mean().detach()),
            "q_target_min": float(q_target.min().detach()),
            "q_target_max": float(q_target.max().detach()),
        }

        # ── Actor + alphas ──
        if self.actor_enabled():
            alpha_arm = self.log_alpha_arm.exp()
            alpha_gripper = self.log_alpha_gripper.exp()
            actor_loss, arm_log_prob, gripper_log_prob, u_sample, _ = compute_actor_loss(
                self.actor,
                self.critic,
                batch["curr_images"],
                batch["curr_proprio"],
                batch["nominal"],
                mask,
                alpha_arm,
                alpha_gripper,
            )
            actor_snapshot = {
                name: p.detach().clone()
                for name, p in self.actor.named_parameters()
            }
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            self.opt_actor.step()

            alpha_arm_loss = compute_alpha_loss(
                self.log_alpha_arm, arm_log_prob, self.target_entropy_arm
            )
            alpha_gripper_loss = compute_alpha_loss(
                self.log_alpha_gripper,
                gripper_log_prob,
                self.target_entropy_gripper,
            )
            alpha_loss = alpha_arm_loss + alpha_gripper_loss
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_grad_norm = _grad_norm(
                [self.log_alpha_arm, self.log_alpha_gripper]
            )
            self.opt_alpha.step()
            # Log the alpha actually used by the *next* policy update (i.e.
            # the post-step value), not the stale pre-step one.
            alpha_arm = self.log_alpha_arm.detach().exp()
            alpha_gripper = self.log_alpha_gripper.detach().exp()
            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach()),
                    "actor_grad_norm": actor_grad_norm,
                    "actor_param_delta": _param_delta(
                        self.actor.named_parameters(), actor_snapshot
                    ),
                    "arm_residual_abs_mean": float(
                        u_sample.detach().abs().mean()
                    ),
                    "arm_residual_abs_max": float(
                        u_sample.detach().abs().max()
                    ),
                    "alpha_grad_norm": alpha_grad_norm,
                    "alpha_arm": float(alpha_arm),
                    "alpha_gripper": float(alpha_gripper),
                    "gripper_entropy": float(
                        -gripper_log_prob.detach().clamp_min(1e-8).mean()
                    ),
                }
            )

        # ── Target polyak ──
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

        self.update_counter += 1
        self.utd_token_pool -= self.min_steps_per_update
        metrics.update(
            {
                "update_counter": float(self.update_counter),
                "residual_scale": float(self.residual_scale()),
                "executed_steps": float(self.executed_step_counter),
                "online_size": float(self.buffer.sizes()["online"]),
                "demo_size": float(self.buffer.sizes()["demo"]),
                "policy_lag_max": float(self.policy_lag_max),
                "policy_lag_warned": float(self.policy_lag_warned),
                "policy_lag_rejected": float(self.policy_lag_rejected),
                "buffer_accepted": float(self.buffer.accepted_count),
                "buffer_duplicates": float(self.buffer.duplicate_count),
                "buffer_rejected": float(self.buffer.rejected_count),
                "buffer_evicted": float(self.buffer.evicted_count),
                "buffer_online_bytes": float(
                    self.buffer.memory_usage()["online_bytes"]
                ),
                "buffer_demo_bytes": float(
                    self.buffer.memory_usage()["demo_bytes"]
                ),
                "over_memory_watermark": float(
                    self.buffer.over_memory_watermark()
                ),
                "can_update": 1.0,
            }
        )
        return metrics

    # ── Persistence (Pi0.5 weights are never part of this state) ───────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha_arm": self.log_alpha_arm.detach().cpu().clone(),
            "log_alpha_gripper": self.log_alpha_gripper.detach().cpu().clone(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "buffer": self.buffer.state_dict(),
            "executed_step_counter": self.executed_step_counter,
            "update_counter": self.update_counter,
            "utd_token_pool": self.utd_token_pool,
            "policy_lag_max": self.policy_lag_max,
            "policy_lag_warned": self.policy_lag_warned,
            "policy_lag_rejected": self.policy_lag_rejected,
            "last_sync_ok": bool(self.last_sync_ok),
            "codec_fingerprint": self.codec.fingerprint(),
            "base_fingerprint": self.base_fingerprint,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["codec_fingerprint"] != self.codec.fingerprint():
            raise ValueError("codec fingerprint mismatch on learner resume")
        if state["base_fingerprint"] != self.base_fingerprint:
            raise ValueError("base fingerprint mismatch on learner resume")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.log_alpha_arm.data.copy_(state["log_alpha_arm"])
        self.log_alpha_gripper.data.copy_(state["log_alpha_gripper"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.opt_alpha.load_state_dict(state["opt_alpha"])
        for optimizer in (self.opt_actor, self.opt_critic, self.opt_alpha):
            for group_state in optimizer.state.values():
                for key, value in group_state.items():
                    if isinstance(value, torch.Tensor):
                        group_state[key] = value.to(self.device)
        self.buffer.load_state_dict(state["buffer"])
        self.executed_step_counter = int(state["executed_step_counter"])
        self.update_counter = int(state["update_counter"])
        self.utd_token_pool = float(state["utd_token_pool"])
        self.policy_lag_max = int(state.get("policy_lag_max", 0))
        self.policy_lag_warned = int(state.get("policy_lag_warned", 0))
        self.policy_lag_rejected = int(state.get("policy_lag_rejected", 0))
        self.last_sync_ok = bool(state.get("last_sync_ok", True))
