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

"""EnvWorker-side finalization of chunk transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.rollout import RolloutChunkAudit
from rlinf.algorithms.residual_hil_rlpd.transition import (
    ResidualChunkTransition,
    build_residual_chunk_transition,
)


@dataclass
class ChunkEnvFeedback:
    """Per-step real execution feedback returned by the env worker."""

    executed_actions: np.ndarray  # [H, 8], zeros for non-executed tail
    executed_action_mask: np.ndarray  # [H]
    gripper_bypass_mask: np.ndarray  # [H] actually used (incl. human merges)
    human_intervention_mask: np.ndarray  # [H]
    handoff_hold_mask: np.ndarray  # [H]
    rewards: np.ndarray  # [H]
    terminations: np.ndarray  # [H]
    truncations: np.ndarray  # [H]
    reward_label_valid: bool = True


def finalize_chunk_transition(
    *,
    curr_obs: dict[str, Any],
    next_obs: dict[str, Any],
    audit: RolloutChunkAudit,
    feedback: ChunkEnvFeedback,
    codec: ResidualCodec,
    source: int,
    base_fingerprint: str,
    episode_id: int,
    chunk_id: int,
    gamma: float = 0.99,
    next_nominal_actions: np.ndarray | None = None,
) -> tuple[ResidualChunkTransition | None, bool, str]:
    """Finalize one chunk decision; never invents executed actions."""
    transition, valid, reason = build_residual_chunk_transition(
        curr_obs=curr_obs,
        next_obs=next_obs,
        nominal_actions=audit.nominal_actions,
        next_nominal_actions=next_nominal_actions,
        sampled_arm_residual=audit.sampled_arm_residual,
        sampled_gripper_mode=audit.sampled_gripper_mode,
        commanded_actions=audit.commanded_actions,
        executed_actions=feedback.executed_actions,
        gripper_bypass_mask=feedback.gripper_bypass_mask,
        rewards=feedback.rewards,
        executed_action_mask=feedback.executed_action_mask,
        human_intervention_mask=feedback.human_intervention_mask,
        handoff_hold_mask=feedback.handoff_hold_mask,
        terminations=feedback.terminations,
        truncations=feedback.truncations,
        reward_label_valid=feedback.reward_label_valid,
        codec=codec,
        source=source,
        policy_version=audit.policy_version,
        base_fingerprint=base_fingerprint,
        gamma=gamma,
        episode_id=episode_id,
        chunk_id=chunk_id,
    )
    return transition, valid, reason
