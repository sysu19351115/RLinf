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

"""Chunk-SMDP transition schema for the residual HIL-RLPD algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rlinf.algorithms.residual_hil_rlpd.action_codec import (
    ARM_RESIDUAL_DIM,
    ResidualCodec,
    invert_gripper_mode,
)

CHUNK_LEN = 10

# Transition sources.
SOURCE_ONLINE = 0
SOURCE_ONLINE_INTERVENTION = 1
SOURCE_OFFLINE_DEMO = 2

# Validation / build status.
STATUS_OK = "ok"
STATUS_NON_CONTIGUOUS_EXECUTED_MASK = "non_contiguous_executed_mask"
STATUS_K_ZERO = "k_zero"
STATUS_OUT_OF_SUPPORT = "out_of_support"
STATUS_NON_FINITE = "non_finite"
STATUS_REWARD_LABEL_INVALID = "reward_label_invalid"
STATUS_SHAPE_MISMATCH = "shape_mismatch"
STATUS_FINGERPRINT_MISMATCH = "fingerprint_mismatch"
STATUS_UNKNOWN_TERMINATION = "unknown_termination"
STATUS_MISSING_NEXT_NOMINAL = "missing_next_nominal"


@dataclass
class ResidualChunkTransition:
    """One chunk-level SMDP transition.

    Field semantics follow the plan: ``executed_actions`` are the real
    controller-accepted commands; ``actions`` are always derived by inverting
    executed actions relative to nominal actions, never copied from model
    samples.
    """

    curr_obs: dict[str, Any]
    next_obs: dict[str, Any]
    nominal_actions: np.ndarray  # [H, 8]
    next_nominal_actions: np.ndarray  # [H, 8]; zeros when unknown
    sampled_arm_residual: np.ndarray  # [H, 6]
    sampled_gripper_mode: np.ndarray  # [H]
    commanded_actions: np.ndarray  # [H, 8]
    gripper_bypass_mask: np.ndarray  # [H]
    executed_actions: np.ndarray  # [H, 8]
    actions_arm: np.ndarray  # [H, 6]
    actions_gripper: np.ndarray  # [H, 3] one-hot
    rewards: np.ndarray  # [H]
    executed_action_mask: np.ndarray  # [H]
    human_intervention_mask: np.ndarray  # [H]
    handoff_hold_mask: np.ndarray  # [H]
    terminations: np.ndarray  # [H]
    truncations: np.ndarray  # [H]
    bootstrap_mask: np.ndarray  # [1]
    discount_steps: np.ndarray  # [1] = K
    discounted_return: float
    transition_valid: np.ndarray  # [1]
    reward_label_valid: np.ndarray  # [1]
    source: int
    policy_version: np.ndarray  # [1]
    base_fingerprint: str
    codec_fingerprint: str
    episode_id: int = -1
    chunk_id: int = -1
    status_reason: str = STATUS_OK

    def validate(self, codec: ResidualCodec) -> None:
        """Fail closed on any schema violation."""
        h = self.nominal_actions.shape[0]
        expected = {
            "nominal_actions": (h, 8),
            "next_nominal_actions": (h, 8),
            "sampled_arm_residual": (h, 6),
            "sampled_gripper_mode": (h,),
            "commanded_actions": (h, 8),
            "gripper_bypass_mask": (h,),
            "executed_actions": (h, 8),
            "actions_arm": (h, 6),
            "actions_gripper": (h, 3),
            "rewards": (h,),
            "executed_action_mask": (h,),
            "human_intervention_mask": (h,),
            "handoff_hold_mask": (h,),
            "terminations": (h,),
            "truncations": (h,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        if self.codec_fingerprint != codec.fingerprint():
            raise ValueError(
                "codec fingerprint mismatch between transition and learner"
            )
        if not np.isfinite(self.actions_arm).all():
            raise ValueError("actions_arm must be finite")
        if not (
            (self.actions_gripper.sum(axis=-1, keepdims=True) == 1.0)
            & (self.actions_gripper >= 0.0)
        ).all():
            raise ValueError("actions_gripper must be one-hot")


def _is_prefix(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(~mask)
    return not idx.size or bool(idx[0] == mask.size - len(idx))


def build_residual_chunk_transition(
    *,
    curr_obs: dict[str, Any],
    next_obs: dict[str, Any],
    nominal_actions: np.ndarray,
    next_nominal_actions: np.ndarray | None = None,
    sampled_arm_residual: np.ndarray,
    sampled_gripper_mode: np.ndarray,
    commanded_actions: np.ndarray,
    executed_actions: np.ndarray,
    gripper_bypass_mask: np.ndarray,
    rewards: np.ndarray,
    executed_action_mask: np.ndarray,
    human_intervention_mask: np.ndarray,
    handoff_hold_mask: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    reward_label_valid: bool,
    codec: ResidualCodec,
    source: int,
    policy_version: int,
    base_fingerprint: str,
    gamma: float = 0.99,
    episode_id: int = -1,
    chunk_id: int = -1,
) -> tuple[ResidualChunkTransition, bool, str]:
    """Build a validated chunk transition from real executed actions.

    Returns:
        ``(transition, valid, reason)``. Invalid transitions must never enter
        a training buffer.
    """
    h = len(nominal_actions)
    if h != CHUNK_LEN:
        return (
            ResidualChunkTransition(
                curr_obs=curr_obs,
                next_obs=next_obs,
                nominal_actions=np.asarray(nominal_actions, dtype=np.float32),
                next_nominal_actions=np.zeros((h, 8), dtype=np.float32),
                sampled_arm_residual=np.asarray(sampled_arm_residual, dtype=np.float32),
                sampled_gripper_mode=np.asarray(sampled_gripper_mode, dtype=np.int64),
                commanded_actions=np.asarray(commanded_actions, dtype=np.float32),
                gripper_bypass_mask=np.asarray(gripper_bypass_mask, dtype=bool),
                executed_actions=np.asarray(executed_actions, dtype=np.float32),
                actions_arm=np.zeros((h, ARM_RESIDUAL_DIM), dtype=np.float32),
                actions_gripper=np.zeros((h, 3), dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                executed_action_mask=np.asarray(executed_action_mask, dtype=bool),
                human_intervention_mask=np.asarray(human_intervention_mask, dtype=bool),
                handoff_hold_mask=np.asarray(handoff_hold_mask, dtype=bool),
                terminations=np.asarray(terminations, dtype=bool),
                truncations=np.asarray(truncations, dtype=bool),
                bootstrap_mask=np.zeros(1, dtype=bool),
                discount_steps=np.zeros(1, dtype=np.int64),
                discounted_return=0.0,
                transition_valid=np.zeros(1, dtype=bool),
                reward_label_valid=np.asarray([reward_label_valid], dtype=bool),
                source=source,
                policy_version=np.asarray([policy_version], dtype=np.int64),
                base_fingerprint=base_fingerprint,
                codec_fingerprint=codec.fingerprint(),
                episode_id=episode_id,
                chunk_id=chunk_id,
                status_reason=STATUS_SHAPE_MISMATCH,
            ),
            False,
            STATUS_SHAPE_MISMATCH,
        )

    nominal = np.asarray(nominal_actions, dtype=np.float32)
    next_nominal = np.asarray(
        next_nominal_actions
        if next_nominal_actions is not None
        else np.zeros((h, 8), dtype=np.float32),
        dtype=np.float32,
    )
    executed = np.asarray(executed_actions, dtype=np.float32)
    sampled_residual = np.asarray(sampled_arm_residual, dtype=np.float32)
    sampled_gripper = np.asarray(sampled_gripper_mode, dtype=np.int64)
    executed_mask = np.asarray(executed_action_mask, dtype=bool)
    human_mask = np.asarray(human_intervention_mask, dtype=bool)
    handoff_mask = np.asarray(handoff_hold_mask, dtype=bool)
    term = np.asarray(terminations, dtype=bool)
    trunc = np.asarray(truncations, dtype=bool)
    reward_values = np.asarray(rewards, dtype=np.float32)
    bypass = np.asarray(gripper_bypass_mask, dtype=bool)

    if not _is_prefix(executed_mask):
        return _invalid(
            curr_obs,
            next_obs,
            nominal,
            sampled_residual,
            sampled_gripper,
            executed,
            bypass,
            reward_values,
            executed_mask,
            human_mask,
            handoff_mask,
            term,
            trunc,
            reward_label_valid,
            codec,
            source,
            policy_version,
            base_fingerprint,
            episode_id,
            chunk_id,
            STATUS_NON_CONTIGUOUS_EXECUTED_MASK,
        )
    k = int(executed_mask.sum())
    if k == 0:
        return _invalid(
            curr_obs,
            next_obs,
            nominal,
            sampled_residual,
            sampled_gripper,
            executed,
            bypass,
            reward_values,
            executed_mask,
            human_mask,
            handoff_mask,
            term,
            trunc,
            reward_label_valid,
            codec,
            source,
            policy_version,
            base_fingerprint,
            episode_id,
            chunk_id,
            STATUS_K_ZERO,
        )
    if not reward_label_valid:
        return _invalid(
            curr_obs,
            next_obs,
            nominal,
            sampled_residual,
            sampled_gripper,
            executed,
            bypass,
            reward_values,
            executed_mask,
            human_mask,
            handoff_mask,
            term,
            trunc,
            reward_label_valid,
            codec,
            source,
            policy_version,
            base_fingerprint,
            episode_id,
            chunk_id,
            STATUS_REWARD_LABEL_INVALID,
        )
    if not (
        np.isfinite(nominal).all()
        and np.isfinite(executed).all()
        and np.isfinite(sampled_residual).all()
        and np.isfinite(reward_values).all()
    ):
        return _invalid(
            curr_obs,
            next_obs,
            nominal,
            sampled_residual,
            sampled_gripper,
            executed,
            bypass,
            reward_values,
            executed_mask,
            human_mask,
            handoff_mask,
            term,
            trunc,
            reward_label_valid,
            codec,
            source,
            policy_version,
            base_fingerprint,
            episode_id,
            chunk_id,
            STATUS_NON_FINITE,
        )

    actions_arm = np.zeros((h, ARM_RESIDUAL_DIM), dtype=np.float32)
    actions_gripper = np.zeros((h, 3), dtype=np.float32)
    for i in range(k):
        u, out_of_support = codec.invert(nominal[i, :7], executed[i, :7])
        if out_of_support:
            return _invalid(
                curr_obs,
                next_obs,
                nominal,
                sampled_residual,
                sampled_gripper,
                executed,
                bypass,
                reward_values,
                executed_mask,
                human_mask,
                handoff_mask,
                term,
                trunc,
                reward_label_valid,
                codec,
                source,
                policy_version,
                base_fingerprint,
                episode_id,
                chunk_id,
                STATUS_OUT_OF_SUPPORT,
            )
        actions_arm[i] = u
        mode = invert_gripper_mode(
            executed[i][7],
            int(sampled_gripper[i]),
            human_intervened=bool(human_mask[i]),
        )
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[mode] = 1.0
        actions_gripper[i] = one_hot

    discounted_return = float(np.sum(reward_values[:k] * (gamma ** np.arange(k))))
    any_ending = bool(np.any(term[:k]) or np.any(trunc[:k]))
    bootstrap = np.asarray([not any_ending], dtype=bool)
    if bool(bootstrap[0]) and not np.any(np.abs(next_nominal) > 0):
        return _invalid(
            curr_obs,
            next_obs,
            nominal,
            sampled_residual,
            sampled_gripper,
            executed,
            bypass,
            reward_values,
            executed_mask,
            human_mask,
            handoff_mask,
            term,
            trunc,
            reward_label_valid,
            codec,
            source,
            policy_version,
            base_fingerprint,
            episode_id,
            chunk_id,
            STATUS_MISSING_NEXT_NOMINAL,
        )
    discount_steps = np.asarray([k], dtype=np.int64)

    transition = ResidualChunkTransition(
        curr_obs=curr_obs,
        next_obs=next_obs,
        nominal_actions=nominal,
        next_nominal_actions=next_nominal,
        sampled_arm_residual=sampled_residual,
        sampled_gripper_mode=sampled_gripper,
        commanded_actions=np.asarray(commanded_actions, dtype=np.float32),
        gripper_bypass_mask=bypass,
        executed_actions=executed,
        actions_arm=actions_arm,
        actions_gripper=actions_gripper,
        rewards=reward_values,
        executed_action_mask=executed_mask,
        human_intervention_mask=human_mask,
        handoff_hold_mask=handoff_mask,
        terminations=term,
        truncations=trunc,
        bootstrap_mask=bootstrap,
        discount_steps=discount_steps,
        discounted_return=discounted_return,
        transition_valid=np.asarray([True], dtype=bool),
        reward_label_valid=np.asarray([reward_label_valid], dtype=bool),
        source=source,
        policy_version=np.asarray([policy_version], dtype=np.int64),
        base_fingerprint=base_fingerprint,
        codec_fingerprint=codec.fingerprint(),
        episode_id=episode_id,
        chunk_id=chunk_id,
        status_reason=STATUS_OK,
    )
    return transition, True, STATUS_OK


def _invalid(
    curr_obs,
    next_obs,
    nominal,
    sampled_residual,
    sampled_gripper,
    executed,
    bypass,
    reward_values,
    executed_mask,
    human_mask,
    handoff_mask,
    term,
    trunc,
    reward_label_valid,
    codec,
    source,
    policy_version,
    base_fingerprint,
    episode_id,
    chunk_id,
    reason,
) -> tuple[ResidualChunkTransition, bool, str]:
    h = nominal.shape[0]
    return (
        ResidualChunkTransition(
            curr_obs=curr_obs,
            next_obs=next_obs,
            nominal_actions=nominal,
            next_nominal_actions=np.zeros((h, 8), dtype=np.float32),
            sampled_arm_residual=sampled_residual,
            sampled_gripper_mode=sampled_gripper,
            commanded_actions=np.asarray(executed, dtype=np.float32),
            gripper_bypass_mask=bypass,
            executed_actions=executed,
            actions_arm=np.zeros((h, ARM_RESIDUAL_DIM), dtype=np.float32),
            actions_gripper=np.zeros((h, 3), dtype=np.float32),
            rewards=reward_values,
            executed_action_mask=executed_mask,
            human_intervention_mask=human_mask,
            handoff_hold_mask=handoff_mask,
            terminations=term,
            truncations=trunc,
            bootstrap_mask=np.zeros(1, dtype=bool),
            discount_steps=np.zeros(1, dtype=np.int64),
            discounted_return=0.0,
            transition_valid=np.zeros(1, dtype=bool),
            reward_label_valid=np.asarray([reward_label_valid], dtype=bool),
            source=source,
            policy_version=np.asarray([policy_version], dtype=np.int64),
            base_fingerprint=base_fingerprint,
            codec_fingerprint=codec.fingerprint(),
            episode_id=episode_id,
            chunk_id=chunk_id,
            status_reason=reason,
        ),
        False,
        reason,
    )
