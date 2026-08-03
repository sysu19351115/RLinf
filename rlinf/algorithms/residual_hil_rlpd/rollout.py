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

"""Composition of the frozen base policy with the residual policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.action_codec import (
    FORCE_OPEN,
    KEEP_NOMINAL,
    ResidualCodec,
    combine_gripper_mode,
)
from rlinf.models.embodiment.residual_dobot_policy import ResidualDobotActor


def preprocess_image(images: torch.Tensor) -> torch.Tensor:
    """Single shared camera-frame preprocessor (BHWC uint8 -> BCHW float [0,1]).

    - 3-D input (single HWC frame) gains a batch dim.
    - uint8 frames are normalized to ``[0, 1]``; float input is passed through
      unchanged (assumed already normalized).
    - BHWC (``[..., H, W, C]`` with C == 3) is permuted to BCHW.

    Raises:
        ValueError: When the resulting channel dimension is not 3.
    """
    tensor = images
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            "preprocess_image expects a 3-channel image after conversion, "
            f"got shape={tuple(tensor.shape)}"
        )
    return tensor.contiguous()


@dataclass
class RolloutChunkAudit:
    """Everything the finalizer needs to reconstruct a real transition."""

    nominal_actions: np.ndarray  # [H, 8] absolute, OpenPI-de-normalized
    sampled_arm_residual: np.ndarray  # [H, 6] normalized model samples
    sampled_gripper_mode: np.ndarray  # [H]
    commanded_actions: np.ndarray  # [H, 8]
    gripper_bypass_mask: np.ndarray  # [H]
    policy_version: int


class ResidualRolloutPolicy:
    """Frozen base + residual actor composition for one chunk.

    The base policy callable must return normalized absolute actions
    ``[H, 8]`` (pose wxyz + gripper). The residual actor outputs normalized
    corrections; the codec composes them into commanded absolute actions.
    """

    def __init__(
        self,
        base_policy: Callable[[torch.Tensor], torch.Tensor],
        residual_actor: ResidualDobotActor,
        codec: ResidualCodec,
        device: str = "cpu",
        policy_translation_scale_m: Sequence[float] | None = None,
        policy_rotation_scale_deg: Sequence[float] | None = None,
    ):
        self.base_policy = base_policy
        self.residual_actor = residual_actor.to(device).eval()
        self.codec = codec
        self.device = device
        self.policy_translation_scale = np.asarray(
            policy_translation_scale_m
            if policy_translation_scale_m is not None
            else codec.translation_scale_m,
            dtype=np.float32,
        )
        self.policy_rotation_scale = np.deg2rad(
            np.asarray(
                policy_rotation_scale_deg
                if policy_rotation_scale_deg is not None
                else codec.rotation_scale_deg,
                dtype=np.float32,
            )
        )

    @torch.no_grad()
    def rollout_chunk(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        residual_scale: float = 1.0,
        deterministic: bool = False,
        policy_version: int = 0,
    ) -> tuple[np.ndarray, RolloutChunkAudit]:
        """Generate one commanded chunk.

        Args:
            images: ``[1, C, H, W]`` camera frame.
            proprio: ``[1, 8]`` pose + gripper state.
            residual_scale: Ramp multiplier applied to sampled residuals.
            deterministic: Use the actor mean (evaluation / Gate D).
            policy_version: Learner policy version for audit.

        Returns:
            ``(commanded_actions, audit)``.
        """
        base_nominal = self.base_policy(images)
        nominal = base_nominal.detach().float().cpu().numpy().reshape(-1, 8)
        images = images.to(self.device)
        proprio = proprio.to(self.device)
        nominal_t = torch.as_tensor(nominal, dtype=torch.float32, device=self.device)

        u_norm, gripper_mode, _, _ = self.residual_actor.sample(
            images,
            proprio,
            nominal_t.unsqueeze(0),
            deterministic=deterministic,
        )
        u_norm = (u_norm * residual_scale).squeeze(0).cpu().numpy()
        gripper_mode = gripper_mode.squeeze(0).cpu().numpy()
        scale_ratio = np.concatenate(
            [
                self.policy_translation_scale
                / np.asarray(self.codec.translation_scale_np, dtype=np.float32),
                self.policy_rotation_scale
                / np.asarray(self.codec.rotation_scale_rad_np, dtype=np.float32),
            ]
        )
        if residual_scale > 0.0:
            u_norm = u_norm * scale_ratio
        else:
            u_norm = np.zeros_like(u_norm)
            gripper_mode = np.zeros_like(gripper_mode)

        commanded = np.zeros_like(nominal)
        bypass = np.zeros(nominal.shape[0], dtype=bool)
        for i in range(nominal.shape[0]):
            composed_pose = self.codec.compose(
                torch.as_tensor(nominal[i, :7], dtype=torch.float32),
                torch.as_tensor(u_norm[i], dtype=torch.float32),
            ).numpy()
            if residual_scale > 0.0:
                trans_delta = np.abs(composed_pose[:3] - nominal[i, :3])
                rot_delta = np.abs(u_norm[i, 3:]) * np.asarray(
                    self.codec.rotation_scale_rad_np, dtype=np.float32
                )
                if np.any(
                    trans_delta > self.policy_translation_scale * 1.001
                ) or np.any(rot_delta > self.policy_rotation_scale * 1.001):
                    raise RuntimeError(
                        "Residual commanded action exceeds policy limits"
                    )
            gripper, bypass_i = combine_gripper_mode(
                nominal[i, 7], int(gripper_mode[i])
            )
            commanded[i] = np.concatenate([composed_pose, [gripper]])
            bypass[i] = bypass_i

        audit = RolloutChunkAudit(
            nominal_actions=nominal.astype(np.float32),
            sampled_arm_residual=u_norm.astype(np.float32),
            sampled_gripper_mode=gripper_mode.astype(np.int64),
            commanded_actions=commanded.astype(np.float32),
            gripper_bypass_mask=bypass,
            policy_version=int(policy_version),
        )
        return commanded.astype(np.float32), audit


class Pi05ResidualComposer:
    """Compose frozen-Pi0.5 outputs with the residual actor on the rollout node.

    The HF rollout worker already produces nominal absolute actions by calling
    the frozen Pi0.5 (``predict_action_batch``); this composer turns them into
    commanded absolute actions plus the audit dict the env worker needs.
    """

    def __init__(
        self,
        residual_actor: ResidualDobotActor,
        codec: ResidualCodec,
        device: str = "cuda",
        policy_translation_scale_m: Sequence[float] | None = None,
        policy_rotation_scale_deg: Sequence[float] | None = None,
        gripper_max_switches_per_chunk: int = 2,
        gripper_min_hold_steps: int = 5,
        gripper_debounce_chunks: int = 2,
        gripper_allow_force_open: bool = True,
    ):
        self.residual_actor = residual_actor.to(device).eval()
        self.codec = codec
        self.device = device
        self.policy_translation_scale = np.asarray(
            policy_translation_scale_m
            if policy_translation_scale_m is not None
            else codec.translation_scale_m,
            dtype=np.float32,
        )
        self.policy_rotation_scale = np.deg2rad(
            np.asarray(
                policy_rotation_scale_deg
                if policy_rotation_scale_deg is not None
                else codec.rotation_scale_deg,
                dtype=np.float32,
            )
        )
        self.gripper_max_switches_per_chunk = int(gripper_max_switches_per_chunk)
        self.gripper_min_hold_steps = int(gripper_min_hold_steps)
        self.gripper_debounce_chunks = int(gripper_debounce_chunks)
        # Safety knob: while the discrete gripper policy is still learning, it
        # may be restricted to KEEP / FORCE_CLOSE so it can never drop an
        # object mid-task with an untrained FORCE_OPEN.
        self.gripper_allow_force_open = bool(gripper_allow_force_open)
        # Model-side gripper rate limiting state (P2-4).
        self._last_effective_mode = int(0)  # KEEP_NOMINAL
        self._gripper_hold_remaining = 0
        self._gripper_debounce_remaining = 0
        self._switch_count_in_chunk = 0

    @torch.no_grad()
    def compose_chunk(
        self,
        nominal_actions: torch.Tensor,
        env_obs: dict[str, torch.Tensor],
        *,
        residual_scale: float = 1.0,
        gripper_enabled: bool = False,
        deterministic: bool = False,
        policy_version: int = 0,
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Return ``(commanded_actions, audit_dict)`` for a ``[B, H, 8]`` batch."""
        nominal = nominal_actions.detach().float().cpu().numpy()
        images = preprocess_image(env_obs["main_images"]).to(self.device)
        proprio = env_obs["prev_states"].to(self.device)
        nominal_t = torch.as_tensor(nominal, dtype=torch.float32, device=self.device)
        u_norm, gripper_mode, _, _ = self.residual_actor.sample(
            images,
            proprio,
            nominal_t,
            deterministic=deterministic,
        )
        u_norm_np = u_norm.cpu().numpy()
        raw_gripper_np = gripper_mode.cpu().numpy()
        effective_gripper_np = raw_gripper_np.copy()
        scale_ratio = np.concatenate(
            [
                self.policy_translation_scale
                / np.asarray(self.codec.translation_scale_np, dtype=np.float32),
                self.policy_rotation_scale
                / np.asarray(self.codec.rotation_scale_rad_np, dtype=np.float32),
            ]
        )
        batch = nominal.shape[0]
        commanded = np.zeros_like(nominal)
        bypass = np.zeros(nominal.shape[:2], dtype=bool)
        self._switch_count_in_chunk = 0
        # Cooldown is measured in whole chunks (P2-4).
        if self._gripper_debounce_remaining > 0:
            self._gripper_debounce_remaining -= 1
        for b in range(batch):
            for i in range(nominal.shape[1]):
                if residual_scale <= 0.0:
                    # Base-only phase: keep nominal entirely.
                    u_codec = np.zeros(6, dtype=np.float32)
                    gripper_mode_i = int(0)  # KEEP_NOMINAL
                    effective_gripper_np[b, i] = gripper_mode_i
                else:
                    # codec expects normalized units scaled by DATA limit:
                    # u_codec = u_norm * (policy/data) * ramp.
                    u_codec = u_norm_np[b, i] * scale_ratio * residual_scale
                    candidate_mode = int(raw_gripper_np[b, i])
                    if not gripper_enabled:
                        # Discrete gripper has no magnitude: either it is
                        # enabled with rate limits, or it stays KEEP.
                        gripper_mode_i = int(0)
                    else:
                        if (
                            not self.gripper_allow_force_open
                            and candidate_mode == FORCE_OPEN
                        ):
                            # FORCE_OPEN disabled: fall back to keeping the
                            # nominal (frozen VLA) gripper command.
                            candidate_mode = KEEP_NOMINAL
                        gripper_mode_i = self._apply_gripper_rate_limit(
                            candidate_mode
                        )
                    effective_gripper_np[b, i] = gripper_mode_i
                composed_pose = self.codec.compose(
                    torch.as_tensor(nominal[b, i, :7], dtype=torch.float32),
                    torch.as_tensor(u_codec, dtype=torch.float32),
                ).numpy()
                if residual_scale > 0.0:
                    trans_delta = np.abs(composed_pose[:3] - nominal[b, i, :3])
                    rot_delta = np.abs(u_codec[3:]) * np.asarray(
                        self.codec.rotation_scale_rad_np, dtype=np.float32
                    )
                    if np.any(trans_delta > self.policy_translation_scale * 1.001):
                        raise RuntimeError(
                            "Residual commanded translation exceeds policy "
                            f"limits: delta={trans_delta}, "
                            f"limits={self.policy_translation_scale}"
                        )
                    if np.any(rot_delta > self.policy_rotation_scale * 1.001):
                        raise RuntimeError(
                            "Residual commanded rotation exceeds policy "
                            f"limits: delta={rot_delta}, "
                            f"limits={self.policy_rotation_scale}"
                        )
                gripper, bypass_i = combine_gripper_mode(
                    float(nominal[b, i, 7]), gripper_mode_i
                )
                commanded[b, i] = np.concatenate([composed_pose, [gripper]])
                bypass[b, i] = bypass_i
        single_env = batch == 1
        nominal_out = nominal[0] if single_env else nominal
        u_out = u_norm_np[0] if single_env else u_norm_np
        gripper_out = effective_gripper_np[0] if single_env else effective_gripper_np
        raw_gripper_out = raw_gripper_np[0] if single_env else raw_gripper_np
        commanded_out = commanded[0] if single_env else commanded
        bypass_out = bypass[0] if single_env else bypass
        audit = {
            "nominal_actions": np.asarray(nominal_out, dtype=np.float32),
            "sampled_arm_residual": np.asarray(u_out, dtype=np.float32),
            "sampled_gripper_mode": np.asarray(gripper_out, dtype=np.int64),
            "raw_sampled_gripper_mode": np.asarray(raw_gripper_out, dtype=np.int64),
            "commanded_actions": np.asarray(commanded_out, dtype=np.float32),
            "gripper_bypass_mask": np.asarray(bypass_out, dtype=bool),
            "policy_version": int(policy_version),
        }
        # Keep the [B, H, 8] batch dim for the RolloutResult transport
        # contract (``_split_rollout_result`` splits along the batch dim);
        # the audit dict intentionally stays per-env [H, ...] for the
        # env-worker finalizer.
        return (
            torch.as_tensor(
                commanded,
                dtype=nominal_actions.dtype,
                device=nominal_actions.device,
            ),
            audit,
        )

    def _apply_gripper_rate_limit(self, candidate_mode: int) -> int:
        """Min-hold / debounce / max-switch filtering for model gripper modes.

        State persists across chunks: a freshly switched mode is held for
        ``gripper_min_hold_steps`` steps; another switch is blocked for
        ``gripper_debounce_chunks`` chunks; per-chunk switch count is capped.
        """
        if candidate_mode == self._last_effective_mode:
            if self._gripper_hold_remaining > 0:
                self._gripper_hold_remaining -= 1
            return candidate_mode
        # Candidate differs from the last executed mode.
        if self._gripper_hold_remaining > 0:
            self._gripper_hold_remaining -= 1
            return self._last_effective_mode
        if self._gripper_debounce_remaining > 0:
            return self._last_effective_mode
        if self._switch_count_in_chunk >= self.gripper_max_switches_per_chunk:
            return self._last_effective_mode
        # Accept the switch.
        self._switch_count_in_chunk += 1
        self._last_effective_mode = candidate_mode
        self._gripper_hold_remaining = self.gripper_min_hold_steps
        self._gripper_debounce_remaining = self.gripper_debounce_chunks
        return candidate_mode
