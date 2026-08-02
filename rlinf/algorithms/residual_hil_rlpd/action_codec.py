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

"""Reversible SE(3) residual action codec for the Dobot task.

Pose convention is ``[p_x, p_y, p_z, q_w, q_x, q_y, q_z]`` (wxyz). The residual
``u = [δp_norm(3), δr_norm(3)]`` lives in the nominal TCP/tool frame and is
expressed in normalized units ``[-1, 1]``; physical scales come from the YAML
data limits. ``compose_arm_residual`` (torch, rollout GPU) and
``invert_executed_action`` (numpy/scipy, env-worker CPU) implement the same
left-invariant semantics:

``R_exec = R_nom Exp(δr)``, ``p_exec = p_nom + R_nom δp``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

POSE_DIM = 7
ARM_RESIDUAL_DIM = 6

# Gripper correction modes.
KEEP_NOMINAL = 0
FORCE_CLOSE = 1
FORCE_OPEN = 2
GRIPPER_MODES = (KEEP_NOMINAL, FORCE_CLOSE, FORCE_OPEN)

_CODEC_SCHEMA_VERSION = "dobot_residual_v1"


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (torch)
# ---------------------------------------------------------------------------


def _quat_normalize_torch(q: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if torch.any(norm <= 1e-12):
        raise ValueError("Cannot normalize a zero-norm quaternion.")
    return q / norm


def _quat_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions on the last dim."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_rotate_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by wxyz quaternions ``q`` (last-dim aligned)."""
    w, x, y, z = q.unbind(-1)
    t = 2.0 * torch.linalg.cross(torch.stack([x, y, z], dim=-1), v, dim=-1)
    return (
        v
        + w.unsqueeze(-1) * t
        + torch.linalg.cross(torch.stack([x, y, z], dim=-1), t, dim=-1)
    )


def _exp_rotvec_torch(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector -> wxyz quaternion (batch over last dim)."""
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = angle / 2.0
    axis = torch.where(
        angle > 1e-12,
        rotvec / torch.clamp(angle, min=1e-12),
        torch.zeros_like(rotvec),
    )
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def _log_rotvec_torch(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion (w >= 0 hemisphere) -> axis-angle vector."""
    q = _quat_normalize_torch(q)
    w = q[..., 0:1]
    v = q[..., 1:]
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, torch.clamp(w, min=-1.0, max=1.0))
    axis = torch.where(
        v_norm > 1e-12,
        v / torch.clamp(v_norm, min=1e-12),
        torch.zeros_like(v),
    )
    return axis * angle


def _pose_to_quat_rot_torch(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a [..., 7] pose into (position, normalized wxyz quaternion)."""
    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"Expected pose dim {POSE_DIM}, got {pose.shape[-1]}")
    position = pose[..., :3]
    quat = _quat_normalize_torch(pose[..., 3:])
    return position, quat


def compose_arm_residual_torch(
    nominal_pose: torch.Tensor,
    u_norm: torch.Tensor,
    translation_scale_m: torch.Tensor,
    rotation_scale_rad: torch.Tensor,
) -> torch.Tensor:
    """Compose normalized residuals onto nominal poses (pure torch).

    Args:
        nominal_pose: ``[..., 7]`` absolute poses (wxyz).
        u_norm: ``[..., 6]`` normalized residuals in ``[-1, 1]``.
        translation_scale_m: ``[3]`` or broadcastable translation scales.
        rotation_scale_rad: ``[3]`` or broadcastable rotation scales.

    Returns:
        ``[..., 7]`` commanded absolute poses.
    """
    if u_norm.shape[-1] != ARM_RESIDUAL_DIM:
        raise ValueError(
            f"Expected residual dim {ARM_RESIDUAL_DIM}, got {u_norm.shape[-1]}"
        )
    if torch.any(u_norm.abs() > 1.0 + 1e-6):
        raise ValueError(
            "Normalized residuals must lie in [-1, 1]; refusing to silently "
            "clip out-of-support actions."
        )
    position, quat = _pose_to_quat_rot_torch(nominal_pose)
    delta_p = u_norm[..., :3] * translation_scale_m
    delta_r = u_norm[..., 3:] * rotation_scale_rad

    delta_quat = _exp_rotvec_torch(delta_r)
    exec_quat = _quat_mul_torch(quat, delta_quat)
    exec_quat = _quat_normalize_torch(exec_quat)
    # Keep the same hemisphere as the nominal quaternion.
    dot = (exec_quat * quat).sum(dim=-1, keepdim=True)
    exec_quat = torch.where(dot < 0.0, -exec_quat, exec_quat)

    exec_position = position + _quat_rotate_torch(quat, delta_p)
    return torch.cat([exec_position, exec_quat], dim=-1)


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (numpy/scipy)
# ---------------------------------------------------------------------------


def _quat_normalize_numpy(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-norm quaternion.")
    return q / norm


def invert_executed_action_numpy(
    nominal_pose: np.ndarray,
    executed_pose: np.ndarray,
    translation_scale_m: np.ndarray,
    rotation_scale_rad: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Invert a real executed pose into normalized residuals.

    Args:
        nominal_pose: ``[7]`` wxyz pose used for the chunk step.
        executed_pose: ``[7]`` wxyz pose actually sent to and accepted by the
            controller (not a post-hoc measurement).
        translation_scale_m: ``[3]`` data-limit translation scales.
        rotation_scale_rad: ``[3]`` data-limit rotation scales.

    Returns:
        ``(u_norm, out_of_support)``; ``out_of_support=True`` when any axis
        exceeds the data limit (never silently clipped).
    """
    nominal = np.asarray(nominal_pose, dtype=np.float64)
    executed = np.asarray(executed_pose, dtype=np.float64)
    if nominal.shape != (POSE_DIM,) or executed.shape != (POSE_DIM,):
        raise ValueError("invert_executed_action_numpy expects [7] poses")
    if not (np.isfinite(nominal).all() and np.isfinite(executed).all()):
        raise ValueError("Poses must be finite.")

    nom_quat = _quat_normalize_numpy(nominal[3:])
    exec_quat = _quat_normalize_numpy(executed[3:])
    if np.dot(nom_quat, exec_quat) < 0.0:
        exec_quat = -exec_quat

    nom_rot = Rotation.from_quat([nom_quat[1], nom_quat[2], nom_quat[3], nom_quat[0]])
    delta_pos = executed[:3] - nominal[:3]
    delta_p = nom_rot.inv().apply(delta_pos)

    relative_rot = nom_rot.inv() * Rotation.from_quat(
        [exec_quat[1], exec_quat[2], exec_quat[3], exec_quat[0]]
    )
    delta_r = relative_rot.as_rotvec()

    u = np.concatenate(
        [
            np.asarray(delta_p, dtype=np.float64)
            / np.asarray(translation_scale_m, dtype=np.float64),
            np.asarray(delta_r, dtype=np.float64)
            / np.asarray(rotation_scale_rad, dtype=np.float64),
        ]
    )
    out_of_support = bool(np.any(np.abs(u) > 1.0 + 1e-6))
    return u, out_of_support


# ---------------------------------------------------------------------------
# Gripper modes
# ---------------------------------------------------------------------------


def combine_gripper_mode(
    nominal_gripper: float,
    mode: int,
) -> tuple[float, bool]:
    """Map a gripper correction mode to (gripper command, bypass flag)."""
    if mode == KEEP_NOMINAL:
        return float(nominal_gripper), False
    if mode == FORCE_CLOSE:
        return 0.0, True
    if mode == FORCE_OPEN:
        return 1.0, True
    raise ValueError(f"Unknown gripper mode {mode!r}")


def invert_gripper_mode(
    executed_gripper: float,
    sampled_mode: int,
    human_intervened: bool,
) -> int:
    """Recover the training gripper mode from the real executed command."""
    if sampled_mode not in GRIPPER_MODES:
        raise ValueError(f"Unknown gripper mode {sampled_mode!r}")
    if human_intervened:
        # The executed binary gripper is the ground truth for human commands.
        return FORCE_CLOSE if float(executed_gripper) < 0.5 else FORCE_OPEN
    return int(sampled_mode)


def gripper_one_hot(mode: int) -> np.ndarray:
    if mode not in GRIPPER_MODES:
        raise ValueError(f"Unknown gripper mode {mode!r}")
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[mode] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Codec config + fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualCodec:
    """Physical scales and conventions for the residual action space."""

    translation_scale_m: Sequence[float] = (0.1, 0.1, 0.1)
    rotation_scale_deg: Sequence[float] = (30.0, 30.0, 30.0)
    quaternion_order: str = "wxyz"
    frame: str = "tool_frame_left_invariant"

    @property
    def translation_scale_np(self) -> np.ndarray:
        return np.asarray(self.translation_scale_m, dtype=np.float64)

    @property
    def rotation_scale_rad_np(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.rotation_scale_deg, dtype=np.float64))

    @property
    def translation_scale_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.translation_scale_np, dtype=torch.float32)

    @property
    def rotation_scale_rad_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.rotation_scale_rad_np, dtype=torch.float32)

    def compose(
        self,
        nominal_pose: torch.Tensor,
        u_norm: torch.Tensor,
    ) -> torch.Tensor:
        return compose_arm_residual_torch(
            nominal_pose,
            u_norm,
            self.translation_scale_torch,
            self.rotation_scale_rad_torch,
        )

    def invert(
        self,
        nominal_pose: np.ndarray,
        executed_pose: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        return invert_executed_action_numpy(
            nominal_pose,
            executed_pose,
            self.translation_scale_np,
            self.rotation_scale_rad_np,
        )

    def fingerprint(self) -> str:
        payload = {
            "schema": _CODEC_SCHEMA_VERSION,
            "translation_scale_m": [float(v) for v in self.translation_scale_m],
            "rotation_scale_deg": [float(v) for v in self.rotation_scale_deg],
            "quaternion_order": self.quaternion_order,
            "frame": self.frame,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
