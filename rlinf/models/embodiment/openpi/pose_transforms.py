# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SE(3) pose-delta transforms for the Dobot pose-mode policy.

These three transforms — :class:`DeltaPose`, :class:`AbsolutePose`, and
:class:`DeltaActions_Prev` — are **not** part of the upstream openpi package
installed in this environment. They were authored in our openpi fork
(``cnb.cool/THU-HiGroup/ELM/openpi``) for the Dobot CF5AF ``pose`` policy
(``pi05_dobot_t265_pose_train``) and are reproduced here verbatim so the RLinf
training/inference pipeline can run pose-mode without patching site-packages.

They implement **SE(3) relative-pose encoding** via 4×4 homogeneous matrices
(``T_delta = T_ref⁻¹ @ T_curr`` / ``T_curr = T_prev @ T_rel``), with quaternion
hemisphere-continuity handling. This is mathematically distinct from the
upstream ``DeltaActions`` (naive elementwise subtraction) and is required for
Cartesian pose actions — naive subtraction would be wrong for orientation and
for translation expressed in a rotated frame.

Conventions (must match the rest of the stack):
    - Pose layout: ``[x, y, z, qw, qx, qy, qz]`` (w-first quaternion).
    - The gripper dim (last) is **always** left absolute; only the first 7 dims
      (pose block) are converted to deltas (``mask = make_bool_mask(7, -1)``).

These classes satisfy the ``openpi.transforms.DataTransformFn`` Protocol
(structurally — no inheritance needed) and are composable with
``openpi.transforms.Group`` via ``.push(inputs=..., outputs=...)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

# DataTransformFn is a runtime_checkable Protocol in openpi; importing it for
# clarity (structural typing — we don't strictly need to inherit).
from openpi.transforms import DataTransformFn

# Type alias matching openpi's internal convention.
DataDict = dict


# ---------------------------------------------------------------------------
# Quaternion / homogeneous-matrix helpers (pure numpy)
# ---------------------------------------------------------------------------


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize the last axis of a quaternion array."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    return q / norm


def _pose_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    """Convert pose ``[x, y, z, qw, qx, qy, qz]`` to a 4×4 homogeneous matrix.

    Args:
        pose: Pose vector with shape ``(..., 7)``.

    Returns:
        4×4 homogeneous transformation matrix with shape ``(..., 4, 4)``.
    """
    pos = pose[..., :3]  # [x, y, z]
    quat = pose[..., 3:7]  # [qw, qx, qy, qz]

    # Normalize quaternion
    quat = _quat_normalize(quat)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Build rotation matrix (3x3)
    r = np.zeros(quat.shape[:-1] + (3, 3), dtype=pose.dtype)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)

    # Build homogeneous transformation matrix
    t = np.zeros(quat.shape[:-1] + (4, 4), dtype=pose.dtype)
    t[..., :3, :3] = r
    t[..., :3, 3] = pos
    t[..., 3, 3] = 1.0

    return t


def _homogeneous_to_pose(
    t: np.ndarray, reference_quat: np.ndarray | None = None
) -> np.ndarray:
    """Convert a 4×4 homogeneous matrix to pose ``[x, y, z, qw, qx, qy, qz]``.

    Args:
        t: 4×4 homogeneous transformation matrix with shape ``(..., 4, 4)``.
        reference_quat: Optional reference quaternion ``[qw, qx, qy, qz]`` for
            continuity. If the dot product with the extracted quaternion is
            negative, the extracted quaternion is negated (same hemisphere).
            Shape should be ``(..., 4)``.

    Returns:
        Pose vector with shape ``(..., 7)``.
    """
    pos = t[..., :3, 3]  # Extract position
    r = t[..., :3, :3]  # Extract rotation matrix

    # Convert rotation matrix to quaternion
    trace = np.trace(r, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(1 + trace, 0)) / 2
    x = np.copysign(
        np.sqrt(np.maximum(1 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2], 0)) / 2,
        r[..., 2, 1] - r[..., 1, 2],
    )
    y = np.copysign(
        np.sqrt(np.maximum(1 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2], 0)) / 2,
        r[..., 0, 2] - r[..., 2, 0],
    )
    z = np.copysign(
        np.sqrt(np.maximum(1 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2], 0)) / 2,
        r[..., 1, 0] - r[..., 0, 1],
    )

    quat = np.stack([w, x, y, z], axis=-1)
    quat = _quat_normalize(quat)

    # Ensure quaternion continuity with reference if provided
    if reference_quat is not None:
        # If dot product is negative, flip quaternion to same hemisphere
        dot_product = np.sum(quat * reference_quat, axis=-1, keepdims=True)
        quat = np.where(dot_product < 0, -quat, quat)

    return np.concatenate([pos, quat], axis=-1)


def _invert_homogeneous(t: np.ndarray) -> np.ndarray:
    """Compute the inverse of a homogeneous transformation matrix.

    For ``T = [R  t]`` the inverse is ``T^-1 = [R^T  -R^T @ t]``.
                                   ``[0  1]``                 ``[0       1   ]``

    Args:
        t: 4×4 homogeneous transformation matrix with shape ``(..., 4, 4)``.

    Returns:
        Inverse matrix with shape ``(..., 4, 4)``.
    """
    r = t[..., :3, :3]
    trans = t[..., :3, 3]

    r_t = np.transpose(r, axes=[*range(len(r.shape) - 2), -1, -2])
    t_inv = -np.einsum("...ij,...j->...i", r_t, trans)

    t_out = np.zeros_like(t)
    t_out[..., :3, :3] = r_t
    t_out[..., :3, 3] = t_inv
    t_out[..., 3, 3] = 1.0

    return t_out


# ---------------------------------------------------------------------------
# Block-level pose transforms (operating on a [pos_slice, quat_slice] block)
# ---------------------------------------------------------------------------


def _apply_delta_pose_block(
    output: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray,
    pos_slice: slice,
    quat_slice: slice,
    handled: np.ndarray,
    expand_reference: bool = False,
    reference_quat: np.ndarray | None = None,
) -> None:
    """Apply delta pose transformation: ``T_delta = T_ref⁻¹ @ T_curr``.

    This correctly handles position transformation in the rotated reference
    frame (i.e. the delta is expressed in the reference frame, not the world
    frame).

    Args:
        output: Output array to write delta values to.
        current: Current pose values ``[x, y, z, qw, qx, qy, qz]``.
        reference: Reference pose to subtract from (``prev_state`` for state,
            ``state`` for actions).
        pos_slice: Slice for position dimensions (e.g., ``slice(0, 3)``).
        quat_slice: Slice for quaternion dimensions (e.g., ``slice(3, 7)``).
        handled: Boolean array marking which dimensions have been handled.
        expand_reference: If True, expand reference dims for broadcasting with
            action sequences.
        reference_quat: Optional reference quaternion for ensuring continuity
            in delta quaternion. If provided, delta quaternion will be flipped
            if dot product is negative.
    """
    # Extract pose components
    curr_pose = np.concatenate(
        [current[..., pos_slice], current[..., quat_slice]], axis=-1
    )

    if expand_reference:
        ref_pose = np.concatenate(
            [
                np.expand_dims(reference[..., pos_slice], axis=-2),
                np.expand_dims(reference[..., quat_slice], axis=-2),
            ],
            axis=-1,
        )
    else:
        ref_pose = np.concatenate(
            [reference[..., pos_slice], reference[..., quat_slice]], axis=-1
        )

    # Convert to homogeneous transformation matrices
    t_curr = _pose_to_homogeneous(curr_pose)
    t_ref = _pose_to_homogeneous(ref_pose)

    # Compute relative transform in local frame: T_delta = T_ref^-1 @ T_curr
    t_ref_inv = _invert_homogeneous(t_ref)
    t_delta = np.einsum("...ij,...jk->...ik", t_ref_inv, t_curr)

    # Convert back to pose representation with quaternion continuity checking
    delta_pose = _homogeneous_to_pose(t_delta, reference_quat=reference_quat)

    # Write output
    output[..., pos_slice] = delta_pose[..., :3]
    output[..., quat_slice] = delta_pose[..., 3:7]

    handled[pos_slice] = True
    handled[quat_slice] = True


def _apply_absolute_pose_block(
    output: np.ndarray,
    prev: np.ndarray,
    rel: np.ndarray,
    pos_slice: slice,
    quat_slice: slice,
    handled: np.ndarray,
    expand_state: bool = False,
    reference_quat: np.ndarray | None = None,
) -> None:
    """Apply absolute pose transformation: ``T_curr = T_prev @ T_rel``.

    This correctly handles position transformation in the reference frame.

    Args:
        output: Output array to write absolute values to.
        prev: Previous absolute pose values ``[x, y, z, qw, qx, qy, qz]``.
        rel: Relative pose values to add (delta pose).
        pos_slice: Slice for position dimensions (e.g., ``slice(0, 3)``).
        quat_slice: Slice for quaternion dimensions (e.g., ``slice(3, 7)``).
        handled: Boolean array marking which dimensions have been handled.
        expand_state: If True, expand state dims for broadcasting with action
            sequences.
        reference_quat: Optional reference quaternion for ensuring continuity
            in resulting quaternion.
    """
    # Extract pose components
    prev_pose = np.concatenate([prev[..., pos_slice], prev[..., quat_slice]], axis=-1)

    rel_pose = np.concatenate([rel[..., pos_slice], rel[..., quat_slice]], axis=-1)

    # Convert to homogeneous transformation matrices
    t_prev = _pose_to_homogeneous(prev_pose)
    t_rel = _pose_to_homogeneous(rel_pose)

    # Expand state dimension for broadcasting with action sequence if needed
    if expand_state:
        t_prev = np.expand_dims(t_prev, axis=-3)

    # Compute absolute pose: T_curr = T_prev @ T_rel
    t_curr = np.einsum("...ij,...jk->...ik", t_prev, t_rel)

    # Convert back to pose representation with quaternion continuity checking
    curr_pose = _homogeneous_to_pose(t_curr, reference_quat=reference_quat)

    # Write output
    output[..., pos_slice] = curr_pose[..., :3]
    output[..., quat_slice] = curr_pose[..., 3:7]

    handled[pos_slice] = True
    handled[quat_slice] = True


# ---------------------------------------------------------------------------
# Top-level transforms (satisfy openpi.transforms.DataTransformFn Protocol)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeltaPose(DataTransformFn):
    """Repacks absolute pose state/actions into delta (relative) pose space.

    Uses ``prev_state`` (previous frame's absolute pose) as the reference:
    - ``state`` delta: ``T_delta = T_prev_state⁻¹ @ T_state``
    - ``actions`` delta: ``T_delta = T_state⁻¹ @ T_actions`` (with broadcasting
      over the action horizon)
    - ``rtc_obs.prev_action`` delta: same as actions (for RTC chunk broker).

    No-op if ``prev_state`` is absent or ``mask`` is None — this is the safety
    escape hatch (sending relative poses directly to the robot without a
    reference would be non-physical).

    The gripper dim (last, masked False) is left absolute.
    """

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "prev_state" not in data or self.mask is None:
            return data

        prev_state, state = data["prev_state"], data["state"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], state.shape[-1])
        delta = state.copy()
        handled = np.zeros(dims, dtype=bool)

        # Process left and right arm pose blocks with quaternion continuity.
        # Single-arm Dobot only hits the first block (dims >= 7, mask[:7].all()).
        if dims >= 7 and mask[:7].all():
            _apply_delta_pose_block(
                delta,
                state,
                prev_state,
                slice(0, 3),
                slice(3, 7),
                handled,
                reference_quat=state[..., 3:7],
            )
        if dims >= 15 and mask[8:15].all():
            _apply_delta_pose_block(
                delta,
                state,
                prev_state,
                slice(8, 11),
                slice(11, 15),
                handled,
                reference_quat=state[..., 11:15],
            )

        data["state"] = delta

        # Convert actions into relative pose w.r.t current state
        if "actions" in data:
            actions = data["actions"]
            adims = min(dims, actions.shape[-1])
            a_delta = actions.copy()
            a_handled = np.zeros(adims, dtype=bool)

            if adims >= 7 and mask[:7].all():
                _apply_delta_pose_block(
                    a_delta,
                    actions,
                    state,
                    slice(0, 3),
                    slice(3, 7),
                    a_handled,
                    expand_reference=True,
                    reference_quat=actions[..., 3:7],
                )
            if adims >= 15 and mask[8:15].all():
                _apply_delta_pose_block(
                    a_delta,
                    actions,
                    state,
                    slice(8, 11),
                    slice(11, 15),
                    a_handled,
                    expand_reference=True,
                    reference_quat=actions[..., 11:15],
                )

            data["actions"] = a_delta

        # Convert prev_action into relative pose w.r.t current state (RTC)
        if "rtc_obs" in data and "prev_action" in data["rtc_obs"]:
            prev_actions = data["rtc_obs"]["prev_action"].copy()
            prev_adims = min(dims, prev_actions.shape[-1])
            prev_a_delta = prev_actions.copy()
            prev_a_handled = np.zeros(prev_adims, dtype=bool)

            if prev_adims >= 7 and mask[:7].all():
                _apply_delta_pose_block(
                    prev_a_delta,
                    prev_actions,
                    state,
                    slice(0, 3),
                    slice(3, 7),
                    prev_a_handled,
                    expand_reference=True,
                    reference_quat=prev_actions[..., 3:7],
                )
            if prev_adims >= 15 and mask[8:15].all():
                _apply_delta_pose_block(
                    prev_a_delta,
                    prev_actions,
                    state,
                    slice(8, 11),
                    slice(11, 15),
                    prev_a_handled,
                    expand_reference=True,
                    reference_quat=prev_actions[..., 11:15],
                )

            data["rtc_obs"]["prev_action"] = prev_a_delta

        return data


@dataclasses.dataclass(frozen=True)
class AbsolutePose(DataTransformFn):
    """Reconstruct absolute pose state/actions from relative pose.

    Uses ``prev_state`` absolute pose and ``state`` relative pose (in local
    frame) to compute current absolute ``state`` via
    ``T_curr = T_prev @ T_delta``, then converts relative ``actions`` w.r.t.
    state into absolute ``actions``.

    No-op if ``prev_state`` is absent or ``mask`` is None.
    """

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "prev_state" not in data or self.mask is None:
            return data

        prev_state, rel_state = data["prev_state"], data["state"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], rel_state.shape[-1])

        abs_state = rel_state.copy()
        handled = np.zeros(dims, dtype=bool)

        # Process left and right arm pose blocks with quaternion continuity.
        if dims >= 7 and mask[:7].all():
            _apply_absolute_pose_block(
                abs_state,
                prev_state,
                rel_state,
                slice(0, 3),
                slice(3, 7),
                handled,
                reference_quat=prev_state[..., 3:7],
            )
        if dims >= 15 and mask[8:15].all():
            _apply_absolute_pose_block(
                abs_state,
                prev_state,
                rel_state,
                slice(8, 11),
                slice(11, 15),
                handled,
                reference_quat=prev_state[..., 11:15],
            )

        data["state"] = abs_state

        # Convert relative actions (w.r.t current state) into absolute actions.
        if "actions" in data:
            actions = data["actions"]
            action_dims = actions.shape[-1]
            adims = min(dims, action_dims)
            abs_actions = actions.copy()
            a_handled = np.zeros(adims, dtype=bool)

            # Left arm pose block
            if adims >= 7 and mask[:7].all():
                _apply_absolute_pose_block(
                    abs_actions,
                    abs_state,
                    actions,
                    slice(0, 3),
                    slice(3, 7),
                    a_handled,
                    expand_state=True,
                    reference_quat=abs_state[..., 3:7],
                )

            # Right arm pose block
            if adims >= 15 and mask[8:15].all():
                _apply_absolute_pose_block(
                    abs_actions,
                    abs_state,
                    actions,
                    slice(8, 11),
                    slice(11, 15),
                    a_handled,
                    expand_state=True,
                    reference_quat=abs_state[..., 11:15],
                )

            data["actions"] = abs_actions

        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions_Prev(DataTransformFn):
    """Convert ``rtc_obs.prev_action`` to delta action space (RTC inference).

    Subtracts ``state`` from ``rtc_obs.prev_action`` on the masked dims, in
    place. This mirrors :class:`openpi.transforms.DeltaActions` but operates
    on the previous-action chunk carried by the RTC action-chunk broker
    instead of the current ``actions`` field.

    No-op if ``rtc_obs`` is absent or ``mask`` is None.
    """

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "rtc_obs" not in data or self.mask is None:
            return data

        state = data["state"].copy()
        actions = data["rtc_obs"]["prev_action"].copy()
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data["rtc_obs"]["prev_action"] = actions

        return data
