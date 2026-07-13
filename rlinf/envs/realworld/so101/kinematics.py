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

"""SO101 single-arm kinematics using pinocchio and the official URDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

# Motor names in the SO101 URDF and in the 12-dim RLinf action vector.
_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Indices of arm joints within a 6-dim single-arm vector.
_ARM_INDICES = np.arange(5, dtype=np.int64)
_GRIPPER_INDEX = 5


def _default_urdf_path() -> Path:
    return Path(__file__).resolve().parent / "urdf" / "so101_new_calib.urdf"


class SO101ArmKinematics:
    """Forward / inverse kinematics for one SO101 arm.

    The public interface uses RLinf units:

    * arm joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
      wrist_roll) in ``[-100, 100]`` with ``0`` at mid-range;
    * gripper in ``[0, 100]`` where ``0`` is closed and ``100`` is open.

    Internally the URDF uses radians; the new-calibration URDF defines the
    virtual zero of each joint as the midpoint of its limits.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        end_effector_link: str = "gripper_frame_link",
    ):
        urdf_path = Path(urdf_path) if urdf_path is not None else _default_urdf_path()
        if not urdf_path.exists():
            raise FileNotFoundError(f"SO101 URDF not found: {urdf_path}")

        try:
            import pinocchio
        except ImportError as exc:
            raise ImportError(
                "SO101ArmKinematics requires pinocchio. "
                "Install it with: uv pip install pin"
            ) from exc

        self._pinocchio = pinocchio
        self.urdf_path = urdf_path
        self.end_effector_link = end_effector_link

        self.model, self.collision_model, self.visual_model = (
            pinocchio.buildModelsFromUrdf(str(urdf_path))
        )
        self.data = self.model.createData()

        self.frame_id = self.model.getFrameId(end_effector_link)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(
                f"End-effector link '{end_effector_link}' not found in URDF"
            )

        # Build joint-name -> q-index mapping and collect limits.
        self._joint_ids: list[int] = []
        self._q_indices: list[int] = []
        self._lower: np.ndarray = np.empty(6, dtype=np.float64)
        self._upper: np.ndarray = np.empty(6, dtype=np.float64)
        self._ranges: np.ndarray = np.empty(6, dtype=np.float64)

        for i, name in enumerate(_ARM_JOINT_NAMES):
            jid = self.model.getJointId(name)
            if jid >= len(self.model.joints):
                raise ValueError(f"Joint '{name}' not found in URDF")
            self._joint_ids.append(jid)
            # For 1-DoF joints, idx_qs gives the index in the configuration vector.
            q_idx = self.model.idx_qs[jid]
            self._q_indices.append(q_idx)
            self._lower[i] = self.model.lowerPositionLimit[q_idx]
            self._upper[i] = self.model.upperPositionLimit[q_idx]

        self._ranges = self._upper - self._lower
        # Guard against zero-range joints.
        self._ranges = np.where(self._ranges < 1e-12, 1e-12, self._ranges)

        self._logger = get_logger()
        self._logger.info(
            f"SO101ArmKinematics loaded from {urdf_path} "
            f"with EE frame '{end_effector_link}'"
        )

    # ── Unit conversions ─────────────────────────────────────────────────────

    def rlinf_to_urdf(self, q_rlinf: np.ndarray) -> np.ndarray:
        """Convert a 6-dim RLinf joint vector to URDF radians."""
        q = np.asarray(q_rlinf, dtype=np.float64).reshape(6).copy()
        q_urdf = np.empty(6, dtype=np.float64)

        # Arm joints: RLinf [-100, 100] -> LeRobot [0, 100] -> URDF radians.
        for k, idx in enumerate(_ARM_INDICES):
            lerobot = q[idx] / 2.0 + 50.0
            q_urdf[idx] = self._lower[k] + (lerobot / 100.0) * self._ranges[k]

        # Gripper: RLinf [0, 100] -> URDF radians.
        q_urdf[_GRIPPER_INDEX] = (
            self._lower[_GRIPPER_INDEX]
            + (q[_GRIPPER_INDEX] / 100.0) * self._ranges[_GRIPPER_INDEX]
        )
        return q_urdf

    def urdf_to_rlinf(self, q_urdf: np.ndarray) -> np.ndarray:
        """Convert a 6-dim URDF radian vector to RLinf units."""
        q = np.asarray(q_urdf, dtype=np.float64).reshape(6).copy()
        q_rlinf = np.empty(6, dtype=np.float64)

        for k, idx in enumerate(_ARM_INDICES):
            lerobot = (q[idx] - self._lower[k]) / self._ranges[k] * 100.0
            q_rlinf[idx] = (lerobot - 50.0) * 2.0

        q_rlinf[_GRIPPER_INDEX] = (
            (q[_GRIPPER_INDEX] - self._lower[_GRIPPER_INDEX])
            / self._ranges[_GRIPPER_INDEX]
            * 100.0
        )
        return q_rlinf

    def clip_urdf(self, q_urdf: np.ndarray) -> np.ndarray:
        """Clamp a URDF configuration to joint limits."""
        return np.clip(q_urdf, self._lower, self._upper)

    # ── Forward kinematics ───────────────────────────────────────────────────

    def forward(
        self, q_rlinf: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return end-effector position and quaternion (xyzw) in URDF base frame."""
        q_urdf = self.rlinf_to_urdf(q_rlinf)
        q_full = self._to_full_config(q_urdf)

        self._pinocchio.framesForwardKinematics(self.model, self.data, q_full)
        oMf = self.data.oMf[self.frame_id]

        position = np.asarray(oMf.translation, dtype=np.float64).copy()
        quat = self._pinocchio.Quaternion(oMf.rotation)
        quaternion = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)
        return position, quaternion

    # ── Inverse kinematics ───────────────────────────────────────────────────

    def inverse(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init_rlinf: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-5,
        damping: float = 1e-3,
        dt: float = 0.5,
    ) -> np.ndarray:
        """Solve IK for a target pose and return a RLinf 6-dim joint vector.

        Args:
            target_position: Target EE position (3,).
            target_quaternion: Target EE orientation as quaternion ``[x, y, z, w]``.
            q_init_rlinf: Initial guess in RLinf units.
            max_iter: Maximum Gauss-Newton iterations.
            tol: Convergence tolerance on the pose error norm.
            damping: Damping factor for the Jacobian pseudo-inverse.
            dt: Integration step size (reduces overshoot).

        Returns:
            6-dim RLinf joint configuration.
        """
        target_position = np.asarray(target_position, dtype=np.float64)
        target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        q_urdf = self.rlinf_to_urdf(q_init_rlinf)

        oMdes = self._pinocchio.SE3(
            self._quat_to_matrix(target_quaternion), target_position
        )

        damping_sq = damping * damping
        identity6 = np.eye(6)

        for _ in range(max_iter):
            q_full = self._to_full_config(q_urdf)
            self._pinocchio.framesForwardKinematics(self.model, self.data, q_full)
            oMf = self.data.oMf[self.frame_id]

            # Spatial error (twist) in the local frame; LOCAL Jacobian matches it.
            dMi = oMdes.inverse() * oMf
            err = self._pinocchio.log6(dMi).vector
            if float(np.linalg.norm(err)) < tol:
                break

            # Jacobian of the EE frame in the local frame.
            J_full = self._pinocchio.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                self.frame_id,
                self._pinocchio.ReferenceFrame.LOCAL,
            )
            # Select columns corresponding to our 6 controlled joints.
            J = np.empty((6, 6), dtype=np.float64)
            for i, q_idx in enumerate(self._q_indices):
                J[:, i] = J_full[:, q_idx]

            # Standard Gauss-Newton update with integration step.
            JJt = J @ J.T
            v = -J.T @ np.linalg.solve(JJt + damping_sq * identity6, err)
            q_full = self._pinocchio.integrate(
                self.model, q_full, v * dt
            )
            q_urdf = self.clip_urdf(self._from_full_config(q_full))

        return self.urdf_to_rlinf(q_urdf)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _to_full_config(self, q_urdf_6: np.ndarray) -> np.ndarray:
        """Map the 6 controlled joints to the full URDF configuration vector."""
        q_full = np.zeros(self.model.nq, dtype=np.float64)
        for q_idx, value in zip(self._q_indices, q_urdf_6):
            q_full[q_idx] = float(value)
        return q_full

    def _from_full_config(self, q_full: np.ndarray) -> np.ndarray:
        """Extract the 6 controlled joints from the full URDF configuration."""
        return np.array([q_full[q_idx] for q_idx in self._q_indices], dtype=np.float64)

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert ``[x, y, z, w]`` quaternion to a 3x3 rotation matrix."""
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )


def make_dual_arm_kinematics(
    urdf_path: str | Path | None = None,
    end_effector_link: str = "gripper_frame_link",
) -> dict[str, SO101ArmKinematics]:
    """Return left/right SO101 arm kinematics instances."""
    return {
        "left": SO101ArmKinematics(urdf_path=urdf_path, end_effector_link=end_effector_link),
        "right": SO101ArmKinematics(urdf_path=urdf_path, end_effector_link=end_effector_link),
    }
