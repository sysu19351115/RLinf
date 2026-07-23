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

"""State snapshot for the Dobot CR5AF 6-DOF arm + Damiao gripper.

All joint quantities are in **radians** (converted from Dobot's native degrees at the
controller boundary). The TCP pose uses **meters + quaternion ``[qw, qx, qy, qz]``**
(w-first), matching the SDK's upper-layer convention. The gripper is **normalized
``[0, 1]``** where ``0 = closed`` and ``1 = open``.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DobotRobotState:
    """State snapshot for the Dobot CR5AF 6-DOF robot + Damiao gripper.

    All Cartesian quantities are expressed in the robot base frame. The optional
    ``wrench`` is only populated when the six-axis force/torque sensor is enabled.
    """

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions ``[q1, ..., q6]`` in radians."""

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """End-effector pose ``[x, y, z, qw, qx, qy, qz]`` (m / quaternion, w-first)."""

    gripper_position: float = 0.0
    """Gripper normalized position ``[0, 1]`` (0=closed, 1=open)."""

    gripper_open: bool = False
    """``True`` when the gripper is open (normalized position >= 0.5)."""

    action_mode: str = "joint"
    """The action mode this state was read under: ``"joint"`` or ``"cartesian"``."""

    servo_accepted: bool = True
    """Whether the last servo command was accepted by the safety guard."""

    arm_joint_velocity: Optional[np.ndarray] = None
    """Joint velocities ``[dq1, ..., dq6]`` in rad/s (optional, not always read)."""

    wrench: Optional[np.ndarray] = None
    """End-effector wrench ``[Fx, Fy, Fz, Mx, My, Mz]`` in N / N-m (optional)."""

    wrench_online: bool = False
    """Whether the force/torque sensor was online for this frame."""

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
