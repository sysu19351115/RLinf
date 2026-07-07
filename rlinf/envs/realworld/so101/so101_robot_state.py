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
"""Robot state dataclass for the SO101 bimanual arm."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SO101RobotState:
    """State of the SO101 bimanual follower arms.

    The state layout is:
      - left arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (5 joints)
      - left gripper (1)
      - right arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (5 joints)
      - right gripper (1)
    """

    arm_joint_position: np.ndarray
    """All 12 motor positions ``(12,)`` in radians / arbitrary gripper units."""

    arm_joint_velocity: np.ndarray | None = None
    """Optional motor velocities ``(12,)``."""

    arm_joint_torque: np.ndarray | None = None
    """Optional motor torques ``(12,)``."""

    left_gripper_position: float = 0.0
    """Left gripper opening position."""

    right_gripper_position: float = 0.0
    """Right gripper opening position."""

    left_gripper_open: bool = True
    """Whether the left gripper is considered open."""

    right_gripper_open: bool = True
    """Whether the right gripper is considered open."""
