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

import os
import sys
import time
from typing import Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .rebot_robot_state import RebotArmRobotState


class RebotArmController(Worker):
    """RebotArm robot arm controller.

    Wraps the ``reBotArm_control_py`` SDK (CAN bus) as a distributed
    :class:`Worker` so it can be placed on any node in the cluster.

    ``reBotArm`` already runs an internal 500 Hz control loop that reads
    ``_q_target`` and sends POS_VEL commands to the motors.  This controller
    simply sets the joint target via :meth:`reBotArm.servo_joints` and reads
    state via :meth:`reBotArm.get_joint_positions` / FK.

    All ``reBotArm_control_py`` and ``scipy`` imports are deferred to
    :meth:`__init__` so this module can be imported on GPU-only nodes that
    do not have the robot SDK installed.
    """

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    @staticmethod
    def launch_controller(
        can_interface: str = "can0",
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
    ):
        """Launch a :class:`RebotArmController` on the specified node.

        Args:
            can_interface: CAN socket interface name.
            env_idx: Environment index for naming.
            node_rank: Cluster node rank to place on.
            worker_rank: Worker rank for naming.

        Returns:
            RebotArmController: The launched remote controller instance.
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return (
            RebotArmController.create_group(can_interface)
            .launch(
                cluster=cluster,
                placement_strategy=placement,
                name=f"RebotArmController-{worker_rank}-{env_idx}",
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, can_interface: str = "can0"):
        super().__init__()
        self._logger = get_logger()

        # Ensure the rebot SDK is importable (it lives alongside this module).
        _rebot_dir = os.path.dirname(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..",
        )))
        if _rebot_dir not in sys.path:
            sys.path.insert(0, _rebot_dir)

        from reBotArm_control_py import reBotArm
        from scipy.spatial.transform import Rotation as R

        self._R = R

        self._arm = reBotArm()
        self._arm.connect()
        self._arm.power_on()
        self._logger.info(
            f"RebotArmController connected on '{can_interface}' and powered on."
        )

    def close(self):
        """Safely shut down the arm."""
        try:
            self._arm.stop()
            self._arm.power_off()
            self._arm.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_robot_up(self) -> bool:
        """Return ``True`` when the arm is connected and powered."""
        return self._arm.is_connected and self._arm.is_powered

    def get_state(self) -> RebotArmRobotState:
        """Compute and return the current robot state.

        Reads joint positions / velocities / torques from the hardware and
        performs FK to obtain the TCP pose.
        """
        q, dq, tau = self._arm.get_state()
        q = np.array(q, dtype=np.float64)
        dq = np.array(dq, dtype=np.float64)
        tau = np.array(tau, dtype=np.float64)

        pos_xyz, rpy = self._arm.get_end_effector_pose()
        pos_xyz = np.array(pos_xyz, dtype=np.float64)
        rpy = np.array(rpy, dtype=np.float64)

        tcp_quat = self._R.from_euler("xyz", rpy).as_quat()
        tcp_pose = np.concatenate([pos_xyz, tcp_quat])

        gripper_pos = self._arm.get_gripper_position()
        gripper_open = gripper_pos > 2.0  # heuristic: mid-point between open(4.7) and close(0.0)

        return RebotArmRobotState(
            arm_joint_position=q,
            arm_joint_velocity=dq,
            arm_joint_torque=tau,
            tcp_pose=tcp_pose,
            tcp_vel=np.zeros(6),
            gripper_position=float(gripper_pos),
            gripper_open=gripper_open,
        )

    def get_end_effector_pose(self):
        """Read the current end-effector pose.

        Returns:
            tuple: ``(position, rpy)`` where position is ``(3,)`` in meters
            and rpy is ``(3,)`` in radians.
        """
        pos_xyz, rpy = self._arm.get_end_effector_pose()
        return np.array(pos_xyz, dtype=np.float64), np.array(rpy, dtype=np.float64)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_joints(self, q_target: np.ndarray) -> None:
        """Set the target joint position (non-blocking).

        Calls ``reBotArm.servo_joints`` to immediately write the target.
        The internal 500 Hz control loop picks it up and drives the motors.

        Args:
            q_target: Desired joint positions ``(6,)`` in radians.
        """
        self._arm.servo_joints(q_target)

    def reset_joint(
        self,
        reset_qpos: list[float],
        duration: float = 3.0,
    ) -> None:
        """Move to a joint reset configuration using trajectory planning.

        Uses the built-in ``move_joints`` which performs joint-space
        linear interpolation and blocks until complete.

        Args:
            reset_qpos: Target joint positions ``(6,)`` in radians.
            duration: Time in seconds for the motion.
        """
        self._arm.move_joints(reset_qpos, duration=duration, wait=True)

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def open_gripper(self) -> None:
        """Open the gripper."""
        self._arm.move_open_gripper()

    def close_gripper(self) -> None:
        """Close the gripper."""
        self._arm.move_close_gripper()

    def move_gripper(self, position: float) -> None:
        """Set the gripper target position.

        Args:
            position: Target gripper motor position.
        """
        self._arm.move_gripper(position)
