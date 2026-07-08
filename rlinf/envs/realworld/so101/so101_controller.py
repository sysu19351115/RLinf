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
"""SO101 bimanual robot controller as a distributed Ray Worker.

Wraps the LeRobot ``ManipulatorRobot`` (``So101RobotConfig`` with two follower
arms) so it can be placed on the robot node of a RLinf cluster.  All LeRobot
imports are deferred to ``__init__`` so this module can be imported on
training-only nodes that do not have the robot SDK or serial hardware attached.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState

# Motor ordering expected by the SO101 environment and OpenPI policy.
_LEFT_MOTOR_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
)
_RIGHT_MOTOR_NAMES = (
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)

# LeRobot per-arm motor names.  Order within an arm must match the policy.
_ARM_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Camera name mapping from OpenPI convention to LeRobot camera keys.
_CAMERA_NAME_MAP = {
    "cam_high": "left_global",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}


def _package_dir() -> Path:
    """Return the directory containing this module."""
    return Path(__file__).resolve().parent


def _state_from_robot_obs(arm_joint_position: np.ndarray) -> np.ndarray:
    """Validate and return the 12-dim motor position vector from a LeRobot obs.

    ``arm_joint_position`` is expected to concatenate left-arm then right-arm
    positions in the order defined by ``_ARM_MOTOR_NAMES``.
    """
    q = np.asarray(arm_joint_position, dtype=np.float64)
    if q.shape != (len(_LEFT_MOTOR_NAMES) + len(_RIGHT_MOTOR_NAMES),):
        raise ValueError(f"Expected 12-dim joint position, got shape {q.shape}")
    return q


def _images_from_robot_obs(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Extract and convert camera images from a LeRobot observation.

    Returns images in CHW RGB uint8 format (the OpenPI convention).
    """
    out: dict[str, np.ndarray] = {}
    for openpi_name, robot_name in _CAMERA_NAME_MAP.items():
        if robot_name not in images:
            raise KeyError(f"Robot observation missing camera key: {robot_name}")
        image = np.asarray(images[robot_name])
        if image.ndim == 3 and image.shape[-1] == 3:
            # LeRobot OpenCVCamera returns HWC RGB; transpose to CHW.
            image = np.transpose(image, (2, 0, 1))
        out[openpi_name] = image.astype(np.uint8)
    return out


def _action_vector_to_robot_action(action: np.ndarray) -> torch.Tensor:
    """Convert a 12-dim action vector into a tensor for ``ManipulatorRobot``.

    The action is expected in the same calibrated units produced by LeRobot:
    arm joints in ``[-100, 100]`` and gripper in ``[0, 100]``.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 2:
        if action.shape[0] == 0:
            raise ValueError(f"Expected 12-dim action, got shape {action.shape}")
        action = action[0]
    expected = len(_LEFT_MOTOR_NAMES) + len(_RIGHT_MOTOR_NAMES)
    if action.shape != (expected,):
        raise ValueError(f"Expected {expected}-dim action, got shape {action.shape}")
    return torch.from_numpy(action)


class SO101Controller(Worker):
    """SO101 bimanual arm controller.

    Wraps LeRobot ``ManipulatorRobot`` as a :class:`Worker` so it can be placed
    on the robot node of a RLinf cluster.
    """

    @staticmethod
    def launch_controller(
        *,
        left_follower_port: str,
        right_follower_port: str,
        left_wrist_camera: dict[str, Any],
        right_wrist_camera: dict[str, Any],
        left_global_camera: dict[str, Any],
        robot_id: str = "bi",
        max_relative_target: float | None = 5.0,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
    ):
        """Launch a :class:`SO101Controller` on the specified node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return SO101Controller.create_group(
            left_follower_port=left_follower_port,
            right_follower_port=right_follower_port,
            left_wrist_camera=left_wrist_camera,
            right_wrist_camera=right_wrist_camera,
            left_global_camera=left_global_camera,
            robot_id=robot_id,
            max_relative_target=max_relative_target,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"SO101Controller-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        *,
        left_follower_port: str,
        right_follower_port: str,
        left_wrist_camera: dict[str, Any],
        right_wrist_camera: dict[str, Any],
        left_global_camera: dict[str, Any],
        robot_id: str = "bi",
        max_relative_target: float | None = 5.0,
    ):
        super().__init__()
        self._logger = get_logger()

        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        def _camera_config(spec: dict[str, Any]) -> OpenCVCameraConfig:
            index_or_path = spec["index_or_path"]
            try:
                camera_index = int(index_or_path)
            except (ValueError, TypeError):
                camera_index = str(index_or_path)
            return OpenCVCameraConfig(
                camera_index=camera_index,
                width=int(spec.get("width", 1920)),
                height=int(spec.get("height", 1080)),
                fps=int(spec.get("fps", 30)),
            )

        def _arm_config(port: str) -> FeetechMotorsBusConfig:
            return FeetechMotorsBusConfig(
                port=port,
                motors={
                    name: (idx, "sts3215")
                    for idx, name in enumerate(_ARM_MOTOR_NAMES, start=1)
                },
            )

        calibration_dir = _package_dir() / "calibration"
        config = So101RobotConfig(
            calibration_dir=str(calibration_dir),
            follower_arms={
                "left": _arm_config(left_follower_port),
                "right": _arm_config(right_follower_port),
            },
            cameras={
                "left_global": _camera_config(left_global_camera),
                "left_wrist": _camera_config(left_wrist_camera),
                "right_wrist": _camera_config(right_wrist_camera),
            },
            max_relative_target=max_relative_target,
        )

        self._robot = ManipulatorRobot(config)
        self._robot.connect()
        self._logger.info(
            f"SO101Controller connected: left={left_follower_port}, right={right_follower_port}"
        )

    def close(self):
        """Disconnect the robot."""
        try:
            if hasattr(self, "_robot") and self._robot is not None:
                self._robot.disconnect()
        except Exception:
            pass

    def is_robot_up(self) -> bool:
        """Return ``True`` when the robot is connected."""
        return bool(getattr(self._robot, "is_connected", False))

    def get_state(self) -> SO101RobotState:
        """Return the current robot state (joint positions + gripper heuristics)."""
        q = _state_from_robot_obs(self._robot.capture_observation()["observation.state"].numpy())

        left_gripper_pos = float(q[5])
        right_gripper_pos = float(q[11])

        return SO101RobotState(
            arm_joint_position=q,
            left_gripper_position=left_gripper_pos,
            right_gripper_position=right_gripper_pos,
            left_gripper_open=left_gripper_pos > 50.0,
            right_gripper_open=right_gripper_pos > 50.0,
        )

    def get_observation(self) -> dict[str, Any]:
        """Return the full observation in OpenPI-compatible format.

        Keys: ``state`` (12,), ``images`` {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        and ``prompt`` is omitted here (filled in by the env).
        """
        obs = self._robot.capture_observation()
        return {
            "state": _state_from_robot_obs(obs["observation.state"].numpy()),
            "images": _images_from_robot_obs(
                {key.removeprefix("observation.images."): value.numpy() for key, value in obs.items() if key.startswith("observation.images.")}
            ),
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send a 12-dim absolute position target to the robot."""
        action_tensor = _action_vector_to_robot_action(action)
        self._robot.send_action(action_tensor)

    def move_to_pose(
        self,
        target_joints: np.ndarray,
        *,
        init_steps: int = 60,
        init_fps: float = 30.0,
    ) -> None:
        """Smoothly move to a target joint configuration.

        Uses linear interpolation in joint space.  This is used during reset
        to go to the initial pose.
        """
        import time

        target_joints = np.asarray(target_joints, dtype=np.float32)
        expected = len(_LEFT_MOTOR_NAMES) + len(_RIGHT_MOTOR_NAMES)
        if target_joints.shape != (expected,):
            raise ValueError(
                f"Expected target_joints shape {(expected,)}, got {target_joints.shape}"
            )

        current = self._robot.capture_observation()["observation.state"].numpy()
        sleep_s = 1.0 / init_fps if init_fps > 0 else 0.0
        for step in range(1, init_steps + 1):
            alpha = step / init_steps
            command = current + (target_joints - current) * alpha
            self._robot.send_action(torch.from_numpy(command.astype(np.float32)))
            if sleep_s > 0:
                time.sleep(sleep_s)
