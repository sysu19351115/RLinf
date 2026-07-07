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

Wraps the LeRobot ``BiSOFollower`` robot so it can be placed on the robot node
of a RLinf cluster.  All LeRobot imports are deferred to ``__init__`` so this
module can be imported on training-only nodes that do not have the robot SDK
or serial hardware attached.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState

# Motor ordering expected by the SO101 environment and OpenPI policy.
_MOTOR_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)

# Camera name mapping from OpenPI convention to LeRobot robot observation keys.
_CAMERA_NAME_MAP = {
    "cam_high": "left_global",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}


def _state_from_robot_obs(robot_obs: dict[str, Any]) -> np.ndarray:
    """Extract the 12-dim motor position vector from a LeRobot observation."""
    positions = []
    for name in _MOTOR_NAMES:
        key = f"{name}.pos"
        if key not in robot_obs:
            raise KeyError(f"Robot observation missing motor key: {key}")
        positions.append(float(robot_obs[key]))
    return np.asarray(positions, dtype=np.float64)


def _images_from_robot_obs(robot_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract and convert camera images from a LeRobot observation.

    Returns images in CHW RGB uint8 format (the OpenPI convention).
    """
    import cv2

    images: dict[str, np.ndarray] = {}
    for openpi_name, robot_name in _CAMERA_NAME_MAP.items():
        if robot_name not in robot_obs:
            raise KeyError(f"Robot observation missing camera key: {robot_name}")
        image = np.asarray(robot_obs[robot_name])
        if image.ndim == 3 and image.shape[-1] == 3:
            # HWC -> CHW
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
        images[openpi_name] = image.astype(np.uint8)
    return images


def _action_vector_to_robot_action(action: np.ndarray) -> dict[str, float]:
    """Convert a 12-dim action vector into LeRobot motor position commands."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 2:
        if action.shape[0] == 0:
            raise ValueError(f"Expected 12-dim action, got shape {action.shape}")
        action = action[0]
    if action.shape != (len(_MOTOR_NAMES),):
        raise ValueError(f"Expected 12-dim action, got shape {action.shape}")
    return {
        f"{name}.pos": float(action[index]) for index, name in enumerate(_MOTOR_NAMES)
    }


class SO101Controller(Worker):
    """SO101 bimanual arm controller.

    Wraps LeRobot ``BiSOFollower`` as a :class:`Worker` so it can be placed on
    the robot node of a RLinf cluster.
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
        """Launch a :class:`SO101Controller} on the specified node."""
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

        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
        from lerobot.robots.so_follower import SOFollowerConfig

        def _camera_config(spec: dict[str, Any]) -> OpenCVCameraConfig:
            index_or_path = spec["index_or_path"]
            try:
                index_or_path = int(index_or_path)
            except (ValueError, TypeError):
                index_or_path = Path(index_or_path)
            return OpenCVCameraConfig(
                index_or_path=index_or_path,
                width=int(spec.get("width", 1920)),
                height=int(spec.get("height", 1080)),
                fps=int(spec.get("fps", 30)),
                fourcc=spec.get("fourcc", "MJPG"),
            )

        config = BiSOFollowerConfig(
            id=robot_id,
            left_arm_config=SOFollowerConfig(
                port=left_follower_port,
                max_relative_target=max_relative_target,
                cameras={
                    "wrist": _camera_config(left_wrist_camera),
                    "global": _camera_config(left_global_camera),
                },
            ),
            right_arm_config=SOFollowerConfig(
                port=right_follower_port,
                max_relative_target=max_relative_target,
                cameras={"wrist": _camera_config(right_wrist_camera)},
            ),
        )
        self._robot = BiSOFollower(config)
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
        robot_obs = self._robot.get_observation()
        q = _state_from_robot_obs(robot_obs)

        left_gripper_pos = float(robot_obs.get("left_gripper.pos", q[5]))
        right_gripper_pos = float(robot_obs.get("right_gripper.pos", q[11]))

        return SO101RobotState(
            arm_joint_position=q,
            left_gripper_position=left_gripper_pos,
            right_gripper_position=right_gripper_pos,
            left_gripper_open=left_gripper_pos > 0.5,
            right_gripper_open=right_gripper_pos > 0.5,
        )

    def get_observation(self) -> dict[str, Any]:
        """Return the full observation in OpenPI-compatible format.

        Keys: ``state`` (12,), ``images`` {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        and ``prompt`` is omitted here (filled in by the env).
        """
        robot_obs = self._robot.get_observation()
        return {
            "state": _state_from_robot_obs(robot_obs),
            "images": _images_from_robot_obs(robot_obs),
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send a 12-dim absolute position target to the robot."""
        robot_action = _action_vector_to_robot_action(action)
        self._robot.send_action(robot_action)

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
        if target_joints.shape != (len(_MOTOR_NAMES),):
            raise ValueError(
                f"Expected target_joints shape {(len(_MOTOR_NAMES),)}, got {target_joints.shape}"
            )

        current = _state_from_robot_obs(self._robot.get_observation())
        sleep_s = 1.0 / init_fps if init_fps > 0 else 0.0
        for step in range(1, init_steps + 1):
            alpha = step / init_steps
            command = current + (target_joints - current) * alpha
            self._robot.send_action(_action_vector_to_robot_action(command))
            if sleep_s > 0:
                time.sleep(sleep_s)
