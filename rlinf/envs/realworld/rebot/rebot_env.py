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

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .rebot_robot_state import RebotArmRobotState

# reBotArm B601-RS joint limits (rad) — from hardware config.
_DEFAULT_JOINT_LIMIT_LOW = np.array([-3.14, -1.57, -3.14, -3.14, -3.14, -3.14])
_DEFAULT_JOINT_LIMIT_HIGH = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])


@dataclass
class RebotArmRobotConfig:
    """Configuration for :class:`RebotArmEnv`.

    Hardware connection fields (``can_interface``, ``camera_serials``) are
    populated automatically from ``RebotArmHWInfo`` when ``None``.
    """

    can_interface: Optional[str] = None
    """CAN socket interface name (e.g. ``"can0"``)."""

    camera_serials: Optional[list[str]] = None
    """Ordered list of camera serial numbers for observations."""

    camera_type: Optional[str] = None
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    enable_gripper: bool = True
    """Whether the gripper is attached."""

    enable_camera_player: bool = True
    """Display a live camera window during episodes."""

    is_dummy: bool = False
    """When ``True``, skip all hardware calls (useful for offline training)."""

    task_description: str = ""
    """Language prompt describing the task (e.g. 'pick objects and insert into the tray box')."""

    use_dense_reward: bool = False
    """Use distance-based dense reward instead of binary 0/1."""

    step_frequency: float = 10.0
    """Maximum environment steps per second."""

    # Target and reset poses.
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    """Target end-effector pose ``[x, y, z, rx, ry, rz]`` (m / Euler XYZ)."""

    reset_qpos: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    """Joint configuration to move to at the start of each episode."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW.copy()
    )
    """Lower joint limits ``(6,)`` in radians used to clamp actions."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH.copy()
    )
    """Upper joint limits ``(6,)`` in radians used to clamp actions."""

    max_num_steps: int = 100
    """Episode truncation horizon."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    )
    """Per-axis tolerances ``[x, y, z, rx, ry, rz]`` for the success check."""

    binary_gripper_threshold: float = 0.5
    """Action magnitude threshold for open/close gripper transitions."""

    enable_gripper_penalty: bool = True
    """Subtract penalty from reward on each gripper state change."""

    gripper_penalty: float = 0.1
    """Reward penalty per gripper action."""

    success_hold_steps: int = 1
    """Number of consecutive steps in the target zone required for success."""

    # Pick-and-place specific fields (used by RebotArmPickAndPlaceEnv).
    pick_pose: Optional[np.ndarray] = None
    """Pick target pose ``[x, y, z]`` (optional)."""

    place_pose: Optional[np.ndarray] = None
    """Place target pose ``[x, y, z]`` (optional)."""

    save_video_path: Optional[str] = None
    """Path to save episode videos. ``None`` disables saving."""


class RebotArmEnv(gym.Env):
    """RebotArm 6-DOF robot environment with joint-space actions.

    Action space:  ``Box((7,))`` — ``[q1, ..., q6, gripper]``
    Observation:   ``Dict{state: Dict{...}, frames: Dict{wrist_i: ...}}``

    The first six action dimensions are **absolute joint positions** in
    radians, bounded by the configured joint limits.  The seventh element
    is a binary open/close gripper command in ``[-1, 1]`` using
    ``binary_gripper_threshold``.

    Reward is computed in Cartesian space by comparing the FK-computed TCP
    pose to ``target_ee_pose``.
    """

    def __init__(
        self,
        config: Optional[RebotArmRobotConfig] = None,
        override_cfg: Optional[dict] = None,
        worker_info: Optional[WorkerInfo] = None,
        hardware_info=None,
        env_idx: int = 0,
        env_cfg=None,
        **kwargs,
    ):
        self._logger = get_logger()
        self.config = config if config is not None else RebotArmRobotConfig()
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0

        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        # Apply overrides from env_cfg.init_params.
        if env_cfg is not None and hasattr(env_cfg, "init_params"):
            for key, value in env_cfg.init_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Apply override_cfg dict (from RealWorldEnv / reward worker injection).
        if override_cfg:
            for key, value in override_cfg.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self._num_steps = 0
        self._success_hold_counter = 0
        self._gripper_is_open = True

        if not self.config.is_dummy:
            self._setup_hardware()

        if self.config.camera_serials is None:
            self.config.camera_serials = []
        if not self.config.camera_serials:
            self._logger.info(
                "No camera serials configured. "
                "Observations will not contain camera frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time:.0f}s for RebotArm to be ready."
                )

        self._controller.reset_joint(self.config.reset_qpos).wait()
        time.sleep(1.0)
        self._state = self._controller.get_state().wait()[0]

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_hardware(self):
        from .rebot_controller import RebotArmController

        if self.config.can_interface is None and self.hardware_info is not None:
            self.config.can_interface = getattr(
                self.hardware_info.config, "can_interface", "can0"
            )
        if self.config.camera_serials is None and self.hardware_info is not None:
            self.config.camera_serials = getattr(
                self.hardware_info.config, "camera_serials", []
            )
        if self.config.camera_type is None and self.hardware_info is not None:
            self.config.camera_type = getattr(
                self.hardware_info.config, "camera_type", "realsense"
            )

        controller_node_rank = getattr(
            self.hardware_info.config, "controller_node_rank", None
        ) if self.hardware_info is not None else None
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self._controller = RebotArmController.launch_controller(
            can_interface=self.config.can_interface or "can0",
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
        )

    def _init_action_obs_spaces(self):
        """Initialise action and observation spaces."""
        self._joint_limit_low = np.array(
            self.config.joint_limit_low, dtype=np.float64
        )
        self._joint_limit_high = np.array(
            self.config.joint_limit_high, dtype=np.float64
        )

        action_low = np.append(self._joint_limit_low, -1.0).astype(np.float32)
        action_high = np.append(self._joint_limit_high, 1.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        frame_spaces = {}
        num_cameras = len(self.config.camera_serials or [])
        if num_cameras == 0:
            num_cameras = 1  # dummy space for gym compatibility

        for k in range(num_cameras):
            frame_spaces[f"wrist_{k + 1}"] = gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # Pi0.5 rebot policy expects 7-dim state:
                        # 6 joint angles + 1 gripper position.
                        "arm_joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                    }
                ),
                "frames": gym.spaces.Dict(frame_spaces),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ── Core gym API ─────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: ``(7,)`` float array.
                ``action[:6]`` are absolute joint positions in radians.
                ``action[6]`` is the gripper command (binary open/close).
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            q_target = np.clip(
                action[:6], self._joint_limit_low, self._joint_limit_high
            )
            self._controller.move_joints(q_target).wait()

            gripper_action = float(action[6])
            is_gripper_effective = self._gripper_action(gripper_action)
        else:
            is_gripper_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._state = self._controller.get_state().wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_effective)

        terminated = (reward >= 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps

        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def reset(self, seed=None, options=None):
        """Reset the environment to the rest configuration."""
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0
        self.go_to_rest()
        self._num_steps = 0
        self._state = self._controller.get_state().wait()[0]
        self._gripper_is_open = True
        return self._get_observation(), {}

    def go_to_rest(self):
        """Move to the rest configuration."""
        self._controller.reset_joint(self.config.reset_qpos).wait()
        time.sleep(0.5)

    # ── Reward ───────────────────────────────────────────────────────────────

    def _calc_step_reward(
        self,
        observation: dict,
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute reward from FK-based TCP pose vs target pose."""
        if not self.config.is_dummy:
            euler_angles = np.abs(
                R.from_quat(self._state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                self._success_hold_counter += 1
                reward = 1.0
            else:
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = float(
                        np.exp(-500.0 * np.sum(np.square(target_delta[:3])))
                    )
                else:
                    reward = 0.0

            if (
                self.config.enable_gripper_penalty
                and is_gripper_action_effective
            ):
                reward -= self.config.gripper_penalty

            reward = max(0.0, min(1.0, reward))
            return reward
        return 0.0

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            if not frames:
                frames = {
                    "wrist_1": np.zeros((128, 128, 3), dtype=np.uint8)
                }
            # Pi0.5 rebot policy expects 7-dim state: 6 joints + gripper.
            state = {
                "arm_joint_position": self._state.arm_joint_position,
                "gripper_position": np.array(
                    [float(self._state.gripper_position)]
                ),
            }
            return copy.deepcopy({"state": state, "frames": frames})
        return self._base_observation_space.sample()

    # ── Cameras ──────────────────────────────────────────────────────────────

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        if not self.config.camera_serials:
            return
        camera_type = self.config.camera_type or "realsense"
        for i, serial in enumerate(self.config.camera_serials):
            info = CameraInfo(
                name=f"wrist_{i + 1}",
                serial_number=serial,
                camera_type=camera_type,
            )
            camera = create_camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized = cv2.resize(cropped, reshape_size)
        return cropped, resized

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][
                    camera._camera_info.name
                ].shape[:2][::-1]
                _, resized = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = resized[..., ::-1]
                display_frames[camera._camera_info.name] = resized
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} not producing frames."
                )
        self.camera_player.put_frame(display_frames)
        return frames

    # ── Gripper ──────────────────────────────────────────────────────────────

    def _gripper_action(self, position: float) -> bool:
        """Execute a binary gripper open/close.

        Returns:
            ``True`` if a gripper state transition occurred.
        """
        if not self.config.enable_gripper:
            return False

        if position <= -self.config.binary_gripper_threshold and not self._gripper_is_open:
            self._controller.close_gripper().wait()
            self._gripper_is_open = False
            time.sleep(0.3)
            return True

        if position >= self.config.binary_gripper_threshold and self._gripper_is_open:
            self._controller.open_gripper().wait()
            self._gripper_is_open = True
            time.sleep(0.3)
            return True

        return False

    # ── Utilities ────────────────────────────────────────────────────────────

    @property
    def task_description(self) -> str:
        """Language instruction for this task."""
        return self.config.task_description

    @property
    def target_ee_pose(self) -> np.ndarray:
        """Target EEF pose as ``[x, y, z, qx, qy, qz, qw]``."""
        return np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler(
                    "xyz", self.config.target_ee_pose[3:].copy()
                ).as_quat(),
            ]
        ).copy()
