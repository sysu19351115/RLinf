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
"""SO101 bimanual robot environment for RLinf real-world RL."""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

# Default joint limits are intentionally wide; the hardware-level
# max_relative_target provides the primary safety bound.
# Arm joints are exposed to the policy in [-100, 100] (0 = mid-range), while
# LeRobot's LINEAR calibration internally maps them to [0, 100]. The conversion
# is handled in SO101Controller. Grippers are [0, 100] on both sides.
_DEFAULT_JOINT_LIMIT_LOW = np.array(
    [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0] * 2, dtype=np.float64
)
_DEFAULT_JOINT_LIMIT_HIGH = np.array(
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0] * 2, dtype=np.float64
)

# Camera ordering used in the observation space and RealWorldEnv main_image_key.
_CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _camera_spec(
    index_or_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    fourcc: str = "MJPG",
) -> dict[str, Any]:
    """Build a camera spec dict compatible with SO101Controller."""
    return {
        "index_or_path": index_or_path,
        "width": width,
        "height": height,
        "fps": fps,
        "fourcc": fourcc,
    }


@dataclass
class SO101RobotConfig:
    """Configuration for :class:`SO101Env`.

    Hardware connection fields are populated from ``hardware_info`` when ``None``.
    """

    left_follower_port: Optional[str] = None
    """Serial port for the left follower arm (e.g. ``"/dev/ttyACM2"``)."""

    right_follower_port: Optional[str] = None
    """Serial port for the right follower arm (e.g. ``"/dev/ttyACM3"``)."""

    left_wrist_camera: dict[str, Any] = field(default_factory=dict)
    """Camera spec for the left wrist camera."""

    right_wrist_camera: dict[str, Any] = field(default_factory=dict)
    """Camera spec for the right wrist camera."""

    left_global_camera: dict[str, Any] = field(default_factory=dict)
    """Camera spec for the left global (high) camera."""

    robot_id: str = "bi"
    """Robot ID passed to BiSOFollower."""

    max_relative_target: float = 5.0
    """Max relative target passed to SOFollowerConfig."""

    initial_joints: Optional[np.ndarray] = None
    """12-dim initial joint configuration used at reset."""

    end_joints: Optional[np.ndarray] = None
    """Optional 12-dim end-of-episode joint configuration."""

    init_steps: int = 60
    """Number of interpolation steps for reset motion."""

    init_fps: float = 30.0
    """Frequency of reset interpolation."""

    tele_mode: bool = False
    """If ``True``, do not send actions to the motors (read-only safety mode)."""

    is_dummy: bool = False
    """Skip all hardware calls."""

    task_description: str = ""
    """Language instruction for the task."""

    use_dense_reward: bool = False
    """Use distance-based dense reward instead of binary 0/1."""

    use_reward_model: bool = False
    """Use a learned vision-based reward model."""

    reward_worker_cfg: Optional[dict] = None
    """Configuration dict passed to the embodied reward worker."""

    reward_worker_node_rank: Optional[int] = None
    """Node rank on which to place the reward worker."""

    reward_worker_node_group: Optional[str] = None
    """Optional node group label for reward worker placement."""

    reward_worker_hardware_rank: int = 0
    """GPU/hardware rank for the reward worker."""

    reward_image_key: Optional[str] = None
    """Key in ``observation['frames']`` to use for reward model inference."""

    step_frequency: float = 10.0
    """Maximum environment steps per second."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW.copy()
    )
    """Lower joint limits ``(12,)`` in radians used to clamp actions."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH.copy()
    )
    """Upper joint limits ``(12,)`` in radians used to clamp actions."""

    max_num_steps: int = 100
    """Episode truncation horizon."""

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Target end-effector pose placeholder for dense/sparse reward."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    )
    """Per-axis tolerances for the legacy geometric success check."""

    success_hold_steps: int = 1
    """Number of consecutive steps in the target zone required for success."""

    save_video_path: Optional[str] = None
    """Path to save episode videos. ``None`` disables saving."""


class SO101Env(gym.Env):
    """SO101 bimanual robot environment with joint-space actions.

    Action space: ``Box((12,))`` — absolute motor position targets for the 12
    motors (left 5 joints + gripper, right 5 joints + gripper).

    Observation: ``Dict{state: Dict{arm_joint_position: (12,)}, frames: Dict{wrist_i: (H,W,3)}}``
    """

    def __init__(
        self,
        config: Optional[SO101RobotConfig] = None,
        override_cfg: Optional[dict] = None,
        worker_info: Optional[WorkerInfo] = None,
        hardware_info=None,
        env_idx: int = 0,
        env_cfg=None,
        **kwargs,
    ):
        self._logger = get_logger()
        self.config = config if config is not None else SO101RobotConfig()
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
        self._reward_worker = None

        if not self.config.is_dummy:
            self._setup_hardware()
        if self.config.use_reward_model:
            self._setup_reward_worker()

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time:.0f}s for SO101 to be ready."
                )

        self._state = self._controller.get_state().wait()[0]
        self._move_to_initial_pose()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_hardware(self):
        from .so101_controller import SO101Controller

        hw = self.hardware_info.config if self.hardware_info is not None else None

        def _from_hw(key: str, default: Any = None):
            return getattr(hw, key, default) if hw is not None else default

        left_port = self.config.left_follower_port or _from_hw(
            "left_follower_port", "/dev/ttyACM2"
        )
        right_port = self.config.right_follower_port or _from_hw(
            "right_follower_port", "/dev/ttyACM3"
        )

        left_wrist = self.config.left_wrist_camera or _from_hw(
            "left_wrist_camera",
            {"index_or_path": "/dev/video0"},
        )
        right_wrist = self.config.right_wrist_camera or _from_hw(
            "right_wrist_camera",
            {"index_or_path": "/dev/video1"},
        )
        left_global = self.config.left_global_camera or _from_hw(
            "left_global_camera",
            {"index_or_path": "/dev/video2"},
        )

        controller_node_rank = _from_hw("controller_node_rank", None)
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self._controller = SO101Controller.launch_controller(
            left_follower_port=left_port,
            right_follower_port=right_port,
            left_wrist_camera=left_wrist,
            right_wrist_camera=right_wrist,
            left_global_camera=left_global,
            robot_id=self.config.robot_id,
            max_relative_target=self.config.max_relative_target,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
        )

    def _setup_reward_worker(self):
        if not self.config.use_reward_model:
            return
        if self.config.reward_worker_cfg is None:
            raise ValueError(
                "use_reward_model=True but reward_worker_cfg is not provided."
            )

        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        reward_node_rank = self.config.reward_worker_node_rank
        if reward_node_rank is None:
            reward_node_rank = self.node_rank

        self._reward_worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=self.config.reward_worker_cfg,
            node_rank=reward_node_rank,
            node_group_label=self.config.reward_worker_node_group,
            hardware_rank=self.config.reward_worker_hardware_rank,
            env_idx=self.env_idx,
            worker_rank=self.env_worker_rank,
        )
        self._reward_worker.init_worker().wait()
        self._logger.info(
            f"Reward worker initialized for env {self.env_idx} on node {reward_node_rank}"
        )

    def _init_action_obs_spaces(self):
        self._joint_limit_low = np.array(self.config.joint_limit_low, dtype=np.float64)
        self._joint_limit_high = np.array(
            self.config.joint_limit_high, dtype=np.float64
        )

        self.action_space = gym.spaces.Box(
            self._joint_limit_low.astype(np.float32),
            self._joint_limit_high.astype(np.float32),
        )

        frame_spaces = {}
        for name in _CAMERA_ORDER:
            frame_spaces[name] = gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "arm_joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(12,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(frame_spaces),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ── Core gym API ──────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy and not self.config.tele_mode:
            q_target = np.clip(action, self._joint_limit_low, self._joint_limit_high)
            self._controller.send_action(q_target).wait()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._state = self._controller.get_state().wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(observation)

        terminated = (reward >= 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps

        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def reset(self, seed=None, options=None):
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0
        self._move_to_initial_pose()
        self._num_steps = 0
        if not self.config.is_dummy:
            self._state = self._controller.get_state().wait()[0]
        return self._get_observation(), {}

    def _move_to_initial_pose(self):
        if self.config.is_dummy or self.config.initial_joints is None:
            return
        self._controller.move_to_pose(
            self.config.initial_joints,
            init_steps=self.config.init_steps,
            init_fps=self.config.init_fps,
        ).wait()
        time.sleep(0.5)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _calc_step_reward(self, observation: dict) -> float:
        if self.config.is_dummy and not self.config.use_reward_model:
            return 0.0

        if self.config.use_reward_model:
            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            return max(0.0, min(1.0, float(reward)))

        # Legacy dense/sparse reward based on a placeholder target pose.
        # SO101 does not have FK in the base env; this is a fallback only.
        delta = np.abs(
            self._state.arm_joint_position[:6] - self.config.target_ee_pose[:6]
        )
        is_in_target_zone = np.all(delta[:3] <= self.config.reward_threshold[:3])

        if is_in_target_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(np.exp(-500.0 * np.sum(np.square(delta[:3]))))
            else:
                reward = 0.0

        return max(0.0, min(1.0, reward))

    def _compute_reward_model(self, observation: dict[str, Any]) -> float:
        if self._reward_worker is None:
            raise RuntimeError(
                "Reward worker is not initialized but use_reward_model=True."
            )

        frames = observation.get("frames", {})
        if not frames:
            raise ValueError("No frames available for reward model inference.")

        image_key = self.config.reward_image_key
        if image_key is None:
            image_key = sorted(frames.keys())[0]
        if image_key not in frames:
            raise KeyError(
                f"reward_image_key '{image_key}' not found in frames. "
                f"Available keys: {list(frames.keys())}"
            )

        image_batch = np.expand_dims(frames[image_key], axis=0)
        reward_output = self._reward_worker.compute_image_rewards(image_batch).wait()[0]
        if hasattr(reward_output, "detach"):
            reward_output = reward_output.detach().cpu().numpy()
        reward_array = np.asarray(reward_output).reshape(-1)
        return float(reward_array[0])

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            obs = self._controller.get_observation().wait()[0]
            frames = self._crop_and_resize_frames(obs["images"])
            if not frames:
                frames = {
                    name: np.zeros((128, 128, 3), dtype=np.uint8)
                    for name in _CAMERA_ORDER
                }
            state = {"arm_joint_position": self._state.arm_joint_position}
            return copy.deepcopy({"state": state, "frames": frames})
        return self._base_observation_space.sample()

    def _crop_and_resize_frames(
        self, raw_frames: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        frames = {}
        for name in _CAMERA_ORDER:
            if name not in raw_frames:
                continue
            img = raw_frames[name]
            # CHW -> HWC if necessary.
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            h, w = img.shape[:2]
            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cropped = img[start_y : start_y + crop_size, start_x : start_x + crop_size]
            resized = cv2.resize(cropped, (128, 128))
            frames[name] = resized
        return frames

    # ── Utilities ─────────────────────────────────────────────────────────────

    def close(self):
        """Release hardware resources."""
        if not self.config.is_dummy and hasattr(self, "_controller"):
            self._controller.close().wait()
        super().close()

    @property
    def task_description(self) -> str:
        return self.config.task_description
