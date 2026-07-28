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

"""Dobot CR5AF 6-DOF arm environment with joint + Cartesian dual-mode control.

Action space (joint mode):    ``Box((7,))``  — ``[j1..j6 (rad), gripper (0..1)]``
Action space (cartesian mode): ``Box((8,))`` — ``[x,y,z (m), qw,qx,qy,qz, gripper (0..1)]``

Observation space: ``Dict{state: Dict{...}, frames: Dict{cam_left_wrist: ..., cam_high?}}``

The state block is 7-dim (joint mode: ``[j1..j6 rad, gripper]``) or 8-dim (pose
mode: ``[x,y,z m, qw,qx,qy,qz, gripper]``). In pose mode, ``prev_state`` is also
tracked (first frame = itself) for the server-side ``DeltaPose`` transform.

Reward is computed either by a learned vision-based reward model, by Cartesian
distance to ``target_ee_pose``, or sparse (default).
"""

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.envs.realworld.dobot.dobot_action_split import RelativeGripperBinarizer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

# Dobot CR5AF joint limits (rad) — conservative defaults.
_DEFAULT_JOINT_LIMIT_LOW = np.array([-3.14, -2.61, -3.14, -3.14, -3.14, -3.14])
_DEFAULT_JOINT_LIMIT_HIGH = np.array([3.14, 2.61, 3.14, 3.14, 3.14, 3.14])

# Image size: pi0.5 training resolution (NOT 128×128 like rebot).
_IMAGE_SIZE = 224
# Camera order: cam_left_wrist is always required, cam_high is optional.
_CAMERA_ORDER = ("cam_high", "cam_left_wrist")


@dataclass
class DobotRobotConfig:
    """Configuration for :class:`DobotEnv`.

    Hardware connection fields (``ip``, ``gripper_port``, ``camera_serials``) are
    populated automatically from :class:`DobotHWInfo` when ``None``.
    """

    # ── Connection ────────────────────────────────────────────────────────────
    ip: Optional[str] = None
    """Dobot controller IP (e.g. ``"192.168.5.1"``)."""

    speed: int = 100
    """Global MovJ speed percentage (0-100)."""

    user_index: int = 0
    """User coordinate system index (must match collection)."""

    tool_index: int = 0
    """Tool coordinate system index (must match collection)."""

    gripper_port: str = "/dev/ttyACM0"
    """Damiao gripper serial bridge device path."""

    enable_gripper: bool = True
    """Whether the gripper is attached and should be controlled."""

    gripper_closed_deg: float = 0.0
    """Gripper closed calibration (motor degrees). Note: closed > open in motor space."""

    gripper_open_deg: float = -320.0
    """Gripper open calibration (motor degrees, negative)."""

    gripper_relative_threshold: Optional[float] = None
    """Optional normalized deadband for stateful binary gripper execution.

    When enabled, the model/human gripper target remains continuous until its
    distance from the measured gripper position exceeds this threshold. The
    executed command then latches to fully closed (0.0) or fully open (1.0).
    """

    enable_ft_sensor: bool = True
    """Enable the six-axis force/torque sensor (required for ForceVLA)."""

    payload: Optional[tuple] = None
    """Optional payload ``(load_kg, x_mm, y_mm, z_mm)``."""

    # ── Dual mode (core) ───────────────────────────────────────────────────────
    action_mode: str = "joint"
    """``"joint"`` (ServoJ, 7-dim action) or ``"cartesian"`` (ServoP, 8-dim action)."""

    state_mode: str = "joint"
    """``"joint"`` (7-dim state) or ``"pose"`` (8-dim state + prev_state).
    ``"pose"`` requires ``action_mode="cartesian"``."""

    # ── Cameras ─────────────────────────────────────────────────────────────────
    camera_serials: Optional[list[str]] = None
    """Ordered list of camera serials. Index 0 = cam_high (optional),
    index 1+ = wrist cameras. ``cam_left_wrist`` is the primary observation."""

    camera_type: Optional[str] = None
    """Camera backend: ``"realsense"``, ``"zed"``, ``"lumos"`` or generic USB via
    ``"opencv"`` / ``"usb"`` / ``"v4l2"`` (LeRobot ``OpenCVCamera``)."""

    camera_resolution: tuple[int, int] = (640, 480)
    """Requested USB/color stream resolution as ``(width, height)``."""

    camera_fps: int = 30
    """Requested color stream frame rate."""

    camera_fourcc: Optional[str] = None
    """Optional V4L2 pixel format FOURCC (e.g. ``"MJPG"``) for the OpenCV USB
    backend. ``None`` keeps the backend default. Set ``"MJPG"`` to capture at
    high resolutions (e.g. 1920x1080) where the default YUYV stream is
    bandwidth-limited; the env center-crops to a square (``min(h, w)``) and
    resizes to 224x224 before returning the observation."""

    enable_high_camera: bool = True
    """Whether a high/scene camera (``cam_high``) is attached."""

    enable_camera_player: bool = True
    """Display a live camera window during episodes."""

    # ── Misc ─────────────────────────────────────────────────────────────────────
    is_dummy: bool = False
    """When ``True``, skip all hardware calls (useful for offline training)."""

    task_description: str = ""
    """Language prompt describing the task."""

    step_frequency: float = 30.0
    """Maximum environment steps per second (Dobot servo rate)."""

    initial_joint_pos: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    )
    """Reset joint pose ``[j1..j6 rad, gripper norm]`` (7-dim). Joint space
    is always used for reset, even in cartesian mode."""

    reset_fps: float = 30.0
    """ServoJ frequency used by the reset trajectory."""

    reset_min_duration_s: float = 3.0
    """Minimum duration of the adaptive minimum-jerk reset trajectory."""

    reset_max_duration_s: float = 20.0
    """Maximum permitted reset duration; longer required trajectories fail closed."""

    reset_max_velocity_deg_s: float = 20.0
    """Per-joint reset velocity limit in degrees/second."""

    reset_max_acceleration_deg_s2: float = 30.0
    """Per-joint reset acceleration limit in degrees/second²."""

    reset_max_step_deg: float = 1.0
    """Maximum planned per-frame joint increment, below the SDK 2° slew guard."""

    reset_feedback_interval_frames: int = 3
    """Read feedback and check RobotMode every N reset frames."""

    reset_max_tracking_error_deg: float = 8.0
    """Maximum ServoJ command-to-feedback error during reset."""

    reset_final_tolerance_deg: float = 1.0
    """Maximum joint error after reset is complete."""

    reset_final_hold_frames: int = 10
    """Number of low-speed target hold frames before final convergence check."""

    reset_gripper_release_settle_s: float = 0.5
    """Wait after physically opening the gripper before moving the arm."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW.copy()
    )
    """Lower joint limits ``(6,)`` in radians."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH.copy()
    )
    """Upper joint limits ``(6,)`` in radians."""

    max_num_steps: int = 500
    """Episode truncation horizon."""

    # ── Reward ───────────────────────────────────────────────────────────────────
    use_dense_reward: bool = False
    """Use distance-based dense reward instead of sparse 0/1."""

    use_reward_model: bool = False
    """Use a learned vision-based reward model instead of geometry."""

    reward_mode: str = "per_step"
    """Reward computation mode: ``per_step`` or ``terminal``.

    ``terminal`` evaluates the reward worker only once, when the episode reaches
    ``max_num_steps``. Earlier steps receive zero reward.
    """

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

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Target end-effector pose ``[x, y, z, rx, ry, rz]`` (m / Euler XYZ) — geometry reward."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
    )
    """Per-axis tolerances ``[x, y, z, rx, ry, rz]`` for the geometric success check."""

    success_hold_steps: int = 1
    """Number of consecutive steps in the target zone required for success."""

    enable_gripper_penalty: bool = False
    """Subtract penalty from reward on each gripper state change."""

    gripper_penalty: float = 0.1
    """Reward penalty per gripper action."""

    _VALID_ACTION_MODES = ("joint", "cartesian")
    _VALID_STATE_MODES = ("joint", "pose")
    _VALID_REWARD_MODES = ("per_step", "terminal")

    def __post_init__(self):
        """Validate mode-compatibility constraints."""
        if self.action_mode not in self._VALID_ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {self._VALID_ACTION_MODES}, "
                f"got {self.action_mode!r}"
            )
        if self.state_mode not in self._VALID_STATE_MODES:
            raise ValueError(
                f"state_mode must be one of {self._VALID_STATE_MODES}, "
                f"got {self.state_mode!r}"
            )
        if self.state_mode == "pose" and self.action_mode != "cartesian":
            raise ValueError(
                "state_mode='pose' requires action_mode='cartesian': "
                "位姿策略输出 8 维绝对位姿动作，必须由 ServoP 执行"
            )
        if self.reward_mode not in self._VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {self._VALID_REWARD_MODES}, "
                f"got {self.reward_mode!r}"
            )
        if self.gripper_relative_threshold is not None and not (
            0.0 < float(self.gripper_relative_threshold) < 1.0
        ):
            raise ValueError(
                "gripper_relative_threshold must be in (0, 1), "
                f"got {self.gripper_relative_threshold!r}"
            )
        self._validate_reset_config()

    def _validate_reset_config(self) -> None:
        if self.reset_fps <= 0:
            raise ValueError("reset_fps must be positive")
        if (
            self.reset_min_duration_s <= 0
            or self.reset_max_duration_s < self.reset_min_duration_s
        ):
            raise ValueError(
                "reset durations must satisfy "
                "0 < reset_min_duration_s <= reset_max_duration_s"
            )
        if (
            self.reset_max_velocity_deg_s <= 0
            or self.reset_max_acceleration_deg_s2 <= 0
            or self.reset_max_step_deg <= 0
        ):
            raise ValueError(
                "reset velocity, acceleration, and per-frame step limits "
                "must be positive"
            )
        if self.reset_feedback_interval_frames <= 0:
            raise ValueError("reset_feedback_interval_frames must be positive")
        if (
            self.reset_max_tracking_error_deg <= 0
            or self.reset_final_tolerance_deg <= 0
        ):
            raise ValueError("reset tracking and final tolerances must be positive")
        if self.reset_final_hold_frames < 0:
            raise ValueError("reset_final_hold_frames must be non-negative")
        if self.reset_gripper_release_settle_s < 0:
            raise ValueError("reset_gripper_release_settle_s must be non-negative")


class DobotEnv(gym.Env):
    """Dobot CR5AF 6-DOF robot environment with joint + Cartesian dual-mode control.

    Action / state layout depends on ``action_mode`` / ``state_mode``:

    - **joint mode** (default): action 7-dim ``[j1..j6 rad, gripper]``,
      state 7-dim (same layout). Policy: ``pi05_dobot_joint``.
    - **cartesian / pose mode**: action 8-dim ``[x,y,z m, qw,qx,qy,qz, gripper]``,
      state 8-dim (same layout) + ``prev_state``. Policy: ``pi05_dobot_pose``.

    Reward is computed via a learned reward model (if ``use_reward_model``),
    Cartesian distance to ``target_ee_pose`` (if ``use_dense_reward``), or sparse.
    """

    def __init__(
        self,
        config: Optional[DobotRobotConfig] = None,
        override_cfg: Optional[dict] = None,
        worker_info: Optional[WorkerInfo] = None,
        hardware_info=None,
        env_idx: int = 0,
        env_cfg=None,
        **kwargs,
    ):
        self._logger = get_logger()
        self.config = config if config is not None else DobotRobotConfig()
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
        self._reward_worker = None
        self._servo_rejected_count = 0
        self._state = None
        self._prev_state: Optional[np.ndarray] = None
        self._terminal_reward_computed = False

        # Merge hardware_info connection fields into config (independent of
        # is_dummy so dummy-mode tests with hardware_info also see the values).
        self._merge_hardware_info()

        # Re-validate after all overrides (env_cfg.init_params, override_cfg,
        # hardware_info) have been applied — __post_init__ ran before them.
        self._validate_config()
        self._gripper_binarizer = (
            RelativeGripperBinarizer(self.config.gripper_relative_threshold)
            if self.config.gripper_relative_threshold is not None
            else None
        )
        self._skip_gripper_binarizer = False

        if not self.config.is_dummy:
            self._setup_hardware()
        if self.config.use_reward_model:
            self._setup_reward_worker()

        if self.config.camera_serials is None:
            self.config.camera_serials = []
        if not self.config.camera_serials:
            self._logger.info(
                "No camera serials configured. Observations will use dummy frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the arm to be servo-ready.
        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time:.0f}s for Dobot to be ready."
                )
                break

        # Use the same guarded ServoJ-only reset path at startup and between
        # episodes. Keeping a single entry point prevents the constructor from
        # drifting back to legacy fixed-step reset arguments.
        self.go_to_rest()

        # Read initial state (and prev_state for pose mode).
        self._state, self._prev_state = self._read_state_and_prev()

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    # ── Setup ────────────────────────────────────────────────────────────────

    def _merge_hardware_info(self):
        """Merge connection fields from ``hardware_info.config`` into ``self.config``.

        Runs in both dummy and real mode. ``action_mode`` / ``state_mode`` are
        only overridden from hardware_info if the config still has its default
        (so env-level override_cfg takes precedence for mode selection).
        """
        if self.hardware_info is None:
            return
        hw = self.hardware_info.config
        for key in (
            "ip",
            "speed",
            "user_index",
            "tool_index",
            "gripper_port",
            "enable_gripper",
            "gripper_closed_deg",
            "gripper_open_deg",
            "enable_ft_sensor",
            "camera_serials",
            "camera_type",
            "camera_resolution",
            "camera_fps",
            "camera_fourcc",
        ):
            if hasattr(hw, key) and getattr(hw, key, None) is not None:
                setattr(self.config, key, getattr(hw, key))
        # action_mode / state_mode: only override if config still at default.
        _defaults = DobotRobotConfig()
        if self.config.action_mode == _defaults.action_mode and hasattr(
            hw, "action_mode"
        ):
            self.config.action_mode = hw.action_mode
        if self.config.state_mode == _defaults.state_mode and hasattr(hw, "state_mode"):
            self.config.state_mode = hw.state_mode

    def _validate_config(self):
        """Re-validate config after all overrides have been applied."""
        c = self.config
        if c.action_mode not in DobotRobotConfig._VALID_ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {DobotRobotConfig._VALID_ACTION_MODES}, "
                f"got {c.action_mode!r}"
            )
        if c.state_mode not in DobotRobotConfig._VALID_STATE_MODES:
            raise ValueError(
                f"state_mode must be one of {DobotRobotConfig._VALID_STATE_MODES}, "
                f"got {c.state_mode!r}"
            )
        if c.state_mode == "pose" and c.action_mode != "cartesian":
            raise ValueError(
                "state_mode='pose' requires action_mode='cartesian': "
                "位姿策略输出 8 维绝对位姿动作，必须由 ServoP 执行"
            )
        if c.reward_mode not in DobotRobotConfig._VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of "
                f"{DobotRobotConfig._VALID_REWARD_MODES}, "
                f"got {c.reward_mode!r}"
            )
        if c.gripper_relative_threshold is not None and not (
            0.0 < float(c.gripper_relative_threshold) < 1.0
        ):
            raise ValueError(
                "gripper_relative_threshold must be in (0, 1), "
                f"got {c.gripper_relative_threshold!r}"
            )
        c._validate_reset_config()
        resolution = tuple(int(v) for v in c.camera_resolution)
        if len(resolution) != 2 or any(v <= 0 for v in resolution):
            raise ValueError(
                f"camera_resolution must contain two positive integers, "
                f"got {c.camera_resolution!r}"
            )
        if int(c.camera_fps) <= 0:
            raise ValueError(f"camera_fps must be positive, got {c.camera_fps!r}")

    def _setup_hardware(self):
        """Launch the controller (real mode only)."""
        from .dobot_controller import DobotController

        controller_node_rank = (
            getattr(self.hardware_info.config, "controller_node_rank", None)
            if self.hardware_info is not None
            else None
        )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self._controller = DobotController.launch_controller(
            ip=self.config.ip or "192.168.5.1",
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            speed=self.config.speed,
            user_index=self.config.user_index,
            tool_index=self.config.tool_index,
            action_mode=self.config.action_mode,
            state_mode=self.config.state_mode,
            gripper_port=self.config.gripper_port,
            enable_gripper=self.config.enable_gripper,
            gripper_closed_deg=self.config.gripper_closed_deg,
            gripper_open_deg=self.config.gripper_open_deg,
            enable_ft_sensor=self.config.enable_ft_sensor,
            payload=self.config.payload,
        )

    def _setup_reward_worker(self):
        """Launch the embodied reward worker if reward model is enabled."""
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
        """Initialize action and observation spaces based on action_mode/state_mode."""
        self._joint_limit_low = np.array(self.config.joint_limit_low, dtype=np.float64)
        self._joint_limit_high = np.array(
            self.config.joint_limit_high, dtype=np.float64
        )

        # Action space: 7-dim (joint) or 8-dim (cartesian).
        if self.config.action_mode == "cartesian":
            # [x,y,z m, qw,qx,qy,qz, gripper 0..1]
            action_low = np.array([-2, -2, -2, -1, -1, -1, -1, 0.0], dtype=np.float32)
            action_high = np.array([2, 2, 2, 1, 1, 1, 1, 1.0], dtype=np.float32)
        else:
            # [j1..j6 rad, gripper 0..1]
            action_low = np.append(self._joint_limit_low, 0.0).astype(np.float32)
            action_high = np.append(self._joint_limit_high, 1.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        # Observation space: state (7 or 8-dim) + frames (cameras).
        is_pose = self.config.state_mode == "pose"
        state_dim = 8 if is_pose else 7

        if is_pose:
            # Pose mode: single 8-dim block [x,y,z, qw,qx,qy,qz, gripper].
            state_spaces = {
                "ee_pose_state": gym.spaces.Box(-np.inf, np.inf, shape=(state_dim,)),
            }
        else:
            # Joint mode: arm_joint_position(6) + gripper_position(1) = 7-dim total.
            # RealWorldEnv concatenates sorted state dict values into `states`,
            # so the policy sees a 7-dim vector. This matches rebot's layout.
            state_spaces = {
                "arm_joint_position": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_position": gym.spaces.Box(0.0, 1.0, shape=(1,)),
            }

        # Frames: cam_left_wrist always present; cam_high optional.
        frame_spaces = {}
        available_cameras = self._available_camera_names()
        for cam_name in available_cameras:
            frame_spaces[cam_name] = gym.spaces.Box(
                0, 255, shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
            )
        if not frame_spaces:
            # Dummy frame space for gym compatibility when no cameras.
            frame_spaces["cam_left_wrist"] = gym.spaces.Box(
                0, 255, shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
            )

        observation_spaces = {
            "state": gym.spaces.Dict(state_spaces),
            "frames": gym.spaces.Dict(frame_spaces),
        }
        if is_pose:
            observation_spaces["prev_state"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(state_dim,), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _available_camera_names(self) -> list[str]:
        """Return the list of camera names based on config."""
        names = ["cam_left_wrist"]
        if self.config.enable_high_camera:
            names.append("cam_high")
        return names

    def _initial_joints_rad(self) -> np.ndarray:
        """Extract the 6 joint angles (rad) from ``initial_joint_pos`` (7-dim)."""
        jp = list(self.config.initial_joint_pos)
        if len(jp) < 6:
            jp = jp + [0.0] * (6 - len(jp))
        return np.asarray(jp[:6], dtype=np.float64)

    def _prepare_executed_action(self, action: np.ndarray) -> np.ndarray:
        """Return the clipped command that will be sent to the controller.

        The input action is never mutated. When relative gripper binarization is
        enabled, only the final gripper component is changed; arm components
        remain continuous. Set ``skip_gripper_binarizer=True`` (via
        :meth:`set_gripper_bypass`) to pass the gripper value through directly —
        used by the HIL keyboard wrapper to issue absolute open/close commands.
        """
        executed_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            self.action_space.low,
            self.action_space.high,
        ).copy()
        if self._gripper_binarizer is None or self._skip_gripper_binarizer:
            self._skip_gripper_binarizer = False
            # Sync binarizer state so MODEL→ENGAGE→MODEL transitions don't
            # produce stale latch decisions.
            if self._gripper_binarizer is not None:
                self._gripper_binarizer._state = float(executed_action[-1])
            return executed_action

        if self._state is None:
            current_gripper = float(self.config.initial_joint_pos[-1])
        else:
            current_gripper = float(self._state.gripper_position)
        executed_action[-1] = self._gripper_binarizer.map(
            output_norm=float(executed_action[-1]),
            current_norm=current_gripper,
        )
        return executed_action

    # ── Core gym API ─────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: ``(7,)`` (joint) or ``(8,)`` (cartesian) float array.
        """
        start_time = time.time()

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # Safety: hard-reject NaN/Inf — do NOT send anything to the robot.
        # Replacing with zeros/ones is unsafe (non-physical targets); the
        # correct behavior is to skip the entire frame (no arm, no gripper).
        if not np.all(np.isfinite(action)):
            self._logger.error(
                f"Non-finite action (NaN/Inf) — hard-rejecting frame, "
                f"NO motion sent to robot. action={action}"
            )
            self._num_steps += 1
            step_time = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
            if not self.config.is_dummy:
                self._state, self._prev_state = self._read_state_and_prev()
            observation = self._get_observation()
            reward = self._calc_step_reward(
                observation, is_gripper_action_effective=False
            )
            terminated = False
            if self.config.reward_mode != "terminal":
                terminated = (reward >= 1.0) and (
                    self._success_hold_counter >= self.config.success_hold_steps
                )
            truncated = self._num_steps >= self.config.max_num_steps
            return (
                observation,
                reward,
                terminated,
                truncated,
                {
                    "action_command_accepted": False,
                    "action_rejection_reason": "non_finite_action",
                },
            )

        executed_action = self._prepare_executed_action(action)

        is_gripper_effective = False
        accepted = True
        if not self.config.is_dummy:
            # Periodic RobotMode health check (throttled to ~1 Hz).
            self._check_robot_health()

            accepted = self._controller.send_action(
                executed_action, action_mode=self.config.action_mode
            ).wait()[0]
            if not accepted:
                self._servo_rejected_count += 1
                if self._servo_rejected_count % 30 == 0:
                    self._logger.warning(
                        "Dobot servo rejected "
                        f"{self._servo_rejected_count} consecutive frames"
                    )
            else:
                self._servo_rejected_count = 0

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._state, self._prev_state = self._read_state_and_prev()

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_effective)

        terminated = False
        if self.config.reward_mode != "terminal":
            terminated = (reward >= 1.0) and (
                self._success_hold_counter >= self.config.success_hold_steps
            )
        truncated = self._num_steps >= self.config.max_num_steps

        info = {
            "executed_action": executed_action.copy(),
            "action_command_accepted": bool(accepted),
        }
        if not accepted:
            info["action_rejection_reason"] = "safety_guard"
        return observation, reward, terminated, truncated, info

    @property
    def num_steps(self):
        return self._num_steps

    def reset(self, seed=None, options=None):
        """Reset the environment to the rest configuration."""
        self._num_steps = 0
        self._success_hold_counter = 0
        self._terminal_reward_computed = False
        if self.config.is_dummy:
            if self._gripper_binarizer is not None:
                self._gripper_binarizer.reset()
            return self._get_observation(), {}

        self._success_hold_counter = 0
        self.go_to_rest()
        self._num_steps = 0

        # Clear pose tracker so first frame's prev_state = itself.
        self._controller.reset_pose_tracker().wait()
        self._state, self._prev_state = self._read_state_and_prev()
        if self._gripper_binarizer is not None:
            self._gripper_binarizer.reset()
        return self._get_observation(), {}

    def go_to_rest(self):
        """Release the gripper, then reset with a guarded ServoJ trajectory."""
        self._controller.assert_ready().wait()
        self._controller.open_gripper().wait()
        self._gripper_is_open = True
        time.sleep(float(self.config.reset_gripper_release_settle_s))
        self._controller.reset_to_pose(
            self._initial_joints_rad(),
            reset_fps=self.config.reset_fps,
            min_duration_s=self.config.reset_min_duration_s,
            max_duration_s=self.config.reset_max_duration_s,
            max_velocity_deg_s=self.config.reset_max_velocity_deg_s,
            max_acceleration_deg_s2=self.config.reset_max_acceleration_deg_s2,
            max_step_deg=self.config.reset_max_step_deg,
            feedback_interval_frames=self.config.reset_feedback_interval_frames,
            max_tracking_error_deg=self.config.reset_max_tracking_error_deg,
            final_tolerance_deg=self.config.reset_final_tolerance_deg,
            final_hold_frames=self.config.reset_final_hold_frames,
        ).wait()
        time.sleep(0.5)

    def _check_robot_health(self):
        """Throttled (~1 Hz) RobotMode check.

        ServoJ/ServoP **silently fail** under E-stop / alarm / disable / pause,
        while feedback reads still succeed. This is the only way to detect them.
        Raises RuntimeError if the arm is not in a servo-ready mode.
        """
        now = time.monotonic()
        if (
            hasattr(self, "_last_health_check_ts")
            and now - self._last_health_check_ts < 1.0
        ):
            return
        self._last_health_check_ts = now
        if not self._controller.is_robot_up().wait()[0]:
            raise RuntimeError(
                "Dobot not in servo-ready mode (RobotMode not in {5,7,8}). "
                "Possible E-stop / alarm / disabled."
            )

    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions ``[j1..j6 rad, gripper norm]`` (7-dim).

        Required by HIL keyboard intervention wrapper and the collector.
        """
        if self.config.is_dummy:
            return np.zeros(7, dtype=np.float64)
        state = self._controller.get_state().wait()[0]
        return np.append(
            np.asarray(state.arm_joint_position, dtype=np.float64),
            float(state.gripper_position),
        )

    def get_pose_state(self) -> np.ndarray:
        """Return fresh ``[x,y,z,qw,qx,qy,qz,gripper]`` without touching the prev tracker.

        Reads the current TCP pose + gripper via ``controller.get_state()``
        (which does **not** update :class:`PoseStateTracker`). This keeps the
        observation ``prev_state`` semantics intact while giving the HIL
        wrapper a safe initial target for ENGAGE.

        Raises:
            RuntimeError: if ``state_mode != "pose"``.
        """
        if self.config.state_mode != "pose":
            raise RuntimeError("get_pose_state requires state_mode='pose'")
        if self.config.is_dummy:
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
        state = self._controller.get_state().wait()[0]
        pose = np.asarray(state.tcp_pose, dtype=np.float64).reshape(7)
        return np.concatenate([pose, [float(state.gripper_position)]])

    def reset_servo_smoothing(self) -> None:
        """Restart slew smoothing from current feedback at a control-owner switch.

        Delegates to ``controller.engage()`` (which calls
        ``_robot.reset_smoothing()``). In dummy mode this is a no-op.
        """
        if not self.config.is_dummy:
            self._controller.engage().wait()

    def set_gripper_bypass(self, skip: bool) -> None:
        """Skip the RelativeGripperBinarizer for the next step.

        Used by the HIL keyboard wrapper to issue absolute gripper commands
        (0/1) that should not be re-latched by the relative binarizer. The flag
        is auto-consumed after one step.
        """
        self._skip_gripper_binarizer = bool(skip)

    def _read_state_and_prev(self) -> tuple:
        """Read (state, prev_state) from the controller.

        Returns:
            ``(DobotRobotState, prev_state|None)`` — prev_state is the 8-dim
            pose state in pose mode (first frame = itself), else None.
        """
        if self.config.is_dummy:
            return None, None
        return self._controller.get_state_and_prev_state().wait()[0]

    # ── Reward ───────────────────────────────────────────────────────────────

    def _calc_step_reward(
        self,
        observation: dict,
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute reward from reward model / geometry / sparse."""
        if self.config.is_dummy and not self.config.use_reward_model:
            return 0.0

        if self.config.use_reward_model:
            if self.config.reward_mode == "terminal":
                if (
                    self._num_steps >= self.config.max_num_steps
                    and not self._terminal_reward_computed
                ):
                    self._terminal_reward_computed = True
                    reward = self._compute_reward_model(observation)
                    return max(0.0, min(1.0, float(reward)))
                return 0.0

            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty
            return max(0.0, min(1.0, float(reward)))

        if self._state is None:
            return 0.0

        # Legacy geometry-based reward using TCP pose.
        tcp = np.asarray(self._state.tcp_pose, dtype=np.float64)
        # tcp_pose layout: [x,y,z, qw,qx,qy,qz] (w-first).
        # scipy's from_quat expects [qx,qy,qz,qw] — reorder.
        quat_xyzw = tcp[3:].copy()
        quat_xyzw = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])
        # as_euler returns radians; convert to degrees to match target_ee_pose
        # (which uses degrees in the Euler components, per YAML convention).
        euler_angles = np.degrees(np.abs(R.from_quat(quat_xyzw).as_euler("xyz")))
        position = np.hstack([tcp[:3], euler_angles])
        target_delta = np.abs(position - self.config.target_ee_pose)

        # Check BOTH position (first 3) and orientation (last 3) against thresholds.
        is_in_target_zone = np.all(target_delta <= self.config.reward_threshold)

        if is_in_target_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(np.exp(-500.0 * np.sum(np.square(target_delta[:3]))))
            else:
                reward = 0.0

        if self.config.enable_gripper_penalty and is_gripper_action_effective:
            reward -= self.config.gripper_penalty

        reward = max(0.0, min(1.0, reward))
        return reward

    def _compute_reward_model(self, observation: dict[str, Any]) -> float:
        """Run reward model inference on the current camera frame."""
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
        reward_observations = {
            "main_images": image_batch,
        }
        reward_output = self._reward_worker.compute_image_rewards(
            reward_observations
        ).wait()[0]
        if hasattr(reward_output, "detach"):
            reward_output = reward_output.detach().cpu().numpy()
        reward_array = np.asarray(reward_output).reshape(-1)
        return float(reward_array[0])

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            if not frames:
                frames = {
                    "cam_left_wrist": np.zeros(
                        (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
                    )
                }
            state = self._build_state_dict()
            obs = {"state": state, "frames": frames}
            # Forward prev_state for pose-mode envs (consumed by DeltaPose/AbsolutePose
            # via RealWorldEnv._wrap_obs → obs_processor → policy transform).
            if self.config.state_mode == "pose" and self._prev_state is not None:
                obs["prev_state"] = np.asarray(self._prev_state, dtype=np.float32)
            return copy.deepcopy(obs)
        obs = self._base_observation_space.sample()
        if self.config.state_mode == "pose":
            # Keep dummy pose observations structurally equivalent to real
            # observations. The first frame uses itself as prev_state.
            state = np.array(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
                dtype=np.float32,
            )
            obs["state"]["ee_pose_state"] = state.copy()
            obs["prev_state"] = state.copy()
        return obs

    def _build_state_dict(self) -> dict:
        """Build the ``state`` sub-dict from ``self._state`` + ``self._prev_state``."""
        if self._state is None:
            return self.observation_space["state"].sample()

        is_pose = self.config.state_mode == "pose"
        if is_pose:
            from .dobot_pose_obs import pose_state

            ee_state = pose_state(
                np.asarray(self._state.tcp_pose, dtype=np.float32),
                float(self._state.gripper_position),
            )
            return {"ee_pose_state": ee_state}
        # joint mode: arm_joint_position(6) + gripper_position(1) separately.
        # RealWorldEnv concatenates sorted dict values → 7-dim `states`.
        joints = np.asarray(self._state.arm_joint_position, dtype=np.float32)
        gripper = np.array([float(self._state.gripper_position)], dtype=np.float32)
        return {
            "arm_joint_position": joints,
            "gripper_position": gripper,
        }

    # ── Cameras ──────────────────────────────────────────────────────────────

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        if not self.config.camera_serials:
            return
        camera_type = self.config.camera_type or "realsense"
        for i, serial in enumerate(self.config.camera_serials):
            cam_name = (
                f"wrist_{i + 1}"
                if i > 0 or not self.config.enable_high_camera
                else "cam_high"
            )
            if i == 0 and self.config.enable_high_camera:
                cam_name = "cam_high"
            elif i == 1 or (i == 0 and not self.config.enable_high_camera):
                cam_name = "cam_left_wrist"
            info = CameraInfo(
                name=cam_name,
                serial_number=serial,
                camera_type=camera_type,
                resolution=tuple(int(v) for v in self.config.camera_resolution),
                fps=int(self.config.camera_fps),
                fourcc=self.config.camera_fourcc,
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
        cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
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

    # ── Utilities ────────────────────────────────────────────────────────────

    def close(self):
        """Release cameras, video player, and the robot controller."""
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if not self.config.is_dummy and hasattr(self, "_cameras"):
            self._close_cameras()
        # Disable + disconnect the arm (stops motion, releases TCP).
        if not self.config.is_dummy and hasattr(self, "_controller"):
            try:
                self._controller.close().wait()[0]
            except Exception as e:
                self._logger.warning(f"Error closing DobotController: {e}")
        super().close()

    @property
    def task_description(self) -> str:
        """Language instruction for this task."""
        return self.config.task_description

    @property
    def target_ee_pose(self) -> np.ndarray:
        """Target EEF pose as ``[x, y, z, qw, qx, qy, qz]`` (w-first)."""
        quat_xyzw = R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat()
        # scipy returns [qx,qy,qz,qw]; convert to w-first [qw,qx,qy,qz].
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([self.config.target_ee_pose[:3], quat_wxyz]).copy()

    @property
    def controller(self):
        """Underlying DobotController (for HIL kinematics access)."""
        return self._controller if hasattr(self, "_controller") else None
