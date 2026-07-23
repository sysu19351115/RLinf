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

"""Dobot CR5AF controller as a distributed :class:`Worker`.

Wraps the ``dobot_zhiyu`` SDK (:class:`DobotRobot`) and the Damiao gripper
(:class:`DamiaoGripper`) behind a unified, normalized-unit RPC interface so it
can be placed on any node in the cluster via Ray. The env (and HIL collector)
call every method through a returned Future (``.wait()``).

Unit boundary (conversions happen **only** here, once):
    - Joints: **radians** in / out (SDK speaks degrees internally).
    - Pose:   **meters + quaternion ``[x, y, z, qw, qx, qy, qz]``** (w-first).
    - Gripper: **normalized ``[0, 1]``** (0=closed, 1=open; sign-flipped vs motor).

All SDK / scipy / motorbridge imports are deferred to :meth:`__init__` so this
module can be imported on GPU-only nodes that lack the robot SDK.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .dobot_robot_state import DobotRobotState

_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi
# RobotMode 中允许继续伺服的状态：5=使能空闲、7=运动中、8=单步运动。
_SERVO_READY_MODES = frozenset({5, 7, 8})


class DobotController(Worker):
    """Dobot CR5AF arm + Damiao gripper controller.

    Wraps the ``dobot_zhiyu`` SDK (TCP) and the Damiao motor gripper
    (motorbridge, FORCE_POS mode) as a distributed :class:`Worker` so it can be
    placed on any node in the cluster.

    Every public method returns a Future (call ``.wait()`` on the env side).
    """

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    @staticmethod
    def launch_controller(
        ip: str = "192.168.5.1",
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        **dobot_kwargs,
    ):
        """Launch a :class:`DobotController` on the specified node.

        Args:
            ip: Dobot controller IP.
            env_idx: Environment index for naming.
            node_rank: Cluster node rank to place on.
            worker_rank: Worker rank for naming.
            **dobot_kwargs: Forwarded to :meth:`DobotController.__init__`.

        Returns:
            The launched remote controller instance.
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return DobotController.create_group(ip, **dobot_kwargs).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"DobotController-{worker_rank}-{env_idx}",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        ip: str = "192.168.5.1",
        speed: int = 100,
        user_index: int = 0,
        tool_index: int = 0,
        action_mode: str = "joint",
        state_mode: str = "joint",
        gripper_port: str = "/dev/ttyACM0",
        gripper_norm: bool = True,
        enable_gripper: bool = True,
        gripper_required: bool = False,
        gripper_closed_deg: float = 0.0,
        gripper_open_deg: float = -320.0,
        gripper_max_velocity_rad_s: float = 10.0,
        gripper_force_ratio: float = 0.05,
        gripper_home_on_start: bool = False,
        payload: Optional[tuple] = None,
        enable_ft_sensor: bool = True,
        **dobot_kwargs,
    ):
        super().__init__()
        self._logger = get_logger()

        # Inject the dobot_zhiyu SDK onto sys.path (it is a git submodule, not
        # pip-installed). Path: rlinf/envs/realworld/dobot/dobot_controller.py
        #   -> up 5 levels = repo root -> third_party/dobot_zhiyu/src
        _repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )
        _sdk_src = os.path.join(_repo_root, "third_party", "dobot_zhiyu", "src")
        if _sdk_src not in sys.path:
            sys.path.insert(0, _sdk_src)

        # Deferred imports: GPU-only nodes must be able to import this module
        # without the robot SDK / motorbridge installed.
        from dobot_control import DobotRobot  # noqa: F401

        from .dobot_action_split import split_follower_action
        from .dobot_pose_obs import PoseStateTracker

        self._split_follower_action = split_follower_action
        self._PoseStateTracker = PoseStateTracker

        self._robot = DobotRobot(
            ip=ip, speed=speed, user=user_index, tool=tool_index, **dobot_kwargs
        )
        # Only import + construct the gripper when it's actually enabled, so
        # nodes without motorbridge can still use the controller with
        # enable_gripper=False.
        if enable_gripper:
            from .dobot_gripper import DamiaoGripper

            self._gripper = DamiaoGripper(
                port=gripper_port,
                closed_deg=gripper_closed_deg,
                open_deg=gripper_open_deg,
                max_velocity_rad_s=gripper_max_velocity_rad_s,
                force_ratio=gripper_force_ratio,
            )
        else:
            self._gripper = None
        self._action_mode = action_mode
        self._state_mode = state_mode
        self._gripper_norm = gripper_norm
        self._enable_gripper = bool(enable_gripper)
        self._gripper_required = bool(gripper_required)
        self._gripper_home_on_start = bool(gripper_home_on_start)
        self._payload = payload
        self._enable_ft_sensor = enable_ft_sensor
        self._enabled = False
        self._follower_engaged = False
        # PoseStateTracker for pose-mode prev_state (reset in reset_pose_tracker).
        self._pose_tracker = PoseStateTracker()

        # Connect + enable immediately (matching rebot's connect-in-__init__ pattern).
        # This runs on the remote Ray actor; by the time launch_controller returns,
        # the arm is connected, enabled, and ready for servo.
        self.enable()

    # ------------------------------------------------------------------
    # Enable / disable / close
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Connect + enable the arm + (optional) payload + FT sensor + gripper."""
        self._robot.connect()
        self._robot.enable()
        if self._payload is not None:
            load, x, y, z = self._payload
            self._robot.set_payload(load, x, y, z)
        if self._enable_ft_sensor:
            self._robot.enable_ft_sensor(True)
        if self._enable_gripper:
            try:
                self._gripper.connect()
                if self._gripper_home_on_start:
                    self._gripper.home()
                else:
                    current = self._gripper.get_normalized()
                    self._gripper.move_normalized(current)
            except Exception as e:
                try:
                    self._gripper.close()
                except Exception:
                    pass
                if self._gripper_required:
                    try:
                        self._robot.disable_robot()
                    except Exception:
                        pass
                    try:
                        self._robot.close()
                    except Exception:
                        pass
                    raise RuntimeError(f"夹爪初始化失败: {e}") from e
                self._logger.warning(f"夹爪连接失败（继续运行）: {e}")
        self._enabled = True
        self._logger.info(
            f"DobotController connected on '{self._robot.ip}' and enabled "
            f"(action_mode={self._action_mode}, state_mode={self._state_mode})."
        )

    def disable(self) -> None:
        """Disable the arm and gripper (idempotent)."""
        if self._enabled:
            if self._gripper is not None:
                try:
                    self._gripper.disable()
                except Exception:
                    pass
            try:
                self._robot.disable_robot()
            except Exception:
                pass
            self._enabled = False

    def close(self) -> None:
        """Safely shut down the arm + gripper (idempotent)."""
        self._follower_engaged = False
        if self._gripper is not None:
            try:
                self._gripper.close()
            except Exception:
                pass
        try:
            self._robot.close()
        except Exception:
            pass
        self._enabled = False

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_robot_up(self) -> bool:
        """Return ``True`` when the arm is in a servo-ready mode.

        Checks ``RobotMode ∈ {5, 7, 8}`` (enable-idle / running / single-move).
        E-stop / alarm (9) / disabled (4) / paused (10) all cause ServoJ/ServoP
        to **silently fail** while feedback reads still succeed — this is the
        only way to detect them.
        """
        try:
            return self._robot.get_mode() in _SERVO_READY_MODES
        except Exception:
            return False

    def assert_ready(self) -> None:
        """Raise ``RuntimeError`` if the arm is not servo-ready."""
        mode = self._robot.get_mode()
        if mode not in _SERVO_READY_MODES:
            raise RuntimeError(
                f"机械臂不可控：RobotMode={mode}({self._robot._mode_name(mode)})，"
                "可能已按下急停、触发报警或被失能/暂停"
            )

    # ------------------------------------------------------------------
    # Servo control
    # ------------------------------------------------------------------

    def engage(self) -> None:
        """Prepare for a servo stream: reset the slew smoother.

        Must be called before the first ServoJ/ServoP after enable/reset so
        slewing restarts from the current pose.
        """
        self._robot.reset_smoothing()
        self._follower_engaged = True

    def release(self) -> None:
        """Exit a servo stream."""
        self._follower_engaged = False

    def drive_arm_joints(self, q_rad: np.ndarray) -> bool:
        """ServoJ to joint targets. Input: 6 joints in radians. Returns accepted."""
        q_deg = (np.asarray(q_rad, dtype=float).reshape(6) * _RAD2DEG).tolist()
        return self._robot.servo_joints(q_deg)

    def drive_arm_pose(self, pose_quat_m: np.ndarray) -> bool:
        """ServoP to a Cartesian target. Input: ``[x,y,z(m), qw,qx,qy,qz]``."""
        return self._robot.servo_pose(np.asarray(pose_quat_m, dtype=float).reshape(7))

    def drive_gripper(self, norm_01: float) -> None:
        """Drive the gripper to a normalized ``[0,1]`` target (skips if absent)."""
        if self._gripper is None or not self._gripper.enabled:
            return
        if self._gripper_norm:
            self._gripper.move_normalized(float(norm_01))
        else:
            self._gripper.move_to(float(norm_01))

    def send_action(
        self, action: np.ndarray, action_mode: Optional[str] = None
    ) -> bool:
        """Unified drive: dispatch arm by mode + always drive gripper.

        Args:
            action: 7-dim (joint) ``[j1..j6 rad, gripper]`` or 8-dim (cartesian)
                ``[x,y,z m, qw,qx,qy,qz, gripper]``.
            action_mode: Override ``self._action_mode`` for this call (``"joint"`` /
                ``"cartesian"``). Defaults to the configured mode.

        Returns:
            ``True`` if the arm command was accepted (False = safety guard reject).
        """
        mode = action_mode or self._action_mode
        arm, gripper = self._split_follower_action(action, mode)
        if mode == "cartesian":
            accepted = self.drive_arm_pose(arm)
        else:
            accepted = self.drive_arm_joints(arm)
        self.drive_gripper(gripper)
        return bool(accepted)

    # ------------------------------------------------------------------
    # Reset / homing (joint space, non-servo)
    # ------------------------------------------------------------------

    def move_joints(self, q_rad: np.ndarray, duration: float = 3.0) -> None:
        """MovJ to joint targets (radians), blocking until reached.

        Used for reset/homing only — does NOT participate in the servo loop.
        """
        q_deg = (np.asarray(q_rad, dtype=float).reshape(6) * _RAD2DEG).tolist()
        self._robot.move_j(q_deg)
        self._robot.wait_until_reached(q_deg, timeout=max(duration, 5.0))
        self._follower_engaged = False

    def reset_to_pose(
        self,
        joints_rad: np.ndarray,
        init_steps: int = 60,
        init_fps: int = 30,
    ) -> None:
        """Move to a joint reset pose via joint-space linear interpolation.

        Args:
            joints_rad: Target joint positions (6,) in radians.
            init_steps: Number of interpolation steps.
            init_fps: Interpolation frequency (Hz).
        """
        start = np.asarray(self.get_joint_status(), dtype=float)
        target = np.asarray(joints_rad, dtype=float).reshape(6)
        dt = 1.0 / float(init_fps)
        self._robot.reset_smoothing()
        for i in range(1, init_steps + 1):
            alpha = float(i) / float(init_steps)
            q = start + (target - start) * alpha
            q_deg = (q * _RAD2DEG).tolist()
            self._robot.servo_joints(q_deg)
            import time

            time.sleep(dt)
        self._follower_engaged = False

    def reset_pose_tracker(self) -> None:
        """Clear the pose-mode ``prev_state`` tracker (call on env reset)."""
        self._pose_tracker.reset()

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def open_gripper(self) -> None:
        """Open the gripper (normalized 1.0)."""
        if self._gripper is not None and self._gripper.enabled:
            self._gripper.move_normalized(1.0)

    def close_gripper(self) -> None:
        """Close the gripper (normalized 0.0)."""
        if self._gripper is not None and self._gripper.enabled:
            self._gripper.move_normalized(0.0)

    def move_gripper(self, norm_01: float) -> None:
        """Set the gripper target (normalized ``[0,1]``)."""
        self.drive_gripper(norm_01)

    # ------------------------------------------------------------------
    # FT sensor (ForceVLA only)
    # ------------------------------------------------------------------

    def tare_ft_sensor(self) -> None:
        """Zero the six-axis force/torque sensor (tare).

        Must be called with the end-effector hanging free; use the same taring
        procedure for collection and inference so the wrench distribution matches.
        """
        self._robot.six_force_home()

    # ------------------------------------------------------------------
    # State reading (single feedback read for multi-modal)
    # ------------------------------------------------------------------

    def get_joint_status(self) -> np.ndarray:
        """Return 6 joint positions in radians."""
        deg = self._robot.get_joint_positions()
        return np.asarray(deg, dtype=float)[:6] * _DEG2RAD

    def get_gripper_status(self) -> float:
        """Return gripper normalized ``[0,1]`` (or motor rad if gripper_norm=False)."""
        if self._gripper is None or not self._gripper.enabled:
            return 0.0
        if self._gripper_norm:
            return self._gripper.get_normalized()
        return self._gripper.get_position()

    def get_end_pose(self) -> np.ndarray:
        """Return TCP pose ``[x,y,z(m), qw,qx,qy,qz]`` (7,)."""
        return np.asarray(self._robot.get_tcp_pose(), dtype=float).reshape(7)

    def get_state_and_pose(self) -> tuple:
        """Single feedback read returning ``(state[7], pose[7])``.

        ``state = [j1..j6 rad, gripper norm]``; ``pose = [x,y,z m, qw,qx,qy,qz]``.
        """
        st = self._robot.read_state_from_feedback(
            include_joint=True, include_pose=True, include_wrench=False
        )
        joints_rad = np.asarray(st["joint_positions"], dtype=float)[:6] * _DEG2RAD
        pose = np.asarray(st["tcp_pose"], dtype=float).reshape(7)
        state = np.concatenate([joints_rad, [self.get_gripper_status()]])
        return state, pose

    def get_state_pose_wrench(self) -> tuple:
        """Single feedback read returning ``(state[7], pose[7], wrench[6], online)``."""
        st = self._robot.read_state_from_feedback(
            include_joint=True, include_pose=True, include_wrench=True
        )
        joints_rad = np.asarray(st["joint_positions"], dtype=float)[:6] * _DEG2RAD
        pose = np.asarray(st["tcp_pose"], dtype=float).reshape(7)
        state = np.concatenate([joints_rad, [self.get_gripper_status()]])
        wrench = np.asarray(st["wrench"], dtype=float)[:6]
        return state, pose, wrench, bool(st["wrench_online"])

    def get_wrench(self) -> tuple:
        """Return ``(wrench[6], online)``; wrench holds last valid frame on bad data."""
        wrench, online = self._robot.get_wrench()
        return np.asarray(wrench, dtype=float)[:6], bool(online)

    def get_state(self) -> DobotRobotState:
        """Read the current state and return a :class:`DobotRobotState` snapshot.

        Uses a single feedback read for joints + pose (+ wrench if FT enabled).
        """
        if self._enable_ft_sensor:
            state, pose, wrench, online = self.get_state_pose_wrench()
        else:
            state, pose = self.get_state_and_pose()
            wrench, online = None, False
        joints_rad = state[:6]
        gripper_norm = float(state[6])
        return DobotRobotState(
            arm_joint_position=joints_rad,
            tcp_pose=pose,
            gripper_position=gripper_norm,
            gripper_open=gripper_norm >= 0.5,
            action_mode=self._action_mode,
            servo_accepted=True,
            wrench=wrench,
            wrench_online=online,
        )

    def get_state_and_prev_state(self) -> tuple:
        """Read state + return ``(DobotRobotState, prev_state|None)``.

        In pose mode, ``prev_state`` is the previous frame's 8-dim pose state
        (first frame = itself). In joint mode, returns ``None`` (no prev_state needed).
        """
        st = self.get_state()
        if self._state_mode != "pose":
            return st, None
        # Build 8-dim pose state [x,y,z, qw,qx,qy,qz, gripper]
        from .dobot_pose_obs import pose_state

        cur = pose_state(st.tcp_pose, st.gripper_position)
        prev = self._pose_tracker.update(cur)
        return st, prev

    # ------------------------------------------------------------------
    # Drag teaching (collection use)
    # ------------------------------------------------------------------

    def start_drag(self) -> None:
        """Enter joint drag mode (human pushes the arm to demonstrate)."""
        self._robot.start_drag()

    def stop_drag(self) -> None:
        """Exit drag mode."""
        self._robot.stop_drag()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def action_mode(self) -> str:
        return self._action_mode

    @property
    def state_mode(self) -> str:
        return self._state_mode

    @property
    def has_gripper(self) -> bool:
        return self._gripper is not None and self._gripper.enabled

    @property
    def follower_engaged(self) -> bool:
        return self._follower_engaged
