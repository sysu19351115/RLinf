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

"""Dobot keyboard-based 6D end-effector intervention wrapper.

Unlike :class:`SO101KeyboardIntervention` (which solves IK from a 6D-EE target
to joint space), this wrapper operates directly in Cartesian space: the target
is an 8-dim absolute pose ``[x, y, z, qw, qx, qy, qz, gripper]`` and no inverse
kinematics is needed. This matches the Dobot pose policy's action space.

In ENGAGE mode the operator drives the TCP position (base frame XYZ),
orientation (tool-frame roll/pitch/yaw), and gripper (normalized [0, 1]) with
the keyboard. The wrapper returns the 8-dim absolute pose action directly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.utils.logging import get_logger


class _DummyKeyboardListener:
    """No-op keyboard listener used in dummy mode when evdev is unavailable."""

    def pop_pressed_keys(self) -> list[str]:
        return []

    def get_key(self) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Quaternion helpers (env/wrapper/dataset always use wxyz; scipy uses xyzw).
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _normalize_same_hemisphere(new_q: np.ndarray, reference_q: np.ndarray) -> np.ndarray:
    """Normalize *new_q* and flip sign to stay on the same hemisphere as *reference_q*."""
    new_q = np.asarray(new_q, dtype=np.float64)
    norm = np.linalg.norm(new_q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid target quaternion (zero or non-finite norm).")
    new_q = new_q / norm
    if np.dot(new_q, reference_q) < 0:
        new_q = -new_q
    return new_q


class DobotKeyboardIntervention(gym.ActionWrapper):
    """Keyboard 6D-EE Cartesian intervention for the Dobot CR5AF robot.

    Operates in ``state_mode='pose'`` / ``action_mode='cartesian'`` only.
    The action is an 8-dim absolute pose::

        [x, y, z, qw, qx, qy, qz, gripper]

    Args:
        env: Wrapped Dobot environment. Must expose ``get_pose_state()`` and
            ``reset_servo_smoothing()`` via the unwrapped env.
        position_delta: Position increment per key event (meters).
        rotation_delta: Rotation increment per key event (radians).
        gripper_delta: Gripper increment per key event (normalized [0, 1]).
        workspace_low: Lower XYZ bounds of the safe workspace (3-dim).
        workspace_high: Upper XYZ bounds of the safe workspace (3-dim).
        toggle_key: Key that toggles MODEL / ENGAGE.
        model_key: Key that forces return to MODEL.
        done_key: Key that ends the current episode (saves and starts next).
        abort_key: Key that discards the current episode (resets without saving).
        quit_keys: Keys that discard the current episode and exit the program.
        start_in_engage: If ``True``, begin in ENGAGE instead of MODEL.
        listener: Optional pre-constructed keyboard listener (for testing).
            ``None`` creates a real :class:`KeyboardListener` (or a dummy one
            in dummy mode when evdev is unavailable).
    """

    def __init__(
        self,
        env: gym.Env,
        position_delta: float = 0.002,
        rotation_delta: float = 0.02,
        gripper_delta: float = 0.05,
        workspace_low: np.ndarray | None = None,
        workspace_high: np.ndarray | None = None,
        base_frame_euler_deg: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        toggle_key: str = "h",
        model_key: str = "m",
        done_key: str = "Key.enter",
        abort_key: str = "Key.backspace",
        quit_keys: tuple[str, ...] = ("Key.esc",),
        start_in_engage: bool = False,
        listener=None,
    ):
        super().__init__(env)

        # ── Validate env mode ────────────────────────────────────────────────
        config = getattr(self.unwrapped, "config", None)
        if config is None:
            raise ValueError("DobotKeyboardIntervention requires a DobotEnv with a config.")
        if getattr(config, "action_mode", None) != "cartesian":
            raise ValueError(
                "DobotKeyboardIntervention requires action_mode='cartesian', "
                f"got action_mode={config.action_mode!r}."
            )
        if getattr(config, "state_mode", None) != "pose":
            raise ValueError(
                "DobotKeyboardIntervention requires state_mode='pose', "
                f"got state_mode={config.state_mode!r}."
            )

        # ── Validate deltas ─────────────────────────────────────────────────
        if position_delta <= 0:
            raise ValueError(f"position_delta must be positive, got {position_delta}.")
        if rotation_delta <= 0:
            raise ValueError(f"rotation_delta must be positive, got {rotation_delta}.")
        if gripper_delta <= 0:
            raise ValueError(f"gripper_delta must be positive, got {gripper_delta}.")

        # ── Validate workspace (optional software safety layer) ──────────────
        # The Dobot controller's SDK already enforces a hard max-jump guard
        # (max_jump_m, default 10 cm) and a slew rate limiter, which reject
        # out-of-bound commands at the hardware level. The software workspace
        # clamp here is an ADDITIONAL layer that prevents slow drift to the
        # workspace edge under continuous key presses (each step is small
        # enough to pass max_jump_m, but accumulates). It is OPTIONAL: leave
        # both bounds None to rely solely on the hardware guard + operator
        # E-stop. Provide both bounds to enable the clamp.
        if (workspace_low is None) != (workspace_high is None):
            raise ValueError(
                "workspace_low and workspace_high must be both set or both None."
            )
        if workspace_low is not None:
            self._workspace_low = np.asarray(workspace_low, dtype=np.float64).reshape(3)
            self._workspace_high = np.asarray(workspace_high, dtype=np.float64).reshape(3)
            if self._workspace_low.shape != (3,) or self._workspace_high.shape != (3,):
                raise ValueError("workspace_low and workspace_high must be 3-dim.")
            if np.any(self._workspace_low >= self._workspace_high):
                raise ValueError(
                    f"workspace_low {self._workspace_low} must be strictly less than "
                    f"workspace_high {self._workspace_high} in all dimensions."
                )
        else:
            self._workspace_low = None
            self._workspace_high = None

        self.position_delta = float(position_delta)
        self.rotation_delta = float(rotation_delta)
        self.gripper_delta = float(gripper_delta)
        self.toggle_key = toggle_key
        self.model_key = model_key
        self.done_key = done_key
        self.abort_key = abort_key
        self.quit_keys = tuple(quit_keys)
        self._start_in_engage = bool(start_in_engage)

        # ── Base-frame rotation (physical → base frame) ──────────────────────
        # Adapts keyboard translation for non-standard robot mounting.
        # Rotates keyboard XYZ increments by the configured Euler angles
        # (xyz intrinsic order, degrees: [rx, ry, rz]) so that operator-intuitive
        # directions map to the correct base-frame axes. Tool-frame rotations
        # (i/k/j/l/u/o) are unaffected (they use right-multiply in the tool
        # frame, independent of the base frame).
        euler = np.asarray(base_frame_euler_deg, dtype=np.float64).reshape(3)
        self._base_frame_euler_deg = euler
        self._base_rotation = Rotation.from_euler(
            "xyz", euler, degrees=True
        ).as_matrix()

        # ── Keyboard listener ────────────────────────────────────────────────
        if listener is not None:
            self.listener = listener
        else:
            is_dummy = getattr(config, "is_dummy", False)
            try:
                self.listener = KeyboardListener()
            except Exception as exc:
                if is_dummy:
                    get_logger().warning(
                        "[DobotKeyboardIntervention] KeyboardListener failed in "
                        "dummy mode: %s. Using dummy keyboard listener.",
                        exc,
                    )
                    self.listener = _DummyKeyboardListener()
                else:
                    raise

        # ── State machine ─────────────────────────────────────────────────────
        self._state = "engage" if self._start_in_engage else "model"
        self._last_key_press_ts: dict[str, float] = {}
        self._key_debounce_s = 0.12
        self._target_log_interval_s = 1.0
        self._last_target_log_ts = 0.0

        # Target pose (wxyz quaternion) and gripper.
        self._target_position: np.ndarray = np.zeros(3, dtype=np.float64)
        self._target_quaternion_wxyz: np.ndarray = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._target_gripper: float = 0.5

        # Episode event flags.
        self._episode_save = False
        self._episode_abort = False
        self._quit_program = False

        # model_action_valid: set by the collector before each step to indicate
        # whether the incoming action came from a real model inference.
        self._model_action_valid = False

    # ── Public API for collector ─────────────────────────────────────────────

    def set_model_action_valid(self, valid: bool) -> None:
        """Set whether the next step's action came from a real model inference.

        Called by the collector before ``env.step()``. Consumed (reset to
        ``False``) after each step so it never leaks across frames.
        """
        self._model_action_valid = bool(valid)

    # ── Reset / lifecycle ────────────────────────────────────────────────────

    def reset(self, **kwargs):
        # Start in a safe MODEL state until the env has been reset and we can
        # read the real TCP pose.
        self._state = "model"
        self._last_key_press_ts.clear()
        self.listener.pop_pressed_keys()
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._target_gripper = 0.5
        self._episode_save = False
        self._episode_abort = False
        self._quit_program = False
        self._model_action_valid = False
        obs, info = self.env.reset(**kwargs)
        # Only now (after env.reset) can we safely read the real TCP pose.
        if self._start_in_engage:
            self._initialize_target_from_current_pose()
            self.get_wrapper_attr("reset_servo_smoothing")()
            self._state = "engage"
        return obs, info

    def close(self):
        self._state = "model"
        return super().close()

    # ── Key handling ─────────────────────────────────────────────────────────

    def _process_press_events(self) -> str | None:
        """Drain pressed keys and return a state-change event, or ``None``."""
        now = time.monotonic()
        pressed = self.listener.pop_pressed_keys()
        if pressed:
            get_logger().info("[DobotKeyboardIntervention] press events: %s", pressed)
        for key in pressed:
            if now - self._last_key_press_ts.get(key, -1.0) < self._key_debounce_s:
                continue
            self._last_key_press_ts[key] = now

            if key in self.quit_keys:
                return "quit"
            if key == self.abort_key:
                return "abort"
            if key == self.model_key:
                return "model"
            if key == self.done_key:
                return "done"
            if key == self.toggle_key:
                return "toggle"
        return None

    def _current_held_key(self) -> str | None:
        key = self.listener.get_key()
        if key is not None:
            get_logger().info("[DobotKeyboardIntervention] held key: %s", key)
        return key

    # ── Target updates ──────────────────────────────────────────────────────

    def _initialize_target_from_current_pose(self) -> None:
        """Read the current TCP pose and set it as the ENGAGE target."""
        pose = np.asarray(
            self.get_wrapper_attr("get_pose_state")(), dtype=np.float64
        ).reshape(8)
        self._target_position = pose[:3].copy()
        self._target_quaternion_wxyz = _normalize_same_hemisphere(
            pose[3:7], self._target_quaternion_wxyz
        )
        self._target_gripper = float(np.clip(pose[7], 0.0, 1.0))

    def _update_target(self) -> None:
        """Apply continuous-key deltas to the target pose/gripper.

        Translation keys (w/s/a/d/q/e) are rotated from the operator's physical
        frame to the base frame via ``_base_rotation`` (3×3). Tool-frame
        rotations (i/k/j/l/u/o) are unaffected (they use right-multiply in the
        tool frame, independent of the base frame).
        """
        key = self._current_held_key()
        if key is None:
            return

        # ── Translation: rotate through _base_rotation (3×3) ──────────────────
        # Keyboard mapping (physical frame): w/s = front/back (+/-X),
        # a/d = left/right (+/-Y), q/e = up/down (+/-Z). These map directly to
        # the standard Dobot base frame where +X is "forward" and +Y is "left".
        phys_xyz: np.ndarray | None = None
        if key == "w":
            phys_xyz = np.array([self.position_delta, 0.0, 0.0])
        elif key == "s":
            phys_xyz = np.array([-self.position_delta, 0.0, 0.0])
        elif key == "a":
            phys_xyz = np.array([0.0, self.position_delta, 0.0])
        elif key == "d":
            phys_xyz = np.array([0.0, -self.position_delta, 0.0])
        elif key == "q":
            phys_xyz = np.array([0.0, 0.0, self.position_delta])
        elif key == "e":
            phys_xyz = np.array([0.0, 0.0, -self.position_delta])

        if phys_xyz is not None:
            base_xyz = self._base_rotation @ phys_xyz
            self._target_position += base_xyz
        elif key == "i":
            self._rotate_target("x", self.rotation_delta)
        elif key == "k":
            self._rotate_target("x", -self.rotation_delta)
        elif key == "l":
            self._rotate_target("y", self.rotation_delta)
        elif key == "j":
            self._rotate_target("y", -self.rotation_delta)
        elif key == "o":
            self._rotate_target("z", self.rotation_delta)
        elif key == "u":
            self._rotate_target("z", -self.rotation_delta)
        elif key in (",", "Key.comma"):
            self._target_gripper = 0.0
        elif key in (".", "Key.dot"):
            self._target_gripper = 1.0
        else:
            return

        now = time.monotonic()
        if now - self._last_target_log_ts >= self._target_log_interval_s:
            self._last_target_log_ts = now
            euler = Rotation.from_quat(
                _wxyz_to_xyzw(self._target_quaternion_wxyz)
            ).as_euler("xyz", degrees=True)
            get_logger().info(
                "[DobotKeyboardIntervention] key=%s "
                "target_pos=[%.4f, %.4f, %.4f] target_euler=[%.2f, %.2f, %.2f] gripper=%.2f",
                key,
                self._target_position[0],
                self._target_position[1],
                self._target_position[2],
                euler[0],
                euler[1],
                euler[2],
                self._target_gripper,
            )

    def _rotate_target(self, axis: str, delta: float) -> None:
        """Rotate the target quaternion in the tool frame (right-multiply)."""
        current = Rotation.from_quat(_wxyz_to_xyzw(self._target_quaternion_wxyz))
        delta_rot = Rotation.from_euler(axis, delta)
        new_rot = current * delta_rot  # right-multiply = tool frame
        new_wxyz = _xyzw_to_wxyz(new_rot.as_quat())
        self._target_quaternion_wxyz = _normalize_same_hemisphere(
            new_wxyz, self._target_quaternion_wxyz
        )

    # ── Action computation ───────────────────────────────────────────────────

    def _build_safe_action(self) -> np.ndarray:
        """Build the 8-dim absolute pose action from the current target."""
        action = np.empty(8, dtype=np.float64)
        # Workspace clamp is optional; skip if not configured (hardware guard
        # handles out-of-bound commands).
        if self._workspace_low is not None:
            action[:3] = np.clip(
                self._target_position, self._workspace_low, self._workspace_high
            )
        else:
            action[:3] = self._target_position
        action[3:7] = self._target_quaternion_wxyz  # already normalized
        action[7] = np.clip(self._target_gripper, 0.0, 1.0)

        # Final safety: finite + unit quaternion.
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action generated: {action}")
        if not np.isclose(np.linalg.norm(action[3:7]), 1.0, atol=1e-6):
            raise ValueError(
                f"Non-unit quaternion in action: norm={np.linalg.norm(action[3:7])}"
            )
        return action

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Transform the incoming model action based on the HIL state.

        Returns:
            ``(action_out, replaced)`` — *action_out* is the action to execute,
            *replaced* is ``True`` if the human target replaced the model action.
        """
        event = self._process_press_events()

        # "Execute human command first, then switch state" trick:
        # If we toggle back to MODEL this step, we still execute the human
        # target for this frame and switch state for the next step.
        was_engage = self._state == "engage"
        next_state = self._state

        if event == "quit":
            next_state = "model"
            self._quit_program = True
        elif event == "abort":
            next_state = "model"
            self._episode_abort = True
        elif event == "done":
            next_state = "model"
            self._episode_save = True
        elif event == "model":
            next_state = "model"
        elif event == "toggle":
            if self._state == "model":
                next_state = "engage"
                self._initialize_target_from_current_pose()
                self.get_wrapper_attr("reset_servo_smoothing")()
                get_logger().info("[DobotKeyboardIntervention] MODEL -> ENGAGE")
            else:
                next_state = "model"
                self.get_wrapper_attr("reset_servo_smoothing")()
                get_logger().info("[DobotKeyboardIntervention] ENGAGE -> MODEL")

        if not was_engage and next_state == "model":
            # Pure model mode: pass through.
            self._state = next_state
            return np.asarray(action, dtype=np.float64), False

        # ENGAGE (or the step where we leave ENGAGE): update target from held
        # keys and build the absolute pose action.
        self._update_target()
        action_out = self._build_safe_action()
        self._state = next_state
        return action_out, True

    def step(self, action):
        model_action = np.asarray(action, dtype=np.float64)
        # Snapshot the valid flag before the step consumes it.
        model_action_valid = self._model_action_valid
        self._model_action_valid = False  # consume

        new_action, replaced = self.action(model_action)
        if replaced:
            # ENGAGE frame: skip RelativeGripperBinarizer so keyboard's
            # absolute 0/1 gripper commands pass through directly.
            self.get_wrapper_attr("set_gripper_bypass")(True)
        obs, rew, done, truncated, info = self.env.step(new_action)

        info["model_action"] = model_action
        if replaced:
            # The base Dobot environment may transform the gripper command
            # (for example, stateful relative binarization). Record the command
            # actually sent to the controller, not the pre-transform target.
            info["intervene_action"] = np.asarray(
                info.get("executed_action", new_action), dtype=np.float64
            ).copy()
            info["intervene_flag"] = np.ones(1, dtype=bool)
            # ENGAGE frames are never valid model predictions.
            model_action_valid = False
        info["model_action_valid"] = np.array([model_action_valid], dtype=bool)

        if self._episode_save:
            rew = 1.0
            done = True
            info["hil_event"] = "save"
            info["success_once"] = np.array([True], dtype=bool)
        elif self._episode_abort:
            info["hil_event"] = "abort"
        if self._quit_program:
            info["quit_program"] = True

        info["hil_state"] = self._state
        return obs, rew, done, truncated, info
