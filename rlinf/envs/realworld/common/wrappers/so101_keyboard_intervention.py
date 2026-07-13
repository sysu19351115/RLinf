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

"""SO101 keyboard-based 6D end-effector intervention wrapper.

Controls one arm at a time. In ENGAGE mode the operator drives the active arm's
end-effector position and orientation with the keyboard; the wrapper solves IK
and returns a 12-dim absolute joint target. The non-active arm simply holds its
current joint configuration.
"""

from __future__ import annotations

import time

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


class SO101KeyboardIntervention(gym.ActionWrapper):
    """Keyboard 6D-EE intervention for the SO101 bimanual robot.

    Args:
        env: Wrapped SO101 environment. Must expose ``get_joint_positions()``.
        active_arm: Which arm is controlled first (``"left"`` or ``"right"``).
        position_delta: Position increment per key event (meters).
        rotation_delta: Rotation increment per key event (radians).
        gripper_delta: Gripper increment per key event (RLinf [0, 100] units).
        urdf_path: Optional path to the SO101 URDF. ``None`` uses the bundled
            official URDF.
        end_effector_link: Name of the EE frame in the URDF.
        toggle_key: Key that toggles MODEL / ENGAGE.
        model_key: Key that forces return to MODEL.
        done_key: Key that ends the current episode (saves it and starts next).
        quit_keys: Keys that save the current episode and exit the program.
        switch_arm_key: Key that cycles the active arm.
    """

    _ARMS = ("left", "right")
    _LEFT_SLICE = slice(0, 6)
    _RIGHT_SLICE = slice(6, 12)

    def __init__(
        self,
        env: gym.Env,
        active_arm: str = "left",
        position_delta: float = 0.005,
        rotation_delta: float = 0.05,
        gripper_delta: float = 5.0,
        urdf_path: str | None = None,
        end_effector_link: str = "gripper_frame_link",
        toggle_key: str = "h",
        model_key: str = "m",
        done_key: str = "Key.enter",
        quit_keys: tuple[str, ...] = ("Key.esc",),
        switch_arm_key: str = "Key.tab",
    ):
        super().__init__(env)

        if active_arm not in self._ARMS:
            raise ValueError(
                f"active_arm must be one of {self._ARMS}, got {active_arm}"
            )
        self._active_arm = active_arm
        self._episode_done = False
        self._quit_program = False

        config = getattr(self.unwrapped, "config", None)
        is_dummy = getattr(config, "is_dummy", False) if config is not None else False
        try:
            self.listener = KeyboardListener()
        except Exception as exc:
            if is_dummy:
                get_logger().warning(
                    "[SO101KeyboardIntervention] KeyboardListener failed in dummy "
                    "mode: %s. Using dummy keyboard listener.",
                    exc,
                )
                self.listener = _DummyKeyboardListener()
            else:
                raise

        self.position_delta = float(position_delta)
        self.rotation_delta = float(rotation_delta)
        self.gripper_delta = float(gripper_delta)
        self.toggle_key = toggle_key
        self.model_key = model_key
        self.done_key = done_key
        self.quit_keys = quit_keys
        self.switch_arm_key = switch_arm_key

        self._state = "model"
        self._last_key_press_ts: dict[str, float] = {}
        self._key_debounce_s = 0.12
        self._target_log_interval_s = 1.0
        self._last_target_log_ts = 0.0

        # Import lazily to avoid a circular import through rlinf.envs.realworld.so101.
        from rlinf.envs.realworld.so101.kinematics import SO101ArmKinematics

        self._kinematics = {
            "left": SO101ArmKinematics(
                urdf_path=urdf_path, end_effector_link=end_effector_link
            ),
            "right": SO101ArmKinematics(
                urdf_path=urdf_path, end_effector_link=end_effector_link
            ),
        }

        # Target pose and gripper for each arm in RLinf units.
        self._target_pos: dict[str, np.ndarray] = {}
        self._target_quat: dict[str, np.ndarray] = {}  # [x, y, z, w]
        self._target_gripper: dict[str, float] = {}

    # ── Reset / lifecycle ────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self._state = "model"
        self._last_key_press_ts.clear()
        self.listener.pop_pressed_keys()
        self._target_pos.clear()
        self._target_quat.clear()
        self._target_gripper.clear()
        self._episode_done = False
        self._quit_program = False
        return self.env.reset(**kwargs)

    # ── Key handling ─────────────────────────────────────────────────────────

    def _process_press_events(self) -> str | None:
        """Drain pressed keys and return any state-change event, or None."""
        now = time.monotonic()
        event = None
        pressed = self.listener.pop_pressed_keys()
        if pressed:
            get_logger().info("[SO101KeyboardIntervention] press events: %s", pressed)
        for key in pressed:
            if now - self._last_key_press_ts.get(key, -1.0) < self._key_debounce_s:
                continue
            self._last_key_press_ts[key] = now

            if key in self.quit_keys:
                return "quit"
            if key == self.model_key:
                return "model"
            if key == self.done_key:
                return "done"
            if key == self.toggle_key:
                return "toggle"
            if key == self.switch_arm_key:
                return "switch_arm"
        return event

    def _current_held_key(self) -> str | None:
        key = self.listener.get_key()
        if key is not None:
            get_logger().info("[SO101KeyboardIntervention] held key: %s", key)
        return key

    # ── State updates ────────────────────────────────────────────────────────

    def _arm_joints(self, q12: np.ndarray, arm: str) -> np.ndarray:
        sl = self._LEFT_SLICE if arm == "left" else self._RIGHT_SLICE
        return np.asarray(q12[sl], dtype=np.float64).copy()

    def _set_arm_joints(self, q12: np.ndarray, arm: str, q6: np.ndarray) -> None:
        sl = self._LEFT_SLICE if arm == "left" else self._RIGHT_SLICE
        q12[sl] = q6

    def _initialize_arm_targets(self, q12: np.ndarray) -> None:
        """Set target pose/gripper from current joints for both arms."""
        for arm in self._ARMS:
            q6 = self._arm_joints(q12, arm)
            pos, quat = self._kinematics[arm].forward(q6)
            self._target_pos[arm] = pos
            self._target_quat[arm] = quat
            self._target_gripper[arm] = float(q6[5])

    def _update_active_target(self, q12: np.ndarray) -> None:
        """Apply continuous-key deltas to the active arm's target pose/gripper."""
        arm = self._active_arm
        key = self._current_held_key()
        if key is None:
            return

        if key in ("w",):
            self._target_pos[arm][0] += self.position_delta
        elif key in ("s",):
            self._target_pos[arm][0] -= self.position_delta
        elif key in ("a",):
            self._target_pos[arm][1] += self.position_delta
        elif key in ("d",):
            self._target_pos[arm][1] -= self.position_delta
        elif key in ("q",):
            self._target_pos[arm][2] += self.position_delta
        elif key in ("e",):
            self._target_pos[arm][2] -= self.position_delta
        elif key in ("i",):
            self._rotate_active_target("x", self.rotation_delta)
        elif key in ("k",):
            self._rotate_active_target("x", -self.rotation_delta)
        elif key in ("j",):
            self._rotate_active_target("y", self.rotation_delta)
        elif key in ("l",):
            self._rotate_active_target("y", -self.rotation_delta)
        elif key in ("u",):
            self._rotate_active_target("z", self.rotation_delta)
        elif key in ("o",):
            self._rotate_active_target("z", -self.rotation_delta)
        elif key in (",", "Key.comma"):
            self._target_gripper[arm] = max(
                0.0, self._target_gripper[arm] - self.gripper_delta
            )
        elif key in (".", "Key.dot"):
            self._target_gripper[arm] = min(
                100.0, self._target_gripper[arm] + self.gripper_delta
            )
        else:
            # Not a pose/gripper control key; avoid logging below.
            return

        now = time.monotonic()
        if now - self._last_target_log_ts >= self._target_log_interval_s:
            self._last_target_log_ts = now
            pos = self._target_pos[arm]
            euler = Rotation.from_quat(self._target_quat[arm]).as_euler(
                "xyz", degrees=True
            )
            get_logger().info(
                "[SO101KeyboardIntervention] active=%s key=%s "
                "target_pos=[%.4f, %.4f, %.4f] target_euler=[%.2f, %.2f, %.2f] gripper=%.1f",
                arm,
                key,
                pos[0],
                pos[1],
                pos[2],
                euler[0],
                euler[1],
                euler[2],
                self._target_gripper[arm],
            )

    def _rotate_active_target(self, axis: str, delta: float) -> None:
        arm = self._active_arm
        delta_rot = Rotation.from_euler(axis, delta)
        current_rot = Rotation.from_quat(self._target_quat[arm])
        # Apply delta in the end-effector (tool) frame.
        new_rot = current_rot * delta_rot
        self._target_quat[arm] = new_rot.as_quat().astype(np.float64)

    # ── Action computation ───────────────────────────────────────────────────

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        event = self._process_press_events()

        # Keep track of whether we were in ENGAGE at the start of this step.
        # If the operator toggles back to MODEL this step, we still want to
        # execute the human command (not the incoming model action, which might
        # be a stale queue entry or a dummy zero) and then switch state for the
        # next step.
        was_engage = self._state == "engage"
        next_state = self._state

        if event == "quit":
            next_state = "model"
            self._quit_program = True
        elif event == "done":
            next_state = "model"
            self._episode_done = True
        elif event == "model":
            next_state = "model"
        elif event == "toggle":
            if self._state == "model":
                next_state = "engage"
                q12 = self._get_follower_q()
                self._initialize_arm_targets(q12)
                get_logger().info(
                    "[SO101KeyboardIntervention] MODEL -> ENGAGE, active=%s",
                    self._active_arm,
                )
            else:
                next_state = "model"
                get_logger().info("[SO101KeyboardIntervention] ENGAGE -> MODEL")
        elif event == "switch_arm" and self._state == "engage":
            self._active_arm = "right" if self._active_arm == "left" else "left"
            # Re-initialize target for the newly active arm from current FK.
            q12 = self._get_follower_q()
            q6 = self._arm_joints(q12, self._active_arm)
            pos, quat = self._kinematics[self._active_arm].forward(q6)
            self._target_pos[self._active_arm] = pos
            self._target_quat[self._active_arm] = quat
            self._target_gripper[self._active_arm] = float(q6[5])
            get_logger().info(
                "[SO101KeyboardIntervention] switched active arm to %s",
                self._active_arm,
            )

        if not was_engage and next_state == "model":
            self._state = next_state
            return np.asarray(action, dtype=np.float64), False

        # ENGAGE (or the step where we leave ENGAGE): read current joints and
        # update target based on held keys.
        q12_current = self._get_follower_q()
        self._update_active_target(q12_current)

        action_out = q12_current.copy()
        for arm in self._ARMS:
            q_init = self._arm_joints(q12_current, arm)
            # For the active arm the target pose is updated by the keyboard.
            # For the non-active arm we keep the target pose stored at the
            # moment we entered ENGAGE (or when the arm was last active). Using
            # a fixed target prevents the arm from drifting under gravity: if we
            # instead re-targeted the current joint position every step, a small
            # sag between reads would become the new target and the arm would
            # slowly fall.
            q6 = self._kinematics[arm].inverse(
                self._target_pos[arm],
                self._target_quat[arm],
                q_init,
            )
            q6[5] = self._target_gripper[arm]
            self._set_arm_joints(action_out, arm, q6)

        # Clamp to action space.
        low = self.action_space.low.astype(np.float64)
        high = self.action_space.high.astype(np.float64)
        action_out = np.clip(action_out, low, high)
        self._state = next_state
        return action_out, True

    def step(self, action):
        model_action = np.asarray(action, dtype=np.float64)
        new_action, replaced = self.action(model_action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["model_action"] = model_action
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1, dtype=bool)
        if self._episode_done:
            truncated = True
        if self._quit_program:
            info["quit_program"] = True
        info["hil_state"] = self._state
        info["hil_active_arm"] = self._active_arm
        return obs, rew, done, truncated, info

    def close(self):
        self._state = "model"
        return super().close()

    def _get_follower_q(self) -> np.ndarray:
        return np.asarray(
            self.get_wrapper_attr("get_joint_positions")(), dtype=np.float64
        )
