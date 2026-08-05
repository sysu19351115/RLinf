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
from collections.abc import Mapping, Sequence
from typing import Any

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

    def get_keys(self) -> frozenset[str]:
        return frozenset()

    def is_connected(self) -> bool:
        return True

    def fatal_error(self) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Quaternion helpers (env/wrapper/dataset always use wxyz; scipy uses xyzw).
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _normalize_same_hemisphere(
    new_q: np.ndarray, reference_q: np.ndarray
) -> np.ndarray:
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
        safe_model_handoff: Hold until the current action chunk ends before
            allowing MODEL control to resume.
        handoff_max_position_jump_m: Maximum position difference between the
            first fresh model action and current TCP feedback.
        handoff_max_rotation_jump_deg: Maximum orientation difference between
            the first fresh model action and current TCP feedback.
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
        allow_motion_intervention: bool = True,
        episode_control_mode: str = "collector",
        wait_for_start_on_reset: bool = False,
        start_key: str = "y",
        start_gate_timeout_s: float | None = None,
        safe_model_handoff: bool = True,
        handoff_max_position_jump_m: float = 0.005,
        handoff_max_rotation_jump_deg: float = 2.0,
        human_stage_reward: Mapping[str, Any] | None = None,
        listener=None,
    ):
        super().__init__(env)

        # ── Validate env mode ────────────────────────────────────────────────
        config = getattr(self.unwrapped, "config", None)
        if config is None:
            raise ValueError(
                "DobotKeyboardIntervention requires a DobotEnv with a config."
            )
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
        if episode_control_mode not in (
            "collector",
            "online",
            "online_chunk_boundary",
        ):
            raise ValueError(
                "episode_control_mode must be 'collector', 'online', or "
                "'online_chunk_boundary', "
                f"got {episode_control_mode!r}."
            )
        if start_gate_timeout_s is not None and start_gate_timeout_s <= 0:
            raise ValueError(
                "start_gate_timeout_s must be positive or None, "
                f"got {start_gate_timeout_s}."
            )
        if handoff_max_position_jump_m <= 0:
            raise ValueError(
                "handoff_max_position_jump_m must be positive, "
                f"got {handoff_max_position_jump_m}."
            )
        if handoff_max_rotation_jump_deg <= 0:
            raise ValueError(
                "handoff_max_rotation_jump_deg must be positive, "
                f"got {handoff_max_rotation_jump_deg}."
            )
        if (
            episode_control_mode == "online"
            and allow_motion_intervention
            and not safe_model_handoff
        ):
            raise ValueError(
                "online motion intervention requires safe_model_handoff=True."
            )

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
            self._workspace_high = np.asarray(workspace_high, dtype=np.float64).reshape(
                3
            )
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
        self._allow_motion_intervention = bool(allow_motion_intervention)
        if start_in_engage and not self._allow_motion_intervention:
            raise ValueError(
                "start_in_engage=True is incompatible with "
                "allow_motion_intervention=False."
            )
        self._start_in_engage = bool(start_in_engage)
        self._episode_control_mode = episode_control_mode
        self._wait_for_start_on_reset = bool(wait_for_start_on_reset)
        self._start_key = start_key
        self._start_gate_timeout_s = start_gate_timeout_s
        self._safe_model_handoff = bool(safe_model_handoff)
        self._handoff_max_position_jump_m = float(handoff_max_position_jump_m)
        self._handoff_max_rotation_jump_deg = float(handoff_max_rotation_jump_deg)

        # ── Human stage-label rewards (HIL-RLPD dense reward design) ────────
        # Keys 1/2/3 mark "grasp success", "arrived before the hook" and
        # "confirmed hang".  The state machine only moves forward; event
        # rewards are emitted once per stage; per-step hold rewards are small
        # so the terminal success/failure signal stays dominant.
        hsr = dict(human_stage_reward or {})
        # Explicit enable switch (P1): configs that do not opt in keep the
        # legacy reward semantics exactly (env reward passthrough, success=1,
        # failure=0).  ``human_stage_reward`` must be a non-None dict with
        # ``enabled: true`` for the stage-label rewards to take effect.
        self._human_stage_enabled = (
            bool(hsr.get("enabled", True)) if human_stage_reward is not None else False
        )
        if self._human_stage_enabled:
            stage_keys = tuple(str(k) for k in hsr.get("stage_keys", ("1", "2", "3")))
            stage_hold = [float(v) for v in hsr.get("hold", (0.0, 0.001, 0.002, 0.0))]
            stage_event = [float(v) for v in hsr.get("event", (0.0, 0.02, 0.05, 0.20))]
            success_reward = float(hsr.get("success", 1.0))
            failure_reward = float(hsr.get("failure", -1.0))
            intervention_penalty = float(hsr.get("intervention_penalty", 0.0))
            control_keys = set(quit_keys) | {
                abort_key,
                done_key,
                model_key,
                toggle_key,
                start_key,
            }
            overlap = set(stage_keys) & control_keys
            if overlap:
                raise ValueError(
                    "human_stage_reward.stage_keys must not overlap control keys "
                    "(quit/abort/done/model/toggle/start); "
                    f"overlap={sorted(overlap)}"
                )
            if len(stage_keys) != 3 or len(set(stage_keys)) != 3:
                raise ValueError(
                    "human_stage_reward.stage_keys must contain exactly 3 unique "
                    f"keys, got {stage_keys!r}"
                )
            if len(stage_hold) != 4 or len(stage_event) != 4:
                raise ValueError(
                    "human_stage_reward.hold/event must be length-4 vectors "
                    "(stages 0..3), got "
                    f"hold={stage_hold}, event={stage_event}"
                )
            if not (
                all(np.isfinite(stage_hold))
                and all(np.isfinite(stage_event))
                and np.isfinite(success_reward)
                and np.isfinite(failure_reward)
                and np.isfinite(intervention_penalty)
            ):
                raise ValueError("human_stage_reward values must all be finite")
            if success_reward < 0.0 or failure_reward > 0.0:
                raise ValueError(
                    "human_stage_reward.success must be >= 0 and failure must be "
                    f"<= 0, got success={success_reward}, failure={failure_reward}"
                )
            if intervention_penalty > 0.0:
                raise ValueError(
                    "human_stage_reward.intervention_penalty must be <= 0 "
                    f"(got {intervention_penalty})"
                )
            self.stage_keys = stage_keys
            self._stage_key_to_index = {
                key: index for index, key in enumerate(stage_keys, start=1)
            }
            self._stage_hold_rewards = stage_hold
            self._stage_event_rewards = stage_event
            self._success_reward = success_reward
            self._failure_reward = failure_reward
            self._intervention_penalty = intervention_penalty
        else:
            self.stage_keys = ()
            self._stage_key_to_index = {}
            self._stage_hold_rewards = (0.0, 0.0, 0.0, 0.0)
            self._stage_event_rewards = (0.0, 0.0, 0.0, 0.0)
            self._success_reward = 1.0
            self._failure_reward = 0.0
            self._intervention_penalty = 0.0
        self._stage = 0
        self._pending_event_reward = 0.0

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
        self._episode_closed = False
        self._pending_episode_label: str | None = None
        self._is_chunk_boundary = False
        self._label_conflict_this_step = False

        # model_action_valid: set by the collector before each step to indicate
        # whether the incoming action came from a real model inference.
        self._model_action_valid = False
        self._request_replan_this_step = False
        self._handoff_hold_this_step = False
        self._handoff_rejected_this_step = False

    # ── Public API for collector ─────────────────────────────────────────────

    def set_model_action_valid(self, valid: bool) -> None:
        """Set whether the next step's action came from a real model inference.

        Called by the collector before ``env.step()``. Consumed (reset to
        ``False``) after each step so it never leaks across frames.
        """
        self._model_action_valid = bool(valid)

    def set_chunk_boundary(self, is_boundary: bool) -> None:
        """Mark whether the next step is the last action in its action chunk.

        ``RealWorldEnv.chunk_step`` owns the actual chunk size and calls this
        before every sub-step. Keeping the boundary outside this wrapper avoids
        duplicating ``num_action_chunks`` in keyboard configuration.
        """
        self._is_chunk_boundary = bool(is_boundary)

    def complete_model_handoff(self) -> None:
        """Arm MODEL control after the stale action chunk has been exhausted."""
        if self._state != "model_pending":
            return
        self.get_wrapper_attr("reset_servo_smoothing")()
        self._state = "model_armed"
        get_logger().info("[DobotKeyboardIntervention] MODEL_PENDING -> MODEL_ARMED")

    def wait_for_start_key(self, key: str | None = None) -> None:
        """Block until the operator presses *key* to start the episode.

        Polls the keyboard listener at ~20 Hz. In dummy mode (where the
        listener is a ``_DummyKeyboardListener``), returns immediately so
        automated tests are not blocked.

        Call this *before* ``env.reset()`` so the reset obs is fresh.
        """
        if key is None:
            key = self._start_key
        if isinstance(self.listener, _DummyKeyboardListener):
            return
        get_logger().info(
            "[DobotKeyboardIntervention] Press '%s' to start the next episode.", key
        )
        deadline = (
            None
            if self._start_gate_timeout_s is None
            else time.monotonic() + self._start_gate_timeout_s
        )
        while True:
            pressed = self.listener.pop_pressed_keys()
            if any(pressed_key in self.quit_keys for pressed_key in pressed):
                raise RuntimeError("Operator cancelled the start gate.")
            if key in pressed:
                break
            fatal_error = (
                self.listener.fatal_error()
                if hasattr(self.listener, "fatal_error")
                else None
            )
            if fatal_error is not None:
                raise RuntimeError(
                    f"Keyboard listener failed while waiting to start: {fatal_error}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                connected = (
                    self.listener.is_connected()
                    if hasattr(self.listener, "is_connected")
                    else None
                )
                raise TimeoutError(
                    "Timed out waiting for the operator start key "
                    f"{key!r}; keyboard_connected={connected}."
                )
            time.sleep(0.05)
        get_logger().info(
            "[DobotKeyboardIntervention] Start key '%s' pressed. Recording begins.",
            key,
        )

    # ── Reset / lifecycle ────────────────────────────────────────────────────

    def reset(self, **kwargs):
        if self._wait_for_start_on_reset:
            self.wait_for_start_key()
        # Start in a safe MODEL state until the env has been reset and we can
        # read the real TCP pose.
        self._state = "model"
        self._last_key_press_ts.clear()
        self.listener.pop_pressed_keys()
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._target_gripper = 0.5
        self._model_action_valid = False
        self._request_replan_this_step = False
        self._handoff_hold_this_step = False
        self._handoff_rejected_this_step = False
        obs, info = self.env.reset(**kwargs)
        # Clear the episode latch only after the underlying reset succeeds.
        # A failed reset must leave all future actions blocked.
        self._episode_save = False
        self._episode_abort = False
        self._quit_program = False
        self._episode_closed = False
        self._pending_episode_label = None
        self._is_chunk_boundary = False
        self._label_conflict_this_step = False
        self._stage = 0
        self._pending_event_reward = 0.0
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

            if self._human_stage_enabled and key in self.stage_keys:
                new_stage = self._stage_key_to_index[key]
                if new_stage > self._stage:
                    self._stage = new_stage
                    self._pending_event_reward = self._stage_event_rewards[new_stage]
                    get_logger().info(
                        "[HumanStage] key=%s stage->%d event=%.4f",
                        key,
                        new_stage,
                        self._pending_event_reward,
                    )
                else:
                    get_logger().warning(
                        "[HumanStage] ignored non-monotonic stage key %s "
                        "(current stage=%d)",
                        key,
                        self._stage,
                    )
                continue

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

    def _current_held_keys(self) -> frozenset[str]:
        """Return all held keys, with fallback for legacy injected listeners."""
        if hasattr(self.listener, "get_keys"):
            keys = frozenset(self.listener.get_keys())
        else:
            key = self.listener.get_key()
            keys = frozenset((key,)) if key is not None else frozenset()
        if keys:
            get_logger().info("[DobotKeyboardIntervention] held keys: %s", sorted(keys))
        return keys

    def _listener_failure_reason(self) -> str | None:
        """Return a fail-closed reason when keyboard intervention is unavailable."""
        fatal_error = (
            self.listener.fatal_error()
            if hasattr(self.listener, "fatal_error")
            else None
        )
        if fatal_error is not None:
            return "keyboard_listener_error"
        if hasattr(self.listener, "is_connected") and not self.listener.is_connected():
            return "keyboard_disconnected"
        return None

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
        keys = self._current_held_keys()
        if not keys:
            return

        # ── Translation: rotate through _base_rotation (3×3) ──────────────────
        # Mirror-mode keyboard mapping for face-to-face operation:
        # w/s = back/front (operator's perspective is mirrored),
        # a/d = right/left, q/e = up/down (up is unchanged).
        direction = np.array(
            [
                float("s" in keys) - float("w" in keys),
                float("d" in keys) - float("a" in keys),
                float("q" in keys) - float("e" in keys),
            ],
            dtype=np.float64,
        )
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0.0:
            # Keep diagonal/3-axis movement at the same total speed as a
            # single-axis command instead of increasing it by sqrt(2)/sqrt(3).
            phys_xyz = direction * (self.position_delta / direction_norm)
            base_xyz = self._base_rotation @ phys_xyz
            self._target_position += base_xyz
        else:
            # Rotation and gripper remain single-key controls. Translation has
            # priority whenever a non-cancelled translation combination exists.
            key = self.listener.get_key()
            if key is None:
                return
            if key == "i":
                self._rotate_target("x", self.rotation_delta)
            elif key == "k":
                self._rotate_target("x", -self.rotation_delta)
            elif key == "j":
                self._rotate_target("y", self.rotation_delta)
            elif key == "l":
                self._rotate_target("y", -self.rotation_delta)
            elif key == "u":
                self._rotate_target("z", self.rotation_delta)
            elif key == "o":
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
                "[DobotKeyboardIntervention] keys=%s "
                "target_pos=[%.4f, %.4f, %.4f] target_euler=[%.2f, %.2f, %.2f] gripper=%.2f",
                sorted(keys),
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

    def _build_feedback_hold_action(self) -> np.ndarray:
        """Return the measured TCP pose as a fail-closed hold command."""
        pose = np.asarray(
            self.get_wrapper_attr("get_pose_state")(), dtype=np.float64
        ).reshape(8)
        pose = pose.copy()
        pose[3:7] = _normalize_same_hemisphere(pose[3:7], self._target_quaternion_wxyz)
        pose[7] = float(np.clip(pose[7], 0.0, 1.0))
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite feedback hold pose: {pose}")
        return pose

    def _model_handoff_is_safe(
        self, model_action: np.ndarray
    ) -> tuple[bool, float, float]:
        """Validate a fresh model chunk's first absolute pose against feedback."""
        if not np.isfinite(model_action).all():
            return False, float("inf"), float("inf")
        current = self._build_feedback_hold_action()
        position_jump_m = float(np.linalg.norm(model_action[:3] - current[:3]))
        model_quat = _normalize_same_hemisphere(model_action[3:7], current[3:7])
        quat_dot = float(np.clip(abs(np.dot(model_quat, current[3:7])), 0.0, 1.0))
        rotation_jump_deg = float(np.degrees(2.0 * np.arccos(quat_dot)))
        safe = (
            position_jump_m <= self._handoff_max_position_jump_m
            and rotation_jump_deg <= self._handoff_max_rotation_jump_deg
        )
        return safe, position_jump_m, rotation_jump_deg

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Transform the incoming model action based on the HIL state.

        Returns:
            ``(action_out, replaced)`` — *action_out* is the action to execute,
            *replaced* is ``True`` if the human target replaced the model action.
        """
        event = self._process_press_events()
        self._request_replan_this_step = False
        self._handoff_hold_this_step = False
        self._handoff_rejected_this_step = False
        self._label_conflict_this_step = False

        # "Execute human command first, then switch state" trick:
        # If we toggle back to MODEL this step, we still execute the human
        # target for this frame and switch state for the next step.
        was_engage = self._state == "engage"
        was_handoff = self._state in ("model_pending", "model_armed")
        next_state = self._state

        if event == "quit":
            next_state = "model"
            self._quit_program = True
            self._pending_episode_label = None
            self._episode_closed = True
        elif event == "abort":
            next_state = "model"
            if self._episode_control_mode == "online_chunk_boundary":
                if self._pending_episode_label is None:
                    self._pending_episode_label = "failure"
                elif self._pending_episode_label != "failure":
                    self._label_conflict_this_step = True
            else:
                self._episode_abort = True
                if self._episode_control_mode == "online":
                    self._episode_closed = True
        elif event == "done":
            next_state = "model"
            if self._episode_control_mode == "online_chunk_boundary":
                if self._pending_episode_label is None:
                    self._pending_episode_label = "success"
                elif self._pending_episode_label != "success":
                    self._label_conflict_this_step = True
            else:
                self._episode_save = True
                self._episode_closed = True
        elif event == "model":
            if self._safe_model_handoff and self._state == "engage":
                next_state = "model_pending"
                self._request_replan_this_step = True
                self.get_wrapper_attr("reset_servo_smoothing")()
                get_logger().info("[DobotKeyboardIntervention] ENGAGE -> MODEL_PENDING")
            elif self._state not in ("model_pending", "model_armed"):
                next_state = "model"
        elif event == "toggle":
            if not self._allow_motion_intervention:
                # Autonomous evaluation deliberately keeps the keyboard
                # listener for start/success/failure/quit labels while making
                # it impossible for an operator key to replace model actions.
                next_state = "model"
                get_logger().warning(
                    "[DobotKeyboardIntervention] Ignoring motion-intervention "
                    "toggle because allow_motion_intervention=False."
                )
            elif self._state == "model":
                next_state = "engage"
                self._initialize_target_from_current_pose()
                self.get_wrapper_attr("reset_servo_smoothing")()
                get_logger().info("[DobotKeyboardIntervention] MODEL -> ENGAGE")
            elif self._state == "engage":
                if self._safe_model_handoff:
                    next_state = "model_pending"
                    self._request_replan_this_step = True
                    get_logger().info(
                        "[DobotKeyboardIntervention] ENGAGE -> MODEL_PENDING"
                    )
                else:
                    next_state = "model"
                self.get_wrapper_attr("reset_servo_smoothing")()
            else:
                next_state = "engage"
                self._initialize_target_from_current_pose()
                self.get_wrapper_attr("reset_servo_smoothing")()
                get_logger().info(
                    "[DobotKeyboardIntervention] MODEL handoff cancelled -> ENGAGE"
                )

        if event == "quit":
            self._state = next_state
            self._handoff_hold_this_step = True
            return self._build_feedback_hold_action(), False

        if (
            not was_engage
            and was_handoff
            and event in ("abort", "done")
            and self._episode_control_mode != "online_chunk_boundary"
        ):
            self._state = next_state
            self._handoff_hold_this_step = True
            return self._build_feedback_hold_action(), False

        if not was_engage and next_state == "model_pending":
            self._state = next_state
            self._handoff_hold_this_step = True
            return self._build_feedback_hold_action(), False

        if not was_engage and next_state == "model_armed":
            model_action = np.asarray(action, dtype=np.float64).reshape(8)
            safe, position_jump_m, rotation_jump_deg = self._model_handoff_is_safe(
                model_action
            )
            if not safe:
                self._handoff_hold_this_step = True
                self._handoff_rejected_this_step = True
                get_logger().error(
                    "[DobotKeyboardIntervention] Rejecting unsafe fresh model "
                    "handoff: position_jump=%.4fm (limit=%.4fm), "
                    "rotation_jump=%.2fdeg (limit=%.2fdeg)",
                    position_jump_m,
                    self._handoff_max_position_jump_m,
                    rotation_jump_deg,
                    self._handoff_max_rotation_jump_deg,
                )
                return self._build_feedback_hold_action(), False
            self._state = "model"
            get_logger().info("[DobotKeyboardIntervention] MODEL_ARMED -> MODEL")
            return model_action, False

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
        if self._episode_closed:
            raise RuntimeError(
                "The operator has ended this episode; reset() is required "
                "before another action can be executed."
            )
        model_action = np.asarray(action, dtype=np.float64)
        listener_failure = self._listener_failure_reason()
        if listener_failure is not None:
            # Do not execute an unmonitored model action. Sending the measured
            # pose back as the absolute target produces a local hold while the
            # episode is failed closed.
            hold_action = np.asarray(
                self.get_wrapper_attr("get_pose_state")(), dtype=np.float64
            ).reshape(8)
            self.get_wrapper_attr("set_gripper_bypass")(True)
            obs, rew, done, truncated, info = self.env.step(hold_action)
            if self._human_stage_enabled:
                # Safety abort: not an operator failure label; the transition
                # is marked invalid downstream, so no stage/terminal reward.
                rew = 0.0
            self._state = "model"
            self._model_action_valid = False
            self._pending_episode_label = None
            info["model_action"] = model_action
            info["model_action_valid"] = np.array([False], dtype=bool)
            info["hil_event"] = "abort"
            info["termination_reason"] = listener_failure
            info["keyboard_connected"] = False
            info["reward_label_valid"] = False
            info["success_once"] = np.array([False], dtype=bool)
            if self._episode_control_mode in ("online", "online_chunk_boundary"):
                self._episode_closed = True
                info["operator_episode_end"] = True
                info["operator_success"] = False
                truncated = True
            info["hil_state"] = self._state
            return obs, rew, done, truncated, info

        # Snapshot the valid flag before the step consumes it.
        model_action_valid = self._model_action_valid
        self._model_action_valid = False  # consume

        new_action, replaced = self.action(model_action)
        if replaced or self._handoff_hold_this_step:
            # ENGAGE frame: skip RelativeGripperBinarizer so keyboard's
            # absolute 0/1 gripper commands pass through directly. Handoff
            # holds also preserve the measured gripper position exactly.
            self.get_wrapper_attr("set_gripper_bypass")(True)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # HIL-RLPD: when the human stage-label feature is enabled it owns the
        # reward signal and the env's own task reward is dropped to avoid
        # double counting.  When disabled, the env reward passes through
        # unchanged (legacy semantics).
        if self._human_stage_enabled:
            rew = float(self._stage_hold_rewards[self._stage])
            if self._pending_event_reward != 0.0:
                rew += self._pending_event_reward
                self._pending_event_reward = 0.0
            if replaced or self._handoff_hold_this_step:
                rew += self._intervention_penalty

        info["model_action"] = model_action
        info["request_replan"] = np.array([self._request_replan_this_step], dtype=bool)
        info["handoff_hold"] = np.array([self._handoff_hold_this_step], dtype=bool)
        info["handoff_rejected"] = np.array(
            [self._handoff_rejected_this_step], dtype=bool
        )
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
        elif self._handoff_hold_this_step:
            # Reuse the action-override transport while keeping the human label
            # false. EnvWorker combines this with handoff_hold_mask when
            # replacing the stale rollout action in the trajectory.
            info["intervene_action"] = np.asarray(
                info.get("executed_action", new_action), dtype=np.float64
            ).copy()
            info["intervene_flag"] = np.zeros(1, dtype=bool)
            model_action_valid = False
        info["model_action_valid"] = np.array([model_action_valid], dtype=bool)
        if self._pending_episode_label is not None:
            info["operator_label_pending"] = self._pending_episode_label
        if self._label_conflict_this_step:
            info["operator_label_conflict_ignored"] = True

        if self._handoff_rejected_this_step:
            truncated = True
            if self._human_stage_enabled:
                # Safety abort: not an operator failure label; the transition
                # is marked invalid downstream, so no stage/terminal reward.
                rew = 0.0
            self._episode_closed = True
            self._pending_episode_label = None
            info["termination_reason"] = "unsafe_model_handoff"
            info["operator_episode_end"] = False
            info["operator_success"] = False
            info["reward_label_valid"] = False
            info["success_once"] = np.array([False], dtype=bool)

        if (
            self._episode_control_mode == "online_chunk_boundary"
            and self._is_chunk_boundary
            and self._pending_episode_label is not None
            and not self._handoff_rejected_this_step
        ):
            if self._pending_episode_label == "success":
                self._episode_save = True
            else:
                self._episode_abort = True
            self._pending_episode_label = None
            self._episode_closed = True

        if self._episode_save:
            if self._human_stage_enabled:
                rew += self._success_reward
                get_logger().info(
                    "[HumanStage] terminal=success reward=%.4f", self._success_reward
                )
            else:
                rew = 1.0
            done = True
            truncated = False
            info["hil_event"] = "save"
            info["termination_reason"] = "operator_success"
            info["operator_episode_end"] = True
            info["operator_success"] = True
            info["reward_label_valid"] = True
            info["success_once"] = np.array([True], dtype=bool)
        elif self._episode_abort:
            info["hil_event"] = "abort"
            info["termination_reason"] = (
                "operator_failure"
                if self._episode_control_mode == "online_chunk_boundary"
                else "operator_abort"
            )
            info["success_once"] = np.array([False], dtype=bool)
            if self._episode_control_mode in ("online", "online_chunk_boundary"):
                info["operator_episode_end"] = True
                info["operator_success"] = False
                info["reward_label_valid"] = True
                done = False
                truncated = True
                if self._human_stage_enabled:
                    rew += self._failure_reward
                    get_logger().info(
                        "[HumanStage] terminal=failure reward=%.4f",
                        self._failure_reward,
                    )
        if self._quit_program:
            info["quit_program"] = True
            info["operator_shutdown_requested"] = True
            info["termination_reason"] = "operator_quit"
            info["reward_label_valid"] = False
            info["success_once"] = np.array([False], dtype=bool)
            if self._episode_control_mode in ("online", "online_chunk_boundary"):
                info["operator_episode_end"] = True
                info["operator_success"] = False
                truncated = True

        info["hil_state"] = self._state
        # Edge events are reported once. _episode_closed remains latched for
        # terminal events so missing a consumer can never resume motion.
        self._episode_save = False
        self._episode_abort = False
        self._quit_program = False
        self._is_chunk_boundary = False
        return obs, rew, done, truncated, info
