# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Dobot keyboard HIL intervention.

These tests never access ``/dev/input``, cameras, or real hardware.
``DobotKeyboardIntervention`` accepts an injectable ``listener`` for
deterministic testing; ``DobotEnv.get_pose_state`` / ``reset_servo_smoothing``
are exercised via dummy config.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Sequence

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.common.wrappers.dobot_keyboard_intervention import (
    DobotKeyboardIntervention,
)
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_WORKSPACE_LOW = np.array([-0.5, -0.5, 0.0])
_DUMMY_WORKSPACE_HIGH = np.array([0.5, 0.5, 0.5])
# Must match DobotEnv._get_observation dummy pose for consistency.
_DUMMY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)


class FakeListener:
    """Deterministic keyboard listener for testing.

    ``press_sequence`` feeds edge-press events (consumed by
    ``pop_pressed_keys``); ``held`` is the current key held down
    (consumed by ``get_key``).
    """

    def __init__(
        self,
        press_sequence: Sequence[str] | None = None,
        held: str | None = None,
    ):
        self._presses: deque[str] = deque(press_sequence or [])
        self._held_keys = {held} if held is not None else set()
        self._latest_held = held
        self._connected = True
        self._fatal_error = None

    def pop_pressed_keys(self) -> list[str]:
        if self._presses:
            return [self._presses.popleft()]
        return []

    def get_key(self) -> str | None:
        return self._latest_held

    def get_keys(self) -> frozenset[str]:
        return frozenset(self._held_keys)

    def press(self, key: str) -> None:
        """Helper to inject a key press after construction."""
        self._presses.append(key)

    def set_held(self, key: str | None) -> None:
        """Helper to set the currently held key."""
        self._held_keys = {key} if key is not None else set()
        self._latest_held = key

    def set_held_keys(self, *keys: str) -> None:
        """Helper to set all currently held keys."""
        self._held_keys = set(keys)
        self._latest_held = keys[-1] if keys else None

    def is_connected(self) -> bool:
        return self._connected

    def fatal_error(self) -> str | None:
        return self._fatal_error


def _bare_keyboard_listener() -> KeyboardListener:
    """Construct KeyboardListener state without opening an evdev device."""
    listener = KeyboardListener.__new__(KeyboardListener)
    listener.state_lock = threading.Lock()
    listener.latest_data = {"key": None}
    listener._held_keys = {}
    listener._press_events = deque()
    listener._connected = threading.Event()
    listener._connected.set()
    listener._fatal_error = None
    return listener


def _dummy_pose_env(max_num_steps: int = 100) -> DobotEnv:
    return DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            max_num_steps=max_num_steps,
            step_frequency=10_000.0,
        )
    )


def _make_wrapper(
    env: DobotEnv,
    listener: FakeListener | None = None,
    **kwargs,
) -> DobotKeyboardIntervention:
    defaults = {
        "position_delta": 0.002,
        "rotation_delta": 0.02,
        "gripper_delta": 0.05,
        "workspace_low": _DUMMY_WORKSPACE_LOW,
        "workspace_high": _DUMMY_WORKSPACE_HIGH,
    }
    defaults.update(kwargs)
    if listener is None:
        listener = FakeListener()
    return DobotKeyboardIntervention(env, listener=listener, **defaults)


# ===========================================================================
# Task 1: DobotEnv.get_pose_state / reset_servo_smoothing
# ===========================================================================


class TestKeyboardListenerHeldKeys:
    def test_pressing_two_keys_keeps_both_until_each_is_released(self):
        listener = _bare_keyboard_listener()

        listener._update_key_state("w", 1)
        listener._update_key_state("a", 1)

        assert listener.get_keys() == frozenset({"w", "a"})
        assert listener.get_key() == "a"

        listener._update_key_state("a", 0)

        assert listener.get_keys() == frozenset({"w"})
        assert listener.get_key() == "w"

    def test_clear_held_keys_removes_stale_motion_after_disconnect(self):
        listener = _bare_keyboard_listener()
        listener._update_key_state("w", 1)
        listener._update_key_state("a", 1)

        listener._clear_held_keys()

        assert listener.get_keys() == frozenset()
        assert listener.get_key() is None

    def test_listener_health_state_is_observable(self):
        listener = _bare_keyboard_listener()

        assert listener.is_connected()
        assert listener.fatal_error() is None

        listener._connected.clear()
        listener._fatal_error = "device read failed"

        assert not listener.is_connected()
        assert listener.fatal_error() == "device read failed"

    def test_unexpected_listener_thread_error_is_fail_closed(self):
        class FailingDevice:
            path = "/dev/input/fake-keyboard"

            @staticmethod
            def read_loop():
                raise RuntimeError("unexpected read failure")
                yield

        listener = _bare_keyboard_listener()
        listener.device = FailingDevice()
        listener._update_key_state("w", 1)

        listener._listen_loop()

        assert not listener.is_connected()
        assert "unexpected read failure" in listener.fatal_error()
        assert listener.get_keys() == frozenset()


class TestEpisodeControlMode:
    def test_rejects_unknown_mode(self):
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="episode_control_mode"):
            _make_wrapper(env, episode_control_mode="unsafe")
        env.close()

    def test_collector_abort_keeps_episode_open(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="collector")
        w.reset()
        w.listener.press("Key.backspace")

        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert info["hil_event"] == "abort"
        assert info["termination_reason"] == "operator_abort"
        assert "operator_episode_end" not in info
        env.close()

    def test_online_abort_truncates_as_failure(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="online")
        w.reset()
        w.listener.press("Key.backspace")

        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        # Without human_stage_reward the legacy semantics stay: failure = 0.
        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "operator_abort"
        assert not bool(np.asarray(info["success_once"]).any())
        assert info["operator_episode_end"] is True
        assert info["operator_success"] is False
        assert info["hil_state"] == "model"
        env.close()

    def test_online_done_terminates_as_success(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="online")
        w.reset()
        w.listener.press("Key.enter")

        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert reward == 1.0
        assert terminated
        assert not truncated
        assert info["termination_reason"] == "operator_success"
        assert bool(np.asarray(info["success_once"]).all())
        assert info["operator_episode_end"] is True
        assert info["operator_success"] is True
        assert info["hil_state"] == "model"
        env.close()

    def test_chunk_boundary_mode_defers_success_and_keeps_actions_pass_through(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            episode_control_mode="online_chunk_boundary",
            allow_motion_intervention=False,
            safe_model_handoff=False,
        )
        w.reset()
        first_action = _DUMMY_POSE.copy()
        first_action[0] = 0.1
        w.set_chunk_boundary(False)
        w.listener.press("Key.enter")

        _, reward, terminated, truncated, info = w.step(first_action)

        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert info["operator_label_pending"] == "success"
        assert "operator_episode_end" not in info
        np.testing.assert_allclose(info["executed_action"], first_action)

        final_action = _DUMMY_POSE.copy()
        final_action[0] = 0.2
        w.set_chunk_boundary(True)
        _, reward, terminated, truncated, info = w.step(final_action)

        assert reward == 1.0
        assert terminated
        assert not truncated
        assert info["termination_reason"] == "operator_success"
        assert info["operator_episode_end"] is True
        assert info["operator_success"] is True
        assert info["reward_label_valid"] is True
        np.testing.assert_allclose(info["executed_action"], final_action)
        with pytest.raises(RuntimeError, match="episode"):
            w.step(_DUMMY_POSE.copy())
        env.close()

    def test_chunk_boundary_mode_defers_failure_to_boundary(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            episode_control_mode="online_chunk_boundary",
            allow_motion_intervention=False,
            safe_model_handoff=False,
        )
        w.reset()
        w.set_chunk_boundary(False)
        w.listener.press("Key.backspace")

        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert info["operator_label_pending"] == "failure"

        w.set_chunk_boundary(True)
        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        # Without human_stage_reward the legacy semantics stay: failure = 0.
        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "operator_failure"
        assert info["operator_episode_end"] is True
        assert info["operator_success"] is False
        assert info["reward_label_valid"] is True
        env.close()

    def test_chunk_boundary_mode_first_label_wins(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            episode_control_mode="online_chunk_boundary",
            allow_motion_intervention=False,
            safe_model_handoff=False,
        )
        w.reset()
        w.set_chunk_boundary(False)
        w.listener.press("Key.enter")
        w.step(_DUMMY_POSE.copy())

        w.set_chunk_boundary(False)
        w.listener.press("Key.backspace")
        _, _, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert not terminated
        assert not truncated
        assert info["operator_label_pending"] == "success"
        assert info["operator_label_conflict_ignored"] is True

        w.set_chunk_boundary(True)
        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())
        assert reward == 1.0
        assert terminated
        assert not truncated
        env.close()

    def test_chunk_boundary_reset_clears_pending_label(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            episode_control_mode="online_chunk_boundary",
            allow_motion_intervention=False,
            safe_model_handoff=False,
        )
        w.reset()
        w.set_chunk_boundary(False)
        w.listener.press("Key.enter")
        w.step(_DUMMY_POSE.copy())

        w.reset()
        w.set_chunk_boundary(True)
        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert "operator_label_pending" not in info
        env.close()

    def test_online_quit_truncates_and_requests_shutdown(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="online")
        w.reset()
        w.listener.press("Key.esc")

        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "operator_quit"
        assert info["quit_program"] is True
        assert info["operator_episode_end"] is True
        assert info["operator_success"] is False
        env.close()

    def test_chunk_boundary_quit_is_immediate_hold_with_invalid_label(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            episode_control_mode="online_chunk_boundary",
            allow_motion_intervention=False,
            safe_model_handoff=False,
        )
        w.reset()
        unsafe_model_action = _DUMMY_POSE.copy()
        unsafe_model_action[:3] = [0.4, 0.4, 0.4]
        w.set_chunk_boundary(False)
        w.listener.press("Key.esc")

        _, reward, terminated, truncated, info = w.step(unsafe_model_action)

        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "operator_quit"
        assert info["reward_label_valid"] is False
        np.testing.assert_allclose(info["executed_action"], _DUMMY_POSE)
        env.close()

    def test_online_reset_can_wait_for_start_key(self):
        env = _dummy_pose_env()
        listener = FakeListener(press_sequence=["y"])
        w = _make_wrapper(
            env,
            listener=listener,
            episode_control_mode="online",
            wait_for_start_on_reset=True,
        )

        obs, _ = w.reset()

        assert obs is not None
        assert listener.pop_pressed_keys() == []
        env.close()

    def test_start_gate_can_be_cancelled_with_quit_key(self):
        env = _dummy_pose_env()
        listener = FakeListener(press_sequence=["Key.esc"])
        w = _make_wrapper(
            env,
            listener=listener,
            wait_for_start_on_reset=True,
        )

        with pytest.raises(RuntimeError, match="cancelled"):
            w.reset()

        env.close()

    def test_start_gate_times_out_while_keyboard_is_disconnected(self):
        env = _dummy_pose_env()
        listener = FakeListener()
        listener._connected = False
        w = _make_wrapper(
            env,
            listener=listener,
            wait_for_start_on_reset=True,
            start_gate_timeout_s=0.01,
        )

        with pytest.raises(TimeoutError, match="keyboard_connected=False"):
            w.reset()

        env.close()

    def test_start_gate_fails_on_permanent_listener_error(self):
        env = _dummy_pose_env()
        listener = FakeListener()
        listener._fatal_error = "read failed"
        w = _make_wrapper(
            env,
            listener=listener,
            wait_for_start_on_reset=True,
        )

        with pytest.raises(RuntimeError, match="read failed"):
            w.reset()

        env.close()

    def test_online_episode_latch_blocks_action_until_reset(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="online")
        w.reset()
        w.listener.press("Key.backspace")
        w.step(_DUMMY_POSE.copy())

        with pytest.raises(RuntimeError, match="episode"):
            w.step(_DUMMY_POSE.copy())

        assert env.num_steps == 1
        w.reset()
        w.step(_DUMMY_POSE.copy())
        assert env.num_steps == 1
        env.close()

    def test_collector_abort_event_is_one_shot(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="collector")
        w.reset()
        w.listener.press("Key.backspace")
        _, _, _, _, first_info = w.step(_DUMMY_POSE.copy())

        _, _, _, _, second_info = w.step(_DUMMY_POSE.copy())

        assert first_info["hil_event"] == "abort"
        assert "hil_event" not in second_info
        env.close()

    def test_failed_reset_keeps_episode_latched(self, monkeypatch):
        env = _dummy_pose_env()
        w = _make_wrapper(env, episode_control_mode="online")
        w.reset()
        w.listener.press("Key.enter")
        w.step(_DUMMY_POSE.copy())

        def fail_reset(**kwargs):
            raise RuntimeError("reset failed")

        monkeypatch.setattr(w.env, "reset", fail_reset)
        with pytest.raises(RuntimeError, match="reset failed"):
            w.reset()
        with pytest.raises(RuntimeError, match="episode"):
            w.step(_DUMMY_POSE.copy())
        env.close()

    @pytest.mark.parametrize(
        ("fatal_error", "expected_reason"),
        [
            (None, "keyboard_disconnected"),
            ("read failed", "keyboard_listener_error"),
        ],
    )
    def test_online_keyboard_failure_executes_hold_and_ends_episode(
        self, fatal_error, expected_reason
    ):
        env = _dummy_pose_env()
        listener = FakeListener()
        w = _make_wrapper(
            env,
            listener=listener,
            episode_control_mode="online",
        )
        w.reset()
        listener._connected = False
        listener._fatal_error = fatal_error
        unsafe_model_action = _DUMMY_POSE.copy()
        unsafe_model_action[:3] = [0.4, 0.4, 0.4]

        _, reward, terminated, truncated, info = w.step(unsafe_model_action)

        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["termination_reason"] == expected_reason
        assert info["keyboard_connected"] is False
        np.testing.assert_allclose(info["executed_action"], _DUMMY_POSE)
        with pytest.raises(RuntimeError, match="episode"):
            w.step(unsafe_model_action)
        env.close()

    def test_collector_keyboard_failure_aborts_without_flushing_as_done(self):
        env = _dummy_pose_env()
        listener = FakeListener()
        w = _make_wrapper(
            env,
            listener=listener,
            episode_control_mode="collector",
        )
        w.reset()
        listener._connected = False

        _, _, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert not terminated
        assert not truncated
        assert info["hil_event"] == "abort"
        assert info["termination_reason"] == "keyboard_disconnected"
        assert "operator_episode_end" not in info
        env.close()


class TestGetPoseState:
    def test_dummy_returns_finite_8dim_unit_quaternion(self):
        env = _dummy_pose_env()
        pose = env.get_pose_state()
        assert pose.shape == (8,)
        assert np.isfinite(pose).all()
        assert np.isclose(np.linalg.norm(pose[3:7]), 1.0)
        env.close()

    def test_dummy_pose_matches_env_observation(self):
        env = _dummy_pose_env()
        env.reset()
        pose = env.get_pose_state()
        np.testing.assert_allclose(pose, _DUMMY_POSE, atol=1e-6)
        env.close()

    def test_joint_mode_raises(self):
        env = DobotEnv(
            DobotRobotConfig(
                is_dummy=True,
                action_mode="joint",
                state_mode="joint",
                step_frequency=10_000.0,
            )
        )
        with pytest.raises(RuntimeError, match="pose"):
            env.get_pose_state()
        env.close()

    def test_does_not_touch_pose_tracker(self):
        """get_pose_state must not update PoseStateTracker (prev_state)."""
        env = _dummy_pose_env()
        env.reset()
        env.get_pose_state()
        env.get_pose_state()
        obs = env._get_observation()
        # First frame prev_state should still equal state.
        np.testing.assert_allclose(
            obs["prev_state"], obs["state"]["ee_pose_state"], atol=1e-6
        )
        env.close()


class TestResetServoSmoothing:
    def test_dummy_is_noop(self):
        env = _dummy_pose_env()
        env.reset_servo_smoothing()
        env.close()

    def test_real_calls_engage(self, monkeypatch):
        """When not dummy, reset_servo_smoothing delegates to controller.engage."""
        env = _dummy_pose_env()
        env.config.is_dummy = False
        calls = []

        class _FakeFuture:
            def wait(self):
                return [None]

        class _FakeController:
            def engage(self):
                calls.append("engage")
                return _FakeFuture()

        # DobotEnv only sets _controller in _setup_hardware, which is skipped
        # for dummy. We inject it manually.
        monkeypatch.setattr(env, "_controller", _FakeController(), raising=False)
        env.reset_servo_smoothing()
        assert calls == ["engage"]
        env.close()


# ===========================================================================
# Task 2: DobotKeyboardIntervention wrapper
# ===========================================================================


class TestWrapperInit:
    def test_non_pose_mode_raises(self):
        env = DobotEnv(
            DobotRobotConfig(
                is_dummy=True,
                action_mode="joint",
                state_mode="joint",
                step_frequency=10_000.0,
            )
        )
        with pytest.raises(ValueError, match="cartesian|pose"):
            _make_wrapper(env)

    def test_workspace_optional_without_bounds(self):
        """Workspace is optional: omitting both bounds should not raise."""
        env = _dummy_pose_env()
        w = DobotKeyboardIntervention(env, listener=FakeListener())
        assert w._workspace_low is None
        assert w._workspace_high is None
        w.reset()
        w.step(_DUMMY_POSE.copy())
        env.close()

    def test_workspace_partial_bounds_raises(self):
        """Setting only one bound must raise (all-or-nothing)."""
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="both set or both None"):
            DobotKeyboardIntervention(
                env, listener=FakeListener(), workspace_low=np.array([0, 0, 0])
            )

    def test_workspace_low_geq_high_raises(self):
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="workspace"):
            DobotKeyboardIntervention(
                env,
                listener=FakeListener(),
                workspace_low=np.array([0.1, 0.1, 0.1]),
                workspace_high=np.array([0.1, 0.1, 0.1]),
            )

    def test_negative_delta_raises(self):
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="delta"):
            _make_wrapper(env, position_delta=-0.001)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"handoff_max_position_jump_m": 0.0},
            {"handoff_max_rotation_jump_deg": 0.0},
        ],
    )
    def test_non_positive_handoff_limit_raises(self, kwargs):
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="handoff_max"):
            _make_wrapper(env, **kwargs)
        env.close()

    def test_online_mode_cannot_disable_safe_model_handoff(self):
        env = _dummy_pose_env()
        with pytest.raises(ValueError, match="safe_model_handoff"):
            _make_wrapper(
                env,
                episode_control_mode="online",
                safe_model_handoff=False,
            )
        env.close()


class TestStateMachine:
    def test_model_passthrough_no_event(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env)
        w.reset()
        w.step(_DUMMY_POSE.copy())
        env.close()

    def test_h_enters_engage(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert info["hil_state"] == "engage"
        assert "intervene_action" in info
        assert bool(info["intervene_flag"].all())
        env.close()

    def test_m_defers_model_until_chunk_boundary(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("m")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert info["hil_state"] == "model_pending"
        assert bool(np.asarray(info["request_replan"]).item())

        stale = _DUMMY_POSE.copy()
        stale[0] = 0.05
        _, _, _, _, info = w.step(stale)
        assert info["hil_state"] == "model_pending"
        assert bool(np.asarray(info["handoff_hold"]).item())
        np.testing.assert_allclose(info["executed_action"], _DUMMY_POSE)

        w.complete_model_handoff()
        fresh = _DUMMY_POSE.copy()
        fresh[0] = 0.003
        _, _, _, _, info = w.step(fresh)
        assert info["hil_state"] == "model"
        assert not bool(np.asarray(info["handoff_hold"]).item())
        np.testing.assert_allclose(info["executed_action"], fresh)
        env.close()

    def test_unsafe_first_model_action_after_handoff_fails_closed(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.press("m")
        w.step(_DUMMY_POSE.copy())
        w.complete_model_handoff()

        unsafe = _DUMMY_POSE.copy()
        unsafe[0] = 0.05
        _, _, terminated, truncated, info = w.step(unsafe)

        assert not terminated
        assert truncated
        assert info["hil_state"] == "model_armed"
        assert bool(np.asarray(info["handoff_rejected"]).item())
        assert info["termination_reason"] == "unsafe_model_handoff"
        np.testing.assert_allclose(info["executed_action"], _DUMMY_POSE)
        with pytest.raises(RuntimeError, match="reset"):
            w.step(_DUMMY_POSE.copy())
        env.close()

    def test_non_finite_first_model_action_after_handoff_fails_closed(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.press("m")
        w.step(_DUMMY_POSE.copy())
        w.complete_model_handoff()

        non_finite = _DUMMY_POSE.copy()
        non_finite[0] = np.nan
        _, _, _, truncated, info = w.step(non_finite)

        assert truncated
        assert bool(np.asarray(info["handoff_rejected"]).item())
        np.testing.assert_allclose(info["executed_action"], _DUMMY_POSE)
        env.close()

    def test_model_to_model_no_intervene(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env)
        w.reset()
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert "intervene_flag" not in info
        env.close()


class TestTranslation:
    def test_w_decreases_x(self):
        """Mirror mode: w = operator's forward = robot's back = -X."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())  # w held
        action = info["intervene_action"]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] - 0.002)
        assert action[1] == pytest.approx(_DUMMY_POSE[1])
        env.close()

    def test_a_decreases_y(self):
        """Mirror mode: a = operator's left = robot's right = -Y."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("a")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())  # a held
        action = info["intervene_action"]
        assert action[1] == pytest.approx(_DUMMY_POSE[1] - 0.002)
        assert action[0] == pytest.approx(_DUMMY_POSE[0])
        env.close()

    def test_w_and_a_move_diagonally_at_bounded_total_speed(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.set_held_keys("w", "a")

        _, _, _, _, info = w.step(_DUMMY_POSE.copy())

        delta = info["intervene_action"][:3] - _DUMMY_POSE[:3]
        expected_axis_delta = -0.002 / np.sqrt(2.0)
        np.testing.assert_allclose(
            delta,
            [expected_axis_delta, expected_axis_delta, 0.0],
            atol=1e-9,
        )
        assert np.linalg.norm(delta) == pytest.approx(0.002)
        env.close()

    def test_opposite_translation_keys_cancel(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.set_held_keys("w", "s", "a", "d", "q", "e")

        _, _, _, _, info = w.step(_DUMMY_POSE.copy())

        np.testing.assert_allclose(
            info["intervene_action"][:3],
            _DUMMY_POSE[:3],
            atol=1e-12,
        )
        env.close()

    def test_releasing_one_key_keeps_remaining_direction_active(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.set_held_keys("w", "a")
        _, _, _, _, first_info = w.step(_DUMMY_POSE.copy())
        w.listener.set_held_keys("w")

        _, _, _, _, second_info = w.step(first_info["intervene_action"].copy())

        first = first_info["intervene_action"]
        second = second_info["intervene_action"]
        assert second[0] == pytest.approx(first[0] - 0.002)
        assert second[1] == pytest.approx(first[1])
        env.close()


class TestRotation:
    def test_i_updates_roll_wxyz_output(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), rotation_delta=0.02)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("i")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())  # i held
        action = info["intervene_action"]
        assert np.isclose(np.linalg.norm(action[3:7]), 1.0)
        expected = Rotation.from_euler("x", 0.02).as_quat()  # xyzw
        expected_wxyz = np.array([expected[3], expected[0], expected[1], expected[2]])
        np.testing.assert_allclose(action[3:7], expected_wxyz, atol=1e-10)
        env.close()

    def test_quaternion_no_sign_flip(self):
        """Consecutive rotations must not cause q ↔ -q jumps."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), rotation_delta=0.5)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("i")
        _, _, _, _, info1 = w.step(_DUMMY_POSE.copy())
        _, _, _, _, info2 = w.step(info1["intervene_action"].copy())
        q1 = info1["intervene_action"][3:7]
        q2 = info2["intervene_action"][3:7]
        assert np.dot(q1, q2) > 0


class TestGripper:
    def test_comma_closes_gripper(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held(",")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # Comma sets gripper to fully closed (0.0), absolute not incremental.
        assert action[7] == pytest.approx(0.0)

    def test_period_opens_gripper(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held(".")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # Period sets gripper to fully open (1.0), absolute not incremental.
        assert action[7] == pytest.approx(1.0)


class TestWorkspaceClamp:
    def test_xyz_clamped(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            listener=FakeListener(),
            position_delta=10.0,
            workspace_low=np.array([-0.1, -0.1, 0.0]),
            workspace_high=np.array([0.1, 0.1, 0.5]),
        )
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("d")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # Mirror mode: d = operator's right = robot's left = +Y; 0 + 10.0 clamped to workspace_high Y = 0.1
        assert action[1] == pytest.approx(0.1)


class TestBaseFrameRotation:
    """Verify base_frame_euler_deg rotates translation correctly."""

    def test_euler_zero_no_rotation(self):
        """Mirror mode, [0,0,0]: w decreases X (back), a decreases Y (right)."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 0])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] - 0.002)
        assert action[1] == pytest.approx(_DUMMY_POSE[1])
        env.close()

    def test_rz_90_w_decreases_y(self):
        """rz=90°: w (mirror -X=[-1,0,0]) → base -Y."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # [-1,0,0] rotated 90° around Z → [0,-1,0]
        assert action[0] == pytest.approx(_DUMMY_POSE[0], abs=1e-6)
        assert action[1] == pytest.approx(_DUMMY_POSE[1] - 0.002, abs=1e-6)
        env.close()

    def test_rz_90_a_increases_x(self):
        """rz=90°: a (mirror -Y=[0,-1,0]) → base +X."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("a")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # [0,-1,0] rotated 90° around Z → [1,0,0]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] + 0.002, abs=1e-6)
        assert action[1] == pytest.approx(_DUMMY_POSE[1], abs=1e-6)
        env.close()

    def test_rz_90_q_unchanged(self):
        """rz=90°: q still increases Z (Z-yaw doesn't rotate Z)."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("q")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[2] == pytest.approx(_DUMMY_POSE[2] + 0.002, abs=1e-6)
        env.close()

    def test_rx_180_q_decreases_z(self):
        """rx=180° (upside-down): q (physical +Z) → base -Z."""
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            listener=FakeListener(),
            base_frame_euler_deg=[180, 0, 0],
            workspace_low=np.array([-0.5, -0.5, -0.5]),  # allow negative Z
        )
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("q")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # Upside-down: physical up (+Z) → base -Z
        assert action[2] == pytest.approx(_DUMMY_POSE[2] - 0.002, abs=1e-6)
        env.close()

    def test_rotation_keys_unaffected(self):
        """Base rotation does not affect tool-frame rotation keys."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("i")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert np.isclose(np.linalg.norm(action[3:7]), 1.0)
        expected = Rotation.from_euler("x", 0.02).as_quat()
        expected_wxyz = np.array([expected[3], expected[0], expected[1], expected[2]])
        np.testing.assert_allclose(action[3:7], expected_wxyz, atol=1e-10)
        env.close()


class TestHumanStageReward:
    def test_keyboard_failure_when_enabled_is_reward_neutral(self):
        env = _dummy_pose_env()
        listener = FakeListener()
        w = _make_wrapper(
            env,
            listener=listener,
            episode_control_mode="online",
            human_stage_reward={"failure": -5.0, "success": 5.0},
        )
        w.reset()
        listener._connected = False
        listener._fatal_error = "keyboard_disconnected"

        _, rew, terminated, truncated, info = w.step(_DUMMY_POSE.copy())

        assert rew == 0.0  # safety abort: no stage/terminal reward
        assert not terminated
        assert truncated
        assert info["reward_label_valid"] is False
        env.close()

    def test_unsafe_handoff_when_enabled_is_reward_neutral(self):
        env = _dummy_pose_env()
        w = _make_wrapper(
            env,
            listener=FakeListener(),
            human_stage_reward={"failure": -5.0, "success": 5.0},
        )
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())
        w.listener.press("m")
        w.step(_DUMMY_POSE.copy())
        w.complete_model_handoff()

        unsafe = _DUMMY_POSE.copy()
        unsafe[0] = 0.05
        _, rew, terminated, truncated, info = w.step(unsafe)

        assert rew == 0.0  # safety abort: no stage/terminal reward
        assert not terminated
        assert truncated
        assert info["reward_label_valid"] is False
        env.close()


class TestEpisodeEvents:
    def test_enter_saves_episode(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("Key.enter")
        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())
        assert reward == 1.0
        assert terminated is True
        assert info["hil_event"] == "save"
        assert bool(info["success_once"].all())

    def test_backspace_aborts(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("Key.backspace")
        _, reward, terminated, truncated, info = w.step(_DUMMY_POSE.copy())
        assert info["hil_event"] == "abort"
        assert terminated is False

    def test_esc_quits(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("Key.esc")
        _, _, terminated, truncated, info = w.step(_DUMMY_POSE.copy())
        assert info["quit_program"] is True
        assert terminated is False


class TestModelActionValid:
    def test_set_and_read_model_action_valid(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env)
        w.reset()
        w.set_model_action_valid(True)
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert bool(info["model_action_valid"].all())
        assert w._model_action_valid is False

    def test_engage_frame_not_valid(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.set_model_action_valid(True)
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert not bool(info["model_action_valid"].any())


class TestReset:
    def test_reset_clears_state(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("Key.enter")
        w.step(_DUMMY_POSE.copy())  # save
        w.reset()
        assert w._state == "model"
        assert w._episode_save is False
        assert w._episode_abort is False
        assert w._quit_program is False
        env.close()


class TestStartInEngage:
    """Verify start_in_engage initializes target from real TCP, not zero pose."""

    def test_start_in_engage_initializes_from_current_pose(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), start_in_engage=True)
        w.reset()
        assert w._state == "engage"
        # Target must match the dummy pose, NOT zeros.
        np.testing.assert_allclose(w._target_position, _DUMMY_POSE[:3])
        np.testing.assert_allclose(w._target_quaternion_wxyz, _DUMMY_POSE[3:7])
        assert w._target_gripper == pytest.approx(_DUMMY_POSE[7])

    def test_start_in_engage_first_action_equals_pose(self):
        """The first ENGAGE action must equal the current TCP pose, not zeros."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), start_in_engage=True)
        w.reset()
        # No held key — action should be the current pose hold.
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        np.testing.assert_allclose(action, _DUMMY_POSE, atol=1e-6)

    def test_default_starts_in_model(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), start_in_engage=False)
        w.reset()
        assert w._state == "model"


# ===========================================================================
# Task 3: Factory and apply_dobot_wrappers
# ===========================================================================


class TestFactory:
    """Tests for create_dobot_pick_and_place_env and apply_dobot_wrappers."""

    def _env_cfg(
        self,
        use_keyboard: bool = False,
        use_in_dummy: bool = True,
        **kcfg_overrides,
    ) -> dict:
        kcfg = {
            "position_delta": 0.002,
            "rotation_delta": 0.02,
            "gripper_delta": 0.05,
            "workspace_low": list(_DUMMY_WORKSPACE_LOW),
            "workspace_high": list(_DUMMY_WORKSPACE_HIGH),
            "toggle_key": "h",
            "model_key": "m",
            "done_key": "Key.enter",
            "abort_key": "Key.backspace",
            "quit_keys": ["Key.esc"],
            "start_in_engage": False,
        }
        kcfg.update(kcfg_overrides)
        return {
            "use_keyboard_intervention": use_keyboard,
            "use_intervention_in_dummy": use_in_dummy,
            "keyboard_intervention": kcfg,
        }

    def test_no_keyboard_no_wrap(self):
        from rlinf.envs.realworld.dobot.tasks.pick_and_place import (
            create_dobot_pick_and_place_env,
        )

        env = create_dobot_pick_and_place_env(
            override_cfg={
                "is_dummy": True,
                "action_mode": "cartesian",
                "state_mode": "pose",
                "step_frequency": 10_000.0,
            },
            env_cfg=self._env_cfg(use_keyboard=False),
        )
        # No keyboard wrapper → unwrapped is DobotPickAndPlaceEnv.
        assert type(env).__name__ == "DobotPickAndPlaceEnv"
        env.close()

    def test_keyboard_wraps_dummy(self):
        from rlinf.envs.realworld.common.wrappers import (
            DobotKeyboardIntervention as DKI,
        )
        from rlinf.envs.realworld.dobot.tasks.pick_and_place import (
            create_dobot_pick_and_place_env,
        )

        env = create_dobot_pick_and_place_env(
            override_cfg={
                "is_dummy": True,
                "action_mode": "cartesian",
                "state_mode": "pose",
                "step_frequency": 10_000.0,
            },
            env_cfg=self._env_cfg(use_keyboard=True, use_in_dummy=True),
        )
        assert isinstance(env, DKI)
        env.close()

    def test_joint_mode_with_keyboard_raises(self):
        from rlinf.envs.realworld.dobot.tasks.pick_and_place import (
            create_dobot_pick_and_place_env,
        )

        with pytest.raises(ValueError, match="cartesian|pose"):
            create_dobot_pick_and_place_env(
                override_cfg={
                    "is_dummy": True,
                    "action_mode": "joint",
                    "state_mode": "joint",
                    "step_frequency": 10_000.0,
                },
                env_cfg=self._env_cfg(use_keyboard=True, use_in_dummy=True),
            )

    def test_no_workspace_works(self):
        """Factory should work without workspace (hardware guard is sufficient)."""
        from rlinf.envs.realworld.common.wrappers import (
            DobotKeyboardIntervention as DKI,
        )
        from rlinf.envs.realworld.dobot.tasks.pick_and_place import (
            create_dobot_pick_and_place_env,
        )

        cfg = self._env_cfg(use_keyboard=True, use_in_dummy=True)
        del cfg["keyboard_intervention"]["workspace_low"]
        del cfg["keyboard_intervention"]["workspace_high"]
        env = create_dobot_pick_and_place_env(
            override_cfg={
                "is_dummy": True,
                "action_mode": "cartesian",
                "state_mode": "pose",
                "step_frequency": 10_000.0,
            },
            env_cfg=cfg,
        )
        assert isinstance(env, DKI)
        assert env._workspace_low is None
        env.close()
