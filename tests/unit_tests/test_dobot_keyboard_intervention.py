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

from collections import deque
from typing import Sequence

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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

    def __init__(self, press_sequence: Sequence[str] | None = None, held: str | None = None):
        self._presses: deque[str] = deque(press_sequence or [])
        self._held = held

    def pop_pressed_keys(self) -> list[str]:
        if self._presses:
            return [self._presses.popleft()]
        return []

    def get_key(self) -> str | None:
        return self._held

    def press(self, key: str) -> None:
        """Helper to inject a key press after construction."""
        self._presses.append(key)

    def set_held(self, key: str | None) -> None:
        """Helper to set the currently held key."""
        self._held = key


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

    def test_m_returns_to_model(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener())
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.press("m")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert info["hil_state"] == "model"
        env.close()

    def test_model_to_model_no_intervene(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env)
        w.reset()
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        assert "intervene_flag" not in info
        env.close()


class TestTranslation:
    def test_w_increases_x(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), position_delta=0.002)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())  # w held
        action = info["intervene_action"]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] + 0.002)
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
        w = _make_wrapper(env, listener=FakeListener(), gripper_delta=0.05)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held(",")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[7] == pytest.approx(_DUMMY_POSE[7] - 0.05)

    def test_period_opens_gripper(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), gripper_delta=0.05)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held(".")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[7] == pytest.approx(_DUMMY_POSE[7] + 0.05)

    def test_gripper_clamped_0_1(self):
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), gripper_delta=0.5)
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w._target_gripper = 0.0
        w.listener.set_held(",")
        _, _, _, _, info = w.step(
            np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        )
        action = info["intervene_action"]
        assert action[7] == 0.0


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
        # 'd' decreases Y; 0 - 10.0 clamped to workspace_low Y = -0.1
        assert action[1] == pytest.approx(-0.1)


class TestBaseFrameRotation:
    """Verify base_frame_euler_deg rotates translation correctly."""

    def test_euler_zero_no_rotation(self):
        """Default [0,0,0]: w increases X, a increases Y (unchanged)."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 0])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] + 0.002)
        assert action[1] == pytest.approx(_DUMMY_POSE[1])
        env.close()

    def test_rz_90_w_increases_y(self):
        """rz=90°: w (physical +X) → base +Y."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("w")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        # cos(90°)≈0 (float), sin(90°)=1 → w adds to Y, not X
        assert action[0] == pytest.approx(_DUMMY_POSE[0], abs=1e-6)
        assert action[1] == pytest.approx(_DUMMY_POSE[1] + 0.002, abs=1e-6)
        env.close()

    def test_rz_90_a_decreases_x(self):
        """rz=90°: a (physical +Y) → base -X."""
        env = _dummy_pose_env()
        w = _make_wrapper(env, listener=FakeListener(), base_frame_euler_deg=[0, 0, 90])
        w.reset()
        w.listener.press("h")
        w.step(_DUMMY_POSE.copy())  # enter engage
        w.listener.set_held("a")
        _, _, _, _, info = w.step(_DUMMY_POSE.copy())
        action = info["intervene_action"]
        assert action[0] == pytest.approx(_DUMMY_POSE[0] - 0.002, abs=1e-6)
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
            workspace_low=np.array([-0.5, -0.5, -0.5]),   # allow negative Z
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
        w = _make_wrapper(
            env, listener=FakeListener(), start_in_engage=True
        )
        w.reset()
        assert w._state == "engage"
        # Target must match the dummy pose, NOT zeros.
        np.testing.assert_allclose(w._target_position, _DUMMY_POSE[:3])
        np.testing.assert_allclose(w._target_quaternion_wxyz, _DUMMY_POSE[3:7])
        assert w._target_gripper == pytest.approx(_DUMMY_POSE[7])

    def test_start_in_engage_first_action_equals_pose(self):
        """The first ENGAGE action must equal the current TCP pose, not zeros."""
        env = _dummy_pose_env()
        w = _make_wrapper(
            env, listener=FakeListener(), start_in_engage=True
        )
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

