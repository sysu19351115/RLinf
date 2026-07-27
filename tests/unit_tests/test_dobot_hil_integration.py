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

"""Integration test: model_action_valid propagation through real wrapper stack.

This test constructs the real wrapper hierarchy:
    NoAutoResetSyncVectorEnv → DobotKeyboardIntervention → DobotEnv (dummy)

and verifies that ``set_model_action_valid`` called via the vector env's
``call()`` method reaches the DobotKeyboardIntervention wrapper, and that the
flag appears in the step's info dict with the correct value for MODEL vs
ENGAGE frames.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Sequence

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.wrappers.dobot_keyboard_intervention import (
    DobotKeyboardIntervention,
)
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.wrappers.collect_episode import resolve_collection_save_dir

_DUMMY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
_WS_LOW = np.array([-0.5, -0.5, 0.0])
_WS_HIGH = np.array([0.5, 0.5, 0.5])


class FakeListener:
    """Deterministic keyboard listener for testing."""

    def __init__(
        self, press_sequence: Sequence[str] | None = None, held: str | None = None
    ):
        self._presses: deque[str] = deque(press_sequence or [])
        self._held = held

    def pop_pressed_keys(self) -> list[str]:
        if self._presses:
            return [self._presses.popleft()]
        return []

    def get_key(self) -> str | None:
        return self._held

    def press(self, key: str) -> None:
        self._presses.append(key)

    def set_held(self, key: str | None) -> None:
        self._held = key


def _make_env_stack(gripper_relative_threshold: float | None = None):
    """Create the real wrapper stack: vector env → keyboard wrapper → dummy env."""
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            max_num_steps=100,
            step_frequency=10_000.0,
            gripper_relative_threshold=gripper_relative_threshold,
        )
    )
    kb_wrapper = DobotKeyboardIntervention(
        env,
        position_delta=0.002,
        rotation_delta=0.02,
        gripper_delta=0.05,
        workspace_low=_WS_LOW,
        workspace_high=_WS_HIGH,
        listener=FakeListener(),
    )
    venv = NoAutoResetSyncVectorEnv([lambda: kb_wrapper])
    return venv, kb_wrapper


def _make_realworld_stack(
    episode_control_mode: str = "collector",
    *,
    manual_episode_control_only: bool = True,
    ignore_terminations: bool = False,
    auto_reset: bool = False,
):
    venv, kb = _make_env_stack()
    kb._episode_control_mode = episode_control_mode
    cfg = OmegaConf.create(
        {
            "total_num_envs": 1,
            "main_image_key": "cam_left_wrist",
            "max_episode_steps": 100,
            "auto_reset": auto_reset,
            "ignore_terminations": ignore_terminations,
            "manual_episode_control_only": manual_episode_control_only,
        }
    )
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.env = venv
    env.cfg = cfg
    env.num_envs = 1
    env.main_image_key = "cam_left_wrist"
    env.manual_episode_control_only = manual_episode_control_only
    env.auto_reset = auto_reset
    env.ignore_terminations = ignore_terminations
    env._episode_needs_reset = False
    env._elapsed_steps = np.zeros(1, dtype=np.int64)
    env.prev_step_reward = np.zeros(1, dtype=np.float32)
    env.returns = np.zeros(1, dtype=np.float32)
    env.success_once = np.zeros(1, dtype=bool)
    env.fail_once = np.zeros(1, dtype=bool)
    env.intervened_once = np.zeros(1, dtype=bool)
    env.intervened_steps = np.zeros(1, dtype=np.int64)
    env._is_start = True
    env.task_descriptions = ["Pick up an object with the hole and hang it on a hook."]
    env.use_fixed_reset_state_ids = False
    env.reset_state_ids = None
    return env, venv, kb


class TestModelActionValidPropagation:
    """Verify model_action_valid propagates through the real wrapper stack."""

    def test_call_reaches_keyboard_wrapper(self):
        """vector_env.call('set_model_action_valid', True) reaches the wrapper."""
        venv, kb = _make_env_stack()
        venv.reset()
        venv.call("set_model_action_valid", True)
        assert kb._model_action_valid is True

        venv.call("set_model_action_valid", False)
        assert kb._model_action_valid is False
        venv.close()

    def test_model_frame_valid_true(self):
        """A MODEL-mode inference frame has model_action_valid=True in info."""
        venv, kb = _make_env_stack()
        venv.reset()
        kb.set_model_action_valid(True)
        action = _DUMMY_POSE[None, :].copy()
        _, _, _, _, info = venv.step(action)
        # info is dict-of-arrays from vector env; check env 0.
        assert bool(np.asarray(info["model_action_valid"]).any())
        venv.close()

    def test_engage_frame_valid_false(self):
        """An ENGAGE frame has model_action_valid=False even if set True."""
        venv, kb = _make_env_stack()
        venv.reset()
        # Enter ENGAGE.
        kb.listener.press("h")
        venv.step(_DUMMY_POSE[None, :].copy())
        # Now in ENGAGE. Even if collector sets valid=True, wrapper overrides.
        kb.set_model_action_valid(True)
        _, _, _, _, info = venv.step(_DUMMY_POSE[None, :].copy())
        assert not bool(np.asarray(info["model_action_valid"]).any())
        venv.close()

    def test_engage_action_matches_binary_command_sent_by_environment(self):
        venv, kb = _make_env_stack(gripper_relative_threshold=0.2)
        venv.reset()
        kb.gripper_delta = 0.3
        kb.listener.press("h")
        kb.listener.set_held(",")

        _, _, _, _, info = venv.step(_DUMMY_POSE[None, :].copy())

        executed = np.asarray(info["executed_action"])[0]
        intervene = np.asarray(info["intervene_action"])[0]
        assert executed[-1] == 0.0
        np.testing.assert_array_equal(intervene, executed)
        venv.close()


class TestHGDAggerRealWorldPropagation:
    def test_full_intervention_chunk_exposes_flags_actions_and_prev_state(self):
        env, venv, kb = _make_realworld_stack()
        env.reset()
        kb.listener.press("h")
        kb.listener.set_held("w")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        obs_list, _, _, _, infos_list = env.chunk_step(chunk)

        assert len(obs_list) == 4
        assert "prev_states" in obs_list[-1]
        assert infos_list[-1]["intervene_flag"].shape == (1, 4)
        assert bool(infos_list[-1]["intervene_flag"].all())
        assert infos_list[-1]["intervene_action"].shape == (1, 4 * 8)
        venv.close()

    def test_partial_intervention_chunk_keeps_exact_step_mask(self):
        env, venv, kb = _make_realworld_stack()
        env.reset()
        kb.listener.press("h")
        kb.listener.press("m")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        _, _, _, _, infos_list = env.chunk_step(chunk)

        np.testing.assert_array_equal(
            infos_list[-1]["intervene_flag"].numpy(),
            np.array([[True, True, False, False]]),
        )
        venv.close()

    def test_online_termination_skips_remaining_chunk_actions(self):
        env, venv, kb = _make_realworld_stack(episode_control_mode="online")
        env.reset()
        kb.listener.press("Key.enter")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        _, rewards, terminations, truncations, infos_list = env.chunk_step(chunk)

        assert kb.unwrapped.num_steps == 1
        assert rewards.shape == (1, 4)
        assert terminations.shape == (1, 4)
        assert truncations.shape == (1, 4)
        np.testing.assert_array_equal(
            terminations.numpy(), np.array([[True, False, False, False]])
        )
        assert infos_list[-1]["skipped_action_steps"] == 3
        np.testing.assert_array_equal(
            infos_list[-1]["executed_action_mask"].numpy(),
            np.array([[True, False, False, False]]),
        )
        venv.close()

    @pytest.mark.parametrize(
        ("key", "expected_termination", "expected_truncation"),
        [
            ("Key.enter", True, False),
            ("Key.backspace", False, True),
            ("Key.esc", False, True),
        ],
    )
    @pytest.mark.parametrize("manual_episode_control_only", [False, True])
    @pytest.mark.parametrize("ignore_terminations", [False, True])
    def test_online_operator_end_survives_realworld_filters(
        self,
        key,
        expected_termination,
        expected_truncation,
        manual_episode_control_only,
        ignore_terminations,
    ):
        env, venv, kb = _make_realworld_stack(
            episode_control_mode="online",
            manual_episode_control_only=manual_episode_control_only,
            ignore_terminations=ignore_terminations,
        )
        env.reset()
        kb.listener.press(key)
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        _, _, terminations, truncations, infos_list = env.chunk_step(chunk)

        assert kb.unwrapped.num_steps == 1
        assert bool(terminations.any()) is expected_termination
        assert bool(truncations.any()) is expected_truncation
        assert bool(np.asarray(infos_list[-1]["operator_episode_end"]).any())
        venv.close()

    def test_collector_abort_does_not_end_chunk(self):
        env, venv, kb = _make_realworld_stack(episode_control_mode="collector")
        env.reset()
        kb.listener.press("Key.backspace")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        _, _, terminations, truncations, _ = env.chunk_step(chunk)

        assert kb.unwrapped.num_steps == 4
        assert not bool(terminations.any())
        assert not bool(truncations.any())
        venv.close()

    def test_terminal_episode_rejects_next_chunk_until_reset(self):
        env, venv, kb = _make_realworld_stack(episode_control_mode="online")
        env.reset()
        kb.listener.press("Key.backspace")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)
        env.chunk_step(chunk)

        with pytest.raises(RuntimeError, match="reset"):
            env.chunk_step(chunk)

        assert kb.unwrapped.num_steps == 1
        env.reset()
        env.chunk_step(chunk[:, :1])
        assert kb.unwrapped.num_steps == 1
        venv.close()

    def test_padding_entries_do_not_alias(self):
        env, venv, kb = _make_realworld_stack(episode_control_mode="online")
        env.reset()
        kb.listener.press("Key.enter")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        obs_list, _, _, _, infos_list = env.chunk_step(chunk)

        assert len({id(obs) for obs in obs_list}) == 4
        assert len({id(info) for info in infos_list}) == 4
        infos_list[1]["padding_probe"] = True
        assert "padding_probe" not in infos_list[2]
        venv.close()

    def test_auto_reset_success_allows_a_fresh_next_chunk(self):
        env, venv, kb = _make_realworld_stack(
            episode_control_mode="online",
            auto_reset=True,
        )
        env.reset()
        kb.listener.press("Key.enter")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        env.chunk_step(chunk)
        env.chunk_step(chunk[:, :1])

        assert kb.unwrapped.num_steps == 1
        venv.close()

    def test_online_quit_never_auto_resets_into_another_chunk(self):
        env, venv, kb = _make_realworld_stack(
            episode_control_mode="online",
            auto_reset=True,
        )
        env.reset()
        kb.listener.press("Key.esc")
        chunk = np.repeat(_DUMMY_POSE[None, None, :], 4, axis=1)

        _, _, _, truncations, infos_list = env.chunk_step(chunk)

        assert bool(truncations.any())
        assert bool(np.asarray(infos_list[-1]["operator_shutdown_requested"]).any())
        with pytest.raises(RuntimeError, match="reset"):
            env.chunk_step(chunk)
        assert kb.unwrapped.num_steps == 1
        venv.close()


class TestCollectionSessionDirectory:
    def test_creates_timestamped_directory_and_collision_suffix(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "save_dir": str(tmp_path / "collected_data"),
                "create_session_dir": True,
                "session_name_format": "%Y%m%d_%H%M%S",
                "resume": False,
            }
        )
        now = datetime(2026, 7, 24, 20, 30, 15)

        first = resolve_collection_save_dir(cfg, now=now)
        second = resolve_collection_save_dir(cfg, now=now)

        assert first.endswith("20260724_203015")
        assert second.endswith("20260724_203015_01")
        assert (tmp_path / "collected_data" / "20260724_203015").is_dir()
        assert (tmp_path / "collected_data" / "20260724_203015_01").is_dir()

    def test_resume_requires_explicit_existing_session(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "save_dir": str(tmp_path / "collected_data"),
                "create_session_dir": True,
                "resume": True,
            }
        )
        with pytest.raises(ValueError, match="create_session_dir=false"):
            resolve_collection_save_dir(cfg)

        session = tmp_path / "collected_data" / "20260724_203015"
        session.mkdir(parents=True)
        cfg.create_session_dir = False
        cfg.save_dir = str(session)
        assert resolve_collection_save_dir(cfg) == str(session)
