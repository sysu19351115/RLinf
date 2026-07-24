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
from typing import Sequence

import numpy as np

from rlinf.envs.realworld.common.wrappers.dobot_keyboard_intervention import (
    DobotKeyboardIntervention,
)
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv

_DUMMY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
_WS_LOW = np.array([-0.5, -0.5, 0.0])
_WS_HIGH = np.array([0.5, 0.5, 0.5])


class FakeListener:
    """Deterministic keyboard listener for testing."""

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
        self._presses.append(key)

    def set_held(self, key: str | None) -> None:
        self._held = key


def _make_env_stack():
    """Create the real wrapper stack: vector env → keyboard wrapper → dummy env."""
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            max_num_steps=100,
            step_frequency=10_000.0,
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
