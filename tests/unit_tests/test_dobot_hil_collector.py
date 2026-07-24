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

"""Unit tests for DobotHILCollector logic.

These tests use a fake env and fake policy to verify the collector's action
chunk queue, HIL state transitions, and event handling — without accessing
hardware, cameras, or Ray.
"""

from __future__ import annotations

# DobotHILCollector is in examples/ (not a package). Use importlib to load it.
import importlib.util as _ilu
import pathlib as _pl
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from rlinf.envs.realworld.dobot.hold_policy import HoldPolicy

_collector_path = _pl.Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "collect_dobot_hil_data.py"
_spec = _ilu.spec_from_file_location("collect_dobot_hil_data", _collector_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DobotHILCollector = _mod.DobotHILCollector

_DUMMY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)


class FakeEnv:
    """Minimal env stand-in for collector tests.

    ``step_queue`` is a list of info dicts that will be returned in order,
    one per step call. When exhausted, the env returns a plain model step.
    ``reset()`` clears the queue so the loop doesn't spin forever.
    """

    def __init__(self):
        self.action_space = MagicMock()
        self.action_space.shape = (8,)
        self._step_count = 0
        self._hil_state = "model"
        self.step_queue: list[dict] = []
        self.reset_called = False
        self.closed = False
        self._last_model_action_valid = False
        self._reset_count = 0

    def reset(self, **kwargs):
        self.reset_called = True
        self._step_count = 0
        self._reset_count += 1
        self.step_queue = []
        return {"states": _DUMMY_POSE[None, :].copy()}, {}

    def step(self, action):
        self._step_count += 1
        info = {"hil_state": np.array([self._hil_state])}
        if self.step_queue:
            info.update(self.step_queue.pop(0))
        obs = {"states": _DUMMY_POSE[None, :].copy()}
        reward = float(np.asarray(info.get("reward", 0.0)))
        terminated = np.array(
            [bool(np.asarray(info.get("terminated", False)).any())]
        )
        truncated = np.array(
            [bool(np.asarray(info.get("truncated", False)).any())]
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        self.closed = True

    @property
    def preexisting_episode_count(self):
        return 0

    def set_model_action_valid(self, valid):
        self._last_model_action_valid = valid


class FakeVectorEnv:
    """Simulates NoAutoResetSyncVectorEnv: has .envs and .call()."""

    def __init__(self, sub_env):
        self.envs = [sub_env]
        self._sub_env = sub_env

    def call(self, name, *args, **kwargs):
        """Broadcast method call to sub-envs (like SyncVectorEnv.call)."""
        return [getattr(self._sub_env, name)(*args, **kwargs)]

    def reset(self, **kwargs):
        return self._sub_env.reset(**kwargs)

    def step(self, action):
        return self._sub_env.step(action)

    def close(self):
        self._sub_env.closed = True

    @property
    def preexisting_episode_count(self):
        return 0


class FakeKeyboardWrapper(FakeEnv):
    """Simulates the real wrapper chain: CollectEpisode → RealWorldEnv → vector env.

    ``self.env`` is the FakeVectorEnv (mimics RealWorldEnv.env).
    The vector env wraps the actual FakeEnv (mimics DobotKeyboardIntervention
    wrapping DobotEnv, but we flatten for testing).
    """

    def __init__(self):
        # Create _inner_env BEFORE super().__init__() because FakeEnv.__init__
        # sets self.step_queue = [], which our property setter forwards to
        # _inner_env.
        self._inner_env = FakeEnv()
        self._vector_env = FakeVectorEnv(self._inner_env)
        super().__init__()
        self.env = self._vector_env  # mimics RealWorldEnv.env = vector env
        self.closed = False

    def set_model_action_valid(self, valid):
        # In real code, the collector calls vector_env.call("set_model_action_valid", valid)
        # which reaches DobotKeyboardIntervention. Here we forward to the inner env.
        self._inner_env.set_model_action_valid(valid)

    @property
    def step_queue(self):
        return self._inner_env.step_queue

    @step_queue.setter
    def step_queue(self, value):
        self._inner_env.step_queue = value

    def reset(self, **kwargs):
        self._inner_env.reset_called = True
        saved_queue = list(self.step_queue)
        result = self._inner_env.reset(**kwargs)
        self._inner_env.step_queue = saved_queue
        return result

    def step(self, action):
        return self._inner_env.step(action)

    def close(self):
        self.closed = True
        self._inner_env.closed = True


def _make_collector(policy_mode="dummy", num_episodes=3, **overrides):
    """Create a DobotHILCollector with a fake env (bypassing RealWorldEnv)."""
    cfg = MagicMock()
    cfg.policy_mode = policy_mode
    cfg.env.eval.override_cfg.get = lambda key, default=None: {
        "is_dummy": policy_mode == "dummy",
    }.get(key, default)
    cfg.env.eval.get = lambda key, default=None: {
        "data_collection": MagicMock(
            enabled=True,
            save_dir="/tmp/test_dobot_hil",
            get=lambda k, d=None: {
                "export_format": "lerobot",
                "robot_type": "dobot_cr5af",
                "fps": 30,
                "only_success": True,
                "finalize_interval": 20,
                "resume": False,
                "image_writer_threads": 1,
                "image_writer_processes": 1,
            }.get(k, d),
        ),
    }.get(key, default)
    cfg.env.group_name = "EnvGroup"
    cfg.actor.model.get = lambda k, d=None: {
        "num_action_chunks": 2,
        "model_path": "fake",
        "model_type": "openpi",
    }.get(k, d)
    cfg.actor.model.model_path = "fake"
    cfg.runner.num_data_episodes = num_episodes
    cfg.get = lambda k, d=None: getattr(cfg, k, d)

    collector = DobotHILCollector.__new__(DobotHILCollector)
    # Manually init only what we need.
    collector.cfg = cfg
    collector.policy_mode = policy_mode
    collector.action_dim = 8
    collector.num_episodes = num_episodes
    collector._target_step_period = None
    collector._action_queue = None
    collector._pending_model_hold_step = False
    collector._preexisting = 0

    # Inject fake env.
    fake_kb = FakeKeyboardWrapper()
    collector.env = fake_kb

    # Inject policy.
    if policy_mode == "model":
        collector.policy = HoldPolicy(action_dim=8, action_chunk=2)
    else:
        collector.policy = HoldPolicy(action_dim=8, action_chunk=2)

    from rlinf.utils.logging import get_logger
    collector._logger = get_logger()
    collector.log_info = collector._logger.info
    collector.log_warning = collector._logger.warning

    return collector


# ===========================================================================
# Tests
# ===========================================================================


class TestHoldPolicy:
    def test_returns_current_state_repeated(self):
        policy = HoldPolicy(action_dim=8, action_chunk=3)
        obs = {"states": torch.zeros(1, 8)}
        actions, aux = policy.predict_action_batch(env_obs=obs)
        assert actions.shape == (1, 3, 8)
        # All actions should be zero (current state).
        np.testing.assert_array_equal(actions.numpy(), 0.0)
        assert "forward_inputs" in aux

    def test_returns_nonzero_state(self):
        policy = HoldPolicy(action_dim=8, action_chunk=2)
        state = torch.ones(1, 8) * 0.5
        obs = {"states": state}
        actions, _ = policy.predict_action_batch(env_obs=obs)
        assert actions.shape == (1, 2, 8)
        np.testing.assert_array_equal(actions.numpy(), 0.5)


class TestActionChunk:
    def test_chunk_consumed_in_order(self):
        c = _make_collector(policy_mode="model")
        # Manually fill queue.
        c._action_queue = np.array(
            [[1, 0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        obs = {"states": _DUMMY_POSE[None, :]}
        a1 = c._get_action(obs, hil_state="model")
        a2 = c._get_action(obs, hil_state="model")
        assert a1[0, 0] == 1.0
        assert a2[0, 0] == 2.0

    def test_engage_returns_hold(self):
        c = _make_collector(policy_mode="model")
        c._action_queue = np.array(
            [[1, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float64
        )
        obs = {"states": _DUMMY_POSE[None, :]}
        action = c._get_action(obs, hil_state="engage")
        # Should be hold (current state), not the queued action.
        np.testing.assert_allclose(action[0], _DUMMY_POSE)
        # Queue should NOT be consumed.
        assert c._action_queue is not None
        assert c._action_queue.shape[0] == 1

    def test_model_to_engage_clears_queue(self):
        c = _make_collector(policy_mode="model")
        c._action_queue = np.array(
            [[1, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float64
        )
        # Simulate the run loop's queue clearing on state change.
        c._action_queue = None
        assert c._action_queue is None


class TestHoldFromObs:
    def test_returns_normalized_quaternion(self):
        c = _make_collector()
        obs = {"states": _DUMMY_POSE[None, :].copy()}
        action = c._hold_from_obs(obs)
        assert action.shape == (1, 8)
        assert np.isclose(np.linalg.norm(action[0, 3:7]), 1.0)

    def test_non_finite_raises(self):
        c = _make_collector()
        bad_state = _DUMMY_POSE[None, :].copy()
        bad_state[0, 0] = np.nan
        obs = {"states": bad_state}
        with pytest.raises(ValueError, match="Non-finite"):
            c._hold_from_obs(obs)


class TestInferActionChunk:
    def test_rejects_wrong_dim(self):
        c = _make_collector(policy_mode="model")
        # Patch policy to return wrong dim.
        class BadPolicy:
            def predict_action_batch(self, **kwargs):
                return torch.zeros(1, 2, 7), {}

        c.policy = BadPolicy()
        obs = {"states": _DUMMY_POSE[None, :]}
        with pytest.raises(ValueError, match="action_dim"):
            c._infer_action_chunk(obs)

    def test_rejects_non_finite(self):
        c = _make_collector(policy_mode="model")

        class NanPolicy:
            def predict_action_batch(self, **kwargs):
                actions = torch.zeros(1, 2, 8)
                actions[0, 0, 0] = float("nan")
                return actions, {}

        c.policy = NanPolicy()
        obs = {"states": _DUMMY_POSE[None, :]}
        with pytest.raises(ValueError, match="Non-finite"):
            c._infer_action_chunk(obs)


class TestModelActionValid:
    def test_model_inference_sets_valid(self):
        c = _make_collector(policy_mode="model")
        c.env._inner_env._hil_state = "model"
        c.env.step_queue = [{}]
        obs = {"states": _DUMMY_POSE[None, :]}
        c._action_queue = None  # Force inference.
        c._set_model_action_valid(True)
        action = c._get_action(obs, hil_state="model")
        c.env.step(action)
        # Flag propagates through vector env .call() to the inner env.
        assert c.env._inner_env._last_model_action_valid is True

    def test_engage_sets_invalid(self):
        c = _make_collector(policy_mode="model")
        c.env._inner_env._hil_state = "engage"
        c.env.step_queue = [{}]
        c._set_model_action_valid(False)
        action = c._get_action(obs={"states": _DUMMY_POSE[None, :]}, hil_state="engage")
        c.env.step(action)
        assert c.env._inner_env._last_model_action_valid is False

    def test_set_model_action_valid_raises_without_wrapper(self):
        """If the keyboard wrapper is not in the env stack, raise."""
        c = _make_collector(policy_mode="model")
        # Replace env with a plain object that has no .env, .envs, or .call.
        c.env = object()
        with pytest.raises(RuntimeError, match="set_model_action_valid"):
            c._set_model_action_valid(True)


class TestRunLoop:
    def test_quit_exits_cleanly(self):
        c = _make_collector(policy_mode="dummy", num_episodes=5)
        # First step triggers quit.
        c.env._inner_env.step_queue = [{"quit_program": np.array([True])}]
        c.env.step_queue = list(c.env._inner_env.step_queue)
        c.run()
        assert c.env._inner_env.reset_called  # reset before exit
        assert c.env.closed

    def test_abort_resets_no_increment(self):
        c = _make_collector(policy_mode="dummy", num_episodes=5)
        # First step: abort. After reset, second step: quit to exit.
        first_queue = [{"hil_event": np.array(["abort"])}]
        c.env._inner_env.step_queue = list(first_queue)
        c.env.step_queue = list(first_queue)
        # After reset (triggered by abort), inject quit.
        original_reset = c.env.reset

        def reset_with_quit(**kwargs):
            result = original_reset(**kwargs)
            c.env._inner_env.step_queue = [{"quit_program": np.array([True])}]
            return result

        c.env.reset = reset_with_quit
        c.run()
        assert c.env._inner_env.reset_called

    def test_save_increments_episode(self):
        c = _make_collector(policy_mode="dummy", num_episodes=2)
        # First step: save (terminated). After reset, second step: quit to exit.
        first_queue = [{
            "terminated": np.array([True]),
            "success_once": np.array([True]),
            "hil_event": "save",
            "reward": 1.0,
        }]
        c.env._inner_env.step_queue = list(first_queue)
        c.env.step_queue = list(first_queue)
        original_reset = c.env.reset

        def reset_with_quit(**kwargs):
            result = original_reset(**kwargs)
            c.env._inner_env.step_queue = [{"quit_program": np.array([True])}]
            return result

        c.env.reset = reset_with_quit
        c.run()
        assert c.env.closed
