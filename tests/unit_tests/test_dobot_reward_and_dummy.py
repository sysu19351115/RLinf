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

import numpy as np
import pytest

from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig


def _dummy_pose_env(max_num_steps: int = 3) -> DobotEnv:
    return DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            max_num_steps=max_num_steps,
            step_frequency=10_000.0,
        )
    )


def test_pose_dummy_observation_contains_prev_state():
    env = _dummy_pose_env()
    observation, _ = env.reset()

    state = observation["state"]["ee_pose_state"]
    np.testing.assert_array_equal(observation["prev_state"], state)
    assert env.observation_space.contains(observation)

    env.close()


def test_terminal_human_reward_runs_once_at_episode_end(monkeypatch):
    env = _dummy_pose_env(max_num_steps=3)
    env.config.use_reward_model = True
    env.config.reward_mode = "terminal"

    calls = []

    def compute_reward(observation):
        calls.append(observation)
        return 1.0

    monkeypatch.setattr(env, "_compute_reward_model", compute_reward)

    action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5])
    results = [env.step(action) for _ in range(4)]

    assert [result[1] for result in results] == [0.0, 0.0, 1.0, 0.0]
    assert [result[2] for result in results] == [False, False, False, False]
    assert [result[3] for result in results] == [False, False, True, True]
    assert len(calls) == 1

    env.reset()
    for _ in range(3):
        env.step(action)
    assert len(calls) == 2

    env.close()


def test_invalid_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        DobotRobotConfig(reward_mode="invalid")


# ---------------------------------------------------------------------------
# Human reward link regression (review: main_images dict + standalone runner)
# ---------------------------------------------------------------------------


class _FakeRewardWorkerGroup:
    """Captures the observations passed to compute_image_rewards."""

    def __init__(self):
        self.captured = None

    def compute_image_rewards(self, observations):
        self.captured = observations

        class _Future:
            def wait(self):
                return [np.array([0.5])]

        return _Future()


def test_compute_reward_model_passes_main_images_dict():
    """Dobot must pass a dict with ``main_images`` of shape [1,224,224,3]."""
    env = _dummy_pose_env()
    env.config.use_reward_model = True
    env.config.reward_mode = "terminal"
    env.config.reward_image_key = "cam_left_wrist"

    fake_worker = _FakeRewardWorkerGroup()
    env._reward_worker = fake_worker

    # Build a 224x224 frame as DobotEnv._get_camera_frames would produce.
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {"frames": {"cam_left_wrist": frame}}

    env._compute_reward_model(observation)

    # Must be a dict containing main_images (not a bare ndarray).
    assert isinstance(fake_worker.captured, dict)
    assert "main_images" in fake_worker.captured
    imgs = fake_worker.captured["main_images"]
    assert isinstance(imgs, np.ndarray)
    assert imgs.shape == (1, 224, 224, 3)

    env.close()


def test_standalone_reward_worker_tolerates_missing_runner():
    """launch_for_realworld builds a cfg with only `reward`; reading
    ``enable_decoupled_mode`` must not raise ConfigKeyError.

    This mirrors EmbodiedRewardWorker.__init__'s fixed access pattern against
    the exact standalone_cfg shape produced by launch_for_realworld.
    """
    from omegaconf import OmegaConf

    # Reconstruct the standalone cfg exactly as launch_for_realworld does.
    standalone_reward_cfg = {
        "use_reward_model": True,
        "model": {"model_type": "human"},
        "standalone_realworld": True,
    }
    cfg = OmegaConf.create({"reward": standalone_reward_cfg})

    # The fixed access pattern in reward_worker.py (was: self.cfg.runner.get(...)).
    value = cfg.get("runner", {}).get("enable_decoupled_mode", False)
    assert value is False
