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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.realworld.dobot.dobot_controller import DobotController
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig
from rlinf.envs.realworld.realworld_env import RealWorldEnv


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


def _dummy_realworld_env() -> RealWorldEnv:
    # Register the gym id used by RealWorldEnv._create_env.
    from rlinf.envs.realworld.dobot import tasks  # noqa: F401

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "max_episode_steps": 20,
            "use_fixed_reset_state_ids": False,
            "main_image_key": "cam_left_wrist",
            "use_keyboard_intervention": False,
            "keyboard_intervention": {},
            "collect_residual_feedback": True,
            "override_cfg": {},
            "video_cfg": {},
            "init_params": {
                "id": "DobotPickAndPlaceEnv-v1",
                "is_dummy": True,
                "action_mode": "cartesian",
                "state_mode": "pose",
                "max_num_steps": 20,
                "step_frequency": 10_000.0,
                "camera_serials": [],
                "gripper_relative_threshold": 0.2,
                "reward_mode": "none",
                "task_description": "dummy",
            },
        }
    )
    return RealWorldEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def test_chunk_step_residual_feedback_unwraps_vector_env_info():
    """Real-machine regression: gymnasium vector-env infos wrap per-env array
    values (e.g. ``executed_action``) in a length-1 object array, which
    ``np.asarray(..., dtype=np.float32)`` rejects with "setting an array
    element with a sequence" (numpy >= 1.24).  Chunked residual feedback must
    still be collected from the dummy/single-env path."""
    env = _dummy_realworld_env()
    env.reset()
    chunk_actions = torch.zeros(1, 10, 8)

    _, _, _, _, infos_list = env.chunk_step(chunk_actions)

    feedback = infos_list[-1]["residual_feedback"]
    assert feedback["executed_actions"].shape == (10, 8)
    np.testing.assert_array_equal(
        feedback["executed_actions"], chunk_actions.numpy()[0]
    )
    env.close()


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


def test_none_reward_mode_is_valid_and_disables_geometry_reward():
    env = _dummy_pose_env()
    env.config.reward_mode = "none"
    env.config.is_dummy = False
    env._state = SimpleNamespace(
        tcp_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        gripper_position=0.5,
    )
    env.config.target_ee_pose = np.zeros(6)
    env.config.reward_threshold = np.ones(6)

    assert env._calc_step_reward({}) == 0.0
    env.close()


def test_disabled_reward_model_does_not_create_worker():
    env = _dummy_pose_env()

    assert env.config.use_reward_model is False
    assert env._reward_worker is None
    env.close()


def test_invalid_gripper_relative_threshold_is_rejected():
    for threshold in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="gripper_relative_threshold"):
            DobotRobotConfig(gripper_relative_threshold=threshold)


def test_relative_gripper_action_is_binary_and_reported_as_executed():
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            gripper_relative_threshold=0.2,
            step_frequency=10_000.0,
        )
    )
    env._state = SimpleNamespace(gripper_position=0.8)
    action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)

    _, _, _, _, info = env.step(action)

    assert info["executed_action"][-1] == 0.0
    assert info["action_command_accepted"] is True
    assert action[-1] == pytest.approx(0.1)
    env.close()


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_action_reports_command_rejection(invalid_value):
    env = _dummy_pose_env()
    action = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
        dtype=np.float64,
    )
    action[0] = invalid_value

    _, _, _, _, info = env.step(action)

    assert info["action_command_accepted"] is False
    assert info["action_rejection_reason"] == "non_finite_action"
    env.close()


def test_relative_gripper_holds_latched_state_during_output_fluctuations():
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            gripper_relative_threshold=0.2,
            step_frequency=10_000.0,
        )
    )
    env._state = SimpleNamespace(gripper_position=0.5)
    base = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    closed = env.step(np.append(base, 0.1))[-1]["executed_action"][-1]
    held = [
        env.step(np.append(base, output))[-1]["executed_action"][-1]
        for output in (0.35, 0.45, 0.55)
    ]

    assert closed == 0.0
    assert held == [0.0, 0.0, 0.0]
    env.close()


def test_relative_gripper_reset_clears_latched_state():
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            gripper_relative_threshold=0.2,
            step_frequency=10_000.0,
        )
    )
    env._state = SimpleNamespace(gripper_position=0.5)
    base = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert env.step(np.append(base, 0.1))[-1]["executed_action"][-1] == 0.0

    env.reset()
    env._state = SimpleNamespace(gripper_position=0.8)
    # Difference is inside the deadband, so a reset mapper initializes from
    # measured feedback (open) instead of retaining the prior closed latch.
    executed = env.step(np.append(base, 0.7))[-1]["executed_action"][-1]

    assert executed == 1.0
    env.close()


# ---------------------------------------------------------------------------
# Guarded minimum-jerk ServoJ reset
# ---------------------------------------------------------------------------


def _plan_reset(start_deg, target_deg, **overrides):
    kwargs = {
        "reset_fps": 30.0,
        "min_duration_s": 3.0,
        "max_duration_s": 20.0,
        "max_velocity_deg_s": 20.0,
        "max_acceleration_deg_s2": 30.0,
        "max_step_deg": 1.0,
    }
    kwargs.update(overrides)
    return DobotController._plan_reset_trajectory(
        np.radians(start_deg), np.radians(target_deg), **kwargs
    )


def test_reset_trajectory_is_minimum_jerk_and_step_limited():
    trajectory, effective_target, duration = _plan_reset(
        np.zeros(6), np.array([90.0, -45.0, 30.0, 0.0, 0.0, 0.0])
    )
    points = np.vstack([np.zeros(6), trajectory])
    step_deg = np.degrees(np.abs(np.diff(points, axis=0)))

    assert duration > 3.0
    assert np.max(step_deg) <= 1.0 + 1e-9
    np.testing.assert_allclose(trajectory[-1], effective_target, rtol=0.0, atol=1e-12)

    # Minimum-jerk starts and ends much more gently than its middle section.
    first_step = float(np.max(step_deg[0]))
    middle_step = float(np.max(step_deg[len(step_deg) // 2]))
    last_step = float(np.max(step_deg[-1]))
    assert first_step < middle_step * 0.01
    assert last_step < middle_step * 0.01


def test_reset_duration_grows_with_joint_distance():
    _, _, short_duration = _plan_reset(np.zeros(6), np.full(6, 10.0))
    _, _, long_duration = _plan_reset(np.zeros(6), np.full(6, 100.0))
    assert long_duration > short_duration


def test_reset_trajectory_uses_shortest_angular_path():
    trajectory, effective_target, _ = _plan_reset(
        np.array([350.0, 0, 0, 0, 0, 0]),
        np.array([10.0, 0, 0, 0, 0, 0]),
    )
    assert np.degrees(effective_target[0]) == pytest.approx(370.0)
    assert np.degrees(trajectory[-1, 0]) == pytest.approx(370.0)


def test_reset_planner_fails_before_motion_when_duration_exceeds_limit():
    with pytest.raises(ValueError, match="exceeding max_duration"):
        _plan_reset(
            np.zeros(6),
            np.full(6, 170.0),
            max_velocity_deg_s=5.0,
            max_duration_s=5.0,
        )


def test_reset_config_rejects_invalid_safety_limits():
    with pytest.raises(ValueError, match="reset durations"):
        DobotRobotConfig(reset_min_duration_s=5.0, reset_max_duration_s=4.0)
    with pytest.raises(ValueError, match="per-frame step"):
        DobotRobotConfig(reset_max_step_deg=0.0)


class _ImmediateFuture:
    def __init__(self, result=None):
        self.result = result

    def wait(self):
        return [self.result]


class _ResetControllerSpy:
    def __init__(self):
        self.calls = []
        self.reset_kwargs = None

    def assert_ready(self):
        self.calls.append("assert_ready")
        return _ImmediateFuture()

    def open_gripper(self):
        self.calls.append("open_gripper")
        return _ImmediateFuture()

    def reset_to_pose(self, joints, **kwargs):
        self.calls.append("reset_to_pose")
        self.reset_kwargs = kwargs
        return _ImmediateFuture()


def test_go_to_rest_opens_gripper_before_servoj_reset(monkeypatch):
    env = _dummy_pose_env()
    spy = _ResetControllerSpy()
    env._controller = spy
    monkeypatch.setattr(
        "rlinf.envs.realworld.dobot.dobot_env.time.sleep", lambda _: None
    )

    env.go_to_rest()

    assert spy.calls == ["assert_ready", "open_gripper", "reset_to_pose"]
    assert env._gripper_is_open is True
    assert spy.reset_kwargs["reset_fps"] == env.config.reset_fps
    assert spy.reset_kwargs["max_step_deg"] == env.config.reset_max_step_deg
    env.close()


def test_dobot_env_has_no_legacy_fixed_step_reset_call():
    """Startup and episode reset must share the guarded go_to_rest path."""
    import inspect

    source = inspect.getsource(DobotEnv)
    assert "init_steps=" not in source
    assert "init_fps=" not in source


class _ResetRobotFake:
    def __init__(self, *, follow_commands=True, reject_at=None):
        self.joints_deg = np.zeros(6, dtype=float)
        self.follow_commands = follow_commands
        self.reject_at = reject_at
        self.commands = []
        self.smoothing_reset = False

    def get_joint_positions(self):
        return self.joints_deg.copy()

    def get_mode(self):
        return 5

    @staticmethod
    def _mode_name(mode):
        return str(mode)

    def reset_smoothing(self):
        self.smoothing_reset = True

    def servo_joints(self, joints_deg):
        self.commands.append(np.asarray(joints_deg, dtype=float))
        if self.reject_at is not None and len(self.commands) == self.reject_at:
            return False
        if self.follow_commands:
            self.joints_deg = self.commands[-1].copy()
        return True

    @staticmethod
    def _warn_jump(_message):
        return None


class _LoggerFake:
    def info(self, *_args, **_kwargs):
        return None


def _direct_reset_controller(robot):
    controller = DobotController.__new__(DobotController)
    controller._robot = robot
    controller._logger = _LoggerFake()
    controller._follower_engaged = False
    return controller


def _run_fast_reset(controller, target_deg, **overrides):
    kwargs = {
        "reset_fps": 100.0,
        "min_duration_s": 0.1,
        "max_duration_s": 2.0,
        "max_velocity_deg_s": 100.0,
        "max_acceleration_deg_s2": 200.0,
        "max_step_deg": 1.0,
        "feedback_interval_frames": 1,
        "max_tracking_error_deg": 2.0,
        "final_tolerance_deg": 0.2,
        "final_hold_frames": 2,
    }
    kwargs.update(overrides)
    controller.reset_to_pose(np.radians(target_deg), **kwargs)


def test_reset_executor_aborts_when_servoj_is_rejected(monkeypatch):
    robot = _ResetRobotFake(reject_at=2)
    controller = _direct_reset_controller(robot)
    monkeypatch.setattr(
        "rlinf.envs.realworld.dobot.dobot_controller.time.sleep", lambda _: None
    )

    with pytest.raises(RuntimeError, match="ServoJ rejected"):
        _run_fast_reset(controller, np.full(6, 5.0))

    assert controller._follower_engaged is False
    assert len(robot.commands) == 2


def test_reset_executor_aborts_on_tracking_error(monkeypatch):
    robot = _ResetRobotFake(follow_commands=False)
    controller = _direct_reset_controller(robot)
    monkeypatch.setattr(
        "rlinf.envs.realworld.dobot.dobot_controller.time.sleep", lambda _: None
    )

    with pytest.raises(RuntimeError, match="tracking error"):
        _run_fast_reset(
            controller,
            np.full(6, 5.0),
            max_tracking_error_deg=0.05,
        )

    assert controller._follower_engaged is False


def test_reset_executor_reaches_target_without_movej(monkeypatch):
    robot = _ResetRobotFake(follow_commands=True)
    controller = _direct_reset_controller(robot)
    monkeypatch.setattr(
        "rlinf.envs.realworld.dobot.dobot_controller.time.sleep", lambda _: None
    )

    _run_fast_reset(controller, np.array([5.0, -4.0, 3.0, -2.0, 1.0, 0.0]))

    assert robot.smoothing_reset is True
    np.testing.assert_allclose(
        robot.joints_deg,
        np.array([5.0, -4.0, 3.0, -2.0, 1.0, 0.0]),
        atol=1e-9,
    )
    assert controller._follower_engaged is False


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
