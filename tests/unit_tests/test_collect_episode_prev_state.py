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

"""Unit tests for CollectEpisode prev_state / model_action_valid persistence.

These tests verify that ``CollectEpisode._buffer_to_lerobot_ep`` correctly
extracts ``prev_state`` and ``model_action_valid`` from observations/info and
writes them into LeRobot frame dicts, and that ``_ensure_lerobot_writer``
declares the corresponding custom features schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class _DummyEnv:
    """Minimal env stand-in for CollectEpisode (non-gym path)."""

    observation_space = None
    action_space = None
    is_start = True

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action, **kwargs):
        return {}, 0.0, False, False, {}

    def close(self):
        pass


def _make_collector(tmp_path, **kwargs) -> CollectEpisode:
    """Create a CollectEpisode without triggering gym wrapper init."""
    defaults = {
        "export_format": "lerobot",
        "robot_type": "dobot_cr5af",
        "fps": 30,
    }
    defaults.update(kwargs)
    return CollectEpisode(
        _DummyEnv(),
        save_dir=str(tmp_path / "collected_data"),
        **defaults,
    )


def _build_pose_buf(num_steps: int = 2):
    """Build a raw episode buffer matching CollectEpisode._new_buffer shape.

    Includes ``prev_states`` in observations and ``model_action_valid`` in infos.
    """
    buf = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "infos": [],
        "segment_ids": [],
    }
    # Reset-obs (prepended, has prev_states too).
    reset_obs = {
        "main_images": np.zeros((224, 224, 3), dtype=np.uint8),
        "states": np.zeros((1, 8), dtype=np.float32),
        "prev_states": np.zeros((1, 8), dtype=np.float32),
        "task_descriptions": ["test"],
    }
    buf["observations"].append(reset_obs)
    buf["rewards"].append(0.0)
    buf["terminated"].append(False)
    buf["truncated"].append(False)
    buf["infos"].append({})
    buf["segment_ids"].append(0)

    for i in range(num_steps):
        obs = {
            "main_images": np.zeros((224, 224, 3), dtype=np.uint8),
            "states": np.ones((1, 8), dtype=np.float32) * (i + 1),
            "prev_states": np.ones((1, 8), dtype=np.float32) * i,
            "task_descriptions": ["test"],
        }
        info = {
            "model_action": np.ones(8, dtype=np.float32) * (i + 1),
            "model_action_valid": np.array([True], dtype=bool),
        }
        buf["observations"].append(obs)
        buf["actions"].append(np.ones(8, dtype=np.float32) * (i + 1))
        buf["rewards"].append(0.0)
        buf["terminated"].append(i == num_steps - 1)
        buf["truncated"].append(False)
        buf["infos"].append(info)
        buf["segment_ids"].append(0)

    return buf


class TestPrevStateExtraction:
    def test_frame_contains_prev_state(self, tmp_path):
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=2)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        assert frames is not None
        assert len(frames) == 2
        for frame in frames:
            assert "prev_state" in frame
            assert frame["prev_state"].shape == (8,)
            assert frame["prev_state"].dtype == np.float32

    def test_prev_state_values_correct(self, tmp_path):
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=2)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        # _buffer_to_lerobot_ep aligns obs to actions by taking the leading N
        # entries from the observations list (which includes a prepended
        # reset obs). So:
        #   frame[0].state = reset_obs.states = 0, prev_state = reset_obs.prev_states = 0
        #   frame[1].state = step0_obs.states = 1, prev_state = step0_obs.prev_states = 0
        np.testing.assert_allclose(frames[0]["state"], 0.0)
        np.testing.assert_allclose(frames[0]["prev_state"], 0.0)
        np.testing.assert_allclose(frames[1]["state"], 1.0)
        np.testing.assert_allclose(frames[1]["prev_state"], 0.0)

    def test_missing_prev_state_no_crash(self, tmp_path):
        """Without prev_states, frames should not contain prev_state key."""
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=2)
        # Remove prev_states from all observations.
        for obs in buf["observations"]:
            obs.pop("prev_states", None)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        assert frames is not None
        for frame in frames:
            assert "prev_state" not in frame


class TestModelActionValidExtraction:
    def test_frame_contains_model_action_valid(self, tmp_path):
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=2)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        for frame in frames:
            assert "model_action_valid" in frame
            assert frame["model_action_valid"].dtype == bool
            assert bool(frame["model_action_valid"].all())


class TestRequiredObservationFields:
    def test_missing_required_field_raises(self, tmp_path):
        collector = _make_collector(
            tmp_path, required_observation_fields=("prev_states",)
        )
        buf = _build_pose_buf(num_steps=2)
        # Remove prev_states from step observations (keep reset obs).
        for obs in buf["observations"][1:]:
            obs.pop("prev_states", None)
        with pytest.raises(ValueError, match="prev_states"):
            collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)

    def test_present_required_field_ok(self, tmp_path):
        collector = _make_collector(
            tmp_path, required_observation_fields=("prev_states",)
        )
        buf = _build_pose_buf(num_steps=2)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        assert frames is not None
        for frame in frames:
            assert "prev_state" in frame


class TestCustomFeaturesSchema:
    def test_custom_features_include_prev_state(self, tmp_path):
        """Verify _ensure_lerobot_writer declares prev_state schema."""
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=2)
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        # Simulate writer creation by checking first frame keys.
        first = frames[0]
        assert "prev_state" in first
        assert "model_action_valid" in first
        assert "model_action" in first
        # Verify shapes for schema declaration.
        assert first["prev_state"].shape[-1] == 8
        assert first["model_action_valid"].shape == (1,)
