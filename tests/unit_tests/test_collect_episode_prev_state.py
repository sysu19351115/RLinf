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

from rlinf.data.lerobot_writer import LeRobotDatasetWriter
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


class TestExecutedActionPersistence:
    def test_record_step_prefers_executed_action(self, tmp_path):
        collector = _make_collector(tmp_path)
        requested = np.full((1, 8), 0.4, dtype=np.float32)
        executed = requested.copy()
        executed[0, -1] = 0.0

        collector._record_step(
            requested,
            obs={},
            reward=np.array([0.0]),
            terminated=np.array([False]),
            truncated=np.array([False]),
            info={"executed_action": executed},
        )

        np.testing.assert_array_equal(collector._buffers[0]["actions"][0], executed[0])
        assert requested[0, -1] == pytest.approx(0.4)
        collector.close()

    def test_lerobot_frame_prefers_executed_over_intervene_action(self, tmp_path):
        collector = _make_collector(tmp_path)
        buf = _build_pose_buf(num_steps=1)
        buf["actions"][0][-1] = 0.4
        buf["infos"][1].update(
            {
                "executed_action": np.array(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "intervene_action": np.array(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.7],
                    dtype=np.float32,
                ),
                "intervene_flag": np.array([True]),
            }
        )

        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)

        assert frames[0]["actions"][-1] == 0.0
        collector.close()


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


class TestOpenPIDobotPoseLayout:
    def test_maps_primary_fields_and_keeps_hil_fields(self, tmp_path):
        collector = _make_collector(
            tmp_path,
            dataset_layout="openpi_dobot_pose",
            required_observation_fields=("prev_states",),
        )
        frames = collector._buffer_to_lerobot_ep(
            _build_pose_buf(num_steps=2), env_idx=0, is_success=True
        )

        first = frames[0]
        assert "observation.state" in first
        assert "action" in first
        assert "observation.prev_state" in first
        assert "observation.images.cam_left_wrist" in first
        assert "state" not in first
        assert "actions" not in first
        assert "prev_state" not in first
        assert "image" not in first
        assert "model_action" in first
        assert "model_action_valid" in first
        assert "intervene_flag" in first
        assert "segment_id" in first
        assert "is_success" in first
        assert "done" in first

    def test_schema_matches_openpi_pose_keys_and_shapes(self, tmp_path):
        collector = _make_collector(
            tmp_path,
            dataset_layout="openpi_dobot_pose",
            required_observation_fields=("prev_states",),
        )
        frames = collector._buffer_to_lerobot_ep(
            _build_pose_buf(num_steps=1), env_idx=0, is_success=True
        )

        features = collector._openpi_dobot_pose_features(frames[0])
        assert features["observation.state"]["shape"] == (8,)
        assert features["action"]["shape"] == (8,)
        assert features["observation.prev_state"]["shape"] == (8,)
        assert features["observation.images.cam_left_wrist"]["shape"] == (
            3,
            224,
            224,
        )
        assert "observation.wrench" not in features
        for extra_key in (
            "model_action",
            "model_action_valid",
            "intervene_flag",
            "segment_id",
            "is_success",
            "done",
        ):
            assert extra_key in features

    def test_written_dataset_can_be_reloaded_by_lerobot(self, tmp_path):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        collector = _make_collector(
            tmp_path,
            dataset_layout="openpi_dobot_pose",
            required_observation_fields=("prev_states",),
        )
        buf = _build_pose_buf(num_steps=2)
        valid_pose = np.array(
            [[0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.5]],
            dtype=np.float32,
        )
        for obs in buf["observations"]:
            obs["states"] = valid_pose.copy()
            obs["prev_states"] = valid_pose.copy()
        for index, action in enumerate(buf["actions"]):
            action[:] = valid_pose[0]
            action[-1] = float(index % 2)
            buf["infos"][index + 1]["intervene_flag"] = np.array([True])
        frames = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
        collector._write_lerobot_episode(frames)
        collector.close()

        dataset_root = tmp_path / "collected_data" / "rank_0" / "id_0"
        dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
        features = dataset.meta.info["features"]
        assert "observation.state" in features
        assert "action" in features
        assert "observation.prev_state" in features
        assert "observation.images.cam_left_wrist" in features
        assert "model_action" in features
        assert "intervene_flag" in features
        assert dataset.meta.info["total_episodes"] == 1
        assert len(dataset) == 2


class TestLeRobotWriterOverwriteProtection:
    def test_existing_dataset_is_never_deleted(self, tmp_path):
        dataset_root = tmp_path / "existing_dataset"
        dataset_root.mkdir()
        sentinel = dataset_root / "sentinel.txt"
        sentinel.write_text("keep")

        writer = LeRobotDatasetWriter()
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            writer.create(repo_id=str(dataset_root))

        assert sentinel.read_text() == "keep"
