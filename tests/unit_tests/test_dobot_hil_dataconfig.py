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

"""Unit tests for DobotDataConfig dataset_layout (native vs rlinf_hil)."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from rlinf.models.embodiment.openpi.dataconfig.dobot_dataconfig import DobotDataConfig


def _make_config(**kwargs) -> DobotDataConfig:
    return DobotDataConfig(**kwargs)


def _invoke_create(cfg: DobotDataConfig):
    """Call DobotDataConfig.create with a mocked model_config."""
    model_config = MagicMock()
    assets_dirs = pathlib.Path("/tmp/nonexistent_assets")
    return cfg.create(assets_dirs, model_config)


def _get_repack_structure(data_config) -> dict:
    """Extract the repack_structure dict from the DataConfig's repack_transforms."""
    repack_group = data_config.repack_transforms
    # Group has .inputs list; the first is RepackTransform with .structure.
    repack_transform = repack_group.inputs[0]
    return repack_transform.structure


class TestNativeLayout:
    """Default 'native' layout must preserve existing checkpoint compatibility."""

    def test_native_joint_repack(self):
        cfg = _make_config(use_pose=False, dataset_layout="native")
        dc = _invoke_create(cfg)
        mapping = _get_repack_structure(dc)
        assert mapping["observation/image"] == "observation.images.cam_left_wrist"
        assert mapping["observation/state"] == "observation.state"
        assert mapping["actions"] == "action"
        assert mapping["prompt"] == "prompt"
        # Joint mode has no prev_state.
        assert "observation/prev_state" not in mapping

    def test_native_pose_repack(self):
        cfg = _make_config(use_pose=True, use_delta_joint_actions=False, dataset_layout="native")
        dc = _invoke_create(cfg)
        mapping = _get_repack_structure(dc)
        assert mapping["observation/prev_state"] == "observation.prev_state"

    def test_default_is_native(self):
        cfg = _make_config()
        assert cfg.dataset_layout == "native"


class TestRlinfHilLayout:
    """'rlinf_hil' layout must use flat keys from CollectEpisode."""

    def test_rlinf_hil_joint_repack(self):
        cfg = _make_config(use_pose=False, dataset_layout="rlinf_hil")
        dc = _invoke_create(cfg)
        mapping = _get_repack_structure(dc)
        assert mapping["observation/image"] == "image"
        assert mapping["observation/state"] == "state"
        assert mapping["actions"] == "actions"
        assert mapping["prompt"] == "task"
        assert "observation/prev_state" not in mapping

    def test_rlinf_hil_pose_repack_includes_prev_state(self):
        cfg = _make_config(
            use_pose=True, use_delta_joint_actions=False, dataset_layout="rlinf_hil"
        )
        dc = _invoke_create(cfg)
        mapping = _get_repack_structure(dc)
        assert mapping["observation/prev_state"] == "prev_state"
        assert mapping["observation/image"] == "image"
        assert mapping["observation/state"] == "state"
        assert mapping["actions"] == "actions"
        assert mapping["prompt"] == "task"

    def test_pose_mode_state_dim_is_8(self):
        cfg = _make_config(
            use_pose=True, use_delta_joint_actions=False, dataset_layout="rlinf_hil"
        )
        dc = _invoke_create(cfg)
        # The repack doesn't carry state_dim, but the data_transforms do.
        # Verify pose transforms (DeltaPose/AbsolutePose) are present.
        transforms = dc.data_transforms.inputs
        transform_names = [type(t).__name__ for t in transforms]
        assert "DeltaPose" in transform_names
        # AbsolutePose is in outputs, not inputs.
        output_names = [type(t).__name__ for t in dc.data_transforms.outputs]
        assert "AbsolutePose" in output_names
