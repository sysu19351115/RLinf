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

"""Unit tests for checkpoint verification, markers, and retention."""

from __future__ import annotations

import os

import pytest

from rlinf.utils.checkpoint_utils import (
    prune_old_checkpoints,
    verify_checkpoint_files,
    write_completed_marker,
)


def _write(path, size: int = 4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


def test_verify_dcp_checkpoint_ok(tmp_path):
    actor = tmp_path / "actor"
    _write(actor / "dcp_checkpoint" / ".metadata")
    _write(actor / "dcp_checkpoint" / "__0_0.distcp", size=1024)
    _write(actor / "model_state_dict" / "full_weights.pt", size=2048)

    verify_checkpoint_files(str(actor), require_full_weights=True)


def test_verify_local_shard_checkpoint_ok(tmp_path):
    actor = tmp_path / "actor"
    _write(actor / "local_shard_checkpoint" / "checkpoint_rank_0.pt", size=1024)
    _write(actor / "model_state_dict" / "full_weights.pt", size=2048)

    verify_checkpoint_files(str(actor), require_full_weights=True)


def test_verify_missing_checkpoint_raises(tmp_path):
    actor = tmp_path / "actor"
    os.makedirs(actor, exist_ok=True)

    with pytest.raises(RuntimeError, match="neither dcp_checkpoint"):
        verify_checkpoint_files(str(actor), require_full_weights=False)


def test_verify_empty_distcp_raises(tmp_path):
    actor = tmp_path / "actor"
    _write(actor / "dcp_checkpoint" / ".metadata")
    _write(actor / "dcp_checkpoint" / "__0_0.distcp", size=0)

    with pytest.raises(RuntimeError, match="no non-empty \\*\\.distcp"):
        verify_checkpoint_files(str(actor), require_full_weights=False)


def test_verify_missing_full_weights_raises(tmp_path):
    actor = tmp_path / "actor"
    _write(actor / "dcp_checkpoint" / ".metadata")
    _write(actor / "dcp_checkpoint" / "__0_0.distcp", size=1024)

    with pytest.raises(RuntimeError, match="full_weights.pt"):
        verify_checkpoint_files(str(actor), require_full_weights=True)


def test_verify_relative_path_raises():
    with pytest.raises(ValueError, match="must be absolute"):
        verify_checkpoint_files("relative/actor", require_full_weights=False)


def test_write_completed_marker(tmp_path):
    actor = tmp_path / "actor"
    os.makedirs(actor, exist_ok=True)

    marker = write_completed_marker(str(actor), step=42)

    assert marker == str(actor / "COMPLETED")
    content = (actor / "COMPLETED").read_text()
    assert "global_step=42" in content


def test_prune_keeps_newest_checkpoints(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    for step in (1, 2, 3, 4, 5):
        (checkpoints_dir / f"global_step_{step}").mkdir(parents=True)

    removed = prune_old_checkpoints(str(checkpoints_dir), keep_last=2)

    assert removed == [
        str(checkpoints_dir / f"global_step_{step}") for step in (1, 2, 3)
    ]
    remaining = sorted(path.name for path in checkpoints_dir.iterdir() if path.is_dir())
    assert remaining == ["global_step_4", "global_step_5"]


def test_prune_ignores_non_step_dirs_and_keep_all(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "global_step_1").mkdir(parents=True)
    (checkpoints_dir / "global_step_2").mkdir(parents=True)
    (checkpoints_dir / "other_dir").mkdir()

    assert prune_old_checkpoints(str(checkpoints_dir), keep_last=0) == []
    removed = prune_old_checkpoints(str(checkpoints_dir), keep_last=1)

    assert removed == [str(checkpoints_dir / "global_step_1")]
    assert (checkpoints_dir / "other_dir").exists()
