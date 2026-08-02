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

"""Checkpoint verification, completion markers, and retention helpers."""

from __future__ import annotations

import glob
import os
import re
import shutil
import time


def verify_checkpoint_files(
    actor_save_path: str,
    require_full_weights: bool = True,
) -> None:
    """Verify a checkpoint directory contains complete, non-empty artifacts.

    Supports both DCP (``dcp_checkpoint/``) and local-shard
    (``local_shard_checkpoint/``) formats written by the FSDP strategy, plus
    the optional full model weights. Raises ``RuntimeError`` when required
    artifacts are missing or empty.

    Args:
        actor_save_path: Absolute actor checkpoint directory.
        require_full_weights: Whether ``model_state_dict/full_weights.pt``
            must exist and be non-empty.
    """
    if not os.path.isabs(actor_save_path):
        raise ValueError(f"Checkpoint path must be absolute; Got: {actor_save_path!r}")

    dcp_dir = os.path.join(actor_save_path, "dcp_checkpoint")
    local_shard_dir = os.path.join(actor_save_path, "local_shard_checkpoint")
    has_dcp = os.path.isdir(dcp_dir)
    has_local_shard = os.path.isdir(local_shard_dir)
    if not has_dcp and not has_local_shard:
        raise RuntimeError(
            "Checkpoint is incomplete: neither dcp_checkpoint/ nor "
            f"local_shard_checkpoint/ exists under {actor_save_path}"
        )

    if has_dcp:
        metadata = os.path.join(dcp_dir, ".metadata")
        if not os.path.isfile(metadata) or os.path.getsize(metadata) == 0:
            raise RuntimeError(
                "Checkpoint is incomplete: dcp_checkpoint/.metadata is missing or empty"
            )
        distcp_files = glob.glob(os.path.join(dcp_dir, "*.distcp"))
        if not distcp_files or any(os.path.getsize(path) == 0 for path in distcp_files):
            raise RuntimeError(
                "Checkpoint is incomplete: no non-empty *.distcp files "
                "under dcp_checkpoint/"
            )

    if has_local_shard:
        shard_files = glob.glob(os.path.join(local_shard_dir, "checkpoint_rank_*.pt"))
        if not shard_files or any(os.path.getsize(path) == 0 for path in shard_files):
            raise RuntimeError(
                "Checkpoint is incomplete: no non-empty checkpoint_rank_*.pt "
                "files under local_shard_checkpoint/"
            )

    if require_full_weights:
        full_weights = os.path.join(
            actor_save_path, "model_state_dict", "full_weights.pt"
        )
        if not os.path.isfile(full_weights) or os.path.getsize(full_weights) == 0:
            raise RuntimeError(
                "Checkpoint is incomplete: model_state_dict/full_weights.pt "
                "is missing or empty"
            )


def write_completed_marker(actor_save_path: str, step: int) -> str:
    """Write a ``COMPLETED`` marker into a verified checkpoint directory.

    Args:
        actor_save_path: Absolute actor checkpoint directory.
        step: Global step of the checkpoint.

    Returns:
        The marker file path.
    """
    marker_path = os.path.join(actor_save_path, "COMPLETED")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"global_step={step}\nsaved_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return marker_path


def prune_old_checkpoints(
    checkpoints_dir: str,
    keep_last: int,
) -> list[str]:
    """Delete the oldest ``global_step_*`` checkpoint directories.

    Only directories matching ``global_step_<int>`` directly under
    ``checkpoints_dir`` are considered; the newest ``keep_last`` are kept.
    Callers must only invoke this after the newest checkpoint has been fully
    verified (e.g. ``verify_checkpoint_files`` + ``write_completed_marker``).

    Args:
        checkpoints_dir: Parent directory containing ``global_step_*`` dirs.
        keep_last: Number of newest checkpoints to keep; ``<= 0`` keeps all.

    Returns:
        The list of removed checkpoint directories.
    """
    if keep_last <= 0 or not os.path.isdir(checkpoints_dir):
        return []

    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(checkpoints_dir, "global_step_*")):
        if not os.path.isdir(path):
            continue
        match = re.search(r"global_step_(\d+)$", path)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))

    candidates.sort(key=lambda item: item[0])
    removed: list[str] = []
    for _, path in candidates[: max(0, len(candidates) - keep_last)]:
        shutil.rmtree(path, ignore_errors=False)
        removed.append(path)
    return removed
