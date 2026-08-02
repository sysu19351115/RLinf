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
import hashlib
import json
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


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Content SHA-256 of one file (used by the checkpoint manifest)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_checkpoint_manifest(
    checkpoint_dir: str,
    *,
    step: int,
    files: dict[str, str],
    extra: dict | None = None,
) -> str:
    """Write ``manifest.json`` with per-file SHA-256 and byte sizes (P2-7)."""
    manifest: dict = {
        "schema_version": 1,
        "global_step": int(step),
        "files": {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for rel_path, abs_path in files.items():
        if not os.path.isfile(abs_path):
            raise RuntimeError(
                f"manifest file missing during save: {abs_path}"
            )
        manifest["files"][rel_path] = {
            "sha256": sha256_file(abs_path),
            "bytes": os.path.getsize(abs_path),
        }
    if extra:
        manifest["extra"] = extra
    path = os.path.join(checkpoint_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def verify_checkpoint_manifest(
    checkpoint_dir: str, *, reject_unknown_files: bool = False
) -> dict:
    """Verify ``manifest.json`` hashes/sizes; rejects missing or corrupt files.

    Args:
        checkpoint_dir: Absolute checkpoint directory.
        reject_unknown_files: When True, any file not listed in the manifest
            (besides ``manifest.json`` and ``COMPLETED``) is rejected, so a
            tampered/partial directory cannot silently restore extra state.
    """
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"checkpoint {checkpoint_dir} is missing manifest.json"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported checkpoint manifest schema_version")
    for rel_path, meta in manifest.get("files", {}).items():
        full = os.path.join(checkpoint_dir, rel_path)
        if not os.path.isfile(full):
            raise RuntimeError(
                f"checkpoint file missing: {rel_path}"
            )
        if int(meta["bytes"]) != os.path.getsize(full):
            raise RuntimeError(
                f"checkpoint file size mismatch: {rel_path}"
            )
        if str(meta["sha256"]) != sha256_file(full):
            raise RuntimeError(
                f"checkpoint file hash mismatch: {rel_path}"
            )
    if reject_unknown_files:
        expected = set(manifest.get("files", {})) | {"manifest.json", "COMPLETED"}
        for root, _dirs, files in os.walk(checkpoint_dir):
            for name in files:
                rel = os.path.relpath(
                    os.path.join(root, name), checkpoint_dir
                )
                if rel not in expected:
                    raise RuntimeError(
                        f"checkpoint contains unknown file: {rel}"
                    )
    return manifest


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
