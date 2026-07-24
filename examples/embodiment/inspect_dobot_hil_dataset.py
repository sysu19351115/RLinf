# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inspect and validate a Dobot HIL LeRobot dataset.

Checks schema (field dtypes/shapes), numerical safety (quaternion norms,
gripper range, workspace bounds, prev_state continuity), and episode
integrity (last frame done, at least one intervene frame).

Usage::

    python examples/embodiment/inspect_dobot_hil_dataset.py \\
        logs/dobot_hil_collect/collected_data/<SESSION>/rank_0/id_0

    # Allow constant images (dummy mode only):
    python examples/embodiment/inspect_dobot_hil_dataset.py \\
        logs/dobot_hil_collect/collected_data/<SESSION>/rank_0/id_0 \\
        --allow-dummy-images

Exit code is non-zero if any structural or safety invariant fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_dataset(data_path: str):
    """Load a LeRobot dataset and return (dataset, feature_schema)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(data_path).expanduser().resolve()
    ds = LeRobotDataset(repo_id=root.name, root=root)
    features = ds.meta.info.get("features", {})
    return ds, features


def _check_schema(features: dict) -> list[str]:
    """Verify required fields exist with correct dtype/shape."""
    errors = []
    is_openpi = "observation.state" in features
    required = {
        ("observation.state" if is_openpi else "state"): ("float32", 8),
        ("action" if is_openpi else "actions"): ("float32", 8),
        "done": ("bool", None),
        "is_success": ("bool", None),
        "intervene_flag": ("bool", None),
        ("observation.images.cam_left_wrist" if is_openpi else "image"): (None, None),
    }
    for field, (dtype, dim) in required.items():
        if field not in features:
            errors.append(f"Missing required field: {field}")
            continue
        f = features[field]
        if dtype and f.get("dtype") != dtype:
            errors.append(f"Field {field} has dtype {f.get('dtype')}, expected {dtype}")
        if dim and f.get("shape") and f["shape"][-1] != dim:
            errors.append(
                f"Field {field} has shape {f.get('shape')}, expected last dim {dim}"
            )
    # prev_state is required for pose mode.
    prev_state_key = "observation.prev_state" if is_openpi else "prev_state"
    if prev_state_key not in features:
        errors.append(f"Missing {prev_state_key} (required for pose mode)")
    return errors


def _check_episode(ds, ep_idx: int, allow_dummy_images: bool) -> tuple[list[str], dict]:
    """Check one episode. Returns (errors, stats)."""
    errors = []
    stats = {
        "frames": 0,
        "intervened_frames": 0,
        "xyz_min": np.full(3, np.inf),
        "xyz_max": np.full(3, -np.inf),
        "gripper_min": np.inf,
        "gripper_max": -np.inf,
        "quat_norm_max_err": 0.0,
        "prev_state_max_err": 0.0,
        "has_intervene": False,
    }

    if hasattr(ds, "get_episode"):
        frames = list(ds.get_episode(ep_idx))
    else:
        start = int(ds.episode_data_index["from"][ep_idx])
        end = int(ds.episode_data_index["to"][ep_idx])
        frames = [ds[i] for i in range(start, end)]
    if not frames:
        errors.append(f"Episode {ep_idx}: empty")
        return errors, stats

    prev_state = None
    for i, frame in enumerate(frames):
        stats["frames"] += 1
        state = np.asarray(
            frame.get("observation.state", frame.get("state", [])),
            dtype=np.float32,
        ).flatten()
        action = np.asarray(
            frame.get("action", frame.get("actions", [])), dtype=np.float32
        ).flatten()
        intervene = bool(np.asarray(frame.get("intervene_flag", False)).any())

        if state.shape[0] != 8:
            errors.append(
                f"Episode {ep_idx} frame {i}: state dim {state.shape[0]} != 8"
            )
            continue
        if not np.isfinite(state).all():
            errors.append(f"Episode {ep_idx} frame {i}: non-finite state")
        if not np.isfinite(action).all():
            errors.append(f"Episode {ep_idx} frame {i}: non-finite action")

        # Quaternion norm.
        q = state[3:7]
        qnorm = np.linalg.norm(q)
        err = abs(qnorm - 1.0)
        if err > stats["quat_norm_max_err"]:
            stats["quat_norm_max_err"] = err
        if err > 1e-3:
            errors.append(
                f"Episode {ep_idx} frame {i}: quaternion norm {qnorm:.6f} deviates > 1e-3"
            )

        # Gripper range.
        g = state[7]
        stats["gripper_min"] = min(stats["gripper_min"], g)
        stats["gripper_max"] = max(stats["gripper_max"], g)
        if g < -0.01 or g > 1.01:
            errors.append(f"Episode {ep_idx} frame {i}: gripper {g:.3f} out of [0, 1]")

        # XYZ bounds.
        stats["xyz_min"] = np.minimum(stats["xyz_min"], state[:3])
        stats["xyz_max"] = np.maximum(stats["xyz_max"], state[:3])

        # Intervene.
        if intervene:
            stats["intervened_frames"] += 1
            stats["has_intervene"] = True

        # prev_state continuity.
        ps = frame.get("observation.prev_state", frame.get("prev_state"))
        if ps is not None:
            ps = np.asarray(ps, dtype=np.float32).flatten()
            if prev_state is not None:
                cont_err = np.max(np.abs(ps - prev_state))
                if cont_err > stats["prev_state_max_err"]:
                    stats["prev_state_max_err"] = cont_err
            prev_state = ps.copy()

    # Last frame done.
    last_frame = frames[-1]
    done = bool(np.asarray(last_frame.get("done", False)).any())
    if not done:
        errors.append(f"Episode {ep_idx}: last frame done=False")

    return errors, stats


def main():
    parser = argparse.ArgumentParser(description="Inspect Dobot HIL LeRobot dataset.")
    parser.add_argument("data_path", help="Path to the LeRobot dataset (rank_N/id_M).")
    parser.add_argument(
        "--allow-dummy-images",
        action="store_true",
        help="Allow constant/all-black images (dummy mode only).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: dataset path does not exist: {data_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Inspecting dataset: {data_path}")

    ds, features = _load_dataset(str(data_path))

    # ── Schema checks ───────────────────────────────────────────────────────
    schema_errors = _check_schema(features)
    for e in schema_errors:
        print(f"SCHEMA ERROR: {e}", file=sys.stderr)

    # ── Per-episode checks ──────────────────────────────────────────────────
    num_episodes = ds.meta.info.get("total_episodes", 0)
    all_errors = list(schema_errors)
    total_frames = 0
    total_intervened = 0
    intervened_episodes = 0

    for ep_idx in range(num_episodes):
        errors, stats = _check_episode(ds, ep_idx, args.allow_dummy_images)
        all_errors.extend(errors)
        total_frames += stats["frames"]
        total_intervened += stats["intervened_frames"]
        if stats["has_intervene"]:
            intervened_episodes += 1
        for e in errors:
            print(f"EPISODE {ep_idx}: {e}", file=sys.stderr)
        print(
            f"  Episode {ep_idx}: {stats['frames']} frames, "
            f"{stats['intervened_frames']} intervened, "
            f"quat_err={stats['quat_norm_max_err']:.6f}, "
            f"prev_err={stats['prev_state_max_err']:.6f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Total frames: {total_frames}")
    if num_episodes > 0:
        print(f"Mean episode length: {total_frames / num_episodes:.1f}")
    print(f"Intervened episodes: {intervened_episodes}/{num_episodes}")
    if total_frames > 0:
        print(f"Intervened frame ratio: {total_intervened / total_frames:.3f}")

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} error(s) found.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
