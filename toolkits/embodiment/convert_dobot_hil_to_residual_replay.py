#!/usr/bin/env python3
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

"""Convert collected Dobot HIL episodes into residual RLPD replay data.

Input format (synthetic or LeRobot-derived):

``episodes = [{
    "episode_id": int,
    "observations": {"main_images": np.ndarray [T, C, H, W],
                    "prev_states": np.ndarray [T, 8]},
    "actions": np.ndarray [T, 8],          # real executed absolute actions
    "human_intervention_mask": np.ndarray [T] (optional),
    "handoff_hold_mask": np.ndarray [T] (optional),
    "rewards": np.ndarray [T] (optional),
    "terminated": bool,
}]
``

Nominal chunks must be provided by the frozen Pi0.5 base policy (same
checkpoint used at rollout time). The CLI accepts a JSON/``.npz`` file mapping
``(episode_id, chunk_id) -> nominal [10, 8]``; a companion script generates it
by running the frozen checkpoint over chunk-start observations.

Trust boundary: ``npz``/``torch`` payloads from an unknown source can execute
arbitrary code via pickle.  This tool therefore refuses object-array ``npz``
files unless ``--trusted-source`` is passed, and the resulting replay file must
only be loaded inside this project's own training run (a directory that the
operator controls).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
    compute_base_fingerprint,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    CHUNK_LEN,
    SOURCE_OFFLINE_DEMO,
    ResidualChunkTransition,
    build_residual_chunk_transition,
)


def _to_bhwc(images: np.ndarray) -> np.ndarray:
    """Normalize camera frames to BHWC (matching the online rollout layout).

    Accepts either channels-first ``[T, C, H, W]`` (LeRobot-style) or already
    ``[T, H, W, C]`` batches, plus single ``[C, H, W]`` / ``[H, W, C]``
    frames (e.g. ``final_obs``).  Online transitions always store BHWC, so
    offline conversions must use the same layout or mixed batches would break
    ``transitions_to_torch_batch``.
    """
    images = np.asarray(images)
    if images.ndim == 3:
        if images.shape[-1] == 3:
            return images
        if images.shape[0] == 3:
            return np.moveaxis(images, 0, -1)
    elif images.ndim == 4:
        if images.shape[1] == 3 and images.shape[-1] != 3:
            return np.moveaxis(images, 1, -1)
        if images.shape[-1] == 3:
            return images
    raise ValueError(
        "Cannot determine image layout (need 3 channels as C or last dim), "
        f"got shape {images.shape}"
    )


def convert_episodes_to_residual_replay(
    episodes: list[dict[str, Any]],
    *,
    nominal_provider: Callable[[int, int], np.ndarray],
    next_nominal_provider: Callable[[int, int], np.ndarray],
    codec: ResidualCodec,
    base_fingerprint: str,
    gamma: float = 0.99,
    episode_id_offset: int = 0,
    detail: bool = False,
) -> tuple[list[ResidualChunkTransition], dict[str, Any]]:
    """Convert episodes into chunk transitions; returns (transitions, report)."""
    transitions: list[ResidualChunkTransition] = []
    report: dict[str, Any] = {
        "episodes": len(episodes),
        "transitions": 0,
        "rejected": 0,
        "rejected_reasons": {},
        "out_of_support": 0,
        "details": [] if detail else None,
    }

    def _detail(episode_id: int, chunk_id: int, status: str, reason: str = "") -> None:
        if report["details"] is not None:
            report["details"].append(
                {
                    "episode_id": int(episode_id),
                    "chunk_id": int(chunk_id),
                    "status": status,
                    "reason": reason,
                }
            )

    for episode in episodes:
        episode_id = int(episode["episode_id"])
        observations = episode["observations"]
        images = _to_bhwc(np.asarray(observations["main_images"]))
        states = np.asarray(observations["prev_states"])
        actions = np.asarray(episode["actions"], dtype=np.float32)
        time_len = actions.shape[0]
        if time_len == 0:
            report["rejected"] += 1
            _detail(episode_id, -1, "rejected", "empty_episode")
            continue
        human_mask = np.asarray(
            episode.get(
                "human_intervention_mask",
                np.zeros(time_len, dtype=bool),
            ),
            dtype=bool,
        )
        handoff_mask = np.asarray(
            episode.get("handoff_hold_mask", np.zeros(time_len, dtype=bool)),
            dtype=bool,
        )
        if "rewards" not in episode:
            report["rejected"] += 1
            report["rejected_reasons"]["missing_rewards"] = (
                report["rejected_reasons"].get("missing_rewards", 0) + 1
            )
            _detail(episode_id, -1, "rejected", "missing_rewards")
            continue
        rewards = np.asarray(episode["rewards"], dtype=np.float32)
        terminated = bool(episode.get("terminated", False))

        for chunk_id, start in enumerate(range(0, time_len, CHUNK_LEN)):
            end = min(start + CHUNK_LEN, time_len)
            k = end - start
            try:
                nominal = np.asarray(
                    nominal_provider(episode_id, chunk_id), dtype=np.float32
                )
            except KeyError:
                report["rejected"] += 1
                report["rejected_reasons"]["missing_nominal"] = (
                    report["rejected_reasons"].get("missing_nominal", 0) + 1
                )
                _detail(episode_id, chunk_id, "rejected", "missing_nominal")
                continue
            if nominal.shape != (CHUNK_LEN, 8):
                report["rejected"] += 1
                report["rejected_reasons"]["bad_nominal_shape"] = (
                    report["rejected_reasons"].get("bad_nominal_shape", 0) + 1
                )
                _detail(episode_id, chunk_id, "rejected", "bad_nominal_shape")
                continue

            executed = np.zeros((CHUNK_LEN, 8), dtype=np.float32)
            executed_mask = np.zeros(CHUNK_LEN, dtype=bool)
            executed_mask[:k] = True
            executed[:k] = actions[start:end]
            executed_human = np.zeros(CHUNK_LEN, dtype=bool)
            executed_human[:k] = human_mask[start:end]
            executed_handoff = np.zeros(CHUNK_LEN, dtype=bool)
            executed_handoff[:k] = handoff_mask[start:end]
            chunk_rewards = np.zeros(CHUNK_LEN, dtype=np.float32)
            chunk_rewards[:k] = rewards[start:end]
            terminations = np.zeros(CHUNK_LEN, dtype=bool)
            truncations = np.zeros(CHUNK_LEN, dtype=bool)
            is_last = chunk_id == (time_len - 1) // CHUNK_LEN
            if terminated and is_last:
                terminations[k - 1] = True
            elif is_last and not terminated:
                truncations[k - 1] = True

            next_nominal = None
            if not is_last:
                try:
                    next_nominal = np.asarray(
                        next_nominal_provider(episode_id, chunk_id),
                        dtype=np.float32,
                    )
                except KeyError:
                    report["rejected"] += 1
                    report["rejected_reasons"]["missing_next_nominal"] = (
                        report["rejected_reasons"].get("missing_next_nominal", 0) + 1
                    )
                    _detail(
                        episode_id, chunk_id, "rejected", "missing_next_nominal"
                    )
                    continue
                if next_nominal.shape != (CHUNK_LEN, 8):
                    report["rejected"] += 1
                    report["rejected_reasons"]["bad_next_nominal_shape"] = (
                        report["rejected_reasons"].get("bad_next_nominal_shape", 0) + 1
                    )
                    _detail(
                        episode_id, chunk_id, "rejected", "bad_next_nominal_shape"
                    )
                    continue

            curr_obs = {
                "main_images": images[start],
                "prev_states": states[start],
            }
            if end < time_len:
                next_idx = end
            else:
                next_idx = end - 1
            raw_final = episode.get("final_obs") or {}
            final_images = raw_final.get("main_images", images[next_idx])
            final_states = raw_final.get("prev_states", states[next_idx])
            next_obs = {
                "main_images": _to_bhwc(np.asarray(final_images)),
                "prev_states": np.asarray(final_states),
            }
            transition, valid, reason = build_residual_chunk_transition(
                curr_obs=curr_obs,
                next_obs=next_obs,
                nominal_actions=nominal,
                next_nominal_actions=next_nominal,
                sampled_arm_residual=np.zeros((CHUNK_LEN, 6), dtype=np.float32),
                sampled_gripper_mode=np.zeros(CHUNK_LEN, dtype=np.int64),
                commanded_actions=executed.copy(),
                executed_actions=executed,
                gripper_bypass_mask=np.zeros(CHUNK_LEN, dtype=bool),
                rewards=chunk_rewards,
                executed_action_mask=executed_mask,
                human_intervention_mask=executed_human,
                handoff_hold_mask=executed_handoff,
                terminations=terminations,
                truncations=truncations,
                reward_label_valid=True,
                codec=codec,
                source=SOURCE_OFFLINE_DEMO,
                policy_version=-1,
                base_fingerprint=base_fingerprint,
                gamma=gamma,
                episode_id=episode_id + episode_id_offset,
                chunk_id=chunk_id,
            )
            if not valid:
                report["rejected"] += 1
                report["rejected_reasons"][reason] = (
                    report["rejected_reasons"].get(reason, 0) + 1
                )
                _detail(episode_id, chunk_id, "rejected", reason)
                if reason == "out_of_support":
                    report["out_of_support"] += 1
                continue
            transitions.append(transition)
            report["transitions"] += 1
            _detail(episode_id, chunk_id, "accepted")

    return transitions, report


def _save_replay(
    transitions: list[ResidualChunkTransition],
    report: dict[str, Any],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "residual_replay.pt")
    torch.save(
        {"transitions": transitions, "report": report},
        path,
    )
    report_path = os.path.join(output_dir, "conversion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Dobot HIL episodes into residual RLPD replay."
    )
    parser.add_argument("--episodes", required=True, help="Path to episodes .npz")
    parser.add_argument("--nominals", required=True, help="Path to nominal chunks .npz")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model-path",
        default="",
        help="Frozen base checkpoint dir; used to compute the content "
        "fingerprint (must match the rollout/learner model).",
    )
    parser.add_argument(
        "--norm-stats-path",
        default=None,
        help="OpenPI norm_stats.json path used for the fingerprint.",
    )
    parser.add_argument(
        "--base-fingerprint",
        default=None,
        help="Override the computed content fingerprint (advanced).",
    )
    parser.add_argument(
        "--trusted-source",
        action="store_true",
        help="Allow loading object-array .npz episodes (pickle). Only pass "
        "when the file is produced by this project's own collector on a "
        "machine you control.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--episode-id-offset",
        type=int,
        default=1 << 40,
        help="Offline episode-id offset to avoid collision with online ids.",
    )
    args = parser.parse_args()

    try:
        data = np.load(args.episodes, allow_pickle=False)
    except ValueError as exc:
        if not args.trusted_source:
            raise ValueError(
                "episode .npz contains an object array (pickle). Refusing to "
                "load an untrusted file; pass --trusted-source only when the "
                "file was produced by this project's own collector."
            ) from exc
        data = np.load(args.episodes, allow_pickle=True)
    episodes = list(data["episodes"])
    try:
        nominals = np.load(args.nominals, allow_pickle=False)
    except ValueError as exc:
        if not args.trusted_source:
            raise ValueError(
                "nominals .npz contains an object array (pickle). Refusing to "
                "load an untrusted file; pass --trusted-source only when the "
                "file was produced by this project's own collector."
            ) from exc
        nominals = np.load(args.nominals, allow_pickle=True)
    nominal_map = {
        (int(key[0]), int(key[1])): value
        for key, value in nominals["nominals"].item().items()
    }

    def nominal_provider(episode_id: int, chunk_id: int) -> np.ndarray:
        return nominal_map[(episode_id, chunk_id)]

    def next_nominal_provider(episode_id: int, chunk_id: int) -> np.ndarray:
        return nominal_map[(episode_id, chunk_id + 1)]

    codec = ResidualCodec()
    base_fingerprint = args.base_fingerprint
    if base_fingerprint is None:
        base_fingerprint = compute_base_fingerprint(
            model_path=args.model_path,
            norm_stats_path=args.norm_stats_path,
            codec=codec,
        )
    print(f"base_fingerprint={base_fingerprint}")
    transitions, report = convert_episodes_to_residual_replay(
        episodes,
        nominal_provider=nominal_provider,
        next_nominal_provider=next_nominal_provider,
        codec=codec,
        base_fingerprint=base_fingerprint,
        episode_id_offset=args.episode_id_offset,
        detail=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str))
    if not args.dry_run:
        output_dir = os.path.join(
            args.output,
            f"residual_replay_{time.strftime('%Y%m%d-%H%M%S')}",
        )
        saved = _save_replay(transitions, report, output_dir)
        print(f"Saved replay to {saved}")


if __name__ == "__main__":
    main()
