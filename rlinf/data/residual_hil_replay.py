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

"""Schema-aware dual replay buffer with auditable, byte-budgeted sampling."""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_hil_rlpd.rollout import preprocess_image
from rlinf.algorithms.residual_hil_rlpd.transition import (
    SOURCE_OFFLINE_DEMO,
    SOURCE_ONLINE_INTERVENTION,
    ResidualChunkTransition,
)


def _array_bytes(value: Any) -> int:
    """Best-effort resident bytes of a transition field (arrays dominate)."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_array_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v) for v in value)
    if isinstance(value, np.generic):
        return int(value.itemsize)
    return 0


def transition_resident_bytes(transition: ResidualChunkTransition) -> int:
    """Estimate resident bytes for one chunk transition (P2-1)."""
    total = 0
    for field in dataclasses.fields(transition):
        total += _array_bytes(getattr(transition, field.name))
    return total


class ResidualHilReplayBuffer:
    """Online replay + demo (intervention) replay with auditable sampling.

    Rules:
    - Every valid transition enters ``online``.
    - Transitions with any executed human intervention additionally enter
      ``demo`` (with their original source preserved).  ``demo`` holds the
      *same object references* as ``online``, so images are never duplicated.
    - A transition id ``(source, episode_id, chunk_id)`` may only appear once
      per buffer; duplicate inserts are counted and ignored.
    - Storage is bounded by transition counts *and* byte budgets; eviction is
      O(1) via ``deque.popleft`` (ring semantics).
    - Sampling uses the configured ``demo_ratio`` (default 0.5), auditable per
      batch.
    - When usage passes ``memory_high_watermark``, ``can_train()`` returns
      False so the learner pauses instead of OOM-ing the run.
    """

    def __init__(
        self,
        *,
        codec_fingerprint: str | None = None,
        base_fingerprint: str | None = None,
        min_demo_size: int = 1,
        max_online_transitions: int = 20_000,
        max_demo_transitions: int = 5_000,
        max_online_bytes: float = 40e9,
        max_demo_bytes: float = 10e9,
        memory_high_watermark: float = 0.9,
        demo_ratio: float = 0.5,
        seed: int = 0,
    ):
        if not 0.0 < demo_ratio < 1.0:
            raise ValueError(f"demo_ratio must be in (0, 1), got {demo_ratio}")
        self.min_demo_size = int(min_demo_size)
        self.codec_fingerprint = codec_fingerprint
        self.base_fingerprint = base_fingerprint
        self.max_online_transitions = int(max_online_transitions)
        self.max_demo_transitions = int(max_demo_transitions)
        self.max_online_bytes = float(max_online_bytes)
        self.max_demo_bytes = float(max_demo_bytes)
        self.memory_high_watermark = float(memory_high_watermark)
        self.demo_ratio = float(demo_ratio)
        self._online: deque[ResidualChunkTransition] = deque()
        self._demo: deque[ResidualChunkTransition] = deque()
        self._online_ids: set[tuple[int, int, int]] = set()
        self._demo_ids: set[tuple[int, int, int]] = set()
        self._online_bytes = 0
        self._demo_bytes = 0
        # Auditable counters (P2-9).
        self.accepted_count = 0
        self.duplicate_count = 0
        self.rejected_count = 0
        self.evicted_count = 0
        self.reason_counts: dict[str, int] = {}
        self._rng = np.random.default_rng(seed)

    # ── Insertion ───────────────────────────────────────────────────────────

    def add(self, transition: ResidualChunkTransition) -> bool:
        """Insert one transition; returns False when rejected/duplicate."""
        if not bool(transition.transition_valid[0]):
            self._reject("invalid_transition")
            return False
        if (
            self.codec_fingerprint is not None
            and transition.codec_fingerprint != self.codec_fingerprint
        ):
            self._reject("codec_fingerprint_mismatch")
            return False
        if (
            self.base_fingerprint is not None
            and transition.base_fingerprint != self.base_fingerprint
        ):
            self._reject("base_fingerprint_mismatch")
            return False
        transition_id = (
            int(transition.source),
            int(transition.episode_id),
            int(transition.chunk_id),
        )
        added_any = False
        was_duplicate = False
        if transition_id not in self._online_ids:
            self._online_ids.add(transition_id)
            self._online.append(transition)
            self._online_bytes += transition_resident_bytes(transition)
            self._trim(
                self._online,
                self._online_ids,
                self.max_online_transitions,
                self.max_online_bytes,
                "_online_bytes",
            )
            added_any = True
        else:
            was_duplicate = True
        if (
            transition.source in (SOURCE_ONLINE_INTERVENTION, SOURCE_OFFLINE_DEMO)
            or bool(transition.human_intervention_mask.any())
        ):
            if transition_id not in self._demo_ids:
                self._demo_ids.add(transition_id)
                self._demo.append(transition)  # same object as online
                self._demo_bytes += transition_resident_bytes(transition)
                self._trim(
                    self._demo,
                    self._demo_ids,
                    self.max_demo_transitions,
                    self.max_demo_bytes,
                    "_demo_bytes",
                )
                added_any = True
            else:
                was_duplicate = True
        if was_duplicate and not added_any:
            self.duplicate_count += 1
        if added_any:
            self.accepted_count += 1
        return added_any

    def _reject(self, reason: str) -> None:
        self.rejected_count += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def _trim(
        self,
        buffer: deque[ResidualChunkTransition],
        ids: set[tuple[int, int, int]],
        max_transitions: int,
        max_bytes: float,
        bytes_attr: str,
    ) -> None:
        """Evict oldest entries (O(1) popleft) until count and byte budgets hold."""
        while len(buffer) > max_transitions or (
            max_bytes > 0 and getattr(self, bytes_attr) > max_bytes
        ):
            dropped = buffer.popleft()
            ids.discard(
                (
                    int(dropped.source),
                    int(dropped.episode_id),
                    int(dropped.chunk_id),
                )
            )
            setattr(
                self,
                bytes_attr,
                max(0, getattr(self, bytes_attr) - transition_resident_bytes(dropped)),
            )
            self.evicted_count += 1

    # ── Status ──────────────────────────────────────────────────────────────

    def waiting_for_demo(self) -> bool:
        return len(self._demo) < self.min_demo_size

    def sizes(self) -> dict[str, int]:
        return {"online": len(self._online), "demo": len(self._demo)}

    def memory_usage(self) -> dict[str, float]:
        return {
            "online_bytes": float(self._online_bytes),
            "demo_bytes": float(self._demo_bytes),
            "online_bytes_ratio": (
                self._online_bytes / self.max_online_bytes
                if self.max_online_bytes > 0
                else 0.0
            ),
            "demo_bytes_ratio": (
                self._demo_bytes / self.max_demo_bytes
                if self.max_demo_bytes > 0
                else 0.0
            ),
        }

    def over_memory_watermark(self) -> bool:
        usage = self.memory_usage()
        return max(usage["online_bytes_ratio"], usage["demo_bytes_ratio"]) >= (
            self.memory_high_watermark
        )

    def can_train(self) -> bool:
        """False when waiting for demos or past the memory high watermark."""
        return not self.waiting_for_demo() and not self.over_memory_watermark()

    def source_counts(self) -> dict[str, int]:
        counts = {str(k): 0 for k in range(3)}
        for transition in self._online:
            counts[str(int(transition.source))] += 1
        return counts

    def stats(self) -> dict[str, Any]:
        return {
            **self.sizes(),
            **self.memory_usage(),
            "accepted_count": self.accepted_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "evicted_count": self.evicted_count,
            "reason_counts": dict(self.reason_counts),
        }

    # ── Symmetric sampling ──────────────────────────────────────────────────

    def sample(
        self, batch_size: int
    ) -> tuple[list[ResidualChunkTransition], np.ndarray]:
        """Sample ``batch_size`` transitions with the configured demo ratio.

        Args:
            batch_size: Must be even (kept for backward compatibility with the
                strict 50/50 contract; larger ratios still divide evenly).

        Returns:
            ``(transitions, source_mask)`` where ``source_mask[i] == 1`` marks
            demo samples.
        """
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")
        if self.waiting_for_demo():
            raise RuntimeError(
                "waiting_for_demo_buffer: not enough demo transitions "
                f"({len(self._demo)} < {self.min_demo_size})"
            )
        demo_count = int(round(batch_size * self.demo_ratio))
        online_count = batch_size - demo_count
        if online_count < 1 or demo_count < 1:
            raise RuntimeError(
                f"demo_ratio={self.demo_ratio} cannot fill batch {batch_size} "
                "with both sources"
            )
        online_idx = self._rng.integers(0, len(self._online), size=online_count)
        demo_idx = self._rng.integers(0, len(self._demo), size=demo_count)
        transitions = [self._online[int(i)] for i in online_idx] + [
            self._demo[int(i)] for i in demo_idx
        ]
        source_mask = np.concatenate(
            [
                np.zeros(online_count, dtype=np.int64),
                np.ones(demo_count, dtype=np.int64),
            ]
        )
        return transitions, source_mask

    def audit_ratio(
        self, transitions: list[ResidualChunkTransition], source_mask: np.ndarray
    ) -> float:
        if not transitions:
            return 0.0
        return float(np.mean(source_mask))

    # ── Persistence ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": list(self._online),
            "demo": list(self._demo),
            "online_ids": sorted(self._online_ids),
            "demo_ids": sorted(self._demo_ids),
            "online_bytes": self._online_bytes,
            "demo_bytes": self._demo_bytes,
            "rng_state": self._rng.bit_generator.state,
            "accepted_count": self.accepted_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "evicted_count": self.evicted_count,
            "reason_counts": dict(self.reason_counts),
            "min_demo_size": self.min_demo_size,
            "max_online_transitions": self.max_online_transitions,
            "max_demo_transitions": self.max_demo_transitions,
            "max_online_bytes": self.max_online_bytes,
            "max_demo_bytes": self.max_demo_bytes,
            "demo_ratio": self.demo_ratio,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._online = deque(state["online"])
        self._demo = deque(state["demo"])
        self._online_ids = set(state["online_ids"])
        self._demo_ids = set(state["demo_ids"])
        self._online_bytes = int(state.get("online_bytes", 0))
        self._demo_bytes = int(state.get("demo_bytes", 0))
        self.accepted_count = int(state.get("accepted_count", 0))
        self.duplicate_count = int(state.get("duplicate_count", 0))
        self.rejected_count = int(state.get("rejected_count", 0))
        self.evicted_count = int(state.get("evicted_count", 0))
        self.reason_counts = dict(state.get("reason_counts", {}))
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
        self.min_demo_size = int(state["min_demo_size"])
        self.max_online_transitions = int(state["max_online_transitions"])
        self.max_demo_transitions = int(state["max_demo_transitions"])
        self.max_online_bytes = float(state["max_online_bytes"])
        self.max_demo_bytes = float(state["max_demo_bytes"])
        self.demo_ratio = float(state.get("demo_ratio", 0.5))


def transitions_to_torch_batch(
    transitions: list[ResidualChunkTransition],
) -> dict[str, torch.Tensor]:
    """Pack transitions into a flat torch batch for the learner."""
    batch: dict[str, torch.Tensor] = {}

    def _tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _stack(name: str) -> torch.Tensor:
        arrays = [getattr(t, name) for t in transitions]
        if name in ("bootstrap_mask", "discount_steps", "policy_version"):
            return torch.as_tensor(
                np.stack(arrays, axis=0).reshape(len(arrays)),
                dtype=torch.float32,
            )
        return torch.as_tensor(np.stack(arrays, axis=0), dtype=torch.float32)

    batch["actions_arm"] = _stack("actions_arm")
    batch["actions_gripper"] = _stack("actions_gripper")
    batch["executed_action_mask"] = _stack("executed_action_mask")
    batch["bootstrap_mask"] = _stack("bootstrap_mask")
    batch["discount_steps"] = _stack("discount_steps")
    batch["policy_version"] = _stack("policy_version")
    batch["discounted_return"] = torch.as_tensor(
        [t.discounted_return for t in transitions], dtype=torch.float32
    )

    def _obs_stack(key: str) -> torch.Tensor:
        tensors = [_tensor(t.curr_obs[key]) for t in transitions]
        stacked = torch.stack(tensors, dim=0)
        if stacked.ndim >= 3 and stacked.shape[1] == 1:
            return stacked.squeeze(1)
        return stacked

    def _next_obs_stack(key: str) -> torch.Tensor:
        tensors = [_tensor(t.next_obs[key]) for t in transitions]
        stacked = torch.stack(tensors, dim=0)
        if stacked.ndim >= 3 and stacked.shape[1] == 1:
            return stacked.squeeze(1)
        return stacked

    batch["curr_images"] = preprocess_image(_obs_stack("main_images"))
    batch["curr_proprio"] = _obs_stack("prev_states").float()
    batch["nominal"] = _stack("nominal_actions")
    batch["next_images"] = preprocess_image(_next_obs_stack("main_images"))
    batch["next_proprio"] = _next_obs_stack("prev_states").float()
    batch["next_nominal"] = _stack("next_nominal_actions")
    return batch
