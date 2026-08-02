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

"""Strict distributed message schema for the residual HIL-RLPD stack (P2-6).

All cross-process messages carry ``type``, ``schema_version``, ``run_id``,
``policy_version`` and ``sender_rank``.  Receivers validate every field and
fail closed (reject the message / fall back to nominal-only) rather than
acting on partially understood data.
"""

from __future__ import annotations

import math
import time
from typing import Any

MESSAGE_SCHEMA_VERSION = 1

MESSAGE_TYPE_TRANSITION = "residual_transition"
MESSAGE_TYPE_SCALE = "residual_scale"
MESSAGE_TYPE_SCALE_ACK = "residual_scale_ack"


def _require_finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"residual message field {field} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"residual message field {field} is not finite")
    return number


def validate_message(
    message: Any,
    expected_type: str,
    *,
    run_id: str = "",
) -> dict[str, Any]:
    """Validate a message; raises ``ValueError`` on any schema violation."""
    if not isinstance(message, dict):
        raise ValueError(
            f"residual message must be a dict, got {type(message).__name__}"
        )
    if message.get("type") != expected_type:
        raise ValueError(
            f"residual message type mismatch: expected {expected_type!r}, "
            f"got {message.get('type')!r}"
        )
    schema_version = message.get("schema_version", None)
    if schema_version is None or int(schema_version) != MESSAGE_SCHEMA_VERSION:
        raise ValueError(
            "residual message schema_version mismatch: "
            f"expected {MESSAGE_SCHEMA_VERSION}, got {schema_version!r}"
        )
    if run_id and message.get("run_id", "") != run_id:
        raise ValueError(
            "residual message run_id mismatch: "
            f"expected {run_id!r}, got {message.get('run_id')!r}"
        )
    if not isinstance(message.get("sender_rank", 0), int):
        raise ValueError("residual message sender_rank must be an int")
    return message


def build_scale_message(
    scale: float,
    policy_version: int,
    *,
    run_id: str = "",
    sender_rank: int = 0,
    gripper_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "type": MESSAGE_TYPE_SCALE,
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "sender_rank": int(sender_rank),
        "policy_version": int(policy_version),
        "scale": float(scale),
        "gripper_enabled": bool(gripper_enabled),
    }


def build_transition_message(
    transition: Any,
    *,
    policy_version: int,
    run_id: str = "",
    sender_rank: int = 0,
) -> dict[str, Any]:
    """Strict envelope for env -> actor transition messages (P2-6/P0)."""
    return {
        "type": MESSAGE_TYPE_TRANSITION,
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "sender_rank": int(sender_rank),
        "policy_version": int(policy_version),
        "transition": transition,
    }


def parse_scale_message(
    message: Any,
    *,
    applied_version: int,
    run_id: str = "",
) -> tuple[float, bool]:
    """Parse a scale message, binding it to the already-applied weight version.

    Fail-closed: any schema violation, version mismatch or out-of-range value
    yields ``(0.0, False)`` (nominal-only, gripper KEEP), never the previous
    state.
    """
    try:
        validated = validate_message(
            message, MESSAGE_TYPE_SCALE, run_id=run_id
        )
        if int(validated["policy_version"]) != int(applied_version):
            raise ValueError(
                "scale message policy_version does not match the applied "
                f"weight version: {validated['policy_version']} != {applied_version}"
            )
        scale = _require_finite(validated["scale"], "scale")
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"scale out of range [0, 1]: {scale}")
        return scale, bool(validated.get("gripper_enabled", False))
    except (ValueError, KeyError, TypeError):
        return 0.0, False


def build_ack_message(
    ack_type: str,
    policy_version: int,
    *,
    run_id: str = "",
    sender_rank: int = 0,
) -> dict[str, Any]:
    if ack_type != MESSAGE_TYPE_SCALE_ACK:
        raise ValueError(f"unknown ack type {ack_type!r}")
    return {
        "type": ack_type,
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "sender_rank": int(sender_rank),
        "policy_version": int(policy_version),
        "ok": True,
    }


def validate_transition_message(
    message: Any,
    *,
    run_id: str = "",
) -> dict[str, Any]:
    """Validate a transition envelope; the payload is checked on ingest."""
    return validate_message(message, MESSAGE_TYPE_TRANSITION, run_id=run_id)


class ScaleAckTracker:
    """Timeout/retry state machine for the rollout scale ACK (P2-6).

    The actor broadcasts a scale message bound to an applied weight version;
    the rollout replies ``residual_scale_ack``.  This tracker counts timeouts
    and retries.  A lost ACK is informational (the rollout only *applies* the
    scale when its own applied weight version matches), but repeated timeouts
    are surfaced in metrics so operators can see the sync channel degrading.
    """

    def __init__(self, *, timeout_s: float = 10.0, max_retries: int = 2):
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retries = 0
        self.timeouts = 0
        self.acks_received = 0
        self.last_ack_ok = False
        self._pending_version: int | None = None
        self._deadline: float | None = None

    def expect_ack(self, version: int, now: float | None = None) -> float:
        """Start waiting for an ACK at the given weight version."""
        self._pending_version = int(version)
        self._deadline = (time.monotonic() if now is None else float(now)) + (
            self.timeout_s
        )
        return self._deadline

    def record_ack(self, version: int) -> bool:
        """Return True when the ACK matches the pending version."""
        if self._pending_version is not None and int(version) == self._pending_version:
            self.acks_received += 1
            self.last_ack_ok = True
            self._pending_version = None
            self._deadline = None
            return True
        self.last_ack_ok = False
        return False

    def on_timeout(self, now: float | None = None) -> bool:
        """Handle a timeout; returns True when a retry is allowed."""
        self.timeouts += 1
        self.last_ack_ok = False
        if self._pending_version is None:
            return False
        self.retries += 1
        if self.retries > self.max_retries:
            self._pending_version = None
            self._deadline = None
            return False
        self._deadline = (time.monotonic() if now is None else float(now)) + (
            self.timeout_s
        )
        return True

    def is_pending(self, now: float | None = None) -> bool:
        if self._pending_version is None:
            return False
        if self._deadline is not None and (now if now is not None else time.monotonic()) > self._deadline:
            self.on_timeout(now)
            return self._pending_version is not None
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "scale_ack_retries": self.retries,
            "scale_ack_timeouts": self.timeouts,
            "scale_acks_received": self.acks_received,
            "scale_ack_ok": bool(self.last_ack_ok),
        }
