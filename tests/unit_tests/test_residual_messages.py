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

"""Strict message schema (P2-6) tests."""

from __future__ import annotations

import pytest

from rlinf.algorithms.residual_hil_rlpd.messages import (
    MESSAGE_SCHEMA_VERSION,
    MESSAGE_TYPE_SCALE,
    MESSAGE_TYPE_SCALE_ACK,
    MESSAGE_TYPE_TRANSITION,
    ScaleAckTracker,
    build_ack_message,
    build_scale_message,
    parse_scale_message,
    validate_message,
    validate_transition_message,
)


def test_scale_message_round_trip_and_version_binding():
    message = build_scale_message(0.3, policy_version=7, run_id="run-1")
    scale, gripper_enabled = parse_scale_message(
        message, applied_version=7, run_id="run-1"
    )
    assert scale == 0.3
    assert gripper_enabled is False

    # Wrong applied weight version -> fail closed to 0.0 / disabled.
    scale, gripper_enabled = parse_scale_message(
        message, applied_version=6, run_id="run-1"
    )
    assert scale == 0.0
    assert gripper_enabled is False


def test_scale_gripper_flag_and_range():
    message = build_scale_message(
        0.5, policy_version=1, gripper_enabled=True
    )
    scale, gripper_enabled = parse_scale_message(
        message, applied_version=1
    )
    assert scale == 0.5
    assert gripper_enabled is True

    bad = build_scale_message(1.5, policy_version=1)
    scale, gripper_enabled = parse_scale_message(bad, applied_version=1)
    assert scale == 0.0
    assert gripper_enabled is False


def test_malformed_messages_fail_closed():
    assert parse_scale_message(None, applied_version=0) == (0.0, False)
    assert parse_scale_message(
        {"type": "other"}, applied_version=0
    ) == (0.0, False)
    bad_schema = build_scale_message(0.5, policy_version=0)
    bad_schema["schema_version"] = 999
    assert parse_scale_message(bad_schema, applied_version=0) == (0.0, False)


def test_validate_message_rejects_wrong_type_and_run_id():
    with pytest.raises(ValueError):
        validate_message(
            {"type": "other", "schema_version": MESSAGE_SCHEMA_VERSION},
            MESSAGE_TYPE_SCALE,
        )
    message = build_scale_message(0.5, policy_version=0, run_id="run-a")
    with pytest.raises(ValueError):
        validate_message(message, MESSAGE_TYPE_SCALE, run_id="run-b")


def test_transition_envelope_validation():
    message = {
        "type": MESSAGE_TYPE_TRANSITION,
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "run_id": "",
        "sender_rank": 0,
        "policy_version": 3,
        "transition": object(),
    }
    validated = validate_transition_message(message)
    assert validated["policy_version"] == 3
    with pytest.raises(ValueError):
        validate_transition_message({"type": MESSAGE_TYPE_SCALE})


def test_ack_message_validation():
    ack = build_ack_message(MESSAGE_TYPE_SCALE_ACK, policy_version=4, run_id="r")
    assert ack["type"] == MESSAGE_TYPE_SCALE_ACK
    assert ack["policy_version"] == 4
    validated = validate_message(ack, MESSAGE_TYPE_SCALE_ACK, run_id="r")
    assert validated["ok"] is True
    with pytest.raises(ValueError):
        build_ack_message("unknown_ack", policy_version=0)


def test_scale_ack_tracker_timeout_and_retry():
    tracker = ScaleAckTracker(timeout_s=1.0, max_retries=2)
    tracker.expect_ack(version=4, now=100.0)
    assert tracker.is_pending(now=100.5)
    assert tracker.on_timeout(now=101.5) is True  # first retry allowed
    assert tracker.retries == 1
    assert tracker.timeouts == 1
    # ACK for the wrong version is rejected.
    assert tracker.record_ack(version=3) is False
    # Correct ACK after retry.
    assert tracker.record_ack(version=4) is True
    assert tracker.acks_received == 1
    assert tracker.last_ack_ok is True
    assert not tracker.is_pending(now=200.0)


def test_scale_ack_tracker_exhausts_retries():
    tracker = ScaleAckTracker(timeout_s=1.0, max_retries=2)
    tracker.expect_ack(version=4, now=100.0)
    assert tracker.on_timeout(now=101.0) is True
    assert tracker.on_timeout(now=102.0) is True
    assert tracker.on_timeout(now=103.0) is False  # retries exhausted
    assert tracker.retries == 3
    assert not tracker.is_pending(now=104.0)
    stats = tracker.stats()
    assert stats["scale_ack_timeouts"] == 3
