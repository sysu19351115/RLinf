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

"""EnvWorker diagnostics tests for Dobot HG-DAgger."""

from types import SimpleNamespace

import numpy as np
import torch

from rlinf.workers.env.env_worker import EnvWorker


def test_extracts_operator_reason_and_chunk_execution_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["operator_abort"]),
            "skipped_action_steps": 3,
            "executed_action_mask": torch.tensor([[True, False, False, False]]),
            "handoff_hold_mask": torch.tensor([[False, True, False, False]]),
        }
    )

    torch.testing.assert_close(
        metrics["episode_end/operator_abort"], torch.tensor([1.0])
    )
    torch.testing.assert_close(
        metrics["episode_end/skipped_action_steps"], torch.tensor([3.0])
    )
    torch.testing.assert_close(
        metrics["episode_end/executed_action_fraction"], torch.tensor([0.25])
    )
    torch.testing.assert_close(
        metrics["control/handoff_hold_fraction"], torch.tensor([0.25])
    )


def test_does_not_emit_false_operator_reason_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["timeout"]),
            "executed_action_mask": torch.ones((1, 4), dtype=torch.bool),
        }
    )

    assert not any(key.startswith("episode_end/operator_") for key in metrics)
    torch.testing.assert_close(
        metrics["episode_end/executed_action_fraction"], torch.tensor([1.0])
    )


def test_extracts_keyboard_failure_reason():
    metrics = EnvWorker._extract_operator_metrics(
        {"termination_reason": np.array(["keyboard_disconnected"])}
    )

    torch.testing.assert_close(
        metrics["episode_end/keyboard_disconnected"], torch.tensor([1.0])
    )


def test_extracts_controller_rejection_metrics_and_audit_code():
    infos = {
        "termination_reason": np.array(["controller_rejection"]),
        "executed_action_mask": torch.tensor([[True, False, False, False]]),
        "episode_id": torch.tensor([4]),
        "episode_step_ids": torch.tensor([[0, 1, -1, -1]]),
    }

    metrics = EnvWorker._extract_operator_metrics(infos)
    audit_info = EnvWorker._extract_trajectory_audit_info(infos)

    torch.testing.assert_close(
        metrics["episode_end/controller_rejection"], torch.tensor([1.0])
    )
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([6]))


def test_extracts_unsafe_model_handoff_metrics_and_audit_code():
    infos = {
        "termination_reason": np.array(["unsafe_model_handoff"]),
        "executed_action_mask": torch.tensor([[True, False]]),
        "episode_id": torch.tensor([5]),
        "episode_step_ids": torch.tensor([[10, -1]]),
    }

    metrics = EnvWorker._extract_operator_metrics(infos)
    audit_info = EnvWorker._extract_trajectory_audit_info(infos)

    torch.testing.assert_close(
        metrics["episode_end/unsafe_model_handoff"], torch.tensor([1.0])
    )
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([7]))


def test_extracts_trajectory_audit_info_from_final_info():
    audit_info = EnvWorker._extract_trajectory_audit_info(
        {
            "final_info": {
                "executed_action_mask": torch.tensor([[True, True, False, False]]),
                "termination_reason": np.array(["operator_abort"]),
                "episode_id": torch.tensor([9]),
                "episode_step_ids": torch.tensor([[20, 21, -1, -1]]),
                "handoff_hold_mask": torch.tensor([[False, True, False, False]]),
            }
        }
    )

    assert audit_info["executed_action_mask"].dtype == torch.bool
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([2]))
    torch.testing.assert_close(audit_info["episode_id"], torch.tensor([9]))
    torch.testing.assert_close(
        audit_info["episode_step_ids"], torch.tensor([[20, 21, -1, -1]])
    )
    torch.testing.assert_close(
        audit_info["handoff_hold_mask"],
        torch.tensor([[False, True, False, False]]),
    )


def test_missing_handoff_hold_mask_defaults_to_false():
    audit_info = EnvWorker._extract_trajectory_audit_info(
        {
            "executed_action_mask": torch.tensor([[True, True]]),
            "episode_id": torch.tensor([1]),
            "episode_step_ids": torch.tensor([[0, 1]]),
        }
    )

    torch.testing.assert_close(
        audit_info["handoff_hold_mask"], torch.tensor([[False, False]])
    )


def test_handoff_hold_overrides_action_without_becoming_human_label():
    class RolloutRecorder:
        def __init__(self):
            self.actions = None
            self.override_flags = None
            self.human_flags = None

        def update_last_actions(self, actions, flags):
            self.actions = actions
            self.override_flags = flags

        def mark_last_step_with_intervene_flags(self, flags):
            self.human_flags = flags

    recorder = RolloutRecorder()
    worker = EnvWorker.__new__(EnvWorker)
    worker.rollout_results = [recorder]
    human_flags = torch.tensor([[True, False, False, False]])
    handoff_hold_mask = torch.tensor([[False, False, True, True]])
    override_actions = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    env_output = SimpleNamespace(
        intervene_actions=override_actions,
        intervene_flags=human_flags,
        env_infos={"handoff_hold_mask": handoff_hold_mask},
    )

    worker._apply_last_action_overrides(0, env_output)

    torch.testing.assert_close(recorder.actions, override_actions)
    torch.testing.assert_close(
        recorder.override_flags,
        torch.tensor([[True, False, True, True]]),
    )
    torch.testing.assert_close(recorder.human_flags, human_flags)


def test_missing_execution_metadata_does_not_create_audit_record():
    assert EnvWorker._extract_trajectory_audit_info({}) == {}
