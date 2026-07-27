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

import numpy as np
import torch

from rlinf.workers.env.env_worker import EnvWorker


def test_extracts_operator_reason_and_chunk_execution_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["operator_abort"]),
            "skipped_action_steps": 3,
            "executed_action_mask": torch.tensor([[True, False, False, False]]),
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
