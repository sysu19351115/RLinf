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

"""Dobot pose data-contract tests for trajectory-replay HG-DAgger."""

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)

_ACTION_DIM = 8
_CHUNK_SIZE = 4


def _forward_inputs() -> dict[str, torch.Tensor]:
    return {
        "action": torch.zeros((1, _CHUNK_SIZE * _ACTION_DIM)),
        "model_action": torch.ones((1, _CHUNK_SIZE * _ACTION_DIM)),
        "observation/state": torch.zeros((1, _ACTION_DIM)),
        "observation/prev_state": torch.full((1, _ACTION_DIM), 0.25),
        "observation/image": torch.zeros((1, 3, 8, 8)),
    }


def _rollout_result() -> EmbodiedRolloutResult:
    result = EmbodiedRolloutResult(max_episode_length=100)
    result.append_step_result(
        ChunkStepResult(
            actions=torch.zeros((1, _CHUNK_SIZE * _ACTION_DIM)),
            forward_inputs=_forward_inputs(),
            rewards=torch.zeros((1, _CHUNK_SIZE)),
            terminations=torch.zeros((1, _CHUNK_SIZE), dtype=torch.bool),
            truncations=torch.zeros((1, _CHUNK_SIZE), dtype=torch.bool),
            dones=torch.zeros((1, _CHUNK_SIZE), dtype=torch.bool),
            audit_info={
                "executed_action_mask": torch.ones((1, _CHUNK_SIZE), dtype=torch.bool),
                "termination_reason_code": torch.zeros((1,), dtype=torch.int64),
                "episode_id": torch.tensor([12], dtype=torch.int64),
                "episode_step_ids": torch.arange(_CHUNK_SIZE).reshape(1, -1),
            },
            versions=torch.tensor([[7]]),
        )
    )
    return result


def test_full_intervention_replaces_action_and_preserves_prev_state():
    result = _rollout_result()
    expert_action = torch.arange(
        _CHUNK_SIZE * _ACTION_DIM, dtype=torch.float32
    ).reshape(1, -1)
    flags = torch.ones((1, _CHUNK_SIZE), dtype=torch.bool)

    result.update_last_actions(expert_action, flags)
    trajectory = result.to_trajectory()
    extracted = trajectory.extract_intervene_traj(mode="all")

    assert extracted is not None
    assert len(extracted) == 1
    expert_trajectory = extracted[0]
    torch.testing.assert_close(expert_trajectory.actions[0, 0], expert_action[0])
    torch.testing.assert_close(
        expert_trajectory.forward_inputs["action"][0, 0], expert_action[0]
    )
    assert "model_action" not in expert_trajectory.forward_inputs
    torch.testing.assert_close(
        expert_trajectory.forward_inputs["observation/prev_state"][0, 0],
        torch.full((_ACTION_DIM,), 0.25),
    )
    assert expert_trajectory.intervene_flags.all()
    assert expert_trajectory.audit_info["executed_action_mask"].all()
    torch.testing.assert_close(
        expert_trajectory.audit_info["episode_id"], torch.tensor([[12]])
    )
    torch.testing.assert_close(expert_trajectory.versions, torch.tensor([[[7]]]))
    expert_trajectory.validate_expert_replay_contract(["observation/prev_state"])


def test_partial_intervention_chunk_is_not_an_expert_sample():
    result = _rollout_result()
    expert_action = torch.ones((1, _CHUNK_SIZE * _ACTION_DIM))
    flags = torch.tensor([[True, True, False, True]])

    result.update_last_actions(expert_action, flags)
    trajectory = result.to_trajectory()

    assert trajectory.extract_intervene_traj(mode="all") is None


def test_partially_executed_chunk_is_not_an_expert_sample():
    result = _rollout_result()
    expert_action = torch.ones((1, _CHUNK_SIZE * _ACTION_DIM))
    flags = torch.ones((1, _CHUNK_SIZE), dtype=torch.bool)
    result.update_last_actions(expert_action, flags)
    result.update_last_audit_info(
        {
            "executed_action_mask": torch.tensor([[True, True, False, False]]),
            "termination_reason_code": torch.tensor([2]),
            "episode_id": torch.tensor([12]),
            "episode_step_ids": torch.tensor([[0, 1, -1, -1]]),
        }
    )

    trajectory = result.to_trajectory()

    assert trajectory.extract_intervene_traj(mode="all") is None


def test_audit_info_rejects_inconsistent_chunk_shapes():
    result = _rollout_result()

    with pytest.raises(ValueError, match="episode_step_ids"):
        result.update_last_audit_info(
            {
                "executed_action_mask": torch.ones((1, _CHUNK_SIZE), dtype=torch.bool),
                "termination_reason_code": torch.tensor([0]),
                "episode_id": torch.tensor([12]),
                "episode_step_ids": torch.ones((1, 2), dtype=torch.int64),
            }
        )


def test_expert_contract_rejects_missing_prev_state():
    result = _rollout_result()
    result.update_last_actions(
        torch.ones((1, _CHUNK_SIZE * _ACTION_DIM)),
        torch.ones((1, _CHUNK_SIZE), dtype=torch.bool),
    )
    trajectory = result.to_trajectory()
    del trajectory.forward_inputs["observation/prev_state"]
    expert = trajectory.extract_intervene_traj(mode="all")[0]

    with pytest.raises(ValueError, match="observation/prev_state"):
        expert.validate_expert_replay_contract(["observation/prev_state"])


def test_expert_contract_rejects_nonfinite_actions():
    result = _rollout_result()
    expert_action = torch.ones((1, _CHUNK_SIZE * _ACTION_DIM))
    expert_action[0, 0] = torch.nan
    result.update_last_actions(
        expert_action, torch.ones((1, _CHUNK_SIZE), dtype=torch.bool)
    )
    expert = result.to_trajectory().extract_intervene_traj(mode="all")[0]

    with pytest.raises(ValueError, match="NaN or Inf"):
        expert.validate_expert_replay_contract(["observation/prev_state"])


def test_expert_contract_rejects_noncontiguous_step_ids():
    result = _rollout_result()
    result.update_last_actions(
        torch.ones((1, _CHUNK_SIZE * _ACTION_DIM)),
        torch.ones((1, _CHUNK_SIZE), dtype=torch.bool),
    )
    result.update_last_audit_info(
        {
            "executed_action_mask": torch.ones((1, _CHUNK_SIZE), dtype=torch.bool),
            "termination_reason_code": torch.tensor([0]),
            "episode_id": torch.tensor([12]),
            "episode_step_ids": torch.tensor([[0, 1, 3, 4]]),
        }
    )
    expert = result.to_trajectory().extract_intervene_traj(mode="all")[0]

    with pytest.raises(ValueError, match="not contiguous"):
        expert.validate_expert_replay_contract(["observation/prev_state"])
