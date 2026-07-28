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

"""Regression tests for review Issues #1 and #2.

Issue #1: _window_to_trajectory must preserve multi-D forward_inputs
          (images [C, H, W]) without dropping the channel dimension.
Issue #2: termination/padding chunks with sentinel step_ids (all -1)
          must be handled by clear_env, not raise in _validate_chunk.
"""

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    AuditedExecutionChunk,
    ExpertAnchoredWindowAssembler,
)
from rlinf.workers.actor.fsdp_dagger_policy_worker import (
    EmbodiedDAGGERFSDPPolicy,
)

_ACTION_DIM = 8
_CHUNK = 10
_WINDOW = 50


def _make_window_with_images(human_steps: int = 10) -> dict:
    """Build a completed window dict carrying image forward_inputs."""
    total = _WINDOW
    model_steps = total - human_steps
    human_mask = torch.tensor(
        [True] * human_steps + [False] * model_steps, dtype=torch.bool
    )
    step_ids = torch.arange(total)
    return {
        "action": torch.randn(total, _ACTION_DIM),
        "human_action_mask": human_mask,
        "executed_action_mask": torch.ones(total, dtype=torch.bool),
        "episode_step_ids": step_ids,
        "episode_id": 42,
        "forward_inputs": {
            "observation/prev_state": torch.randn(_ACTION_DIM),
            "observation/state": torch.randn(_ACTION_DIM),
            "observation/image": torch.randn(3, 224, 224),
            "observation/wrist_image": torch.randn(3, 224, 224),
            "model_action": torch.randn(_WINDOW * 32),
            "chains": torch.randn(4, _WINDOW, 32),
            "denoise_inds": torch.arange(4),
        },
        "model_version": torch.tensor([7]),
        "human_steps": human_steps,
        "model_steps": model_steps,
        "human_fraction": human_steps / total,
    }


def test_multid_forward_inputs_preserved():
    """Image tensors must get [1, 1, C, H, W], not lose channel dim."""
    window = _make_window_with_images(human_steps=30)
    traj = EmbodiedDAGGERFSDPPolicy._window_to_trajectory(window, _ACTION_DIM)

    assert traj.forward_inputs["observation/image"].shape == (1, 1, 3, 224, 224)
    assert traj.forward_inputs["observation/wrist_image"].shape == (
        1,
        1,
        3,
        224,
        224,
    )
    assert traj.forward_inputs["observation/prev_state"].shape == (1, 1, 8)
    assert traj.forward_inputs["observation/state"].shape == (1, 1, 8)
    assert traj.forward_inputs["human_action_mask"].shape == (1, 1, _WINDOW)
    assert "model_action" not in traj.forward_inputs
    assert "chains" not in traj.forward_inputs
    assert "denoise_inds" not in traj.forward_inputs
    assert traj.versions.shape == (1, 1)
    assert traj.versions.item() == 7
    assert traj.intervene_flags.shape == (1, 1, _WINDOW * _ACTION_DIM)
    expected_flags = (
        traj.audit_info["human_action_mask"]
        .unsqueeze(-1)
        .expand(-1, -1, -1, _ACTION_DIM)
    )
    assert torch.equal(traj.intervene_flags, expected_flags.reshape(1, 1, -1))


def test_min_human_steps_counts_steps_not_action_elements():
    window = _make_window_with_images(human_steps=10)
    traj = EmbodiedDAGGERFSDPPolicy._window_to_trajectory(window, _ACTION_DIM)

    with pytest.raises(ValueError, match="only 10 human steps"):
        traj.validate_expert_replay_contract(min_human_steps=11)


def test_multid_forward_inputs_values_not_corrupted():
    """Ensure pixel data is byte-for-byte preserved, not reshuffled."""
    window = _make_window_with_images(human_steps=10)
    original = window["forward_inputs"]["observation/image"]
    traj = EmbodiedDAGGERFSDPPolicy._window_to_trajectory(window, _ACTION_DIM)
    result = traj.forward_inputs["observation/image"]
    assert torch.equal(result.squeeze(0).squeeze(0), original)


def test_model_weights_id_derived():
    """Issue #5: model_weights_id should carry the version, not be empty."""
    window = _make_window_with_images(human_steps=10)
    traj = EmbodiedDAGGERFSDPPolicy._window_to_trajectory(window, _ACTION_DIM)
    assert traj.model_weights_id == "v7"


def _make_chunk(
    ep_id: int = 0,
    start_step: int = 0,
    human: bool = True,
    action_dim: int = _ACTION_DIM,
) -> AuditedExecutionChunk:
    return AuditedExecutionChunk(
        episode_id=ep_id,
        episode_step_ids=torch.arange(start_step, start_step + _CHUNK),
        action=torch.randn(_CHUNK, action_dim),
        human_action_mask=(
            torch.ones(_CHUNK, dtype=torch.bool)
            if human
            else torch.zeros(_CHUNK, dtype=torch.bool)
        ),
        executed_action_mask=torch.ones(_CHUNK, dtype=torch.bool),
        forward_inputs={
            "observation/prev_state": torch.randn(action_dim),
            "observation/image": torch.randn(3, 224, 224),
        },
        model_version=torch.tensor([1]),
        termination_reason_code=0,
    )


def test_assembler_preserves_multid_forward_inputs():
    """End-to-end: trajectory with images through assembler to window."""
    asm = ExpertAnchoredWindowAssembler(
        execution_chunk_steps=_CHUNK,
        training_window_steps=_WINDOW,
        window_stride_steps=_CHUNK,
        min_human_steps_per_window=10,
        action_dim=_ACTION_DIM,
    )
    for i in range(5):
        human = i == 0
        asm.ingest_chunk(0, _make_chunk(ep_id=1, start_step=i * _CHUNK, human=human))
    windows = asm.emit_windows()
    assert len(windows) == 1
    assert windows[0]["forward_inputs"]["observation/image"].shape == (3, 224, 224)
    assert windows[0]["forward_inputs"]["observation/prev_state"].shape == (8,)

    traj = EmbodiedDAGGERFSDPPolicy._window_to_trajectory(windows[0], _ACTION_DIM)
    assert traj.forward_inputs["observation/image"].shape == (1, 1, 3, 224, 224)


def test_termination_chunk_with_sentinel_step_ids_does_not_raise():
    """A termination chunk with all-(-1) step_ids must clear, not crash."""
    asm = ExpertAnchoredWindowAssembler(
        execution_chunk_steps=_CHUNK,
        training_window_steps=_WINDOW,
        window_stride_steps=_CHUNK,
        min_human_steps_per_window=10,
        action_dim=_ACTION_DIM,
    )
    asm.ingest_chunk(0, _make_chunk(ep_id=1, start_step=0, human=True))
    assert len(asm._anchors.get(0, [])) == 1

    sentinel_chunk = AuditedExecutionChunk(
        episode_id=1,
        episode_step_ids=torch.full((_CHUNK,), -1, dtype=torch.long),
        action=torch.randn(_CHUNK, _ACTION_DIM),
        human_action_mask=torch.zeros(_CHUNK, dtype=torch.bool),
        executed_action_mask=torch.zeros(_CHUNK, dtype=torch.bool),
        forward_inputs={
            "observation/prev_state": torch.randn(_ACTION_DIM),
        },
        model_version=torch.tensor([1]),
        termination_reason_code=1,
    )
    asm.ingest_chunk(0, sentinel_chunk)
    assert 0 not in asm._anchors
    assert asm.emit_windows() == []


def test_padding_chunk_with_sentinel_step_ids_does_not_raise():
    """An unexecuted padding chunk with sentinel step_ids must clear."""
    asm = ExpertAnchoredWindowAssembler(
        execution_chunk_steps=_CHUNK,
        training_window_steps=_WINDOW,
        window_stride_steps=_CHUNK,
        min_human_steps_per_window=10,
        action_dim=_ACTION_DIM,
    )
    asm.ingest_chunk(0, _make_chunk(ep_id=1, start_step=0, human=True))

    padding_chunk = AuditedExecutionChunk(
        episode_id=1,
        episode_step_ids=torch.full((_CHUNK,), -1, dtype=torch.long),
        action=torch.randn(_CHUNK, _ACTION_DIM),
        human_action_mask=torch.zeros(_CHUNK, dtype=torch.bool),
        executed_action_mask=torch.zeros(_CHUNK, dtype=torch.bool),
        forward_inputs={"observation/prev_state": torch.randn(_ACTION_DIM)},
        model_version=torch.tensor([1]),
        termination_reason_code=0,
    )
    asm.ingest_chunk(0, padding_chunk)
    assert 0 not in asm._anchors
    assert asm.emit_windows() == []
