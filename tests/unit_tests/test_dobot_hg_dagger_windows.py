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

"""Tests for ExpertAnchoredWindowAssembler hybrid 50-step window assembly."""

from __future__ import annotations

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    TERMINATION_REASON_CODES,
    AuditedExecutionChunk,
    ExpertAnchoredWindowAssembler,
)


def _make_chunk(
    step_start,
    human=True,
    episode_id=1,
    action_dim=8,
    chunk_size=10,
    executed=True,
    termination_code=0,
    model_version=1,
):
    return AuditedExecutionChunk(
        episode_id=episode_id,
        episode_step_ids=torch.arange(step_start, step_start + chunk_size),
        action=torch.full((chunk_size, action_dim), 0.5),
        human_action_mask=torch.full((chunk_size,), human, dtype=torch.bool),
        executed_action_mask=torch.full((chunk_size,), executed, dtype=torch.bool),
        forward_inputs={"observation/prev_state": torch.full((1, action_dim), 0.25)},
        model_version=torch.tensor(model_version),
        termination_reason_code=termination_code,
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"execution_chunk_steps": 0}, "execution_chunk_steps"),
        ({"training_window_steps": 0}, "training_window_steps"),
        ({"window_stride_steps": 0}, "window_stride_steps"),
        ({"min_human_steps_per_window": 0}, "min_human_steps_per_window"),
        ({"action_dim": 0}, "action_dim"),
        ({"training_window_steps": 45}, "multiple of execution_chunk_steps"),
        ({"window_stride_steps": 15}, "multiple of execution_chunk_steps"),
        ({"min_human_steps_per_window": 51}, "must not exceed"),
    ],
)
def test_invalid_assembler_configuration_fails_fast(kwargs, error):
    with pytest.raises(ValueError, match=error):
        ExpertAnchoredWindowAssembler(**kwargs)


def test_three_human_then_four_model_emits_three_windows():
    assembler = ExpertAnchoredWindowAssembler()
    # Three fully-human chunks become anchors at steps 0, 10, 20.
    for i in range(3):
        chunk = _make_chunk(i * 10, human=True)
        # Make forward_inputs unique per chunk to verify anchoring.
        chunk.forward_inputs = {
            "observation/prev_state": torch.full((1, 8), float(i * 10))
        }
        assembler.ingest_chunk(0, chunk)
    # Four model chunks complete the windows.
    for i in range(3, 7):
        assembler.ingest_chunk(0, _make_chunk(i * 10, human=False))

    windows = assembler.emit_windows()
    assert len(windows) == 3

    # Observations come from s0, s10, s20 (the anchor chunks).
    for w, expected_start in zip(windows, [0, 10, 20]):
        obs = w["forward_inputs"]["observation/prev_state"]
        assert torch.allclose(obs, torch.full((1, 8), float(expected_start)))

    # Action patterns: HHHMM, HHMMM, HMMMM (H=10 human steps, M=10 model steps).
    expected_human_steps = [30, 20, 10]
    expected_model_steps = [20, 30, 40]
    for i, w in enumerate(windows):
        assert w["human_steps"] == expected_human_steps[i]
        assert w["model_steps"] == expected_model_steps[i]
        mask = w["human_action_mask"]
        assert mask.shape == (50,)
        assert mask[: expected_human_steps[i]].all()
        assert not mask[expected_human_steps[i] :].any()


def test_candidate_not_emitted_until_fifth_chunk():
    assembler = ExpertAnchoredWindowAssembler()
    # One human anchor + three model chunks = four chunks, no emission.
    assembler.ingest_chunk(0, _make_chunk(0, human=True))
    for i in range(1, 4):
        assembler.ingest_chunk(0, _make_chunk(i * 10, human=False))
    assert len(assembler.emit_windows()) == 0

    # Fifth chunk completes the window.
    assembler.ingest_chunk(0, _make_chunk(40, human=False))
    windows = assembler.emit_windows()
    assert len(windows) == 1


def test_episode_id_change_discards_incomplete():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=1))
    assembler.ingest_chunk(0, _make_chunk(10, human=False, episode_id=2))
    assert len(assembler.emit_windows()) == 0
    assert assembler.pending_count == 0


def test_natural_episode_change_resets_anchor_stride():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(490, human=True, episode_id=1))

    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=2))
    assert assembler.pending_count == 1

    for step_start in range(10, 50, 10):
        assembler.ingest_chunk(0, _make_chunk(step_start, human=False, episode_id=2))

    windows = assembler.emit_windows()
    assert len(windows) == 1
    assert windows[0]["episode_id"] == 2
    assert windows[0]["episode_step_ids"][0].item() == 0


def test_step_id_gap_discards_incomplete():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=1))
    # Gap: expected step 10 but received step 20.
    assembler.ingest_chunk(0, _make_chunk(20, human=False, episode_id=1))
    assert len(assembler.emit_windows()) == 0
    assert assembler.pending_count == 0


def test_unexecuted_chunk_discards():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=1))
    rejected = _make_chunk(10, human=False, episode_id=1)
    rejected.executed_action_mask[5] = False
    assembler.ingest_chunk(0, rejected)
    assert len(assembler.emit_windows()) == 0
    assert assembler.pending_count == 0


def test_termination_chunk_discards():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=1))
    term_code = TERMINATION_REASON_CODES["operator_success"]
    assembler.ingest_chunk(
        0, _make_chunk(10, human=False, episode_id=1, termination_code=term_code)
    )
    assert len(assembler.emit_windows()) == 0
    assert assembler.pending_count == 0


def test_pending_survives_rollout_batch():
    assembler = ExpertAnchoredWindowAssembler()
    # Feed three chunks (one human + two model) in the first batch.
    assembler.ingest_chunk(0, _make_chunk(0, human=True))
    assembler.ingest_chunk(0, _make_chunk(10, human=False))
    assembler.ingest_chunk(0, _make_chunk(20, human=False))
    assert len(assembler.emit_windows()) == 0

    # Feed two more chunks in a subsequent batch.
    assembler.ingest_chunk(0, _make_chunk(30, human=False))
    assembler.ingest_chunk(0, _make_chunk(40, human=False))
    windows = assembler.emit_windows()
    assert len(windows) == 1


def test_model_version_change_discards_pending_anchor():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, model_version=1))

    assembler.ingest_chunk(0, _make_chunk(10, human=False, model_version=2))

    assert assembler.emit_windows() == []
    assert assembler.pending_count == 0


def test_window_metrics_report_emits_drops_reasons_and_pending():
    assembler = ExpertAnchoredWindowAssembler()
    assembler.ingest_chunk(0, _make_chunk(0, human=True, model_version=1))
    assert assembler.get_metrics()["pending_anchor"] == 1

    assembler.ingest_chunk(0, _make_chunk(10, human=False, model_version=2))
    metrics = assembler.get_metrics()
    assert metrics["window_dropped"] == 1
    assert metrics["window_drop_reason/model_version_change"] == 1
    assert metrics["pending_anchor"] == 0

    for step_start in range(0, 50, 10):
        assembler.ingest_chunk(
            0,
            _make_chunk(
                step_start,
                human=step_start == 0,
                episode_id=2,
                model_version=2,
            ),
        )

    metrics = assembler.get_metrics()
    assert metrics["window_emitted"] == 1
    assert metrics["window_dropped"] == 1
    assert metrics["pending_anchor"] == 0


def test_window_audit_fields():
    assembler = ExpertAnchoredWindowAssembler()
    # One human chunk + four model chunks = one 50-step window.
    assembler.ingest_chunk(0, _make_chunk(0, human=True, episode_id=42))
    for i in range(1, 5):
        assembler.ingest_chunk(0, _make_chunk(i * 10, human=False, episode_id=42))

    windows = assembler.emit_windows()
    assert len(windows) == 1
    w = windows[0]

    assert w["human_steps"] == 10
    assert w["model_steps"] == 40
    assert w["human_fraction"] == 0.2
    assert w["episode_id"] == 42

    step_ids = w["episode_step_ids"]
    assert step_ids.shape == (50,)
    assert step_ids[0].item() == 0
    assert step_ids[-1].item() == 49
    diffs = step_ids[1:].to(torch.long) - step_ids[:-1].to(torch.long)
    assert torch.equal(diffs, torch.ones(49, dtype=torch.long))

    assert w["action"].shape == (50, 8)
    assert w["human_action_mask"].shape == (50,)
    assert w["executed_action_mask"].shape == (50,)
    assert w["human_action_mask"][:10].all()
    assert not w["human_action_mask"][10:].any()
    assert w["executed_action_mask"].all()
    assert w["model_version"] is not None
    assert torch.equal(w["model_version"], torch.tensor(1))
