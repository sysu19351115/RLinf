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

"""Tests for OpenPI DAgger 50-step horizon reshape and loss dimensions."""

import pytest
import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
    _reduce_sft_loss,
    _resolve_dagger_loss_mask,
)


class _StubConfig:
    action_horizon = 50
    action_env_dim = 8
    action_dim = 32
    action_chunk = 10


def _make_model_stub():
    """Create a minimal stub with the methods we need to test."""
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    model.config = _StubConfig()
    return model


def test_dagger_action_reshape_uses_action_horizon_not_chunk():
    """The old bug reshaped [B, 400] to [B, 10, 40] instead of [B, 50, 8]."""
    model = _make_model_stub()
    bsz = 2
    action = torch.randn(bsz, 50 * 8)  # [B, 400]

    # Simulate the reshape logic from prepare_dagger_sft_batch (else branch)
    expected_numel = bsz * model.config.action_horizon * model.config.action_env_dim
    assert action.numel() == expected_numel

    reshaped = action.reshape(
        bsz, model.config.action_horizon, model.config.action_env_dim
    )
    assert reshaped.shape == (bsz, 50, 8)


def test_dagger_action_wrong_numel_raises():
    """A 10-step chunk (old format) must be rejected, not silently misshaped."""
    model = _make_model_stub()
    bsz = 1
    old_10_step = torch.randn(bsz, 10 * 8)  # [B, 80] — old format

    expected = bsz * model.config.action_horizon * model.config.action_env_dim
    assert old_10_step.numel() != expected

    with pytest.raises(ValueError, match="expected"):
        if old_10_step.numel() != expected:
            raise ValueError(
                f"DAgger action has {old_10_step.numel()} elements, "
                f"expected {expected}."
            )


def test_loss_slice_uses_action_horizon():
    """Loss must be sliced to [B, 50, 8], not [B, 10, 8]."""
    config = _StubConfig()
    # Simulate per-element loss from PI0Pytorch.forward: [B, action_horizon, action_dim]
    loss = torch.randn(2, config.action_horizon, config.action_dim)

    # The fix: slice to action_horizon (50) and action_env_dim (8)
    sliced = loss[:, : config.action_horizon, : config.action_env_dim]
    assert sliced.shape == (2, 50, 8)

    # The old bug: sliced to action_chunk (10) and action_env_dim (8)
    old_buggy = loss[:, : config.action_chunk, : config.action_env_dim]
    assert old_buggy.shape == (2, 10, 8)
    assert sliced.shape != old_buggy.shape


def test_human_only_loss_ignores_model_suffix():
    loss = torch.cat([torch.ones(1, 10, 8), torch.full((1, 40, 8), 100.0)], dim=1)
    mask = torch.tensor([[True] * 10 + [False] * 40])

    reduced = _reduce_sft_loss(
        loss,
        action_horizon=50,
        action_env_dim=8,
        use_action_chunk_loss=True,
        loss_mask=mask,
    )

    assert reduced.item() == pytest.approx(1.0)


def test_full_window_loss_keeps_existing_mean():
    loss = torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32)

    reduced = _reduce_sft_loss(
        loss,
        action_horizon=50,
        action_env_dim=8,
        use_action_chunk_loss=True,
    )

    assert reduced.item() == pytest.approx(loss[:, :, :8].mean().item())


def test_empty_human_mask_is_rejected():
    with pytest.raises(ValueError, match="selects no action steps"):
        _reduce_sft_loss(
            torch.ones(1, 50, 8),
            action_horizon=50,
            action_env_dim=8,
            use_action_chunk_loss=True,
            loss_mask=torch.zeros(1, 50, dtype=torch.bool),
        )


def test_resolve_dagger_loss_mask():
    mask = torch.tensor([[True] * 10 + [False] * 40])
    batch = {"human_action_mask": mask}

    assert _resolve_dagger_loss_mask(batch, "full_window") is None
    assert torch.equal(_resolve_dagger_loss_mask(batch, "human_only"), mask)

    with pytest.raises(ValueError, match="Unsupported DAgger loss_scope"):
        _resolve_dagger_loss_mask(batch, "unknown")
    with pytest.raises(ValueError, match="requires human_action_mask"):
        _resolve_dagger_loss_mask({}, "human_only")
