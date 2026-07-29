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

"""Tests for OpenPI PyTorch prediction and execution horizon separation."""

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch import _resolve_action_horizon


def test_explicit_action_horizon_is_independent_from_execution_chunk():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 10,
            "openpi": {"action_horizon": 50},
        }
    )

    assert _resolve_action_horizon(cfg) == 50
    assert cfg.num_action_chunks == 10


def test_action_horizon_defaults_to_execution_chunk_for_legacy_configs():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 10,
            "openpi": {},
        }
    )

    assert _resolve_action_horizon(cfg) == 10


def test_non_positive_action_horizon_is_rejected():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 10,
            "openpi": {"action_horizon": 0},
        }
    )

    with pytest.raises(ValueError, match="action_horizon must be positive"):
        _resolve_action_horizon(cfg)
