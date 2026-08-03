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

"""Example config contract tests for residual HIL-RLPD."""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.algorithms.residual_hil_rlpd import validate_residual_hil_rlpd_config
from rlinf.config import normalize_runner_paths
from rlinf.utils import omega_resolver  # noqa: F401

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def test_residual_hil_rlpd_example_config_is_valid(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_async_residual_hil_rlpd_pi05")
    OmegaConf.resolve(cfg)
    normalize_runner_paths(cfg)

    validate_residual_hil_rlpd_config(cfg)

    assert cfg.algorithm.loss_type == "residual_hil_rlpd"
    assert cfg.actor.model.model_type == "residual_dobot_policy"
    assert cfg.actor.model.base_policy.trainable is False
    assert cfg.actor.global_batch_size % 2 == 0
    assert cfg.env.train.keyboard_intervention.safe_model_handoff is True
    assert cfg.env.eval.keyboard_intervention.allow_motion_intervention is False
    assert cfg.algorithm.residual_hil_rlpd.gripper_residual_enabled is False
    # Model/norm paths are hoisted to the top-level dobot block and referenced
    # by both actor and rollout model configs (single truth source).
    assert cfg.actor.model.model_path == cfg.dobot.model_path
    assert cfg.actor.model.base_policy.model_path == cfg.dobot.model_path
    assert cfg.actor.model.openpi_data.norm_stats_path == cfg.dobot.norm_stats_path
    assert cfg.rollout.model.model_path == cfg.dobot.model_path
    assert cfg.rollout.model.base_policy.model_path == cfg.dobot.model_path
    assert (
        cfg.rollout.model.openpi_data.norm_stats_path == cfg.dobot.norm_stats_path
    )
    assert os.path.isabs(cfg.runner.logger.log_path)


def test_workspace_check_can_be_opted_out_explicitly(monkeypatch):
    """``safety_workspace_check_enabled: False`` makes the calibration-specific
    workspace box optional; without the explicit opt-out the keys stay
    mandatory."""
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_async_residual_hil_rlpd_pi05")
    OmegaConf.resolve(cfg)
    normalize_runner_paths(cfg)
    rlpd = cfg.algorithm.residual_hil_rlpd

    # The example config opts out of the calibration-specific box check, so
    # the workspace keys may be absent.
    assert rlpd.safety_workspace_check_enabled is False
    validate_residual_hil_rlpd_config(cfg)

    # Re-enabling the check without the keys must fail loudly.
    with open_dict(rlpd):
        rlpd.safety_workspace_check_enabled = True
    try:
        validate_residual_hil_rlpd_config(cfg)
    except ValueError as exc:
        assert "safety_workspace_min_m" in str(exc)
    else:
        raise AssertionError("workspace box must be required when check is on")
