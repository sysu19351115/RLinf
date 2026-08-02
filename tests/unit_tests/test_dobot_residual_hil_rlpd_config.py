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

"""Config contract tests for the residual HIL-RLPD algorithm."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from rlinf.algorithms.residual_hil_rlpd import validate_residual_hil_rlpd_config


def _rlpd_block():
    return OmegaConf.create(
        {
            "translation_data_limit_m": [0.1, 0.1, 0.1],
            "rotation_data_limit_deg": [30.0, 30.0, 30.0],
            "translation_policy_limit_m": [0.01, 0.01, 0.01],
            "rotation_policy_limit_deg": [5.0, 5.0, 5.0],
            "demo_ratio": 0.5,
            "utd_ratio": 1.0,
            "max_update_backlog": 200,
            "critic_only_updates": 500,
            "residual_scale_ramp_updates": 2000,
            "base_only_collect_steps": 100,
            "num_q_heads": 10,
            "num_q_sample": 2,
            "backup_entropy": False,
            "alpha_arm_init": 0.1,
            "alpha_gripper_init": 0.05,
            "min_demo_size": 1,
            "max_online_transitions": 20000,
            "max_demo_transitions": 5000,
            "max_online_bytes": 4e10,
            "max_demo_bytes": 1e10,
            "memory_high_watermark": 0.9,
            "policy_lag_warn_threshold": 50,
            "policy_lag_reject_threshold": 500,
            "gripper_enable_after_updates": 1500,
            "gripper_max_switches_per_chunk": 2,
            "gripper_min_hold_steps": 5,
            "gripper_debounce_chunks": 2,
            "safety_workspace_min_m": [-0.35, -0.55, 0.02],
            "safety_workspace_max_m": [0.60, 0.55, 0.75],
            "safety_max_translation_delta_m": 0.02,
            "safety_max_rotation_delta_deg": 10.0,
            "safety_hold_chunks": 3,
        }
    )


def _env(mode: str):
    return OmegaConf.create(
        {
            "total_num_envs": 1,
            "override_cfg": {
                "action_mode": "cartesian",
                "state_mode": "pose",
            },
            "keyboard_intervention": {
                "safe_model_handoff": True,
                "allow_motion_intervention": mode == "train",
            },
        }
    )


def _valid_cfg():
    return OmegaConf.create(
        {
            "run_id": "test-run",
            "dobot": {"ip": "192.168.5.2"},
            "env": {"train": _env("train"), "eval": _env("eval")},
            "algorithm": {
                "loss_type": "residual_hil_rlpd",
                "adv_type": "embodied_sac",
                "residual_hil_rlpd": _rlpd_block(),
            },
            "actor": {
                "global_batch_size": 64,
                "model": {
                    "num_action_chunks": 10,
                    "base_policy": {"trainable": False},
                },
            },
            "rollout": {
                "collect_transitions": True,
                "model": {"num_action_chunks": 10},
            },
        }
    )


def test_valid_config_passes():
    validate_residual_hil_rlpd_config(_valid_cfg())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda cfg: cfg.pop("dobot"), "only supports the Dobot"),
        (
            lambda cfg: cfg.env.train.override_cfg.__setitem__("action_mode", "joint"),
            "action_mode=cartesian",
        ),
        (
            lambda cfg: cfg.env.eval.override_cfg.__setitem__("state_mode", "joint"),
            "state_mode=pose",
        ),
        (
            lambda cfg: cfg.env.train.__setitem__("total_num_envs", 2),
            "total_num_envs == 1",
        ),
        (
            lambda cfg: cfg.algorithm.__setitem__("adv_type", "gae"),
            "adv_type=embodied_sac",
        ),
        (
            lambda cfg: cfg.actor.model.__setitem__("num_action_chunks", 4),
            "num_action_chunks == 10",
        ),
        (
            lambda cfg: cfg.rollout.__setitem__("collect_transitions", False),
            "collect_transitions=True",
        ),
        (
            lambda cfg: cfg.actor.__setitem__("global_batch_size", 63),
            "even actor.global_batch_size",
        ),
        (
            lambda cfg: cfg.actor.model.base_policy.__setitem__("trainable", True),
            "stay frozen",
        ),
        (
            lambda cfg: cfg.env.train.keyboard_intervention.__setitem__(
                "safe_model_handoff", False
            ),
            "safe_model_handoff=True",
        ),
        (
            lambda cfg: cfg.env.eval.keyboard_intervention.__setitem__(
                "allow_motion_intervention", True
            ),
            "allow_motion_intervention=False",
        ),
        (
            lambda cfg: cfg.algorithm.pop("residual_hil_rlpd"),
            "algorithm.residual_hil_rlpd block",
        ),
        (
            lambda cfg: cfg.algorithm.residual_hil_rlpd.__setitem__(
                "translation_policy_limit_m", [0.2, 0.2, 0.2]
            ),
            "translation_policy_limit_m must be <=",
        ),
        (
            lambda cfg: cfg.algorithm.residual_hil_rlpd.__setitem__("demo_ratio", 1.0),
            "demo_ratio must be in",
        ),
        (
            lambda cfg: cfg.algorithm.residual_hil_rlpd.__setitem__(
                "backup_entropy", True
            ),
            "backup_entropy=True is not yet supported",
        ),
    ],
)
def test_invalid_configs_are_rejected(mutate, match):
    cfg = _valid_cfg()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_residual_hil_rlpd_config(cfg)


def test_missing_rlpd_key_is_rejected():
    cfg = _valid_cfg()
    cfg.algorithm.residual_hil_rlpd.pop("utd_ratio")
    with pytest.raises(ValueError, match="utd_ratio"):
        validate_residual_hil_rlpd_config(cfg)
