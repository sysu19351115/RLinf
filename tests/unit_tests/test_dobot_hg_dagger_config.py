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

"""Hydra contract tests for Dobot HG-DAgger deployment configs."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from rlinf.utils import omega_resolver  # noqa: F401

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
_REQUIRED_OVERRIDES = [
    "actor.model.model_path=/tmp/dobot_model",
    "actor.model.openpi_data.norm_stats_path=/tmp/dobot_norm_stats.json",
    "actor.optim.lr=1e-6",
]


def _compose(config_name: str, monkeypatch) -> DictConfig:
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=_REQUIRED_OVERRIDES)
    OmegaConf.resolve(cfg)
    return cfg


def _assert_algorithm_and_safety_contract(cfg: DictConfig) -> None:
    assert cfg.runner.only_eval is False
    assert cfg.runner.val_check_interval == -1
    assert cfg.algorithm.loss_type == "embodied_dagger"
    assert cfg.algorithm.update_epoch == 1
    assert cfg.algorithm.dagger.only_save_expert is True
    assert cfg.algorithm.replay_buffer.min_buffer_size == 4
    assert cfg.algorithm.replay_buffer.auto_save is True
    assert cfg.algorithm.replay_buffer.trajectory_format == "pt"

    assert cfg.env.train.total_num_envs == 1
    assert cfg.env.train.auto_reset is True
    assert cfg.env.train.ignore_terminations is False
    assert cfg.env.train.override_cfg.is_dummy is True
    assert cfg.env.train.override_cfg.action_mode == "cartesian"
    assert cfg.env.train.override_cfg.state_mode == "pose"
    assert cfg.env.train.override_cfg.manual_episode_control_only is True
    assert cfg.env.train.use_keyboard_intervention is True
    assert cfg.env.train.keyboard_intervention.episode_control_mode == "online"
    assert cfg.env.train.keyboard_intervention.wait_for_start_on_reset is True
    assert cfg.env.train.data_collection.enabled is False

    assert cfg.actor.model.model_type == "openpi"
    assert cfg.actor.model.action_dim == 8
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.model.openpi.config_name == "pi05_dobot_pose"
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path

    # Hybrid 50-step window contract
    d = cfg.algorithm.dagger
    assert d.execution_chunk_steps == 10
    assert d.training_window_steps == 50
    assert d.window_stride_steps == 10
    assert d.window_anchor == "full_human_chunk"
    assert d.min_human_steps_per_window == 10
    assert d.loss_scope == "human_only"
    assert list(d.required_forward_input_keys) == [
        "observation/prev_state",
        "observation/image",
    ]
    assert cfg.algorithm.replay_buffer.schema_version.endswith("_v2")
    assert str(cfg.algorithm.replay_buffer.auto_save_path).endswith(
        "logs/dobot_hg_dagger/replay_buffer_h50"
    )
    assert d.training_window_steps == cfg.actor.model.openpi.action_horizon
    assert d.execution_chunk_steps == cfg.actor.model.num_action_chunks
    assert d.window_stride_steps == d.execution_chunk_steps
    assert d.training_window_steps % d.execution_chunk_steps == 0


def test_single_node_config_is_safe_and_resolvable(monkeypatch):
    cfg = _compose("dobot_hg_dagger_openpi", monkeypatch)

    _assert_algorithm_and_safety_contract(cfg)
    assert cfg.cluster.num_nodes == 1
    for component in ("actor", "rollout", "env"):
        assert cfg.cluster.component_placement[component].node_group == "dobot"
    assert cfg.cluster.node_groups[0].node_ranks == 0
    assert cfg.cluster.node_groups[0].hardware.configs[0].node_rank == 0


def test_two_node_config_changes_only_deployment(monkeypatch):
    single = _compose("dobot_hg_dagger_openpi", monkeypatch)
    two_node = _compose("dobot_hg_dagger_openpi_2node", monkeypatch)

    _assert_algorithm_and_safety_contract(two_node)
    assert two_node.cluster.num_nodes == 2
    assert two_node.cluster.component_placement.actor.node_group == "inference"
    assert two_node.cluster.component_placement.rollout.node_group == "robot"
    assert two_node.cluster.component_placement.env.node_group == "robot"
    assert two_node.cluster.node_groups[1].hardware.configs[0].node_rank == 1

    for key in ("algorithm", "env", "rollout", "actor", "reward", "critic"):
        assert OmegaConf.to_container(
            two_node[key], resolve=True
        ) == OmegaConf.to_container(single[key], resolve=True)


def test_required_training_values_fail_closed(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    monkeypatch.delenv("DOBOT_HG_DAGGER_MODEL_PATH", raising=False)
    monkeypatch.delenv("DOBOT_HG_DAGGER_NORM_STATS_PATH", raising=False)
    monkeypatch.delenv("DOBOT_HG_DAGGER_LR", raising=False)
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_hg_dagger_openpi")

    with pytest.raises(InterpolationResolutionError):
        OmegaConf.resolve(cfg)
