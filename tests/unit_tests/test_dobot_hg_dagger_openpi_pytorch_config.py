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

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils import omega_resolver  # noqa: F401

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def _compose(config_name, monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    monkeypatch.setenv("DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("DOBOT_HG_DAGGER_LR", "1e-6")
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def _assert_contract(cfg):
    assert cfg.actor.model.model_type == "openpi_pytorch"
    assert cfg.actor.model.openpi.task == "dagger"
    assert cfg.actor.model.precision == "fp32"
    assert cfg.rollout.model.precision == "bf16"
    assert cfg.actor.model.openpi.action_horizon == 50
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.model.action_dim == 8
    assert cfg.actor.model.openpi.model_action_dim == 32
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.add_value_head is False
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "bf16"
    assert cfg.algorithm.dagger.loss_scope == "human_only"
    assert cfg.algorithm.dagger.execution_chunk_steps == 10
    assert cfg.algorithm.dagger.training_window_steps == 50
    assert cfg.algorithm.dagger.min_human_steps_per_window == 10
    assert cfg.algorithm.reward_label_validity.enabled is False
    assert cfg.env.train.override_cfg.is_dummy is True
    assert cfg.env.train.override_cfg.manual_episode_control_only is True
    assert cfg.env.train.auto_reset is True
    assert cfg.env.train.ignore_terminations is False
    assert cfg.env.train.keyboard_intervention.allow_motion_intervention is True
    assert cfg.env.train.keyboard_intervention.episode_control_mode == "online"
    assert cfg.env.train.keyboard_intervention.safe_model_handoff is True
    assert cfg.env.train.keyboard_intervention.wait_for_start_on_reset is True
    assert cfg.env.train.terminal_padding.enabled is False


def test_single_node_openpi_pytorch_config(monkeypatch):
    cfg = _compose("dobot_hg_dagger_openpi_pytorch", monkeypatch)
    _assert_contract(cfg)
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor.node_group == "dobot"
    assert cfg.cluster.component_placement.rollout.node_group == "dobot"
    assert cfg.cluster.component_placement.env.node_group == "dobot"


def test_two_node_openpi_pytorch_config(monkeypatch):
    cfg = _compose("dobot_hg_dagger_openpi_pytorch_2node", monkeypatch)
    _assert_contract(cfg)
    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.component_placement.actor.node_group == "inference"
    assert cfg.cluster.component_placement.rollout.node_group == "robot"
    assert cfg.cluster.component_placement.env.node_group == "robot"
    assert cfg.cluster.node_groups[0].node_ranks == 0
    assert cfg.cluster.node_groups[1].node_ranks == 1
    assert cfg.cluster.node_groups[1].hardware.configs[0].node_rank == 1
    assert cfg.dobot.ip == "192.168.5.2"
    assert cfg.dobot.tool_index == 2
    assert cfg.dobot.initial_joint_pos == [
        -5.87,
        -0.799,
        -2.175,
        1.89,
        1.424,
        0.621,
        0.9,
    ]
    hardware_cfg = cfg.cluster.node_groups[1].hardware.configs[0]
    assert hardware_cfg.ip == cfg.dobot.ip
    assert hardware_cfg.tool_index == cfg.dobot.tool_index
    assert cfg.env.train.override_cfg.initial_joint_pos == cfg.dobot.initial_joint_pos
