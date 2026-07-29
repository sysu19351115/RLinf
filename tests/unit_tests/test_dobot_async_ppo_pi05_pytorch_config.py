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

"""Hydra contract tests for the Dobot OpenPI PyTorch async PPO config."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils import omega_resolver  # noqa: F401

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def test_dobot_openpi_pytorch_async_ppo_config_is_resolvable(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_async_ppo_pi05_pytorch")
    OmegaConf.resolve(cfg)

    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.component_placement.actor.node_group == "cloud"
    assert cfg.cluster.component_placement.rollout.node_group == "robot"
    assert cfg.cluster.component_placement.env.node_group == "robot"

    model = cfg.actor.model
    assert model.model_type == "openpi_pytorch"
    assert model.precision == "bf16"
    assert model.add_value_head is True
    assert model.num_action_chunks == 10
    assert model.action_dim == 8
    assert model.num_steps == 4
    assert model.openpi.task == "rl"
    assert model.openpi.config_name == "pi05_dobot_pose"
    assert model.openpi.action_horizon == 50
    assert model.openpi.action_chunk == 10
    assert model.openpi.action_env_dim == 8
    assert model.openpi.num_steps == 4
    assert model.openpi.model_action_dim == 32
    assert model.openpi.num_images_in_input == 2
    assert model.openpi.joint_logprob is False
    assert model.openpi.value_after_vlm is True
    assert model.openpi.value_vlm_mode == "mean_token"
    assert model.openpi.detach_critic_input is True
    assert model.openpi.train_expert_only is True
    assert model.openpi.noise_method == "flow_sde"
    assert model.openpi.noise_level == 0.4
    assert model.openpi.ignore_last is True

    assert cfg.rollout.model.model_path == model.model_path
    assert cfg.rollout.model.precision == model.precision
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.algorithm.loss_type == "decoupled_actor_critic"
