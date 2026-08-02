# Copyright 2025 The RLinf Authors.
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

import asyncio
import gc
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.rlt.transition import update_rlt_transitions
from rlinf.data.embodied_io_struct import (
    TERMINATION_REASON_CODES,
    UNKNOWN_TERMINATION_REASON_CODE,
    ChunkStepResult,
    EmbodiedLerobotRolloutResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, CommMapper, Worker
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
    validate_reward_label_validity_config,
)
from rlinf.workers.env.history_manager import HistoryManager


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.residual_hil_rlpd_mode = (
            str(self.cfg.algorithm.get("loss_type", "")) == "residual_hil_rlpd"
        )
        self._run_id = str(cfg.get("run_id", ""))
        self._pending_residual_transitions = []
        self._residual_pending_ctx = None
        self._residual_chunk_counter = 0
        # P2-4/P2-5 fail-closed state: gripper inhibit after human takeover /
        # rejection, and independent safety-barrier hold after a violation.
        self._residual_gripper_inhibit_remaining = 0
        self._residual_gripper_suppressed_mask: np.ndarray | None = None
        self._residual_safety_hold_remaining = 0
        self._residual_safety_violations = 0
        self._residual_safety_barrier = None
        self._residual_safety_violated_chunk = False
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        if self.residual_hil_rlpd_mode and self.stage_num != 1:
            raise ValueError(
                "residual_hil_rlpd supports a single pipeline stage "
                f"(single env), got pipeline_stage_num={self.stage_num}"
            )
        self._terminal_end_monotonic: list[float | None] = [None] * self.stage_num
        self._terminal_to_reset_latency_s: list[float] = []
        self.reward_label_validity_enabled = validate_reward_label_validity_config(cfg)
        self.enable_rlt = (
            OmegaConf.select(self.cfg, "algorithm.loss_type", default="") == "rlt_ac"
        )

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        self.train_terminal_padding_enabled = self._terminal_padding_enabled(
            train_env_cfg
        )
        self.eval_terminal_padding_enabled = self._terminal_padding_enabled(
            eval_env_cfg
        )

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
            else False
        )
        if self.enable_train:
            self.enable_online_lerobot = bool(
                OmegaConf.select(
                    self.cfg,
                    "algorithm.dagger.online_lerobot.enabled",
                    default=False,
                )
            )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
            if self.residual_hil_rlpd_mode and (
                self.train_num_envs_per_stage != 1 or self.train_batch_size != 1
            ):
                raise ValueError(
                    "residual_hil_rlpd supports a single env per stage "
                    f"(single env), got train_num_envs_per_stage="
                    f"{self.train_num_envs_per_stage}, train_batch_size="
                    f"{self.train_batch_size}"
                )
            self._validate_terminal_padding_contract(
                mode="train",
                env_cfg=train_env_cfg,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        else:
            self.enable_online_lerobot = False
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
            self._validate_terminal_padding_contract(
                mode="eval",
                env_cfg=eval_env_cfg,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.actor_split_num = (
            1 if not self.enable_train else self.get_actor_split_num()
        )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()

        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if self.env_decoupled_mode and (
            self.train_terminal_padding_enabled or self.eval_terminal_padding_enabled
        ):
            raise ValueError(
                "Dobot terminal padding does not support runner.enable_decoupled_mode."
            )

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    @staticmethod
    def _terminal_padding_enabled(env_cfg: Any) -> bool:
        if env_cfg is None:
            return False
        padding_cfg = env_cfg.get("terminal_padding", {})
        return bool(padding_cfg.get("enabled", False))

    @classmethod
    def _validate_terminal_padding_contract(
        cls,
        *,
        mode: str,
        env_cfg: Any,
        num_envs_per_stage: int,
    ) -> bool:
        """Fail closed: terminal padding currently supports one Dobot per stage."""
        if not cls._terminal_padding_enabled(env_cfg):
            return False

        if num_envs_per_stage != 1:
            raise ValueError(
                f"env.{mode}.terminal_padding.enabled requires exactly one "
                f"environment per stage; got {num_envs_per_stage}."
            )
        if bool(env_cfg.get("auto_reset", False)):
            raise ValueError(
                f"env.{mode}.terminal_padding.enabled requires auto_reset=False."
            )

        init_params = env_cfg.get("init_params", {})
        env_id = str(init_params.get("id", ""))
        if not env_id.startswith("Dobot"):
            raise ValueError(
                f"env.{mode}.terminal_padding.enabled is restricted to Dobot "
                f"real-world environments; got init_params.id={env_id!r}."
            )
        keyboard_cfg = env_cfg.get("keyboard_intervention", {})
        if (
            not bool(env_cfg.get("use_keyboard_intervention", False))
            or keyboard_cfg.get("episode_control_mode") != "online_chunk_boundary"
        ):
            raise ValueError(
                f"env.{mode}.terminal_padding.enabled requires Dobot keyboard "
                "episode_control_mode='online_chunk_boundary'."
            )
        return True

    @staticmethod
    def _terminal_padding_triggered(
        *,
        enabled: bool,
        dones: torch.Tensor | None,
    ) -> bool:
        if not enabled or dones is None:
            return False
        if dones.ndim == 0 or dones.shape[0] != 1:
            raise RuntimeError(
                "terminal padding received a non-singleton environment batch after "
                f"configuration validation: shape={tuple(dones.shape)}."
            )
        return bool(dones[0].any())

    def _prepare_rollout_results(self, rollout_results: list | None = None) -> list:
        if self.enable_online_lerobot and rollout_results is not None:
            for stage_rollout in rollout_results:
                stage_rollout.rewards.clear()
            return rollout_results

        collect_only_success = bool(
            OmegaConf.select(
                self.cfg,
                "algorithm.dagger.online_lerobot.only_success",
                default=False,
            )
        )
        max_episode_length = self.cfg.env.train.max_episode_steps
        if self.enable_online_lerobot:
            return [
                EmbodiedLerobotRolloutResult(
                    max_episode_length=max_episode_length,
                    num_envs=self.train_num_envs_per_stage,
                    only_success=collect_only_success,
                    num_action_chunks=self.model_cfg.num_action_chunks,
                    action_dim=self.model_cfg.action_dim,
                )
                for _ in range(self.stage_num)
            ]
        return [
            EmbodiedRolloutResult(max_episode_length=max_episode_length)
            for _ in range(self.stage_num)
        ]

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        # Skip when there is only a single env worker — a single-rank barrier
        # still creates a Gloo distributed process group, whose server socket
        # can be discovered by cross-node weight-sync groups, causing c10d timeouts.
        if self._world_size > 1:
            self.broadcast(
                True,
                groups=[(self._group_name, list(range(self._world_size)))],
            )

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.eval_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.eval_env_list
                ), "eval envs must have an offload method to enable offload!"

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

        self._init_env()

    def update_env_cfg(self):
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.cfg.env.train.get(
                    "enable_init_offload", True
                ):
                    get_env_attr(self.env_list[i], "offload")()
            if self.enable_eval:
                if self.eval_enable_offload:
                    get_env_attr(self.eval_env_list[i], "offload")()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
        gripper_bypass_mask: np.ndarray | None = None,
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
            env_cfg=self.cfg.env.train,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(
                chunk_actions,
                gripper_bypass_mask=gripper_bypass_mask,
            )
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()
        control_infos = (
            infos["final_info"]
            if isinstance(infos, dict) and "final_info" in infos
            else infos
        )
        if isinstance(control_infos, dict):
            env_info.update(self._extract_operator_metrics(control_infos))

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if (
            self.cfg.env.train.auto_reset
            and chunk_dones.any()
            and "final_info" in infos
        ):
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
        }
        return env_output, env_info, chunk_step_payload

    def _finalize_residual_transition(
        self,
        rollout_result: Any,
        env_output: EnvOutput,
        stage_id: int,
        *,
        chunk_start_obs: dict[str, Any],
        next_nominal: np.ndarray | None = None,
        safety_violated: bool = False,
    ):
        """Build one residual chunk transition from real executed feedback."""
        from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
        from rlinf.algorithms.residual_hil_rlpd.finalizer import (
            ChunkEnvFeedback,
            finalize_chunk_transition,
        )
        from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
            compute_base_fingerprint,
        )
        from rlinf.algorithms.residual_hil_rlpd.rollout import (
            RolloutChunkAudit,
        )
        from rlinf.algorithms.residual_hil_rlpd.transition import (
            SOURCE_ONLINE,
            SOURCE_ONLINE_INTERVENTION,
        )

        feedback_data = (env_output.env_infos or {}).get("residual_feedback")
        if feedback_data is None:
            return None
        if safety_violated:
            # P2-5: the chunk was replaced with nominal before execution; the
            # violating residual must never enter replay.
            self._logger.warning(
                "[ResidualHIL] rejecting transition for safety-violated chunk "
                f"{self._residual_chunk_counter}"
            )
            return None
        audit_data = getattr(rollout_result, "audit_info", None)
        if not audit_data:
            raise RuntimeError("residual_hil_rlpd rollout result is missing audit_info")
        audit_dict = audit_data[-1] if isinstance(audit_data, list) else audit_data
        sampled_gripper_mode = np.asarray(
            audit_dict["sampled_gripper_mode"], dtype=np.int64
        )
        # P2-4: if the env worker suppressed the model gripper override this
        # chunk (human takeover / safety hold), the *training label* must match
        # the executed command (KEEP), not the raw model sample.
        suppressed_mask = getattr(self, "_residual_gripper_suppressed_mask", None)
        if (
            suppressed_mask is not None
            and suppressed_mask.size == sampled_gripper_mode.size
        ):
            sampled_gripper_mode = sampled_gripper_mode.copy()
            sampled_gripper_mode[suppressed_mask] = 0  # KEEP_NOMINAL
        audit = RolloutChunkAudit(
            nominal_actions=np.asarray(audit_dict["nominal_actions"], dtype=np.float32),
            sampled_arm_residual=np.asarray(
                audit_dict["sampled_arm_residual"], dtype=np.float32
            ),
            sampled_gripper_mode=sampled_gripper_mode,
            commanded_actions=np.asarray(
                audit_dict["commanded_actions"], dtype=np.float32
            ),
            gripper_bypass_mask=np.asarray(
                audit_dict["gripper_bypass_mask"], dtype=bool
            ),
            policy_version=int(audit_dict.get("policy_version", 0)),
        )
        feedback = ChunkEnvFeedback(
            executed_actions=np.asarray(
                feedback_data["executed_actions"], dtype=np.float32
            ),
            executed_action_mask=np.asarray(
                feedback_data["executed_action_mask"], dtype=bool
            ),
            gripper_bypass_mask=np.asarray(
                feedback_data["gripper_bypass_mask"], dtype=bool
            ),
            human_intervention_mask=np.asarray(
                feedback_data["human_intervention_mask"], dtype=bool
            ),
            handoff_hold_mask=np.asarray(
                feedback_data["handoff_hold_mask"], dtype=bool
            ),
            rewards=np.asarray(feedback_data["rewards"], dtype=np.float32),
            terminations=np.asarray(feedback_data["terminations"], dtype=bool),
            truncations=np.asarray(feedback_data["truncations"], dtype=bool),
            reward_label_valid=bool(
                np.all(feedback_data.get("reward_label_valid", np.array([False])))
            ),
        )
        rlpd_cfg = self.cfg.algorithm.residual_hil_rlpd
        codec = ResidualCodec(
            translation_scale_m=tuple(rlpd_cfg.translation_data_limit_m),
            rotation_scale_deg=tuple(rlpd_cfg.rotation_data_limit_deg),
        )
        source = (
            SOURCE_ONLINE_INTERVENTION
            if feedback.human_intervention_mask.any()
            else SOURCE_ONLINE
        )
        transition, valid, _ = finalize_chunk_transition(
            curr_obs=chunk_start_obs,
            next_obs=env_output.final_obs or env_output.obs,
            audit=audit,
            feedback=feedback,
            codec=codec,
            source=source,
            base_fingerprint=compute_base_fingerprint(
                str(self.cfg.actor.model.base_policy.get("model_path", "")),
                norm_stats_path=str(
                    self.cfg.actor.model.get("openpi_data", {}).get(
                        "norm_stats_path", None
                    )
                ),
                codec=codec,
            ),
            episode_id=int(
                np.asarray(
                    (env_output.env_infos or {}).get("episode_id", [stage_id])
                ).reshape(-1)[0]
            ),
            chunk_id=self._residual_chunk_counter,
            gamma=float(self.cfg.algorithm.gamma),
            next_nominal_actions=next_nominal,
        )
        return transition if valid else None

    def _residual_safety_check(
        self,
        rollout_result: Any,
        nominal_actions: np.ndarray | None,
        bypass_mask: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """P2-5 pre-execution safety barrier for one chunk.

        Checks the *commanded* chunk (nominal + residual) against the
        independent safety limits before ``env_interact_step``.  On violation:
        the chunk is replaced with nominal in-place (never executed as-is), a
        configurable hold is entered, and the transition is marked invalid at
        finalization.

        Returns:
            ``(bypass_mask, suppressed_mask)`` where ``suppressed_mask`` is
            non-None when the chunk's gripper label must be forced to KEEP.
        """
        if nominal_actions is None or nominal_actions.size == 0:
            return bypass_mask, None
        from rlinf.algorithms.residual_hil_rlpd.safety import (
            VIOLATION_WORKSPACE,
            ResidualSafetyBarrier,
            SafetyLimits,
        )

        if self._residual_safety_barrier is None:
            self._residual_safety_barrier = ResidualSafetyBarrier(
                SafetyLimits.from_config(self.cfg.algorithm.residual_hil_rlpd)
            )
        commanded_np = rollout_result.actions.detach().cpu().float().numpy()
        if commanded_np.ndim == 3:
            commanded_np = commanded_np[0]
        violations = self._residual_safety_barrier.check_chunk(
            nominal_actions, commanded_np
        )
        if not violations:
            return bypass_mask, None
        self._residual_safety_violations += 1
        self._residual_safety_violated_chunk = True
        self._residual_safety_hold_remaining = int(
            self.cfg.algorithm.residual_hil_rlpd.get("safety_hold_chunks", 3)
        )
        rollout_result.actions[..., :7] = torch.as_tensor(
            nominal_actions[..., :7],
            device=rollout_result.actions.device,
        )
        rollout_result.actions[..., 7] = torch.as_tensor(
            nominal_actions[..., 7],
            device=rollout_result.actions.device,
        )
        bypass_mask = np.zeros_like(bypass_mask)
        suppressed_mask = np.ones(nominal_actions.shape[0], dtype=bool)
        # Log the actual violating coordinates (and the nominal chunk's xyz
        # envelope) so the operator can calibrate safety_workspace_min/max_m:
        # a repeated workspace trip with nominal itself outside the box means
        # the limits, not the residual, are wrong for this robot/calibration.
        violation_details = "; ".join(
            f"{v.code}@{v.step}: {v.detail}" for v in violations[:3]
        )
        if any(v.code == VIOLATION_WORKSPACE for v in violations):
            nominal_xyz = np.asarray(nominal_actions[..., :3], dtype=np.float64)
            violation_details += (
                " | nominal xyz range "
                f"[{nominal_xyz.min(axis=0).tolist()} .. "
                f"{nominal_xyz.max(axis=0).tolist()}]"
            )
        self._logger.warning(
            "[ResidualHIL] pre-execution safety violations on chunk "
            f"{self._residual_chunk_counter}: {violation_details}; chunk "
            f"replaced with nominal and "
            f"{self._residual_safety_hold_remaining}-chunk hold entered"
        )
        return bypass_mask, suppressed_mask

    def _residual_observe_chunk(
        self,
        rollout_result: Any,
        env_output: EnvOutput,
        chunk_start_obs: dict[str, Any],
        stage_id: int,
    ) -> None:
        """Defer building chunk k until chunk k+1's nominal is known."""
        feedback_data = (env_output.env_infos or {}).get("residual_feedback")
        if feedback_data is None:
            return
        # P2-4: human takeover, handoff hold or controller rejection disables
        # model gripper override for the next ``gripper_debounce_chunks``.
        human = np.asarray(feedback_data.get("human_intervention_mask", []), dtype=bool)
        handoff = np.asarray(feedback_data.get("handoff_hold_mask", []), dtype=bool)
        accepted = np.asarray(feedback_data.get("executed_action_mask", []), dtype=bool)
        rejected = accepted.size > 0 and not bool(accepted.all())
        if bool(human.any()) or bool(handoff.any()) or rejected:
            debounce = int(
                self.cfg.algorithm.residual_hil_rlpd.get("gripper_debounce_chunks", 2)
            )
            self._residual_gripper_inhibit_remaining = max(
                self._residual_gripper_inhibit_remaining, debounce
            )
        safety_violated = bool(self._residual_safety_violated_chunk)
        self._residual_safety_violated_chunk = False
        ctx = (
            rollout_result,
            env_output,
            chunk_start_obs,
            stage_id,
            safety_violated,
        )
        if self._residual_pending_ctx is not None:
            previous = self._residual_pending_ctx
            self._residual_pending_ctx = ctx
            next_nominal = self._audit_nominal_actions(rollout_result)
            self._finalize_and_enqueue(previous, next_nominal=next_nominal)
        else:
            self._residual_pending_ctx = ctx
        # A terminated chunk ends the episode: hemisphere tracking restarts.
        if self._residual_safety_barrier is not None and (
            bool(np.asarray(feedback_data.get("terminations", [])).any())
            or bool(np.asarray(feedback_data.get("truncations", [])).any())
        ):
            self._residual_safety_barrier.reset_tracking()

    @staticmethod
    def _audit_nominal_actions(rollout_result: Any) -> np.ndarray | None:
        audit_data = getattr(rollout_result, "audit_info", None)
        if not audit_data:
            return None
        audit_dict = audit_data[-1] if isinstance(audit_data, list) else audit_data
        return np.asarray(audit_dict.get("nominal_actions", []), dtype=np.float32)

    def _finalize_and_enqueue(
        self,
        ctx,
        *,
        next_nominal: np.ndarray | None,
    ) -> None:
        rollout_result, env_output, chunk_start_obs, stage_id, safety_violated = ctx
        transition = self._finalize_residual_transition(
            rollout_result,
            env_output,
            stage_id,
            chunk_start_obs=chunk_start_obs,
            next_nominal=next_nominal,
            safety_violated=safety_violated,
        )
        self._residual_chunk_counter += 1
        if transition is not None:
            self._pending_residual_transitions.append(transition)

    def _flush_residual_pending(self) -> None:
        if self._residual_pending_ctx is not None:
            ctx = self._residual_pending_ctx
            self._residual_pending_ctx = None
            self._finalize_and_enqueue(ctx, next_nominal=None)

    @staticmethod
    def _extract_operator_metrics(infos: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert operator-control diagnostics into numeric logger metrics."""
        metrics: dict[str, torch.Tensor] = {}
        reasons = infos.get("termination_reason")
        if reasons is not None:
            reason_values = np.asarray(reasons).reshape(-1)
            for reason in (
                "operator_success",
                "operator_abort",
                "operator_failure",
                "operator_quit",
                "episode_timeout",
                "keyboard_disconnected",
                "keyboard_listener_error",
                "controller_rejection",
                "unsafe_model_handoff",
            ):
                reason_mask = reason_values == reason
                if reason_mask.any():
                    metrics[f"episode_end/{reason}"] = torch.as_tensor(
                        reason_mask, dtype=torch.float32
                    )
        if "reward_label_valid" in infos:
            metrics["rollout/reward_label_valid"] = torch.as_tensor(
                infos["reward_label_valid"], dtype=torch.float32
            ).reshape(-1)

        if "skipped_action_steps" in infos:
            metrics["episode_end/skipped_action_steps"] = torch.as_tensor(
                infos["skipped_action_steps"], dtype=torch.float32
            ).reshape(-1)
        if "executed_action_mask" in infos:
            executed_mask = torch.as_tensor(
                infos["executed_action_mask"], dtype=torch.float32
            )
            if executed_mask.ndim == 1:
                executed_mask = executed_mask.unsqueeze(0)
            metrics["episode_end/executed_action_fraction"] = executed_mask.mean(dim=-1)
        if "handoff_hold_mask" in infos:
            handoff_hold_mask = torch.as_tensor(
                infos["handoff_hold_mask"], dtype=torch.float32
            )
            if handoff_hold_mask.ndim == 1:
                handoff_hold_mask = handoff_hold_mask.unsqueeze(0)
            metrics["control/handoff_hold_fraction"] = handoff_hold_mask.mean(dim=-1)
        return metrics

    @staticmethod
    def _extract_trajectory_audit_info(
        infos: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        """Extract replay audit tensors without adding them to model inputs."""
        if not isinstance(infos, dict):
            return {}
        control_infos = infos["final_info"] if "final_info" in infos else infos
        required_keys = {
            "executed_action_mask",
            "episode_id",
            "episode_step_ids",
        }
        if not required_keys.issubset(control_infos):
            return {}

        executed_action_mask = torch.as_tensor(
            control_infos["executed_action_mask"], dtype=torch.bool
        ).cpu()
        if executed_action_mask.dim() == 1:
            executed_action_mask = executed_action_mask.unsqueeze(0)
        episode_step_ids = torch.as_tensor(
            control_infos["episode_step_ids"], dtype=torch.int64
        ).cpu()
        if episode_step_ids.dim() == 1:
            episode_step_ids = episode_step_ids.unsqueeze(0)
        episode_id = torch.as_tensor(
            control_infos["episode_id"], dtype=torch.int64
        ).reshape(-1)
        reward_label_valid = torch.as_tensor(
            control_infos.get(
                "reward_label_valid",
                torch.ones(executed_action_mask.shape[0], dtype=torch.bool),
            ),
            dtype=torch.bool,
        ).reshape(-1)
        if reward_label_valid.shape != (executed_action_mask.shape[0],):
            raise ValueError(
                "reward_label_valid must have one value per environment, got "
                f"{tuple(reward_label_valid.shape)}."
            )
        handoff_hold_mask = torch.as_tensor(
            control_infos.get(
                "handoff_hold_mask",
                torch.zeros_like(executed_action_mask),
            ),
            dtype=torch.bool,
        ).cpu()
        if handoff_hold_mask.dim() == 1:
            handoff_hold_mask = handoff_hold_mask.unsqueeze(0)
        if handoff_hold_mask.shape != executed_action_mask.shape:
            raise ValueError(
                "handoff_hold_mask must match executed_action_mask, got "
                f"{tuple(handoff_hold_mask.shape)} and "
                f"{tuple(executed_action_mask.shape)}."
            )

        reason_values = np.asarray(
            control_infos.get("termination_reason", ["none"])
        ).reshape(-1)
        if reason_values.size == 1 and executed_action_mask.shape[0] > 1:
            reason_values = np.repeat(reason_values, executed_action_mask.shape[0])
        reason_codes = torch.as_tensor(
            [
                TERMINATION_REASON_CODES.get(
                    str(reason), UNKNOWN_TERMINATION_REASON_CODE
                )
                for reason in reason_values
            ],
            dtype=torch.int64,
        )
        return {
            "executed_action_mask": executed_action_mask.contiguous(),
            "termination_reason_code": reason_codes.contiguous(),
            "episode_id": episode_id.contiguous(),
            "episode_step_ids": episode_step_ids.contiguous(),
            "handoff_hold_mask": handoff_hold_mask.contiguous(),
            "reward_label_valid": reward_label_valid.contiguous(),
        }

    def _update_last_rollout_audit(self, stage_id: int, env_output: EnvOutput) -> None:
        audit_info = self._extract_trajectory_audit_info(env_output.env_infos)
        if audit_info:
            self.rollout_results[stage_id].update_last_audit_info(audit_info)

    def _apply_last_action_overrides(
        self, stage_id: int, env_output: EnvOutput
    ) -> None:
        """Persist human and handoff-hold actions without relabeling holds."""
        if env_output.intervene_actions is None:
            return
        human_flags = env_output.intervene_flags
        if human_flags is None:
            return

        override_flags = human_flags
        infos = env_output.env_infos
        control_infos = (
            infos["final_info"]
            if isinstance(infos, dict) and "final_info" in infos
            else infos
        )
        if isinstance(control_infos, dict) and "handoff_hold_mask" in control_infos:
            handoff_hold_mask = torch.as_tensor(
                control_infos["handoff_hold_mask"],
                dtype=torch.bool,
                device=human_flags.device,
            )
            if handoff_hold_mask.dim() == 1:
                handoff_hold_mask = handoff_hold_mask.unsqueeze(0)
            if handoff_hold_mask.shape != human_flags.shape:
                raise ValueError(
                    "handoff_hold_mask must match intervene_flags, got "
                    f"{tuple(handoff_hold_mask.shape)} and "
                    f"{tuple(human_flags.shape)}."
                )
            override_flags = torch.logical_or(human_flags, handoff_hold_mask)

        self.rollout_results[stage_id].update_last_actions(
            env_output.intervene_actions,
            override_flags,
        )
        if not torch.equal(override_flags, human_flags):
            self.rollout_results[stage_id].mark_last_step_with_intervene_flags(
                human_flags
            )

    def env_evaluate_step(
        self,
        raw_actions: torch.Tensor,
        stage_id: int,
        gripper_bypass_mask: np.ndarray | None = None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(
                chunk_actions,
                gripper_bypass_mask=gripper_bypass_mask,
            )
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            current_terminations = chunk_terminations.any(dim=1)
            current_truncations = chunk_truncations.any(dim=1)
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
                operator_metrics = self._extract_operator_metrics(final_info)
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()
                operator_metrics = self._extract_operator_metrics(infos)
            else:
                operator_metrics = {}

            done_mask_cpu = newly_done.cpu()
            for key, value in operator_metrics.items():
                if value.ndim > 0 and value.shape[0] == done_mask_cpu.shape[0]:
                    value = value[done_mask_cpu]
                env_info[key] = value.cpu()

            # Standalone real-world evaluation is autonomous by contract.
            # Keep an explicit metric instead of asking downstream users to
            # infer it from the absence of intervention.
            if "success_no_intervened" in env_info:
                env_info["autonomous_success"] = env_info["success_no_intervened"].to(
                    dtype=torch.float32
                )
            if "episode_len" in env_info:
                step_frequency = float(
                    self.cfg.env.eval.override_cfg.get("step_frequency", 1.0)
                )
                if step_frequency <= 0:
                    raise ValueError(
                        "env.eval.override_cfg.step_frequency must be positive "
                        "to report episode_duration_s."
                    )
                env_info["episode_duration_s"] = (
                    env_info["episode_len"].to(dtype=torch.float32) / step_frequency
                )

            control_infos = infos.get("final_info", infos)
            has_reason = (
                isinstance(control_infos, dict)
                and control_infos.get("termination_reason") is not None
            )
            if not has_reason:
                # Gym time limits end through truncation. Other unlabeled
                # terminations are reported separately instead of being
                # silently mixed with operator-labelled outcomes.
                timeout_mask = newly_done & current_truncations
                environment_mask = newly_done & current_terminations
                if timeout_mask.any():
                    env_info["episode_end/timeout"] = timeout_mask[newly_done].to(
                        dtype=torch.float32
                    )
                if environment_mask.any():
                    env_info["episode_end/environment_termination"] = environment_mask[
                        newly_done
                    ].to(dtype=torch.float32)

        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            env_infos=infos if isinstance(infos, dict) else None,
            rlt_switch_flags=rlt_switch_flags,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer batch dim for routed shards; supports RolloutResult and plain tensor payloads.

        When the channel carries a non-``RolloutResult`` shard (e.g. reward tensor or eval
        actions) into a rollout recv, avoid assuming dataclass fields and delegate or use
        the leading dimension of dense arrays.
        """

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if isinstance(data, RolloutResult):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                    self.env_list[i].wait_for_video_saves()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                    self.eval_env_list[i].wait_for_video_saves()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
            reward_input.update({"dones": dones})

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.rollout_results[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(reward_assign_length, rollout_rewards_length)
            for reward_assign_length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                if self.enable_online_lerobot:
                    rollout_results = getattr(self, "rollout_results", None)
                    if rollout_results is not None:
                        rollout_results[stage_id].reset_episode_buffers()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    env_infos=infos if isinstance(infos, dict) else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _build_rollout_input_data(
        self,
        env_batch: dict[str, Any],
        *,
        rollout_padding: bool | None = None,
    ) -> dict[str, Any]:
        data = {
            "obs": env_batch["obs"],
            "final_obs": env_batch["final_obs"],
        }
        if rollout_padding is not None:
            states = env_batch["obs"].get("states")
            if isinstance(states, torch.Tensor):
                batch_size = states.shape[0]
            else:
                task_descriptions = env_batch["obs"].get("task_descriptions")
                if task_descriptions is None:
                    raise ValueError("Cannot infer rollout-padding batch size.")
                batch_size = len(task_descriptions)
            data["rollout_padding"] = torch.full(
                (batch_size,), rollout_padding, dtype=torch.bool
            )
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
            data["intervene_flags"] = env_batch.get("intervene_flags", None)
        return data

    def _make_terminal_padding_env_output(
        self,
        source: EnvOutput,
    ) -> EnvOutput:
        """Build one shape-compatible chunk without stepping the environment."""
        batch_size = (
            source.dones.shape[0]
            if source.dones is not None
            else self.train_num_envs_per_stage
        )
        chunk_size = self.model_cfg.num_action_chunks
        zero_done = torch.zeros((batch_size, chunk_size), dtype=torch.bool)
        reward_dtype = (
            source.rewards.dtype if source.rewards is not None else torch.float32
        )
        zero_reward = torch.zeros((batch_size, chunk_size), dtype=reward_dtype)

        episode_id = torch.full((batch_size,), -1, dtype=torch.int64)
        source_infos = source.env_infos
        if isinstance(source_infos, dict):
            control_infos = source_infos.get("final_info", source_infos)
            if isinstance(control_infos, dict) and "episode_id" in control_infos:
                episode_id = torch.as_tensor(
                    control_infos["episode_id"], dtype=torch.int64
                ).reshape(-1)

        padding_infos = {
            "executed_action_mask": torch.zeros(
                (batch_size, chunk_size), dtype=torch.bool
            ),
            "action_command_accepted_mask": torch.zeros(
                (batch_size, chunk_size), dtype=torch.bool
            ),
            "episode_step_ids": torch.full(
                (batch_size, chunk_size), -1, dtype=torch.int64
            ),
            "episode_id": episode_id,
            "handoff_hold_mask": torch.zeros(
                (batch_size, chunk_size), dtype=torch.bool
            ),
            "termination_reason": np.full(batch_size, "none", dtype=object),
            "reward_label_valid": torch.ones(batch_size, dtype=torch.bool),
            "skipped_action_steps": chunk_size,
            "rollout_padding": True,
        }
        return EnvOutput(
            obs=source.obs,
            final_obs=None,
            rewards=zero_reward,
            dones=zero_done,
            terminations=zero_done.clone(),
            truncations=zero_done.clone(),
            env_infos=padding_infos,
            intervene_actions=None,
            intervene_flags=None,
            rlt_switch_flags=None,
        )

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_to(
                group_name=self.cfg.rollout.group_name,
                channel=rollout_channel,
                data=self._build_rollout_input_data(env_batch),
                mode="train",
                tag="rollout_results",
                route_key=stage_id if not self.env_decoupled_mode else None,
                decoupled_mode=self.env_decoupled_mode,
            )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        reset_started_at = time.monotonic()
        for stage_id, terminal_at in enumerate(self._terminal_end_monotonic):
            if terminal_at is None:
                continue
            self._terminal_to_reset_latency_s.append(reset_started_at - terminal_at)
            self._terminal_end_monotonic[stage_id] = None
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        rollout_result.clear()
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    @Worker.timer("env/send_lerobot_episodes")
    async def send_lerobot_episodes(
        self, episodes: list[list[dict]], channel: Channel
    ) -> None:
        if not episodes:
            return
        if self.actor_split_num <= 1:
            chunks = [episodes]
        else:
            chunks = split_list(
                episodes,
                self.actor_split_num,
                enforce_divisible_batch=False,
            )
        for chunk in chunks:
            if not chunk:
                continue
            channel.put(chunk, async_op=True)

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results = self._prepare_rollout_results(
            getattr(self, "rollout_results", None)
        )
        env_metrics = defaultdict(list)
        rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num
        valid_chunks = 0
        padded_chunks = 0
        train_terminal_padding_enabled = getattr(
            self,
            "train_terminal_padding_enabled",
            False,
        )
        terminal_to_reset_latency_s = getattr(
            self,
            "_terminal_to_reset_latency_s",
            [],
        )

        for epoch in range(self.rollout_epoch):
            terminal_padding_active = [False] * self.stage_num
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)
            if terminal_to_reset_latency_s:
                env_metrics["rollout/terminal_to_reset_latency_s"].append(
                    torch.tensor(terminal_to_reset_latency_s, dtype=torch.float32)
                )
                terminal_to_reset_latency_s.clear()

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    if train_terminal_padding_enabled:
                        if terminal_padding_active[stage_id]:
                            padded_chunks += self.train_num_envs_per_stage
                        else:
                            valid_chunks += self.train_num_envs_per_stage
                    curr_obs = env_output.obs
                    self._update_last_rollout_audit(stage_id, env_output)
                    self._apply_last_action_overrides(stage_id, env_output)

                    reward_model_output = None
                    if (
                        reward_channel is not None
                        and chunk_step_idx != 0
                        and not terminal_padding_active[stage_id]
                    ):
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="train_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.train_batch_size,
                        merge_fn=RolloutResult.merge_rollout_results,
                        infer_batch_size_fn=self._infer_rollout_batch_size,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=(
                            rollout_result.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            rollout_result.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )

                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if rollout_result.intervene_flags is not None:
                        self.rollout_results[
                            stage_id
                        ].mark_last_step_with_intervene_flags(
                            rollout_result.intervene_flags
                        )
                    if (
                        self.enable_rlt
                        and self.collect_transitions
                        and not terminal_padding_active[stage_id]
                    ):
                        update_rlt_transitions(
                            stage_id,
                            rlt_pending_obs,
                            self.rollout_results,
                            rollout_result,
                            cache_current=True,
                        )

                    if terminal_padding_active[stage_id]:
                        env_output = self._make_terminal_padding_env_output(env_output)
                        env_info = {}
                        terminal_ended_this_step = False
                    else:
                        terminal_ended_this_step = False
                        chunk_start_obs = env_output.obs
                        bypass_mask = None
                        suppressed_mask = None
                        nominal_actions = None
                        if getattr(self, "residual_hil_rlpd_mode", False):
                            audit_data = getattr(rollout_result, "audit_info", None)
                            if audit_data:
                                audit_dict = (
                                    audit_data[-1]
                                    if isinstance(audit_data, list)
                                    else audit_data
                                )
                                bypass_mask = np.asarray(
                                    audit_dict.get("gripper_bypass_mask", []),
                                    dtype=bool,
                                )
                                nominal_actions = np.asarray(
                                    audit_dict.get("nominal_actions", []),
                                    dtype=np.float32,
                                )
                            else:
                                nominal_actions = None
                            force_safety_hold = self._residual_safety_hold_remaining > 0
                            if force_safety_hold:
                                self._residual_safety_hold_remaining -= 1
                            suppress_gripper = (
                                self._residual_gripper_inhibit_remaining > 0
                            )
                            if suppress_gripper:
                                self._residual_gripper_inhibit_remaining -= 1
                            if (
                                nominal_actions is not None
                                and nominal_actions.size > 0
                                and (force_safety_hold or suppress_gripper)
                            ):
                                # P2-4: after a human takeover / rejection the
                                # model gripper override is suppressed for the
                                # debounce window (executed command -> KEEP).
                                # P2-5: after a safety violation the whole
                                # residual is held at nominal.
                                if force_safety_hold:
                                    rollout_result.actions[..., :7] = torch.as_tensor(
                                        nominal_actions[..., :7],
                                        device=rollout_result.actions.device,
                                    )
                                if force_safety_hold or suppress_gripper:
                                    rollout_result.actions[..., 7] = torch.as_tensor(
                                        nominal_actions[..., 7],
                                        device=rollout_result.actions.device,
                                    )
                                bypass_mask = np.zeros_like(bypass_mask)
                                suppressed_mask = np.ones(
                                    nominal_actions.shape[0], dtype=bool
                                )
                            self._residual_gripper_suppressed_mask = suppressed_mask
                        # P2-5: independent safety barrier runs *before* the
                        # chunk reaches the servo.  On violation the chunk is
                        # replaced with nominal here (never executed as-is),
                        # a hold is entered, and the transition is marked
                        # invalid at finalization.
                        bypass_mask, barrier_suppressed = self._residual_safety_check(
                            rollout_result, nominal_actions, bypass_mask
                        )
                        if barrier_suppressed is not None:
                            suppressed_mask = barrier_suppressed
                            self._residual_gripper_suppressed_mask = suppressed_mask
                        env_step_kwargs = {}
                        if bypass_mask is not None:
                            env_step_kwargs["gripper_bypass_mask"] = bypass_mask
                        env_output, env_info, chunk_step_payload = (
                            self.env_interact_step(
                                rollout_result.actions,
                                stage_id,
                                **env_step_kwargs,
                            )
                        )
                        if getattr(self, "residual_hil_rlpd_mode", False):
                            self._residual_observe_chunk(
                                rollout_result,
                                env_output,
                                chunk_start_obs,
                                stage_id,
                            )
                        stage_rollout = self.rollout_results[stage_id]
                        if isinstance(stage_rollout, EmbodiedLerobotRolloutResult):
                            stage_rollout.append_chunk_episode_data(
                                rollout_result=rollout_result,
                                **chunk_step_payload,
                            )
                        if (
                            not self.cfg.env.train.auto_reset
                            and self._terminal_padding_triggered(
                                enabled=train_terminal_padding_enabled,
                                dones=env_output.dones,
                            )
                        ):
                            terminal_padding_active[stage_id] = True
                            terminal_ended_this_step = True
                            if self._terminal_end_monotonic[stage_id] is None:
                                self._terminal_end_monotonic[stage_id] = (
                                    time.monotonic()
                                )
                            self.record_env_metrics(env_metrics, env_info)
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(
                            env_batch,
                            rollout_padding=(
                                terminal_padding_active[stage_id]
                                if train_terminal_padding_enabled
                                else None
                            ),
                        ),
                        mode="train",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    if (
                        self.collect_transitions
                        and not self.enable_rlt
                        and not terminal_padding_active[stage_id]
                    ):
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record and not terminal_ended_this_step:
                        self.record_env_metrics(env_metrics, env_info)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self._update_last_rollout_audit(stage_id, env_output)
                self._apply_last_action_overrides(stage_id, env_output)

                reward_model_output = None
                if reward_channel is not None and not terminal_padding_active[stage_id]:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    batch_size=self.train_batch_size,
                    merge_fn=RolloutResult.merge_rollout_results,
                    infer_batch_size_fn=self._infer_rollout_batch_size,
                    decoupled_mode=self.env_decoupled_mode,
                )
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                chunk_step_result = ChunkStepResult(
                    actions=rollout_result.forward_inputs.get("action", None),
                    prev_logprobs=(
                        rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None
                    ),
                    prev_values=(
                        rollout_result.prev_values if self.collect_prev_infos else None
                    ),
                    forward_inputs=rollout_result.forward_inputs,
                    versions=rollout_result.versions,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)
                if self.enable_rlt and self.collect_transitions:
                    update_rlt_transitions(
                        stage_id,
                        rlt_pending_obs,
                        self.rollout_results,
                        rollout_result,
                        cache_current=False,
                    )

            if self.use_training_pipeline and actor_channel is not None:
                await self.send_rollout_trajectories_pipeline(
                    self.rollout_results, actor_channel
                )
                self.rollout_results = self._prepare_rollout_results(
                    getattr(self, "rollout_results", None)
                )

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if not self.use_training_pipeline and actor_channel is not None:
            if self.enable_online_lerobot:
                for stage_id in range(self.stage_num):
                    episodes = self.rollout_results[stage_id].drain_episodes()
                    await self.send_lerobot_episodes(episodes, actor_channel)
            else:
                for stage_id in range(self.stage_num):
                    await self.send_rollout_trajectories(
                        self.rollout_results[stage_id], actor_channel
                    )

        if getattr(self, "residual_hil_rlpd_mode", False):
            self._flush_residual_pending()

        if (
            getattr(self, "residual_hil_rlpd_mode", False)
            and self._pending_residual_transitions
            and actor_channel is not None
        ):
            from rlinf.algorithms.residual_hil_rlpd.messages import (
                build_transition_message,
            )

            for transition in self._pending_residual_transitions:
                self.send_to(
                    group_name=self.cfg.actor.group_name,
                    channel=actor_channel,
                    data=build_transition_message(
                        transition,
                        policy_version=int(
                            np.asarray(transition.policy_version).reshape(-1)[0]
                        ),
                        run_id=self._run_id,
                        sender_rank=int(getattr(self, "_rank", 0)),
                    ),
                )
            self._pending_residual_transitions.clear()

        if train_terminal_padding_enabled:
            total_chunks = valid_chunks + padded_chunks
            env_metrics["rollout/valid_chunks"].append(
                torch.tensor([valid_chunks], dtype=torch.float32)
            )
            env_metrics["rollout/padded_chunks"].append(
                torch.tensor([padded_chunks], dtype=torch.float32)
            )
            env_metrics["rollout/padding_fast_path_count"].append(
                torch.tensor([padded_chunks], dtype=torch.float32)
            )
            env_metrics["rollout/padding_fraction"].append(
                torch.tensor(
                    [padded_chunks / total_chunks if total_chunks else 0.0],
                    dtype=torch.float32,
                )
            )

        if getattr(self, "residual_hil_rlpd_mode", False):
            env_metrics["residual/safety_violations"].append(
                torch.tensor([self._residual_safety_violations], dtype=torch.float32)
            )
            env_metrics["residual/chunk_counter"].append(
                torch.tensor([self._residual_chunk_counter], dtype=torch.float32)
            )
            env_metrics["residual/gripper_inhibit_remaining"].append(
                torch.tensor(
                    [self._residual_gripper_inhibit_remaining],
                    dtype=torch.float32,
                )
            )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    @Worker.timer("evaluate")
    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
        eval_terminal_padding_enabled = getattr(
            self,
            "eval_terminal_padding_enabled",
            False,
        )
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            eval_padding_active = [False] * self.stage_num
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(env_batch),
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    raw_chunk_actions = (
                        rollout_results.actions
                        if hasattr(rollout_results, "actions")
                        else rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    bypass_mask = None
                    if getattr(self, "residual_hil_rlpd_mode", False) and hasattr(
                        rollout_results, "audit_info"
                    ):
                        audit_data = rollout_results.audit_info
                        audit_dict = (
                            audit_data[-1]
                            if isinstance(audit_data, list)
                            else audit_data
                        )
                        bypass_mask = np.asarray(
                            audit_dict.get("gripper_bypass_mask", []),
                            dtype=bool,
                        )
                    if eval_padding_active[stage_id]:
                        env_output = self._make_terminal_padding_env_output(env_output)
                        env_info = {}
                    else:
                        env_output, env_info = self.env_evaluate_step(
                            raw_chunk_actions,
                            stage_id,
                            gripper_bypass_mask=bypass_mask,
                        )
                        if (
                            not self.cfg.env.eval.auto_reset
                            and self._terminal_padding_triggered(
                                enabled=eval_terminal_padding_enabled,
                                dones=env_output.dones,
                            )
                        ):
                            eval_padding_active[stage_id] = True

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(
                            env_batch,
                            rollout_padding=(
                                eval_padding_active[stage_id]
                                if eval_terminal_padding_enabled
                                else None
                            ),
                        ),
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def compute_advantages_and_returns(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Advantages/returns are rollout-level quantities, so compute them before
        # splitting. After this point each channel item is an actor micro-batch that can
        # be trained directly without reconstructing the full rollout batch on actor.
        assert not (
            self.use_training_pipeline and self.cfg.algorithm.adv_type == "opd"
        ), (
            "OPD does not support runner.use_training_pipeline=True because "
            "teacher_logprobs are computed on actor workers after rollout."
        )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "prev_logprobs": rollout_batch.get("prev_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True)
            and not self.use_training_pipeline,
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_pipeline_batch(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        batch = convert_trajectories_to_batch([trajectory])
        batch = preprocess_embodied_batch(
            batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            reward_label_validity=self.reward_label_validity_enabled,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        return self.compute_advantages_and_returns(batch)

    def pack_pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        if self.shuffle_rollout:
            shuffle_id = torch.randperm(
                batch_size, generator=self.shuffle_generators[actor_rank]
            )
        else:
            shuffle_id = torch.arange(batch_size)

        flatten_batch = flatten_embodied_batch(batch, shuffle_id)
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self,
        rollout_results: list[EmbodiedRolloutResult],
        channel: Channel,
    ) -> None:
        pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
        batches_by_actor_rank: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(
            list
        )

        with self.worker_timer("prepare_micro_batches"):
            for stage_id, rollout_result in enumerate(rollout_results):
                actor_splits = self.pipeline_stage_actor_splits[stage_id]
                trajectories = rollout_result.to_splited_trajectories_by_sizes(
                    [split_size for _, split_size in actor_splits]
                )

                for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                    batch = self.prepare_pipeline_batch(trajectory)
                    pending_batches.append((actor_rank, batch))
                    batches_by_actor_rank[actor_rank].append(batch)

            if self.cfg.algorithm.get("normalize_advantages", True):
                for actor_rank, batches in sorted(batches_by_actor_rank.items()):
                    local_adv_stats = sum(
                        masked_stats(batch["advantages"], batch.get("loss_mask"))
                        for batch in batches
                    )
                    env_ranks = self.pipeline_actor_env_ranks[actor_rank]
                    if len(env_ranks) > 1:
                        global_adv_stats = sum(
                            self.broadcast(
                                local_adv_stats if self._rank == src_rank else None,
                                groups=[(self._group_name, env_ranks)],
                                src=(self._group_name, src_rank),
                            )
                            for src_rank in env_ranks
                        )
                    else:
                        # Single env worker: broadcast is a no-op, skip Gloo creation
                        global_adv_stats = local_adv_stats
                    for batch in batches:
                        batch["advantages"] = normalize_from_stats(
                            batch["advantages"], global_adv_stats
                        )

            for actor_rank, batch in pending_batches:
                for micro_batch in self.pack_pipeline_micro_batches(batch, actor_rank):
                    channel.put(
                        micro_batch,
                        key=self.pipeline_actor_keys[actor_rank],
                        async_op=True,
                    )
