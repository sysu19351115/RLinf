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

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_id = -1
        self._episode_needs_reset = False
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        """
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        self._episode_id += 1
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        self._episode_needs_reset = False
        return extracted_obs, infos

    def _info_bool_array(self, infos, key):
        """Return a vectorized boolean info value for every real-world env."""
        values = np.asarray(infos.get(key, False), dtype=bool)
        if values.ndim == 0:
            values = np.full(self.num_envs, bool(values), dtype=bool)
        values = values.reshape(-1)
        if values.shape != (self.num_envs,):
            raise ValueError(
                f"info[{key!r}] must have shape ({self.num_envs},), got {values.shape}."
            )
        return values

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        state = raw_obs["state"]
        full_states = np.concatenate([state[k] for k in sorted(state)], axis=-1)
        obs["states"] = full_states

        # Forward prev_states for pose-mode envs (e.g. DobotCartesianEnv).
        # If present in raw_obs, pass through to the policy pipeline.
        if "prev_state" in raw_obs:
            obs["prev_states"] = raw_obs["prev_state"]

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]
        raw_images = OrderedDict(sorted(frames.items()))
        raw_images.pop(self.main_image_key)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if self._episode_needs_reset:
            raise RuntimeError(
                "The real-world episode has ended; reset() is required before "
                "another action can be executed."
            )
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        infos["episode_id"] = np.full(self.num_envs, self._episode_id, dtype=np.int64)
        infos["episode_step_id"] = self.elapsed_steps.astype(np.int64) - 1
        terminations = np.asarray(terminations, dtype=bool).copy()
        truncations = np.asarray(truncations, dtype=bool).copy()
        operator_episode_end = self._info_bool_array(infos, "operator_episode_end")
        operator_success = self._info_bool_array(infos, "operator_success")
        # max_episode_steps: null → external wrapper owns episode end.
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            # Preserve lower-level safety/timeout truncations. The outer horizon
            # is an additional stop condition, never a replacement.
            truncations = np.logical_or(truncations, timeout_truncations)

        # Operator control-plane events take precedence over environment task
        # semantics and must survive ignore_terminations.
        terminations[operator_episode_end] = operator_success[operator_episode_end]
        truncations[operator_episode_end] = ~operator_success[operator_episode_end]

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations = np.logical_and(terminations, operator_episode_end)

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)
        if "rlt_switch_flags" in infos:
            infos["rlt_switch_flags"] = to_tensor(
                np.asarray(infos["rlt_switch_flags"], dtype=bool)
            )

        dones = terminations | truncations
        if dones.any():
            self._episode_needs_reset = True
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_rlt_switch_flags = []
        raw_chunk_episode_step_ids = []
        raw_chunk_action_command_accepted = []
        chunk_episode_id = None
        stopped_early = False
        shutdown_requested = False
        controller_rejection_detected = False
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if chunk_episode_id is None:
                chunk_episode_id = torch.as_tensor(
                    infos["episode_id"], dtype=torch.int64
                )
            raw_chunk_episode_step_ids.append(
                torch.as_tensor(infos["episode_step_id"], dtype=torch.int64)
            )
            if "action_command_accepted" in infos:
                command_accepted = self._info_bool_array(
                    infos, "action_command_accepted"
                )
            else:
                # Other real-world environments may not expose controller
                # acknowledgement. A step that returned normally remains
                # backward-compatible and is considered accepted.
                command_accepted = np.ones(self.num_envs, dtype=bool)
            raw_chunk_action_command_accepted.append(
                torch.as_tensor(command_accepted, dtype=torch.bool)
            )
            command_rejected = np.logical_not(command_accepted)
            if command_rejected.any():
                reason_values = np.asarray(
                    infos.get(
                        "termination_reason",
                        np.full(self.num_envs, "none", dtype=object),
                    ),
                    dtype=object,
                )
                if reason_values.ndim == 0:
                    reason_values = np.full(
                        self.num_envs, reason_values.item(), dtype=object
                    )
                reason_values = reason_values.reshape(-1)
                if reason_values.shape != (self.num_envs,):
                    raise ValueError(
                        "info['termination_reason'] must have shape "
                        f"({self.num_envs},), got {reason_values.shape}."
                    )
                reason_values[command_rejected] = "controller_rejection"
                infos["termination_reason"] = reason_values
                infos["controller_rejection"] = command_rejected.copy()
                truncations = torch.logical_or(
                    truncations,
                    torch.as_tensor(command_rejected, dtype=torch.bool),
                )
                controller_rejection_detected = True
                self._episode_needs_reset = True
            shutdown_requested = shutdown_requested or bool(
                self._info_bool_array(infos, "operator_shutdown_requested").any()
            )
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            if (
                command_rejected.any()
                or torch.logical_or(terminations, truncations).all()
            ):
                # RealWorldEnv currently permits exactly one env per worker.
                # Never execute stale actions after an operator/fault/timeout
                # has ended that episode. Pad tensors below so the rollout
                # contract retains its configured action-chunk shape.
                stopped_early = i + 1 < chunk_size
                break

        executed_steps = len(chunk_rewards)
        if stopped_early:
            skipped_steps = chunk_size - executed_steps
            terminal_obs = obs_list[-1]
            terminal_info = infos_list[-1]
            zero_reward = torch.zeros_like(chunk_rewards[-1])
            zero_done = torch.zeros_like(raw_chunk_terminations[-1])
            zero_intervene_action = (
                torch.zeros_like(raw_chunk_intervene_actions[-1])
                if raw_chunk_intervene_actions
                else None
            )
            zero_intervene_flag = (
                torch.zeros_like(raw_chunk_intervene_flag[-1])
                if raw_chunk_intervene_flag
                else None
            )
            for _ in range(skipped_steps):
                obs_list.append(copy.deepcopy(terminal_obs))
                infos_list.append(copy.deepcopy(terminal_info))
                chunk_rewards.append(zero_reward.clone())
                raw_chunk_terminations.append(zero_done.clone())
                raw_chunk_truncations.append(zero_done.clone())
                raw_chunk_episode_step_ids.append(
                    torch.full((self.num_envs,), -1, dtype=torch.int64)
                )
                raw_chunk_action_command_accepted.append(
                    torch.zeros(self.num_envs, dtype=torch.bool)
                )
                if raw_chunk_intervene_actions:
                    raw_chunk_intervene_actions.append(zero_intervene_action.clone())
                    raw_chunk_intervene_flag.append(zero_intervene_flag.clone())

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
            infos_list[-1] = infos_last
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = torch.stack(
                raw_chunk_rlt_switch_flags, dim=1
            )
            infos_list[-1] = infos_last
        action_command_accepted_mask = torch.stack(
            raw_chunk_action_command_accepted, dim=1
        )
        step_reached_mask = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)
        step_reached_mask[:, :executed_steps] = True
        executed_action_mask = torch.logical_and(
            step_reached_mask, action_command_accepted_mask
        )
        infos_last["action_command_accepted_mask"] = action_command_accepted_mask
        infos_last["executed_action_mask"] = executed_action_mask
        infos_last["skipped_action_steps"] = chunk_size - executed_steps
        infos_last["episode_id"] = chunk_episode_id
        infos_last["episode_step_ids"] = torch.stack(raw_chunk_episode_step_ids, dim=1)
        infos_list[-1] = infos_last

        if (
            past_dones.any()
            and self.auto_reset
            and not shutdown_requested
            and not controller_rejection_detected
        ):
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
