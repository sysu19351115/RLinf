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
"""GymAlohaEnv: RLinf-compatible wrapper around the gym_aloha MuJoCo simulator."""

from __future__ import annotations

import gym_aloha  # noqa: F401  triggers gymnasium registration side-effect

import copy
from typing import Any, Optional, Union

import gymnasium as gymnasium
import numpy as np
import torch
from PIL import Image

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


def _resize_with_pad(img: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Resize image to target size with zero-padding (preserves aspect ratio).

    Args:
        img: Input image as a numpy array [H, W, C].
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized uint8 image of shape [target_h, target_w, C].
    """
    if img.shape[:2] == (target_h, target_w):
        img = img.astype(np.uint8) if img.dtype != np.uint8 else img
        return img
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    pil_img = Image.fromarray(img)
    h, w = img.shape[:2]
    ratio = max(w / target_w, h / target_h)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new(resized.mode, (target_w, target_h), 0)
    padded.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.array(padded)


class GymAlohaEnv:
    """RLinf-compatible wrapper around the ``gym_aloha`` MuJoCo simulator.

    The environment runs the ``AlohaTransferCube-v0`` task by default.
    Each instance manages ``num_envs`` parallel gymnasium environments.

    Args:
        cfg: Hydra ``DictConfig`` for this env (e.g. ``cfg.env.train``).
        num_envs: Number of parallel environments this worker manages.
        seed_offset: Unique seed offset for this worker process.
        total_num_processes: Total number of env worker processes.
        worker_info: Opaque worker metadata from the scheduler.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        self.cfg = cfg
        self._num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        self.group_size: int = cfg.group_size
        self.num_group: int = num_envs // self.group_size
        self.auto_reset: bool = cfg.auto_reset
        self.ignore_terminations: bool = cfg.ignore_terminations
        self.use_rel_reward: bool = cfg.use_rel_reward
        self.use_step_penalty: bool = getattr(cfg, "use_step_penalty", False)
        self.max_episode_steps: int = cfg.max_episode_steps
        self.reward_coef: float = float(getattr(cfg, "reward_coef", 1.0))
        self.task_name: str = getattr(cfg, "task_name", "Transfer cube")

        self._is_start = True
        self._generator = np.random.default_rng(seed=self.seed)

        task = getattr(cfg, "gym_task", "gym_aloha/AlohaTransferCube-v0")
        self._envs: list[gymnasium.Env] = []
        for i in range(num_envs):
            env = gymnasium.make(
                task,
                obs_type="pixels_agent_pos",
                render_mode=None,
                max_episode_steps=self.max_episode_steps,
            )
            env.reset(seed=self.seed + i)
            self._envs.append(env)

        self.prev_step_reward = np.zeros(num_envs, dtype=np.float32)
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)

        self.success_once = np.zeros(num_envs, dtype=bool)
        self.returns = np.zeros(num_envs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def info_logging_keys(self):
        return ["is_success"]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs_list: list[dict]) -> dict[str, Any]:
        """Convert a list of gymnasium observations into the RLinf obs dict."""
        images = []
        states = []
        for obs in raw_obs_list:
            img = _resize_with_pad(obs["pixels"]["top"])  # [H,W,C] uint8
            images.append(img)
            states.append(obs["agent_pos"])  # (14,) float64

        return {
            "main_images": torch.from_numpy(np.stack(images)).contiguous(),
            "wrist_images": None,
            "states": torch.from_numpy(np.stack(states)).float(),
            "task_descriptions": [self.task_name] * len(raw_obs_list),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            env_idx = np.arange(self._num_envs)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info: dict[str, Any] = {}
        self.returns += step_reward.numpy()
        episode_info["success_once"] = torch.from_numpy(self.success_once.copy())
        episode_info["return"] = torch.from_numpy(self.returns.copy())
        episode_info["episode_len"] = torch.from_numpy(self._elapsed_steps.copy())
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[list[int], np.ndarray]] = None,
        reset_state_ids: Optional[Any] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the specified environments and return their observations.

        Args:
            env_idx: Indices of environments to reset (all if None).
            reset_state_ids: Ignored for gym_aloha (single task).

        Returns:
            ``(obs, infos)`` tuple.
        """
        if env_idx is None:
            env_idx = np.arange(self._num_envs)
        env_idx = np.atleast_1d(env_idx)

        raw_obs_list = []
        for i in env_idx:
            obs, _ = self._envs[i].reset()
            raw_obs_list.append(obs)

        self._reset_metrics(env_idx)
        self._is_start = False

        obs = self._wrap_obs(raw_obs_list)
        infos: dict[str, Any] = {}
        return obs, infos

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Execute one environment step.

        Args:
            actions: Action array of shape ``(num_envs, 14)``.
            auto_reset: Whether to auto-reset terminated/truncated envs.

        Returns:
            ``(obs, step_reward, terminations, truncations, infos)`` tuple.
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1

        raw_obs_list: list[dict] = []
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        terminations = np.zeros(self._num_envs, dtype=bool)
        truncations = np.zeros(self._num_envs, dtype=bool)
        info_list: list[dict] = []

        for i in range(self._num_envs):
            obs, reward, terminated, truncated, info = self._envs[i].step(actions[i])
            raw_obs_list.append(obs)
            # reward warpper
            if reward >= 4:
                reward = 1
            else:
                reward = 0
            rewards[i] = reward
            terminations[i] = terminated
            truncated = truncated or (self._elapsed_steps[i] >= self.max_episode_steps)
            truncations[i] = truncated
            info_list.append(info)

        obs = self._wrap_obs(raw_obs_list)
        step_reward = self._calc_step_reward(rewards)

        # Update success tracking: in gym_aloha, terminated == (reward == 4) == success
        self.success_once = self.success_once | terminations

        terminations_t = torch.from_numpy(terminations)
        truncations_t = torch.from_numpy(truncations)
        infos = list_of_dict_to_dict_of_list(info_list)
        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            if "is_success" in infos:
                infos["episode"]["success_at_end"] = terminations_t.clone()
            terminations_t = torch.zeros_like(terminations_t)

        dones = terminations_t | truncations_t
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations_t, truncations_t, infos

    def chunk_step(
        self,
        chunk_actions: torch.Tensor,
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute an action chunk (multiple steps) and aggregate results.

        Args:
            chunk_actions: Tensor of shape ``(num_envs, chunk_size, 14)``.

        Returns:
            ``(obs_list, chunk_rewards, chunk_terminations,
              chunk_truncations, infos_list)`` where rewards/terminations/
            truncations have shape ``(num_envs, chunk_size)``.
        """
        chunk_size = chunk_actions.shape[1]
        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []

        for i in range(chunk_size):
            obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)  # (B, C)
        raw_term_t = torch.stack(raw_chunk_terminations, dim=1)
        raw_trunc_t = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_term_t.any(dim=1)
        past_truncations = raw_trunc_t.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_term_t)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_trunc_t)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_term_t.clone()
            chunk_truncations = raw_trunc_t.clone()

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    # ------------------------------------------------------------------
    # Reward & auto-reset
    # ------------------------------------------------------------------

    def _calc_step_reward(self, raw_rewards: np.ndarray) -> torch.Tensor:
        """Compute the step reward from the raw gymnasium reward.

        Converts the binary success reward to a per-step relative reward if
        ``use_rel_reward`` is enabled, and applies ``reward_coef`` scaling.
        ``use_step_penalty`` subtracts 1.0 per step.
        """
        step_penalty = -1.0 if self.use_step_penalty else 0.0
        reward = self.reward_coef * raw_rewards + step_penalty

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward.copy()
            return torch.from_numpy(reward_diff).float()
        return torch.from_numpy(reward).float()

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset done environments, stashing the terminal observation."""
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)

        env_idx = np.arange(self._num_envs)[dones.cpu().numpy()]
        obs, _ = self.reset(env_idx=env_idx.tolist())

        for key in final_obs:
            val = final_obs[key]
            new_val = obs.get(key)
            if isinstance(val, torch.Tensor):
                val = val.clone()
                val[env_idx] = new_val
            elif isinstance(val, list):
                for j, idx in enumerate(env_idx):
                    val[idx] = new_val[j]
            final_obs[key] = val

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return final_obs, infos

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all underlying gymnasium environments."""
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
        self._envs = []
