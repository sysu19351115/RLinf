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

"""Human-in-the-loop data collection for the Dobot CR5AF robot.

Loads a pi0.5 checkpoint (or a hold / dummy policy), rolls out the policy in
the real Dobot environment, and lets a human operator take over via keyboard-
based 6D Cartesian end-effector control:

* ``w``/``s``, ``a``/``d``, ``q``/``e`` -- move the TCP along base X/Y/Z
* ``i``/``k``, ``j``/``l``, ``u``/``o`` -- rotate the TCP around tool X/Y/Z
* ``,``/``.`` -- close/open the gripper (normalized [0, 1])
* ``h`` -- toggle MODEL / ENGAGE
* ``m`` -- return to MODEL control immediately
* ``Enter`` -- save the current episode and start a new one
* ``Backspace`` -- discard the current episode (reset without saving)
* ``ESC`` -- discard the current episode, reset to initial pose, and exit

Three ``policy_mode`` values are supported:

* ``dummy`` -- no hardware, no model; tests the software pipeline end-to-end.
* ``hold`` -- connects to the real robot but does not load a model; always
  suggests the current pose as the action. Use this for safety verification.
* ``model`` -- loads the PI0.5 checkpoint and uses it for MODEL-mode actions.

Collected episodes are exported in LeRobot format by the ``CollectEpisode``
wrapper using OpenPI's Dobot pose field names. HIL diagnostic fields such as
``model_action``, ``model_action_valid``, and ``intervene_flag`` are retained.

Launch (dummy mode for verification)::

    python examples/embodiment/collect_dobot_hil_data.py \\
        --config-name dobot_hil_collect policy_mode=dummy
"""

from __future__ import annotations

import time
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.envs.realworld.dobot.hold_policy import HoldPolicy
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers import CollectEpisode
from rlinf.envs.wrappers.collect_episode import resolve_collection_save_dir
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.logging import get_logger

_ACTION_DIM = 8


class DobotHILCollector(Worker):
    """Single-process collector that runs a policy with keyboard HIL for Dobot."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._logger = get_logger()

        # ── Validate policy_mode / is_dummy consistency ──────────────────────
        policy_mode = str(getattr(cfg, "policy_mode", "model"))
        if policy_mode not in ("dummy", "hold", "model"):
            raise ValueError(
                f"policy_mode must be 'dummy', 'hold', or 'model', got {policy_mode!r}."
            )
        self.policy_mode = policy_mode
        is_dummy = bool(cfg.env.eval.override_cfg.get("is_dummy", False))
        if policy_mode == "dummy" and not is_dummy:
            raise ValueError(
                "policy_mode='dummy' requires env.eval.override_cfg.is_dummy=true."
            )
        if policy_mode in ("hold", "model") and is_dummy:
            raise ValueError(
                f"policy_mode='{policy_mode}' requires env.eval.override_cfg.is_dummy=false."
            )

        eval_cfg = cfg.env.eval
        dc_cfg = eval_cfg.get("data_collection")
        collection_save_dir = None
        if dc_cfg and getattr(dc_cfg, "enabled", False):
            collection_save_dir = resolve_collection_save_dir(dc_cfg)
            self.log_info(f"Dobot HIL dataset session: {collection_save_dir}")

        self.env = RealWorldEnv(
            eval_cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        if dc_cfg and getattr(dc_cfg, "enabled", False):
            self.env = CollectEpisode(
                self.env,
                save_dir=collection_save_dir,
                export_format=dc_cfg.get("export_format", "lerobot"),
                robot_type=dc_cfg.get("robot_type", "dobot_cf5af"),
                fps=dc_cfg.get("fps", 30),
                only_success=dc_cfg.get("only_success", True),
                finalize_interval=dc_cfg.get("finalize_interval", 20),
                resume=bool(dc_cfg.get("resume", False)),
                image_writer_threads=int(dc_cfg.get("image_writer_threads", 1)),
                image_writer_processes=int(dc_cfg.get("image_writer_processes", 1)),
                required_observation_fields=("prev_states",),
                dataset_layout=dc_cfg.get("dataset_layout", "openpi_dobot_pose"),
            )
            self._preexisting = int(getattr(self.env, "preexisting_episode_count", 0))
        else:
            self._preexisting = 0

        self.action_dim = int(self.env.action_space.shape[-1])
        if self.action_dim != _ACTION_DIM:
            raise ValueError(
                f"Dobot HIL requires action_dim={_ACTION_DIM} (cartesian pose), "
                f"got action_dim={self.action_dim}."
            )

        self.policy = self._load_policy()
        self.num_episodes = int(cfg.runner.num_data_episodes)
        self._target_step_period = (
            1.0 / float(dc_cfg.get("fps", 30)) if dc_cfg else None
        )

        # Action-chunk queue: cleared on HIL mode switches.
        self._action_queue: np.ndarray | None = None
        # After ENGAGE -> MODEL, execute one hold step for a fresh observation.
        self._pending_model_hold_step = False
        # Whether to wait for the operator's 'y' key after each reset.
        self._wait_for_start_key = bool(
            eval_cfg.get("wait_for_start_key", True)
        )

    def _load_policy(self):
        model_cfg = self.cfg.actor.model
        if self.policy_mode == "dummy":
            self.log_info("Using dummy policy (no hardware, no model).")
            action_chunk = int(model_cfg.get("num_action_chunks", 1))
            return HoldPolicy(action_dim=self.action_dim, action_chunk=action_chunk)
        if self.policy_mode == "hold":
            self.log_info(
                "Using hold policy (real robot, no model inference). "
                "Actions will hold the current pose."
            )
            action_chunk = int(model_cfg.get("num_action_chunks", 1))
            return HoldPolicy(action_dim=self.action_dim, action_chunk=action_chunk)

        self.log_info(f"Loading policy from {model_cfg.model_path}")
        from rlinf.models.embodiment.openpi import get_model

        return get_model(model_cfg)

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Add optional image keys required by OpenPI obs_processor.

        Dobot only has one camera (cam_left_wrist), so ``wrist_images`` and
        ``extra_view_images`` are absent from the env obs. obs_processor
        accesses them directly (not via .get()), so we must fill them with None.
        """
        obs = dict(obs)
        if "wrist_images" not in obs:
            obs["wrist_images"] = None
        if "extra_view_images" not in obs:
            obs["extra_view_images"] = None
        return obs

    def _get_current_pose_state(self) -> np.ndarray | None:
        """Read the latest 8-dim pose state from the hardware."""
        try:
            env = self.env
            while hasattr(env, "env") and not hasattr(env, "envs"):
                env = env.env
            if not hasattr(env, "envs"):
                return None
            pose = env.envs[0].get_pose_state()
            return np.asarray(pose, dtype=np.float64)
        except Exception as e:
            self.log_warning(f"Failed to read current pose state: {e}")
            return None

    def _hold_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        """Build a safe hold action from the current observation state.

        Returns shape ``(1, action_dim)``. The quaternion is normalized to
        ensure safety.
        """
        state = obs["states"]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float64).reshape(1, self.action_dim)
        if not np.isfinite(state).all():
            raise ValueError(
                "Non-finite state in observation; cannot build hold action."
            )
        # Normalize quaternion (indices 3:7, wxyz).
        q = state[0, 3:7]
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            state[0, 3:7] = q / norm
        return state

    def _infer_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        """Run policy inference and return the predicted action chunk.

        Returned shape is ``(action_chunk, action_dim)``.
        """
        obs = self._prepare_obs(obs)
        with torch.no_grad():
            actions, _ = self.policy.predict_action_batch(
                env_obs=obs,
                mode="eval",
                compute_values=False,
            )
        actions = actions.detach().cpu().numpy()
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 2 and actions.shape[0] == 1:
            actions = actions[0:1, :]
        elif actions.ndim == 1:
            actions = actions[None, :]
        # Validate.
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Policy output action_dim={actions.shape[-1]} != expected {self.action_dim}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite action from policy inference; rejecting.")
        return np.asarray(actions, dtype=np.float64)

    def _get_action(self, obs: dict[str, Any], hil_state: str = "model") -> np.ndarray:
        """Return the next action for the vectorized environment.

        Shape is ``(1, action_dim)``. In ENGAGE mode the intervention wrapper
        replaces the action, so we return a hold action without consuming the
        queue.
        """
        if hil_state == "engage":
            # ENGAGE: wrapper replaces action; return safe hold as fallback.
            return self._hold_from_obs(obs)

        if self._pending_model_hold_step:
            self._pending_model_hold_step = False
            current_pose = self._get_current_pose_state()
            if current_pose is not None:
                self.log_info(
                    "[HIL] ENGAGE -> MODEL hold step; acquiring fresh observation"
                )
                return current_pose[None, :].copy()
            self.log_warning(
                "[HIL] failed to read pose for hold step; falling back to obs hold"
            )
            return self._hold_from_obs(obs)

        if self._action_queue is None or self._action_queue.shape[0] == 0:
            self._action_queue = self._infer_action_chunk(obs)
        action = self._action_queue[0:1, :].copy()
        self._action_queue = self._action_queue[1:, :]
        return action

    def run(self):
        progress = tqdm(
            total=self.num_episodes,
            initial=self._preexisting,
            desc="Collecting Dobot HIL episodes",
        )
        try:
            self._wait_for_episode_start()
            obs, _ = self.env.reset()
            self._action_queue = None
            self._pending_model_hold_step = False
            episodes_done = self._preexisting
            prev_hil_state = "model"

            while episodes_done < self.num_episodes:
                iter_start = time.perf_counter()

                # Determine if this action is a real model prediction.
                is_model_action = (
                    prev_hil_state == "model"
                    and not self._pending_model_hold_step
                    and (
                        self._action_queue is not None
                        and self._action_queue.shape[0] > 0
                    )
                )
                # If the queue is empty, _get_action will infer (model action).
                if prev_hil_state == "model" and (
                    self._action_queue is None or self._action_queue.shape[0] == 0
                ):
                    is_model_action = self.policy_mode == "model"

                action = self._get_action(obs, hil_state=prev_hil_state)

                # Set model_action_valid flag for the wrapper.
                self._set_model_action_valid(is_model_action)

                next_obs, reward, terminated, truncated, info = self.env.step(action)

                if "hil_state" in info:
                    hil_state = info["hil_state"][0]
                    if hil_state != prev_hil_state:
                        self.log_info(f"[HIL] state={hil_state}")
                        self._action_queue = None
                        if hil_state == "model" and prev_hil_state == "engage":
                            self._pending_model_hold_step = True
                        prev_hil_state = hil_state

                obs = next_obs

                # ── HIL event handling (priority order) ───────────────────────
                # 1. quit: reset, then exit.
                quit_program = bool(np.asarray(info.get("quit_program", False)).any())
                if quit_program:
                    self.log_info(
                        "Quit requested by operator (ESC); resetting to initial "
                        "pose and exiting without saving."
                    )
                    try:
                        self.env.reset()
                    except Exception as e:
                        self.log_warning(f"Failed to reset before exit: {e}")
                    break

                # 2. abort: reset, clear state, do not increment episode count.
                hil_event = info.get("hil_event")
                if hil_event is not None:
                    hil_event = (
                        hil_event[0]
                        if isinstance(hil_event, (np.ndarray, list))
                        else hil_event
                    )
                if hil_event == "abort":
                    self.log_info(
                        "Abort requested by operator (Backspace); resetting "
                        "without saving current episode."
                    )
                    self._wait_for_episode_start()
                    obs, _ = self.env.reset()
                    self._action_queue = None
                    self._pending_model_hold_step = False
                    prev_hil_state = "model"
                    continue

                # 3. terminated/truncated: save event confirmed, increment.
                done = bool(terminated.any().item() or truncated.any().item())
                if done:
                    episodes_done += 1
                    progress.update(1)
                    if episodes_done < self.num_episodes:
                        self.log_info(f"Episode {episodes_done} saved; continuing.")
                        self._wait_for_episode_start()
                        obs, _ = self.env.reset()
                        self._action_queue = None
                        self._pending_model_hold_step = False
                        prev_hil_state = "model"
                    else:
                        self.log_info(f"Episode {episodes_done} saved; reached limit.")

                elapsed = time.perf_counter() - iter_start
                if self._target_step_period is not None:
                    sleep_for = self._target_step_period - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        finally:
            progress.close()
            self.env.close()
            self.log_info("Dobot HIL data collection finished.")

    def _set_model_action_valid(self, valid: bool) -> None:
        """Set the model_action_valid flag on the keyboard wrapper.

        Walks the wrapper chain (CollectEpisode → RealWorldEnv → vector env).
        When a vector env (with ``.envs`` and ``.call``) is reached, broadcasts
        via ``call("set_model_action_valid", valid)`` to the underlying
        DobotKeyboardIntervention. Raises if the wrapper is not found.
        """
        env = self.env
        while True:
            if hasattr(env, "set_model_action_valid"):
                env.set_model_action_valid(valid)
                return
            if hasattr(env, "envs") and hasattr(env, "call"):
                # Vector env: broadcast to sub-envs (DobotKeyboardIntervention).
                env.call("set_model_action_valid", valid)
                return
            if hasattr(env, "env"):
                env = env.env
            else:
                break
        raise RuntimeError(
            "Could not find DobotKeyboardIntervention with set_model_action_valid "
            "in the env stack. Ensure the keyboard wrapper is applied."
        )

    def _wait_for_episode_start(self) -> None:
        """Block until the operator presses 'y' to start the next episode.

        Skipped in dummy mode (the wrapper's listener is a no-op) or when
        ``wait_for_start_key`` is disabled in config. Call this *before*
        ``env.reset()`` so the returned obs is fresh.
        """
        if not self._wait_for_start_key:
            return
        env = self.env
        while True:
            if hasattr(env, "wait_for_start_key"):
                env.wait_for_start_key()
                return
            if hasattr(env, "envs") and hasattr(env, "call"):
                env.call("wait_for_start_key")
                return
            if hasattr(env, "env"):
                env = env.env
            else:
                break


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="dobot_hil_collect",
)
def main(cfg: DictConfig):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = DobotHILCollector.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
