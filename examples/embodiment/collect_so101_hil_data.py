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

"""Human-in-the-loop data collection for the SO101 bimanual robot.

Loads a pi0.5 checkpoint (or a dummy policy for verification), rolls out the
policy in the real SO101 environment, and lets a human operator take over via
keyboard-based 6D end-effector control:

* ``w``/``s``, ``a``/``d``, ``q``/``e`` -- move the active arm EE along X/Y/Z
* ``i``/``k``, ``j``/``l``, ``u``/``o`` -- rotate the active arm EE around X/Y/Z
* ``,``/``.`` -- close/open the active arm gripper
* ``Tab`` -- switch active arm (left/right)
* ``h`` -- toggle MODEL / ENGAGE
* ``m`` -- return to MODEL control immediately
* ``ESC`` -- quit the program; the robot first returns to ``initial_joints`` and then shuts down

Collected episodes are exported in LeRobot format by the ``CollectEpisode``
wrapper. The wrapper records the *executed* action (model or human) together
with the intervention flag.

Launch (dummy mode for verification):
    python examples/embodiment/collect_so101_hil_data.py --config-name so101_hil_collect
"""

from __future__ import annotations

import time
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers import CollectEpisode
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.logging import get_logger


class DummyPolicy:
    """Stand-in policy for dummy-mode verification.

    Returns zero actions with the correct action-chunk shape.
    """

    def __init__(self, action_dim: int, action_chunk: int = 1):
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bsize = int(env_obs["states"].shape[0])
        actions = torch.zeros(
            bsize, self.action_chunk, self.action_dim, dtype=torch.float32
        )
        return actions, {"forward_inputs": {"action": actions.reshape(bsize, -1)}}


class SO101HILCollector(Worker):
    """Single-process collector that runs a policy with leader-arm HIL."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._logger = get_logger()

        eval_cfg = cfg.env.eval
        self.env = RealWorldEnv(
            eval_cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        dc_cfg = eval_cfg.get("data_collection")
        if dc_cfg and getattr(dc_cfg, "enabled", False):
            self.env = CollectEpisode(
                self.env,
                save_dir=dc_cfg.save_dir,
                export_format=dc_cfg.get("export_format", "lerobot"),
                robot_type=dc_cfg.get("robot_type", "so101"),
                fps=dc_cfg.get("fps", 10),
                only_success=dc_cfg.get("only_success", False),
                finalize_interval=dc_cfg.get("finalize_interval", 100),
                resume=bool(dc_cfg.get("resume", False)),
                image_writer_threads=int(dc_cfg.get("image_writer_threads", 1)),
                image_writer_processes=int(dc_cfg.get("image_writer_processes", 1)),
            )
            self._preexisting = int(getattr(self.env, "preexisting_episode_count", 0))
        else:
            self._preexisting = 0

        self.action_dim = int(self.env.action_space.shape[-1])
        self.policy = self._load_policy()
        self.num_episodes = int(cfg.runner.num_data_episodes)
        self._target_step_period = (
            1.0 / float(dc_cfg.get("fps", 10)) if dc_cfg else None
        )

        # Action-chunk queue: the policy predicts a chunk of future actions; we
        # execute them one per step and only replan when the queue is empty.
        # This matches how action-chunk policies such as OpenPI are meant to be
        # rolled out. The queue is cleared on HIL mode switches so stale plans
        # are never executed after human intervention.
        self._action_queue: np.ndarray | None = None

        # After the operator switches from ENGAGE back to MODEL, we execute one
        # hold step at the current joint positions. This lets env.step() produce
        # a fresh observation that reflects the actual robot pose after human
        # intervention, before the policy is asked to predict again.
        self._pending_model_hold_step = False

    def _load_policy(self):
        model_cfg = self.cfg.actor.model
        if getattr(self.cfg, "use_dummy_policy", False):
            self.log_info("Using dummy policy for verification.")
            action_chunk = int(model_cfg.get("num_action_chunks", 1))
            return DummyPolicy(action_dim=self.action_dim, action_chunk=action_chunk)

        self.log_info(f"Loading policy from {model_cfg.model_path}")
        from rlinf.models.embodiment.openpi import get_model

        return get_model(model_cfg)

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Add optional wrist_images key required by OpenPI obs_processor."""
        obs = dict(obs)
        if "wrist_images" not in obs:
            obs["wrist_images"] = None
        return obs

    def _get_current_joint_positions(self) -> np.ndarray | None:
        """Read the latest follower-arm joint positions from the hardware."""
        try:
            env = self.env
            while hasattr(env, "env") and not hasattr(env, "envs"):
                env = env.env
            if not hasattr(env, "envs"):
                return None
            q = env.envs[0].get_joint_positions()
            return np.asarray(q, dtype=np.float64)
        except Exception as e:
            self.log_warning(f"Failed to read current joint positions: {e}")
            return None

    def _infer_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        """Run policy inference and return the predicted action chunk.

        Returned shape is ``(action_chunk, action_dim)``. The policy outputs
        absolute joint targets via its own output transforms; we just normalize
        the tensor shape here.
        """
        obs = self._prepare_obs(obs)
        with torch.no_grad():
            actions, _ = self.policy.predict_action_batch(
                env_obs=obs,
                mode="eval",
                compute_values=False,
            )
        actions = actions.detach().cpu().numpy()
        # Normalize policy outputs to (action_chunk, action_dim).
        if actions.ndim == 3:
            # (batch, action_chunk, action_dim) -> drop batch dim.
            actions = actions[0]
        elif actions.ndim == 2 and actions.shape[0] == 1:
            # (1, action_dim) -> single-step action.
            actions = actions[0:1, :]
        elif actions.ndim == 1:
            # (action_dim,) -> single-step action.
            actions = actions[None, :]
        return np.asarray(actions, dtype=np.float64)

    def _get_action(self, obs: dict[str, Any], hil_state: str = "model") -> np.ndarray:
        """Return the next action for the vectorized environment.

        The returned array has shape ``(1, action_dim)`` because
        ``NoAutoResetSyncVectorEnv`` expects a batch of actions, even with
        ``num_envs=1``.

        In MODEL mode this pops from the action-chunk queue (refilling when
        empty). In ENGAGE mode the intervention wrapper replaces the action with
        the human IK command, so we return a dummy action without consuming the
        queue; this leaves the queued model actions intact for when control
        returns to MODEL.

        After switching back from ENGAGE to MODEL, the first MODEL step is a
        hold at the current joint positions so that ``env.step()`` produces a
        fresh observation for the next policy inference.
        """
        if hil_state == "engage":
            return np.zeros((1, self.action_dim), dtype=np.float64)

        if self._pending_model_hold_step:
            self._pending_model_hold_step = False
            current_q = self._get_current_joint_positions()
            if current_q is not None:
                self.log_info(
                    "[HIL] ENGAGE -> MODEL hold step; acquiring fresh observation"
                )
                return current_q[None, :].copy()
            self.log_warning(
                "[HIL] failed to read joints for hold step; falling back to inference"
            )

        if self._action_queue is None or self._action_queue.shape[0] == 0:
            self._action_queue = self._infer_action_chunk(obs)
        action = self._action_queue[0:1, :].copy()
        self._action_queue = self._action_queue[1:, :]
        return action

    def run(self):
        obs, _ = self.env.reset()
        self._action_queue = None
        self._pending_model_hold_step = False
        progress = tqdm(
            total=self.num_episodes,
            initial=self._preexisting,
            desc="Collecting SO101 HIL episodes",
        )
        episodes_done = self._preexisting
        prev_hil_state = "model"

        while episodes_done < self.num_episodes:
            iter_start = time.perf_counter()

            t0 = time.perf_counter()
            action = self._get_action(obs, hil_state=prev_hil_state)
            t_action = time.perf_counter() - t0

            t0 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            t_step = time.perf_counter() - t0

            if "hil_state" in info:
                hil_state = info["hil_state"][0]
                if hil_state != prev_hil_state:
                    self.log_info(f"[HIL] state={hil_state}")
                    # Clear stale predictions so the next step starts fresh.
                    self._action_queue = None
                    # When returning to MODEL, request one hold step so the next
                    # policy inference sees a fresh observation of the actual
                    # robot pose after human intervention.
                    if hil_state == "model" and prev_hil_state == "engage":
                        self._pending_model_hold_step = True
                    prev_hil_state = hil_state

            obs = next_obs

            quit_program = bool(np.asarray(info.get("quit_program", False)).any())
            if quit_program:
                self.log_info(
                    "Quit requested by operator (ESC); resetting to initial pose "
                    "and exiting without saving the current episode."
                )
                try:
                    self.env.reset()
                except Exception as e:
                    self.log_warning(
                        f"Failed to reset to initial pose before exit: {e}"
                    )
                break

            done = bool(terminated.any().item() or truncated.any().item())
            if done:
                episodes_done += 1
                progress.update(1)
                if episodes_done < self.num_episodes:
                    self.log_info(
                        f"Episode {episodes_done} saved; continuing to next episode."
                    )
                    obs, _ = self.env.reset()
                    self._action_queue = None
                    self._pending_model_hold_step = False
                else:
                    self.log_info(
                        f"Episode {episodes_done} saved; reached episode limit."
                    )

            elapsed = time.perf_counter() - iter_start
            self.log_info(
                f"[PERF] action={t_action:.3f}s step={t_step:.3f}s total={elapsed:.3f}s"
            )

            if self._target_step_period is not None:
                sleep_for = self._target_step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        progress.close()
        self.env.close()
        self.log_info("SO101 HIL data collection finished.")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="so101_hil_collect",
)
def main(cfg: DictConfig):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = SO101HILCollector.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
