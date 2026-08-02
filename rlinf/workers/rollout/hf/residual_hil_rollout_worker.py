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

"""HF rollout worker binding the frozen Pi0.5 outputs to the residual actor."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.residual_hil_rlpd.config import build_residual_codec
from rlinf.algorithms.residual_hil_rlpd.messages import (
    MESSAGE_TYPE_SCALE_ACK,
    build_ack_message,
    parse_scale_message,
)
from rlinf.algorithms.residual_hil_rlpd.rollout import Pi05ResidualComposer
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.models.embodiment.residual_dobot_policy import ResidualDobotActor
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


class ResidualHILRolloutWorker(AsyncMultiStepRolloutWorker):
    """``AsyncMultiStepRolloutWorker`` composing Pi0.5 with the residual actor.

    The inherited path calls the frozen Pi0.5 through
    ``predict_action_batch(env_obs)`` and returns the nominal absolute
    actions; this subclass composes the residual actor on top of them and
    attaches the audit dict the env worker needs for transition finalization.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        rlpd_cfg = cfg.algorithm.residual_hil_rlpd
        self.residual_actor = ResidualDobotActor().to(self.device)
        self.residual_composer = Pi05ResidualComposer(
            residual_actor=self.residual_actor,
            codec=build_residual_codec(cfg),
            device=self.device,
            policy_translation_scale_m=tuple(rlpd_cfg.translation_policy_limit_m),
            policy_rotation_scale_deg=tuple(rlpd_cfg.rotation_policy_limit_deg),
            gripper_max_switches_per_chunk=int(
                rlpd_cfg.gripper_max_switches_per_chunk
            ),
            gripper_min_hold_steps=int(rlpd_cfg.gripper_min_hold_steps),
            gripper_debounce_chunks=int(rlpd_cfg.gripper_debounce_chunks),
        )
        self._last_env_obs: dict[str, torch.Tensor] | None = None
        # Fail-closed: no scale message -> nominal-only (base) execution.
        self._residual_scale = 0.0
        self._gripper_enabled = False
        self._run_id = str(cfg.get("run_id", ""))

    def set_residual_scale(self, scale: float) -> None:
        self._residual_scale = float(scale)
        self._gripper_enabled = False

    @staticmethod
    def _parse_scale_message(message: Any, applied_version: int) -> float:
        """Fail-closed scale update; ``applied_version`` is the weight version
        already applied by ``PatchWeightSyncer`` (scale must bind to it)."""
        scale, _ = parse_scale_message(
            message, applied_version=applied_version
        )
        return scale

    def _predict_rollout_actions(self, env_obs, **kwargs):
        self._last_env_obs = env_obs
        return super()._predict_rollout_actions(env_obs, **kwargs)

    def _build_rollout_result(
        self,
        actions: torch.Tensor,
        result: dict[str, Any],
        *,
        final_obs: dict[str, Any] | None = None,
    ) -> RolloutResult:
        rollout_result = super()._build_rollout_result(
            actions, result, final_obs=final_obs
        )
        if self._last_env_obs is None:
            raise RuntimeError(
                "residual_hil_rlpd rollout lost the env observation; "
                "_predict_rollout_actions must run before result building"
            )
        commanded, audit = self.residual_composer.compose_chunk(
            actions,
            self._last_env_obs,
            residual_scale=self._residual_scale,
            gripper_enabled=self._gripper_enabled,
            deterministic=False,
            policy_version=self.version,
        )
        rollout_result.actions = commanded
        rollout_result.audit_info = audit
        return rollout_result

    async def sync_model_from_actor(self):
        """Feed learner weight patches into the residual actor (not Pi0.5)."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.residual_actor.state_dict(),
                recv=recv_func,
                send=send_func,
            )
        self.version = await self.weight_syncer.apply(self.residual_actor, recv_func)
        try:
            scale_message = await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
            self._residual_scale, self._gripper_enabled = parse_scale_message(
                scale_message,
                applied_version=self.version,
                run_id=self._run_id,
            )
            await self._send_scale_ack()
        except Exception:
            self._logger.warning(
                "[ResidualHIL] Failed to receive residual scale; "
                "staying nominal-only",
            )
            self._residual_scale = 0.0
            self._gripper_enabled = False
        return True

    async def _send_scale_ack(self) -> None:
        """Best-effort ACK of the applied scale (P2-6); never blocks training."""
        try:
            await self.send(
                build_ack_message(
                    MESSAGE_TYPE_SCALE_ACK,
                    self.version,
                    run_id=self._run_id,
                    sender_rank=int(getattr(self, "_rank", 0)),
                ),
                dst_group_name=self.actor_group_name,
                dst_rank=getattr(self, "actor_weight_src_rank", 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
        except Exception:
            self._logger.warning("[ResidualHIL] scale ACK send failed (ignored)")
