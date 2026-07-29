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

"""Unified deterministic-rollout and masked-SFT model for HG-DAgger."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.utils.freeze import (
    freeze_paligemma_vlm,
)


class OpenPiPytorchDaggerActionModel(OpenPiPytorchEvalActionModel):
    """Deterministic Pi0 rollout plus masked online flow-matching SFT."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            preserve_dagger_anchor_inputs=True,
        )

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT training; "
                f"got {forward_type!r}."
            )
        return self.sft_forward(**kwargs)

    def freeze_vlm(self) -> int:
        """Freeze the shared PaliGemma side before FSDP wrapping."""
        return freeze_paligemma_vlm(self.model)

    @staticmethod
    def _remove_replay_step_dim(value: Any, *, batch_size: int) -> Any:
        """Remove only the replay trajectory dimension ``[B, 1, ...]``."""
        if not hasattr(value, "shape") or len(value.shape) < 2:
            return value
        if value.shape[0] != batch_size:
            raise ValueError(
                "DAgger replay tensors must share the action batch dimension; "
                f"got first dim {value.shape[0]}, expected {batch_size}."
            )
        return value[:, 0] if value.shape[1] == 1 else value

    def prepare_dagger_sft_batch(
        self, batch: dict[str, Any], loss_scope: str = "full_window"
    ) -> dict[str, Any]:
        """Transform one HG-DAgger replay batch into model training inputs."""
        if "action" not in batch:
            raise ValueError("DAgger replay forward_inputs are missing 'action'.")
        raw_actions = batch["action"]
        if not torch.is_tensor(raw_actions):
            raw_actions = torch.as_tensor(raw_actions)
        if raw_actions.ndim < 2:
            raise ValueError(
                "DAgger action must include a batch dimension and flattened window; "
                f"got shape {tuple(raw_actions.shape)}."
            )
        batch_size = raw_actions.shape[0]
        action_horizon = int(self.model.action_horizon)
        expected = batch_size * action_horizon * self.action_env_dim
        if raw_actions.numel() != expected:
            raise ValueError(
                f"DAgger action has {raw_actions.numel()} elements, expected "
                f"{expected} (batch={batch_size}, action_horizon={action_horizon}, "
                f"action_env_dim={self.action_env_dim})."
            )
        actions = raw_actions.reshape(batch_size, action_horizon, self.action_env_dim)
        if not torch.isfinite(actions).all():
            raise ValueError("DAgger replay actions contain NaN or Inf.")

        obs_dict = {
            key: self._remove_replay_step_dim(value, batch_size=batch_size)
            for key, value in batch.items()
            if key.startswith("observation/")
        }
        if "observation/prev_state" not in obs_dict:
            raise ValueError(
                "Dobot DAgger replay is missing 'observation/prev_state'; "
                "relative pose actions cannot be reconstructed safely."
            )
        for key in ("tokenized_prompt", "tokenized_prompt_mask"):
            if key in batch:
                obs_dict[key] = self._remove_replay_step_dim(
                    batch[key], batch_size=batch_size
                )

        obs_dict["actions"] = actions
        obs_dict["prompt"] = ["empty"] * batch_size
        processed = self.input_transform(obs_dict, transpose=False)
        for key in ("tokenized_prompt", "tokenized_prompt_mask"):
            if key in obs_dict:
                processed[key] = obs_dict[key]
        if "actions" not in processed:
            raise ValueError(
                "OpenPI input transforms dropped DAgger actions; the configured "
                "data transform is incompatible with online DAgger training."
            )
        model_actions = processed.pop("actions")
        if not torch.is_tensor(model_actions):
            model_actions = torch.as_tensor(model_actions)
        expected_shape = (
            batch_size,
            action_horizon,
            int(self.model.action_dim),
        )
        if tuple(model_actions.shape) != expected_shape:
            raise ValueError(
                "Transformed DAgger actions must have shape "
                f"{expected_shape}; got {tuple(model_actions.shape)}."
            )
        if not torch.isfinite(model_actions).all():
            raise ValueError("Transformed DAgger actions contain NaN or Inf.")

        if loss_scope == "human_only":
            if "human_action_mask" not in batch:
                raise ValueError(
                    "DAgger loss_scope='human_only' requires "
                    "human_action_mask in replay forward_inputs."
                )
            loss_mask = self._remove_replay_step_dim(
                batch["human_action_mask"], batch_size=batch_size
            )
            loss_mask = torch.as_tensor(loss_mask, dtype=torch.bool)
        elif loss_scope == "full_window":
            loss_mask = torch.ones(batch_size, action_horizon, dtype=torch.bool)
        else:
            raise ValueError(
                f"Unsupported DAgger loss_scope {loss_scope!r}; "
                "expected 'human_only' or 'full_window'."
            )
        if tuple(loss_mask.shape) != (batch_size, action_horizon):
            raise ValueError(
                "DAgger loss mask must have shape "
                f"[{batch_size}, {action_horizon}]; got {tuple(loss_mask.shape)}."
            )
        if not loss_mask.any():
            raise ValueError("DAgger loss mask selects no human action steps.")

        return {
            "observation": self._observation_dict_to_device(processed),
            "actions": model_actions.to(
                device=self.device, dtype=torch.float32
            ).contiguous(),
            "loss_mask": loss_mask.to(device=self.device).contiguous(),
        }

    def sft_forward(
        self,
        data: dict[str, Any],
        use_action_chunk_loss: bool = True,
    ) -> torch.Tensor:
        """Compute flow-matching loss over human steps and environment dims."""
        if not use_action_chunk_loss:
            raise ValueError(
                "OpenPI PyTorch DAgger requires action-chunk-aware loss reduction."
            )
        required = {"observation", "actions", "loss_mask"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"DAgger SFT batch is missing keys: {sorted(missing)}.")

        errors = self.model.compute_loss(
            data["observation"],
            data["actions"],
            train=True,
            reduce_action_dim=False,
        )
        expected_prefix = (
            data["actions"].shape[0],
            int(self.model.action_horizon),
        )
        if errors.ndim != 3 or tuple(errors.shape[:2]) != expected_prefix:
            raise ValueError(
                "Pi0 DAgger loss must have shape [B, action_horizon, D]; "
                f"got {tuple(errors.shape)}."
            )
        if errors.shape[-1] < self.action_env_dim:
            raise ValueError(
                f"Pi0 loss action dim {errors.shape[-1]} is smaller than "
                f"environment action dim {self.action_env_dim}."
            )
        errors = errors[:, :, : self.action_env_dim]
        mask = torch.as_tensor(data["loss_mask"], device=errors.device)
        if tuple(mask.shape) != tuple(errors.shape[:2]):
            raise ValueError(
                "DAgger loss mask must match [B, action_horizon]; "
                f"got {tuple(mask.shape)} for loss {tuple(errors.shape)}."
            )
        selected_steps = mask.to(dtype=errors.dtype).sum()
        if selected_steps.item() <= 0:
            raise ValueError("DAgger loss mask selects no human action steps.")
        loss = (errors * mask.to(dtype=errors.dtype).unsqueeze(-1)).sum() / (
            selected_steps * self.action_env_dim
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError(f"DAgger produced an invalid scalar loss: {loss}.")
        return loss
