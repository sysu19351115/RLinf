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

"""Hybrid continuous/discrete residual actor and Q ensemble.

The actor outputs, for every step of a nominal chunk, a tanh-Gaussian arm
residual (normalized ``[-1, 1]^6``) and a categorical gripper correction
(``KEEP / FORCE_CLOSE / FORCE_OPEN``). The critic consumes the whole chunk
residual and produces a scalar chunk-level Q per ensemble head. Actor and
critic encoders are strictly separate so the two optimizers never share a
parameter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

GRIPPER_NUM_MODES = 3


def _init_linear(module: nn.Linear, scale: float = 1.0) -> None:
    nn.init.orthogonal_(module.weight, gain=scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _ImageEncoder(nn.Module):
    """Small CNN over the (single) USB camera frame."""

    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = hidden

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).flatten(1)


class _MlpEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _tanh_gaussian_log_prob(
    pre_activation: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of a tanh-squashed Gaussian sample."""
    std = log_std.exp().clamp_min(1e-6)
    gaussian_log_prob = (
        -0.5 * ((pre_activation - mean) / std) ** 2
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )
    # tanh correction: d tanh(x)/dx = 1 - tanh^2(x)
    correction = torch.log(1.0 - sample**2 + 1e-6)
    return (gaussian_log_prob - correction).sum(dim=-1)


class ResidualDobotActor(nn.Module):
    """Per-step hybrid residual actor for one nominal chunk."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        proprio_dim: int = 8,
        nominal_dim: int = 8,
        chunk_len: int = 10,
        arm_dim: int = 6,
        hidden: int = 256,
        init_log_std: float = -4.0,
        positional_embed: int = 16,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.arm_dim = arm_dim
        self.image_encoder = _ImageEncoder(image_channels, hidden // 4)
        self.proprio_encoder = _MlpEncoder(proprio_dim, hidden // 2, hidden // 2)
        self.nominal_encoder = _MlpEncoder(nominal_dim * chunk_len, hidden, hidden // 2)
        self.positional_embedding = nn.Parameter(
            torch.randn(chunk_len, positional_embed) * 0.02
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                self.image_encoder.out_dim
                + hidden // 2
                + hidden // 2
                + positional_embed,
                hidden,
            ),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.arm_head = nn.Linear(hidden, arm_dim)
        self.log_std = nn.Parameter(
            torch.full((arm_dim,), init_log_std, dtype=torch.float32)
        )
        self.gripper_head = nn.Linear(hidden, GRIPPER_NUM_MODES)
        # Zero-initialized mean so the first policy is "keep nominal".
        nn.init.zeros_(self.arm_head.weight)
        nn.init.zeros_(self.arm_head.bias)
        nn.init.zeros_(self.gripper_head.weight)
        # KEEP-dominant prior so base-only phases never randomly force the
        # gripper; identical on learner and rollout (initial-sync safe).
        nn.init.zeros_(self.gripper_head.bias)
        with torch.no_grad():
            self.gripper_head.bias[0] = 2.0

    def _features(
        self, images: torch.Tensor, proprio: torch.Tensor, nominal: torch.Tensor
    ) -> torch.Tensor:
        batch = images.shape[0]
        image_feat = self.image_encoder(images)
        proprio_feat = self.proprio_encoder(proprio)
        nominal_feat = self.nominal_encoder(nominal.flatten(1))
        pos = self.positional_embedding.unsqueeze(0).expand(batch, -1, -1)
        per_step = torch.cat(
            [
                image_feat.unsqueeze(1).expand(-1, self.chunk_len, -1),
                proprio_feat.unsqueeze(1).expand(-1, self.chunk_len, -1),
                nominal_feat.unsqueeze(1).expand(-1, self.chunk_len, -1),
                pos,
            ],
            dim=-1,
        )
        return self.fusion(per_step)

    def forward(
        self, images: torch.Tensor, proprio: torch.Tensor, nominal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mean, log_std, gripper_logits)`` with ``[B, H, ...]``."""
        features = self._features(images, proprio, nominal)
        mean = self.arm_head(features)
        log_std = self.log_std.expand_as(mean)
        gripper_logits = self.gripper_head(features)
        return mean, log_std, gripper_logits

    def sample(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        nominal: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample per-step arm residuals and gripper modes.

        Returns:
            ``(u_norm, gripper_mode, log_prob, entropy)``; ``u_norm`` is
            tanh-squashed into ``[-1, 1]``.
        """
        mean, log_std, gripper_logits = self.forward(images, proprio, nominal)
        if deterministic:
            pre_activation = mean
            u_norm = torch.tanh(mean)
        else:
            std = log_std.exp().clamp_min(1e-6)
            noise = torch.randn_like(mean)
            pre_activation = mean + std * noise
            u_norm = torch.tanh(pre_activation)

        arm_log_prob = _tanh_gaussian_log_prob(pre_activation, mean, log_std, u_norm)
        gripper_probs = torch.softmax(gripper_logits, dim=-1)
        gripper_mode = (
            torch.argmax(gripper_logits, dim=-1)
            if deterministic
            else torch.distributions.Categorical(gripper_probs).sample()
        )
        gripper_log_prob = (
            torch.log_softmax(gripper_logits, dim=-1)
            .gather(-1, gripper_mode.unsqueeze(-1))
            .squeeze(-1)
        )
        log_prob = arm_log_prob + gripper_log_prob
        entropy = -torch.distributions.Categorical(gripper_probs).entropy()
        return u_norm, gripper_mode, log_prob, entropy


class ResidualDobotCritic(nn.Module):
    """Chunk-level Q ensemble with LayerNorm MLP heads."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        proprio_dim: int = 8,
        nominal_dim: int = 8,
        chunk_len: int = 10,
        arm_dim: int = 6,
        num_q_heads: int = 10,
        hidden: int = 256,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.image_encoder = _ImageEncoder(image_channels, hidden // 4)
        self.proprio_encoder = _MlpEncoder(proprio_dim, hidden // 2, hidden // 2)
        self.nominal_encoder = _MlpEncoder(nominal_dim * chunk_len, hidden, hidden // 2)
        action_dim = arm_dim * chunk_len + GRIPPER_NUM_MODES * chunk_len + chunk_len
        self.action_encoder = _MlpEncoder(action_dim, hidden, hidden // 2)
        in_dim = self.image_encoder.out_dim + hidden // 2 + hidden // 2 + hidden // 2
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(num_q_heads)
            ]
        )

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        nominal: torch.Tensor,
        arm_residual: torch.Tensor,
        gripper_one_hot: torch.Tensor,
        executed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q ``[B, num_q_heads]``."""
        image_feat = self.image_encoder(images)
        proprio_feat = self.proprio_encoder(proprio)
        nominal_feat = self.nominal_encoder(nominal.flatten(1))
        action_feat = self.action_encoder(
            torch.cat(
                [
                    arm_residual.flatten(1),
                    gripper_one_hot.flatten(1),
                    executed_mask.flatten(1).to(arm_residual.dtype),
                ],
                dim=-1,
            )
        )
        features = torch.cat(
            [image_feat, proprio_feat, nominal_feat, action_feat], dim=-1
        )
        return torch.stack([head(features).squeeze(-1) for head in self.heads], dim=-1)
