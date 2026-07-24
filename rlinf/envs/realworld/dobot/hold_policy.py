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

"""Hold policy for Dobot HIL data collection.

Returns the current state as the action, repeated for the full action chunk.
Used in ``policy_mode='hold'`` for real-robot safety verification without
loading a model, and in ``policy_mode='dummy'`` for software pipeline testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class HoldPolicy:
    """Policy that always returns the current state as the action.

    The action is the current 8-dim pose state repeated for the full action
    chunk. This is safe because it commands the robot to stay where it is.

    Args:
        action_dim: Dimension of the action (8 for Dobot Cartesian pose).
        action_chunk: Number of future actions in the chunk.
    """

    def __init__(self, action_dim: int = 8, action_chunk: int = 1):
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = env_obs["states"]
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = np.asarray(state)
        bsize = int(state_np.shape[0])
        # Repeat the current state across the action chunk.
        actions = np.broadcast_to(
            state_np[:, None, :], (bsize, self.action_chunk, self.action_dim)
        ).copy()
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        return actions_t, {"forward_inputs": {"action": actions_t.reshape(bsize, -1)}}
