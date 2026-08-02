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

"""Lightweight residual policy for the frozen-Pi0.5 HIL-RLPD stack."""

import torch

from rlinf.models.embodiment.residual_dobot_policy.policy import (
    GRIPPER_NUM_MODES,
    ResidualDobotActor,
    ResidualDobotCritic,
)


class ResidualBasePolicyWrapper(torch.nn.Module):
    """Wrap the frozen Pi0.5 base so the HF worker treats it as the model."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def predict_action_batch(self, env_obs, **kwargs):
        return self.base_model.predict_action_batch(env_obs, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.base_model.load_state_dict(*args, **kwargs)

    def set_global_step(self, *args, **kwargs):
        if hasattr(self.base_model, "set_global_step"):
            return self.base_model.set_global_step(*args, **kwargs)
        return None

    def eval(self):
        self.base_model.eval()
        return self

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return self


__all__ = [
    "GRIPPER_NUM_MODES",
    "ResidualDobotActor",
    "ResidualDobotCritic",
    "ResidualBasePolicyWrapper",
]
