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
"""Data configuration for the reBot Arm B601 single-arm robot with pi0.5.

The reBot arm uses 6 joint angles + 1 gripper as absolute position targets.
Actions are converted to delta actions for training (joints only, gripper
stays absolute), matching the OpenPI ``LeRobotRebotDataConfig`` pattern.
"""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import rebot_policy


@dataclasses.dataclass(frozen=True)
class RebotDataConfig(DataConfigFactory):
    """Data configuration for reBot Arm single-arm datasets.

    State layout:  ``[j1, ..., j6, gripper]`` — 7‑dim absolute joint positions.
    Action layout: same as state (absolute).  During training the
    pipeline converts actions to delta (joints only) via
    ``DeltaActions(6)`` / ``AbsoluteActions(6)`` so the model learns
    relative displacements.
    """

    default_prompt: str | None = None

    def generate_observations(
        self,
        image: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> dict:
        """Creates an input example for the reBot policy."""
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[rebot_policy.RebotPolicyInputs(model_type=model_config.model_type)],
            outputs=[rebot_policy.RebotPolicyOutputs()],
        )

        # Convert absolute joint targets to deltas for training.
        # delta_action_mask: first 6 dims (joints) are delta,
        # last 1 dim (gripper) stays absolute.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[
                _transforms.DeltaActions(delta_action_mask),
                _transforms.DeltaActions_Prev(delta_action_mask),
            ],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
