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
"""Data configuration for the SO101 bimanual robot with pi0.5.

The SO101 uses 12-dim absolute position targets:
  [left 5 joints + gripper, right 5 joints + gripper].

During training, joint targets are converted to deltas while the grippers
stay absolute, matching the OpenPI ``LeRobotSO101DataConfig`` pattern.
"""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


@dataclasses.dataclass(frozen=True)
class SO101DataConfig(DataConfigFactory):
    """Data configuration for SO101 bimanual datasets.

    State layout: ``[left 5 joints + gripper, right 5 joints + gripper]`` — 12-dim.
    Action layout: same as state (absolute).  During training the pipeline
    converts actions to delta (joints only) via ``DeltaActions`` /
    ``AbsoluteActions`` so the model learns relative displacements.
    """

    default_prompt: str | None = None

    def generate_observations(
        self,
        image: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> dict:
        """Creates an input example for the SO101 policy."""
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
            inputs=[so101_policy.SO101PolicyInputs(model_type=model_config.model_type)],
            outputs=[so101_policy.SO101PolicyOutputs()],
        )

        # Convert absolute joint targets to deltas for training.
        # delta_action_mask: first 5 and last 5 dims (joints) are delta,
        # the 6th and 12th dims (grippers) stay absolute.
        delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
