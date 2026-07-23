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
"""Data configuration for the Dobot CR5AF single-arm robot with pi0.5.

Two mutually exclusive variants, selected by ``use_pose``:

- **joint variant** (``use_pose=False``, default): 7-dim
  ``[j1..j6 rad, gripper]`` state/action. Joints are converted to deltas via
  the upstream ``DeltaActions`` / ``AbsoluteActions`` (naive elementwise
  subtraction); the gripper stays absolute. Corresponds to
  ``pi05_dobot_joint``.

- **pose variant** (``use_pose=True``): 8-dim
  ``[x,y,z m, qw,qx,qy,qz, gripper]`` state/action. The pose block is
  converted to **SE(3) relative poses** via our custom
  :class:`~rlinf.models.embodiment.openpi.pose_transforms.DeltaPose` /
  :class:`~rlinf.models.embodiment.openpi.pose_transforms.AbsolutePose`
  (4×4 homogeneous matrices, NOT naive subtraction) using the dataset's
  ``observation.prev_state`` (gripper stays absolute). Corresponds to
  ``pi05_dobot_pose``.

Repack maps LeRobot dataset keys to flat keys that match
:class:`~rlinf.models.embodiment.openpi.policies.dobot_policy.DobotPolicyInputs`:
    - ``observation.images.cam_left_wrist`` → ``observation/image``
    - ``observation.state`` → ``observation/state``
    - ``observation.prev_state`` → ``observation/prev_state``  (pose only)
    - ``action`` → ``actions``
    - ``prompt`` → ``prompt``
"""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dobot_policy
from rlinf.models.embodiment.openpi.pose_transforms import (
    AbsolutePose,
    DeltaActions_Prev,
    DeltaPose,
)


@dataclasses.dataclass(frozen=True)
class DobotDataConfig(DataConfigFactory):
    """Data configuration for Dobot CR5AF single-arm datasets.

    Args:
        use_pose: ``False`` (joint variant, 7-dim) or ``True`` (pose variant,
            8-dim). The two are mutually exclusive with ``use_delta_joint_actions``.
        use_delta_joint_actions: Joint-mode delta actions (default True). Must
            be ``False`` when ``use_pose=True``.
        default_prompt: Optional default language prompt.
    """

    use_pose: bool = False
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    def generate_observations(
        self,
        image: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> dict:
        """Creates an input example for the Dobot policy (joint mode)."""
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if self.use_pose and self.use_delta_joint_actions:
            raise ValueError("use_pose and use_delta_joint_actions cannot both be True")

        state_dim = 8 if self.use_pose else 7

        # ── Repack: LeRobot nested keys → flat keys matching DobotPolicyInputs ─
        repack_structure = {
            "observation/image": "observation.images.cam_left_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.use_pose:
            # DeltaPose needs the previous absolute pose, stored per-frame.
            repack_structure["observation/prev_state"] = "observation.prev_state"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        # ── Data transforms: policy I/O + delta encoding ───────────────────────
        data_transforms = _transforms.Group(
            inputs=[
                dobot_policy.DobotPolicyInputs(state_dim=state_dim),
            ],
            outputs=[dobot_policy.DobotPolicyOutputs(action_dim=state_dim)],
        )

        if self.use_pose:
            # Pose mode: SE(3) delta via homogeneous matrices (custom transforms).
            # mask = (T,T,T,T,T,T,T,F) — 7 pose dims delta, gripper absolute.
            delta_pose_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[DeltaPose(delta_pose_mask)],
                outputs=[AbsolutePose(delta_pose_mask)],
            )
        else:
            # Joint mode: naive elementwise delta (upstream transforms).
            # mask = (T,T,T,T,T,T,F) — 6 joint dims delta, gripper absolute.
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.DeltaActions(delta_action_mask),
                    DeltaActions_Prev(delta_action_mask),
                ],
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
            action_sequence_keys=("action",),
        )
