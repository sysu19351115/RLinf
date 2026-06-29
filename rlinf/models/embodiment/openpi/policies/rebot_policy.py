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
"""Policy transforms for the reBot Arm B601 single-arm robot with pi0.5 model.

Observation layout (from RebotArmEnv):
  - state: ``[j1, ..., j6, gripper]`` — 7‑dim absolute joint positions.
  - frames: ``wrist_1`` — wrist camera image ``(H, W, 3)`` uint8.

The policy maps the single wrist camera to ``left_wrist_0_rgb`` and fills
the remaining camera slots with zero images, matching OpenPI's 3‑camera
format.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_rebot_example() -> dict:
    """Creates a random input example for the reBot single-arm policy."""
    return {
        "observation/image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(7).astype(np.float32),
        "prompt": "pick objects and place at the target location",
    }


def _parse_image(image) -> np.ndarray:
    """Parse an image to uint8 ``[H, W, C]`` format."""
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RebotPolicyInputs(transforms.DataTransformFn):
    """Converts RLinf RebotArm observations to OpenPI model input format.

    Expected input keys (from the RLinf observation pipeline):
        - ``observation/image``: wrist camera image, uint8 ``[H, W, C]``
        - ``observation/state``: 7‑dim joint state ``[j1..j6, gripper]``
        - ``prompt``: task description string (optional)

    Produces the standard OpenPI 3‑camera format with the wrist image
    mapped to ``left_wrist_0_rgb`` and the remaining slots zero‑filled.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"]).astype(np.float32)

        zero_image = np.zeros_like(wrist_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": zero_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": zero_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RebotPolicyOutputs(transforms.DataTransformFn):
    """Extracts reBot actions from the OpenPI model output.

    The model predicts a padded action tensor; this transform slices
    the first 7 dimensions (6 joints + 1 gripper).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
