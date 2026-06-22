# Copyright 2025 The RLinf Authors.
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
"""Policy transforms for the gym_aloha environment with the pi0 model."""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.policies.aloha_policy import (
    _decode_state,
    _encode_actions,
    _encode_actions_inv,
)


def make_gym_aloha_example() -> dict:
    """Creates a random input example for the GymAloha policy."""
    return {
        "observation/image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(14).astype(np.float32),
        "prompt": "Transfer cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse an image to uint8 [H, W, C] format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GymAlohaInputs(transforms.DataTransformFn):
    """Converts gym_aloha observations to the pi0 model input format.

    Expected input keys (from the RLinf observation pipeline):
        - ``observation/image``: base camera image, uint8 [H, W, C]
        - ``observation/state``: 14-dim joint state
        - ``prompt``: task description string

    Produces the standard pi0 model input format with a single
    ``base_0_rgb`` image and zero-padded wrist images.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"])

        state = _decode_state(state, adapt_to_pi=True)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=True)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GymAlohaOutputs(transforms.DataTransformFn):
    """Extracts gym_aloha actions from the pi0 model output.

    The pi0 model predicts 32 action dimensions internally, but the
    Aloha robot only needs the first 14 (left arm 6 + gripper 1
    + right arm 6 + gripper 1). This transform slices the action
    tensor and applies the required joint-flip and gripper
    conversion.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=True)}
