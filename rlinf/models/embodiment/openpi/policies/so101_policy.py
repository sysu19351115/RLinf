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
"""Policy transforms for the SO101 bimanual robot with pi0.5 model.

Observation layout (from SO101Env):
  - state: ``[left 5 joints + gripper, right 5 joints + gripper]`` — 12-dim.
  - frames: ``cam_high`` (left global), ``cam_left_wrist``, ``cam_right_wrist``.

The policy maps these into OpenPI's standard 3-camera format with the same
names expected by the OpenPI SO101 data transform.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 bimanual policy."""
    return {
        "observation/state": np.random.rand(12).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object and place it into the box",
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
class SO101PolicyInputs(transforms.DataTransformFn):
    """Converts RLinf SO101 observations to OpenPI model input format.

    Expected input keys (from the RLinf observation pipeline):
        - ``observation/image``: global/wrist camera image, uint8 ``[H, W, C]``
        - ``observation/state``: 12-dim joint state
        - ``prompt``: task description string (optional)

    The ``observation`` dict from the OpenPI action model contains the main
    image under ``observation/image`` and state under ``observation/state``.
    The extra-view images are passed separately as ``observation/extra_view_image``
    and are stacked into the remaining camera slots.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"]).astype(np.float32)

        # Default masks: base image present, others absent unless provided.
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": np.zeros_like(base_image),
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_,
            "right_wrist_0_rgb": np.False_,
        }

        if "observation/extra_view_image" in data:
            extra = data["observation/extra_view_image"]
            if isinstance(extra, (list, tuple)):
                extra = [np.asarray(x) for x in extra]
            else:
                extra = np.asarray(extra)
                # The env stacks multiple extra-view images along the first axis,
                # giving shape [N, H, W, C]. Split into separate [H, W, C] images.
                if extra.ndim == 4 and extra.shape[0] > 1:
                    extra = [extra[i] for i in range(extra.shape[0])]
                else:
                    extra = [extra]
            if len(extra) >= 1:
                images["left_wrist_0_rgb"] = _parse_image(extra[0])
                image_masks["left_wrist_0_rgb"] = np.True_
            if len(extra) >= 2:
                images["right_wrist_0_rgb"] = _parse_image(extra[1])
                image_masks["right_wrist_0_rgb"] = np.True_

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101PolicyOutputs(transforms.DataTransformFn):
    """Extracts SO101 actions from the OpenPI model output.

    The model predicts a padded action tensor; this transform slices the
    first 12 dimensions (left 5 joints + gripper, right 5 joints + gripper).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :12])}
