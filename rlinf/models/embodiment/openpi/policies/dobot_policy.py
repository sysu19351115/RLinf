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
"""Policy transforms for the Dobot CR5AF single-arm robot with pi0.5 model.

Supports **two mutually exclusive modes** (selected by ``state_dim``):

- **joint mode** (``state_dim=7``): state/action ``[j1..j6 rad, gripper]``.
  Corresponds to ``pi05_dobot_joint``.

- **pose mode** (``state_dim=8``): state/action
  ``[x,y,z m, qw,qx,qy,qz, gripper]`` + ``prev_state``.
  Corresponds to ``pi05_dobot_pose``.

Key alignment with the RLinf RealWorldEnv pipeline:

At **inference**, :class:`RealWorldEnv` routes ``main_image_key``
(``cam_left_wrist`` by default) into ``main_images``, and the remaining
cameras (``cam_high``) into ``extra_view_images``. The
``obs_processor`` in ``openpi_action_model.py`` then maps:
    - ``main_images`` → ``observation/image``
    - ``extra_view_images`` → ``observation/extra_view_image``
    - ``states`` → ``observation/state``
    - ``task_descriptions`` → ``prompt``

It **never** produces ``observation/wrist_image`` (that key only exists for
sim environments). Therefore this transform reads:
    - ``observation/image`` → ``left_wrist_0_rgb`` (the wrist camera)
    - ``observation/extra_view_image`` → ``base_0_rgb`` (cam_high, optional)
    - ``observation/prev_state`` → forwarded for ``DeltaPose`` / ``AbsolutePose``

At **training**, the :class:`DobotDataConfig` repack produces the same flat
keys (``observation/image``, ``observation/state``, etc.) from the LeRobot
dataset's nested columns.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms


def make_dobot_example(state_dim: int = 7) -> dict:
    """Creates a random input example for the Dobot single-arm policy.

    Args:
        state_dim: 7 (joint) or 8 (pose).

    Returns:
        A dict matching the post-repack RLinf observation layout. ``prev_state``
        is only included for pose mode (``state_dim == 8``) since it is required
        by :class:`DeltaPose` / :class:`AbsolutePose` and unused by joint mode.
    """
    example = {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(state_dim).astype(np.float32),
        "actions": np.ones((15, state_dim), dtype=np.float32),
        "prompt": "pick up the plug and plug it into the socket",
    }
    if state_dim == 8:
        example["observation/prev_state"] = np.random.rand(state_dim).astype(np.float32)
    return example


def _parse_image(image) -> np.ndarray:
    """Parse an image to uint8 ``[H, W, C]`` format.

    Accepts HWC uint8 (inference) or CHW float32 in ``[0, 1]`` (training).
    """
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_extra_view_image(images) -> np.ndarray:
    """Parse ``extra_view_images`` which may be [N, H, W, C] or [N, n_extra, H, W, C].

    Returns the first extra image as HWC uint8.
    """
    images = np.asarray(images)
    if images.ndim == 5:  # [N, n_extra, H, W, C]
        images = images[0, 0]  # first env, first extra
    elif images.ndim == 4:  # [N, H, W, C] or [n_extra, H, W, C]
        images = images[0]
    elif images.ndim == 3:  # [H, W, C]
        pass
    return _parse_image(images)


@dataclasses.dataclass(frozen=True)
class DobotPolicyInputs(transforms.DataTransformFn):
    """Converts RLinf Dobot observations to OpenPI model input format.

    Args:
        state_dim: 7 (joint mode) or 8 (pose mode). Controls state validation
            and prev_state forwarding.
    """

    state_dim: int = 7

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state with {self.state_dim} dims, got {state.shape[-1]}."
            )

        # observation/image = main_images = cam_left_wrist (the wrist camera).
        # This is the primary observation and always present.
        wrist_image = _parse_image(data["observation/image"])

        # observation/extra_view_image = extra_view_images = cam_high (optional).
        # When absent, zero-fill and mask out the base_0_rgb slot.
        has_high = (
            "observation/extra_view_image" in data
            and data["observation/extra_view_image"] is not None
        )
        high_image = (
            _parse_extra_view_image(data["observation/extra_view_image"])
            if has_high
            else None
        )
        zero_image = np.zeros_like(wrist_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": high_image if high_image is not None else zero_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": zero_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if high_image is not None else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Pose mode: forward prev_state (required by DeltaPose / AbsolutePose).
        if (
            "observation/prev_state" in data
            and data["observation/prev_state"] is not None
        ):
            inputs["prev_state"] = np.asarray(
                data["observation/prev_state"], dtype=np.float32
            )

        # Actions only present during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"]).copy()

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        # RTC inference (ActionChunkBroker_RTC) carries prev_action/prev_state;
        # must be forwarded for DeltaActions_Prev / DeltaPose.
        if "rtc_obs" in data:
            inputs["rtc_obs"] = data["rtc_obs"].copy()

        return inputs


@dataclasses.dataclass(frozen=True)
class DobotPolicyOutputs(transforms.DataTransformFn):
    """Extracts Dobot actions from the OpenPI model output.

    The model predicts a padded action tensor; this transform slices the first
    ``action_dim`` dimensions.

    Args:
        action_dim: 7 (joint) or 8 (pose).
    """

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
