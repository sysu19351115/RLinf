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
"""Human-in-the-loop reward model for embodied RL."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Optional

import numpy as np
import requests
import torch
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

logger = logging.getLogger(__name__)


class HumanRewardModel(BaseRewardModel):
    """A reward model that asks a human to score each episode via a web UI.

    The model expects a standalone ``HumanRewardServer`` (see
    ``tmp/human_reward_server.py``) running at ``human_reward_url``. At inference
    time it uploads the final frame and blocks until the human submits a reward.

    This is intended for use with ``reward_mode: terminal`` so the human is only
    asked once per episode.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.human_reward_url: str = cfg.get(
            "human_reward_url", "http://127.0.0.1:12345"
        )
        self.task_description: str = cfg.get("task_description", "")
        self.timeout: float = float(cfg.get("timeout", 600.0))
        self.server_wait_timeout: float = float(
            cfg.get("server_wait_timeout", 60.0)
        )
        self.default_reward: float = float(cfg.get("default_reward", 0.0))
        self._counter = 0

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HumanRewardModel is a frozen inference-time reward model; "
            "training via forward() is not supported."
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image as a base64 JPEG data URL content."""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_first_image(observations: dict[str, Any]) -> Image.Image:
        """Extract the first image from observations as a PIL RGB image."""
        images = observations.get("main_images")
        if images is None:
            raise ValueError(
                "HumanRewardModel expects observations['main_images'] to be present."
            )

        if isinstance(images, torch.Tensor):
            arr = images.detach().cpu().numpy()
        elif isinstance(images, np.ndarray):
            arr = images
        else:
            raise TypeError(f"Unsupported image input type: {type(images)}")

        # Handle batch dimension.
        if arr.ndim == 4:
            arr = arr[0]

        # Handle CHW -> HWC.
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape after preprocessing: {arr.shape}")

        if arr.dtype == np.uint8:
            return Image.fromarray(arr).convert("RGB")
        return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")

    def _submit_episode(self, episode_id: str, image: Image.Image) -> None:
        url = f"{self.human_reward_url.rstrip('/')}/submit_episode"
        payload = {
            "episode_id": episode_id,
            "task": self.task_description,
            "image_base64": self._encode_image(image),
        }
        response = requests.post(url, json=payload, timeout=30.0)
        response.raise_for_status()

    def _poll_reward(self, episode_id: str) -> float:
        url = f"{self.human_reward_url.rstrip('/')}/get_reward"
        deadline = time.time() + self.timeout

        while time.time() < deadline:
            remaining = min(self.server_wait_timeout, deadline - time.time())
            if remaining <= 0:
                break
            try:
                response = requests.get(
                    url,
                    params={"episode_id": episode_id},
                    timeout=remaining + 5.0,
                )
                response.raise_for_status()
                data = response.json()
                source = data.get("source", "unknown")
                reward = float(data["reward"])
                if source != "timeout":
                    logger.info(
                        "Human reward received for episode %s: reward=%.1f (source=%s)",
                        episode_id,
                        reward,
                        source,
                    )
                    return reward
                # Server returned a timeout/empty response; loop again.
            except requests.Timeout:
                pass
            except Exception as e:
                logger.warning("Failed to poll human reward server: %s", e)
                time.sleep(1.0)

        logger.warning(
            "Timed out waiting for human reward for episode %s; returning default %.1f",
            episode_id,
            self.default_reward,
        )
        return self.default_reward

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        image = self._extract_first_image(observations)

        self._counter += 1
        episode_id = f"human_{self._counter}_{int(time.time() * 1000)}"

        logger.info(
            "Requesting human reward for episode %s (timeout=%.0fs)",
            episode_id,
            self.timeout,
        )
        self._submit_episode(episode_id, image)
        reward = self._poll_reward(episode_id)
        return torch.tensor([reward], dtype=torch.float32)
