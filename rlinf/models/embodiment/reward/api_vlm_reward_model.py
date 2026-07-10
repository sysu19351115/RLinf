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
"""Remote API-based VLM reward model (OpenAI-compatible, Moonshot by default)."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import numpy as np
import requests
import torch
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)

logger = logging.getLogger(__name__)


class APIVLMRewardModel(BaseRewardModel):
    """A frozen VLM reward model backed by a remote vision-language API.

    This implementation targets Moonshot's OpenAI-compatible chat-completion API
    by default, but can be pointed at any provider using the same message format
    by setting ``api_url`` and ``model_name`` in config.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.api_url: str = cfg.get("api_url", "https://api.moonshot.cn/v1/chat/completions")
        self.model_name: str = cfg.get("model_name", "moonshot-v1-8k-vision-preview")
        self.api_key_env: str = cfg.get("api_key_env", "MOONSHOT_API_KEY")
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key for reward model is not set. "
                f"Please set the environment variable {self.api_key_env}."
            )

        self.max_retries: int = int(cfg.get("max_retries", 3))
        self.timeout: float = float(cfg.get("timeout", 60.0))
        self.system_prompt: Optional[str] = cfg.get("system_prompt", None)

        temperature = float(cfg.get("temperature", 1.0))
        # Moonshot's kimi-k2.6 only accepts temperature == 1.0.
        if temperature != 1.0:
            logger.warning(
                "Moonshot model %s only supports temperature=1.0; "
                "clamping configured temperature %.2f to 1.0.",
                self.model_name,
                temperature,
            )
            temperature = 1.0

        self.gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
            "temperature": temperature,
        }

        self.setup_input_builder()
        self.setup_reward_parser()

    def setup_input_builder(self) -> None:
        builder_name = self.cfg.get("input_builder_name", "api_vlm_input_builder")
        self.input_builder = get_input_builder(builder_name)(
            **self.cfg.get("input_builder_params", {}),
            _processor=None,
        )

    def setup_reward_parser(self) -> None:
        parser_name = self.cfg.get("reward_parser_name", "smolvlm_reward_parser")
        self.reward_parser = get_reward_parser(parser_name)(
            **self.cfg.get("reward_parser_params", {})
        )

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "APIVLMRewardModel is a frozen inference-time reward model; "
            "training via forward() is not supported."
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image as a base64 JPEG data URL."""
        buffer = io.BytesIO()
        # Convert to RGB to ensure JPEG compatibility.
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_messages(self, image: Image.Image, prompt_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(image)},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        )
        return messages

    def _call_api(self, messages: list[dict[str, Any]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.gen_kwargs["max_new_tokens"],
            "temperature": self.gen_kwargs["temperature"],
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                last_error = e
                detail = ""
                try:
                    detail = response.text
                except Exception:
                    pass
                logger.warning(
                    "API reward model request failed (attempt %d/%d): %s; response: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                    detail,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
            except Exception as e:
                last_error = e
                logger.warning(
                    "API reward model request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"API reward model failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        prepared = self.input_builder.build_inputs(observations, device=None)
        images_list = prepared.get("images_list") or prepared.get("images", [])
        prompt_texts_list = prepared.get("prompt_texts_list") or prepared.get(
            "prompt_texts", []
        )

        if not images_list or not prompt_texts_list:
            raise ValueError(
                "APIVLMRewardModel received no images or prompts from input builder."
            )

        outputs: list[str] = []
        for images, prompt_texts in zip(images_list, prompt_texts_list):
            # Each sample may contain multiple images; we use the first image for
            # the reward model, consistent with VLMRewardModel behavior.
            if not images:
                raise ValueError("No images available for API reward model inference.")
            image = images[0]
            prompt_text = prompt_texts[0] if prompt_texts else ""
            messages = self._build_messages(image, prompt_text)
            output = self._call_api(messages)
            outputs.append(output)

        rewards = self.reward_parser.parse_rewards(outputs)
        return rewards
