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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers import apply_so101_wrappers

from .pick_and_place import SO101PickAndPlaceEnv as SO101PickAndPlaceEnv


def create_so101_pick_and_place_env(
    override_cfg: dict[str, Any] | None = None,
    worker_info: Any = None,
    hardware_info: Any = None,
    env_idx: int = 0,
    env_cfg: Mapping[str, Any] | None = None,
) -> gym.Env:
    """Factory for the SO101 pick-and-place task with optional HIL wrappers.

    All arguments except ``override_cfg`` are optional so the factory can also
    be used directly via ``gym.make`` for testing and verification.
    """
    env = SO101PickAndPlaceEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
        env_cfg=env_cfg,
    )
    return apply_so101_wrappers(env, env_cfg or {})


register(
    id="SO101PickAndPlaceEnv-v1",
    entry_point="rlinf.envs.realworld.so101.tasks:create_so101_pick_and_place_env",
)
