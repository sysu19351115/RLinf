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
"""SO101 bimanual pick-and-place task.

This task is intended to be used with a learned vision-based reward model.
The legacy geometric reward path is kept as a fallback for dummy testing.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rlinf.scheduler import WorkerInfo

from ..so101_env import SO101Env, SO101RobotConfig


@dataclass
class SO101PickAndPlaceConfig(SO101RobotConfig):
    """Configuration for :class:`SO101PickAndPlaceEnv`."""

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Target EEF pose placeholder for geometric reward."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
    )
    """Per-axis success tolerances."""

    use_dense_reward: bool = True
    """Use distance-based dense reward."""

    max_num_steps: int = 200
    """Episode truncation horizon."""


class SO101PickAndPlaceEnv(SO101Env):
    """SO101 bimanual pick-and-place task.

    Inherits all observation and action spaces from :class:`SO101Env`.
    """

    def __init__(
        self,
        override_cfg: Optional[dict] = None,
        worker_info: Optional[WorkerInfo] = None,
        hardware_info=None,
        env_idx: int = 0,
        env_cfg=None,
        **kwargs,
    ):
        super().__init__(
            config=SO101PickAndPlaceConfig(),
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=env_cfg,
            **kwargs,
        )
