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

"""RebotArm Pick-and-Place task.

Targets a single TCP pose ``target_ee_pose``.  The robot must move its
end-effector within ``reward_threshold`` of that pose.  If
``use_dense_reward`` is enabled, a continuous exponential distance-based
reward is emitted; otherwise the reward is sparse (0/1).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rlinf.scheduler import WorkerInfo

from ..rebot_env import RebotArmEnv, RebotArmRobotConfig


@dataclass
class RebotArmPickAndPlaceConfig(RebotArmRobotConfig):
    """Configuration for :class:`RebotArmPickAndPlaceEnv`."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.3, -0.2, 0.15, 3.14, 0.0, 0.0])
    )
    """Target EEF pose for the pick-and-place task."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
    )
    """Per-axis success tolerances ``[x, y, z, rx, ry, rz]``."""

    use_dense_reward: bool = True
    """Use distance-based dense reward."""

    max_num_steps: int = 200
    """Episode truncation horizon."""


class RebotArmPickAndPlaceEnv(RebotArmEnv):
    """RebotArm pick-and-place task.

    The task requires the robot to move its end-effector to the
    ``target_ee_pose`` within the configured ``reward_threshold``
    tolerance.

    Inherits all observation and action spaces from :class:`RebotArmEnv`.
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
            config=RebotArmPickAndPlaceConfig(),
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=env_cfg,
            **kwargs,
        )
