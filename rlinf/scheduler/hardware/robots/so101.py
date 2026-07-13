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
"""Hardware configuration and enumeration for the SO101 bimanual robot."""

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)
from .auto_config import RobotAutoConfig


@dataclass
class SO101ArmHWInfo(HardwareInfo):
    """Hardware information for an SO101 bimanual robotic system."""

    config: "SO101ArmConfig"


@Hardware.register()
class SO101ArmRobot(Hardware):
    """Hardware policy for SO101 bimanual robots (LeRobot BiSOFollower)."""

    HW_TYPE = "SO101Arm"

    @classmethod
    def enumerate(
        cls,
        node_rank: int,
        configs: Optional[list["SO101ArmConfig"]] = None,
    ) -> Optional[HardwareResource]:
        """Enumerate SO101 robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware
                resources. None if no SO101 hardware is configured for this
                node.
        """
        assert configs is not None, (
            "SO101Arm hardware requires explicit configurations."
        )
        robot_configs: list["SO101ArmConfig"] = []
        for config in configs:
            if isinstance(config, SO101ArmConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=SO101ArmConfig,
            node_rank=node_rank,
            count_fields=("left_follower_port",),
        )

        if robot_configs:
            so101_infos = []
            for config in robot_configs:
                if not config.disable_validate:
                    cls._validate_serial_ports(config, node_rank)
                so101_infos.append(
                    SO101ArmHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )
            return HardwareResource(type=cls.HW_TYPE, infos=so101_infos)
        return None

    @staticmethod
    def _validate_serial_ports(config: "SO101ArmConfig", node_rank: int) -> None:
        """Warn if the serial ports are not visible on this node."""
        ports = [config.left_follower_port, config.right_follower_port]
        for port in ports:
            if not os.path.exists(port):
                warnings.warn(
                    f"Serial port '{port}' not found on node rank {node_rank}. "
                    "The SO101 controller may fail to start."
                )


@NodeHardwareConfig.register_hardware_config(SO101ArmRobot.HW_TYPE)
@dataclass
class SO101ArmConfig(HardwareConfig):
    """Configuration for an SO101 bimanual robot."""

    left_follower_port: str = "/dev/ttyACM2"
    """Serial port for the left follower arm."""

    right_follower_port: str = "/dev/ttyACM3"
    """Serial port for the right follower arm."""

    left_wrist_camera: Optional[dict[str, Any]] = None
    """Camera spec for the left wrist camera.
    If ``None`` (default), the env falls back to a placeholder path."""

    right_wrist_camera: Optional[dict[str, Any]] = None
    """Camera spec for the right wrist camera.
    If ``None`` (default), the env falls back to a placeholder path."""

    left_global_camera: Optional[dict[str, Any]] = None
    """Camera spec for the left global (high) camera.
    If ``None`` (default), the env falls back to a placeholder path."""

    robot_id: str = "bi"
    """Robot ID passed to BiSOFollower."""

    max_relative_target: float = 5.0
    """Max relative target passed to SOFollowerConfig."""

    controller_node_rank: Optional[int] = None
    """Node rank where :class:`SO101Controller` should run.
    When ``None`` (default), co-located with the env worker."""

    disable_validate: bool = False
    """Whether to skip serial port validation during enumeration."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in SO101Arm config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.left_wrist_camera is None:
            self.left_wrist_camera = {"index_or_path": "/dev/video0"}
        if self.right_wrist_camera is None:
            self.right_wrist_camera = {"index_or_path": "/dev/video1"}
        if self.left_global_camera is None:
            self.left_global_camera = {"index_or_path": "/dev/video2"}
