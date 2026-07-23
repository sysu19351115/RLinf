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

"""Hardware registration for Dobot CR5AF (6-DOF arm + Damiao gripper, TCP).

This module registers the ``Dobot`` hardware type so that YAML configs with
``hardware: type: Dobot`` resolve to :class:`DobotConfig`. The config carries
all connection parameters (IP, gripper port, camera serials, dual-mode settings)
that flow into :class:`DobotEnv` via ``hardware_info.config``.
"""

import socket
import warnings
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)
from .auto_config import RobotAutoConfig


@dataclass
class DobotHWInfo(HardwareInfo):
    """Hardware information for a Dobot CR5AF robotic system."""

    config: "DobotConfig"


@Hardware.register()
class DobotRobot(Hardware):
    """Hardware policy for Dobot robots (TCP, 6-DOF + Damiao gripper)."""

    HW_TYPE = "Dobot"

    @classmethod
    def enumerate(
        cls,
        node_rank: int,
        configs: Optional[list["DobotConfig"]] = None,
    ) -> Optional[HardwareResource]:
        """Enumerate Dobot robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware
                resources. ``None`` if no Dobot hardware is configured for this
                node.
        """
        assert configs is not None, "Dobot hardware requires explicit configurations."
        robot_configs: list["DobotConfig"] = []
        for config in configs:
            if isinstance(config, DobotConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=DobotConfig,
            node_rank=node_rank,
            count_fields=("ip",),
        )

        if robot_configs:
            dobot_infos = []
            for config in robot_configs:
                if not config.disable_validate:
                    cls._validate_ip_reachable(config.ip, node_rank)
                dobot_infos.append(
                    DobotHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )
            return HardwareResource(type=cls.HW_TYPE, infos=dobot_infos)
        return None

    @staticmethod
    def _validate_ip_reachable(ip: str, node_rank: int) -> None:
        """Warn if the Dobot controller TCP port is not reachable.

        Probes port 29999 (dashboard) with a short timeout. This is a best-effort
        connectivity check — warnings are non-fatal (the controller may still start
        if the robot comes online later).
        """
        try:
            with socket.create_connection((ip, 29999), timeout=2.0):
                pass
        except OSError:
            warnings.warn(
                f"Dobot controller at {ip}:29999 is not reachable on node "
                f"rank {node_rank}. The DobotController may fail to start."
            )


@NodeHardwareConfig.register_hardware_config(DobotRobot.HW_TYPE)
@dataclass
class DobotConfig(HardwareConfig):
    """Configuration for a Dobot CR5AF robot.

    Connection fields flow into :class:`DobotEnv` via ``hardware_info.config``
    and are merged into :class:`DobotRobotConfig` in ``DobotEnv._setup_hardware``.
    """

    ip: str = "192.168.5.1"
    """Dobot controller IP address."""

    speed: int = 100
    """Global MovJ speed percentage (0-100)."""

    user_index: int = 0
    """User coordinate system index (must match data collection)."""

    tool_index: int = 0
    """Tool coordinate system index (must match data collection)."""

    action_mode: str = "joint"
    """``"joint"`` (ServoJ, 7-dim) or ``"cartesian"`` (ServoP, 8-dim)."""

    state_mode: str = "joint"
    """``"joint"`` (7-dim state) or ``"pose"`` (8-dim state + prev_state).
    ``"pose"`` requires ``action_mode="cartesian"``."""

    gripper_port: str = "/dev/ttyACM0"
    """Damiao gripper serial bridge device path."""

    enable_gripper: bool = True
    """Whether the gripper is attached and should be controlled."""

    gripper_closed_deg: float = 0.0
    """Gripper closed calibration (motor degrees)."""

    gripper_open_deg: float = -320.0
    """Gripper open calibration (motor degrees, negative)."""

    enable_ft_sensor: bool = True
    """Enable the six-axis force/torque sensor (required for ForceVLA)."""

    camera_serials: Optional[list[str]] = None
    """Optional list of camera serial numbers.
    Pass ``[]`` or leave ``None`` to run without cameras."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"``, ``"zed"``, ``"lumos"`` or generic USB via
    ``"opencv"`` / ``"usb"`` / ``"v4l2"`` (LeRobot ``OpenCVCamera``)."""

    camera_resolution: tuple[int, int] = (640, 480)
    """Requested USB/color stream resolution as ``(width, height)``."""

    camera_fps: int = 30
    """Requested color stream frame rate."""

    camera_fourcc: Optional[str] = None
    """Optional V4L2 FOURCC (e.g. ``"MJPG"``) for the OpenCV USB backend.
    ``None`` keeps the backend default; set ``"MJPG"`` for high-resolution
    capture (e.g. 1920x1080) where YUYV is bandwidth-limited."""

    controller_node_rank: Optional[int] = None
    """Node rank where :class:`DobotController` should run.
    When ``None`` (default), co-located with the env worker."""

    disable_validate: bool = False
    """Whether to skip IP reachability validation during enumeration."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Dobot config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
        # Validate dual-mode compatibility (same constraint as DobotRobotConfig).
        if self.state_mode == "pose" and self.action_mode != "cartesian":
            raise ValueError(
                "state_mode='pose' requires action_mode='cartesian': "
                "位姿策略输出 8 维绝对位姿动作，必须由 ServoP 执行"
            )
