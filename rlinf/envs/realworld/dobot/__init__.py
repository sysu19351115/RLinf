"""Dobot CR5AF 真机环境集成（关节角 + 位姿双模式）。

环境走 RealWorld 派发：``env_type=realworld`` + ``init_params.id`` 指向本包
注册的 gym id（如 ``DobotPickAndPlaceEnv-v1``）。
"""

from .dobot_env import DobotEnv, DobotRobotConfig
from .dobot_robot_state import DobotRobotState

__all__ = [
    "DobotEnv",
    "DobotRobotConfig",
    "DobotRobotState",
]
