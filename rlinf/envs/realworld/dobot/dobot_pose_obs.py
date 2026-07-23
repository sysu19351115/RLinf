"""pose 模式观测的纯函数与状态跟踪（无硬件依赖，便于离线测试）。

对应 pi05_dobot_t265_pose_train：观测 state 为 8 维
``[x,y,z(m), qw,qx,qy,qz, gripper]``，并附带 ``prev_state``（上一帧 state，
首帧为自身），与训练数据（convert_to_lerobot.py state_mode=pose 的
``observation.prev_state``）约定一致，供 server 侧 DeltaPose 把绝对位姿
转为相对上一帧的位姿。
"""

from __future__ import annotations

import numpy as np


def pose_state(pose: np.ndarray, gripper: float) -> np.ndarray:
    """组装 pose 模式的 8 维 state ``[x,y,z(m), qw,qx,qy,qz, gripper]``。"""
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError(
            f"pose must have 7 values [x,y,z,qw,qx,qy,qz], got {pose.shape[0]}"
        )
    return np.concatenate([pose, np.array([gripper], dtype=np.float32)])


class PoseStateTracker:
    """维护 pose 模式观测的 prev_state：上一帧 8 维 pose state，首帧为自身。"""

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None

    def update(self, state: np.ndarray) -> np.ndarray:
        prev = self._prev if self._prev is not None else state
        self._prev = state
        return prev

    def reset(self) -> None:
        self._prev = None
