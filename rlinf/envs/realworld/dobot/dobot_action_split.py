"""统一动作按 action_mode 拆分（纯函数，无硬件依赖，便于离线测试）。"""

from __future__ import annotations

import numpy as np


def split_follower_action(action, action_mode: str) -> tuple[np.ndarray, float]:
    """按 action_mode 把统一动作拆成 (臂目标, 夹爪归一化)。

    - ``joint``: 7 维 ``[j1..j6 rad, gripper]`` → (6 维关节, 夹爪)
    - ``cartesian``: 8 维 ``[x,y,z(m), qw,qx,qy,qz, gripper]`` → (7 维位姿, 夹爪)

    维度不符直接抛 ValueError：关节策略的 7 维动作误发给 cartesian（或反之）
    必须 fail-fast，静默错切会产生非物理目标。
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    if action_mode == "cartesian":
        if action.shape[0] != 8:
            raise ValueError(
                "cartesian action must have 8 values [x,y,z,qw,qx,qy,qz, gripper], "
                f"got {action.shape[0]}"
            )
        return action[:7], float(action[7])
    if action.shape[0] != 7:
        raise ValueError(
            f"joint action must have 7 values [j1..j6, gripper], got {action.shape[0]}"
        )
    return action[:6], float(action[6])


def binarize_gripper_action(action, threshold: float) -> np.ndarray:
    """把动作末位（夹爪归一化 [0,1]）按阈值二值化。

    ``<= threshold`` → 全闭 0.0，``> threshold`` → 全开 1.0。归一化空间
    0=闭、1=开（与电机角空间方向相反，勿配反）。threshold 必须在 (0, 1)
    开区间。返回新数组，不修改原动作。
    """
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"gripper binary threshold must be in (0, 1), got {threshold}")
    action = np.asarray(action, dtype=float).reshape(-1).copy()
    if action.shape[0] < 1:
        raise ValueError("action must not be empty")
    action[-1] = 0.0 if action[-1] <= threshold else 1.0
    return action


class RelativeGripperBinarizer:
    """夹爪相对开闭：按「模型输出 − 当前角」的差值二值化，带保持。

    全部在归一化 [0,1] 空间（0=闭，1=开）：
    - ``output − current > threshold`` → 全开 1.0；
    - ``current − output > threshold`` → 全闭 0.0；
    - 都不超过 → 保持上一次的全开/全闭状态。

    首帧尚无历史状态且未触发时，按当前角就近取整（>= 0.5 视为开）。
    """

    def __init__(self, threshold: float):
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"gripper relative threshold must be in (0, 1), got {threshold}"
            )
        self._threshold = threshold
        self._state: float | None = None

    def reset(self) -> None:
        self._state = None

    def map(self, output_norm: float, current_norm: float) -> float:
        output = float(output_norm)
        current = float(current_norm)
        if output - current > self._threshold:
            self._state = 1.0
        elif current - output > self._threshold:
            self._state = 0.0
        elif self._state is None:
            self._state = 1.0 if current >= 0.5 else 0.0
        return self._state
