#!/usr/bin/env python3
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
"""Record current SO101 joint positions as initial/end pose files."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

_MOTOR_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)

_ARM_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _make_robot(
    left_follower_port: str, right_follower_port: str, max_relative_target: float
):
    package_dir = Path(__file__).resolve().parent

    def _arm_config(port: str) -> FeetechMotorsBusConfig:
        return FeetechMotorsBusConfig(
            port=port,
            motors={
                name: (idx, "sts3215")
                for idx, name in enumerate(_ARM_MOTOR_NAMES, start=1)
            },
        )

    config = So101RobotConfig(
        calibration_dir=str(package_dir / "calibration"),
        leader_arms={},
        follower_arms={
            "left": _arm_config(left_follower_port),
            "right": _arm_config(right_follower_port),
        },
        max_relative_target=max_relative_target,
    )
    return ManipulatorRobot(config)


def _state_from_robot_obs(arm_joint_position: np.ndarray) -> np.ndarray:
    return np.asarray(arm_joint_position, dtype=np.float32)


def _save(path: str, values: np.ndarray) -> None:
    path = Path(path)
    payload = {
        "motor_names": list(_MOTOR_NAMES),
        "positions": [float(v) for v in values],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record SO101 current joint positions."
    )
    parser.add_argument("--left-follower-port", default="/dev/ttyACM2")
    parser.add_argument("--right-follower-port", default="/dev/ttyACM3")
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--pose-kind", choices=("initial", "end"), default="initial")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"rlinf/envs/realworld/so101/{args.pose_kind}_joints.json"
    robot = _make_robot(
        args.left_follower_port, args.right_follower_port, args.max_relative_target
    )
    try:
        robot.connect()
        obs = robot.capture_observation()
        values = _state_from_robot_obs(obs["observation.state"].numpy())
        _save(output, values)
        print(f"Saved {args.pose_kind} joints to {output}")
        for name, value in zip(_MOTOR_NAMES, values):
            print(f"{name}: {float(value):.6f}")
    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()


if __name__ == "__main__":
    main()
