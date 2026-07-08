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
"""Verify that the SO101 + RLinf environment is ready for RL training.

Usage::

    python rlinf/envs/realworld/so101/verify_env.py
    python rlinf/envs/realworld/so101/verify_env.py --skip-camera
    python rlinf/envs/realworld/so101/verify_env.py --skip-hardware
    python rlinf/envs/realworld/so101/verify_env.py \
        --left-follower-port /dev/ttyACM0 \
        --right-follower-port /dev/ttyACM1 \
        --left-wrist-camera /dev/v4l/by-path/...-video-index0 \
        --right-wrist-camera /dev/v4l/by-path/...-video-index0 \
        --left-global-camera /dev/v4l/by-path/...-video-index0

Steps:
    1. LeRobot import check
    2. Serial port presence
    3. ManipulatorRobot connectivity (joint read)
    4. Camera image read [skippable]
    5. RLinf SO101PickAndPlaceEnv (dummy mode)
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_SUCCESS = 0
_FAILURE = 1
_SKIP = 2

_DEFAULT_LEFT_PORT = "/dev/ttyACM2"
_DEFAULT_RIGHT_PORT = "/dev/ttyACM3"
_DEFAULT_LEFT_WRIST_CAMERA = "/dev/video0"
_DEFAULT_RIGHT_WRIST_CAMERA = "/dev/video1"
_DEFAULT_LEFT_GLOBAL_CAMERA = "/dev/video2"


def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m"


def _yellow(msg: str) -> str:
    return f"\033[93m{msg}\033[0m"


def _status(label: str, result: int, detail: str = "") -> None:
    if result == _SUCCESS:
        print(f"  {_green('[ OK ]')} {label}")
    elif result == _SKIP:
        print(f"  {_yellow('[SKIP]')} {label}")
        if detail:
            print(f"         {detail}")
    else:
        print(f"  {_red('[FAIL]')} {label}")
        if detail:
            print(f"         {detail}")


# ──────────────────────────────────────────────────────────────────


def step_lerobot_import() -> tuple[int, str]:
    """Check that LeRobot is importable."""
    try:
        from lerobot.common.robot_devices.robots.configs import (
            So101RobotConfig,  # noqa: F401
        )
        from lerobot.common.robot_devices.robots.manipulator import (
            ManipulatorRobot,  # noqa: F401
        )

        return _SUCCESS, "LeRobot ManipulatorRobot import OK"
    except ImportError as e:
        return _FAILURE, (
            f"LeRobot not importable: {e}. "
            "Make sure you installed the so101 extra: "
            "bash requirements/install.sh embodied --model openpi --env so101"
        )


def step_serial_ports(left_port: str, right_port: str) -> tuple[int, str]:
    """Check that the follower serial ports exist."""
    missing = [p for p in (left_port, right_port) if not os.path.exists(p)]
    if missing:
        return _FAILURE, (
            f"Serial ports not found: {missing}. "
            "Connect the SO101 arms and check /dev/ttyACM* assignments."
        )
    return _SUCCESS, f"Serial ports found: {left_port}, {right_port}"


def step_robot_connect(
    skip_hardware: bool,
    left_port: str,
    right_port: str,
    left_wrist_camera: str,
    right_wrist_camera: str,
    left_global_camera: str,
) -> tuple[int, str]:
    """Connect to the robot and read joint positions."""
    if skip_hardware:
        return _SKIP, "Skipped by --skip-hardware"

    try:
        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        arm_motor_names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
        package_dir = Path(__file__).resolve().parent

        def _camera_config(index_or_path: str) -> OpenCVCameraConfig:
            try:
                camera_index = int(index_or_path)
            except ValueError:
                camera_index = str(index_or_path)
            return OpenCVCameraConfig(
                camera_index=camera_index,
                width=640,
                height=480,
                fps=30,
            )

        def _arm_config(port: str) -> FeetechMotorsBusConfig:
            return FeetechMotorsBusConfig(
                port=port,
                motors={
                    name: (idx, "sts3215")
                    for idx, name in enumerate(arm_motor_names, start=1)
                },
            )

        config = So101RobotConfig(
            calibration_dir=str(package_dir / "calibration"),
            leader_arms={},
            follower_arms={
                "left": _arm_config(left_port),
                "right": _arm_config(right_port),
            },
            cameras={
                "left_global": _camera_config(left_global_camera),
                "left_wrist": _camera_config(left_wrist_camera),
                "right_wrist": _camera_config(right_wrist_camera),
            },
            max_relative_target=5.0,
        )
        robot = ManipulatorRobot(config)
        robot.connect()
        obs = robot.capture_observation()
        q = obs["observation.state"].numpy()
        robot.disconnect()
        return _SUCCESS, f"12 joints readable (positions: {[f'{v:.3f}' for v in q]})"
    except Exception:
        return _FAILURE, f"ManipulatorRobot connection failed:\n{traceback.format_exc()}"


def step_camera(
    skip_camera: bool,
    skip_hardware: bool,
    left_wrist_camera: str,
    right_wrist_camera: str,
    left_global_camera: str,
) -> tuple[int, str]:
    """Read a single frame from each camera."""
    if skip_camera or skip_hardware:
        return _SKIP, "Skipped by --skip-camera or --skip-hardware"

    try:
        import cv2
    except ImportError:
        return _FAILURE, "opencv-python not installed"

    camera_paths = [
        ("left_global", left_global_camera),
        ("left_wrist", left_wrist_camera),
        ("right_wrist", right_wrist_camera),
    ]
    lines = []
    for name, path in camera_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return _FAILURE, f"Cannot open camera {name} at {path}"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return _FAILURE, f"Cannot read frame from camera {name} at {path}"
        lines.append(f"{name}: {frame.shape}")
    return _SUCCESS, "\n".join(lines)


def step_rlinf_env() -> tuple[int, str]:
    """Create a dummy SO101PickAndPlaceEnv and test reset + step."""
    try:
        import gymnasium as gym
    except ImportError as e:
        return _FAILURE, f"gymnasium not installed: {e}"

    try:
        import rlinf.envs.realworld  # noqa: F401
    except Exception as e:
        return _FAILURE, f"Failed to import rlinf.envs.realworld: {e}"

    try:
        env = gym.make(
            "SO101PickAndPlaceEnv-v1",
            override_cfg={
                "is_dummy": True,
                "task_description": "verification task",
            },
        )
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        return _SUCCESS, (
            f"dummy reset + step OK; state shape={obs['state']['arm_joint_position'].shape}, "
            f"reward={reward:.3f}"
        )
    except Exception:
        return _FAILURE, f"RLinf SO101Env dummy test failed:\n{traceback.format_exc()}"


# ──────────────────────────────────────────────────────────────────


def _motor_names() -> tuple[str, ...]:
    return (
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SO101 + RLinf environment.")
    parser.add_argument(
        "--skip-camera", action="store_true", help="Skip camera checks."
    )
    parser.add_argument(
        "--skip-hardware", action="store_true", help="Skip real robot connection."
    )
    parser.add_argument(
        "--left-follower-port",
        default=_DEFAULT_LEFT_PORT,
        help=f"Serial port for the left follower arm (default: {_DEFAULT_LEFT_PORT}).",
    )
    parser.add_argument(
        "--right-follower-port",
        default=_DEFAULT_RIGHT_PORT,
        help=f"Serial port for the right follower arm (default: {_DEFAULT_RIGHT_PORT}).",
    )
    parser.add_argument(
        "--left-wrist-camera",
        default=_DEFAULT_LEFT_WRIST_CAMERA,
        help=f"Path or index for the left wrist camera (default: {_DEFAULT_LEFT_WRIST_CAMERA}).",
    )
    parser.add_argument(
        "--right-wrist-camera",
        default=_DEFAULT_RIGHT_WRIST_CAMERA,
        help=f"Path or index for the right wrist camera (default: {_DEFAULT_RIGHT_WRIST_CAMERA}).",
    )
    parser.add_argument(
        "--left-global-camera",
        default=_DEFAULT_LEFT_GLOBAL_CAMERA,
        help=f"Path or index for the left global/high camera (default: {_DEFAULT_LEFT_GLOBAL_CAMERA}).",
    )
    args = parser.parse_args()

    print("SO101 + RLinf environment verification")
    print("=" * 40)

    steps = [
        ("LeRobot import", step_lerobot_import()),
        (
            "Serial ports",
            step_serial_ports(args.left_follower_port, args.right_follower_port),
        ),
        (
            "ManipulatorRobot connectivity",
            step_robot_connect(
                args.skip_hardware,
                args.left_follower_port,
                args.right_follower_port,
                args.left_wrist_camera,
                args.right_wrist_camera,
                args.left_global_camera,
            ),
        ),
        (
            "Camera read",
            step_camera(
                args.skip_camera,
                args.skip_hardware,
                args.left_wrist_camera,
                args.right_wrist_camera,
                args.left_global_camera,
            ),
        ),
        ("RLinf SO101Env dummy", step_rlinf_env()),
    ]

    for label, (result, detail) in steps:
        _status(label, result, detail)
        time.sleep(0.1)

    if any(result == _FAILURE for _, (result, _) in steps):
        print("\nVerification FAILED.")
        return 1
    print("\nVerification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
