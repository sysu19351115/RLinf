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

import numpy as np
import torch

from rlinf.envs.realworld.so101.so101_controller import (
    _action_vector_to_robot_action,
    _images_from_robot_obs,
    _state_from_robot_obs,
)

_SUCCESS = 0
_FAILURE = 1
_SKIP = 2

_DEFAULT_LEFT_PORT = "/dev/ttyACM0"
_DEFAULT_RIGHT_PORT = "/dev/ttyACM1"
_DEFAULT_LEFT_WRIST_CAMERA = "/dev/video2"
_DEFAULT_RIGHT_WRIST_CAMERA = "/dev/video4"
_DEFAULT_LEFT_GLOBAL_CAMERA = "/dev/video0"


def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m"


def _yellow(msg: str) -> str:
    return f"\033[93m{msg}\033[0m"


def _status(label: str, result: int, detail: str = "") -> None:
    if result == _SUCCESS:
        print(f"  {_green('[ OK ]')} {label}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")
    elif result == _SKIP:
        print(f"  {_yellow('[SKIP]')} {label}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")
    else:
        print(f"  {_red('[FAIL]')} {label}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")


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


def _move_to_rlinf_pose(
    robot, target_rlinf: np.ndarray, steps: int = 60, fps: float = 30.0
) -> np.ndarray:
    """Smoothly move the robot to a 12-dim RLinf pose and return the RLinf state."""
    target_lerobot = _action_vector_to_robot_action(target_rlinf).numpy()
    current_lerobot = robot.capture_observation()["observation.state"].numpy()
    sleep_s = 1.0 / fps if fps > 0 else 0.0
    for step in range(1, steps + 1):
        alpha = step / steps
        command = current_lerobot + (target_lerobot - current_lerobot) * alpha
        robot.send_action(torch.from_numpy(command.astype(np.float32)))
        if sleep_s > 0:
            time.sleep(sleep_s)
    obs = robot.capture_observation()
    return _state_from_robot_obs(obs["observation.state"].numpy())


def _test_range_motion(robot) -> list[str]:
    """Move both arms to the mid-range pose and back to the starting pose.

    Records the current pose, visits the mid-range target
    (arm joints 0 in RLinf -> LeRobot 50%, gripper 50%), then returns to the
    recorded start pose. This verifies the RLinf <-> LeRobot conversion layer
    without driving the arms to their calibrated limits.
    """
    q_start = _state_from_robot_obs(
        robot.capture_observation()["observation.state"].numpy()
    )
    mid_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0] * 2, dtype=np.float32)

    q_mid = _move_to_rlinf_pose(robot, mid_pose, steps=120, fps=30.0)
    time.sleep(0.5)
    q_returned = _move_to_rlinf_pose(robot, q_start, steps=120, fps=30.0)

    arm_indices = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=np.int64)
    gripper_indices = np.array([5, 11], dtype=np.int64)

    mid_arm_err = float(np.max(np.abs(q_mid[arm_indices])))
    mid_gripper_err = float(np.max(np.abs(q_mid[gripper_indices] - 50.0)))
    return_arm_err = float(
        np.max(np.abs(q_returned[arm_indices] - q_start[arm_indices]))
    )
    return_gripper_err = float(
        np.max(np.abs(q_returned[gripper_indices] - q_start[gripper_indices]))
    )

    tolerance = 8.0  # allow a few percent tracking/error margin
    if mid_arm_err > tolerance:
        raise RuntimeError(
            f"Mid-range arm joints off target: max err {mid_arm_err:.2f}"
        )
    if mid_gripper_err > tolerance:
        raise RuntimeError(
            f"Mid-range grippers off target: max err {mid_gripper_err:.2f}"
        )
    if return_arm_err > tolerance:
        raise RuntimeError(
            f"Return arm joints off target: max err {return_arm_err:.2f}"
        )
    if return_gripper_err > tolerance:
        raise RuntimeError(
            f"Return grippers off target: max err {return_gripper_err:.2f}"
        )

    return [
        f"start positions: {[f'{v:.2f}' for v in q_start]}",
        f"mid positions: {[f'{v:.2f}' for v in q_mid]}",
        f"return positions: {[f'{v:.2f}' for v in q_returned]}",
        f"mid arm max err={mid_arm_err:.2f}, gripper max err={mid_gripper_err:.2f}",
        f"return arm max err={return_arm_err:.2f}, gripper max err={return_gripper_err:.2f}",
    ]


def step_robot_connect(
    skip_hardware: bool,
    skip_camera: bool,
    test_control: bool,
    left_port: str,
    right_port: str,
    left_wrist_camera: str,
    right_wrist_camera: str,
    left_global_camera: str,
) -> tuple[int, str]:
    """Connect to the robot and optionally verify that motors can be commanded."""
    if skip_hardware:
        return _SKIP, "Skipped by --skip-hardware"

    try:
        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

        package_dir = Path(__file__).resolve().parent

        def _camera_config(index_or_path: str) -> OpenCVCameraConfig:
            try:
                camera_index = int(index_or_path)
            except ValueError:
                camera_index = str(index_or_path)
            return OpenCVCameraConfig(
                camera_index=camera_index,
            )

        from rlinf.envs.realworld.so101.so101_motor_init import build_so101_manipulator

        cameras = (
            None
            if skip_camera
            else {
                "left_global": _camera_config(left_global_camera),
                "left_wrist": _camera_config(left_wrist_camera),
                "right_wrist": _camera_config(right_wrist_camera),
            }
        )

        robot = build_so101_manipulator(
            left_follower_port=left_port,
            right_follower_port=right_port,
            calibration_dir=package_dir / "calibration",
            cameras=cameras,
            max_relative_target=5.0,
        )
        robot.connect()

        # Read observation exactly like SO101Controller.get_observation().
        obs = robot.capture_observation()
        q = _state_from_robot_obs(obs["observation.state"].numpy())
        if not skip_camera:
            images = _images_from_robot_obs(
                {
                    key.removeprefix("observation.images."): value.numpy()
                    for key, value in obs.items()
                    if key.startswith("observation.images.")
                }
            )
        else:
            images = None

        # Send a small 12-dim action exactly like SO101Controller.send_action().
        action = q.copy()
        action[5] += 1.0  # left gripper
        action[11] += 1.0  # right gripper
        robot.send_action(_action_vector_to_robot_action(action))
        time.sleep(0.5)

        obs2 = robot.capture_observation()
        q2 = _state_from_robot_obs(obs2["observation.state"].numpy())

        detail_lines = [
            f"state shape={q.shape}, positions={[f'{v:.3f}' for v in q]}",
        ]
        if images is not None:
            detail_lines.append(
                f"images={ {k: list(v.shape) for k, v in images.items()} }"
            )
        else:
            detail_lines.append("images skipped")
        detail_lines.append(
            f"12-dim action send ok: left_gripper {q[5]:.3f}->{q2[5]:.3f}, right_gripper {q[11]:.3f}->{q2[11]:.3f}"
        )
        if test_control:
            range_lines = _test_range_motion(robot)
            detail_lines.append("Range motion test passed:")
            detail_lines.extend(range_lines)

        robot.disconnect()
        return _SUCCESS, "\n".join(detail_lines)
    except Exception:
        return (
            _FAILURE,
            f"SO101Controller-level logic test failed:\n{traceback.format_exc()}",
        )


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
        "--test-control",
        action="store_true",
        help="Also verify that each motor responds to small goal-position commands.",
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
                args.skip_camera,
                args.test_control,
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
