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
"""Verify that the ReBot Arm + RLinf environment is ready for RL training.

Usage::

    python rlinf/envs/realworld/rebot/verify_env.py
    python rlinf/envs/realworld/rebot/verify_env.py --skip-camera

Steps:
    1. CAN bus presence
    2. Motor discovery on CAN bus
    3. reBotArm SDK connectivity (joint read + FK)
    4. Camera detection (RealSense D435)  [skippable]
    5. RLinf RebotArmPickAndPlaceEnv (dummy mode)
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_SUCCESS = 0
_FAILURE = 1
_SKIP = 2

_CAN_IFACE = "can0"


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


def step_can_presence() -> int:
    """Check that the CAN interface exists and is UP."""
    can_path = f"/sys/class/net/{_CAN_IFACE}"
    if not os.path.exists(can_path):
        return _FAILURE, (
            f"CAN interface '{_CAN_IFACE}' not found at {can_path}. "
            "Run: sudo ip link set can0 up type can bitrate 1000000 restart-ms 100"
        )
    # Check UP state.
    try:
        out = subprocess.check_output(
            ["ip", "-details", "link", "show", _CAN_IFACE],
            stderr=subprocess.STDOUT,
            text=True,
        )
        if "state UP" not in out:
            return _FAILURE, (
                f"CAN interface '{_CAN_IFACE}' exists but is not UP. "
                "Run: sudo ip link set can0 up"
            )
    except subprocess.CalledProcessError as e:
        return _FAILURE, f"Cannot query CAN interface: {e}"
    return _SUCCESS, ""


def step_motor_scan() -> int:
    """Scan for motors via motorbridge-cli."""
    try:
        out = subprocess.check_output(
            [
                "motorbridge-cli", "scan",
                "--vendor", "robstride",
                "--channel", _CAN_IFACE,
                "--start-id", "1",
                "--end-id", "127",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _FAILURE, "motorbridge-cli not found. Install: pip install motorbridge"
    except subprocess.TimeoutExpired:
        return _FAILURE, "Motor scan timed out (30 s). Check CAN wiring and power."
    except subprocess.CalledProcessError as e:
        return _FAILURE, f"Motor scan failed:\n{e.stdout}"

    # Expect at least 7 motors (6 joints + 1 gripper).
    motor_count = out.count("0x")
    if motor_count < 7:
        return _FAILURE, (
            f"Only {motor_count} motor(s) found (expected >= 7). "
            "Check CAN wiring, terminal resistors, and robot power."
        )
    return _SUCCESS, f"{motor_count} motors found"


def step_rebot_sdk() -> int:
    """Connect to the arm with the reBotArm SDK and read state + FK."""
    try:
        from reBotArm_control_py import reBotArm
    except ImportError as e:
        # Add the SDK path in case it is not on sys.path.
        _rebot_dir = os.path.dirname(os.path.abspath(__file__))
        if _rebot_dir not in sys.path:
            sys.path.insert(0, _rebot_dir)
        try:
            from reBotArm_control_py import reBotArm
        except ImportError:
            return _FAILURE, (
                f"reBotArm_control_py not importable: {e}. "
                "Make sure you installed the rebot extra: "
                "bash requirements/install_local.sh --cpu-only --env rebot"
            )

    try:
        with reBotArm() as arm:
            q = arm.get_joint_positions()
            pos, rpy = arm.get_end_effector_pose()
            msg = (
                f"joint positions (rad): {[f'{v:.3f}' for v in q]}\n"
                f"         end-effector pos (m): {[f'{v:.3f}' for v in pos]}\n"
                f"         end-effector rpy (rad): {[f'{v:.3f}' for v in rpy]}"
            )
    except Exception as e:
        return _FAILURE, f"reBotArm SDK error:\n{traceback.format_exc()}"

    return _SUCCESS, msg


def step_camera() -> int:
    """Detect RealSense D435 cameras."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return _FAILURE, "pyrealsense2 not installed. Run: pip install pyrealsense2"

    devices = list(rs.context().devices)
    if not devices:
        return _FAILURE, "No RealSense devices found. Check USB connection."

    lines = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        lines.append(f"{name}  serial={serial}")
    return _SUCCESS, "\n".join(lines)


def step_rlinf_env() -> int:
    """Create a dummy RebotArmPickAndPlaceEnv and test reset + step."""
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError as e:
        return _FAILURE, f"gymnasium or numpy not installed: {e}"

    try:
        # Trigger gymnasium registration of RebotArmPickAndPlaceEnv-v1.
        import rlinf.envs.realworld  # noqa: F401

        env = gym.make(
            "RebotArmPickAndPlaceEnv-v1",
            override_cfg={"is_dummy": True},
            env_idx=0,
        )
    except Exception as e:
        return _FAILURE, (
            f"Failed to create RebotArmPickAndPlaceEnv-v1:\n"
            f"{traceback.format_exc()}"
        )

    try:
        obs, _info = env.reset()
    except Exception:
        return _FAILURE, f"env.reset() failed:\n{traceback.format_exc()}"

    try:
        action = np.zeros(7, dtype=np.float32)
        obs, reward, term, trunc, _info = env.step(action)
    except Exception:
        return _FAILURE, f"env.step() failed:\n{traceback.format_exc()}"

    state_keys = sorted(obs.get("state", {}).keys())
    frame_keys = sorted(obs.get("frames", {}).keys())
    msg = (
        f"state keys: {state_keys}\n"
        f"         frame keys: {frame_keys}\n"
        f"         reward={reward}, terminated={term}, truncated={trunc}"
    )
    return _SUCCESS, msg


# ──────────────────────────────────────────────────────────────────

_STEPS = [
    ("CAN bus presence", step_can_presence),
    ("Motor discovery", step_motor_scan),
    ("reBotArm SDK connectivity", step_rebot_sdk),
    ("Camera detection (RealSense D435)", step_camera),
    ("RLinf Env (dummy mode)", step_rlinf_env),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ReBot Arm + RLinf environment readiness."
    )
    parser.add_argument(
        "--skip-camera", action="store_true",
        help="Skip the camera detection step."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ReBot Arm + RLinf Environment Verification")
    print("=" * 60)
    print()

    ok, fail, skip = 0, 0, 0
    for label, func in _STEPS:
        if args.skip_camera and "Camera" in label:
            _status(label, _SKIP, "skipped via --skip-camera")
            skip += 1
            continue

        sys.stdout.write(f"[..] {label} ...")
        sys.stdout.flush()
        try:
            result, detail = func()
        except Exception:
            result = _FAILURE
            detail = traceback.format_exc()

        # Clear the progress line.
        sys.stdout.write("\r" + " " * 70 + "\r")
        _status(label, result, detail)
        print()

        if result == _SUCCESS:
            ok += 1
        elif result == _SKIP:
            skip += 1
        else:
            fail += 1

    print("=" * 60)
    print(f"Results: {ok} passed, {fail} failed, {skip} skipped")
    print("=" * 60)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
