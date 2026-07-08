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
"""Convert SO101 calibration files from lerobot_zhiyu format to LeRobot format.

The lerobot_zhiyu fork stores calibration as a per-motor dict with
``range_min``/``range_max`` and uses ``MotorNormMode.RANGE_M100_100`` for arm
joints and ``RANGE_0_100`` for the gripper.

The LeRobot version pinned in RLinf expects calibration files in the format
used by ``lerobot.common.robot_devices.motors.feetech``, i.e. lists of
``homing_offset``, ``drive_mode``, ``start_pos``, ``end_pos``, ``calib_mode``,
and ``motor_names``.

This script maps the lerobot_zhiyu ranges to LeRobot LINEAR calibration so that
after calibration:

- arm joints are in ``[-100, 100]``
- gripper is in ``[0, 100]``

Usage::

    python rlinf/envs/realworld/so101/convert_calibration.py \
        /path/to/lerobot_zhiyu/calibration/robots/so_follower \
        rlinf/envs/realworld/so101/calibration
"""

import argparse
import json
from pathlib import Path

# Order expected by the SO101 policy / RLinf env.
_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def convert_arm_motor(range_min: int, range_max: int) -> tuple[float, float]:
    """Return (start_pos, end_pos) for LeRobot LINEAR mode mapping to [-100, 100]."""
    mid = (range_min + range_max) / 2.0
    return mid, float(range_max)


def convert_gripper(range_min: int, range_max: int) -> tuple[float, float]:
    """Return (start_pos, end_pos) for LeRobot LINEAR mode mapping to [0, 100]."""
    return float(range_min), float(range_max)


def convert_file(src_path: Path) -> dict:
    with open(src_path) as f:
        src = json.load(f)

    homing_offset = []
    drive_mode = []
    start_pos = []
    end_pos = []
    calib_mode = []
    motor_names = []

    for name in _MOTOR_ORDER:
        if name not in src:
            raise KeyError(f"Motor '{name}' not found in {src_path}")
        entry = src[name]
        motor_names.append(name)
        homing_offset.append(0)
        drive_mode.append(entry.get("drive_mode", 0))
        calib_mode.append("LINEAR")

        range_min = int(entry["range_min"])
        range_max = int(entry["range_max"])
        if name == "gripper":
            s, e = convert_gripper(range_min, range_max)
        else:
            s, e = convert_arm_motor(range_min, range_max)
        start_pos.append(s)
        end_pos.append(e)

    return {
        "homing_offset": homing_offset,
        "drive_mode": drive_mode,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "calib_mode": calib_mode,
        "motor_names": motor_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SO101 calibration files.")
    parser.add_argument("src_dir", help="Directory containing lerobot_zhiyu calibration JSONs.")
    parser.add_argument("dst_dir", help="Directory to write LeRobot-format calibration JSONs.")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("left_follower.json", "left_follower.json"),
        ("right_follower.json", "right_follower.json"),
    ]

    for src_name, dst_name in pairs:
        src_path = src_dir / src_name
        if not src_path.exists():
            print(f"Skipping missing source file: {src_path}")
            continue
        converted = convert_file(src_path)
        dst_path = dst_dir / dst_name
        with open(dst_path, "w") as f:
            json.dump(converted, f, indent=4)
        print(f"Wrote {dst_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
