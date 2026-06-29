#!/usr/bin/env python3
"""轨迹回放 + 实时记录 Demo。

用法:
    # 仅回放
    uv run python example/replay_and_record.py records/teach_trajectory_20260611_154039.npy

    # 边回放边记录实际关节角
    uv run python example/replay_and_record.py records/teach_trajectory_20260611_154039.npy --record
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from reBotArm_control_py import reBotArm

RATE = 30.0


def main():
    parser = argparse.ArgumentParser(description="回放示教轨迹并可选记录实际执行")
    parser.add_argument("path", type=str, help="轨迹 .npy 文件路径")
    parser.add_argument(
        "--record", action="store_true", help="边回放边记录实际关节角"
    )
    args = parser.parse_args()

    traj_path = Path(args.path)
    if not traj_path.exists():
        print(f"错误: 文件不存在: {traj_path}")
        sys.exit(1)

    traj = np.load(traj_path)
    print(f"加载轨迹: {traj_path}")
    print(f"  点数: {len(traj)}, shape: {traj.shape}")

    with reBotArm() as arm:
        print("\n机械臂已连接，正在回零...")
        arm.move_home(duration=1.0)
        print("回零完成。\n")

        print(f"开始回放轨迹（{'并记录' if args.record else '不记录'}）...")
        recorded = arm.replay_trajectory(traj=traj, rate=RATE, record=args.record)

        if args.record and recorded is not None and recorded.size > 0:
            records_dir = Path("records")
            records_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = records_dir / f"replay_record_{ts}.npy"
            np.save(save_path, recorded)
            print(f"\n实际执行轨迹已保存: {save_path}")
            print(f"  记录点数: {len(recorded)}, shape: {recorded.shape}")

        print("\n回放完成。")


if __name__ == "__main__":
    main()
