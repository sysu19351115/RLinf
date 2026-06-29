#!/usr/bin/env python3
"""重复定位精度测试。

机械臂在 home（零位）与目标关节角之间往返运动 10 次，
每次到达目标后记录末端位姿（FK），统计与目标位姿的偏差。

用法:
    uv run python example/repeatability_demo.py
"""
import time

import numpy as np

from reBotArm_control_py import reBotArm

DEG = np.pi / 180.0

# 目标关节角（弧度）
TARGET_Q = np.array([1.304, -1.812, -1.141, 0.803, -0.035, 0.136])

ROUNDS = 10
HOME_Q = np.zeros(6)


def main():
    print("=" * 60)
    print(" 重复定位精度测试")
    print("=" * 60)
    print(f" 往返次数: {ROUNDS}")
    print(f" 目标关节角 (deg): {' '.join(f'{v:+5.1f}' for v in np.rad2deg(TARGET_Q))}")
    print()

    with reBotArm() as arm:
        records_pos = np.zeros((ROUNDS, 3))
        records_rpy = np.zeros((ROUNDS, 3))

        # 用 FK 计算目标末端位姿
        target_pos, target_rpy = _compute_target_pose(arm)
        print(f" 目标末端位置 (m):  x={target_pos[0]:+.4f}  y={target_pos[1]:+.4f}  "
              f"z={target_pos[2]:+.4f}")
        print(f" 目标末端姿态 (deg): roll={target_rpy[0]*180/np.pi:+.2f}  "
              f"pitch={target_rpy[1]*180/np.pi:+.2f}  yaw={target_rpy[2]*180/np.pi:+.2f}")
        print()

        for r in range(ROUNDS):
            print(f"--- 第 {r + 1}/{ROUNDS} 轮 ---")

            print("  回零位...")
            arm.move_joints(HOME_Q, duration=1.0)
            time.sleep(1)

            print("  移动到目标...")
            arm.move_joints(TARGET_Q, duration=1.0)

            time.sleep(2)
            pos, rpy = arm.get_end_effector_pose()
            records_pos[r] = pos
            records_rpy[r] = rpy

            p_err = (pos - target_pos) * 1000
            r_err = np.rad2deg(rpy - target_rpy)
            print(f"  实际末端 (m):     x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")
            print(f"  位置误差 (mm):   {' '.join(f'{e:+5.2f}' for e in p_err)}")
            print(f"  姿态误差 (deg):  {' '.join(f'{e:+5.2f}' for e in r_err)}")

        print()
        print("=" * 60)
        print(" 测试结果")
        print("=" * 60)

        mean_pos = records_pos.mean(axis=0)
        std_pos = records_pos.std(axis=0, ddof=1) * 1000
        pos_err_each = np.linalg.norm(records_pos - target_pos, axis=1) * 1000
        peak_pos = records_pos.max(axis=0) - records_pos.min(axis=0)

        mean_rpy = np.rad2deg(records_rpy.mean(axis=0))
        std_rpy = np.rad2deg(records_rpy.std(axis=0, ddof=1))
        peak_rpy = np.rad2deg(records_rpy.max(axis=0) - records_rpy.min(axis=0))

        print(f"\n  --- 位置 ---")
        print(f"  目标 (m):        x={target_pos[0]:+.4f}  y={target_pos[1]:+.4f}  "
              f"z={target_pos[2]:+.4f}")
        print(f"  均值 (m):        x={mean_pos[0]:+.4f}  y={mean_pos[1]:+.4f}  "
              f"z={mean_pos[2]:+.4f}")
        print(f"  标准差 (mm):     x={std_pos[0]:.4f}  y={std_pos[1]:.4f}  "
              f"z={std_pos[2]:.4f}")
        print(f"  峰峰值 (mm):     x={peak_pos[0]*1000:.4f}  y={peak_pos[1]*1000:.4f}  "
              f"z={peak_pos[2]*1000:.4f}")
        print(f"  ─────────────────────────")
        print(f"  欧氏距离误差 (mm): 均值={pos_err_each.mean():.4f}  "
              f"最大={pos_err_each.max():.4f}  "
              f"标准差={pos_err_each.std(ddof=1):.4f}")

        print(f"\n  --- 姿态 ---")
        print(f"  目标 (deg):       roll={target_rpy[0]*180/np.pi:+.2f}  "
              f"pitch={target_rpy[1]*180/np.pi:+.2f}  "
              f"yaw={target_rpy[2]*180/np.pi:+.2f}")
        print(f"  均值 (deg):       roll={mean_rpy[0]:+.2f}  "
              f"pitch={mean_rpy[1]:+.2f}  yaw={mean_rpy[2]:+.2f}")
        print(f"  标准差 (deg):     roll={std_rpy[0]:.4f}  "
              f"pitch={std_rpy[1]:.4f}  yaw={std_rpy[2]:.4f}")
        print(f"  峰峰值 (deg):     roll={peak_rpy[0]:.4f}  "
              f"pitch={peak_rpy[1]:.4f}  yaw={peak_rpy[2]:.4f}")

        print("\n回零位...")
        arm.move_home(duration=2.0)
        time.sleep(1)

    print("\n测试完成，已下电。")


def _compute_target_pose(arm):
    """用 FK 计算目标关节角对应的末端位姿。"""
    from reBotArm_control_py.kinematics import compute_fk, pad_q_for_model
    import pinocchio as pin
    # RS 模型含夹爪自由度，须将 6 维臂关节角补齐到模型维度
    q_full = pad_q_for_model(arm._model, TARGET_Q, arm._n)
    pos, rot, _ = compute_fk(arm._model, q_full)
    rpy = pin.rpy.matrixToRpy(rot)
    return pos, rpy


if __name__ == "__main__":
    main()
