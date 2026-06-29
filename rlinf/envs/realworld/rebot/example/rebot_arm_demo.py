#!/usr/bin/env python3
"""reBotArm 使用示例 —— 演示 6 大核心功能。

用法:
    uv run python example/rebot_arm_demo.py
    REBOT_CHANNEL=/dev/cu.usbmodem1101 uv run python example/rebot_arm_demo.py

功能:
    1. 读取末端位姿
    2. 读取关节角
    3. 读取夹爪
    4. 末端位置控制
    5. 关节角控制
    6. 夹爪控制
    7. 夹爪限力控制（力位混合 FORCE_POS）
"""
import time

import numpy as np

from reBotArm_control_py import reBotArm


def main():
    print("=" * 60)
    print(" reBotArm 演示")
    print("=" * 60)

    with reBotArm() as arm:
        print("\n--- 1. 读取关节角 ---")
        q = arm.get_joint_positions()
        print(f"  关节角 (rad): {q}")

        print("\n--- 2. 读取末端位姿 ---")
        pos, rpy = arm.get_end_effector_pose()
        print(f"  位置 (m):     x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")
        print(f"  姿态 (rad):   roll={rpy[0]:+.3f}  pitch={rpy[1]:+.3f}  yaw={rpy[2]:+.3f}")

        print("\n--- 3. 读取夹爪 ---")
        g = arm.get_gripper_position()
        print(f"  夹爪位置: {g:.3f}")

        # print("\n--- 4. 末端位置控制 ---")
        # print("  移动到 (0.3, 0.0, 0.2) ...")
        # ok = arm.move_pose(0.3, 0.0, 0.3, duration=2.0)
        # print(f"  结果: {'ok' if ok else 'fail'}")

        # print("\n--- 5. 关节角控制 ---")
        # q_target = np.array([1.304, -1.812, -1.141, 0.803, -0.035, 0.136])
        # print(f"  移动到关节角: {q_target}")
        # arm.move_joints(q_target, duration=5)
        # print("  完成")
        # time.sleep(3)

        # print("\n--- 6. 夹爪控制 ---")
        # print("  夹爪张开...")
        # arm.move_open_gripper()
        # time.sleep(1)
        # print("  夹爪闭合...")
        # arm.move_close_gripper()

        print("\n--- 7. 夹爪限力控制（力位混合 FORCE_POS）---")
        # 先张开
        print("  张开夹爪...")
        arm.move_open_gripper()
        time.sleep(1.5)

        # 限力闭合：朝闭合位置(0)运动，但出力只用最大力矩的 30%。
        # 若夹到物体到不了目标位置，电机会停在力矩上限处稳定夹持，不会硬怼夹碎。
        ratio = 0.3
        print(f"  限力闭合（ratio={ratio}，即 {int(ratio*100)}% 最大出力）...")
        arm.move_gripper_force(pos=0.0, ratio=ratio)
        time.sleep(2.0)

        # 读回实际停住的位置：夹到东西时它 != 目标 0，而是顶住物体的位置
        g = arm.get_gripper_position()
        print(f"  夹爪实际停住位置: {g:.3f}（夹到物体时不会等于目标 0）")

        # 想夹得更紧就调大 ratio，例如：
        # arm.move_gripper_force(pos=0.0, ratio=0.6)   # 60% 出力，夹得更紧

        print("  松开...")
        arm.move_open_gripper()
        time.sleep(1.5)

        print("\n--- 回零 ---")
        arm.move_home(duration=2.0)
        time.sleep(1)
        print("  已回零")

    print("\n完成，已安全断开连接。")


if __name__ == "__main__":
    main()
