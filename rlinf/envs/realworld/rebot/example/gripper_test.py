#!/usr/bin/env python3
"""夹爪交互式测试工具（基于新版分组架构 RebotArm.gripper 组）。

直接驱动配置中的 gripper 关节组，演示 MIT / POS_VEL / VEL 三种模式。
电机类型（DM / RS）由 config/rebotarm.yaml 的 hardware_yaml 决定。

用法:
    python example/gripper_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm

HELP = """
夹爪交互测试
-----------
z  - 设零（全部电机）
m  - 切换控制模式 (MIT / POS_VEL / VEL)
c  - 发送控制指令
s  - 显示当前状态
h  - 显示帮助
q  - 停止循环 → 失能 → 退出
"""


class GripperTerminal:
    def __init__(self):
        self.rb = RebotArm()
        self.rb.connect()
        if not self.rb.has_gripper:
            raise RuntimeError("当前配置未定义 gripper 组，请检查 hardware_yaml")
        self.g = self.rb.gripper          # gripper JointGroup
        self.g.mode_mit()
        self.g.enable()
        print(f"使能完成，当前模式: {self.g.mode}")
        self._show_state()

        self._target_pos = 0.0
        self._target_vel = 0.0
        self._mit_tau = 0.0

        self._running = True
        self.rb.start_control_loop(self._loop, rate=100.0)
        print(f"控制循环已启动 {self.rb.rate} Hz")

    def _loop(self, ref: RebotArm, dt: float):
        pos = np.array([self._target_pos])
        if self.g.mode == "mit":
            self.g.send_mit(pos, vel=np.array([self._target_vel]),
                            tau=np.array([self._mit_tau]))
        elif self.g.mode == "pos_vel":
            self.g.send_pos_vel(pos)
        elif self.g.mode == "vel":
            self.g.send_vel(np.array([self._target_vel]))

    def _show_state(self):
        pos, vel, torq = self.rb.get_state()
        p, v, t = float(pos[-1]), float(vel[-1]), float(torq[-1])
        print(f"  pos={p:+.4f} rad  vel={v:+.4f} rad/s  torq={t:+.4f} Nm  [mode={self.g.mode}]")

    def run(self):
        print(HELP)
        while self._running:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "q"

            if not cmd:
                continue

            if cmd == "q":
                print("停止控制循环 → 失能 → 退出...")
                self.rb.stop_control_loop()
                self.g.disable()
                self.rb.disconnect()
                self._running = False
                break

            elif cmd == "h":
                print(HELP)

            elif cmd == "s":
                self._show_state()

            elif cmd == "z":
                print("设零...")
                self.rb.set_zero()

            elif cmd == "m":
                print(f"当前模式: {self.g.mode}，切换到: [0]MIT  [1]POS_VEL  [2]VEL")
                sel = input("  > ").strip()
                if sel == "0":
                    self.g.mode_mit()
                    print("已切换到 MIT")
                elif sel == "1":
                    self.g.mode_pos_vel()
                    print("已切换到 POS_VEL")
                elif sel == "2":
                    self.g.mode_vel()
                    print("已切换到 VEL")
                else:
                    print("无效选择")

            elif cmd == "c":
                if self.g.mode == "mit":
                    try:
                        p = float(input("  pos (rad): ").strip() or "0.0")
                        v = float(input("  vel (rad/s) [0.0]: ").strip() or "0.0")
                        tau = float(input("  tau (Nm) [0.0]: ").strip() or "0.0")
                        self._target_pos, self._target_vel, self._mit_tau = p, v, tau
                        print(f"已更新: pos={p}, vel={v}, tau={tau}")
                    except ValueError:
                        print("输入无效")
                elif self.g.mode == "pos_vel":
                    try:
                        p = float(input("  pos (rad): ").strip() or "0.0")
                        self._target_pos = p
                        print(f"已更新: pos={p}")
                    except ValueError:
                        print("输入无效")
                elif self.g.mode == "vel":
                    try:
                        v = float(input("  vel (rad/s): ").strip() or "0.0")
                        self._target_vel = v
                        print(f"已更新: vel={v}")
                    except ValueError:
                        print("输入无效")
                self._show_state()

            else:
                print(f"未知指令: {cmd}，按 h 查看帮助")


if __name__ == "__main__":
    GripperTerminal().run()
