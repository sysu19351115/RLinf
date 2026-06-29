#!/usr/bin/env python3
"""拖动示教演示程序。

通过键盘按键控制示教流程：

    s      开始示教（进入重力补偿并记录）
    q      暂停示教，自动保存轨迹
    r      回放最近一次保存的轨迹
    ESC    退出程序

用法:
    uv run python example/teach_demo.py
    REBOT_CHANNEL=/dev/cu.usbmodem1101 uv run python example/teach_demo.py
"""
import sys
from pathlib import Path

import numpy as np

from reBotArm_control_py import reBotArm


# ------------------------------------------------------------------
# 跨平台单字符读取（无需回车）
# ------------------------------------------------------------------

def _getch_unix():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _getch_windows():
    import msvcrt
    return msvcrt.getch().decode()


try:
    getch = _getch_unix
except ImportError:
    getch = _getch_windows


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------

def _get_latest_record(dir_path: str = "records") -> Path | None:
    """获取 records 目录下最新的 .npy 轨迹文件。"""
    p = Path(dir_path)
    if not p.exists():
        return None
    npy_files = sorted(p.glob("*.npy"), key=lambda f: f.stat().st_mtime, reverse=True)
    return npy_files[0] if npy_files else None


def main():
    print("=" * 60)
    print(" 拖动示教演示")
    print("=" * 60)

    # 启动时扫描历史轨迹
    last_saved_path = _get_latest_record()
    if last_saved_path:
        print(f"\n检测到历史轨迹: {last_saved_path}")

    with reBotArm() as arm:
        print("\n机械臂已连接，正在回零...")
        arm.move_home(duration=1.0)
        print("回零完成，等待指令。\n")
        print("  [s]     开始示教（进入重力补偿并记录）")
        print("  [q]     暂停示教，自动保存轨迹")
        print("  [r]     回放最近一次保存的轨迹")
        print("  [ESC]   退出程序\n")

        while True:
            ch = getch()

            # ESC 键（0x1b）或 Ctrl+C（0x03）
            if ch in ("\x1b", "\x03"):
                print("\n退出...")
                if arm._teach_on:
                    arm.teach_mode(enable=False)
                break

            if ch == "s":
                if not arm._teach_on:
                    arm.teach_mode(enable=True)
                    print("  [开始] 已进入示教模式，可徒手拖动机械臂")
                else:
                    print("  [开始] 已在示教模式中")

            elif ch == "q":
                if arm._teach_on:
                    saved = arm.teach_mode(enable=False)
                    if saved:
                        last_saved_path = saved
                    q = arm.get_joint_positions()
                    deg = np.rad2deg(q)
                    traj = arm.get_teach_trajectory()
                    print(f"\n  [暂停] 关节角 (rad): {' '.join(f'{v:+.3f}' for v in q)}")
                    print(f"         关节角 (deg): {' '.join(f'{v:+.1f}' for v in deg)}")
                    if traj.size > 0:
                        print(f"         记录轨迹点数: {len(traj)}")
                    if last_saved_path:
                        print(f"         轨迹已保存: {last_saved_path}")
                else:
                    print("  [暂停] 当前未在示教模式")

            elif ch == "r":
                if last_saved_path is None:
                    print("  [回放] 没有可用的轨迹，请先示教并保存")
                    continue
                if arm._teach_on:
                    arm.teach_mode(enable=False)
                print(f"  [回放] 正在回放: {last_saved_path}")
                try:
                    arm.replay_trajectory(path=str(last_saved_path), rate=30)
                    print("  [回放] 完成")
                except Exception as e:
                    print(f"  [回放] 失败: {e}")

    print("\n已安全断开连接。")


if __name__ == "__main__":
    main()
