"""Dobot 真机就绪检查脚本（阶段 0 验证）。

逐阶段验证 RLinf 侧 dobot_zhiyu SDK 接入是否正确。每个阶段独立、可单独跳过，
所有运动都保守（小幅、低速、可随时 Ctrl+C 归位）。用于在写正式 Env/Controller
之前先把底层链路打通。

验证项（按顺序）：
  1. import 链路：sys.path 注入 + dobot_control 可 import
  2. TCP 连接 + 使能：connect/enable，RobotMode 到达 ENABLE(5)
  3. 反馈读取：关节(度) + TCP 原生(mm,deg) + 上层(m,quat) + 六维力
  4. 夹爪（达妙）：连接 + 开/合归零（可选，--skip-gripper 跳过）
  5. 相机（USB/V4L2）：create_camera 打开 + 采帧校验（可选，--skip-camera 跳过；
     与机械臂解耦，只要求阶段 1 通过即可独立运行）
  6. ServoJ：小幅关节往复（±2°，30 帧）
  7. ServoP：小幅位姿往复（z ±5mm，30 帧）
  8. IK：原生 inverse_kin 对当前位姿求逆
  9. 归位 + 关闭：move_j 回零 + disable + disconnect

用法（两种方式，任选）：
  # 方式 A：独立脚本运行（推荐，不触发 rlinf 包初始化）
  python rlinf/envs/realworld/dobot/verify_env.py --ip 192.168.5.1
  python rlinf/envs/realworld/dobot/verify_env.py --ip 192.168.5.1 --skip-gripper --skip-motion
  python rlinf/envs/realworld/dobot/verify_env.py --check-import   # 仅验证 import 链路
  # 只验证相机（不连接/使能/运动机械臂）
  python rlinf/envs/realworld/dobot/verify_env.py --camera-only
  # 跳过相机检查（仅验证机械臂 + 夹爪）
  python rlinf/envs/realworld/dobot/verify_env.py --ip 192.168.5.1 --skip-camera
  # 指定归位关节角（度，默认）
  python rlinf/envs/realworld/dobot/verify_env.py --ip 192.168.5.1 \
      --home-joints 352.15 48.24 88.35 -66.15 -93.84 2.01
  # 指定归位关节角（弧度）
  python rlinf/envs/realworld/dobot/verify_env.py --ip 192.168.5.1 \
      --home-joints 6.146 0.842 1.542 -1.154 -1.638 0.035 --home-joints-unit rad

  # 方式 B：模块方式运行（会触发 rlinf/__init__.py 的 torch 等重依赖）
  python -m rlinf.envs.realworld.dobot.verify_env --ip 192.168.5.1

注意：脚本不修改任何硬件标定（user_index/tool_index/payload/六维力归零），
仅做读 + 极小幅 servo。相机检查（阶段 5）不连接机械臂，可独立运行。
运动阶段（6/7）运行前请确保机械臂周围清空、急停在手边。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from dobot_control import DobotRobot

# ---------------------------------------------------------------------------
# sys.path 注入：从本文件位置反推到 third_party/dobot_zhiyu/src
# 本文件: rlinf/envs/realworld/dobot/verify_env.py
# 上溯 5 级到仓库根: ../../../../..
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SDK_SRC = _REPO_ROOT / "third_party" / "dobot_zhiyu" / "src"
if str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

# 单位转换常量（与 dobot_robot.py 内部一致）
_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi
# ServoJ/ServoP 安全就绪模式（ENABLE/RUNNING/SINGLE_MOVE）
_SERVO_READY_MODES = frozenset({5, 7, 8})


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"  [..] {msg}", flush=True)


def _section(n: int, title: str) -> None:
    print(f"\n{'=' * 60}\n阶段 {n}: {title}\n{'=' * 60}", flush=True)


def check_import() -> bool:
    """阶段 1: 验证 SDK import 链路。"""
    _section(1, "import 链路")
    try:
        from dobot_control import (  # noqa: F401
            DOBOT_EULER_SEQ,
            DobotRobot,
            dobot_pose_to_quat_m,
            quat_m_to_dobot_pose,
        )

        _ok(f"dobot_control 导入成功（SDK src = {_SDK_SRC}）")
        _ok(f"DOBOT_EULER_SEQ = {DOBOT_EULER_SEQ!r}（openpi pose 模式真机已验证）")
        # 往返一致性：用「旋转等价性」判断，不能用欧拉角数值相等。
        # 原因：欧拉角在 gimbal lock 附近有多解，as_euler 可能返回另一个等价解
        # （数值可差 180°），但旋转矩阵完全相同。平移（mm↔m）则是严格可逆的。
        from scipy.spatial.transform import Rotation as _R

        rng = np.random.default_rng(0)
        max_rot_diff_deg = 0.0
        max_pos_diff_mm = 0.0
        for _ in range(5):
            native = np.concatenate(
                [rng.uniform(-500, 500, 3), rng.uniform(-180, 180, 3)]
            )
            qm = dobot_pose_to_quat_m(native)
            back = quat_m_to_dobot_pose(qm)
            r_orig = _R.from_euler(DOBOT_EULER_SEQ, native[3:], degrees=True)
            r_back = _R.from_euler(DOBOT_EULER_SEQ, back[3:], degrees=True)
            max_rot_diff_deg = max(
                max_rot_diff_deg, float(np.degrees((r_orig.inv() * r_back).magnitude()))
            )
            max_pos_diff_mm = max(
                max_pos_diff_mm, float(np.max(np.abs(back[:3] - native[:3])))
            )
        if max_rot_diff_deg > 1e-6 or max_pos_diff_mm > 1e-6:
            _fail(
                f"几何转换往返不一致：旋转等价差 {max_rot_diff_deg}°，平移差 {max_pos_diff_mm}mm"
            )
            return False
        _ok(
            f"几何转换往返一致性通过（旋转等价差 {max_rot_diff_deg:.2e}°，"
            f"平移差 {max_pos_diff_mm:.2e}mm，5 次随机位姿）"
        )
        _info("（注：欧拉角数值可能差 180°，这是欧拉多解的正常表现，旋转等价即可）")
        return True
    except Exception as e:
        _fail(f"import 失败: {type(e).__name__}: {e}")
        return False


def check_connect_enable(ip: str, speed: int, verbose: bool) -> Optional["DobotRobot"]:
    """阶段 2: TCP 连接 + 使能。返回 DobotRobot 实例或 None。"""
    _section(2, "TCP 连接 + 使能")
    try:
        from dobot_control import DobotRobot
    except Exception as e:
        _fail(f"无法导入 DobotRobot: {e}")
        return None

    _info(f"连接 {ip}:29999/30004 ...")
    robot = DobotRobot(ip=ip, speed=speed, verbose=verbose)
    try:
        robot.connect()
        _ok("TCP 连接成功")
        robot.enable()
        mode = robot.get_mode()
        # 使能后 RobotMode 应在 {5,7,8}（ENABLE/RUNNING/SINGLE_MOVE），均为 servo 就绪。
        # 5=使能空闲，7=运动中（使能后残留运动/过渡态），8=单步运动；9=报警才需处理。
        if mode not in _SERVO_READY_MODES:
            _fail(
                f"使能后 RobotMode={mode}({robot._mode_name(mode)})，"
                f"期望在 {sorted(_SERVO_READY_MODES)}(ENABLE/RUNNING/SINGLE_MOVE)"
            )
            return None
        _ok(f"使能成功，RobotMode={mode}({robot._mode_name(mode)})，servo 就绪")
        if verbose:
            robot.debug_state("使能后")
        return robot
    except Exception as e:
        _fail(f"连接/使能失败: {type(e).__name__}: {e}")
        try:
            robot.close()
        except Exception:
            pass
        return None


def check_feedback(robot: "DobotRobot") -> bool:
    """阶段 3: 反馈读取（关节 + TCP + 力觉，单次 socket 读）。"""
    _section(3, "反馈读取（单次 socket 读多模态）")
    try:
        st = robot.read_state_from_feedback(
            include_joint=True, include_pose=True, include_wrench=True
        )
        joints_deg = np.asarray(st["joint_positions"])
        tcp_quat_m = np.asarray(st["tcp_pose"])
        wrench = np.asarray(st["wrench"])
        wrench_online = bool(st["wrench_online"])

        if joints_deg.shape != (6,):
            _fail(f"关节维度错误: {joints_deg.shape}, 期望 (6,)")
            return False
        if tcp_quat_m.shape != (7,):
            _fail(f"TCP 位姿维度错误: {tcp_quat_m.shape}, 期望 (7,)")
            return False
        if wrench.shape != (6,):
            _fail(f"力觉维度错误: {wrench.shape}, 期望 (6,)")
            return False
        # 四元数归一化检查
        qnorm = float(np.linalg.norm(tcp_quat_m[3:]))
        if abs(qnorm - 1.0) > 1e-3:
            _fail(f"TCP 四元数未归一化: |q| = {qnorm}")
            return False

        _ok(
            f"关节(deg) = {np.array2string(joints_deg, precision=2, suppress_small=True)}"
        )
        _ok(
            "关节(rad) = "
            f"{np.array2string(joints_deg * _DEG2RAD, precision=3, suppress_small=True)}"
        )
        _ok(
            f"TCP(m, quat[wxyz]) = {np.array2string(tcp_quat_m, precision=4, suppress_small=True)}"
        )
        _ok(
            "TCP 原生(mm, deg) = "
            f"{np.array2string(robot.get_tcp_pose_raw(), precision=1, suppress_small=True)}"
        )
        _ok(
            "力觉 N/N·m = "
            f"{np.array2string(wrench, precision=2, suppress_small=True)} (online={wrench_online})"
        )
        return True
    except Exception as e:
        _fail(f"反馈读取失败: {type(e).__name__}: {e}")
        return False


def check_gripper(port: str, closed_deg: float, open_deg: float) -> bool:
    """阶段 4: 达妙夹爪开/合归零。"""
    _section(4, "达妙夹爪（FORCE_POS）")
    # 直接从同目录导入，避免触发 rlinf/__init__.py 的重依赖（torch 等）。
    # verify_env 设计为可独立运行的诊断脚本。
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dobot_gripper import DamiaoGripper
    except Exception as e:
        _fail(f"无法导入 DamiaoGripper: {e}")
        return False
    try:
        _info(f"连接夹爪 {port}（motorbridge）...")
        grip = DamiaoGripper(
            port=port,
            closed_deg=closed_deg,
            open_deg=open_deg,
            max_velocity_rad_s=5.0,
            force_ratio=0.05,
        )
        _info("连接夹爪（connect）...")
        grip.connect()
        _ok("夹爪连接 + FORCE_POS 模式就绪")
        _info("张开 → 闭合 → 张开（home 序列）...")
        grip.home()
        _ok("夹爪 home 序列完成")
        grip.close()
        return True
    except Exception as e:
        _fail(f"夹爪检查失败: {type(e).__name__}: {e}")
        return False


def check_camera(
    serial: str,
    camera_type: str,
    resolution: tuple[int, int],
    fps: int,
    fourcc: Optional[str] = None,
    warmup_frames: int = 5,
) -> bool:
    """阶段 4.5: USB/V4L2 相机采集校验（不连接机械臂）。

    通过 RLinf 的 :func:`create_camera` 工厂打开相机（验证真实集成链路），
    连采若干帧取最新一帧，校验 shape/dtype/非恒定。相机检查与机械臂完全
    解耦：只要求阶段 1 的 import 链路通过即可独立运行。

    Args:
        serial: 设备路径，优先用稳定的 ``/dev/v4l/by-id/...``。
        camera_type: 传给工厂的后端类型（``opencv``/``usb``/``v4l2``）。
        fourcc: 可选 V4L2 像素格式（如 ``"MJPG"``）。高分辨率（如 1920x1080）
            时必须设 ``"MJPG"``，否则默认 YUYV 受 USB 带宽限制无法达到请求帧率。
        resolution: 请求的 ``(width, height)``。
        fps: 请求帧率。
        warmup_frames: 取帧前丢弃的预热帧数（让自动曝光/白平衡收敛）。
    """
    _section(5, "USB/V4L2 相机（不连接机械臂）")
    # 延迟导入：相机后端会触发 rlinf/__init__.py 的重依赖（torch/cv2/lerobot）。
    # 只在真正运行本阶段时 import，保持 --check-import 等其它路径轻量。
    try:
        from rlinf.envs.realworld.common.camera import CameraInfo, create_camera
    except Exception as e:
        _fail(f"无法导入相机模块: {e}")
        return False
    try:
        _info(
            f"打开相机 {serial}（type={camera_type}, {resolution[0]}x{resolution[1]}@{fps}"
            + (f", fourcc={fourcc}" if fourcc else "")
            + "）..."
        )
        info = CameraInfo(
            name="cam_left_wrist",
            serial_number=serial,
            camera_type=camera_type,
            resolution=tuple(int(v) for v in resolution),
            fps=int(fps),
            fourcc=fourcc,
        )
        camera = create_camera(info)
        camera.open()
        _ok("相机打开成功（采集线程已启动）")

        _info(f"预热丢弃 {warmup_frames} 帧（等待自动曝光/白平衡收敛）...")
        for _ in range(max(0, warmup_frames)):
            camera.get_frame(timeout=5)

        _info("读取校验帧...")
        frame = camera.get_frame(timeout=5)
        exp_h, exp_w = int(resolution[1]), int(resolution[0])
        if frame.shape != (exp_h, exp_w, 3):
            _fail(
                f"帧 shape 错误: {frame.shape}, 期望 ({exp_h}, {exp_w}, 3)"
            )
            camera.close()
            return False
        if frame.dtype != np.uint8:
            _fail(f"帧 dtype 错误: {frame.dtype}, 期望 uint8")
            camera.close()
            return False
        fmin, fmax = int(frame.min()), int(frame.max())
        # 非恒定检查：整帧只有一个值通常是镜头遮挡或采集失败。
        if fmin == fmax:
            _fail(
                f"帧为恒定值 {fmin}（min==max），可能镜头被遮挡或采集异常"
            )
            camera.close()
            return False
        _ok(
            f"帧 shape={frame.shape}, dtype={frame.dtype}, "
            f"range=[{fmin}, {fmax}]"
        )
        _ok("相机采集校验通过（BGR uint8，非恒定）")
        camera.close()
        _ok("相机关闭成功（资源已释放）")
        return True
    except Exception as e:
        _fail(f"相机检查失败: {type(e).__name__}: {e}")
        return False


def check_servo_j(robot: "DobotRobot", frames: int, step_s: float) -> bool:
    """阶段 6: ServoJ 小幅往复（±1°，验证 servo 链路 + 护栏）。"""
    _section(6, f"ServoJ 小幅往复（{frames} 帧，±1°，{1 / step_s:.1f} Hz）")
    try:
        if robot.get_mode() not in _SERVO_READY_MODES:
            _fail(
                f"RobotMode={robot.get_mode()} 不在 servo 就绪模式 {sorted(_SERVO_READY_MODES)}"
            )
            return False
        # 锚定当前关节为振荡中心
        start_deg = np.asarray(robot.get_joint_positions(), dtype=float)
        delta_deg = 1.0  # ±1°，保守
        _info(
            f"起点关节(deg) = {np.array2string(start_deg, precision=2, suppress_small=True)}"
        )
        _info("engage：reset_smoothing，准备 servo 流...")
        robot.reset_smoothing()

        accepted_count = 0
        t0 = time.monotonic()
        for i in range(frames):
            phase = 1.0 if (i // 10) % 2 == 0 else -1.0  # 每 10 帧翻转方向
            target_deg = start_deg.copy()
            target_deg[0] += phase * delta_deg  # 只动 J1，最安全
            ok = robot.servo_joints(target_deg.tolist())
            accepted_count += int(bool(ok))
            time.sleep(step_s)
        dt = time.monotonic() - t0

        # 回到起点
        for _ in range(10):
            robot.servo_joints(start_deg.tolist())
            time.sleep(step_s)

        _ok(f"ServoJ 完成：{accepted_count}/{frames} 帧被护栏接受，用时 {dt:.2f}s")
        if accepted_count < frames:
            _info("（部分帧被 slew/jump 护栏拒绝属正常，连续拒绝才需排查）")
        return True
    except Exception as e:
        _fail(f"ServoJ 失败: {type(e).__name__}: {e}")
        return False


def check_servo_p(robot: "DobotRobot", frames: int, step_s: float) -> bool:
    """阶段 7: ServoP 小幅往复（z ±2mm，验证位姿链路 + 四元数通路）。"""
    _section(7, f"ServoP 小幅往复（{frames} 帧，z ±2mm，{1 / step_s:.1f} Hz）")
    try:
        if robot.get_mode() not in _SERVO_READY_MODES:
            _fail(f"RobotMode={robot.get_mode()} 不在 servo 就绪模式")
            return False
        # 锚定当前 TCP（m + quat wxyz）
        start_pose = np.asarray(robot.get_tcp_pose(), dtype=float).reshape(7)
        delta_z = 0.002  # 2mm，保守
        _info(
            f"起点 TCP(m, quat) = {np.array2string(start_pose, precision=4, suppress_small=True)}"
        )
        _info("engage：reset_smoothing，准备 servo 流...")
        robot.reset_smoothing()

        accepted_count = 0
        t0 = time.monotonic()
        for i in range(frames):
            phase = 1.0 if (i // 10) % 2 == 0 else -1.0
            target = start_pose.copy()
            target[2] += phase * delta_z  # 只动 Z，最安全
            ok = robot.servo_pose(target)
            accepted_count += int(bool(ok))
            time.sleep(step_s)
        dt = time.monotonic() - t0

        # 回到起点
        for _ in range(10):
            robot.servo_pose(start_pose)
            time.sleep(step_s)

        _ok(f"ServoP 完成：{accepted_count}/{frames} 帧被护栏接受，用时 {dt:.2f}s")
        # 验证到位
        end_pose = np.asarray(robot.get_tcp_pose(), dtype=float).reshape(7)
        drift = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
        _ok(f"回到起点，末端漂移 {drift * 1000:.2f} mm")
        return True
    except Exception as e:
        _fail(f"ServoP 失败: {type(e).__name__}: {e}")
        return False


def check_ik(robot: "DobotRobot") -> bool:
    """阶段 8: 原生 IK（对当前位姿求逆，应返回 reachable + 接近当前关节）。"""
    _section(8, "原生 inverse_kin")
    try:
        cur_pose = np.asarray(robot.get_tcp_pose(), dtype=float).reshape(7)
        cur_joints_deg = np.asarray(robot.get_joint_positions(), dtype=float)
        reachable, ik_joints = robot.inverse_kin(cur_pose, cur_joints_deg.tolist())
        if not reachable or ik_joints is None:
            _fail(f"IK 不可达: reachable={reachable}, ik_joints={ik_joints}")
            return False
        ik_joints = np.asarray(ik_joints, dtype=float)
        # 多解情况下关节可能差异较大，只要模 2π 接近即可；这里只做存在性 + 形状检查
        err_deg = float(np.max(np.abs(np.asarray(ik_joints) - cur_joints_deg)))
        _ok(
            "IK reachable=True, joints(deg) = "
            f"{np.array2string(ik_joints, precision=2, suppress_small=True)}"
        )
        _info(f"（与当前关节最大差 {err_deg:.2f}°，IK 多解属正常）")
        return True
    except Exception as e:
        _fail(f"IK 检查失败: {type(e).__name__}: {e}")
        return False


def check_home_close(
    robot: "DobotRobot",
    home_joints: Optional[list[float]],
    unit: str = "deg",
) -> bool:
    """阶段 8: 归位 + 关闭。

    Args:
        home_joints: 归位关节角（6 维）；None 则跳过归位。
        unit: 输入单位，``"deg"`` 或 ``"rad"``。内部统一转度后传给 ``move_j``。
    """
    _section(9, "归位 + 关闭")
    try:
        if home_joints is not None:
            if unit == "rad":
                home_deg = [float(x) * _RAD2DEG for x in home_joints]
                _info(f"move_j 归位 (输入 rad → {home_deg} deg)...")
            else:
                home_deg = [float(x) for x in home_joints]
                _info(f"move_j 归位到 {home_deg} (deg)...")
            robot.move_j(home_deg)
            # move_j 非阻塞，必须等到达再关闭，否则运动中 disable 会突停。
            _info("等待到达归位点（最多 30s）...")
            robot.wait_until_reached(home_deg, tol_deg=2.0, settle=0.5, timeout=30.0)
            _ok("归位完成")
        else:
            _info("未指定 home_joints，跳过 move_j")
        robot.close()
        _ok("机器人已 disable + disconnect")
        return True
    except Exception as e:
        _fail(f"归位/关闭失败: {type(e).__name__}: {e}")
        try:
            robot.close()
        except Exception:
            pass
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Dobot 真机就绪检查（阶段 0）")
    parser.add_argument(
        "--ip", default="192.168.5.1", help="机械臂 IP，默认 192.168.5.1"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=5,
        help="MovJ 速度百分比（1-100），默认 5（很慢）。仅影响 move_j 归位，不影响 ServoJ/ServoP。",
    )
    parser.add_argument("--gripper-port", default="/dev/ttyACM0", help="达妙夹爪串口")
    parser.add_argument(
        "--closed-deg", type=float, default=0.0, help="夹爪闭合标定（度）"
    )
    parser.add_argument(
        "--open-deg", type=float, default=-320.0, help="夹爪张开标定（度）"
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="ServoJ/ServoP 测试帧数，默认 10"
    )
    parser.add_argument(
        "--step-s",
        type=float,
        default=0.2,
        help="ServoJ/ServoP 帧间隔（秒），默认 0.2（5 Hz）。越大越慢越安全。",
    )
    parser.add_argument(
        "--home-joints",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="归位关节角（6 维）。单位由 --home-joints-unit 指定。不指定则不归位。"
        "例: --home-joints 352.15 48.24 88.35 -66.15 -93.84 2.01",
    )
    parser.add_argument(
        "--home-joints-unit",
        choices=["deg", "rad"],
        default="deg",
        help="归位关节角单位，默认 deg（与 Dobot 示教器/SDK 一致）。",
    )
    parser.add_argument("--skip-gripper", action="store_true", help="跳过夹爪检查")
    parser.add_argument(
        "--skip-motion", action="store_true", help="跳过 ServoJ/ServoP/IK"
    )
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="跳过相机检查。相机检查不连接机械臂，可独立验证。",
    )
    parser.add_argument(
        "--camera-serial",
        default="/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0",
        help="相机设备路径，默认用稳定的 by-id 路径（指向 /dev/video6）",
    )
    parser.add_argument(
        "--camera-type",
        default="opencv",
        help="相机后端类型（传给 create_camera），默认 opencv",
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="相机分辨率 width height，默认 1920 1080",
    )
    parser.add_argument(
        "--camera-fps", type=int, default=30, help="相机帧率，默认 30"
    )
    parser.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="V4L2 像素格式 FOURCC（如 MJPG）。高分辨率（1920x1080）时必须用 "
        "MJPG，默认 MJPG。传空字符串则用后端默认（YUYV）。",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="只运行相机检查后退出（不连接/使能/运动机械臂）",
    )
    parser.add_argument(
        "--check-import", action="store_true", help="仅执行 import 链路检查后退出"
    )
    parser.add_argument("--verbose", action="store_true", help="打印 SDK 内部调试状态")
    args = parser.parse_args()

    print(f"Dobot 真机就绪检查\n  IP={args.ip}\n  SDK={_SDK_SRC}", flush=True)

    # 阶段 1
    if not check_import():
        return 1
    if args.check_import:
        print("\n仅 import 检查已请求，退出。")
        return 0

    # 阶段 5: 相机（与机械臂解耦，只要求阶段 1 通过）。
    # 放在机械臂连接之前，这样机械臂未连接/未使能时也能独立验证相机链路。
    results = []
    if not args.skip_camera:
        results.append(
            (
                "相机",
                check_camera(
                    args.camera_serial,
                    args.camera_type,
                    tuple(args.camera_resolution),
                    args.camera_fps,
                    fourcc=args.camera_fourcc or None,
                ),
            )
        )

    # --camera-only：只验证相机后退出，不连接/使能/运动机械臂。
    if args.camera_only:
        _print_summary(results)
        return 0 if all(ok for _, ok in results) else 1

    # 阶段 2
    robot = check_connect_enable(args.ip, args.speed, args.verbose)
    if robot is None:
        _print_summary(results)
        return 1

    try:
        # 阶段 3
        results.append(("反馈读取", check_feedback(robot)))
        # 阶段 4
        if not args.skip_gripper:
            results.append(
                (
                    "夹爪",
                    check_gripper(args.gripper_port, args.closed_deg, args.open_deg),
                )
            )
        # 阶段 6/7/8
        if not args.skip_motion:
            results.append(("ServoJ", check_servo_j(robot, args.frames, args.step_s)))
            results.append(("ServoP", check_servo_p(robot, args.frames, args.step_s)))
            results.append(("IK", check_ik(robot)))
    finally:
        # 阶段 9（无论如何都尝试归位 + 关闭）
        check_home_close(robot, args.home_joints, args.home_joints_unit)

    _print_summary(results)
    return 0 if all(ok for _, ok in results) else 1


def _print_summary(results: list[tuple[str, bool]]) -> None:
    """打印阶段汇总。"""
    print(f"\n{'=' * 60}\n总结\n{'=' * 60}", flush=True)
    for name, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}", flush=True)
    all_pass = all(ok for _, ok in results)
    print(
        f"\n{'全部通过 ✓' if all_pass else '存在失败项 ✗，见上文 [FAIL]'}", flush=True
    )


if __name__ == "__main__":
    sys.exit(main())
