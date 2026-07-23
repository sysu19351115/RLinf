"""达妙电机夹爪（motorbridge 直驱，FORCE_POS 力位混合模式）。

基于已验证的 dm_gripper.py 实现，只依赖 ``motorbridge``。
电机 ID 为 0x01（独立单电机，非 reBot 臂上的 0x07）。
Dobot 夹爪恢复原 lightweight_teleop 标定：闭合=0°，张开=-320°。
配置层使用度，只有下发 motorbridge FORCE_POS 时转换为弧度。
"""

from __future__ import annotations

import math
import time
from typing import Optional

from motorbridge import Controller, Mode

_FORCE_POS = Mode.FORCE_POS


class DamiaoGripper:
    """达妙电机驱动的夹爪，FORCE_POS 力位混合控制。

    控制参数：
    - ``max_velocity_rad_s``：最大速度限制（rad/s），默认 10.0
    - ``force_ratio``：出力比例 0-1，0.05 表示最大出力的 5%
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 921600,
        motor_id: int = 0x01,
        feedback_id: int = 0x11,
        model: str = "4310",
        closed_deg: float = 0.0,
        open_deg: float = -320.0,
        max_velocity_rad_s: float = 10.0,
        force_ratio: float = 0.05,
        sleep=time.sleep,
    ):
        self.port = port
        self.baudrate = baudrate
        self.motor_id = motor_id
        self.feedback_id = feedback_id
        self.model = model
        self._closed_deg = float(closed_deg)
        self._open_deg = float(open_deg)
        if self._closed_deg == self._open_deg:
            raise ValueError("gripper open and closed positions must differ")
        self._max_velocity_rad_s = float(max_velocity_rad_s)
        if self._max_velocity_rad_s <= 0:
            raise ValueError("gripper max velocity must be positive")
        self._force_ratio = float(force_ratio)
        if not 0 < self._force_ratio <= 1:
            raise ValueError("gripper force ratio must be between 0 and 1")
        self._sleep = sleep

        self._ctrl: Optional[Controller] = None
        self._motor = None
        self._enabled = False

    # ---- 生命周期 ----
    def connect(self):
        """建串口桥 + 注册电机 + 使能 + 切 FORCE_POS 模式。

        对 /dev/ttyACM* 统一用 from_dm_serial（与 reBot SDK 一致）。
        控制模式用 FORCE_POS（力位混合），不是 MIT——已验证 dm_gripper.py
        可用，MIT 模式在此硬件上不可用。
        """
        if self.port.startswith("/dev/tty"):
            self._ctrl = Controller.from_dm_serial(self.port, self.baudrate)
        else:
            self._ctrl = Controller(self.port)
        self._motor = self._ctrl.add_damiao_motor(
            self.motor_id, self.feedback_id, self.model
        )
        self._sleep(0.2)
        self._ctrl.enable_all()
        self._sleep(0.3)  # 电机刚使能内部需要初始化时间

        # ensure_mode 带重试：电机刚使能时可能首帧不响应
        last_err = None
        for attempt in range(3):
            try:
                self._motor.ensure_mode(_FORCE_POS, 1000)
                break
            except Exception as e:
                last_err = e
                print(
                    f"⚠️ ensure_mode(FORCE_POS) 第 {attempt + 1}/3 次失败: {e}，重试..."
                )
                self._sleep(0.3)
        else:
            raise RuntimeError(f"达妙电机切 FORCE_POS 模式 3 次均失败: {last_err}")

        self._enabled = True
        return self

    def enable(self):
        if self._ctrl is not None:
            self._ctrl.enable_all()
            self._enabled = True

    def disable(self):
        if self._ctrl is not None:
            self._ctrl.disable_all()
            self._enabled = False

    def close(self):
        """安全关停（幂等）。"""
        if self._ctrl is None:
            return
        try:
            self.disable()
        except Exception:
            pass
        try:
            self._ctrl.shutdown()
        except Exception:
            pass
        try:
            self._ctrl.close()
        except Exception:
            pass
        self._ctrl = None
        self._motor = None
        self._enabled = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---- 控制 ----
    def move_to(self, pos_rad: float):
        """下发 FORCE_POS 指令到目标电机位（rad）。

        每次下发前重新 ensure_mode，与已验证的 dm_gripper.py 保持一致。
        """
        if not self._enabled or self._motor is None:
            raise RuntimeError("Gripper not connected. Call connect() first.")
        self._motor.ensure_mode(_FORCE_POS, 1000)
        self._motor.send_force_pos(pos_rad, self._max_velocity_rad_s, self._force_ratio)

    def move_to_deg(self, target_deg: float):
        """下发以度表示的目标；越界目标钳制到配置行程。"""
        low = min(self._closed_deg, self._open_deg)
        high = max(self._closed_deg, self._open_deg)
        self.move_to(math.radians(min(high, max(low, float(target_deg)))))

    def move_normalized(self, norm_01: float):
        """归一化值 [0,1] → 电机位 → 下发。0=闭合，1=张开。"""
        self.move_to(self.denormalize(norm_01))

    def _full_travel_settle_s(self) -> float:
        travel_s = (
            math.radians(abs(self._open_deg - self._closed_deg))
            / self._max_velocity_rad_s
        )
        return travel_s * 1.5 + 0.5

    def home(self):
        """按原遥操作流程全开后全闭，结束于闭合位置。

        每段行程后读反馈校验是否真的到位：FORCE_POS 出力（force_ratio）
        过小时电机可能纹丝不动，指令本身不会报错，必须靠位置反馈发现。
        """
        settle_s = self._full_travel_settle_s()
        print(
            f"🔧 夹爪复位：全开({self._open_deg:.0f}°) → 全闭({self._closed_deg:.0f}°) ..."
        )
        self.move_to_deg(self._open_deg)
        self._sleep(settle_s)
        self._warn_if_not_reached(self._open_deg, "全开")
        self.move_to_deg(self._closed_deg)
        self._sleep(settle_s)
        self._warn_if_not_reached(self._closed_deg, "全闭")

    def _warn_if_not_reached(
        self, target_deg: float, label: str, tolerance_deg: float = 15.0
    ):
        try:
            actual_deg = math.degrees(self.get_position())
        except Exception as e:
            print(f"⚠️ 夹爪复位后读取位置失败（无法确认{label}到位）: {e}")
            return
        if abs(actual_deg - target_deg) > tolerance_deg:
            print(
                f"⚠️ 夹爪未到达{label}位置：目标 {target_deg:.0f}°，实际 {actual_deg:.1f}°。"
                f"force_ratio={self._force_ratio} 可能出力不足，或软件零点未设在合爪位置。"
            )

    def open(self, *, wait: bool = False):
        """张开到 ``pos_open``。"""
        self.move_to_deg(self._open_deg)
        if wait:
            self._sleep(self._full_travel_settle_s())

    def close_gripper(self):
        """闭合到 ``pos_close``。"""
        self.move_to_deg(self._closed_deg)

    # ---- 读取 ----
    def _request_feedback(self, tries: int = 10) -> object:
        """请求并轮询一次状态反馈（最多重试 tries 次）。"""
        if self._motor is None or self._ctrl is None:
            raise RuntimeError("gripper is not connected")
        last_status = None
        for _ in range(tries):
            self._motor.request_feedback()
            self._ctrl.poll_feedback_once()
            self._sleep(0.005)
            st = self._motor.get_state()
            # 达妙状态字节：0=失能、1=使能，都是有效反馈；8 及以上为故障
            # （超压/欠压/过流/过温/丢通讯/过载），此时位置不可信
            if st is not None and getattr(st, "status_code", -1) in (0, 1):
                return st
            last_status = getattr(st, "status_code", None)
        raise RuntimeError(
            f"failed to receive valid gripper feedback after {tries} attempts "
            f"(last status={last_status})"
        )

    def get_position(self) -> float:
        """读当前电机位（rad）；无有效反馈时抛出异常。"""
        st = self._request_feedback()
        return float(st.pos)

    def get_normalized(self) -> float:
        """读当前电机位并转归一化 [0,1]。"""
        return self.normalize(self.get_position())

    # ---- 归一化转换 ----
    def normalize(self, pos_rad: float) -> float:
        """电机位（rad）→ 归一化 [0,1]。clip 防越界。"""
        denom = self.pos_open - self.pos_close
        if abs(denom) < 1e-9:
            return 0.0
        return min(1.0, max(0.0, (float(pos_rad) - self.pos_close) / denom))

    def denormalize(self, norm_01: float) -> float:
        """归一化 [0,1] → 电机位（rad）。"""
        normalized = min(1.0, max(0.0, float(norm_01)))
        return self.pos_close + normalized * (self.pos_open - self.pos_close)

    # ---- 标定值 ----
    @property
    def pos_close(self) -> float:
        return math.radians(self._closed_deg)

    @property
    def pos_open(self) -> float:
        return math.radians(self._open_deg)

    @property
    def closed_deg(self) -> float:
        return self._closed_deg

    @property
    def open_deg(self) -> float:
        return self._open_deg

    @property
    def enabled(self) -> bool:
        return self._enabled
