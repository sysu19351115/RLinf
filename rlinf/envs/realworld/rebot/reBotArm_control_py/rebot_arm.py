"""reBotArm —— 机械臂高层控制类。

封装底层通信、模式切换与控制循环，对外只暴露语义化的高层接口。

本类构建在官方新版 SDK 的统一分组架构之上：

  - 底层 :class:`~reBotArm_control_py.actuator.RebotArm` 持有 ``arm`` / ``gripper``
    两个 :class:`JointGroup`，自动从 ``config/rebotarm.yaml`` 指向的硬件配置
    （``rebotarm_rs.yaml`` / ``rebotarm_dm.yaml``）读取电机、PID、URDF 等参数。
  - 末端位姿由 :class:`~reBotArm_control_py.controllers.RebotArmEndPose` 控制
    （POS_VEL 模式 + SE(3) 轨迹规划 + CLIK 跟踪）。
  - 夹爪由本类直接驱动（POS_VEL / FORCE_POS），不进入末端控制循环，
    以保留限力夹持等语义。

支持分步操作（连接和上电分离）::

    arm = reBotArm()
    arm.connect()              # 仅建立通信
    q = arm.get_joint_positions()   # 不上电也能读关节角
    arm.power_on()             # 上电：切模式 + 使能 + 启动控制循环
    arm.move_pose(0.3, 0.0, 0.2, duration=2.0)
    arm.disconnect()

也支持上下文管理器一键完成::

    with reBotArm() as arm:
        arm.move_pose(0.3, 0.0, 0.2, duration=2.0)
        arm.move_joints([0.5, -0.3, 0.0, 0.0, 0.0, 0.0], duration=1.5)
        arm.move_gripper(0.5)

DM / RS 电机切换：修改 ``config/rebotarm.yaml`` 的 ``hardware_yaml`` 字段
（``rebotarm_rs.yaml`` / ``rebotarm_dm.yaml``）。该字段同时决定电机驱动与
运动学 URDF，保证执行器与运动学模型始终一致。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
import yaml

from motorbridge import CallError

from .actuator import RebotArm
from .actuator.rebotarm import _resolve_hw_cfg_path
from .kinematics import (
    load_robot_model,
    compute_fk,
    get_end_effector_frame_id,
    pad_q_for_model,
)
from .dynamics import load_dynamics_model, compute_generalized_gravity
from .controllers import RebotArmEndPose


class reBotArm:
    """reBotArm 高层控制接口（6-DOF 机械臂 + 可选夹爪）。

    自动管理控制循环与模式切换，所有运动指令默认阻塞直到完成。

    支持分步操作::

        arm = reBotArm()
        arm.connect()          # 仅建立通信
        q = arm.get_joint_positions()  # 不上电也能读取关节角
        arm.power_on()         # 上电：切模式 + 使能 + 启动控制循环
        arm.move_pose(0.3, 0.0, 0.2)
        arm.disconnect()

    也支持上下文管理器一键完成::

        with reBotArm() as arm:
            arm.move_pose(0.3, 0.0, 0.2)
    """

    def __init__(
        self,
        dt: Optional[float] = None,
    ) -> None:
        """初始化高层控制器。

        硬件类型（DM / RS）由 ``config/rebotarm.yaml`` 的 ``hardware_yaml`` 决定，
        执行器、运动学 URDF、末端帧均据此统一加载。

        Args:
            dt: 控制周期（秒），``None`` 时从配置文件的 ``rate`` 自动推导。
        """
        # ------------------------------------------------------------------
        # 1. 底层执行器（统一分组：arm + gripper）
        # ------------------------------------------------------------------
        self._arm = RebotArm()
        self._arm_group = self._arm.groups["arm"]
        self._n = self._arm_group.num_joints           # 机械臂关节数（通常 6）

        # 夹爪相关字段（在 connect() 后解析具体电机句柄）
        self._has_gripper = self._arm.has_gripper
        self._gripper_group = self._arm.groups.get("gripper")
        self._gripper_motor = None
        gjc = self._gripper_group._jcfgs[0] if self._has_gripper else None
        self._gripper_vlim: float = float(gjc.vlim) if gjc else 3.0
        self._gripper_vendor: str = gjc.vendor if gjc else "damiao"
        self._gripper_name: str = gjc.name if gjc else "gripper"
        # 夹爪 MIT 基准增益（限力夹持时按 ratio 缩放 kp 实现柔顺）
        if self._has_gripper:
            self._gripper_kp0 = self._gripper_group._mit_kp.copy()
            self._gripper_kd0 = self._gripper_group._mit_kd.copy()

        # 夹爪开/合电机位（标定值）：从硬件配置读取 gripper_open / gripper_close，
        # 缺省用 RobStride 标定（闭合 0.0，张开 4.7）。DM 在 rebotarm_dm.yaml 里另配。
        self._gripper_close = 0.0
        self._gripper_open = 4.7
        try:
            hw = yaml.safe_load(_resolve_hw_cfg_path(None).read_text()) or {}
            self._gripper_close = float(hw.get("gripper_close", self._gripper_close))
            self._gripper_open = float(hw.get("gripper_open", self._gripper_open))
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 2. 运动学 / 动力学模型（URDF 路径由硬件配置决定，DM/RS 自动切换）
        # ------------------------------------------------------------------
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)

        self._dyn_model = load_dynamics_model()
        self._dyn_data = self._dyn_model.createData()

        # ------------------------------------------------------------------
        # 3. 末端位姿控制器（机械臂 POS_VEL + 夹爪 MIT，同一控制环按组同步发送）
        # ------------------------------------------------------------------
        self._dt = dt if dt is not None else (1.0 / float(self._arm.rate))
        self._endpos = RebotArmEndPose(
            self._arm, dt=self._dt, arm_control_mode="posvel",
        )
        # 夹爪随末端控制环以 MIT 持续驱动（_loop_cb 每周期 send_mit 到 _gripper_target），
        # 力受限夹持通过缩放夹爪 MIT 的 kp 实现，DM / RS 通用。

        # ------------------------------------------------------------------
        # 4. 状态标志
        # ------------------------------------------------------------------
        self._connected = False
        self._powered = False
        self._teach_on = False
        self._shutting_down = False

        # ------------------------------------------------------------------
        # 5. 示教记录
        # ------------------------------------------------------------------
        self._teach_trajectory: list[np.ndarray] = []
        self._teach_recording = False
        self._teach_sample_counter = 0
        self._teach_sample_every = max(1, int(float(self._arm.rate) / 30))  # 30 Hz 采样
        self._teach_record_dir = Path("records")
        self._teach_record_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 生命周期
    # ==================================================================

    def connect(self) -> None:
        """建立通信（连接总线、注册电机）。

        调用后可读取关节角，但电机尚未使能，控制指令暂时不可用。
        """
        if self._connected:
            return
        self._arm.connect()
        # 电机注册完成后，解析夹爪电机句柄
        if self._has_gripper:
            self._gripper_motor = self._arm._motor_map.get(self._gripper_name)
        self._connected = True

    def disconnect(self) -> None:
        """安全关闭：停运动 → 回零 → 下电 → 关总线。"""
        self._safe_shutdown()

    def _safe_shutdown(self) -> None:
        """统一安全关闭流程：停止当前运动 → 回零位 → 下电 → 断开连接。"""
        if self._shutting_down:
            return
        self._shutting_down = True
        print("\n[reBotArm] 安全关闭中...")

        try:
            self.stop()
        except Exception:
            pass

        if self._powered and not self._teach_on:
            try:
                self._ensure_pose_loop()
                self._endpos.safe_home()
            except KeyboardInterrupt:
                pass
            except Exception:
                pass

        try:
            self.power_off()
        except Exception:
            pass

        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._connected = False
        print("[reBotArm] 安全关闭完成")

    def clear_error(self, retries: int = 3) -> None:
        """清除全部电机（含夹爪）的错误保护状态。

        达妙电机错误码 8~E 需先清除才能使能；RobStride 亦兼容此流程。
        """
        for name, mot in self._arm._motor_map.items():
            for _ in range(retries):
                try:
                    mot.request_feedback()
                except Exception:
                    pass
                for ctrl in self._arm._ctrl_map.values():
                    try:
                        ctrl.poll_feedback_once()
                    except Exception:
                        pass
                st = mot.get_state()
                if st is not None and st.status_code == 0:
                    break
                try:
                    mot.disable()
                except Exception:
                    pass
                time.sleep(0.03)
                try:
                    mot.clear_error()
                except CallError:
                    pass
                time.sleep(0.05)
                if st is not None and st.status_code in (8, 9, 10, 11, 12, 13, 14):
                    print(f"[clear_error] {name}: status_code={st.status_code}，已尝试清除")

    def power_on(self) -> None:
        """上电：清理故障 → 切 POS_VEL 模式 → 使能电机 → 启动控制循环。

        需要先调用 ``connect()``。上电瞬间会将控制目标初始化为当前关节角，
        避免回零猛动。
        """
        if self._powered:
            return
        if not self._connected:
            raise RuntimeError("请先调用 connect()")

        self.clear_error()

        # 机械臂：POS_VEL 模式 + 使能
        self._arm_group.mode_pos_vel()
        self._arm_group.enable()

        # 夹爪：MIT 模式 + 使能（随控制环持续驱动）
        self._enable_gripper()

        # 初始化控制目标为当前位置，避免上电瞬间猛动（臂回零 / 夹爪急合）
        q = self._arm_group.get_positions()
        self._endpos._q_target[:] = q[: self._n]
        if self._has_gripper:
            self._endpos.set_gripper_target(float(self._gripper_group.get_positions()[0]))

        # 启动末端控制循环（臂 POS_VEL + 夹爪 MIT，按组同步发送）
        self._arm.start_control_loop(self._endpos._loop_cb)
        self._endpos._running = True

        self._powered = True

    def power_off(self) -> None:
        """下电：停止控制循环 → 失能电机。

        机械臂下电后处于无力矩状态，须注意重力坠落风险。
        """
        if not self._powered:
            return
        if self._teach_on:
            self._arm.stop_control_loop()
            self._teach_on = False
        self._arm.stop_control_loop()
        self._endpos._running = False
        self._arm.disable_all()   # 失能全部组（含夹爪）
        self._powered = False

    @property
    def is_connected(self) -> bool:
        """是否已建立通信。"""
        return self._connected

    @property
    def is_powered(self) -> bool:
        """电机是否已上电使能。"""
        return self._powered

    def __enter__(self) -> "reBotArm":
        self.connect()
        self.power_on()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is KeyboardInterrupt:
            print("\n[reBotArm] Ctrl+C 检测到，执行安全关闭...")
        self.disconnect()

    # ==================================================================
    # 状态读取
    # ==================================================================

    def get_joint_positions(self) -> np.ndarray:
        """6 个关节的当前角度（弧度），shape=(6,)。"""
        return self._arm_group.get_positions()

    def get_joint_velocities(self) -> np.ndarray:
        """6 个关节的当前角速度（rad/s），shape=(6,)。"""
        return self._arm_group.get_velocities()

    def get_joint_torques(self) -> np.ndarray:
        """6 个关节的当前力矩（N·m），shape=(6,)。"""
        return self._arm.get_state()[2][: self._n]

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取机械臂当前所有关节的状态（位置、速度、力矩）。"""
        pos, vel, torq = self._arm.get_state()
        return pos[: self._n], vel[: self._n], torq[: self._n]

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """末端位姿。

        Returns:
            (position, rpy)
            - position: [x, y, z]  单位 米
            - rpy:      [roll, pitch, yaw]  单位 弧度
        """
        q = self.get_joint_positions()
        q_full = pad_q_for_model(self._model, q, self._n)
        pos, rot, _ = compute_fk(self._model, q_full)
        rpy = pin.rpy.matrixToRpy(rot)
        return pos, rpy

    def get_end_effector_position(self) -> np.ndarray:
        """末端位置 [x, y, z]（米）。"""
        pos, _ = self.get_end_effector_pose()
        return pos

    def get_end_effector_orientation(self) -> np.ndarray:
        """末端姿态 [roll, pitch, yaw]（弧度）。"""
        _, rpy = self.get_end_effector_pose()
        return rpy

    def get_gripper_position(self, tries: int = 6) -> float:
        """夹爪当前位置。可靠读取。"""
        if self._gripper_motor is None:
            return 0.0

        for _ in range(tries):
            try:
                self._gripper_motor.request_feedback()
            except Exception:
                pass
            for ctrl in self._arm._ctrl_map.values():
                try:
                    ctrl.poll_feedback_once()
                except Exception:
                    pass
            try:
                st = self._gripper_motor.get_state()
                if st is not None:
                    return st.pos
            except Exception:
                pass
            time.sleep(0.003)
        return 0.0

    def is_moving(self) -> bool:
        """是否正在执行轨迹运动（move_pose / move_joints）。"""
        return getattr(self._endpos, "_moving", False)

    # ==================================================================
    # 关节角控制
    # ==================================================================

    def move_joints(
        self,
        q_target: np.ndarray,
        duration: float = 2.0,
        wait: bool = True,
    ) -> None:
        """驱动机械臂到目标关节角度。

        Args:
            q_target: 6 个关节目标角度（弧度）。
            duration: 运动时长（秒），默认 2.0 s。
            wait: 为 ``True`` 时阻塞直到到位；
                  为 ``False`` 时立即返回，可通过 ``is_moving`` 轮询。
        """
        q_target = np.asarray(q_target, dtype=np.float64)
        if len(q_target) != self._n:
            raise ValueError(f"q_target 长度应为 {self._n}，实际 {len(q_target)}")

        self._ensure_pose_loop()
        q_start = self.get_joint_positions()
        steps = max(1, int(duration / self._dt))

        # 关节空间线性插值
        traj = [q_start + (i / steps) * (q_target - q_start) for i in range(steps + 1)]
        self._run_traj(traj, duration, wait)

    def servo_joints(self, q_target: np.ndarray) -> None:
        """实时关节伺服：直接将目标写入控制循环，不做轨迹规划。

        适合外部高频流控（轨迹回放、实时遥控等），调用方自行管理时序。

        说明：伺服只更新 6 个机械臂关节目标；夹爪由同一控制环独立地持续
        保持在最近一次 ``move_gripper`` / ``move_gripper_force`` 设定的目标上，
        无需也不应在伺服调用里传夹爪。要动夹爪请单独调 ``move_gripper*``。

        Args:
            q_target: 6 个关节目标角度（弧度）。
        """
        q_target = np.asarray(q_target, dtype=np.float64)
        if len(q_target) != self._n:
            raise ValueError(f"q_target 长度应为 {self._n}，实际 {len(q_target)}")
        self._ensure_pose_loop()
        self._endpos._q_target[:] = q_target

    def servo_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> bool:
        """实时末端伺服：IK 求解后直接写 ``_q_target``，不做轨迹规划。

        适合笛卡尔空间实时流控（视觉跟踪、遥操作等），调用方自行管理时序。

        Args:
            x, y, z: 目标位置（米）。
            roll, pitch, yaw: 目标姿态（弧度）。

        Returns:
            IK 求解成功返回 ``True``，否则 ``False``。
        """
        self._ensure_pose_loop()
        return self._endpos.move_to_ik(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
        )

    # ==================================================================
    # 末端位置控制
    # ==================================================================

    def move_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
        wait: bool = True,
    ) -> bool:
        """驱动末端到目标位姿（带 SE(3) 测地线轨迹规划）。

        Args:
            x, y, z: 目标位置（米）。
            roll, pitch, yaw: 目标姿态（弧度），默认零姿态。
            duration: 运动时长（秒），默认 2.0 s。
            wait: 是否阻塞直到到位。

        Returns:
            规划与执行成功返回 ``True``，否则 ``False``。
        """
        self._ensure_pose_loop()
        ok = self._endpos.move_to_traj(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            duration=duration,
        )
        if not ok:
            return False

        if wait:
            try:
                while self._endpos._moving:
                    time.sleep(self._dt)
            except KeyboardInterrupt:
                self._safe_shutdown()
                raise
        return True

    def move_position(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 2.0,
        wait: bool = True,
    ) -> bool:
        """只控制末端位置，保持当前姿态不变。"""
        _, rpy_cur = self.get_end_effector_pose()
        return self.move_pose(
            x, y, z,
            roll=float(rpy_cur[0]),
            pitch=float(rpy_cur[1]),
            yaw=float(rpy_cur[2]),
            duration=duration,
            wait=wait,
        )

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        duration: float = 2.0,
        wait: bool = True,
    ) -> bool:
        """相对当前末端位置做平移。"""
        pos_cur, rpy_cur = self.get_end_effector_pose()
        return self.move_pose(
            float(pos_cur[0]) + dx,
            float(pos_cur[1]) + dy,
            float(pos_cur[2]) + dz,
            roll=float(rpy_cur[0]),
            pitch=float(rpy_cur[1]),
            yaw=float(rpy_cur[2]),
            duration=duration,
            wait=wait,
        )

    def move_home(self, duration: float = 3.0) -> None:
        """安全回零位（所有关节角度归零）。"""
        self.move_joints(np.zeros(self._n), duration=duration)

    # ==================================================================
    # 夹爪控制（随末端控制环以 MIT 持续驱动）
    # ==================================================================

    def _enable_gripper(self) -> None:
        """夹爪切 MIT 模式并使能（随控制环 send_mit 持续驱动）。"""
        if not self._has_gripper:
            return
        self._gripper_group.mode_mit(
            kp=self._gripper_kp0.copy(),
            kd=self._gripper_kd0.copy(),
        )
        self._gripper_group.enable()

    def move_gripper(self, pos: float, vlim: Optional[float] = None) -> None:
        """设置夹爪目标位置（MIT 持续保持，配置默认刚度）。

        仅更新目标，实际发送由末端控制环每周期完成。``vlim`` 在 MIT
        模式下不适用，保留参数仅为接口兼容，会被忽略。

        Args:
            pos: 目标位置（电机角度，由机械结构决定开合度）。
            vlim: 已忽略（MIT 模式无速度限制参数）。
        """
        if not self._has_gripper:
            return
        # 恢复配置默认刚度（消除上一次限力夹持的软化）
        self._gripper_group._mit_kp[:] = self._gripper_kp0
        self._gripper_group._mit_kd[:] = self._gripper_kd0
        self._endpos.set_gripper_target(float(pos))

    def move_open_gripper(self, vlim: Optional[float] = None) -> None:
        """完全张开夹爪（开合电机位由硬件配置 gripper_open 决定）。"""
        self.move_gripper(self._gripper_open)

    def move_close_gripper(self, vlim: Optional[float] = None) -> None:
        """完全闭合夹爪（开合电机位由硬件配置 gripper_close 决定）。"""
        self.move_gripper(self._gripper_close)

    def move_gripper_force(
        self,
        pos: float,
        ratio: float = 0.3,
        vlim: Optional[float] = None,
    ) -> None:
        """限力夹持（MIT 阻抗控制，DM / RS 通用）。

        通过把夹爪 MIT 的 kp 缩放到基准刚度的 ``ratio`` 倍来软化“虚拟弹簧”，
        夹到物体后以较小的力柔顺夹持，不硬怼、不堵转。控制环持续维持该目标。

        Args:
            pos: 目标位置（电机角度）。
            ratio: 刚度比例，区间 (0, 1]。越小越柔（夹持力越小）。
            vlim: 已忽略（MIT 模式无速度限制参数）。

        说明:
            这是基于 MIT 的“刚度限力”，对所有厂商（含 RobStride）有效。
            它限制的是刚度而非绝对力矩——与达妙固件 FORCE_POS 的“绝对电流上限”
            机制不同；如需绝对限流，请用达妙电机并单独走 FORCE_POS。
        """
        if not self._has_gripper:
            return
        ratio = float(np.clip(ratio, 1e-3, 1.0))
        self._gripper_group._mit_kp[:] = self._gripper_kp0 * ratio
        self._gripper_group._mit_kd[:] = self._gripper_kd0
        self._endpos.set_gripper_target(float(pos))

    # ==================================================================
    # 示教模式（重力补偿，可手动拖动）
    # ==================================================================

    def teach_mode(self, enable: bool = True, kd: float = 1.0) -> Optional[Path]:
        """进入 / 退出示教模式（重力补偿，可手动拖动）。

        进入示教时自动开始记录关节角轨迹，退出时自动停止并保存为 .npy。

        Args:
            enable: ``True`` 进入示教，``False`` 退出并锁定当前位置。
            kd: MIT 阻尼系数，帮助松手后更快停住。

        Returns:
            退出示教时返回保存的 .npy 文件路径，进入示教时返回 ``None``。
        """
        if enable:
            if self._teach_on:
                return None
            # 停掉末端控制循环，切 MIT 浮停
            self._arm.stop_control_loop()
            self._endpos._running = False
            self.clear_error()
            self._arm_group.enable()
            self._powered = True
            self._teach_kd = float(kd)
            ok = self._arm_group.mode_mit(
                kp=np.zeros(self._n),
                kd=np.full(self._n, float(kd)),
            )
            if not ok:
                print("[teach_mode] 部分关节未能切到 MIT 模式")
            self.start_teach_recording()
            self._arm.start_control_loop(self._teach_cb, rate=float(self._arm.rate))
            self._teach_on = True
            return None
        else:
            if not self._teach_on:
                return None
            self._arm.stop_control_loop()
            self._teach_on = False
            saved_path = self.stop_teach_recording()
            q = self._arm_group.get_positions()
            self._arm_group.mode_pos_vel()
            self._endpos._q_target[:] = q[: self._n]
            self._arm_group.send_pos_vel(np.asarray(q, dtype=np.float64))
            # 恢复末端控制循环
            self._arm.start_control_loop(self._endpos._loop_cb)
            self._endpos._running = True
            self._powered = True
            return saved_path

    def _teach_cb(self, _: RebotArm, dt: float) -> None:
        """示教回调：纯重力补偿浮停 + 30Hz 关节角采样。"""
        q = self._arm_group.get_positions(request_feedback=False)
        q_full = pad_q_for_model(self._model, q, self._n)
        tau_g = compute_generalized_gravity(self._dyn_model, q_full, self._dyn_data)
        tau_g = tau_g[: self._n]
        self._arm_group.send_mit(
            pos=q,
            vel=np.zeros(self._n),
            kp=np.zeros(self._n),
            kd=np.full(self._n, getattr(self, "_teach_kd", 1.0)),
            tau=tau_g,
        )

        # 示教期间夹爪保持在最近目标（否则停了末端环夹爪会失力松开）
        if self._has_gripper:
            self._gripper_group.send_mit(
                np.array([self._endpos._gripper_target], dtype=np.float64),
                kp=self._gripper_group._mit_kp,
                kd=self._gripper_group._mit_kd,
            )

        # 30 Hz 采样记录
        if self._teach_recording:
            self._teach_sample_counter += 1
            if self._teach_sample_counter >= self._teach_sample_every:
                self._teach_trajectory.append(q.copy())
                self._teach_sample_counter = 0

    # ==================================================================
    # 示教记录与回放
    # ==================================================================

    def start_teach_recording(self) -> None:
        """开始记录关节角轨迹（清空旧数据）。"""
        self._teach_trajectory.clear()
        self._teach_sample_counter = 0
        self._teach_recording = True
        print("[teach_record] 开始记录，采样率 30 Hz")

    def stop_teach_recording(self, auto_save: bool = True) -> Optional[Path]:
        """停止记录关节角轨迹。

        Args:
            auto_save: 是否自动保存为 .npy 文件。

        Returns:
            保存的文件路径，如果未保存则返回 ``None``。
        """
        self._teach_recording = False
        count = len(self._teach_trajectory)
        print(f"[teach_record] 停止记录，共 {count} 个点")
        if auto_save and count > 0:
            return self.save_teach_trajectory()
        return None

    def get_teach_trajectory(self) -> np.ndarray:
        """获取当前记录的关节角轨迹，shape=(N, 6)，单位弧度。"""
        if not self._teach_trajectory:
            return np.empty((0, self._n))
        return np.stack(self._teach_trajectory)

    def clear_teach_trajectory(self) -> None:
        """清空已记录的轨迹。"""
        self._teach_trajectory.clear()

    def optimize_trajectory(
        self,
        traj: np.ndarray,
        rate: float = 30.0,
    ) -> np.ndarray:
        """离线优化轨迹：贪心覆盖法重参数化。

        沿折线连续行走，每当任意关节累积位移达到 ``vlim / rate`` 时
        放置一个路点，保证相邻点位移不超过电机可达范围。

        Args:
            traj: 原始轨迹，shape=(N, 6)，单位弧度。
            rate: 回放频率（Hz），默认 30 Hz。

        Returns:
            优化后轨迹，shape=(M, 6)，相邻帧位移 ≤ vlim / rate。
        """
        traj = np.asarray(traj, dtype=np.float64)
        if len(traj) < 2:
            return traj

        interval = 1.0 / rate
        max_step = np.array([j.vlim for j in self._arm_group._jcfgs]) * interval
        result = [traj[0].copy()]

        accum = np.zeros(self._n)
        seg_idx = 0
        t = 0.0

        while seg_idx < len(traj) - 1:
            seg_vec = traj[seg_idx + 1] - traj[seg_idx]
            abs_seg = np.abs(seg_vec)
            remaining = 1.0 - t

            done_seg = t * abs_seg
            remain_seg = abs_seg - done_seg

            t_limits = []
            for j in range(self._n):
                if remain_seg[j] > 1e-12:
                    tl = (max_step[j] - accum[j]) / remain_seg[j]
                    t_limits.append(tl)
            if not t_limits:
                break
            t_limit = min(t_limits)

            if t_limit < remaining:
                t += t_limit
                pt = traj[seg_idx] + seg_vec * t
                result.append(pt.copy())
                accum = np.zeros(self._n)
            else:
                accum += remain_seg
                seg_idx += 1
                t = 0.0

        result.append(traj[-1].copy())
        return np.stack(result)

    def save_teach_trajectory(
        self, path: Optional[str] = None, format: str = "npy"
    ) -> Path:
        """保存当前轨迹到文件。

        Args:
            path: 保存路径，``None`` 时自动生成带时间戳的文件名。
            format: 文件格式，目前仅支持 ``"npy"``。

        Returns:
            保存的文件路径。
        """
        traj = self.get_teach_trajectory()
        if traj.size == 0:
            raise RuntimeError("没有可保存的轨迹数据")

        traj = self.optimize_trajectory(traj)
        print(f"[teach_record] 优化后轨迹点数: {len(traj)}")

        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._teach_record_dir / f"teach_trajectory_{ts}.npy"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, traj)
        print(f"[teach_record] 轨迹已保存: {path}  shape={traj.shape}")
        return path

    def replay_trajectory(
        self,
        traj: Optional[np.ndarray] = None,
        path: Optional[str] = None,
        rate: float = 30.0,
        wait: bool = True,
        record: bool = False,
    ) -> Optional[np.ndarray]:
        """回放示教轨迹，逐点驱动机械臂。

        支持直接传入轨迹数组或从 .npy 文件加载。
        开启 ``record`` 时，每帧先记录当前关节角再回放，返回记录的轨迹。

        Args:
            traj: 轨迹数组 shape=(N, 6)，``None`` 时使用最近一次记录的轨迹。
            path: 从 .npy 文件加载轨迹，优先级低于 ``traj``。
            rate: 回放频率（Hz），默认 30 Hz。
            wait: 是否阻塞直到回放完成。
            record: 是否在回放过程中记录关节角。

        Returns:
            ``record=True`` 时返回记录的轨迹 shape=(M, 6)，否则返回 ``None``。
        """
        if traj is not None:
            traj = np.asarray(traj, dtype=np.float64)
        elif path is not None:
            traj = np.load(path)
        else:
            traj = self.get_teach_trajectory()

        if traj.size == 0:
            raise RuntimeError("没有可回放的轨迹数据")

        if traj.ndim != 2 or traj.shape[1] != self._n:
            raise ValueError(f"轨迹 shape 应为 (N, {self._n})，实际 {traj.shape}")

        interval = 1.0 / rate
        duration = len(traj) * interval
        print(f"[replay] 回放轨迹: {len(traj)} 个点, {rate} Hz, 预计时长 {duration:.2f}s")

        recorded: list[np.ndarray] = []

        def _replay():
            for pt in traj:
                if self._endpos._stop_send.is_set():
                    break
                if record:
                    recorded.append(self.get_joint_positions())
                self.move_joints(pt, duration=interval, wait=True)
            self._endpos._moving = False

        if wait:
            _replay()
        else:
            self._endpos._moving = True
            self._endpos._stop_send.clear()
            threading.Thread(target=_replay, daemon=True).start()

        if record:
            return np.stack(recorded) if recorded else np.empty((0, self._n))
        return None

    # ==================================================================
    # 停止 / 急停
    # ==================================================================

    def stop(self) -> None:
        """立即停止当前运动（保持当前位置）。"""
        if self._teach_on:
            self.teach_mode(False)
        else:
            self._endpos._stop_send.set()
            if self._endpos._send_thread is not None:
                self._endpos._send_thread.join(timeout=1.0)
            q = self.get_joint_positions()
            self._endpos._q_target[:] = q[: self._n]

    def estop(self) -> None:
        """紧急停止：失能所有电机。"""
        self._arm.estop()

    # ==================================================================
    # 内部辅助
    # ==================================================================

    def _ensure_pose_loop(self) -> None:
        """确保末端控制循环在运行（POS_VEL 模式），如未上电则自动上电。"""
        if not self._powered:
            self.power_on()
            return
        if not self._arm.control_loop_active:
            q = self.get_joint_positions()
            self._endpos._q_target[:] = q[: self._n]
            if self._has_gripper:
                self._endpos.set_gripper_target(
                    float(self._gripper_group.get_positions()[0])
                )
            self._arm.start_control_loop(self._endpos._loop_cb)
            self._endpos._running = True

    def _run_traj(
        self,
        traj: list[np.ndarray],
        duration: float,
        wait: bool,
    ) -> None:
        """复用 RebotArmEndPose 的轨迹发送线程执行关节空间插值。"""
        # 停掉可能正在运行的旧轨迹
        self._endpos._stop_send.set()
        if self._endpos._send_thread is not None:
            self._endpos._send_thread.join(timeout=1.0)

        self._endpos._traj = traj
        self._endpos._moving = True
        self._endpos._stop_send.clear()
        self._endpos._send_thread = threading.Thread(
            target=self._endpos._send_loop, args=(duration,), daemon=True
        )
        self._endpos._send_thread.start()

        if wait:
            try:
                while self._endpos._moving:
                    time.sleep(self._dt)
            except KeyboardInterrupt:
                self._safe_shutdown()
                raise

    def __repr__(self) -> str:
        return (
            f"reBotArm(joints={self._n}, "
            f"connected={'on' if self._connected else 'off'}, "
            f"powered={'on' if self._powered else 'off'}, "
            f"teach={'on' if self._teach_on else 'off'})"
        )
