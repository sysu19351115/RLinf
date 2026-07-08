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
"""Low-level SO101 / Feetech STS3215 motor initialization helpers.

The LeRobot version pinned in RLinf (commit 0cf8648) does not configure several
STS3215 registers that are required for reliable synchronous reads on the SO101:

* ``Response_Status_Level`` must be 0, otherwise every write command leaves a
  status packet in the RX buffer and the next ``GroupSyncRead`` fails with
  ``Incorrect status packet``.
* ``Return_Delay_Time`` should be 0 for the tight group-read timing used by
  ``GroupSyncRead``.
* ``Phase`` bit 4 must be cleared so position feedback is in
  ``[0, resolution-1]``.

This module performs a one-time low-level setup with ``scservo_sdk`` before the
LeRobot ``ManipulatorRobot`` opens the ports, patches the serial timeout to
match the value used by ``lerobot_zhiyu``, and provides a safer replacement for
``ManipulatorRobot.set_so100_robot_preset`` that skips the problematic 2-byte
``Maximum_Acceleration`` write.
"""

import time
import types
from pathlib import Path
from typing import Any

# STS3215 control-table addresses used here.  These match the official SCS/STS
# memory map and the tables in lerobot.common.robot_devices.motors.feetech.
_RESPONSE_STATUS_LEVEL_ADDR = 8
_RETURN_DELAY_TIME_ADDR = 7
_PHASE_ADDR = 18
_TORQUE_ENABLE_ADDR = 40
_MODE_ADDR = 33
_P_COEFFICIENT_ADDR = 21
_I_COEFFICIENT_ADDR = 23
_D_COEFFICIENT_ADDR = 22
_ACCELERATION_ADDR = 41
_MAXIMUM_ACCELERATION_ADDR = 85

_PROTOCOL_VERSION = 0
_BAUDRATE = 1_000_000
_TIMEOUT_MS = 1000


def _patch_scservo_timeout() -> None:
    """Patch scservo_sdk PortHandler timeout to the lerobot_zhiyu formula.

    The stock timeout is too short for 6 STS3215 motors on a single sync-read
    transaction and produces intermittent ``COMM_RX_TIMEOUT`` / ``COMM_RX_CORRUPT``
    on the later motors in the chain.
    """
    import scservo_sdk as scs

    if getattr(scs.PortHandler.setPacketTimeout, "_rlinf_patched", False):
        return

    def _set_packet_timeout(self, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (
            self.tx_time_per_byte * packet_length
            + self.tx_time_per_byte * 3.0
            + 50.0
        )

    _set_packet_timeout._rlinf_patched = True  # type: ignore[attr-defined]
    scs.PortHandler.setPacketTimeout = _set_packet_timeout


def _open_raw_port(port: str):
    """Open a Feetech serial port at 1 Mbps and return (port_handler, packet_handler)."""
    import scservo_sdk as scs

    _patch_scservo_timeout()

    port_handler = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(_PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise OSError(f"Failed to open SO101 motor port '{port}'.")
    if not port_handler.setBaudRate(_BAUDRATE):
        raise OSError(f"Failed to set baudrate {_BAUDRATE} on '{port}'.")
    port_handler.setPacketTimeoutMillis(_TIMEOUT_MS)

    # Make sure no stale bytes are waiting from a previous session.
    port_handler.ser.reset_input_buffer()
    port_handler.ser.reset_output_buffer()
    return port_handler, packet_handler


def configure_feetech_port(port: str, motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> None:
    """Configure a single SO101 arm port so that LeRobot can sync-read reliably.

    This must be called *before* LeRobot opens the same port.  It writes:

    * ``Response_Status_Level = 0``  (no status replies to writes)
    * ``Return_Delay_Time = 0``      (fastest reply timing)
    * ``Phase &= ~0x10``             (force unsigned position feedback)
    * ``Maximum_Acceleration = 254`` (1 byte, as used by lerobot_zhiyu)
    * ``Acceleration = 254``

    Each register is read back at the end; if any motor did not accept the
    configuration, a ``ConnectionError`` is raised with the offending motor IDs.
    """
    import scservo_sdk as scs

    port_handler, packet_handler = _open_raw_port(port)
    try:
        # 1. Disable status replies from write commands.  This is the main cause of
        # "Incorrect status packet" during the first GroupSyncRead.  Use TxRx here
        # because the motor still replies; the write result is checked.
        for mid in motor_ids:
            comm, _ = packet_handler.write1ByteTxRx(
                port_handler, mid, _RESPONSE_STATUS_LEVEL_ADDR, 0
            )
            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Failed to set Response_Status_Level on {port} ID {mid}: "
                    f"{packet_handler.getTxRxResult(comm)}"
                )
        time.sleep(0.02)
        port_handler.ser.reset_input_buffer()

        # 2. Minimise the per-motor reply delay.  From now on the motor no longer
        # sends status packets, so use TxOnly.
        for mid in motor_ids:
            packet_handler.write1ByteTxOnly(
                port_handler, mid, _RETURN_DELAY_TIME_ADDR, 0
            )
        time.sleep(0.02)
        port_handler.ser.reset_input_buffer()

        # 3. Force position feedback to [0, resolution-1].
        for mid in motor_ids:
            phase, comm, _ = packet_handler.read1ByteTxRx(
                port_handler, mid, _PHASE_ADDR
            )
            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Failed to read Phase on {port} ID {mid}: "
                    f"{packet_handler.getTxRxResult(comm)}"
                )
            if phase & 0x10:
                packet_handler.write1ByteTxOnly(
                    port_handler, mid, _PHASE_ADDR, phase & ~0x10
                )
        time.sleep(0.02)
        port_handler.ser.reset_input_buffer()

        # 4. Use the 1-byte Maximum_Acceleration register that matches the actual
        # STS3215 memory layout (old LeRobot declares it as 2 bytes, which can
        # corrupt adjacent registers).
        for mid in motor_ids:
            packet_handler.write1ByteTxOnly(
                port_handler, mid, _MAXIMUM_ACCELERATION_ADDR, 254
            )
            packet_handler.write1ByteTxOnly(
                port_handler, mid, _ACCELERATION_ADDR, 254
            )
        time.sleep(0.02)
        port_handler.ser.reset_input_buffer()

        # 5. Verify the persistent registers took effect.
        errors = []
        for mid in motor_ids:
            rsl, comm_rsl, _ = packet_handler.read1ByteTxRx(
                port_handler, mid, _RESPONSE_STATUS_LEVEL_ADDR
            )
            phase, comm_phase, _ = packet_handler.read1ByteTxRx(
                port_handler, mid, _PHASE_ADDR
            )
            if comm_rsl != scs.COMM_SUCCESS or rsl != 0:
                errors.append(f"ID{mid} Response_Status_Level={rsl} (comm={comm_rsl})")
            if comm_phase != scs.COMM_SUCCESS or (phase & 0x10):
                errors.append(f"ID{mid} Phase={phase} bit4 set (comm={comm_phase})")
        if errors:
            raise ConnectionError(
                f"SO101 motor init verification failed on {port}: {errors}"
            )
    finally:
        port_handler.closePort()


def _safe_so100_robot_preset(self) -> None:
    """Replacement for ``ManipulatorRobot.set_so100_robot_preset``.

    The upstream implementation writes ``Maximum_Acceleration`` using the
    2-byte entry from its control table, which does not match the STS3215
    layout and can leave the bus in a state where ``GroupSyncRead`` fails.
    This version only writes the registers we have confirmed to be safe.
    """
    for name in self.follower_arms:
        arm = self.follower_arms[name]
        arm.write("Mode", 0)
        arm.write("P_Coefficient", 16)
        arm.write("I_Coefficient", 0)
        arm.write("D_Coefficient", 32)
        arm.write("Acceleration", 254)

    # NOTE: We intentionally do NOT write "Maximum_Acceleration" or "Lock"
    # here; they are configured once by configure_feetech_port() using the
    # correct 1-byte register width.


def patch_robot_preset(robot: Any) -> None:
    """Patch a ManipulatorRobot instance to use the safe SO101 preset."""
    robot.set_so100_robot_preset = types.MethodType(_safe_so100_robot_preset, robot)


def build_so101_manipulator(
    left_follower_port: str,
    right_follower_port: str,
    calibration_dir: str | Path,
    cameras: dict[str, Any] | None = None,
    max_relative_target: float | None = 5.0,
) -> Any:
    """Create a LeRobot ManipulatorRobot with SO101-safe low-level init.

    Returns the robot instance; the caller is responsible for ``connect()`` and
    ``disconnect()``.
    """
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.robots.configs import So101RobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

    # Make sure all PortHandler instances use a timeout that can accommodate 6
    # motors on one sync-read transaction.
    _patch_scservo_timeout()

    # Low-level setup must happen before LeRobot opens the ports.
    configure_feetech_port(left_follower_port)
    configure_feetech_port(right_follower_port)

    arm_motor_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    def _arm_config(port: str) -> FeetechMotorsBusConfig:
        return FeetechMotorsBusConfig(
            port=port,
            motors={
                name: (idx, "sts3215")
                for idx, name in enumerate(arm_motor_names, start=1)
            },
        )

    config = So101RobotConfig(
        calibration_dir=str(calibration_dir),
        leader_arms={},
        follower_arms={
            "left": _arm_config(left_follower_port),
            "right": _arm_config(right_follower_port),
        },
        cameras=cameras or {},
        max_relative_target=max_relative_target,
    )
    robot = ManipulatorRobot(config)
    patch_robot_preset(robot)
    return robot
