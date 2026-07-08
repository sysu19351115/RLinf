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
"""Diagnose SO101 / Feetech STS3215 serial ports.

This is a standalone helper to figure out which motor IDs are reachable on each
port and at which baudrate. It does not modify any motor registers.

Usage::

    python rlinf/envs/realworld/so101/diagnose_ports.py
    python rlinf/envs/realworld/so101/diagnose_ports.py --port /dev/ttyACM0
    python rlinf/envs/realworld/so101/diagnose_ports.py --ids 1 2 3 4 5 6 7 8 9 10
"""

import argparse
import sys
import time

_BAUDRATES = (1_000_000, 500_000, 115_200, 57_600)
_DEFAULT_IDS = tuple(range(1, 11))
_PROTOCOL_VERSION = 0
_PRESENT_POSITION_ADDR = 56
_MODEL_NUMBER_ADDR = 0
_TIMEOUT_MS = 1000


def scan_port(port: str, ids: tuple[int, ...]) -> dict:
    import scservo_sdk as scs

    results = {
        "port": port,
        "open": False,
        "baudrate_ok": None,
        "motors": {},
        "errors": [],
    }

    for baud in _BAUDRATES:
        ph = scs.PortHandler(port)
        pch = scs.PacketHandler(_PROTOCOL_VERSION)
        try:
            if not ph.openPort():
                results["errors"].append(f"Cannot open {port}")
                return results
            results["open"] = True
            if not ph.setBaudRate(baud):
                results["errors"].append(f"Cannot set baudrate {baud} on {port}")
                continue
            ph.setPacketTimeoutMillis(_TIMEOUT_MS)
            ph.ser.reset_input_buffer()
            ph.ser.reset_output_buffer()

            # Ping each ID first; if it answers, read model and position.
            found = {}
            for mid in ids:
                model, comm, err = pch.read2ByteTxRx(ph, mid, _MODEL_NUMBER_ADDR)
                if comm == scs.COMM_SUCCESS:
                    pos, pos_comm, _ = pch.read2ByteTxRx(ph, mid, _PRESENT_POSITION_ADDR)
                    found[mid] = {
                        "model_number": model,
                        "position": pos if pos_comm == scs.COMM_SUCCESS else None,
                    }
            if found:
                results["baudrate_ok"] = baud
                results["motors"] = found
                return results
        finally:
            ph.closePort()
        time.sleep(0.05)

    results["errors"].append(
        f"No motors answered at any baudrate ({_BAUDRATES}) for IDs {ids}"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SO101 serial ports.")
    parser.add_argument(
        "--port",
        action="append",
        default=[],
        help="Serial port to scan (can be given multiple times).",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=list(_DEFAULT_IDS),
        help=f"Motor IDs to probe (default: {list(_DEFAULT_IDS)}).",
    )
    args = parser.parse_args()

    ports = args.port or [f"/dev/ttyACM{i}" for i in range(4)]
    ids = tuple(args.ids)

    print("SO101 port diagnosis")
    print("=" * 50)
    any_found = False
    for port in ports:
        print(f"\nPort: {port}")
        res = scan_port(port, ids)
        if not res["open"]:
            print(f"  [FAIL] Could not open port")
            for e in res["errors"]:
                print(f"         {e}")
            continue
        if res["baudrate_ok"] is None:
            print(f"  [FAIL] No motors responded")
            for e in res["errors"]:
                print(f"         {e}")
            continue
        print(f"  [ OK ] Open; baudrate {res['baudrate_ok']}")
        print(f"  Found {len(res['motors'])} motor(s):")
        for mid, data in sorted(res["motors"].items()):
            pos = data["position"]
            pos_str = f"pos={pos}" if pos is not None else "pos=N/A"
            print(f"    ID {mid:2d}: model={data['model_number']:5d}, {pos_str}")
        any_found = True

    print("\n" + "=" * 50)
    if any_found:
        print("If IDs are not 1-6 on each port, check wiring or update motor IDs.")
        print("Expected: left arm on one port with IDs 1-6, right arm on the other port with IDs 1-6.")
    else:
        print("No motors found. Check power, USB cables, and serial port assignments.")
    return 0 if any_found else 1


if __name__ == "__main__":
    sys.exit(main())
