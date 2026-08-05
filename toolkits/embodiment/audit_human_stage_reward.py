#!/usr/bin/env python3
"""Audit human stage-label rewards from a tee\'d HIL run log.

Parses the structured lines emitted by ``DobotKeyboardIntervention``:

    [HumanStage] key=1 stage->1 event=0.0200
    [HumanStage] terminal=success reward=1.0000
    [HumanStage] terminal=failure reward=-1.0000

Episode boundaries are the "Recording begins." start-gate lines.  The script
reports per-episode stage/event/terminal sums, a hold-score estimate when log
timestamps are available (--fps), aggregate success-vs-failure separation, and
the "pressed 3 then Backspace" rate.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from dataclasses import dataclass, field

STAGE_EVENT_RE = re.compile(
    r"\[HumanStage\] key=(?P<key>\S+) stage->(?P<stage>\d) "
    r"event=(?P<event>[0-9.eE+-]+)"
)
TERMINAL_RE = re.compile(
    r"\[HumanStage\] terminal=(?P<result>success|failure) "
    r"reward=(?P<reward>[0-9.eE+-]+)"
)
START_RE = re.compile(r"Recording begins\.")
TS_PREFIX_RE = re.compile(r"^\s*(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[.,]\d+)?)")
HOLD_DEFAULT = (0.0, 0.001, 0.002, 0.0)


def _parse_ts(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(
            raw.strip().replace(",", ".")
        ).timestamp()
    except ValueError:
        return None


@dataclass
class Episode:
    stages: list[int] = field(default_factory=list)
    events: list[float] = field(default_factory=list)
    terminal: str | None = None
    terminal_reward: float = 0.0
    terminal_ts: float | None = None
    start_ts: float | None = None
    stage_ts: list[float] = field(default_factory=list)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", help="log file (default: stdin)")
    ap.add_argument(
        "--fps", type=float, default=30.0, help="step rate for hold estimate"
    )
    ap.add_argument(
        "--hold", nargs=4, type=float, default=HOLD_DEFAULT, help="hold rewards 0..3"
    )
    args = ap.parse_args()

    episodes: list[Episode] = []
    cur: Episode | None = None
    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    for line in text.splitlines():
        if START_RE.search(line):
            if cur is not None and (cur.stages or cur.terminal):
                episodes.append(cur)
            cur = Episode()
            m = TS_PREFIX_RE.search(line)
            cur.start_ts = _parse_ts(m.group("ts")) if m else None
            continue
        if cur is None:
            continue
        m = STAGE_EVENT_RE.search(line)
        if m:
            cur.stages.append(int(m.group("stage")))
            cur.events.append(float(m.group("event")))
            tm = TS_PREFIX_RE.search(line)
            cur.stage_ts.append(_parse_ts(tm.group("ts")) if tm else 0.0)
            continue
        m = TERMINAL_RE.search(line)
        if m:
            cur.terminal = m.group("result")
            cur.terminal_reward = float(m.group("reward"))
            tm = TS_PREFIX_RE.search(line)
            cur.terminal_ts = _parse_ts(tm.group("ts")) if tm else None
    if cur is not None and (cur.stages or cur.terminal):
        episodes.append(cur)

    if not episodes:
        print("No HumanStage events found in log.")
        return 1

    def hold_est(ep: Episode) -> float | None:
        """Integrate hold rewards over each interval using the stage that is
        active at the interval START (stage 0 before the first key)."""
        if not ep.stage_ts or ep.start_ts is None:
            return None
        starts = [ep.start_ts, *ep.stage_ts]
        if ep.terminal_ts is not None:
            starts.append(ep.terminal_ts)
        total = 0.0
        for j in range(len(starts) - 1):
            dt = starts[j + 1] - starts[j]
            if dt <= 0:
                return None
            active_stage = ep.stages[j - 1] if j > 0 else 0
            total += args.hold[active_stage] * dt * args.fps
        return total

    success_total: list[float] = []
    failure_total: list[float] = []
    p3_abort = 0
    p3 = 0
    print(
        f"{'ep':>3} {'stages':<14} {'events':>7} {'hold~':>8} "
        f"{'terminal':>9} {'total~':>9}"
    )
    for idx, ep in enumerate(episodes, 1):
        h = hold_est(ep)
        total = sum(ep.events) + ep.terminal_reward + (h or 0.0)
        label = ep.terminal or "open"
        if 3 in ep.stages:
            p3 += 1
            if ep.terminal == "failure":
                p3_abort += 1
        print(
            f"{idx:>3} {str(ep.stages):<14} {sum(ep.events):>7.3f} "
            f"{'-' if h is None else f'{h:.3f}':>8} {label:>9} {total:>9.3f}"
        )
        if ep.terminal == "success":
            success_total.append(total)
        elif ep.terminal == "failure":
            failure_total.append(total)
    print()
    open_count = len(episodes) - len(success_total) - len(failure_total)
    print(
        f"episodes={len(episodes)} success={len(success_total)} "
        f"failure={len(failure_total)} open={open_count}"
    )
    if success_total and failure_total:
        avg_s = sum(success_total) / len(success_total)
        avg_f = sum(failure_total) / len(failure_total)
        print(
            f"avg total success={avg_s:.3f}  failure={avg_f:.3f}  gap={avg_s - avg_f:.3f}"
        )
    if p3:
        print(f"pressed-3-then-Backspace: {p3_abort}/{p3} ({p3_abort / p3:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
