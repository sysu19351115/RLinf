# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent last-line safety barrier for real-robot actions (P2-5).

The policy limits are a *design* constraint of the residual actor; this module
is the independent check that runs on every composed chunk before commands can
reach the servo loop.  Any violation fails closed: the residual is dropped
(nominal kept), the transition is marked invalid, and a configurable hold is
entered so the run cannot immediately re-offend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

VIOLATION_NON_FINITE = "non_finite"
VIOLATION_BAD_QUATERNION = "bad_quaternion"
VIOLATION_WORKSPACE = "workspace_out_of_bounds"
VIOLATION_TRANSLATION_DELTA = "translation_delta_exceeded"
VIOLATION_ROTATION_DELTA = "rotation_delta_exceeded"
VIOLATION_QUATERNION_FLIP = "quaternion_hemisphere_flip"


@dataclass(frozen=True)
class SafetyLimits:
    """Physical limits independent of the policy design limits."""

    workspace_min_m: np.ndarray  # [3]
    workspace_max_m: np.ndarray  # [3]
    max_translation_delta_m: float
    max_rotation_delta_deg: float

    @classmethod
    def from_config(cls, rlpd_cfg: Any) -> "SafetyLimits":
        return cls(
            workspace_min_m=np.asarray(
                list(rlpd_cfg.safety_workspace_min_m), dtype=np.float64
            ),
            workspace_max_m=np.asarray(
                list(rlpd_cfg.safety_workspace_max_m), dtype=np.float64
            ),
            max_translation_delta_m=float(
                rlpd_cfg.safety_max_translation_delta_m
            ),
            max_rotation_delta_deg=float(
                rlpd_cfg.safety_max_rotation_delta_deg
            ),
        )


@dataclass(frozen=True)
class SafetyViolation:
    step: int
    code: str
    detail: str


def _check_quaternion(q: np.ndarray) -> list[str]:
    """Quaternion must be finite and near unit norm (wxyz convention)."""
    codes: list[str] = []
    if not np.isfinite(q).all():
        codes.append(VIOLATION_NON_FINITE)
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 1e-2:
        codes.append(VIOLATION_BAD_QUATERNION)
    return codes


class ResidualSafetyBarrier:
    """Stateful per-chunk safety checks with hemisphere continuity tracking."""

    def __init__(
        self,
        limits: SafetyLimits,
        *,
        quat_eps: float = 1e-2,
        hemisphere_eps: float = 0.0,
    ):
        self.limits = limits
        self.quat_eps = float(quat_eps)
        self.hemisphere_eps = float(hemisphere_eps)
        self._prev_quat: np.ndarray | None = None
        self.violation_count = 0
        self.last_violations: list[SafetyViolation] = []

    def check_chunk(
        self,
        nominal: np.ndarray,
        commanded: np.ndarray,
    ) -> list[SafetyViolation]:
        """Check one ``[H, 8]`` commanded chunk against nominal + limits.

        ``commanded`` and ``nominal`` use the wxyz pose convention with the
        gripper in the last column.
        """
        violations: list[SafetyViolation] = []
        nominal = np.asarray(nominal, dtype=np.float64)
        commanded = np.asarray(commanded, dtype=np.float64)
        h = nominal.shape[0]
        rot_limit_rad = math.radians(self.limits.max_rotation_delta_deg)
        for i in range(h):
            step = int(i)
            if not np.isfinite(commanded[i]).all():
                violations.append(
                    SafetyViolation(step, VIOLATION_NON_FINITE, "commanded non-finite")
                )
                continue
            q = commanded[i, 3:7]
            for code in _check_quaternion(q):
                violations.append(
                    SafetyViolation(
                        step, code, f"quaternion={q.tolist()}, norm={np.linalg.norm(q):.4f}"
                    )
                )
            pos = commanded[i, :3]
            if np.any(pos < self.limits.workspace_min_m - 1e-6) or np.any(
                pos > self.limits.workspace_max_m + 1e-6
            ):
                violations.append(
                    SafetyViolation(
                        step,
                        VIOLATION_WORKSPACE,
                        f"pos={pos.tolist()}, "
                        f"min={self.limits.workspace_min_m.tolist()}, "
                        f"max={self.limits.workspace_max_m.tolist()}",
                    )
                )
            trans_delta = float(np.linalg.norm(commanded[i, :3] - nominal[i, :3]))
            if trans_delta > self.limits.max_translation_delta_m + 1e-6:
                violations.append(
                    SafetyViolation(
                        step,
                        VIOLATION_TRANSLATION_DELTA,
                        f"delta={trans_delta:.4f}m > "
                        f"{self.limits.max_translation_delta_m:.4f}m",
                    )
                )
            q_n = nominal[i, 3:7]
            q_c = commanded[i, 3:7]
            if float(np.linalg.norm(q_n)) > 1e-6 and float(np.linalg.norm(q_c)) > 1e-6:
                rel = np.abs(float(np.dot(q_n / np.linalg.norm(q_n), q_c / np.linalg.norm(q_c))))
                rot_delta = 2.0 * math.acos(min(1.0, max(-1.0, rel)))
                if rot_delta > rot_limit_rad + 1e-6:
                    violations.append(
                        SafetyViolation(
                            step,
                            VIOLATION_ROTATION_DELTA,
                            f"delta={math.degrees(rot_delta):.2f}deg > "
                            f"{self.limits.max_rotation_delta_deg:.2f}deg",
                        )
                    )
            if self._prev_quat is not None and float(np.linalg.norm(q)) > 1e-6:
                dot = float(
                    np.dot(q / np.linalg.norm(q), self._prev_quat / np.linalg.norm(self._prev_quat))
                )
                if dot < self.hemisphere_eps:
                    violations.append(
                        SafetyViolation(
                            step,
                            VIOLATION_QUATERNION_FLIP,
                            f"dot={dot:.4f} (hemisphere discontinuity)",
                        )
                    )
            self._prev_quat = q.copy()
        if violations:
            self.violation_count += 1
            self.last_violations = violations
        else:
            self.last_violations = []
        return violations

    def reset_tracking(self) -> None:
        self._prev_quat = None


def build_safety_barrier_from_config(cfg: Any) -> ResidualSafetyBarrier:
    from rlinf.algorithms.residual_hil_rlpd.config import get_residual_rlpd_cfg

    return ResidualSafetyBarrier(
        SafetyLimits.from_config(get_residual_rlpd_cfg(cfg))
    )
