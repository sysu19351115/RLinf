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

"""Safety barrier (P2-5) tests."""

from __future__ import annotations

import numpy as np

from rlinf.algorithms.residual_hil_rlpd.safety import (
    VIOLATION_BAD_QUATERNION,
    VIOLATION_NON_FINITE,
    VIOLATION_QUATERNION_FLIP,
    VIOLATION_ROTATION_DELTA,
    VIOLATION_TRANSLATION_DELTA,
    VIOLATION_WORKSPACE,
    ResidualSafetyBarrier,
    SafetyLimits,
)


def _limits():
    return SafetyLimits(
        workspace_min_m=np.asarray([-0.5, -0.5, 0.0]),
        workspace_max_m=np.asarray([0.5, 0.5, 0.5]),
        max_translation_delta_m=0.02,
        max_rotation_delta_deg=10.0,
    )


def _pose(position, quat_wxyz, gripper=0.5):
    return np.concatenate([position, quat_wxyz, [gripper]]).astype(np.float64)


def test_clean_chunk_passes():
    barrier = ResidualSafetyBarrier(_limits())
    nominal = np.stack([_pose([0.1, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])] * 3)
    commanded = nominal.copy()
    assert barrier.check_chunk(nominal, commanded) == []
    assert barrier.violation_count == 0


def test_workspace_violation_detected():
    barrier = ResidualSafetyBarrier(_limits())
    nominal = np.stack([_pose([0.1, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])])
    commanded = nominal.copy()
    commanded[0, 0] = 0.9  # outside workspace max x=0.5
    violations = barrier.check_chunk(nominal, commanded)
    assert any(v.code == VIOLATION_WORKSPACE for v in violations)
    assert barrier.violation_count == 1


def test_translation_delta_violation_detected():
    barrier = ResidualSafetyBarrier(_limits())
    nominal = np.stack([_pose([0.1, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])])
    commanded = nominal.copy()
    commanded[0, 0] += 0.05  # > 0.02 m delta
    violations = barrier.check_chunk(nominal, commanded)
    assert any(v.code == VIOLATION_TRANSLATION_DELTA for v in violations)


def test_rotation_delta_violation_detected():
    barrier = ResidualSafetyBarrier(_limits())
    nominal = np.stack([_pose([0.1, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])])
    commanded = nominal.copy()
    # 20 degrees around z: w=cos(10deg), z=sin(10deg)
    angle = np.deg2rad(20.0)
    commanded[0, 3:7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    violations = barrier.check_chunk(nominal, commanded)
    assert any(v.code == VIOLATION_ROTATION_DELTA for v in violations)


def test_non_finite_and_bad_quaternion_detected():
    barrier = ResidualSafetyBarrier(_limits())
    nominal = np.stack([_pose([0.1, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])])
    commanded = nominal.copy()
    commanded[0, 3] = np.nan
    violations = barrier.check_chunk(nominal, commanded)
    assert any(v.code == VIOLATION_NON_FINITE for v in violations)

    commanded = nominal.copy()
    commanded[0, 3:7] = [2.0, 0.0, 0.0, 0.0]  # norm 2 -> not unit
    violations = barrier.check_chunk(nominal, commanded)
    assert any(v.code == VIOLATION_BAD_QUATERNION for v in violations)


def test_hemisphere_flip_detected_across_chunks():
    barrier = ResidualSafetyBarrier(_limits())
    q = [1.0, 0.0, 0.0, 0.0]
    nominal = np.stack([_pose([0.1, 0.1, 0.2], q)] * 2)
    commanded = nominal.copy()
    assert barrier.check_chunk(nominal, commanded) == []
    flipped = nominal.copy()
    flipped[0, 3:7] = [-1.0, 0.0, 0.0, 0.0]  # same rotation, opposite sign
    violations = barrier.check_chunk(flipped, flipped)
    assert any(v.code == VIOLATION_QUATERNION_FLIP for v in violations)


def test_reset_tracking_clears_hemisphere_state():
    barrier = ResidualSafetyBarrier(_limits())
    q = [1.0, 0.0, 0.0, 0.0]
    nominal = np.stack([_pose([0.1, 0.1, 0.2], q)])
    assert barrier.check_chunk(nominal, nominal) == []
    barrier.reset_tracking()
    flipped = nominal.copy()
    flipped[0, 3:7] = [-1.0, 0.0, 0.0, 0.0]
    assert barrier.check_chunk(flipped, flipped) == []
