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

"""Property tests for the reversible SE(3) residual codec."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from rlinf.algorithms.residual_hil_rlpd.action_codec import (
    FORCE_CLOSE,
    FORCE_OPEN,
    KEEP_NOMINAL,
    ResidualCodec,
    combine_gripper_mode,
    gripper_one_hot,
    invert_gripper_mode,
)


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    position = rng.uniform(-0.3, 0.3, size=3)
    quat = Rotation.random(random_state=rng).as_quat()  # xyzw
    return np.concatenate([position, [quat[3], quat[0], quat[1], quat[2]]])


def _apply_transform(pose: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_matrix(matrix[:3, :3])
    position = matrix[:3, 3] + matrix[:3, :3] @ pose[:3]
    quat = rotation * Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
    q = quat.as_quat()
    return np.concatenate([position, [q[3], q[0], q[1], q[2]]])


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    matrix[:3, 3] = rng.uniform(-0.2, 0.2, size=3)
    return matrix


@pytest.fixture()
def codec():
    return ResidualCodec()


def test_round_trip(codec):
    rng = np.random.default_rng(0)
    for _ in range(50):
        nominal = _random_pose(rng)
        u = rng.uniform(-0.9, 0.9, size=6)
        commanded = codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        ).numpy()

        u_back, out_of_support = codec.invert(nominal, commanded)

        assert not out_of_support
        np.testing.assert_allclose(u_back, u, atol=2e-5, rtol=0)


def test_zero_residual_is_identity(codec):
    rng = np.random.default_rng(1)
    for _ in range(20):
        nominal = _random_pose(rng)
        commanded = codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.zeros(6, dtype=torch.float32),
        ).numpy()

        np.testing.assert_allclose(commanded[:3], nominal[:3], atol=1e-6)
        dot = abs(np.dot(commanded[3:], nominal[3:]))
        assert dot > 1.0 - 1e-6


def test_negated_quaternion_equivalence(codec):
    rng = np.random.default_rng(2)
    for _ in range(20):
        nominal = _random_pose(rng)
        u = rng.uniform(-0.5, 0.5, size=6)
        nominal_neg = nominal.copy()
        nominal_neg[3:] = -nominal_neg[3:]

        a = codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        ).numpy()
        b = codec.compose(
            torch.as_tensor(nominal_neg, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        ).numpy()

        np.testing.assert_allclose(a[:3], b[:3], atol=1e-5)
        assert abs(np.dot(a[3:], b[3:])) > 1.0 - 1e-5


def test_left_invariance_under_global_transform(codec):
    rng = np.random.default_rng(3)
    for _ in range(20):
        nominal = _random_pose(rng)
        u = rng.uniform(-0.8, 0.8, size=6)
        commanded = codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        ).numpy()
        g = _random_transform(rng)

        u_original, _ = codec.invert(nominal, commanded)
        u_transformed, _ = codec.invert(
            _apply_transform(nominal, g),
            _apply_transform(commanded, g),
        )

        np.testing.assert_allclose(u_transformed, u_original, atol=1e-5, rtol=0)


def test_near_zero_and_near_pi_rotations(codec):
    rng = np.random.default_rng(4)
    # Near-zero rotation stays inside the default 30 deg data limit.
    u = np.zeros(6)
    u[3:] = np.deg2rad(1e-4) / np.deg2rad(30.0)
    nominal = _random_pose(rng)
    commanded = codec.compose(
        torch.as_tensor(nominal, dtype=torch.float32),
        torch.as_tensor(u, dtype=torch.float32),
    ).numpy()
    u_back, out_of_support = codec.invert(nominal, commanded)
    assert not out_of_support
    np.testing.assert_allclose(u_back[3:], u[3:], atol=2e-5, rtol=0)

    # Near-pi single-axis rotation: the composed rotation vector must stay
    # below pi radians (rotvec canonicalizes beyond pi), and the data limit
    # must be wide enough for 179 deg to stay inside [-1, 1].
    wide = ResidualCodec(rotation_scale_deg=(200.0, 200.0, 200.0))
    u = np.zeros(6)
    u[3] = np.deg2rad(179.0) / np.deg2rad(200.0)
    commanded = wide.compose(
        torch.as_tensor(nominal, dtype=torch.float32),
        torch.as_tensor(u, dtype=torch.float32),
    ).numpy()
    u_back, out_of_support = wide.invert(nominal, commanded)
    assert not out_of_support
    np.testing.assert_allclose(u_back[3:], u[3:], atol=2e-5, rtol=0)


def test_zero_or_nan_quaternion_rejected(codec):
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero-norm"):
        codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.zeros(6, dtype=torch.float32),
        )
    nominal[3] = 1.0
    nominal[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        codec.invert(nominal, _random_pose(np.random.default_rng(5)))


def test_out_of_support_is_flagged_not_clipped(codec):
    nominal = np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    executed = nominal.copy()
    executed[:3] += np.array([0.5, 0.0, 0.0])  # 0.5 m >> 0.1 m data limit
    _, out_of_support = codec.invert(nominal, executed)
    assert out_of_support

    u = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="refusing to silently clip"):
        codec.compose(
            torch.as_tensor(nominal, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        )


def test_gripper_mode_semantics():
    assert combine_gripper_mode(0.7, KEEP_NOMINAL) == (0.7, False)
    assert combine_gripper_mode(0.7, FORCE_CLOSE) == (0.0, True)
    assert combine_gripper_mode(0.7, FORCE_OPEN) == (1.0, True)

    assert invert_gripper_mode(0.0, KEEP_NOMINAL, human_intervened=True) == (
        FORCE_CLOSE
    )
    assert invert_gripper_mode(1.0, KEEP_NOMINAL, human_intervened=True) == (FORCE_OPEN)
    assert invert_gripper_mode(0.9, KEEP_NOMINAL, human_intervened=False) == (
        KEEP_NOMINAL
    )

    np.testing.assert_allclose(gripper_one_hot(FORCE_CLOSE), [0, 1, 0])
    np.testing.assert_allclose(gripper_one_hot(FORCE_OPEN), [0, 0, 1])


def test_fingerprint_stable_and_sensitive():
    a = ResidualCodec()
    b = ResidualCodec()
    c = ResidualCodec(translation_scale_m=(0.2, 0.1, 0.1))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
