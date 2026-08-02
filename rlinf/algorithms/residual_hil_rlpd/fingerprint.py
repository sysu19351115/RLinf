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

"""Content-based fingerprints for base policy / norm stats / codec."""

from __future__ import annotations

import hashlib
import os

from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec

_TRANSITION_SCHEMA_VERSION = "dobot_residual_hil_rlpd_v1"


def _file_hash(path: str | None) -> str:
    if not path or not os.path.isfile(path):
        return "no-file"
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _weights_dir_hash(model_path: str | None) -> str:
    """Hash all files under the base checkpoint directory (content-based)."""
    if not model_path or not os.path.isdir(model_path):
        return "no-weights-dir"
    digest = hashlib.sha256()
    for name in sorted(os.listdir(model_path)):
        full = os.path.join(model_path, name)
        if not os.path.isfile(full):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(_file_hash(full).encode("utf-8"))
    return digest.hexdigest()[:16]


def compute_base_fingerprint(
    model_path: str,
    norm_stats_path: str | None = None,
    codec: ResidualCodec | None = None,
) -> str:
    """Composite content fingerprint for the frozen base + codec."""
    payload = "|".join(
        [
            _TRANSITION_SCHEMA_VERSION,
            os.path.realpath(model_path) if model_path else "no-path",
            _weights_dir_hash(model_path),
            _file_hash(norm_stats_path),
            codec.fingerprint() if codec is not None else "no-codec",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
