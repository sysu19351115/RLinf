# Copyright 2025 The RLinf Authors.
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

import json
import os

import torch
from omegaconf import OmegaConf

_REGISTERED = False


def _project_root() -> str:
    """Return the RLinf project root (three levels above this file)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _project_path(rel_path: str) -> str:
    """Resolve a path relative to the RLinf project root.

    Works regardless of Hydra's runtime working directory.
    """
    path = str(rel_path)
    if os.path.isabs(path):
        return path
    return os.path.join(_project_root(), path)


def _json_load(path: str, key: str | None = None):
    """Load a JSON file and optionally return one of its top-level keys.

    Relative paths are resolved against the RLinf project root. This keeps YAML
    resolvers stable even when Hydra changes the working directory at runtime.
    """
    path = _project_path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if key is None else data[key]


def omegaconf_register():
    global _REGISTERED
    if _REGISTERED:  # avoid duplicate
        return
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("not", lambda x: not bool(x))
    OmegaConf.register_new_resolver(
        "torch.dtype", lambda dtype_name: getattr(torch, dtype_name), replace=True
    )
    OmegaConf.register_new_resolver("json_load", _json_load)
    OmegaConf.register_new_resolver("project_path", _project_path)
    _REGISTERED = True


# register when import
omegaconf_register()
