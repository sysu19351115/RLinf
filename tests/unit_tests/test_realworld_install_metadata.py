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

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_project_python_range_matches_installer_support():
    requires_python = SpecifierSet(_pyproject()["project"]["requires-python"])

    assert Version("3.10") in requires_python
    assert Version("3.11") in requires_python
    assert Version("3.12") not in requires_python


def _extra_requirements(extra: str) -> dict[str, Requirement]:
    requirements = [
        Requirement(item)
        for item in _pyproject()["project"]["optional-dependencies"][extra]
    ]
    return {requirement.name: requirement for requirement in requirements}


def test_openpi_robot_extras_keep_numpy1():
    for extra in ("rebot", "so101", "dobot"):
        numpy = _extra_requirements(extra)["numpy"]
        assert Version("1.26.4") in numpy.specifier
        assert Version("2.0.0") not in numpy.specifier


def test_so101_pinocchio_stays_on_compatible_release():
    pin = _extra_requirements("so101")["pin"]

    assert Version("2.7.0") in pin.specifier
    assert Version("3.0.0") not in pin.specifier


def test_cmeel_boost_numpy_metadata_override_is_scoped():
    overrides = _pyproject()["tool"]["uv"]["override-dependencies"]

    scoped_override = next(
        override
        for override in overrides
        if isinstance(override, dict)
        and override.get("package", {}).get("name") == "cmeel-boost"
    )
    numpy = Requirement(scoped_override["dependencies"][0])

    assert numpy.name == "numpy"
    assert Version("1.26.4") in numpy.specifier
    assert Version("3.0.0") not in numpy.specifier


def test_realworld_robot_extras_are_mutually_exclusive():
    conflict_sets = _pyproject()["tool"]["uv"]["conflicts"]
    robot_extras = {"franka", "xsquare_turtle2", "gim_arm", "rebot", "so101", "dobot"}

    assert any(
        robot_extras
        <= {entry["extra"] for entry in conflict_set}
        for conflict_set in conflict_sets
    )


def test_numpy1_robot_extras_conflict_with_agentic_backends():
    conflict_sets = _pyproject()["tool"]["uv"]["conflicts"]
    incompatible_extras = {
        "agentic-sglang",
        "agentic-vllm",
        "franka",
        "xsquare_turtle2",
        "rebot",
        "so101",
        "dobot",
    }

    assert any(
        incompatible_extras
        <= {entry["extra"] for entry in conflict_set}
        for conflict_set in conflict_sets
    )
