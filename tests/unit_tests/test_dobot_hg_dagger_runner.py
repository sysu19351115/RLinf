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

"""Global operator-quit control tests for Dobot HG-DAgger."""

import torch

from rlinf.runners.embodied_runner import operator_quit_requested


def test_operator_quit_detected_across_env_workers():
    results = [
        {"episode_end/operator_abort": torch.tensor([1.0])},
        {"episode_end/operator_quit": torch.tensor([0.0, 1.0])},
    ]

    assert operator_quit_requested(results)


def test_non_quit_episode_end_does_not_stop_training():
    results = [
        {"episode_end/operator_success": torch.tensor([1.0])},
        {"episode_end/operator_abort": torch.tensor([1.0])},
    ]

    assert not operator_quit_requested(results)
