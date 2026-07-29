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

from types import SimpleNamespace

import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as model_lib
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


class _FakeLlm:
    def __call__(self, tokens, **_kwargs):
        return ([tokens[0], tokens[1]],)


def _fake_pi0(batch_size: int, horizon: int, action_dim: int):
    prefix = torch.zeros(batch_size, 1, action_dim)
    suffix = torch.zeros(batch_size, horizon, action_dim)

    return SimpleNamespace(
        action_horizon=horizon,
        embed_dtype=torch.float32,
        llm=_FakeLlm(),
        action_out_proj=lambda value: value,
        embed_prefix=lambda _observation: (
            prefix,
            torch.ones(batch_size, 1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        ),
        embed_suffix=lambda _observation, _actions, _time: (
            suffix,
            torch.ones(batch_size, horizon, dtype=torch.bool),
            torch.zeros(horizon, dtype=torch.bool),
            None,
        ),
    )


def test_compute_loss_can_preserve_action_dimension(monkeypatch):
    batch_size, horizon, action_dim = 2, 3, 4
    fake = _fake_pi0(batch_size, horizon, action_dim)
    actions = torch.zeros(batch_size, horizon, action_dim)
    noise = torch.ones_like(actions)
    time = torch.full((batch_size,), 0.5)

    monkeypatch.setattr(
        model_lib, "preprocess_observation", lambda observation, **_kwargs: observation
    )
    monkeypatch.setattr(
        model_lib, "_observation_to_dtype", lambda observation, _dtype: observation
    )

    unreduced = Pi0.compute_loss(
        fake,
        object(),
        actions,
        train=True,
        noise=noise,
        time=time,
        reduce_action_dim=False,
    )
    default = Pi0.compute_loss(
        fake,
        object(),
        actions,
        train=True,
        noise=noise,
        time=time,
    )

    assert unreduced.shape == (batch_size, horizon, action_dim)
    assert default.shape == (batch_size, horizon)
    torch.testing.assert_close(default, unreduced.mean(dim=-1))
