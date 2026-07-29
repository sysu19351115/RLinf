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

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch.dagger_action_model import (
    OpenPiPytorchDaggerActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import (
    _build_dagger_model,
)


class _FakePi0(nn.Module):
    def __init__(self, *, horizon=50, action_dim=32):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_horizon = horizon
        self.action_dim = action_dim
        self.errors = None

    def compute_loss(self, _observation, actions, **kwargs):
        assert kwargs["reduce_action_dim"] is False
        if self.errors is None:
            return torch.ones_like(actions)
        return self.errors.to(actions.device)


def _model():
    return OpenPiPytorchDaggerActionModel(
        _FakePi0(),
        num_steps=10,
        action_env_dim=8,
        action_chunk=10,
        config_name="pi05_behavior",
    )


def _batch(batch_size=2):
    horizon, env_dim = 50, 8
    return {
        "action": torch.zeros(batch_size, 1, horizon * env_dim),
        "human_action_mask": torch.cat(
            [
                torch.ones(batch_size, 1, 30),
                torch.zeros(batch_size, 1, 20),
            ],
            dim=-1,
        ),
        "observation/image": torch.zeros(batch_size, 1, 3, 8, 8, dtype=torch.uint8),
        "observation/state": torch.zeros(batch_size, 1, env_dim),
        "observation/prev_state": torch.zeros(batch_size, 1, env_dim),
        "tokenized_prompt": torch.zeros(batch_size, 1, 6, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(batch_size, 1, 6, dtype=torch.bool),
    }


def _install_fake_transform(model):
    def _transform(obs, transpose=False):
        del transpose
        actions = torch.as_tensor(obs["actions"])
        padded = torch.nn.functional.pad(actions, (0, model.model.action_dim - 8))
        return {
            "image": {"base_0_rgb": torch.zeros(actions.shape[0], 8, 8, 3)},
            "image_mask": {
                "base_0_rgb": torch.ones(actions.shape[0], dtype=torch.bool)
            },
            "state": torch.zeros(actions.shape[0], model.model.action_dim),
            "tokenized_prompt": torch.zeros(actions.shape[0], 6, dtype=torch.long),
            "tokenized_prompt_mask": torch.ones(actions.shape[0], 6, dtype=torch.bool),
            "actions": padded,
        }

    model.input_transform = _transform


def test_prepare_dagger_batch_normalizes_shape_and_preserves_cached_tokens():
    model = _model()
    _install_fake_transform(model)

    prepared = model.prepare_dagger_sft_batch(_batch(), loss_scope="human_only")

    assert prepared["actions"].shape == (2, 50, 32)
    assert prepared["loss_mask"].shape == (2, 50)
    assert prepared["loss_mask"][:, :30].all()
    assert not prepared["loss_mask"][:, 30:].any()
    assert prepared["observation"].tokenized_prompt.shape == (2, 6)


def test_human_only_loss_excludes_model_recovery_steps():
    model = _model()
    errors = torch.full((1, 50, 32), 100.0)
    errors[:, :30, :8] = 1.0
    model.model.errors = errors
    data = {
        "observation": object(),
        "actions": torch.zeros(1, 50, 32),
        "loss_mask": torch.cat([torch.ones(1, 30), torch.zeros(1, 20)], dim=1).bool(),
    }

    loss = model.sft_forward(data)

    torch.testing.assert_close(loss, torch.tensor(1.0))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda batch: batch.pop("human_action_mask"), "requires human_action_mask"),
        (
            lambda batch: batch.__setitem__("human_action_mask", torch.ones(2, 1, 49)),
            "loss mask must have shape",
        ),
        (
            lambda batch: batch.__setitem__("human_action_mask", torch.zeros(2, 1, 50)),
            "selects no human",
        ),
        (
            lambda batch: batch.__setitem__("action", torch.zeros(2, 1, 399)),
            "action has",
        ),
        (
            lambda batch: batch["action"].fill_(float("nan")),
            "contain NaN or Inf",
        ),
        (
            lambda batch: batch.pop("observation/prev_state"),
            "missing 'observation/prev_state'",
        ),
    ],
)
def test_prepare_dagger_batch_fails_closed(mutate, match):
    model = _model()
    _install_fake_transform(model)
    batch = _batch()
    mutate(batch)

    with pytest.raises(ValueError, match=match):
        model.prepare_dagger_sft_batch(batch, loss_scope="human_only")


def test_dagger_builder_installs_eval_transforms(monkeypatch):
    from rlinf.models.embodiment.openpi_pytorch import transforms_pipeline

    monkeypatch.setattr(
        transforms_pipeline,
        "build_openpi_transforms",
        lambda *_args, **_kwargs: (["input"], ["output"]),
    )
    installed = {}
    monkeypatch.setattr(
        OpenPiPytorchDaggerActionModel,
        "setup_wrappers",
        lambda self, inputs, outputs: installed.update(inputs=inputs, outputs=outputs),
    )
    cfg = OmegaConf.create({"model_path": "/checkpoint"})
    model_cfg = OmegaConf.create({"config_name": "pi05_behavior"})

    result = _build_dagger_model(
        cfg,
        model_cfg,
        _FakePi0(),
        num_steps=10,
        action_chunk=10,
        action_env_dim=8,
    )

    assert isinstance(result, OpenPiPytorchDaggerActionModel)
    assert result.preserve_dagger_anchor_inputs is True
    assert installed == {"inputs": ["input"], "outputs": ["output"]}
