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

import torch.nn as nn

from rlinf.models.embodiment.openpi_pytorch.dagger_action_model import (
    OpenPiPytorchDaggerActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.utils.freeze import (
    freeze_paligemma_vlm,
)


def _experts():
    return nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _experts()
        self.k_proj = _experts()
        self.v_proj = _experts()
        self.o_proj = _experts()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention_norms = _experts()
        self.pre_ffw_norms = _experts()
        self.mlps = _experts()
        self.attn = _Attention()


class _Llm(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([_Block()])
        self.final_norms = _experts()


class _FakePi0(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = nn.Linear(2, 2)
        self.llm = _Llm()
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.action_horizon = 50
        self.action_dim = 32


def _all_frozen(module):
    return all(not parameter.requires_grad for parameter in module.parameters())


def _has_trainable(module):
    return any(parameter.requires_grad for parameter in module.parameters())


def test_freeze_paligemma_leaves_only_action_side_trainable():
    model = _FakePi0()

    frozen = freeze_paligemma_vlm(model)

    assert frozen > 0
    assert _all_frozen(model.img)
    assert _all_frozen(model.llm.embedder)
    block = model.llm.layers[0]
    assert _all_frozen(block.pre_attention_norms[0])
    assert _all_frozen(block.pre_ffw_norms[0])
    assert _all_frozen(block.mlps[0])
    assert _all_frozen(block.attn.q_proj[0])
    assert _all_frozen(model.llm.final_norms[0])
    assert _has_trainable(block.pre_attention_norms[1])
    assert _has_trainable(block.mlps[1])
    assert _has_trainable(block.attn.q_proj[1])
    assert _has_trainable(model.llm.final_norms[1])
    assert _has_trainable(model.action_in_proj)
    assert _has_trainable(model.action_out_proj)
    assert _has_trainable(model.action_time_mlp_in)
    assert _has_trainable(model.state_proj)


def test_dagger_wrapper_adds_no_value_head_and_delegates_freeze():
    core = _FakePi0()
    wrapper = OpenPiPytorchDaggerActionModel(
        core,
        num_steps=10,
        action_env_dim=8,
        action_chunk=10,
    )

    wrapper.freeze_vlm()

    assert not hasattr(wrapper, "value_head")
    assert _all_frozen(core.img)
    assert _has_trainable(core.llm.layers[0].mlps[1])
