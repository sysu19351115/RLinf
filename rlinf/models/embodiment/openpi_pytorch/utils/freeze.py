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

"""Shared train-expert-only freezing policy for OpenPI PyTorch."""

from __future__ import annotations

from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


def freeze_paligemma_vlm(pi0_model: Pi0) -> int:
    """Freeze SigLIP and Gemma expert 0 while leaving action expert 1 trainable."""
    frozen = 0

    def _freeze(module) -> None:
        nonlocal frozen
        if module is None:
            return
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += 1

    _freeze(pi0_model.img)
    llm = pi0_model.llm
    _freeze(llm.embedder)
    for block in llm.layers:
        _freeze(block.pre_attention_norms[0])
        _freeze(block.pre_ffw_norms[0])
        _freeze(block.mlps[0])
        for projections in (
            block.attn.q_proj,
            block.attn.k_proj,
            block.attn.v_proj,
            block.attn.o_proj,
        ):
            _freeze(projections[0])
    _freeze(llm.final_norms[0])
    return frozen
