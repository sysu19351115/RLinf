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
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_dagger_policy_worker import (
    EmbodiedDAGGERFSDPPolicy,
)


class _FakeDaggerModel:
    def __init__(self, loss):
        self.loss = loss
        self.prepared = None
        self.forwarded = None

    def prepare_dagger_sft_batch(self, batch, *, loss_scope):
        self.prepared = (batch, loss_scope)
        return {"prepared": True}

    def __call__(self, **kwargs):
        self.forwarded = kwargs
        return self.loss


def _worker(loss=torch.tensor(1.0)):
    worker = object.__new__(EmbodiedDAGGERFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "openpi_pytorch"}},
            "algorithm": {
                "dagger": {
                    "loss_scope": "human_only",
                    "online_lerobot": {"enabled": False},
                }
            },
        }
    )
    worker.enable_online_lerobot = False
    worker.model = _FakeDaggerModel(loss)
    return worker


def _raw_forward_actor(worker, batch):
    function = EmbodiedDAGGERFSDPPolicy.forward_actor
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function(worker, batch)


def test_worker_routes_openpi_pytorch_to_chunk_aware_human_loss():
    worker = _worker()
    replay = {"action": torch.zeros(1, 400)}

    loss = _raw_forward_actor(worker, replay)

    assert loss.ndim == 0
    assert worker.model.prepared == (replay, "human_only")
    assert worker.model.forwarded == {
        "forward_type": ForwardType.SFT,
        "data": {"prepared": True},
        "use_action_chunk_loss": True,
    }


@pytest.mark.parametrize(
    "loss",
    [
        torch.ones(2),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        1.0,
    ],
)
def test_worker_rejects_non_scalar_or_nonfinite_loss(loss):
    worker = _worker(loss)

    with pytest.raises(ValueError, match="finite scalar loss"):
        _raw_forward_actor(worker, {"action": torch.zeros(1, 400)})
