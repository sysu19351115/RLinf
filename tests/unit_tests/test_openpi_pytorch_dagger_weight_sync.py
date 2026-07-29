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

import asyncio

import pytest
import torch
import torch.nn as nn

from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer
from rlinf.models.embodiment.openpi_pytorch.dagger_action_model import (
    OpenPiPytorchDaggerActionModel,
)
from rlinf.utils.utils import collect_param_names_need_sync


class _TinyPi0(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(2, 2)
        self.action_expert = nn.Linear(2, 2)
        self.action_horizon = 50
        self.action_dim = 32
        for parameter in self.vlm.parameters():
            parameter.requires_grad = False


class _Transport:
    def __init__(self):
        self.sender_to_receiver = asyncio.Queue()
        self.receiver_to_sender = asyncio.Queue()

    async def sender_send(self, value):
        await self.sender_to_receiver.put(value)

    async def sender_recv(self):
        return await self.receiver_to_sender.get()

    async def receiver_send(self, value):
        await self.receiver_to_sender.put(value)

    async def receiver_recv(self):
        return await self.sender_to_receiver.get()


def _wrapper(dtype):
    return OpenPiPytorchDaggerActionModel(
        _TinyPi0().to(dtype=dtype),
        num_steps=5,
        action_env_dim=8,
        action_chunk=10,
        config_name="pi05_dobot_pose",
    )


def _accelerator_device():
    if not torch.cuda.is_available():
        pytest.skip("Patch sync contract requires an accelerator sender.")
    return torch.device("cuda:0")


def _syncer():
    return PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )


def test_actor_and_rollout_wrappers_have_identical_parameter_names():
    actor = _wrapper(torch.float32)
    rollout = _wrapper(torch.bfloat16)

    assert list(actor.state_dict()) == list(rollout.state_dict())
    assert set(dict(actor.named_parameters())) == set(dict(rollout.named_parameters()))
    assert not hasattr(actor, "value_head")
    assert not hasattr(rollout, "value_head")


def test_patch_sync_updates_only_trainable_action_expert_and_casts_to_bf16():
    torch.manual_seed(0)
    device = _accelerator_device()
    actor = _wrapper(torch.float32).to(device)
    rollout = _wrapper(torch.bfloat16).to(device)
    rollout.load_state_dict(
        {
            key: value.to(dtype=rollout.state_dict()[key].dtype)
            for key, value in actor.state_dict().items()
        }
    )
    frozen_before = rollout.model.vlm.weight.detach().clone()
    sender = _syncer()
    receiver = _syncer()
    transport = _Transport()

    async def _run():
        await asyncio.gather(
            sender.init_sender(
                actor.state_dict(),
                collect_param_names_need_sync(actor),
                transport.sender_send,
                transport.sender_recv,
            ),
            receiver.init_receiver(
                rollout.state_dict(),
                transport.receiver_recv,
                transport.receiver_send,
            ),
        )
        with torch.no_grad():
            actor.model.action_expert.weight[0, 0] += 0.5
        await sender.sync(
            actor.state_dict(),
            transport.sender_send,
            version=1,
        )
        first_version = await receiver.apply(rollout, transport.receiver_recv)
        await sender.sync(
            actor.state_dict(),
            transport.sender_send,
            version=2,
        )
        second_version = await receiver.apply(rollout, transport.receiver_recv)
        return first_version, second_version

    first_version, second_version = asyncio.run(_run())

    assert (first_version, second_version) == (1, 2)
    assert rollout.model.action_expert.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        rollout.model.action_expert.weight,
        actor.model.action_expert.weight.to(torch.bfloat16),
    )
    torch.testing.assert_close(rollout.model.vlm.weight, frozen_before)
