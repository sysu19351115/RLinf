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

"""Unit tests for FSDP1 frozen-parameter ignored states."""

from __future__ import annotations

import asyncio
import logging
import os
import socket

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp import ShardingStrategy
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy import fsdp as fsdp_module
from rlinf.hybrid_engines.fsdp.strategy.fsdp import (
    FSDPStrategy,
    _resolve_ignored_states,
)
from rlinf.hybrid_engines.fsdp.utils import create_device_mesh
from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer
from rlinf.scheduler import Worker
from rlinf.utils.utils import collect_param_names_need_sync


class _TinyMixedTrainabilityModel(nn.Module):
    """Tiny stand-in for the Dobot OpenPI PPO actor.

    The frozen embedding and VLM expert mirror the ``train_expert_only``
    freeze applied by OpenPI PyTorch; the action expert and value head stay
    trainable and must never land in ``ignored_states``.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.vlm_expert = nn.Linear(4, 4)
        self.action_expert = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False
        for parameter in self.vlm_expert.parameters():
            parameter.requires_grad = False

    def forward(self, tokens, features):
        x = self.embedding(tokens)
        x = x + features
        x = self.vlm_expert(x)
        x = self.action_expert(x)
        return self.value_head(x)


def _fsdp_cfg(ignore_frozen_params: bool, use_orig_params: bool):
    return OmegaConf.create(
        {
            "ignore_frozen_params": ignore_frozen_params,
            "use_orig_params": use_orig_params,
        }
    )


def test_collects_only_frozen_parameters():
    model = _TinyMixedTrainabilityModel()
    ignored = _resolve_ignored_states(
        model,
        _fsdp_cfg(ignore_frozen_params=True, use_orig_params=True),
        ShardingStrategy.NO_SHARD,
    )

    assert ignored is not None
    assert set(ignored) == {
        *model.embedding.parameters(),
        *model.vlm_expert.parameters(),
    }
    assert all(not parameter.requires_grad for parameter in ignored)
    assert not set(model.action_expert.parameters()) & set(ignored)
    assert not set(model.value_head.parameters()) & set(ignored)
    # A shared parameter object may only appear once in ignored_states.
    assert len(ignored) == len({id(parameter) for parameter in ignored})


def test_disabled_returns_none():
    model = _TinyMixedTrainabilityModel()
    ignored = _resolve_ignored_states(
        model,
        _fsdp_cfg(ignore_frozen_params=False, use_orig_params=True),
        ShardingStrategy.NO_SHARD,
    )

    assert ignored is None


def test_rejects_use_orig_params_false():
    model = _TinyMixedTrainabilityModel()
    with pytest.raises(ValueError, match="requires use_orig_params=True"):
        _resolve_ignored_states(
            model,
            _fsdp_cfg(ignore_frozen_params=True, use_orig_params=False),
            ShardingStrategy.NO_SHARD,
        )


def test_rejects_non_no_shard_strategy():
    model = _TinyMixedTrainabilityModel()
    with pytest.raises(
        ValueError, match="currently supports sharding_strategy=no_shard only"
    ):
        _resolve_ignored_states(
            model,
            _fsdp_cfg(ignore_frozen_params=True, use_orig_params=True),
            ShardingStrategy.FULL_SHARD,
        )


def test_rejects_model_without_frozen_parameters():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no frozen parameters"):
        _resolve_ignored_states(
            model,
            _fsdp_cfg(ignore_frozen_params=True, use_orig_params=True),
            ShardingStrategy.NO_SHARD,
        )


def test_rejects_model_without_trainable_parameters():
    model = nn.Linear(4, 4)
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        _resolve_ignored_states(
            model,
            _fsdp_cfg(ignore_frozen_params=True, use_orig_params=True),
            ShardingStrategy.NO_SHARD,
        )


def _strategy_cfg(ignore_frozen_params: bool, use_orig_params: bool):
    return OmegaConf.create(
        {
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "use_orig_params": use_orig_params,
                "ignore_frozen_params": ignore_frozen_params,
                "mixed_precision": {
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "buffer_dtype": "bf16",
                },
                "cpu_offload": False,
                "forward_prefetch": False,
                "backward_prefetch": None,
                "limit_all_gathers": False,
            },
            "model": {
                "is_lora": False,
                "model_type": "openpi_pytorch",
            },
        }
    )


def test_wrap_model_passes_ignored_states_when_enabled(monkeypatch):
    captured_kwargs = {}

    class _FakeFSDP:
        def __init__(self, module, **kwargs):
            captured_kwargs.update(kwargs)
            self.module = module

    monkeypatch.setattr(fsdp_module, "FSDP", _FakeFSDP)
    monkeypatch.setenv("LOCAL_RANK", "0")
    # The wiring test must stay CPU-only: with FSDP mocked, wrap_model still
    # performs the real ignored-parameter device move.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    model = _TinyMixedTrainabilityModel()
    cfg = _strategy_cfg(ignore_frozen_params=True, use_orig_params=True)
    strategy = FSDPStrategy(cfg=cfg, world_size=1)
    ignored = _resolve_ignored_states(
        model,
        cfg.fsdp_config,
        ShardingStrategy.NO_SHARD,
    )

    strategy.wrap_model(model, device_mesh=object())

    assert captured_kwargs["ignored_states"] == ignored
    assert captured_kwargs["use_orig_params"] is True


def test_wrap_model_rejects_world_size_greater_than_one(monkeypatch):
    captured_kwargs = {}

    class _FakeFSDP:
        def __init__(self, module, **kwargs):
            captured_kwargs.update(kwargs)
            self.module = module

    monkeypatch.setattr(fsdp_module, "FSDP", _FakeFSDP)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    model = _TinyMixedTrainabilityModel()
    cfg = _strategy_cfg(ignore_frozen_params=True, use_orig_params=True)
    strategy = FSDPStrategy(cfg=cfg, world_size=2)

    with pytest.raises(ValueError, match="world_size=1 only"):
        strategy.wrap_model(model, device_mesh=object())


def test_wrap_model_passes_none_ignored_states_when_disabled(monkeypatch):
    captured_kwargs = {}

    class _FakeFSDP:
        def __init__(self, module, **kwargs):
            captured_kwargs.update(kwargs)
            self.module = module

    monkeypatch.setattr(fsdp_module, "FSDP", _FakeFSDP)
    monkeypatch.setenv("LOCAL_RANK", "0")

    model = _TinyMixedTrainabilityModel()
    cfg = _strategy_cfg(ignore_frozen_params=False, use_orig_params=True)
    strategy = FSDPStrategy(cfg=cfg, world_size=1)

    strategy.wrap_model(model, device_mesh=object())

    assert captured_kwargs["ignored_states"] is None
    assert captured_kwargs["use_orig_params"] is True


def _optimizer_contains(optimizer, parameter):
    return any(
        any(parameter is candidate for candidate in group["params"])
        for group in optimizer.param_groups
    )


@pytest.fixture(scope="module")
def fsdp_single_rank():
    """Initialize a single-rank CUDA/NCCL process group for FSDP tests."""
    if not torch.cuda.is_available():
        pytest.skip("FSDP integration tests require CUDA/NCCL")
    if torch.distributed.is_initialized():
        pytest.skip("Torch distributed is already initialized")

    saved_env = {
        key: os.environ.get(key)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    saved_device_type = getattr(Worker, "torch_device_type", None)
    saved_platform = getattr(Worker, "torch_platform", None)

    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]
    os.environ["MASTER_PORT"] = str(master_port)

    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda

    torch.distributed.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_device_type is None:
            Worker.__dict__.pop("torch_device_type", None)
        else:
            Worker.torch_device_type = saved_device_type
        if saved_platform is None:
            Worker.__dict__.pop("torch_platform", None)
        else:
            Worker.torch_platform = saved_platform


def _wrapped_tiny_actor():
    torch.manual_seed(0)
    # Mirror production: from_pretrained(torch_dtype=bf16) loads every weight
    # in bf16, so ignored frozen params keep bf16 while FSDP manages trainable
    # params with the same param_dtype.
    model = _TinyMixedTrainabilityModel().to(dtype=torch.bfloat16)
    cfg = _strategy_cfg(ignore_frozen_params=True, use_orig_params=True)
    strategy = FSDPStrategy(cfg=cfg, world_size=1)
    wrapped = strategy.wrap_model(model, create_device_mesh(1))
    return model, wrapped, strategy


def test_mixed_trainability_fsdp_forward_backward(fsdp_single_rank):
    model, wrapped, _ = _wrapped_tiny_actor()

    tokens = torch.randint(0, 16, (2, 8), device="cuda")
    features = torch.randn(2, 8, 4, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        wrapped.zero_grad(set_to_none=True)
        loss = wrapped(tokens, features).float().sum()
        loss.backward()

    assert model.embedding.weight.grad is None
    assert model.vlm_expert.weight.grad is None
    assert model.action_expert.weight.grad is not None
    assert model.value_head.weight.grad is not None

    # FSDP re-parents the root module under ``_fsdp_wrapped_module``; the
    # original names must still be visible (with that prefix stripped in the
    # production sync path, which collects names before wrapping).
    names = {name for name, _ in wrapped.named_parameters()}
    assert any("action_expert" in name for name in names)
    assert any("value_head" in name for name in names)
    assert any("embedding" in name for name in names)
    assert any("vlm_expert" in name for name in names)


def test_full_state_dict_keeps_frozen_parameters(fsdp_single_rank):
    model, wrapped, strategy = _wrapped_tiny_actor()

    state_dict = strategy.get_model_state_dict(
        wrapped,
        cpu_offload=True,
        full_state_dict=True,
    )

    unwrapped_keys = set(_TinyMixedTrainabilityModel().state_dict().keys())
    assert "embedding.weight" in state_dict
    assert "vlm_expert.weight" in state_dict
    assert "action_expert.weight" in state_dict
    assert "value_head.weight" in state_dict
    assert unwrapped_keys <= set(state_dict.keys())


def test_critic_warmup_parameter_grouping(fsdp_single_rank):
    model, wrapped, _ = _wrapped_tiny_actor()
    cfg = _strategy_cfg(ignore_frozen_params=True, use_orig_params=True)
    cfg.optim = OmegaConf.create(
        {
            "lr": 1e-4,
            "value_lr": 1e-3,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
        }
    )

    manager = object.__new__(FSDPModelManager)
    manager._cfg = cfg
    manager._logger = logging.getLogger("test_fsdp_ignore_frozen_params")
    manager.store_requires_grad_param_name = []

    warmup_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=True,
    )
    assert _optimizer_contains(warmup_optimizer, model.value_head.weight)
    assert not model.action_expert.weight.requires_grad

    regular_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=False,
    )
    assert model.action_expert.weight.requires_grad
    assert _optimizer_contains(regular_optimizer, model.action_expert.weight)
    assert _optimizer_contains(regular_optimizer, model.value_head.weight)


def test_collect_param_names_need_sync_excludes_frozen_params():
    model = _TinyMixedTrainabilityModel()

    names = collect_param_names_need_sync(model)

    assert any("action_expert" in name for name in names)
    assert any("value_head" in name for name in names)
    assert not any("embedding" in name for name in names)
    assert not any("vlm_expert" in name for name in names)


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


def _patch_syncer(init_sync_enabled: bool = False):
    return PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=init_sync_enabled,
    )


def test_patch_sync_full_and_incremental_with_frozen_fsdp(fsdp_single_rank):
    model, wrapped, strategy = _wrapped_tiny_actor()
    actor_state_dict = strategy.get_model_state_dict(
        wrapped,
        cpu_offload=False,
        full_state_dict=True,
    )

    rollout = _TinyMixedTrainabilityModel().to(
        dtype=torch.bfloat16,
        device="cuda",
    )
    sender = _patch_syncer(init_sync_enabled=True)
    receiver = _patch_syncer(init_sync_enabled=True)
    transport = _Transport()

    async def _run():
        await asyncio.gather(
            sender.init_sender(
                actor_state_dict,
                collect_param_names_need_sync(model),
                transport.sender_send,
                transport.sender_recv,
            ),
            receiver.init_receiver(
                rollout.state_dict(),
                transport.receiver_recv,
                transport.receiver_send,
            ),
        )

        # First full sync must bring frozen VLM/embedding and trainable params
        # into agreement; the incremental patch below must not touch frozen ones.
        torch.testing.assert_close(
            rollout.embedding.weight,
            model.embedding.weight.to(torch.bfloat16),
        )
        torch.testing.assert_close(
            rollout.vlm_expert.weight,
            model.vlm_expert.weight.to(torch.bfloat16),
        )
        torch.testing.assert_close(
            rollout.action_expert.weight,
            model.action_expert.weight.to(torch.bfloat16),
        )
        torch.testing.assert_close(
            rollout.value_head.weight,
            model.value_head.weight.to(torch.bfloat16),
        )
        frozen_before = {
            key: rollout.state_dict()[key].detach().clone()
            for key in (
                "embedding.weight",
                "vlm_expert.weight",
                "vlm_expert.bias",
            )
        }

        with torch.no_grad():
            model.action_expert.weight.add_(0.25)
            model.value_head.weight.add_(0.5)
        updated_state_dict = strategy.get_model_state_dict(
            wrapped,
            cpu_offload=False,
            full_state_dict=True,
        )
        await sender.sync(
            updated_state_dict,
            transport.sender_send,
            version=1,
        )
        first_version = await receiver.apply(rollout, transport.receiver_recv)
        return first_version, frozen_before

    first_version, frozen_before = asyncio.run(_run())

    assert first_version == 1
    assert rollout.action_expert.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        rollout.action_expert.weight,
        model.action_expert.weight.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        rollout.value_head.weight,
        model.value_head.weight.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        rollout.embedding.weight,
        frozen_before["embedding.weight"],
    )
    torch.testing.assert_close(
        rollout.vlm_expert.weight,
        frozen_before["vlm_expert.weight"],
    )
    torch.testing.assert_close(
        rollout.vlm_expert.bias,
        frozen_before["vlm_expert.bias"],
    )


def test_checkpoint_round_trip_with_frozen_fsdp(fsdp_single_rank, tmp_path):
    model, wrapped, strategy = _wrapped_tiny_actor()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapped.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    tokens = torch.randint(0, 16, (2, 8), device="cuda")
    features = torch.randn(2, 8, 4, device="cuda", dtype=torch.bfloat16)
    loss = wrapped(tokens, features).float().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    frozen_embedding_before = model.embedding.weight.detach().clone()
    frozen_vlm_before = model.vlm_expert.weight.detach().clone()
    action_before = model.action_expert.weight.detach().clone()
    value_before = model.value_head.weight.detach().clone()
    state_dict_before = strategy.get_model_state_dict(
        wrapped,
        cpu_offload=True,
        full_state_dict=True,
    )

    FSDPStrategy.save_checkpoint(
        wrapped,
        optimizer,
        scheduler,
        save_path=str(tmp_path),
    )

    model2, wrapped2, strategy2 = _wrapped_tiny_actor()
    optimizer2 = torch.optim.AdamW(
        [parameter for parameter in wrapped2.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)

    FSDPStrategy.load_checkpoint(
        wrapped2,
        optimizer2,
        scheduler2,
        load_path=str(tmp_path),
    )

    state_dict_after = strategy2.get_model_state_dict(
        wrapped2,
        cpu_offload=True,
        full_state_dict=True,
    )
    assert set(state_dict_before.keys()) == set(state_dict_after.keys())
    torch.testing.assert_close(model2.embedding.weight, frozen_embedding_before)
    torch.testing.assert_close(model2.vlm_expert.weight, frozen_vlm_before)
    torch.testing.assert_close(model2.action_expert.weight, action_before)
    torch.testing.assert_close(model2.value_head.weight, value_before)
    assert len(optimizer2.state) > 0
    assert scheduler2.last_epoch == scheduler.last_epoch
    assert scheduler2.last_epoch == 1
