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

from rlinf.algorithms.losses import (
    compute_decoupled_ppo_actor_critic_loss,
    compute_decoupled_ppo_actor_loss,
    compute_ppo_actor_loss,
)
from rlinf.hybrid_engines.fsdp import FSDP, ShardingStrategy
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy import fsdp as fsdp_module
from rlinf.hybrid_engines.fsdp.strategy.fsdp import (
    FSDPStrategy,
    _resolve_ignored_states,
)
from rlinf.hybrid_engines.fsdp.utils import create_device_mesh
from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.scheduler import Worker
from rlinf.utils.utils import (
    collect_param_names_need_sync,
    warmup_optimizer_state,
)


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


class _TinyAutoWrappedPPOModel(nn.Module):
    """Tiny PPO actor whose value head matches the real FSDP auto-wrap policy.

    The real ``ValueHead`` class is auto-wrapped as an independent FSDP child
    (``rlinf/hybrid_engines/fsdp/utils.py``), which is required to reproduce
    the critic-warmup writeback regression: during critic-only warmup the root
    FSDP handle (action expert) receives no gradient, and FSDP never restores
    its original-parameter views before the next forward.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.vlm_expert = nn.Linear(4, 4)
        self.action_expert = nn.Linear(4, 4)
        self.value_head = ValueHead(
            input_dim=4,
            hidden_sizes=(4,),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False
        for parameter in self.vlm_expert.parameters():
            parameter.requires_grad = False

    def forward(self, tokens, features):
        prefix = self.embedding(tokens) + features
        prefix = self.vlm_expert(prefix)
        return {
            "logprobs": self.action_expert(prefix),
            "values": self.value_head(prefix.detach()),
        }


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


def _wrapped_tiny_auto_wrapped_ppo_actor():
    torch.manual_seed(0)
    model = _TinyAutoWrappedPPOModel().to(dtype=torch.bfloat16)
    cfg = _strategy_cfg(ignore_frozen_params=True, use_orig_params=True)
    strategy = FSDPStrategy(cfg=cfg, world_size=1)
    wrapped = strategy.wrap_model(model, create_device_mesh(1))
    return model, wrapped, strategy, cfg


def _tiny_inputs():
    tokens = torch.randint(0, 16, (2, 8), device="cuda")
    features = torch.randn(2, 8, 4, device="cuda", dtype=torch.bfloat16)
    return tokens, features


def _manager_for_optimizer_test(cfg):
    if "optim" not in cfg:
        cfg.optim = OmegaConf.create(
            {
                "lr": 0.1,
                "value_lr": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
            }
        )
    manager = object.__new__(FSDPModelManager)
    manager._cfg = cfg
    manager._logger = logging.getLogger("test_fsdp_ignore_frozen_params")
    return manager


def _policy_loss(wrapped, tokens, features, critic_warmup):
    """Build the real decoupled actor+critic loss for the tiny PPO actor."""
    output = wrapped(tokens, features)
    logprobs = output["logprobs"].float()
    values = output["values"].float()
    loss_mask = torch.ones_like(values, dtype=torch.bool)
    advantages = (
        torch.zeros_like(logprobs)
        if critic_warmup
        else torch.randn_like(logprobs) * 0.01
    )
    loss, _ = compute_decoupled_ppo_actor_critic_loss(
        logprobs=logprobs,
        old_logprobs=logprobs.detach(),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        advantages=advantages,
        values=values,
        returns=torch.zeros_like(values),
        prev_values=values.detach(),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=loss_mask,
        critic_warmup=critic_warmup,
    )
    return loss


def _run_critic_step(wrapped, optimizer, tokens, features):
    optimizer.zero_grad(set_to_none=True)
    loss = _policy_loss(wrapped, tokens, features, critic_warmup=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_mixed_trainability_fsdp_forward_backward(fsdp_single_rank):
    model, wrapped, _ = _wrapped_tiny_actor()

    tokens, features = _tiny_inputs()

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
    model, wrapped, _, cfg = _wrapped_tiny_auto_wrapped_ppo_actor()
    manager = _manager_for_optimizer_test(cfg)

    warmup_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=True,
    )
    assert model.action_expert.weight.requires_grad
    assert not _optimizer_contains(warmup_optimizer, model.action_expert.weight)
    assert _optimizer_contains(warmup_optimizer, model.value_head.mlp[0].weight)
    assert not model.embedding.weight.requires_grad
    assert not model.vlm_expert.weight.requires_grad

    regular_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=False,
    )
    assert model.action_expert.weight.requires_grad
    assert _optimizer_contains(regular_optimizer, model.action_expert.weight)
    assert _optimizer_contains(regular_optimizer, model.value_head.mlp[0].weight)


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


def test_critic_warmup_actor_losses_keep_graph_connected():
    logprobs = torch.randn(2, 8, 4, requires_grad=True)
    old_logprobs = torch.randn(2, 8, 4)
    advantages = torch.randn(2, 8, 4)
    loss_mask = torch.ones(2, 8, 4, dtype=torch.bool)

    decoupled_loss, _ = compute_decoupled_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        advantages=advantages,
        loss_mask=loss_mask,
        critic_warmup=True,
    )
    assert float(decoupled_loss) == 0.0
    assert decoupled_loss.requires_grad
    assert decoupled_loss.grad_fn is not None

    ppo_loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        advantages=advantages,
        loss_mask=loss_mask,
        critic_warmup=True,
    )
    assert float(ppo_loss) == 0.0
    assert ppo_loss.requires_grad
    assert ppo_loss.grad_fn is not None


def test_warmup_optimizer_state_ignores_leftover_gradients():
    parameter = nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.randn(4, 4)
    original_value = parameter.detach().clone()
    original_grad = parameter.grad.detach().clone()

    warmup_optimizer_state(optimizer)

    torch.testing.assert_close(parameter, original_value)
    torch.testing.assert_close(parameter.grad, original_grad)
    state = optimizer.state[parameter]
    torch.testing.assert_close(
        state["exp_avg"],
        torch.zeros_like(state["exp_avg"]),
    )
    torch.testing.assert_close(
        state["exp_avg_sq"],
        torch.zeros_like(state["exp_avg_sq"]),
    )
    assert state["step"] == 0


def test_checkpoint_paths_must_be_absolute():
    manager = object.__new__(FSDPModelManager)
    with pytest.raises(ValueError, match="must be absolute"):
        manager.save_checkpoint("relative/checkpoint")

    with pytest.raises(ValueError, match="must be absolute"):
        FSDPStrategy.save_checkpoint(
            model=None,
            optimizers=None,
            lr_schedulers=None,
            save_path="relative/checkpoint",
        )

    with pytest.raises(ValueError, match="must be absolute"):
        FSDPStrategy.load_checkpoint(
            model=None,
            optimizers=None,
            lr_schedulers=None,
            load_path="relative/checkpoint",
        )


def test_critic_warmup_transition_preserves_fsdp_parameter_views(
    fsdp_single_rank,
):
    model, wrapped, strategy, cfg = _wrapped_tiny_auto_wrapped_ppo_actor()
    assert any(
        module is not wrapped and isinstance(module, FSDP)
        for module in wrapped.modules()
    )
    manager = _manager_for_optimizer_test(cfg)
    warmup_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=True,
    )
    assert _optimizer_contains(warmup_optimizer, model.value_head.mlp[0].weight)
    assert not _optimizer_contains(warmup_optimizer, model.action_expert.weight)

    tokens, features = _tiny_inputs()
    action_before = model.action_expert.weight.detach().clone()
    value_before = model.value_head.mlp[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()

    # Two full critic-only warmup steps. The actor term stays graph-connected
    # with a zero scale (matching the fixed losses), so the root FSDP handle
    # participates in backward with zero gradients and its parameter views
    # survive into the next forward.
    for _ in range(2):
        _run_critic_step(wrapped, warmup_optimizer, tokens, features)
    peak_warmup = torch.cuda.max_memory_allocated()

    assert isinstance(model.action_expert.weight, nn.Parameter)
    torch.testing.assert_close(model.action_expert.weight, action_before)
    assert not torch.equal(model.value_head.mlp[0].weight, value_before)
    assert model.action_expert.weight.grad is not None
    assert torch.count_nonzero(model.action_expert.weight.grad) == 0

    # Formal optimizer rebuild: action expert and value head, without any
    # requires_grad mutation.
    regular_optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=False,
    )
    peak_rebuild = torch.cuda.max_memory_allocated()
    assert _optimizer_contains(regular_optimizer, model.action_expert.weight)
    assert _optimizer_contains(regular_optimizer, model.value_head.mlp[0].weight)

    # Adam moments must be initialized from zeros, not from the real gradients
    # still attached to the value head after the last warmup step.
    for group in regular_optimizer.param_groups:
        for parameter in group["params"]:
            state = regular_optimizer.state[parameter]
            assert torch.count_nonzero(state["exp_avg"]) == 0
            assert torch.count_nonzero(state["exp_avg_sq"]) == 0

    # First normal PPO step (third forward) must update the action expert.
    regular_optimizer.zero_grad(set_to_none=True)
    loss = _policy_loss(wrapped, tokens, features, critic_warmup=False)
    loss.backward()
    regular_optimizer.step()
    assert (
        torch.count_nonzero(
            model.action_expert.weight.to(torch.float32)
            - action_before.to(torch.float32)
        )
        > 0
    )

    # One weight-sync round trip after the first PPO step.
    state_dict = strategy.get_model_state_dict(
        wrapped,
        cpu_offload=False,
        full_state_dict=True,
    )
    rollout = _TinyAutoWrappedPPOModel().to(
        dtype=torch.bfloat16,
        device="cuda",
    )
    sender = _patch_syncer(init_sync_enabled=True)
    receiver = _patch_syncer(init_sync_enabled=True)
    transport = _Transport()
    # Production collects sync names before FSDP wrapping; a fresh unwrapped
    # model yields the same clean names without the FSDP ``_fsdp_wrapped_module``
    # prefixes introduced on wrapped submodules.
    sync_names = collect_param_names_need_sync(_TinyAutoWrappedPPOModel())

    async def _sync_once():
        await asyncio.gather(
            sender.init_sender(
                state_dict,
                sync_names,
                transport.sender_send,
                transport.sender_recv,
            ),
            receiver.init_receiver(
                rollout.state_dict(),
                transport.receiver_recv,
                transport.receiver_send,
            ),
        )
        await sender.sync(
            state_dict,
            transport.sender_send,
            version=1,
        )
        return await receiver.apply(rollout, transport.receiver_recv)

    assert asyncio.run(_sync_once()) == 1
    torch.testing.assert_close(
        rollout.action_expert.weight,
        model.action_expert.weight.to(torch.bfloat16),
    )

    # Record per-phase peak memory for the full warmup -> PPO transition and
    # keep it inside a sane budget for the tiny model. This catches gross
    # regressions such as per-step gradient copies, not the 0.81 GiB
    # production-scale duplication (guarded by the optimizer-step test).
    peak_ppo = torch.cuda.max_memory_allocated()
    assert peak_warmup > 0
    assert peak_rebuild > 0
    assert peak_ppo > 0
    assert peak_ppo <= 1024**3


def test_optimizer_step_rebuilds_formal_optimizer_after_warmup(fsdp_single_rank):
    model, wrapped, strategy, cfg = _wrapped_tiny_auto_wrapped_ppo_actor()
    manager = _manager_for_optimizer_test(cfg)
    cfg.optim.total_training_steps = 10
    cfg.optim.clip_grad = 1.0
    manager._strategy = strategy
    manager.model = wrapped
    manager.optimizer = manager.build_optimizer(
        wrapped,
        enable_critic_warmup=True,
    )
    manager.critic_warmup_steps = 2
    manager.optimizer_steps = 0
    manager.grad_scaler = manager.build_grad_scaler(False)
    manager.lr_scheduler = manager.build_lr_scheduler(
        manager.optimizer,
        cfg.optim,
    )

    tokens, features = _tiny_inputs()
    action_before = model.action_expert.weight.detach().clone()

    # Two full critic-only warmup steps driven through the production entry
    # point (unscale -> clip -> step -> optional formal optimizer rebuild).
    for _ in range(2):
        manager.optimizer.zero_grad(set_to_none=True)
        loss = _policy_loss(wrapped, tokens, features, critic_warmup=True)
        loss.backward()
        manager.optimizer_step()

    # The second optimizer_step must have rebuilt the formal optimizer and
    # cleared leftover gradients instead of keeping duplicate zero-gradient
    # copies alive (the 0.81 GiB production-scale transient).
    assert manager.critic_warmup_steps == 0
    assert _optimizer_contains(manager.optimizer, model.action_expert.weight)
    assert _optimizer_contains(manager.optimizer, model.value_head.mlp[0].weight)
    torch.testing.assert_close(model.action_expert.weight, action_before)
    assert all(
        parameter.grad is None
        for parameter in wrapped.parameters()
        if parameter.requires_grad
    )
    for group in manager.optimizer.param_groups:
        for parameter in group["params"]:
            state = manager.optimizer.state[parameter]
            assert torch.count_nonzero(state["exp_avg"]) == 0
            assert torch.count_nonzero(state["exp_avg_sq"]) == 0

    # First normal PPO step through the same entry point updates the action
    # expert, and a subsequent forward keeps the FSDP views valid.
    manager.optimizer.zero_grad(set_to_none=True)
    loss = _policy_loss(wrapped, tokens, features, critic_warmup=False)
    loss.backward()
    manager.optimizer_step()
    assert (
        torch.count_nonzero(
            model.action_expert.weight.to(torch.float32)
            - action_before.to(torch.float32)
        )
        > 0
    )
    output = wrapped(tokens, features)
    assert output["logprobs"].shape == (2, 8, 4)
