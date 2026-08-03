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

"""In-process tests for the residual P0/P1 integration pieces."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.realworld.realworld_env import _stack_residual_feedback
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_residual_hil_rlpd_policy_worker import (
    AsyncResidualHilRLPDWorker,
)


def _worker_cfg():
    return OmegaConf.create(
        {
            "algorithm": {
                "loss_type": "residual_hil_rlpd",
                "gamma": 0.99,
                "residual_hil_rlpd": {
                    "translation_data_limit_m": [0.1, 0.1, 0.1],
                    "rotation_data_limit_deg": [30.0, 30.0, 30.0],
                    "demo_ratio": 0.5,
                    "num_q_heads": 4,
                    "utd_ratio": 1.0,
                    "max_update_backlog": 500,
                    "base_only_collect_steps": 0,
                    "critic_only_updates": 0,
                    "residual_scale_ramp_updates": 50,
                    "num_q_sample": 2,
                    "alpha_arm_init": 0.1,
                    "alpha_gripper_init": 0.05,
                    "min_demo_size": 1,
                    "max_online_transitions": 100,
                    "max_demo_transitions": 100,
                    "max_online_bytes": 1e9,
                    "max_demo_bytes": 1e9,
                    "memory_high_watermark": 0.9,
                    "policy_lag_warn_threshold": 50,
                    "policy_lag_reject_threshold": 500,
                    "gripper_enable_after_updates": 100,
                    "gripper_max_switches_per_chunk": 2,
                    "gripper_min_hold_steps": 5,
                    "gripper_debounce_chunks": 2,
                    "safety_workspace_min_m": [-0.5, -0.5, 0.0],
                    "safety_workspace_max_m": [0.5, 0.5, 0.5],
                    "safety_max_translation_delta_m": 0.02,
                    "safety_max_rotation_delta_deg": 10.0,
                    "safety_hold_chunks": 3,
                },
            },
            "actor": {
                "group_name": "ActorGroup",
                "global_batch_size": 4,
                "model": {"base_policy": {"model_path": "base-fp"}},
            },
            "rollout": {"group_name": "RolloutGroup"},
        }
    )


def _make_worker():
    worker = object.__new__(AsyncResidualHilRLPDWorker)
    worker._is_ray_actor = False
    worker.cfg = _worker_cfg()
    worker._pending = deque()
    worker._lock = threading.Lock()
    worker._stop_flag = threading.Event()
    worker._learner = None
    worker._last_metrics = {}
    worker._run_id = ""
    return worker


class _QueueChannel:
    """Minimal channel stub with the same get()/put() surface the drain
    thread needs; avoids the full Channel/Cluster machinery for in-process
    tests."""

    def __init__(self):
        self._queue = queue.Queue()

    def put(self, item):
        self._queue.put(item)

    def get(self):
        return self._queue.get()


class _FakeSyncer:
    def __init__(self):
        self.sender_initialized = lambda: False
        self.receiver_initialized = lambda: False
        self.calls = []

    async def init_sender(self, **kwargs):
        self.calls.append(("init_sender", kwargs))

    async def sync(self, state_dict, send_func, version):
        self.calls.append(("sync", version))

    async def init_receiver(self, **kwargs):
        self.calls.append(("init_receiver", kwargs))

    async def apply(self, model, recv_func):
        self.calls.append(("apply", model))
        return 42


class _FakeAwaitable:
    async def async_wait(self):
        return None


def test_stack_residual_feedback_pads_tail():
    steps = [
        {
            "executed_action": np.ones(8, dtype=np.float32),
            "action_command_accepted": True,
            "human_intervention": False,
            "handoff_hold": False,
            "gripper_bypass": True,
            "reward": 0.1,
            "termination": False,
            "truncation": False,
            "reward_label_valid": True,
        }
        for _ in range(3)
    ]

    feedback = _stack_residual_feedback(steps, chunk_size=10, action_dim=8)

    assert feedback["executed_actions"].shape == (10, 8)
    assert feedback["executed_action_mask"].sum() == 3
    assert feedback["gripper_bypass_mask"][:3].all()
    assert np.isclose(feedback["rewards"][0], 0.1)
    assert not feedback["executed_action_mask"][3:].any()


def _transition(
    episode_id: int,
    chunk_id: int,
    human: bool = False,
    base_fingerprint: str = "base-fp",
    policy_version: int = 0,
):
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.transition import (
        CHUNK_LEN,
        build_residual_chunk_transition,
    )

    codec = ResidualCodec()
    rng = np.random.default_rng(episode_id * 10 + chunk_id)
    nominal = np.stack(
        [
            np.concatenate([rng.uniform(-0.1, 0.1, size=3), [1.0, 0, 0, 0], [0.5]])
            for _ in range(CHUNK_LEN)
        ]
    ).astype(np.float32)
    u = rng.uniform(-0.3, 0.3, size=(CHUNK_LEN, 6)).astype(np.float32)
    commanded = np.zeros((CHUNK_LEN, 8), dtype=np.float32)
    for i in range(CHUNK_LEN):
        pose = codec.compose(
            torch.as_tensor(nominal[i, :7], dtype=torch.float32),
            torch.as_tensor(u[i], dtype=torch.float32),
        ).numpy()
        commanded[i] = np.concatenate([pose, [nominal[i, 7]]])
    executed = commanded.copy()
    human_mask = np.zeros(CHUNK_LEN, dtype=bool)
    if human:
        executed[5, 7] = 0.0
        human_mask[5] = True
    transition, valid, reason = build_residual_chunk_transition(
        curr_obs={
            "main_images": torch.zeros(1, 3, 16, 16),
            "prev_states": torch.zeros(1, 8),
        },
        next_obs={
            "main_images": torch.zeros(1, 3, 16, 16),
            "prev_states": torch.zeros(1, 8),
        },
        nominal_actions=nominal,
        next_nominal_actions=nominal,
        sampled_arm_residual=u,
        sampled_gripper_mode=np.zeros(CHUNK_LEN, dtype=np.int64),
        commanded_actions=commanded,
        executed_actions=executed,
        gripper_bypass_mask=np.zeros(CHUNK_LEN, dtype=bool),
        rewards=rng.uniform(0, 0.1, size=CHUNK_LEN).astype(np.float32),
        executed_action_mask=np.ones(CHUNK_LEN, dtype=bool),
        human_intervention_mask=human_mask,
        handoff_hold_mask=np.zeros(CHUNK_LEN, dtype=bool),
        terminations=np.zeros(CHUNK_LEN, dtype=bool),
        truncations=np.zeros(CHUNK_LEN, dtype=bool),
        reward_label_valid=True,
        codec=codec,
        source=1 if human else 0,
        policy_version=policy_version,
        base_fingerprint=base_fingerprint,
        episode_id=episode_id,
        chunk_id=chunk_id,
    )
    assert valid, reason
    return transition


def test_worker_in_process_train_sync_checkpoint(tmp_path):
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
        compute_base_fingerprint,
    )

    saved_device_type = getattr(Worker, "torch_device_type", None)
    saved_platform = getattr(Worker, "torch_platform", None)
    Worker.torch_device_type = "cpu"
    Worker.torch_platform = torch.cuda if torch.cuda.is_available() else torch
    worker = _make_worker()
    assert worker.init_worker()

    codec = ResidualCodec()
    expected_fingerprint = compute_base_fingerprint(
        str(worker.cfg.actor.model.get("model_path", "")),
        norm_stats_path=str(
            worker.cfg.actor.model.get("openpi_data", {}).get(
                "norm_stats_path", None
            )
        ),
        codec=codec,
    )
    worker._pending.append(
        _transition(0, 0, base_fingerprint=expected_fingerprint)
    )
    worker._pending.append(
        _transition(0, 1, human=True, base_fingerprint=expected_fingerprint)
    )
    metrics = worker.run_training()
    assert isinstance(metrics, dict)
    assert metrics["demo_size"] >= 1
    assert worker.get_policy_version() >= 0

    fake_syncer = _FakeSyncer()
    worker.weight_syncer = fake_syncer
    worker._is_weight_sender = True
    worker.broadcast = lambda *args, **kwargs: _FakeAwaitable()
    asyncio.run(worker.sync_model_to_rollout())
    assert fake_syncer.calls[0][0] == "init_sender"
    synced_state = fake_syncer.calls[0][1]["state_dict"]
    assert any("arm_head" in name or "gripper_head" in name for name in synced_state)
    assert fake_syncer.calls[1] == ("sync", worker.get_policy_version())

    save_dir = str(tmp_path / "ckpt")
    assert worker.save_checkpoint(save_dir, step=1)
    from rlinf.utils.checkpoint_utils import (
        verify_checkpoint_files,
        write_completed_marker,
    )

    verify_checkpoint_files(save_dir, require_full_weights=True)
    write_completed_marker(save_dir, step=1)
    assert (tmp_path / "ckpt" / "COMPLETED").exists()
    worker2 = _make_worker()
    assert worker2.init_worker()
    assert worker2.load_checkpoint(save_dir)
    for (n1, p1), (n2, p2) in zip(
        worker._learner.actor.state_dict().items(),
        worker2._learner.actor.state_dict().items(),
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    worker.stop()
    if saved_device_type is None:
        Worker.__dict__.pop("torch_device_type", None)
    else:
        Worker.torch_device_type = saved_device_type
    if saved_platform is None:
        Worker.__dict__.pop("torch_platform", None)
    else:
        Worker.torch_platform = saved_platform


def test_run_training_returns_metrics_dict_contract():
    """``AsyncEmbodiedRunner`` expects ``run_training`` to return a metrics
    dict; the previous ``(True, metrics)`` tuple crashed the runner's
    ``_aggregate_numeric_metrics``.  An uninitialized learner must report an
    empty dict so the runner skips the step."""
    worker = _make_worker()
    assert worker.run_training() == {}


def test_run_training_idle_buffer_returns_empty_metrics():
    """Real-machine regression: with no transitions collected yet (e.g. the
    operator has not pressed 'y' to start the first episode) the runner must
    skip steps instead of spinning the global step counter."""
    worker = _make_worker()
    assert worker.init_worker()
    assert worker.run_training() == {}
    worker.stop()


def test_base_fingerprint_is_path_independent(tmp_path):
    """Two nodes mirroring the same frozen checkpoint under different repo
    roots (actor on /home/tyz/... vs env on /home/zylab/...) must compute the
    same base fingerprint; otherwise every online transition is rejected with
    ``base_fingerprint_mismatch`` and training never starts."""
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
        compute_base_fingerprint,
    )

    root_a = tmp_path / "home" / "tyz" / "project" / "RLinf" / "checkpoints" / "base"
    root_b = (
        tmp_path / "home" / "zylab" / "project" / "RLinf" / "checkpoints" / "base"
    )
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    (root_a / "model.safetensors").write_bytes(b"same-weights-bytes")
    (root_b / "model.safetensors").write_bytes(b"same-weights-bytes")
    norm_a = root_a / "norm_stats.json"
    norm_b = root_b / "norm_stats.json"
    norm_a.write_text('{"mean": 1.0}')
    norm_b.write_text('{"mean": 1.0}')
    codec = ResidualCodec()

    fp_a = compute_base_fingerprint(
        str(root_a), norm_stats_path=str(norm_a), codec=codec
    )
    fp_b = compute_base_fingerprint(
        str(root_b), norm_stats_path=str(norm_b), codec=codec
    )
    assert fp_a == fp_b

    # Different weights content must still change the fingerprint.
    (root_b / "model.safetensors").write_bytes(b"different-weights-bytes")
    assert (
        compute_base_fingerprint(
            str(root_b), norm_stats_path=str(norm_b), codec=codec
        )
        != fp_a
    )


def test_gripper_residual_master_switch_disables_override():
    """``gripper_residual_enabled=False`` must keep the gripper on the frozen
    VLA nominal (KEEP) even after the update threshold is crossed, so resuming
    an old model can never re-enable gripper residual from the checkpoint."""
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.learner import (
        ResidualHilRLPDLearner,
    )
    from rlinf.models.embodiment.residual_dobot_policy import (
        ResidualDobotActor,
        ResidualDobotCritic,
    )

    learner = ResidualHilRLPDLearner(
        actor=ResidualDobotActor(hidden=8, image_channels=3),
        critic=ResidualDobotCritic(hidden=8, image_channels=3, num_q_heads=2),
        codec=ResidualCodec(),
        batch_size=2,
        min_demo_size=1,
        device="cpu",
        gripper_residual_enabled=False,
        gripper_enable_after_updates=0,
    )
    learner.update_counter = 10**6
    assert learner.gripper_enabled() is False

    # Enabled (default) still gates on the update threshold.
    learner2 = ResidualHilRLPDLearner(
        actor=ResidualDobotActor(hidden=8, image_channels=3),
        critic=ResidualDobotCritic(hidden=8, image_channels=3, num_q_heads=2),
        codec=ResidualCodec(),
        batch_size=2,
        min_demo_size=1,
        device="cpu",
        gripper_enable_after_updates=0,
    )
    assert learner2.gripper_enabled() is True


def test_residual_scale_parse_is_fail_closed():
    from rlinf.algorithms.residual_hil_rlpd.messages import build_scale_message
    from rlinf.workers.rollout.hf.residual_hil_rollout_worker import (
        ResidualHILRolloutWorker,
    )

    assert (
        ResidualHILRolloutWorker._parse_scale_message(
            build_scale_message(0.5, policy_version=0), applied_version=0
        )
        == 0.5
    )
    # Scale is bound to the applied weight version: mismatch -> fail closed.
    assert (
        ResidualHILRolloutWorker._parse_scale_message(
            build_scale_message(0.5, policy_version=1), applied_version=0
        )
        == 0.0
    )
    out_of_range = build_scale_message(1.5, policy_version=0)
    assert (
        ResidualHILRolloutWorker._parse_scale_message(
            out_of_range, applied_version=0
        )
        == 0.0
    )
    assert (
        ResidualHILRolloutWorker._parse_scale_message(
            {"type": "other"}, applied_version=0
        )
        == 0.0
    )
    assert (
        ResidualHILRolloutWorker._parse_scale_message(None, applied_version=0)
        == 0.0
    )


def test_initial_weight_sync_makes_rollout_equal_learner():
    if not torch.cuda.is_available():
        pytest.skip("initial weight sync requires an accelerator")
    Worker.torch_platform = torch.cuda

    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.learner import ResidualHilRLPDLearner
    from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer
    from rlinf.models.embodiment.residual_dobot_policy import (
        ResidualDobotActor,
        ResidualDobotCritic,
    )

    class _Transport:
        def __init__(self):
            self.a2b = asyncio.Queue()
            self.b2a = asyncio.Queue()

        async def sender_send(self, value):
            await self.a2b.put(value)

        async def sender_recv(self):
            return await self.b2a.get()

        async def receiver_send(self, value):
            await self.b2a.put(value)

        async def receiver_recv(self):
            return await self.a2b.get()

    codec = ResidualCodec()
    learner = ResidualHilRLPDLearner(
        actor=ResidualDobotActor(hidden=32, image_channels=3),
        critic=ResidualDobotCritic(hidden=32, image_channels=3, num_q_heads=2),
        codec=codec,
        batch_size=2,
        min_demo_size=1,
        device="cpu",
    )
    rollout_actor = ResidualDobotActor(hidden=32, image_channels=3)
    sender = PatchWeightSyncer(
        snapshot_device="cuda",
        transport_device="cuda",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
    )
    receiver = PatchWeightSyncer(
        snapshot_device="cuda",
        transport_device="cuda",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
    )
    transport = _Transport()
    learner.actor.to("cuda")
    rollout_actor.to("cuda")
    state = {
        name: value.detach().float()
        for name, value in learner.actor.state_dict().items()
    }

    async def _run():
        await asyncio.gather(
            sender.init_sender(
                state_dict=state,
                param_names_need_sync=list(state.keys()),
                send=transport.sender_send,
                recv=transport.sender_recv,
            ),
            receiver.init_receiver(
                state_dict=rollout_actor.state_dict(),
                recv=transport.receiver_recv,
                send=transport.receiver_send,
            ),
        )
        await sender.sync(state, transport.sender_send, version=1)
        await receiver.apply(rollout_actor, transport.receiver_recv)

    asyncio.run(_run())

    for name, target in learner.actor.state_dict().items():
        torch.testing.assert_close(rollout_actor.state_dict()[name], target)


def test_actor_syncer_matches_rollout_factory_transport_device():
    """P0 regression (real-machine startup crash).

    ``AsyncResidualHilRLPDWorker.sync_model_to_rollout`` used to hand-roll its
    patch syncer from top-level config keys that are not set in the YAML, which
    defaulted the sender transport to CPU while the rollout receiver's factory
    (``WeightSyncer.create``) defaulted it to ``Worker.torch_device_type``.
    The init-sync buckets were therefore shipped on CPU and the receiver
    crashed with ``RuntimeError: Tensor device type does not match the worker
    device type``.  The actor must build its syncer from the same factory and
    transport the state dict on the accelerator.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    saved_device_type = getattr(Worker, "torch_device_type", None)
    saved_platform = getattr(Worker, "torch_platform", None)
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda

    from rlinf.hybrid_engines.weight_syncer import WeightSyncer
    from rlinf.models.embodiment.residual_dobot_policy import ResidualDobotActor

    try:
        worker = object.__new__(AsyncResidualHilRLPDWorker)
        worker.weight_syncer = None
        worker.cfg = OmegaConf.create(
            {
                "weight_syncer": {
                    "type": "patch",
                    "patch": {
                        "snapshot_device": "cpu",
                        "delta_encoding": True,
                        "compression": "none",
                        "init_sync": {
                            "enabled": True,
                            "prefixes": None,
                            "bucket_size": 1 << 20,
                        },
                    },
                }
            }
        )
        worker._build_weight_syncer()
        sender = worker.weight_syncer
        # Rollout side builds from the same config block via the factory.
        receiver = WeightSyncer.create(worker.cfg.weight_syncer)

        assert sender.transport_device.type == Worker.torch_device_type
        assert receiver.transport_device.type == sender.transport_device.type

        torch.manual_seed(0)
        sender_model = ResidualDobotActor(hidden=32, image_channels=3).to("cuda")
        torch.manual_seed(1)
        receiver_model = ResidualDobotActor(hidden=32, image_channels=3).to("cuda")

        class _Transport:
            def __init__(self):
                self.a2b = asyncio.Queue()
                self.b2a = asyncio.Queue()

            async def sender_send(self, value):
                await self.a2b.put(value)

            async def sender_recv(self):
                return await self.b2a.get()

            async def receiver_send(self, value):
                await self.b2a.put(value)

            async def receiver_recv(self):
                return await self.a2b.get()

        state = {
            name: value.detach().float()
            for name, value in sender_model.state_dict().items()
        }
        transport = _Transport()

        async def _run():
            await asyncio.gather(
                sender.init_sender(
                    state_dict=state,
                    param_names_need_sync=list(state.keys()),
                    send=transport.sender_send,
                    recv=transport.sender_recv,
                ),
                receiver.init_receiver(
                    state_dict=receiver_model.state_dict(),
                    recv=transport.receiver_recv,
                    send=transport.receiver_send,
                ),
            )
            await sender.sync(state, transport.sender_send, version=3)
            await receiver.apply(receiver_model, transport.receiver_recv)

        asyncio.run(_run())

        for name, target in sender_model.state_dict().items():
            torch.testing.assert_close(receiver_model.state_dict()[name], target)
    finally:
        if saved_device_type is None:
            Worker.__dict__.pop("torch_device_type", None)
        else:
            Worker.torch_device_type = saved_device_type
        if saved_platform is None:
            Worker.__dict__.pop("torch_platform", None)
        else:
            Worker.torch_platform = saved_platform


def test_env_to_actor_transition_channel_round_trip():
    """P0 regression: the env worker's message envelope must survive a real
    Channel and be ingested by the actor worker (strict schema on)."""
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec
    from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
        compute_base_fingerprint,
    )
    from rlinf.algorithms.residual_hil_rlpd.messages import (
        build_transition_message,
    )
    channel = _QueueChannel()
    worker = _make_worker()
    assert worker.init_worker()
    worker.recv_rollout_trajectories(channel)

    expected_fingerprint = compute_base_fingerprint(
        str(worker.cfg.actor.model.get("model_path", "")),
        norm_stats_path=str(
            worker.cfg.actor.model.get("openpi_data", {}).get(
                "norm_stats_path", None
            )
        ),
        codec=ResidualCodec(),
    )
    transition = _transition(
        0, 0, human=True, base_fingerprint=expected_fingerprint
    )
    # Exactly the envelope the env worker builds before send_to().
    message = build_transition_message(
        transition,
        policy_version=0,
        run_id="",
        sender_rank=0,
    )
    channel.put(message)
    time.sleep(0.3)  # let the drain thread pick the message up
    metrics = worker.run_training()
    assert metrics["online_size"] == 1
    assert metrics["demo_size"] == 1
    assert worker._learner.buffer.sizes()["online"] == 1
    worker.stop()


def test_env_to_actor_rejects_legacy_envelope():
    """P0 regression: an envelope without schema_version/run_id must be
    dropped by the actor (strict validation active), never ingested."""
    channel = _QueueChannel()
    worker = _make_worker()
    assert worker.init_worker()
    worker.recv_rollout_trajectories(channel)
    channel.put(
        {
            "type": "residual_transition",
            "transition": _transition(0, 0),
        }
    )
    time.sleep(0.3)
    # The malformed message is dropped and the buffer stays empty, so the
    # runner must see an empty metrics dict (skip the step).
    metrics = worker.run_training()
    assert metrics == {}
    assert getattr(worker, "_malformed_dropped", 0) >= 1
    worker.stop()
