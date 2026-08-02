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

"""Ray actor worker wrapping the residual HIL-RLPD learner."""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from collections import deque

import torch

from rlinf.algorithms.residual_hil_rlpd.learner import ResidualHilRLPDLearner
from rlinf.algorithms.residual_hil_rlpd.messages import (
    MESSAGE_TYPE_SCALE_ACK,
    MESSAGE_TYPE_TRANSITION,
    ScaleAckTracker,
    build_scale_message,
    validate_message,
)
from rlinf.algorithms.residual_hil_rlpd.transition import (
    ResidualChunkTransition,
)
from rlinf.models.embodiment.residual_dobot_policy import (
    ResidualDobotActor,
    ResidualDobotCritic,
)
from rlinf.scheduler import Worker


class AsyncResidualHilRLPDWorker(Worker):
    """Learner worker for the residual HIL-RLPD stack.

    Messages arriving on the actor channel are dispatched by type:
    ``residual_transition`` payloads feed the learner; trajectory payloads
    from the legacy PPO path are ignored.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._pending: deque[ResidualChunkTransition] = deque()
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._learner: ResidualHilRLPDLearner | None = None
        self._last_metrics: dict[str, float] = {}
        self._malformed_dropped = 0
        self.weight_syncer = None
        self.param_names_need_sync: list[str] = []
        self._run_id = str(cfg.get("run_id", ""))
        self._scale_ack_tracker = ScaleAckTracker()

    def init_worker(self):
        rlpd_cfg = self.cfg.algorithm.residual_hil_rlpd
        device = (
            f"{Worker.torch_device_type}:{int(os.environ.get('LOCAL_RANK', 0))}"
            if Worker.torch_device_type == "cuda"
            else "cpu"
        )
        from rlinf.algorithms.residual_hil_rlpd.config import build_residual_codec

        codec = build_residual_codec(self.cfg)
        from rlinf.algorithms.residual_hil_rlpd.fingerprint import (
            compute_base_fingerprint,
        )

        base_fingerprint = compute_base_fingerprint(
            str(self.cfg.actor.model.get("model_path", "")),
            norm_stats_path=str(
                self.cfg.actor.model.get("openpi_data", {}).get(
                    "norm_stats_path", None
                )
            ),
            codec=codec,
        )
        self._learner = ResidualHilRLPDLearner(
            actor=ResidualDobotActor(),
            critic=ResidualDobotCritic(num_q_heads=int(rlpd_cfg.num_q_heads)),
            codec=codec,
            gamma=float(self.cfg.algorithm.gamma),
            batch_size=int(self.cfg.actor.global_batch_size),
            utd_ratio=float(rlpd_cfg.utd_ratio),
            max_update_backlog=float(rlpd_cfg.max_update_backlog),
            base_only_collect_steps=int(rlpd_cfg.base_only_collect_steps),
            critic_only_updates=int(rlpd_cfg.critic_only_updates),
            residual_scale_ramp_updates=int(rlpd_cfg.residual_scale_ramp_updates),
            min_demo_size=int(rlpd_cfg.min_demo_size),
            max_online_transitions=int(rlpd_cfg.max_online_transitions),
            max_demo_transitions=int(rlpd_cfg.max_demo_transitions),
            max_online_bytes=float(rlpd_cfg.max_online_bytes),
            max_demo_bytes=float(rlpd_cfg.max_demo_bytes),
            memory_high_watermark=float(rlpd_cfg.memory_high_watermark),
            demo_ratio=float(rlpd_cfg.demo_ratio),
            policy_lag_warn_threshold=int(rlpd_cfg.policy_lag_warn_threshold),
            policy_lag_reject_threshold=int(rlpd_cfg.policy_lag_reject_threshold),
            gripper_enable_after_updates=int(
                rlpd_cfg.gripper_enable_after_updates
            ),
            num_q_sample=int(rlpd_cfg.num_q_sample),
            alpha_arm_init=float(rlpd_cfg.alpha_arm_init),
            alpha_gripper_init=float(rlpd_cfg.alpha_gripper_init),
            device=device,
            base_fingerprint=base_fingerprint,
        )
        self.param_names_need_sync = [
            name for name, _ in self._learner.actor.named_parameters()
        ]
        return True

    def recv_rollout_trajectories(self, input_channel):
        """Drain the actor channel and dispatch transition messages."""

        def _drain_loop():
            while not self._stop_flag.is_set():
                message = input_channel.get()
                if message is None:
                    continue
                if (
                    isinstance(message, dict)
                    and message.get("type") == MESSAGE_TYPE_TRANSITION
                ):
                    try:
                        validate_message(
                            message, MESSAGE_TYPE_TRANSITION, run_id=self._run_id
                        )
                    except ValueError:
                        # Strict schema: drop malformed messages, keep training.
                        with self._lock:
                            self._malformed_dropped = (
                                getattr(self, "_malformed_dropped", 0) + 1
                            )
                        continue
                    with self._lock:
                        self._pending.append(message["transition"])

        thread = threading.Thread(target=_drain_loop, daemon=True)
        thread.start()
        return True

    def run_training(self):
        """One training tick: ingest pending transitions and run RLPD updates.

        Returns the metrics dict.  ``AsyncEmbodiedRunner`` expects
        ``actor.run_training()`` to return a metrics dict (an empty dict means
        no training happened and the runner skips the step); the previous
        ``(True, metrics)`` tuple crashed ``_aggregate_numeric_metrics`` with
        ``'tuple' object has no attribute 'items'``.
        """
        if self._learner is None:
            return {}
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for transition in pending:
            self._learner.add_transition(transition)
        metrics: dict[str, float] = {
            "waiting_for_demo": float(self._learner.buffer.waiting_for_demo()),
            "online_size": float(self._learner.buffer.sizes()["online"]),
            "demo_size": float(self._learner.buffer.sizes()["demo"]),
            "pending_transitions": float(len(self._pending)),
            "malformed_message_dropped": float(
                getattr(self, "_malformed_dropped", 0)
            ),
            "residual_scale": float(self._learner.residual_scale()),
            "policy_lag_rejected": float(self._learner.policy_lag_rejected),
            "buffer_bytes": float(
                self._learner.buffer.memory_usage()["online_bytes"]
                + self._learner.buffer.memory_usage()["demo_bytes"]
            ),
            "over_memory_watermark": float(
                self._learner.buffer.over_memory_watermark()
            ),
        }
        updates = 0
        while self._learner.can_update() and updates < 20:
            metrics.update(self._learner.update())
            updates += 1
        metrics["updates"] = float(updates)
        self._last_metrics = metrics
        # AsyncEmbodiedRunner skips the step when the metrics dict is empty;
        # without this the global step counter spins while no training data
        # exists (e.g. before the operator presses 'y' to start the first
        # episode), burning weight syncs and checkpoints.
        return metrics if updates > 0 else {}

    def get_policy_version(self):
        return int(self._learner.update_counter) if self._learner else 0

    def get_residual_scale(self):
        return float(self._learner.residual_scale()) if self._learner else 0.0

    def _build_weight_syncer(self) -> None:
        """Build the actor's patch syncer from the standard config factory.

        The rollout receiver builds its syncer with ``WeightSyncer.create`` on
        the same ``weight_syncer`` config block, where ``transport_device``
        defaults to ``Worker.torch_device_type``.  The previous hand-rolled
        construction read top-level config keys that are not set in the YAML
        (the values live under ``weight_syncer.patch.*``) and therefore
        defaulted the transport to CPU, while the receiver expected accelerator
        tensors; the init-sync buckets then crashed on the receiver in
        ``tensors_record_stream`` with a device-type mismatch.  Using the same
        factory keeps both sides on a single truth source (P2-8).
        """
        if getattr(self, "weight_syncer", None) is not None:
            return
        wcfg = self.cfg.get("weight_syncer", None)
        if wcfg is None:
            raise ValueError(
                "actor.weight_syncer config must be provided for "
                "residual HIL-RLPD weight sync"
            )
        from rlinf.hybrid_engines.weight_syncer import WeightSyncer

        self.weight_syncer = WeightSyncer.create(wcfg)
        self._sync_weight_comm_options = self.weight_syncer.comm_options

    async def sync_model_to_rollout(self):
        """Send residual actor weight patches to the rollout worker."""
        if self._learner is None:
            return None
        self._build_weight_syncer()
        # Keep the source state dict on the accelerator: the patch syncer's
        # CPU-snapshot path requires accelerator source tensors, and the
        # init-sync buckets must land on the same device the rollout receiver
        # expects (Worker.torch_device_type).
        state_dict = {
            name: tensor.detach().float()
            for name, tensor in self._learner.actor.state_dict().items()
        }

        async def send_func(data):
            if not getattr(self, "_is_weight_sender", True):
                return
            await self.broadcast(
                data,
                groups=[
                    (getattr(self, "_group_name", self.cfg.actor.group_name), 0),
                    (
                        getattr(
                            self,
                            "_rollout_group_name",
                            self.cfg.rollout.group_name,
                        ),
                        getattr(self, "_rollout_all_ranks", 0),
                    ),
                ],
                src=(getattr(self, "_group_name", self.cfg.actor.group_name), 0),
                async_op=True,
                options=getattr(self, "_sync_weight_comm_options", None),
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=getattr(
                    self, "_rollout_group_name", self.cfg.rollout.group_name
                ),
                src_rank=0,
                async_op=True,
                options=getattr(self, "_sync_weight_comm_options", None),
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=getattr(self, "_is_weight_sender", True),
            )
        try:
            await self.weight_syncer.sync(
                state_dict, send_func, version=self.get_policy_version()
            )
            self._learner.last_sync_ok = True
        except Exception:
            self._learner.last_sync_ok = False
            raise
        scale_message = build_scale_message(
            self.get_residual_scale(),
            self.get_policy_version(),
            run_id=self._run_id,
            sender_rank=int(getattr(self, "_rank", 0)),
            gripper_enabled=bool(self._learner.gripper_enabled()),
        )
        await self._broadcast_scale_message(scale_message)
        await self._wait_scale_ack(scale_message)
        return True

    async def _broadcast_scale_message(self, message) -> None:
        await self.broadcast(
            message,
            groups=[
                (getattr(self, "_group_name", self.cfg.actor.group_name), 0),
                (
                    getattr(self, "_rollout_group_name", self.cfg.rollout.group_name),
                    getattr(self, "_rollout_all_ranks", 0),
                ),
            ],
            src=(getattr(self, "_group_name", self.cfg.actor.group_name), 0),
            async_op=True,
            options=getattr(self, "_sync_weight_comm_options", None),
        ).async_wait()

    async def _wait_scale_ack(self, scale_message) -> None:
        """Bounded ACK wait with retry (P2-6).

        A lost ACK never changes scale semantics: the rollout only *applies*
        the scale when its own applied weight version matches, so this is a
        channel-health signal with a retry counter, not a safety decision.
        """
        tracker = getattr(self, "_scale_ack_tracker", None)
        if tracker is None:
            tracker = ScaleAckTracker()
            self._scale_ack_tracker = tracker
        tracker.expect_ack(int(scale_message["policy_version"]))
        if not hasattr(self, "_worker_address"):
            # In-process/test workers have no distributed address; skip the
            # ACK wait (the rollout still applies scale only on version match).
            return
        for _ in range(tracker.max_retries + 1):
            try:
                ack = await asyncio.wait_for(
                    self.recv(
                        src_group_name=getattr(
                            self,
                            "_rollout_group_name",
                            self.cfg.rollout.group_name,
                        ),
                        src_rank=0,
                        async_op=True,
                        options=getattr(
                            self, "_sync_weight_comm_options", None
                        ),
                    ).async_wait(),
                    timeout=tracker.timeout_s,
                )
                validate_message(
                    ack, MESSAGE_TYPE_SCALE_ACK, run_id=self._run_id
                )
                tracker.record_ack(int(ack["policy_version"]))
                self._last_metrics["scale_ack_ok"] = 1.0
                return
            except Exception as exc:  # timeout / malformed / comms error
                retry = tracker.on_timeout()
                self._last_metrics["scale_ack_ok"] = 0.0
                self._logger.warning(
                    "[ResidualHIL] scale ACK not received (%s); "
                    "retries=%d",
                    type(exc).__name__,
                    tracker.retries,
                )
                if not retry:
                    return
                await self._broadcast_scale_message(scale_message)

    def save_checkpoint(self, save_path: str, step: int = 0) -> bool:
        if self._learner is None:
            return False
        from rlinf.utils.checkpoint_utils import (
            verify_checkpoint_files,
            write_checkpoint_manifest,
        )

        # P2-7: atomic commit — write to a sibling temp dir, verify, then
        # rename into place.  ``COMPLETED`` is written by the Runner *after*
        # verification; a crash before then leaves only the temp dir, which is
        # never picked up by resume (no COMPLETED marker).
        save_path = os.path.abspath(save_path)
        if os.path.isdir(save_path) and os.listdir(save_path):
            raise RuntimeError(
                f"refusing to overwrite non-empty checkpoint dir: {save_path}"
            )
        tmp_path = f"{save_path}.tmp"
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)
        try:
            learner_state = self._learner.state_dict()
            residual_path = os.path.join(tmp_path, "residual_learner.pt")
            torch.save(
                {"learner": learner_state, "step": step},
                residual_path,
            )
            full_weights_dir = os.path.join(tmp_path, "model_state_dict")
            os.makedirs(full_weights_dir, exist_ok=True)
            full_weights_path = os.path.join(
                full_weights_dir, "full_weights.pt"
            )
            torch.save(learner_state, full_weights_path)
            dcp_dir = os.path.join(tmp_path, "dcp_checkpoint")
            os.makedirs(dcp_dir, exist_ok=True)
            metadata_path = os.path.join(dcp_dir, ".metadata")
            with open(metadata_path, "w") as f:
                f.write(f"global_step={step}\n")
            distcp_path = os.path.join(dcp_dir, "__0_0.distcp")
            with open(distcp_path, "wb") as f:
                f.write(b"residual-learner-placeholder")
            verify_checkpoint_files(tmp_path, require_full_weights=True)
            write_checkpoint_manifest(
                tmp_path,
                step=step,
                files={
                    "residual_learner.pt": residual_path,
                    "model_state_dict/full_weights.pt": full_weights_path,
                    "dcp_checkpoint/.metadata": metadata_path,
                    "dcp_checkpoint/__0_0.distcp": distcp_path,
                },
                extra={
                    "replay_transitions": dict(self._learner.buffer.sizes()),
                },
            )
            # The Runner pre-creates an empty target dir; replace it atomically.
            if os.path.isdir(save_path):
                os.rmdir(save_path)
            os.rename(tmp_path, save_path)
        except Exception:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        return True

    def load_checkpoint(self, load_path: str) -> bool:
        if self._learner is None:
            return False
        from rlinf.utils.checkpoint_utils import verify_checkpoint_manifest

        if not os.path.exists(os.path.join(load_path, "COMPLETED")):
            raise RuntimeError(
                f"resume_dir {load_path} is missing the COMPLETED marker"
            )
        manifest = verify_checkpoint_manifest(
            load_path, reject_unknown_files=True
        )
        state = torch.load(
            os.path.join(load_path, "residual_learner.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self._learner.load_state_dict(state["learner"])
        expected_replay = (manifest.get("extra") or {}).get(
            "replay_transitions", None
        )
        if expected_replay is not None:
            actual = self._learner.buffer.sizes()
            if actual != dict(expected_replay):
                raise RuntimeError(
                    "checkpoint replay size mismatch after resume: "
                    f"expected {expected_replay}, got {actual}"
                )
        return True

    def stop(self):
        self._stop_flag.set()
        return True
