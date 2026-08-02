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

"""EnvWorker diagnostics tests for Dobot HG-DAgger."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
from rlinf.utils.utils import (
    apply_reward_label_validity_mask,
    validate_reward_label_validity_config,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_extracts_operator_reason_and_chunk_execution_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["operator_abort"]),
            "skipped_action_steps": 3,
            "executed_action_mask": torch.tensor([[True, False, False, False]]),
            "handoff_hold_mask": torch.tensor([[False, True, False, False]]),
        }
    )

    torch.testing.assert_close(
        metrics["episode_end/operator_abort"], torch.tensor([1.0])
    )
    torch.testing.assert_close(
        metrics["episode_end/skipped_action_steps"], torch.tensor([3.0])
    )
    torch.testing.assert_close(
        metrics["episode_end/executed_action_fraction"], torch.tensor([0.25])
    )
    torch.testing.assert_close(
        metrics["control/handoff_hold_fraction"], torch.tensor([0.25])
    )


def test_does_not_emit_false_operator_reason_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["timeout"]),
            "executed_action_mask": torch.ones((1, 4), dtype=torch.bool),
        }
    )

    assert not any(key.startswith("episode_end/operator_") for key in metrics)
    torch.testing.assert_close(
        metrics["episode_end/executed_action_fraction"], torch.tensor([1.0])
    )


def test_extracts_timeout_and_reward_label_validity_metrics():
    metrics = EnvWorker._extract_operator_metrics(
        {
            "termination_reason": np.array(["episode_timeout"]),
            "reward_label_valid": torch.tensor([True]),
        }
    )

    torch.testing.assert_close(
        metrics["episode_end/episode_timeout"], torch.tensor([1.0])
    )
    torch.testing.assert_close(
        metrics["rollout/reward_label_valid"], torch.tensor([1.0])
    )


def test_extracts_keyboard_failure_reason():
    metrics = EnvWorker._extract_operator_metrics(
        {"termination_reason": np.array(["keyboard_disconnected"])}
    )

    torch.testing.assert_close(
        metrics["episode_end/keyboard_disconnected"], torch.tensor([1.0])
    )


def test_extracts_controller_rejection_metrics_and_audit_code():
    infos = {
        "termination_reason": np.array(["controller_rejection"]),
        "executed_action_mask": torch.tensor([[True, False, False, False]]),
        "episode_id": torch.tensor([4]),
        "episode_step_ids": torch.tensor([[0, 1, -1, -1]]),
        "reward_label_valid": torch.tensor([False]),
    }

    metrics = EnvWorker._extract_operator_metrics(infos)
    audit_info = EnvWorker._extract_trajectory_audit_info(infos)

    torch.testing.assert_close(
        metrics["episode_end/controller_rejection"], torch.tensor([1.0])
    )
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([6]))
    torch.testing.assert_close(audit_info["reward_label_valid"], torch.tensor([False]))


def test_extracts_unsafe_model_handoff_metrics_and_audit_code():
    infos = {
        "termination_reason": np.array(["unsafe_model_handoff"]),
        "executed_action_mask": torch.tensor([[True, False]]),
        "episode_id": torch.tensor([5]),
        "episode_step_ids": torch.tensor([[10, -1]]),
    }

    metrics = EnvWorker._extract_operator_metrics(infos)
    audit_info = EnvWorker._extract_trajectory_audit_info(infos)

    torch.testing.assert_close(
        metrics["episode_end/unsafe_model_handoff"], torch.tensor([1.0])
    )
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([7]))


def test_extracts_trajectory_audit_info_from_final_info():
    audit_info = EnvWorker._extract_trajectory_audit_info(
        {
            "final_info": {
                "executed_action_mask": torch.tensor([[True, True, False, False]]),
                "termination_reason": np.array(["operator_abort"]),
                "episode_id": torch.tensor([9]),
                "episode_step_ids": torch.tensor([[20, 21, -1, -1]]),
                "handoff_hold_mask": torch.tensor([[False, True, False, False]]),
            }
        }
    )

    assert audit_info["executed_action_mask"].dtype == torch.bool
    torch.testing.assert_close(audit_info["termination_reason_code"], torch.tensor([2]))
    torch.testing.assert_close(audit_info["episode_id"], torch.tensor([9]))
    torch.testing.assert_close(
        audit_info["episode_step_ids"], torch.tensor([[20, 21, -1, -1]])
    )
    torch.testing.assert_close(
        audit_info["handoff_hold_mask"],
        torch.tensor([[False, True, False, False]]),
    )


def test_missing_handoff_hold_mask_defaults_to_false():
    audit_info = EnvWorker._extract_trajectory_audit_info(
        {
            "executed_action_mask": torch.tensor([[True, True]]),
            "episode_id": torch.tensor([1]),
            "episode_step_ids": torch.tensor([[0, 1]]),
        }
    )

    torch.testing.assert_close(
        audit_info["handoff_hold_mask"], torch.tensor([[False, False]])
    )


def test_handoff_hold_overrides_action_without_becoming_human_label():
    class RolloutRecorder:
        def __init__(self):
            self.actions = None
            self.override_flags = None
            self.human_flags = None

        def update_last_actions(self, actions, flags):
            self.actions = actions
            self.override_flags = flags

        def mark_last_step_with_intervene_flags(self, flags):
            self.human_flags = flags

    recorder = RolloutRecorder()
    worker = EnvWorker.__new__(EnvWorker)
    worker.rollout_results = [recorder]
    human_flags = torch.tensor([[True, False, False, False]])
    handoff_hold_mask = torch.tensor([[False, False, True, True]])
    override_actions = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    env_output = SimpleNamespace(
        intervene_actions=override_actions,
        intervene_flags=human_flags,
        env_infos={"handoff_hold_mask": handoff_hold_mask},
    )

    worker._apply_last_action_overrides(0, env_output)

    torch.testing.assert_close(recorder.actions, override_actions)
    torch.testing.assert_close(
        recorder.override_flags,
        torch.tensor([[True, False, True, True]]),
    )
    torch.testing.assert_close(recorder.human_flags, human_flags)


def test_missing_execution_metadata_does_not_create_audit_record():
    assert EnvWorker._extract_trajectory_audit_info({}) == {}


def test_terminal_padding_env_output_is_shape_compatible_and_non_executed():
    worker = EnvWorker.__new__(EnvWorker)
    worker.model_cfg = SimpleNamespace(num_action_chunks=4)
    worker.train_num_envs_per_stage = 1
    source = EnvOutput(
        obs={
            "states": torch.ones((1, 8)),
            "main_images": torch.zeros((1, 2, 2, 3)),
            "task_descriptions": ["test"],
        },
        rewards=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        dones=torch.tensor([[False, False, False, True]]),
        terminations=torch.tensor([[False, False, False, True]]),
        truncations=torch.zeros((1, 4), dtype=torch.bool),
        env_infos={"episode_id": torch.tensor([12])},
    )

    padded = worker._make_terminal_padding_env_output(source)

    assert padded.rewards.shape == (1, 4)
    assert not padded.rewards.any()
    assert not padded.dones.any()
    assert padded.env_infos["rollout_padding"] is True
    assert not padded.env_infos["executed_action_mask"].any()
    torch.testing.assert_close(
        padded.env_infos["episode_step_ids"],
        torch.full((1, 4), -1, dtype=torch.int64),
    )
    torch.testing.assert_close(padded.env_infos["episode_id"], torch.tensor([12]))


def test_rollout_input_marks_every_padding_batch_item():
    worker = EnvWorker.__new__(EnvWorker)
    worker.enable_rlt = False
    env_batch = {
        "obs": {
            "states": torch.ones((2, 8)),
            "task_descriptions": ["a", "b"],
        },
        "final_obs": None,
    }

    payload = worker._build_rollout_input_data(
        env_batch,
        rollout_padding=True,
    )

    torch.testing.assert_close(payload["rollout_padding"], torch.tensor([True, True]))


def test_rollout_padding_result_reuses_shapes_without_mutating_cache():
    source = RolloutResult(
        actions=torch.ones((1, 32)),
        prev_logprobs=torch.ones((1, 4, 8)),
        prev_values=torch.ones((1, 1)),
        bootstrap_values=torch.ones((1, 1)),
        intervene_flags=torch.ones((1, 4), dtype=torch.bool),
        forward_inputs={
            "action": torch.ones((1, 32)),
            "chains": torch.ones((1, 2, 4, 8)),
        },
        versions=torch.ones((1, 4, 8)),
    )
    spec = MultiStepRolloutWorker._capture_rollout_padding_spec(source)
    assert not torch.is_tensor(spec.actions)
    assert all(not torch.is_tensor(value) for value in spec.forward_inputs.values())

    padded = MultiStepRolloutWorker._build_padding_rollout_result(
        spec,
        final=False,
    )

    assert not padded.actions.any()
    assert not padded.intervene_flags.any()
    assert padded.bootstrap_values is None
    assert not padded.prev_logprobs.any()
    torch.testing.assert_close(
        padded.forward_inputs["action"],
        torch.zeros_like(source.forward_inputs["action"]),
    )
    torch.testing.assert_close(padded.versions, torch.full_like(source.versions, -1))
    assert source.actions.all()
    assert source.intervene_flags.all()

    final = MultiStepRolloutWorker._build_padding_rollout_result(
        spec,
        final=True,
    )
    assert not final.actions.any()
    assert final.forward_inputs == {}
    assert final.prev_logprobs is None


@pytest.mark.parametrize("collect_prev_infos", [False, True])
def test_rollout_result_versions_do_not_depend_on_logprobs(collect_prev_infos):
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.collect_prev_infos = collect_prev_infos
    worker.version = 7
    worker.get_bootstrap_values = lambda _final_obs: None

    actions = torch.ones((2, 32))
    prev_logprobs = torch.ones((2, 4, 8)) if collect_prev_infos else None
    result = {
        "prev_logprobs": prev_logprobs,
        "prev_values": torch.ones((2, 1)) if collect_prev_infos else None,
        "forward_inputs": {"action": actions.clone()},
        "expert_label_flag": False,
    }

    rollout_result = worker._build_rollout_result(actions, result)

    if collect_prev_infos:
        torch.testing.assert_close(rollout_result.prev_logprobs, prev_logprobs)
    else:
        assert rollout_result.prev_logprobs is None
    assert rollout_result.versions.shape == (2, 1)
    assert rollout_result.versions.dtype == torch.float32
    torch.testing.assert_close(
        rollout_result.versions,
        torch.full((2, 1), 7.0),
    )


def test_rollout_padding_detection_requires_the_whole_batch():
    assert MultiStepRolloutWorker._is_full_rollout_padding(
        {"rollout_padding": torch.tensor([True, True])}
    )
    assert not MultiStepRolloutWorker._is_full_rollout_padding(
        {"rollout_padding": torch.tensor([True, False])}
    )
    assert not MultiStepRolloutWorker._is_full_rollout_padding({})


def test_disabled_terminal_padding_does_not_allocate_cache_slots():
    assert (
        MultiStepRolloutWorker._new_padding_spec_slots(
            enabled=False,
            num_pipeline_stages=2,
        )
        is None
    )
    assert MultiStepRolloutWorker._new_padding_spec_slots(
        enabled=True,
        num_pipeline_stages=2,
    ) == [None, None]


def _make_rollout_worker_for_padding_loop(
    *,
    padding_enabled,
    payloads,
    capture_spec,
):
    class ImmediateResult:
        def __init__(self, value):
            self.value = value

        async def async_wait(self):
            return self.value

    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"group_name": "EnvGroup"}})
    worker.num_pipeline_stages = 1
    worker.n_train_chunk_steps = 1
    worker.train_terminal_padding_enabled = padding_enabled
    worker.train_batch_size = 1
    worker.collect_prev_infos = True
    worker.enable_opd = False
    worker.rlt_feature_model = None
    worker.model_cfg = SimpleNamespace(num_action_chunks=4)
    worker.version = 3
    worker.update_dagger_beta = lambda: None
    worker.get_bootstrap_values = lambda _final_obs: None
    worker._capture_rollout_padding_spec = capture_spec
    worker.recv_from = lambda **_kwargs: ImmediateResult(payloads.pop(0))
    sent = []
    worker.send_to = lambda **kwargs: sent.append(kwargs["data"])
    predictions = []

    def predict(_obs, **_kwargs):
        predictions.append(True)
        return (
            torch.ones((1, 32)),
            {
                "prev_logprobs": torch.ones((1, 4, 8)),
                "prev_values": torch.ones((1, 1)),
                "forward_inputs": {
                    "action": torch.ones((1, 32)),
                    "chains": torch.ones((1, 2, 4, 8)),
                },
                "expert_label_flag": False,
            },
        )

    worker._predict_rollout_actions = predict
    return worker, predictions, sent


def _run_generate_one_epoch_without_worker_timer(worker):
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__
    asyncio.run(generate_one_epoch(worker, None, None))


def _unwrap_worker_method(method):
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    return method


def test_disabled_terminal_padding_never_captures_rollout_specs():
    normal_payload = {
        "obs": {"states": torch.ones((1, 8))},
        "final_obs": None,
    }

    def fail_capture(_result):
        raise AssertionError("disabled terminal padding must not capture a spec")

    worker, predictions, sent = _make_rollout_worker_for_padding_loop(
        padding_enabled=False,
        payloads=[normal_payload, normal_payload],
        capture_spec=fail_capture,
    )

    _run_generate_one_epoch_without_worker_timer(worker)

    assert len(predictions) == 2
    assert len(sent) == 2


def test_enabled_single_env_padding_skips_terminal_policy_inference():
    normal_payload = {
        "obs": {"states": torch.ones((1, 8))},
        "final_obs": None,
    }
    padding_payload = {
        "obs": {"states": torch.ones((1, 8))},
        "final_obs": None,
        "rollout_padding": torch.tensor([True]),
    }
    worker, predictions, sent = _make_rollout_worker_for_padding_loop(
        padding_enabled=True,
        payloads=[normal_payload, padding_payload],
        capture_spec=MultiStepRolloutWorker._capture_rollout_padding_spec,
    )

    _run_generate_one_epoch_without_worker_timer(worker)

    assert len(predictions) == 1
    assert len(sent) == 2
    assert not sent[-1].actions.any()
    assert sent[-1].forward_inputs == {}


def test_train_single_env_terminal_padding_stops_real_env_calls():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "max_episode_steps": 12,
                }
            },
            "actor": {"model": {"num_action_chunks": 4}},
            "rollout": {
                "group_name": "RolloutGroup",
                "collect_transitions": False,
            },
            "algorithm": {},
        }
    )
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.train_batch_size = 1
    worker.n_train_chunk_steps = 3
    worker.rollout_epoch = 1
    worker.train_terminal_padding_enabled = True
    worker._terminal_end_monotonic = [None]
    worker._terminal_to_reset_latency_s = []
    worker._prefetched_train_bootstrap = None
    worker.model_cfg = worker.cfg.actor.model
    worker.enable_online_lerobot = False
    worker.use_training_pipeline = False
    worker.collect_transitions = False
    worker.collect_prev_infos = True
    worker.enable_rlt = False
    worker.reward_mode = "none"
    worker.history_reward_assign = False
    worker.env_decoupled_mode = False
    worker._bootstrap_and_send_train = lambda _channel: [
        EnvOutput(
            obs={"states": torch.ones((1, 8))},
            rewards=torch.zeros((1, 4)),
            dones=torch.zeros((1, 4), dtype=torch.bool),
            terminations=torch.zeros((1, 4), dtype=torch.bool),
            truncations=torch.zeros((1, 4), dtype=torch.bool),
        )
    ]
    rollout_result = RolloutResult(
        actions=torch.ones((1, 32)),
        prev_logprobs=torch.ones((1, 4, 8)),
        prev_values=torch.ones((1, 1)),
        forward_inputs={"action": torch.ones((1, 32))},
        versions=torch.ones((1, 4, 8)),
    )
    worker.recv_from = lambda **_kwargs: rollout_result
    env_calls = []

    def terminal_env_step(_actions, _stage_id):
        env_calls.append(True)
        return (
            EnvOutput(
                obs={"states": torch.ones((1, 8))},
                rewards=torch.zeros((1, 4)),
                dones=torch.tensor([[False, False, False, True]]),
                terminations=torch.tensor([[False, False, False, True]]),
                truncations=torch.zeros((1, 4), dtype=torch.bool),
            ),
            {},
            {},
        )

    worker.env_interact_step = terminal_env_step
    worker.compute_bootstrap_rewards = lambda env_output, _bootstrap, _reward: (
        env_output.rewards
    )
    worker.send_to = lambda **_kwargs: None
    worker.store_last_obs_and_intervened_info = lambda _outputs: None
    worker.finish_rollout = lambda: None

    run_interact_once = _unwrap_worker_method(EnvWorker._run_interact_once)
    metrics = asyncio.run(
        run_interact_once(
            worker,
            None,
            None,
            None,
            None,
            cooperative_yield=False,
        )
    )

    assert len(env_calls) == 1
    torch.testing.assert_close(metrics["rollout/valid_chunks"], torch.tensor([1.0]))
    torch.testing.assert_close(metrics["rollout/padded_chunks"], torch.tensor([2.0]))


def test_eval_single_env_terminal_padding_stops_real_env_calls():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"eval": {"auto_reset": False}},
            "rollout": {"group_name": "RolloutGroup"},
        }
    )
    worker.stage_num = 1
    worker.eval_rollout_epoch = 1
    worker.eval_num_envs_per_stage = 1
    worker.eval_batch_size = 1
    worker.n_eval_chunk_steps = 2
    worker.eval_terminal_padding_enabled = True
    worker.enable_rlt = False
    worker.eval_prev_done = [torch.zeros(1, dtype=torch.bool)]
    worker.env_decoupled_mode = False
    worker.eval_enable_offload = False
    worker.model_cfg = SimpleNamespace(num_action_chunks=4)
    eval_env = SimpleNamespace(
        is_start=False,
        reset=lambda: ({"states": torch.ones((1, 8))}, {}),
    )
    worker.eval_env_list = [eval_env]
    worker.recv_from = lambda **_kwargs: torch.ones((1, 32))
    worker.send_to = lambda **_kwargs: None
    worker.finish_rollout = lambda mode: None
    env_calls = []

    def terminal_eval_step(_actions, _stage_id, gripper_bypass_mask=None):
        env_calls.append(True)
        return (
            EnvOutput(
                obs={"states": torch.ones((1, 8))},
                rewards=torch.zeros((1, 4)),
                dones=torch.tensor([[False, False, False, True]]),
                terminations=torch.tensor([[False, False, False, True]]),
                truncations=torch.zeros((1, 4), dtype=torch.bool),
            ),
            {},
        )

    worker.env_evaluate_step = terminal_eval_step

    evaluate = _unwrap_worker_method(EnvWorker.evaluate)
    evaluate(worker, None, None)

    assert len(env_calls) == 1


def test_reward_label_invalid_masks_the_entire_trajectory():
    batch = {
        "rewards": torch.ones((3, 2, 1)),
        "loss_mask": torch.ones((3, 2, 1), dtype=torch.bool),
        "audit_info": {
            "reward_label_valid": torch.tensor(
                [
                    [True, True],
                    [True, False],
                    [True, True],
                ]
            )
        },
    }

    masked = apply_reward_label_validity_mask(batch, enabled=True)

    assert masked["loss_mask"][:, 0].all()
    assert not masked["loss_mask"][:, 1].any()


def test_reward_label_validity_creates_mask_when_termination_mask_is_disabled():
    batch = {
        "rewards": torch.ones((2, 2, 4)),
        "audit_info": {
            "reward_label_valid": torch.tensor(
                [
                    [[True], [False]],
                    [[True], [True]],
                ]
            )
        },
    }

    masked = apply_reward_label_validity_mask(batch, enabled=True)

    assert masked["loss_mask"].shape == batch["rewards"].shape
    assert masked["loss_mask"][:, 0].all()
    assert not masked["loss_mask"][:, 1].any()


def test_reward_label_validity_is_noop_unless_explicitly_enabled():
    original_mask = torch.ones((2, 1, 1), dtype=torch.bool)
    batch = {
        "rewards": torch.ones((2, 1, 1)),
        "loss_mask": original_mask.clone(),
        "audit_info": {"reward_label_valid": torch.tensor([[True], [False]])},
    }

    masked = apply_reward_label_validity_mask(batch)

    torch.testing.assert_close(masked["loss_mask"], original_mask)


@pytest.mark.parametrize(
    ("adv_type", "group_size", "error"),
    [
        ("grpo", 1, "adv_type='gae'"),
        ("gae", 2, "group_size=1"),
    ],
)
def test_reward_label_validity_rejects_unsupported_algorithm_contracts(
    adv_type,
    group_size,
    error,
):
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "adv_type": adv_type,
                "group_size": group_size,
                "reward_label_validity": {"enabled": True},
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match=error):
        validate_reward_label_validity_config(cfg)


def test_terminal_padding_contract_rejects_more_than_one_env_per_stage():
    env_cfg = OmegaConf.create(
        {
            "auto_reset": False,
            "init_params": {"id": "DobotPickAndPlaceEnv-v1"},
            "use_keyboard_intervention": True,
            "keyboard_intervention": {"episode_control_mode": "online_chunk_boundary"},
            "terminal_padding": {"enabled": True},
        }
    )

    with pytest.raises(ValueError, match="exactly one environment per stage"):
        EnvWorker._validate_terminal_padding_contract(
            mode="train",
            env_cfg=env_cfg,
            num_envs_per_stage=2,
        )


def test_terminal_padding_contract_is_noop_when_disabled_for_multi_env():
    env_cfg = OmegaConf.create(
        {
            "auto_reset": False,
            "terminal_padding": {"enabled": False},
        }
    )

    assert not EnvWorker._validate_terminal_padding_contract(
        mode="eval",
        env_cfg=env_cfg,
        num_envs_per_stage=2,
    )
    assert not EnvWorker._terminal_padding_triggered(
        enabled=False,
        dones=torch.tensor([[True], [False]]),
    )


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_single_env_terminal_padding_trigger_is_shared_by_train_and_eval(mode):
    env_cfg = OmegaConf.create(
        {
            "auto_reset": False,
            "init_params": {"id": "DobotPickAndPlaceEnv-v1"},
            "use_keyboard_intervention": True,
            "keyboard_intervention": {"episode_control_mode": "online_chunk_boundary"},
            "terminal_padding": {"enabled": True},
        }
    )

    assert EnvWorker._validate_terminal_padding_contract(
        mode=mode,
        env_cfg=env_cfg,
        num_envs_per_stage=1,
    )
    assert EnvWorker._terminal_padding_triggered(
        enabled=True,
        dones=torch.tensor([[False, False, False, True]]),
    )
    assert not EnvWorker._terminal_padding_triggered(
        enabled=True,
        dones=torch.tensor([[False, False, False, False]]),
    )
