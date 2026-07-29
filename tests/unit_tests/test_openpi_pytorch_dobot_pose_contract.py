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

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.openpi.pose_transforms import AbsolutePose
from rlinf.models.embodiment.openpi_pytorch.eval_action_model import (
    OpenPiPytorchEvalActionModel,
    _to_numpy,
)

_PREV_STATE = torch.tensor(
    [[0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.5]],
    dtype=torch.float32,
)


class _DummyPi0(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def sample_actions(self, observation, *, num_steps, noise, rng):
        del observation, num_steps, noise, rng
        return torch.zeros((1, 50, 32), dtype=torch.float32)


def _env_obs(*, include_prev_state: bool = True):
    obs = {
        "states": torch.zeros((1, 8), dtype=torch.float32),
        "main_images": torch.zeros((1, 3, 4, 4), dtype=torch.uint8),
        "task_descriptions": ["test"],
    }
    if include_prev_state:
        obs["prev_states"] = _PREV_STATE.clone()
    return obs


def _make_eval_model(
    captured_output: dict,
    *,
    preserve_dagger_anchor_inputs: bool = True,
):
    model = OpenPiPytorchEvalActionModel(
        _DummyPi0(),
        num_steps=5,
        action_env_dim=8,
        action_chunk=10,
        config_name="pi05_dobot_pose",
        preserve_dagger_anchor_inputs=preserve_dagger_anchor_inputs,
    )

    def input_transform(_obs, transpose=False):
        del transpose
        return {
            "state": torch.zeros((1, 32), dtype=torch.float32),
            "tokenized_prompt": torch.ones((1, 4), dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones((1, 4), dtype=torch.bool),
        }

    def to_observation(processed):
        return SimpleNamespace(
            state=processed["state"],
            tokenized_prompt=processed["tokenized_prompt"],
            tokenized_prompt_mask=processed["tokenized_prompt_mask"],
        )

    def output_transform(outputs):
        captured_output.clear()
        captured_output.update(outputs)
        return {"actions": torch.zeros((1, 10, 8), dtype=torch.float32)}

    model.input_transform = input_transform
    model._observation_dict_to_device = to_observation
    model.output_transform = output_transform
    return model


def test_repack_preserves_prev_state():
    model = _make_eval_model({})

    repacked = model._repack_env_obs(_env_obs())

    torch.testing.assert_close(
        repacked["observation/prev_state"],
        _PREV_STATE,
    )


def test_bfloat16_model_output_can_cross_numpy_transform_boundary():
    converted = _to_numpy(torch.ones(2, dtype=torch.bfloat16))

    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, np.ones(2, dtype=np.float32))


def test_eval_output_transform_receives_prev_state():
    captured_output = {}
    model = _make_eval_model(captured_output)

    model.predict_action_batch(_env_obs(), mode="eval")

    torch.testing.assert_close(captured_output["prev_state"], _PREV_STATE)


def test_eval_without_prev_state_remains_supported():
    captured_output = {}
    model = _make_eval_model(captured_output)

    model.predict_action_batch(_env_obs(include_prev_state=False), mode="eval")

    assert "prev_state" not in captured_output


def test_absolute_pose_zero_delta_round_trip_uses_prev_state():
    previous = _PREV_STATE.numpy()[0]
    relative_identity = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, previous[-1]],
        dtype=np.float32,
    )
    transformed = AbsolutePose(mask=(True,) * 7 + (False,))(
        {
            "prev_state": previous.copy(),
            "state": relative_identity.copy(),
            "actions": np.repeat(relative_identity[None, :], 2, axis=0),
        }
    )

    np.testing.assert_allclose(transformed["state"], previous, atol=1e-6)
    np.testing.assert_allclose(
        transformed["actions"],
        np.repeat(previous[None, :], 2, axis=0),
        atol=1e-6,
    )


def test_rollout_preserves_dagger_anchor_inputs():
    captured_output = {}
    model = _make_eval_model(captured_output)

    actions, result = model.predict_action_batch(_env_obs(), mode="eval")

    forward_inputs = result["forward_inputs"]
    assert actions.shape == (1, 10, 8)
    assert forward_inputs["action"].shape == (1, 10 * 8)
    assert forward_inputs["model_action"].shape == (1, 50 * 32)
    assert "observation/image" in forward_inputs
    assert "observation/state" in forward_inputs
    assert "observation/prev_state" in forward_inputs
    assert "tokenized_prompt" in forward_inputs
    assert "tokenized_prompt_mask" in forward_inputs


def test_regular_eval_keeps_lightweight_forward_inputs():
    model = _make_eval_model({}, preserve_dagger_anchor_inputs=False)

    _, result = model.predict_action_batch(_env_obs(), mode="eval")

    assert set(result["forward_inputs"]) == {"action", "model_action"}


def test_rollout_anchor_inputs_do_not_alias_env_observation():
    captured_output = {}
    model = _make_eval_model(captured_output)
    obs = _env_obs()

    _, result = model.predict_action_batch(obs, mode="eval")
    obs["states"].fill_(9)
    obs["prev_states"].fill_(9)
    obs["main_images"].fill_(9)

    forward_inputs = result["forward_inputs"]
    assert not torch.eq(forward_inputs["observation/state"], 9).any()
    assert not torch.eq(forward_inputs["observation/prev_state"], 9).any()
    assert not torch.eq(forward_inputs["observation/image"], 9).any()
