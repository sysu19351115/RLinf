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

"""Real-checkpoint gate for Dobot OpenPI PyTorch HG-DAgger.

This script never constructs a robot environment. ``--mode rollout`` is safe to
run on the robot node because it only performs model inference on a synthetic
observation. ``--mode actor`` performs one supervised optimizer step and is
intended for the higher-memory actor node after its source tree has been
updated.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils import omega_resolver  # noqa: F401


def _load_cfg(checkpoint: str, precision: str):
    os.environ["DOBOT_HG_DAGGER_PYTORCH_MODEL_PATH"] = checkpoint
    os.environ.setdefault("DOBOT_HG_DAGGER_LR", "1e-6")
    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
    )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="dobot_hg_dagger_openpi_pytorch")
    OmegaConf.resolve(cfg)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    model_cfg.precision = precision
    model_cfg.load_to_device = False
    return cfg, model_cfg


def _synthetic_env_obs():
    # Valid robot-frame pose: xyz + identity quaternion + half-open gripper.
    state = torch.tensor(
        [[0.40, 0.00, 0.25, 1.0, 0.0, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    return {
        "states": state.clone(),
        "prev_states": state.clone(),
        "main_images": torch.zeros((1, 3, 480, 640), dtype=torch.uint8),
        "task_descriptions": [
            "Pick up an object with the hole from the tray and hang it on an empty hook."
        ],
    }


def _load_model(checkpoint: str, precision: str, device: torch.device):
    cfg, model_cfg = _load_cfg(checkpoint, precision)
    model = get_model(model_cfg)
    if model is None:
        raise RuntimeError("OpenPI PyTorch model factory returned None.")
    return cfg, model.to(device)


def run_rollout_gate(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, model = _load_model(args.checkpoint, "bf16", device)
    noise = torch.randn(
        1,
        model.model.action_horizon,
        model.model.action_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    env_obs = _synthetic_env_obs()
    actions, result = model.predict_action_batch(
        env_obs,
        mode="eval",
        noise=noise,
    )
    forward_inputs = result["forward_inputs"]

    assert tuple(actions.shape) == (1, 10, 8)
    assert tuple(forward_inputs["model_action"].shape) == (1, 50 * 32)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(forward_inputs["model_action"]).all()
    for key in (
        "observation/image",
        "observation/state",
        "observation/prev_state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
    ):
        assert key in forward_inputs, f"Missing DAgger replay anchor: {key}"

    replay = {
        key: value
        for key, value in forward_inputs.items()
        if key.startswith("observation/") or key.startswith("tokenized_prompt")
    }
    replay["action"] = actions.detach().cpu().repeat(1, 5, 1).reshape(1, -1)
    replay["human_action_mask"] = torch.cat(
        [torch.ones(1, 30), torch.zeros(1, 20)], dim=1
    ).bool()
    prepared = model.prepare_dagger_sft_batch(replay, loss_scope="human_only")
    assert tuple(prepared["actions"].shape) == (1, 50, 32)
    assert tuple(prepared["loss_mask"].shape) == (1, 50)
    assert int(prepared["loss_mask"].sum().item()) == 30
    assert torch.isfinite(prepared["actions"]).all()

    reference_xyz = env_obs["prev_states"][0, :3]
    first_xyz = actions.detach().cpu()[0, 0, :3]
    position_jump = torch.linalg.vector_norm(first_xyz - reference_xyz).item()
    if position_jump > args.max_position_jump_m:
        raise AssertionError(
            f"First predicted pose jumps {position_jump:.6f}m from prev_state; "
            f"limit is {args.max_position_jump_m:.6f}m."
        )
    print(
        "rollout gate passed:",
        f"actions={tuple(actions.shape)}",
        "model_action=(1, 50, 32)",
        "training_target=(1, 50, 32)",
        "human_steps=30",
        f"first_position_jump_m={position_jump:.6f}",
    )
    if device.type == "cuda":
        print(
            "cuda memory:",
            f"allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
            f"reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f}GiB",
        )


def run_actor_gate(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cfg, model = _load_model(args.checkpoint, "fp32", device)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    env_obs = _synthetic_env_obs()
    with torch.no_grad():
        _, rollout_result = model.predict_action_batch(
            env_obs,
            mode="eval",
            noise=torch.randn(
                1,
                50,
                32,
                device=device,
                dtype=torch.float32,
            ),
        )
    anchor = rollout_result["forward_inputs"]
    previous = env_obs["prev_states"]
    action_window = previous[:, None, :].repeat(1, 50, 1)
    human_mask = torch.zeros(1, 50, dtype=torch.bool)
    human_mask[:, :30] = True
    replay = {
        key: value
        for key, value in anchor.items()
        if key.startswith("observation/") or key.startswith("tokenized_prompt")
    }
    replay["action"] = action_window.reshape(1, -1)
    replay["human_action_mask"] = human_mask
    data = model.prepare_dagger_sft_batch(
        replay,
        loss_scope=cfg.algorithm.dagger.loss_scope,
    )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.actor.optim.lr),
        weight_decay=float(cfg.actor.optim.weight_decay),
    )
    probe = next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "llm.layers" in name
    )
    before = probe.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = model(
            forward_type=ForwardType.SFT,
            data=data,
            use_action_chunk_loss=True,
        )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in trainable if parameter.grad is not None
    ]
    assert gradients, "No action-expert gradients were produced."
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.model.img.parameters()), (
        "Frozen VLM unexpectedly received gradients."
    )
    torch.nn.utils.clip_grad_norm_(trainable, float(cfg.actor.optim.clip_grad))
    optimizer.step()
    assert not torch.equal(before, probe.detach()), "Optimizer changed no probe weight."
    print("actor gate passed:", f"loss={loss.item():.6f}")
    if device.type == "cuda":
        print(
            "cuda memory:",
            f"allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
            f"reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f}GiB",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("rollout", "actor"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-position-jump-m", type=float, default=0.10)
    args = parser.parse_args()
    if args.mode == "rollout":
        run_rollout_gate(args)
    else:
        run_actor_gate(args)


if __name__ == "__main__":
    main()
