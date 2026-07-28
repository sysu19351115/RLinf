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

"""Real-checkpoint forward + backward test for 50-step DAgger.

Verifies that prepare_dagger_sft_batch correctly reshapes to [B, 50, 32]
and sft_forward produces a finite loss. Then attempts backward + optimizer
step to verify gradients are finite and only action-expert params update.
OOM is a failed hardware gate, never a successful forward-only result.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_home"))
os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / ".cache" / "openpi"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

MODEL_PATH = os.environ.get("DOBOT_HG_DAGGER_MODEL_PATH")
NORM_STATS_PATH = os.environ.get("DOBOT_HG_DAGGER_NORM_STATS_PATH")
if not MODEL_PATH or not NORM_STATS_PATH:
    raise RuntimeError(
        "Set DOBOT_HG_DAGGER_MODEL_PATH and "
        "DOBOT_HG_DAGGER_NORM_STATS_PATH before running this hardware gate."
    )
if not Path(MODEL_PATH).is_dir():
    raise FileNotFoundError(f"Model checkpoint directory not found: {MODEL_PATH}")
if not Path(NORM_STATS_PATH).is_file():
    raise FileNotFoundError(f"Normalization stats not found: {NORM_STATS_PATH}")

cfg = OmegaConf.create(
    {
        "model_path": MODEL_PATH,
        "model_type": "openpi",
        "precision": "bf16",
        "num_action_chunks": 10,
        "action_dim": 8,
        "num_steps": 4,
        "add_value_head": False,
        "is_lora": False,
        "load_to_device": True,
        "openpi": {
            "config_name": "pi05_dobot_pose",
            "train_expert_only": True,
            "value_after_vlm": False,
            "num_images_in_input": 2,
            "noise_method": "flow_ode",
            "action_horizon": 50,
            "action_chunk": 10,
            "action_env_dim": 8,
            "num_steps": 4,
        },
        "openpi_data": {
            "repo_id": "",
            "norm_stats_path": NORM_STATS_PATH,
        },
    }
)

from rlinf.models import get_model  # noqa: E402
from rlinf.scheduler import Worker  # noqa: E402

Worker.torch_platform = None
Worker.torch_device_type = "cuda:0"

print("Loading model...")
model = get_model(cfg)
model.eval()

bsz = 1
actions = torch.randn(bsz, 50 * 8)

batch = {
    "action": actions,
    "observation/prev_state": torch.randn(bsz, 8),
    "observation/state": torch.randn(bsz, 8),
    "observation/image": torch.randn(bsz, 3, 224, 224),
    "observation/wrist_image": torch.randn(bsz, 3, 224, 224),
    "human_action_mask": torch.tensor([[True] * 10 + [False] * 40]),
}

print("Preparing DAgger SFT batch...")
data = model.prepare_dagger_sft_batch(batch, loss_scope="human_only")
print(f"  actions shape: {data['actions'].shape}")
assert data["actions"].shape == (bsz, 50, 32), (
    f"Expected (1, 50, 32), got {data['actions'].shape}"
)

print("Forward pass...")
model.train()
loss = model.sft_forward(data=data, use_action_chunk_loss=True)
print(f"  loss: {loss.item()}")
assert torch.isfinite(loss), "Loss is not finite!"

# Check trainable vs frozen.
vlm_named_params = list(model.paligemma_with_expert.paligemma.named_parameters())
assert vlm_named_params, "PaliGemma VLM has no parameters"
trainable_vlm_names = [name for name, p in vlm_named_params if p.requires_grad]
assert not trainable_vlm_names, (
    "All PaliGemma VLM parameters must be frozen; trainable parameters: "
    f"{trainable_vlm_names[:10]}"
)

action_expert_named_params = list(
    model.paligemma_with_expert.gemma_expert.named_parameters()
)
assert action_expert_named_params, "Action expert has no parameters"
trainable_action_expert = [
    (name, p) for name, p in action_expert_named_params if p.requires_grad
]
assert trainable_action_expert, "Action expert has no trainable parameters"
trainable_action_expert_ids = {id(p) for _, p in trainable_action_expert}

trainable_named_params = [
    (name, p) for name, p in model.named_parameters() if p.requires_grad
]
trainable = len(trainable_named_params)
frozen = sum(1 for p in model.parameters() if not p.requires_grad)
print(f"  trainable: {trainable}, frozen: {frozen}")
print(f"  frozen PaliGemma params: {len(vlm_named_params)}")
print(f"  trainable action-expert params: {len(trainable_action_expert)}")

print("\n=== FORWARD CHECKS PASSED ===")

# ── Backward + optimizer step ────────────────────────────────────
# Keep reference copies on CPU so the verification itself does not duplicate
# the frozen VLM and action expert in scarce GPU memory.
frozen_before = {
    id(p): p.detach().cpu().clone() for p in model.parameters() if not p.requires_grad
}
trainable_before = {
    id(p): p.detach().cpu().clone() for p in model.parameters() if p.requires_grad
}

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

print("Attempting backward + optimizer step...")
try:
    optimizer.zero_grad()
    loss.backward()
    missing_grad_names = [name for name, p in trainable_named_params if p.grad is None]
    nonfinite_grad_names = [
        name
        for name, p in trainable_named_params
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    action_expert_grad_names = [
        name for name, p in trainable_action_expert if p.grad is not None
    ]
    print(f"  trainable params without gradients: {len(missing_grad_names)}")
    if missing_grad_names:
        print(f"    examples: {missing_grad_names[:10]}")
    print(f"  action-expert params with gradients: {len(action_expert_grad_names)}")
    assert not nonfinite_grad_names, (
        "Non-finite gradients detected in trainable parameters: "
        f"{nonfinite_grad_names[:10]}"
    )
    assert action_expert_grad_names, (
        "No trainable action-expert parameter received a gradient."
    )

    optimizer.step()
    print("  optimizer step completed.")
except torch.cuda.OutOfMemoryError as e:
    raise RuntimeError(
        "DAgger hardware gate failed: insufficient GPU memory for backward "
        "and optimizer verification."
    ) from e

# Verify frozen weights are unchanged.
frozen_ok = True
for p in model.parameters():
    if not p.requires_grad and id(p) in frozen_before:
        if not torch.equal(p.detach().cpu(), frozen_before[id(p)]):
            frozen_ok = False
            break
assert frozen_ok, "Frozen (VLM) weights changed after optimizer step!"

# Verify at least one trainable action-expert weight changed.
trainable_changed = False
for p in model.parameters():
    if id(p) in trainable_action_expert_ids and id(p) in trainable_before:
        if not torch.equal(p.detach().cpu(), trainable_before[id(p)]):
            trainable_changed = True
            break
assert trainable_changed, "No action-expert weight changed after optimizer step!"

print("  frozen VLM weights: unchanged")
print("  action-expert weights: at least one updated")

# Verify loss is still finite after the step.
print("Verifying post-step forward...")
with torch.no_grad():
    model.eval()
    loss2 = model.sft_forward(data=data, use_action_chunk_loss=True)
assert torch.isfinite(loss2), "Post-step loss is not finite!"
print(f"  post-step loss: {loss2.item()}")

print("\n=== ALL CHECKS PASSED (forward + backward + optimizer) ===")
