# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone single-GPU trainer for the ResNet reward model.

This script bypasses Ray/FSDP and trains the reward model directly with
vanilla PyTorch. It uses the same YAML config as
``examples/reward/train_reward_model.py``.

Usage:
    python examples/reward/train_reward_model_simple.py --config-name so101_reward_training

The best checkpoint is saved to
``{log_path}/{experiment_name}/checkpoints/best_model/actor/model_state_dict/full_weights.pt``
to stay compatible with the RL YAML ``reward_worker_cfg.model.model_path``.
"""

import json
import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlinf.data.datasets.reward_model import RewardBinaryDataset
from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.utils.logging import get_logger

logger = get_logger()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="so101_reward_training",
)
def main(cfg: DictConfig) -> None:
    """Run single-GPU reward model training."""
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model.
    model_cfg = cfg.actor.model
    model = get_reward_model_class(model_cfg.model_type)(model_cfg)
    model.to(device)

    # Build datasets.
    data_cfg = cfg.data
    train_dataset = RewardBinaryDataset(data_cfg.train_data_paths)
    val_dataset = RewardBinaryDataset(data_cfg.val_data_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.actor.micro_batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.actor.micro_batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"train batches/epoch={len(train_loader)}"
    )

    # Optimizer.
    optim_cfg = cfg.actor.optim
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg.lr,
        betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
        eps=optim_cfg.adam_eps,
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    # LR scheduler (constant with warmup).
    total_steps = cfg.actor.optim.get(
        "total_training_steps", cfg.runner.max_epochs * len(train_loader)
    )
    warmup_steps = optim_cfg.get("lr_warmup_steps", 0)

    def lr_lambda(step: int) -> float:
        if step >= total_steps:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint path.
    ckpt_root = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
    best_ckpt_dir = ckpt_root / "checkpoints" / "best_model" / "actor" / "model_state_dict"
    best_ckpt_path = best_ckpt_dir / "full_weights.pt"

    # Training loop.
    best_val_acc = -1.0
    patience = cfg.runner.get("early_stop", {}).get("patience", 5)
    min_delta = cfg.runner.get("early_stop", {}).get("min_delta", 0.001)
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(cfg.runner.max_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.runner.max_epochs} train")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = outputs["loss"]
            loss.backward()
            if optim_cfg.get("clip_grad", None) is not None and optim_cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optim_cfg.clip_grad
                )
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()
            train_acc += outputs["accuracy"].item()
            pbar.set_postfix(
                loss=loss.item(), acc=outputs["accuracy"].item(), lr=optimizer.param_groups[0]["lr"]
            )

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="val"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images, labels)
                val_loss += outputs["loss"].item()
                val_acc += outputs["accuracy"].item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Save best checkpoint.
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(best_ckpt_path))
            logger.info(f"Saved best checkpoint to {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping after {epoch + 1} epochs "
                f"(no improvement for {patience} epochs)."
            )
            break

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
