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

"""Preprocess a LeRobot-format ReBot dataset for ResNet reward model training.

The script loads a LeRobot dataset, labels the last ``success_ratio`` frames of
each demonstration episode as success (1) and the remaining frames as fail (0),
and exports episode-split train/val ``.pt`` files compatible with
``RewardBinaryDataset``.

Example:
    python examples/reward/preprocess_rebot_lerobot.py \
        --dataset-path datasets/rebot_lerobot_data \
        --output-dir logs/rebot_reward_data/processed \
        --image-key observation.images.cam_left_wrist \
        --success-ratio 0.1 \
        --val-split 0.2
"""

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm.auto import tqdm

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.utils.logging import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess LeRobot ReBot dataset for reward model training."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/rebot_reward_data/processed",
        help="Output directory for train.pt and val.pt.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="observation.images.cam_left_wrist",
        help="Image feature key in the LeRobot dataset.",
    )
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=0.1,
        help="Fraction of trailing frames in each episode labeled as success.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of episodes reserved for validation.",
    )
    parser.add_argument(
        "--val-balance-ratio",
        type=float,
        default=2.0,
        help="Validation negative:positive sampling ratio. "
             "Use <=0 to disable balancing and keep all validation frames.",
    )
    parser.add_argument(
        "--fail-episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional episode indices to label entirely as fail.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val episode split and sampling.",
    )
    return parser.parse_args()


def _load_episode_images(
    dataset_root: Path,
    episode_index: int,
    image_key: str,
    chunks_size: int,
) -> list[torch.Tensor]:
    """Load images for one episode directly from its parquet file.

    LeRobot stores image features as ``struct<bytes: binary, path: string>``.
    This function decodes the embedded PNG/JPEG bytes and returns CPU tensors
    in channel-first ``[C, H, W]`` uint8 layout.

    Args:
        dataset_root: Root directory of the LeRobot dataset.
        episode_index: Episode index to load.
        image_key: Column name of the image feature.
        chunks_size: Number of episodes per chunk directory.

    Returns:
        List of image tensors for the episode.
    """
    chunk_idx = episode_index // chunks_size
    parquet_path = (
        dataset_root
        / "data"
        / f"chunk-{chunk_idx:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(str(parquet_path), columns=[image_key])
    column = table[image_key]

    images: list[torch.Tensor] = []
    for i in range(len(column)):
        item = column[i].as_py()
        image_bytes = item["bytes"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        np_image = np.asarray(pil_image)
        # Ensure 3-channel RGB.
        if np_image.ndim == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        np_image = np_image[..., :3]
        # Convert HWC uint8 to CHW torch tensor.
        tensor_image = torch.from_numpy(np_image).permute(2, 0, 1)
        images.append(tensor_image)

    return images


def _balance_validation_set(
    images: list[torch.Tensor],
    labels: list[int],
    neg_pos_ratio: float,
    rng: random.Random,
) -> tuple[list[torch.Tensor], list[int]]:
    """Balance the validation set by sampling negatives to a target ratio.

    Args:
        images: List of validation images.
        labels: List of validation labels.
        neg_pos_ratio: Target negative:positive ratio. For example, 2.0 means
            keep at most 2 negative samples per positive sample.
        rng: Random number generator for deterministic sampling.

    Returns:
        Tuple of (balanced_images, balanced_labels).
    """
    if neg_pos_ratio <= 0:
        return images, labels

    positive_pairs = [(img, lbl) for img, lbl in zip(images, labels) if lbl == 1]
    negative_pairs = [(img, lbl) for img, lbl in zip(images, labels) if lbl == 0]

    num_positive = len(positive_pairs)
    if num_positive == 0:
        logger.warning("No positive samples in validation set; skipping balance.")
        return images, labels

    target_negative = int(num_positive * neg_pos_ratio)
    if len(negative_pairs) > target_negative:
        rng.shuffle(negative_pairs)
        negative_pairs = negative_pairs[:target_negative]

    balanced_pairs = positive_pairs + negative_pairs
    rng.shuffle(balanced_pairs)
    balanced_images = [pair[0] for pair in balanced_pairs]
    balanced_labels = [pair[1] for pair in balanced_pairs]
    return balanced_images, balanced_labels


def preprocess_lerobot_for_reward(
    dataset_path: str,
    output_dir: str,
    image_key: str = "observation.images.cam_left_wrist",
    success_ratio: float = 0.1,
    val_split: float = 0.2,
    val_balance_ratio: float = 2.0,
    fail_episodes: Optional[list[int]] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert a LeRobot ReBot dataset into ResNet reward model training splits.

    Args:
        dataset_path: LeRobot dataset root directory or repo_id.
        output_dir: Directory to write ``train.pt`` and ``val.pt``.
        image_key: Image observation key in the LeRobot dataset.
        success_ratio: Fraction of trailing frames per episode labeled as 1.
        val_split: Fraction of episodes used for validation.
        val_balance_ratio: Target negative:positive ratio in the validation set.
            Use <=0 to keep all validation frames.
        fail_episodes: Optional episode indices labeled entirely as 0.
        seed: Random seed for the episode-level train/val split and sampling.

    Returns:
        Metadata dictionary describing the generated splits.
    """

    # Lazy imports so the script does not require heavy deps at import time.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading LeRobot dataset metadata from {dataset_path}")
    dataset = LeRobotDataset(dataset_path)

    num_episodes = dataset.num_episodes
    episode_from = dataset.episode_data_index["from"].tolist()
    episode_to = dataset.episode_data_index["to"].tolist()
    fail_episode_set = set(fail_episodes) if fail_episodes else set()

    logger.info(
        f"Found {num_episodes} episodes, {len(dataset)} frames, "
        f"image_key='{image_key}'"
    )

    # Load dataset metadata to locate parquet files.
    dataset_root = Path(dataset_path)
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", 1000)

    # Episode-level train/val split to avoid data leakage.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_episodes, generator=generator).tolist()
    val_ep_count = max(1, int(num_episodes * val_split))
    val_episodes = set(perm[:val_ep_count])

    rng = random.Random(seed)

    images: dict[str, list[torch.Tensor]] = {"train": [], "val": []}
    labels: dict[str, list[int]] = {"train": [], "val": []}

    for ep_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        start_idx = episode_from[ep_idx]
        end_idx = episode_to[ep_idx]
        n = end_idx - start_idx
        if n <= 0:
            logger.warning(f"Episode {ep_idx} is empty, skipping.")
            continue

        split = "val" if ep_idx in val_episodes else "train"
        ep_images = _load_episode_images(
            dataset_root, ep_idx, image_key, chunks_size
        )
        if len(ep_images) != n:
            logger.warning(
                f"Episode {ep_idx}: parquet has {len(ep_images)} frames but "
                f"episode_data_index expects {n}; using parquet count."
            )
            n = len(ep_images)

        is_fail = ep_idx in fail_episode_set
        k = max(1, int(n * success_ratio))

        for i, img in enumerate(ep_images):
            if is_fail:
                label = 0
            else:
                label = 1 if i >= n - k else 0

            images[split].append(img)
            labels[split].append(label)

    # Balance the validation set so validation metrics are not dominated
    # by the majority negative class.
    raw_val_images = images["val"]
    raw_val_labels = labels["val"]
    images["val"], labels["val"] = _balance_validation_set(
        raw_val_images,
        raw_val_labels,
        neg_pos_ratio=val_balance_ratio,
        rng=rng,
    )

    metadata = {
        "dataset_path": str(dataset_path),
        "image_key": image_key,
        "success_ratio": success_ratio,
        "val_split": val_split,
        "val_balance_ratio": val_balance_ratio,
        "num_val_raw_samples": len(raw_val_images),
        "num_val_raw_positive": sum(raw_val_labels),
        "fail_episodes": sorted(fail_episode_set),
        "seed": seed,
        "num_train_samples": len(images["train"]),
        "num_val_samples": len(images["val"]),
        "num_train_positive": sum(labels["train"]),
        "num_val_positive": sum(labels["val"]),
    }

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_payload = RewardDatasetPayload(
        images=images["train"],
        labels=labels["train"],
        metadata=metadata,
    )
    val_payload = RewardDatasetPayload(
        images=images["val"],
        labels=labels["val"],
        metadata=metadata,
    )

    train_path = output_dir_path / "train.pt"
    val_path = output_dir_path / "val.pt"
    train_payload.save(str(train_path))
    val_payload.save(str(val_path))

    logger.info(f"Saved train split to {train_path}: {len(train_payload.labels)} samples")
    logger.info(f"Saved val split to {val_path}: {len(val_payload.labels)} samples")
    logger.info(
        f"Train positive ratio: {sum(labels['train']) / max(len(labels['train']), 1):.4f}, "
        f"Val positive ratio: {sum(labels['val']) / max(len(labels['val']), 1):.4f} "
        f"(raw val positive ratio: "
        f"{sum(raw_val_labels) / max(len(raw_val_labels), 1):.4f})"
    )

    return metadata


def main() -> None:
    """Run the LeRobot preprocessing pipeline."""
    args = parse_args()
    metadata = preprocess_lerobot_for_reward(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        image_key=args.image_key,
        success_ratio=args.success_ratio,
        val_split=args.val_split,
        val_balance_ratio=args.val_balance_ratio,
        fail_episodes=args.fail_episodes,
        seed=args.seed,
    )

    print("=" * 80)
    print("ReBot LeRobot reward dataset preprocessing complete")
    print(json.dumps(metadata, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
