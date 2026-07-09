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

"""Convert success/failure image folders into train/test .pt reward datasets.

Expects the input directory to contain ``success/`` and ``failure/`` subfolders.
All images under ``success/`` are labeled 1, and all images under ``failure/``
are labeled 0. The combined set is shuffled and split into ``train.pt`` and
``test.pt`` using ``RewardDatasetPayload``.

Usage:
    python examples/reward/convert_reward_images_to_pt.py \
        --input-dir datasets/so101_reward_images \
        --output-dir logs/so101_reward_data/processed \
        --test-ratio 0.2 \
        --seed 42
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.utils.logging import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert success/failure image folders to reward model .pt files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/so101_reward_images",
        help="Directory containing success/ and failure/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/so101_reward_data/processed",
        help="Directory to write train.pt and test.pt.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of all images reserved for test.pt (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling the dataset split (default: 42).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image extensions to load (default: .png .jpg .jpeg .bmp).",
    )
    return parser.parse_args()


def _load_images_from_folder(
    folder: Path,
    label: int,
    extensions: set[str],
) -> tuple[list[torch.Tensor], list[int]]:
    """Load all images from a folder as CHW uint8 tensors.

    Args:
        folder: Path to the image folder.
        label: Integer label to assign to every loaded image.
        extensions: Allowed image extensions.

    Returns:
        Tuple of (image_tensors, labels).
    """
    images: list[torch.Tensor] = []
    labels: list[int] = []

    if not folder.exists():
        logger.warning(f"Folder not found: {folder}")
        return images, labels

    image_paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )

    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning(f"Could not read image, skipping: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Store as uint8 CHW tensor, consistent with RewardBinaryDataset expectations.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        images.append(tensor)
        labels.append(label)

    return images, labels


def main() -> None:
    """Load images, split, and save train/test reward dataset payloads."""
    args = parse_args()

    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be between 0 and 1, got {args.test_ratio}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(ext.lower() for ext in args.extensions)

    success_images, success_labels = _load_images_from_folder(
        input_dir / "success", label=1, extensions=extensions
    )
    failure_images, failure_labels = _load_images_from_folder(
        input_dir / "failure", label=0, extensions=extensions
    )

    all_images = success_images + failure_images
    all_labels = success_labels + failure_labels

    if len(all_images) == 0:
        raise ValueError(
            f"No images found in {input_dir}/success or {input_dir}/failure."
        )

    indices = list(range(len(all_images)))
    random.shuffle(indices)

    n_test = max(1, int(len(indices) * args.test_ratio))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    def _select(idx_list: list[int]) -> tuple[list[torch.Tensor], list[int]]:
        return [all_images[i] for i in idx_list], [all_labels[i] for i in idx_list]

    train_images, train_labels = _select(train_indices)
    test_images, test_labels = _select(test_indices)

    metadata = {
        "input_dir": str(input_dir),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "num_total_samples": len(all_images),
        "num_success": len(success_images),
        "num_failure": len(failure_images),
        "num_train_samples": len(train_images),
        "num_test_samples": len(test_images),
        "num_train_positive": sum(train_labels),
        "num_test_positive": sum(test_labels),
    }

    train_path = output_dir / "train.pt"
    test_path = output_dir / "test.pt"

    RewardDatasetPayload(
        images=train_images, labels=train_labels, metadata=metadata
    ).save(str(train_path))
    RewardDatasetPayload(
        images=test_images, labels=test_labels, metadata=metadata
    ).save(str(test_path))

    logger.info(f"Saved train split: {train_path} ({len(train_images)} samples)")
    logger.info(f"Saved test split:  {test_path} ({len(test_images)} samples)")
    logger.info(
        f"Total: {len(all_images)} (success={len(success_images)}, "
        f"failure={len(failure_images)})"
    )


if __name__ == "__main__":
    main()
