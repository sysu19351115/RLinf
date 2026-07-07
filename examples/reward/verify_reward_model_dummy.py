#!/usr/bin/env python3
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

"""Dummy-mode verification of the ReBot env + ResNet reward model integration.

This script creates a dummy RebotArmEnv with use_reward_model=True, loads real
images from the preprocessed reward dataset, and checks that the reward worker
returns a sensible scalar reward. It does not require real hardware.

Because the environment is in dummy mode, the robot controller and cameras are
not exercised; only the reward-model inference path is validated.
"""

import argparse
from pathlib import Path

import numpy as np

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.envs.realworld.rebot.rebot_env import RebotArmEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dummy-mode verification of ReBot reward model integration."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=(
            "/home/tyz/project/RLinf/logs/rebot_reward_model/"
            "rebot_reward_training/checkpoints/best_model/"
            "actor/model_state_dict/full_weights.pt"
        ),
        help="Path to the trained ResNet reward model checkpoint.",
    )
    parser.add_argument(
        "--data-pt",
        type=str,
        default=(
            "/home/tyz/project/RLinf/logs/rebot_reward_data/"
            "processed_ratio02_balanced2/train.pt"
        ),
        help="Path to a preprocessed .pt dataset (train.pt or val.pt).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of positive/negative samples to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dummy-mode reward model integration checks."""
    args = parse_args()

    print("=" * 60)
    print("Loading reward dataset to get example images...")
    payload = RewardDatasetPayload.load(args.data_pt)

    rng = np.random.default_rng(args.seed)
    pos_indices = [i for i, label in enumerate(payload.labels) if label == 1]
    neg_indices = [i for i, label in enumerate(payload.labels) if label == 0]

    if not pos_indices or not neg_indices:
        raise ValueError(
            f"Dataset must contain both labels. Found {len(pos_indices)} positive "
            f"and {len(neg_indices)} negative samples."
        )

    num_samples = min(args.num_samples, len(pos_indices), len(neg_indices))
    pos_sampled = rng.choice(pos_indices, size=num_samples, replace=False)
    neg_sampled = rng.choice(neg_indices, size=num_samples, replace=False)

    print(f"Dataset: {args.data_pt}")
    print(f"Total samples: {len(payload.labels)}")
    print(f"Positive samples: {len(pos_indices)}, Negative samples: {len(neg_indices)}")
    print(f"Testing {num_samples} random positive and {num_samples} random negative images.")

    print("=" * 60)
    print("Creating dummy RebotArmEnv with use_reward_model=True...")
    reward_worker_cfg = {
        "use_reward_model": True,
        "model": {
            "model_type": "resnet",
            "model_path": args.checkpoint_path,
            "arch": "resnet18",
            "hidden_dim": 256,
            "dropout": 0.1,
            "image_size": [3, 224, 224],
            "normalize": True,
            "precision": "fp32",
        },
    }

    env = RebotArmEnv(
        override_cfg={
            "is_dummy": True,
            "use_reward_model": True,
            "reward_image_key": "wrist_1",
            "reward_worker_cfg": reward_worker_cfg,
            "enable_gripper_penalty": False,
        },
        env_idx=0,
    )
    print("Environment created. Reward worker initialized.")

    print("=" * 60)
    print("Testing reward computation on sampled images...")
    pos_rewards = []
    neg_rewards = []

    for idx in pos_sampled:
        image = payload.images[idx].permute(1, 2, 0).numpy()  # CHW -> HWC
        obs = {"state": {}, "frames": {"wrist_1": image}}
        reward = env._calc_step_reward(obs, is_gripper_action_effective=False)
        pos_rewards.append(reward)

    for idx in neg_sampled:
        image = payload.images[idx].permute(1, 2, 0).numpy()
        obs = {"state": {}, "frames": {"wrist_1": image}}
        reward = env._calc_step_reward(obs, is_gripper_action_effective=False)
        neg_rewards.append(reward)

    pos_rewards = np.array(pos_rewards)
    neg_rewards = np.array(neg_rewards)

    print(f"  Positive rewards: mean={pos_rewards.mean():.4f}, std={pos_rewards.std():.4f}, min={pos_rewards.min():.4f}, max={pos_rewards.max():.4f}")
    print(f"  Negative rewards: mean={neg_rewards.mean():.4f}, std={neg_rewards.std():.4f}, min={neg_rewards.min():.4f}, max={neg_rewards.max():.4f}")

    # Pairwise comparison: how often positive > negative.
    pairwise_correct = np.sum(pos_rewards > neg_rewards)
    pairwise_tied = np.sum(pos_rewards == neg_rewards)
    print(f"  Pairwise positive>negative: {pairwise_correct}/{num_samples} ({pairwise_correct/num_samples:.1%})")
    if pairwise_tied:
        print(f"  Pairwise ties: {pairwise_tied}/{num_samples}")

    print("=" * 60)
    if pos_rewards.mean() > neg_rewards.mean() and pairwise_correct > num_samples / 2:
        print("✅ PASS: Positive images receive higher reward on average.")
    else:
        print("❌ FAIL: Reward model does not consistently assign higher reward to positive images.")
        print("   Please check the dataset labels and retrain the reward model.")

    print("=" * 60)
    print("Testing env.reset() and env.step() with dummy observations...")
    obs, info = env.reset()
    print(f"  reset observation keys: {list(obs.keys())}")
    print(f"  frames keys: {list(obs.get('frames', {}).keys())}")

    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  step reward: {reward:.4f}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("=" * 60)
    print("Dummy verification complete.")


if __name__ == "__main__":
    main()
