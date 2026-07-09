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

"""Dummy-mode verification of the ResNet reward model integration.

This script creates a dummy real-world env with ``use_reward_model=True``,
loads real images from a preprocessed reward dataset, and checks that the
reward worker returns a sensible scalar reward. It does not require real
hardware.

Usage (SO101):
    python examples/reward/verify_reward_model_dummy.py \
        --env-type so101 \
        --checkpoint-path logs/so101_reward_model/so101_reward_training/checkpoints/best_model/actor/model_state_dict/full_weights.pt \
        --data-pt logs/so101_reward_data/processed/test.pt \
        --reward-image-key cam_high

Usage (ReBot):
    python examples/reward/verify_reward_model_dummy.py \
        --env-type rebot \
        --checkpoint-path /path/to/rebot_reward_model.pt \
        --data-pt /path/to/rebot_reward_data/train.pt \
        --reward-image-key wrist_1
"""

import argparse
from pathlib import Path

import numpy as np

from rlinf.data.datasets.reward_model import RewardDatasetPayload


ENV_DEFAULTS = {
    "so101": {
        "module": "rlinf.envs.realworld.so101.so101_env",
        "class": "SO101Env",
        "reward_image_key": "cam_high",
        "action_dim": 12,
    },
    "rebot": {
        "module": "rlinf.envs.realworld.rebot.rebot_env",
        "class": "RebotArmEnv",
        "reward_image_key": "wrist_1",
        "action_dim": 7,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dummy-mode verification of ResNet reward model integration."
    )
    parser.add_argument(
        "--env-type",
        type=str,
        choices=list(ENV_DEFAULTS.keys()),
        default="so101",
        help="Which real-world env to use for dummy verification.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained ResNet reward model checkpoint.",
    )
    parser.add_argument(
        "--data-pt",
        type=str,
        required=True,
        help="Path to a preprocessed .pt dataset (train.pt or test.pt).",
    )
    parser.add_argument(
        "--reward-image-key",
        type=str,
        default=None,
        help="Observation frame key to pass to the reward model. Default depends on --env-type.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="ResNet architecture used during training.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension of the reward model head.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=3,
        default=[3, 224, 224],
        help="Model input size as C H W.",
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


def _load_env_class(env_type: str):
    """Import the requested env class lazily."""
    import importlib

    info = ENV_DEFAULTS[env_type]
    module = importlib.import_module(info["module"])
    return getattr(module, info["class"])


def _calc_reward(env, observation: dict) -> float:
    """Call the env's reward function, handling signature differences."""
    try:
        return env._calc_step_reward(observation)
    except TypeError:
        return env._calc_step_reward(
            observation, is_gripper_action_effective=False
        )


def main() -> None:
    """Run dummy-mode reward model integration checks."""
    args = parse_args()

    env_info = ENV_DEFAULTS[args.env_type]
    reward_image_key = args.reward_image_key or env_info["reward_image_key"]
    action_dim = env_info["action_dim"]

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
    print(f"Creating dummy {args.env_type} env with use_reward_model=True...")
    reward_worker_cfg = {
        "use_reward_model": True,
        "model": {
            "model_type": "resnet",
            "model_path": args.checkpoint_path,
            "arch": args.arch,
            "hidden_dim": args.hidden_dim,
            "dropout": 0.1,
            "image_size": list(args.image_size),
            "normalize": True,
            "precision": "fp32",
        },
    }

    EnvCls = _load_env_class(args.env_type)
    env = EnvCls(
        override_cfg={
            "is_dummy": True,
            "use_reward_model": True,
            "reward_image_key": reward_image_key,
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
        image = np.ascontiguousarray(
            payload.images[idx].permute(1, 2, 0).numpy()
        )  # CHW -> HWC, writable
        obs = {"state": {}, "frames": {reward_image_key: image}}
        reward = _calc_reward(env, obs)
        pos_rewards.append(reward)

    for idx in neg_sampled:
        image = np.ascontiguousarray(
            payload.images[idx].permute(1, 2, 0).numpy()
        )
        obs = {"state": {}, "frames": {reward_image_key: image}}
        reward = _calc_reward(env, obs)
        neg_rewards.append(reward)

    pos_rewards = np.array(pos_rewards)
    neg_rewards = np.array(neg_rewards)

    print(f"  Positive rewards: mean={pos_rewards.mean():.4f}, std={pos_rewards.std():.4f}, min={pos_rewards.min():.4f}, max={pos_rewards.max():.4f}")
    print(f"  Negative rewards: mean={neg_rewards.mean():.4f}, std={neg_rewards.std():.4f}, min={neg_rewards.min():.4f}, max={neg_rewards.max():.4f}")

    pairwise_correct = np.sum(pos_rewards > neg_rewards)
    pairwise_tied = np.sum(pos_rewards == neg_rewards)
    print(f"  Pairwise positive>negative: {pairwise_correct}/{num_samples} ({pairwise_correct/num_samples:.1%})")
    if pairwise_tied:
        print(f"  Pairwise ties: {pairwise_tied}/{num_samples}")

    print("=" * 60)
    if pos_rewards.mean() > neg_rewards.mean() and pairwise_correct > num_samples / 2:
        print("PASS: Positive images receive higher reward on average.")
    else:
        print("FAIL: Reward model does not consistently assign higher reward to positive images.")
        print("   Please check the dataset labels and retrain the reward model.")

    print("=" * 60)
    print("Testing env.reset() and env.step() with dummy observations...")
    obs, info = env.reset()
    print(f"  reset observation keys: {list(obs.keys())}")
    print(f"  frames keys: {list(obs.get('frames', {}).keys())}")

    action = np.zeros(action_dim, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  step reward: {reward:.4f}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("=" * 60)
    print("Dummy verification complete.")


if __name__ == "__main__":
    main()
