"""Smoke test for the human reward server + HumanRewardModel.

Start the server first:
    .venv/bin/python tmp/human_reward_server.py --port 12345

Then in another terminal run this script and click Success/Failure in the browser.
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.reward.human_reward_model import HumanRewardModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="tmp/vlm_debug_frame.jpg",
        help="Path to an image to submit for scoring.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:12345",
        help="Human reward server URL.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Place all objects into the box to clean the table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = Image.open(args.image).convert("RGB").resize((224, 224))
    images = np.array(img)[None, ...].astype(np.uint8)  # (1, 224, 224, 3)
    observations = {"main_images": images}

    cfg = OmegaConf.create(
        {
            "model_type": "human",
            "human_reward_url": args.url,
            "task_description": args.task,
            "timeout": 600.0,
            "server_wait_timeout": 60.0,
            "default_reward": 0.0,
        }
    )

    print(f"Submitting {args.image} to human reward server at {args.url} ...")
    print("Please click Success or Failure in the browser.")

    model = HumanRewardModel(cfg)
    with torch.no_grad():
        rewards = model.compute_reward(observations)

    print(f"\nReceived reward: {rewards.tolist()}")


if __name__ == "__main__":
    main()
