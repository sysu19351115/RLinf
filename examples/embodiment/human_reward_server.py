"""Standalone human reward server for SO101 real-world RL.

Usage:
    .venv/bin/python tmp/human_reward_server.py --port 12345 --log_dir logs/human_rewards

Open http://<host>:12345 in a browser. When an episode ends, the page shows the
final frame and two buttons: Success (1.0) / Failure (0.0). The reward is
returned to the RLinf reward worker via long-polling on /get_reward.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Shared state protected by _lock.
_state = {
    "episode_id": None,
    "task": "",
    "image_base64": "",
    "reward": 0.0,
    "reward_set": False,
    "submitted_at": 0.0,
}
_lock = threading.Lock()
_condition = threading.Condition(_lock)

args: argparse.Namespace


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <title>Human Reward</title>
  <style>
    body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
    .container { max-width: 800px; margin: auto; background: #fff; padding: 20px;
                 border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    .info { margin: 10px 0; color: #555; }
    .buttons { margin-top: 20px; }
    button { font-size: 20px; padding: 15px 40px; margin-right: 20px;
             border: none; border-radius: 6px; cursor: pointer; }
    .success { background: #4caf50; color: white; }
    .failure { background: #f44336; color: white; }
    .pending { color: #888; font-style: italic; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Episode Reward</h1>
    {% if episode_id %}
      <div class="info"><strong>Episode:</strong> {{ episode_id }}</div>
      <div class="info"><strong>Task:</strong> {{ task }}</div>
      <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="episode frame">
      <div class="buttons">
        <form method="post" action="/reward" style="display:inline;">
          <input type="hidden" name="episode_id" value="{{ episode_id }}">
          <input type="hidden" name="reward" value="1.0">
          <button type="submit" class="success">Success (1.0)</button>
        </form>
        <form method="post" action="/reward" style="display:inline;">
          <input type="hidden" name="episode_id" value="{{ episode_id }}">
          <input type="hidden" name="reward" value="0.0">
          <button type="submit" class="failure">Failure (0.0)</button>
        </form>
      </div>
    {% else %}
      <p class="pending">Waiting for the next episode...</p>
    {% endif %}
  </div>
</body>
</html>
"""


def _log_reward(
    episode_id: str,
    task: str,
    image_base64: str,
    reward: float,
    source: str,
) -> None:
    """Persist the scored episode image and metadata."""
    if not args.log_dir:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.log_dir) / f"{ts}_{episode_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = base64.b64decode(image_base64)
        (out_dir / "frame.jpg").write_bytes(img_bytes)
    except Exception:
        pass

    meta = {
        "episode_id": episode_id,
        "task": task,
        "reward": reward,
        "source": source,
        "timestamp": time.time(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


@app.route("/", methods=["GET"])
def index():
    with _lock:
        return render_template_string(
            INDEX_HTML,
            episode_id=_state["episode_id"],
            task=_state["task"],
            image_base64=_state["image_base64"],
        )


@app.route("/current_episode", methods=["GET"])
def current_episode():
    with _lock:
        return jsonify(
            {
                "episode_id": _state["episode_id"],
                "task": _state["task"],
                "pending": _state["episode_id"] is not None and not _state["reward_set"],
            }
        )


@app.route("/submit_episode", methods=["POST"])
def submit_episode():
    data: dict[str, Any] = request.get_json(force=True, silent=True) or {}
    episode_id = str(data.get("episode_id", ""))
    task = str(data.get("task", ""))
    image_base64 = str(data.get("image_base64", ""))

    if not episode_id or not image_base64:
        return jsonify({"error": "episode_id and image_base64 are required"}), 400

    with _condition:
        # If the previous episode was never scored, log it as timeout/default.
        if _state["episode_id"] is not None and not _state["reward_set"]:
            _log_reward(
                episode_id=_state["episode_id"],
                task=_state["task"],
                image_base64=_state["image_base64"],
                reward=args.default_reward,
                source="timeout_or_overwrite",
            )

        _state["episode_id"] = episode_id
        _state["task"] = task
        _state["image_base64"] = image_base64
        _state["reward"] = args.default_reward
        _state["reward_set"] = False
        _state["submitted_at"] = time.time()
        _condition.notify_all()

    return jsonify({"status": "ok", "episode_id": episode_id})


@app.route("/reward", methods=["POST"])
def reward():
    episode_id = request.form.get("episode_id") or request.json.get("episode_id")
    try:
        reward_value = float(request.form.get("reward") or request.json.get("reward"))
    except (TypeError, ValueError):
        return jsonify({"error": "reward must be a float"}), 400

    with _condition:
        if _state["episode_id"] is None:
            return jsonify({"error": "no pending episode"}), 409
        if episode_id != _state["episode_id"]:
            return jsonify({"error": "episode_id mismatch"}), 409

        _state["reward"] = reward_value
        _state["reward_set"] = True
        _log_reward(
            episode_id=_state["episode_id"],
            task=_state["task"],
            image_base64=_state["image_base64"],
            reward=reward_value,
            source="human",
        )
        _condition.notify_all()

    return jsonify({"status": "ok", "reward": reward_value})


@app.route("/get_reward", methods=["GET"])
def get_reward():
    episode_id = request.args.get("episode_id")
    deadline = time.time() + args.wait_timeout

    with _condition:
        while True:
            if _state["reward_set"] and _state["episode_id"] == episode_id:
                return jsonify({"reward": _state["reward"], "source": "human"})

            if _state["episode_id"] != episode_id:
                # The episode we are waiting for was overwritten or never existed.
                return jsonify(
                    {"reward": args.default_reward, "source": "stale_or_missing"}
                )

            remaining = deadline - time.time()
            if remaining <= 0:
                return jsonify(
                    {"reward": args.default_reward, "source": "timeout"}
                )

            _condition.wait(timeout=min(remaining, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/human_rewards",
        help="Directory to save scored episode frames and metadata.",
    )
    parser.add_argument(
        "--default_reward",
        type=float,
        default=0.0,
        help="Reward returned when the human does not respond in time.",
    )
    parser.add_argument(
        "--wait_timeout",
        type=float,
        default=60.0,
        help="Maximum seconds /get_reward waits before returning default_reward.",
    )
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    print(f"Human reward server running at http://{args.host}:{args.port}")
    print(f"Logs will be saved to: {args.log_dir}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
