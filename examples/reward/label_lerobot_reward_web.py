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
"""Web-based tool for labeling LeRobot episodes as success/failure.

Usage:
    python examples/reward/label_lerobot_reward_web.py \
        --dataset-dir /home/zylab/project/RLinf/datasets/so101_lerobot_data \
        --camera observation.images.left_global \
        --sample-interval 30 \
        --output-dir datasets/so101_reward_images_labeled \
        --port 12345

Controls in the browser:
    ← / → : previous / next image
    S     : mark as success (1) and auto-advance
    F     : mark as failure (0) and auto-advance
    U     : remove label for current image
"""

import argparse
import base64
import json
import shutil
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from rlinf.utils.logging import get_logger

logger = get_logger()
app = Flask(__name__)

LABELS_FILE = "labels.json"

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeRobot Reward Labeler</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a1a;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        #header {
            width: 100%;
            max-width: 1200px;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }
        #progress {
            font-size: 14px;
            color: #aaa;
        }
        #info {
            font-size: 16px;
            font-weight: 600;
        }
        #label-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .label-success { background: #2e7d32; color: #fff; }
        .label-failure { background: #c62828; color: #fff; }
        .label-none { background: #555; color: #ccc; }
        #main {
            flex: 1;
            width: 100%;
            max-width: 1200px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        #image-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            aspect-ratio: 1 / 1;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #aaa;
            display: none;
        }
        #controls {
            margin-top: 24px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.15s;
        }
        .btn:hover { opacity: 0.85; }
        .btn-prev { background: #444; color: #fff; }
        .btn-next { background: #444; color: #fff; }
        .btn-success { background: #2e7d32; color: #fff; }
        .btn-failure { background: #c62828; color: #fff; }
        .btn-unlabel { background: #555; color: #fff; }
        #help {
            margin-top: 16px;
            font-size: 13px;
            color: #888;
            text-align: center;
            line-height: 1.6;
        }
        #toast {
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            border-radius: 6px;
            background: #333;
            color: #fff;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }
        #toast.show { opacity: 1; }
    </style>
</head>
<body>
    <div id="header">
        <div id="progress">Loading...</div>
        <div id="info">Episode - / Frame -</div>
        <div id="label-badge" class="label-none">Unlabeled</div>
    </div>

    <div id="main">
        <div id="image-container">
            <img id="image" src="" alt="Sample">
            <div id="loading">Loading...</div>
        </div>

        <div id="controls">
            <button class="btn btn-prev" onclick="prev()">← Previous</button>
            <button class="btn btn-success" onclick="label(1)">S Success</button>
            <button class="btn btn-failure" onclick="label(0)">F Failure</button>
            <button class="btn btn-unlabel" onclick="unlabel()">U Unlabel</button>
            <button class="btn btn-next" onclick="next()">Next →</button>
        </div>

        <div id="help">
            Keyboard: ← → to navigate, S = success, F = failure, U = unlabel<br>
            Images are auto-saved to success/ or failure/ folder.
        </div>
    </div>

    <div id="toast"></div>

    <script>
        let currentIndex = 0;
        let totalSamples = 0;
        let currentLabel = null;

        async function loadSample(idx) {
            if (idx < 0) idx = 0;
            if (idx >= totalSamples) idx = totalSamples - 1;
            currentIndex = idx;

            document.getElementById('loading').style.display = 'block';
            try {
                const res = await fetch(`/api/sample/${idx}`);
                const data = await res.json();

                document.getElementById('image').src = `data:image/jpeg;base64,${data.image}`;
                document.getElementById('progress').textContent =
                    `Sample ${idx + 1} / ${data.total} | Labeled: ${data.labeled} / ${data.total}`;
                document.getElementById('info').textContent =
                    `Episode ${data.episode_index} / Frame ${data.frame_index}`;

                currentLabel = data.label;
                const badge = document.getElementById('label-badge');
                if (currentLabel === 1) {
                    badge.textContent = 'Success';
                    badge.className = 'label-success';
                } else if (currentLabel === 0) {
                    badge.textContent = 'Failure';
                    badge.className = 'label-failure';
                } else {
                    badge.textContent = 'Unlabeled';
                    badge.className = 'label-none';
                }
            } catch (err) {
                showToast('Failed to load sample: ' + err.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        async function label(value) {
            try {
                const res = await fetch('/api/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index: currentIndex, label: value })
                });
                const data = await res.json();
                if (data.ok) {
                    showToast(value === 1 ? 'Saved as success' : 'Saved as failure');
                    await loadSample(currentIndex);
                    if (currentIndex < totalSamples - 1) {
                        await loadSample(currentIndex + 1);
                    }
                } else {
                    showToast('Error: ' + data.error);
                }
            } catch (err) {
                showToast('Failed to save label: ' + err.message);
            }
        }

        async function unlabel() {
            try {
                const res = await fetch('/api/unlabel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index: currentIndex })
                });
                const data = await res.json();
                if (data.ok) {
                    showToast('Label removed');
                    await loadSample(currentIndex);
                } else {
                    showToast('Error: ' + data.error);
                }
            } catch (err) {
                showToast('Failed to unlabel: ' + err.message);
            }
        }

        function prev() { loadSample(currentIndex - 1); }
        function next() { loadSample(currentIndex + 1); }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 1500);
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') prev();
            else if (e.key === 'ArrowRight') next();
            else if (e.key === 's' || e.key === 'S') label(1);
            else if (e.key === 'f' || e.key === 'F') label(0);
            else if (e.key === 'u' || e.key === 'U') unlabel();
        });

        async function init() {
            const res = await fetch('/api/info');
            const data = await res.json();
            totalSamples = data.total;
            currentIndex = data.current_index;
            await loadSample(currentIndex);
        }

        init();
    </script>
</body>
</html>
"""


class LabelingState:
    """Holds the labeling session state and persists it to disk."""

    def __init__(
        self,
        dataset_dir: Path,
        camera: str,
        sample_interval: int,
        output_dir: Path,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.camera = camera
        self.sample_interval = sample_interval
        self.output_dir = Path(output_dir)
        self.success_dir = self.output_dir / "success"
        self.failure_dir = self.output_dir / "failure"
        self.labels_file = self.output_dir / LABELS_FILE

        self._ensure_output_dir()
        self.samples = self._build_samples()
        self.labels: dict[str, int] = {}
        self.current_index = 0
        self._load_state()

    def _ensure_output_dir(self) -> None:
        """Create output directories. If output_dir already exists, raise."""
        if self.output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {self.output_dir}. "
                "Please delete it manually if you want to start fresh, "
                "or reuse it to resume an existing labeling session."
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.success_dir.mkdir(exist_ok=False)
        self.failure_dir.mkdir(exist_ok=False)
        logger.info(f"Created output directory: {self.output_dir}")

    def _build_samples(self) -> list[dict]:
        """Load parquet files and build a list of uniformly sampled frames."""
        parquet_files = sorted(
            self.dataset_dir.rglob("data/**/*.parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.dataset_dir / 'data'}"
            )

        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        image_col = f"{self.camera}.path"
        if image_col not in df.columns:
            raise KeyError(
                f"Camera column not found: {image_col}. "
                f"Available image columns: "
                f"{[c for c in df.columns if c.endswith('.path')]}."
            )

        # Group by episode and uniformly sample every sample_interval frames.
        samples = []
        for episode_index, group in df.groupby("episode_index"):
            group = group.sort_values("frame_index")
            sampled = group.iloc[:: self.sample_interval]
            for _, row in sampled.iterrows():
                frame_index = int(row["frame_index"])
                image_path = str(row[image_col])
                sample_id = f"episode_{episode_index:03d}_frame_{frame_index:04d}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "episode_index": int(episode_index),
                        "frame_index": frame_index,
                        "image_path": image_path,
                    }
                )

        logger.info(
            f"Loaded {len(samples)} samples from {len(parquet_files)} parquet files "
            f"({df['episode_index'].nunique()} episodes, sample_interval={self.sample_interval})."
        )
        return samples

    def _load_state(self) -> None:
        """Resume from labels.json if it exists."""
        if not self.labels_file.exists():
            return

        with self.labels_file.open("r") as f:
            data = json.load(f)

        loaded_labels = data.get("labels", {})
        self.labels = {str(k): int(v) for k, v in loaded_labels.items()}
        self.current_index = data.get("current_index", 0)

        # Ensure on-disk folders match the loaded labels.
        self._sync_folders_with_labels()

        logger.info(
            f"Resumed labeling session: {len(self.labels)} labeled, "
            f"current_index={self.current_index}."
        )

    def _sync_folders_with_labels(self) -> None:
        """Remove any files in success/failure that are not in labels.json."""
        labeled_ids = set(self.labels.keys())
        for label, folder in [(1, self.success_dir), (0, self.failure_dir)]:
            for path in folder.glob("*.jpg"):
                sample_id = path.stem
                if sample_id not in labeled_ids or self.labels.get(sample_id) != label:
                    logger.warning(f"Removing stale file: {path}")
                    path.unlink()

    def _save_state(self) -> None:
        """Persist current labels and position to labels.json."""
        data = {
            "dataset_dir": str(self.dataset_dir),
            "camera": self.camera,
            "sample_interval": self.sample_interval,
            "current_index": self.current_index,
            "labels": self.labels,
        }
        with self.labels_file.open("w") as f:
            json.dump(data, f, indent=2)

    def get_sample(self, idx: int) -> dict:
        """Return sample metadata and base64-encoded image."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.samples)})")

        sample = self.samples[idx]
        img_path = self.dataset_dir / sample["image_path"]
        img = Image.open(img_path).convert("RGB")

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "sample_id": sample["sample_id"],
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
            "index": idx,
            "total": len(self.samples),
            "labeled": len(self.labels),
            "image": img_b64,
            "label": self.labels.get(sample["sample_id"], None),
        }

    def set_label(self, idx: int, label: int) -> None:
        """Label a sample, copy image to success/failure, and persist state."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")

        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        old_label = self.labels.get(sample_id)

        # Remove old copy if label changed.
        if old_label is not None and old_label != label:
            old_dir = self.success_dir if old_label == 1 else self.failure_dir
            old_file = old_dir / f"{sample_id}.jpg"
            if old_file.exists():
                old_file.unlink()

        # Copy image to the target folder.
        src = self.dataset_dir / sample["image_path"]
        dst_dir = self.success_dir if label == 1 else self.failure_dir
        dst = dst_dir / f"{sample_id}.jpg"
        shutil.copy2(src, dst)

        self.labels[sample_id] = label
        self.current_index = idx
        self._save_state()

    def unlabel(self, idx: int) -> None:
        """Remove the label for a sample and delete its copied image."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")

        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        if sample_id not in self.labels:
            return

        old_label = self.labels.pop(sample_id)
        old_dir = self.success_dir if old_label == 1 else self.failure_dir
        old_file = old_dir / f"{sample_id}.jpg"
        if old_file.exists():
            old_file.unlink()

        self.current_index = idx
        self._save_state()

    def get_info(self) -> dict:
        """Return session progress info."""
        return {
            "total": len(self.samples),
            "labeled": len(self.labels),
            "unlabeled": len(self.samples) - len(self.labels),
            "current_index": self.current_index,
            "dataset_dir": str(self.dataset_dir),
            "camera": self.camera,
            "sample_interval": self.sample_interval,
        }


# Global state set by main().
state: LabelingState | None = None


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/info")
def api_info():
    return jsonify(state.get_info())


@app.route("/api/sample/<int:idx>")
def api_sample(idx: int):
    try:
        return jsonify(state.get_sample(idx))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json(force=True)
    idx = data.get("index")
    label = data.get("label")
    if idx is None or label is None:
        return jsonify({"ok": False, "error": "Missing index or label"}), 400
    try:
        state.set_label(int(idx), int(label))
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"Failed to label sample: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/unlabel", methods=["POST"])
def api_unlabel():
    data = request.get_json(force=True)
    idx = data.get("index")
    if idx is None:
        return jsonify({"ok": False, "error": "Missing index"}), 400
    try:
        state.unlabel(int(idx))
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"Failed to unlabel sample: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web UI for labeling LeRobot episodes as success/failure."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/zylab/project/RLinf/datasets/so101_lerobot_data",
        help="Path to the LeRobot dataset directory.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="observation.images.left_global",
        help="Camera key to display and label.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=30,
        help="Sample one frame every N frames within each episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/so101_reward_images_labeled",
        help="Directory to save success/ and failure/ folders and labels.json.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="Port to bind the web server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global state
    state = LabelingState(
        dataset_dir=Path(args.dataset_dir),
        camera=args.camera,
        sample_interval=args.sample_interval,
        output_dir=Path(args.output_dir),
    )

    logger.info(
        f"Starting web server at http://{args.host}:{args.port}. "
        f"Open this URL in your browser to start labeling."
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
