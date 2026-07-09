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

"""Collect success/failure images from a USB camera for reward model training.

Usage (with GUI preview):
    python examples/reward/collect_reward_images.py \
        --camera /dev/video0 \
        --output-dir datasets/so101_reward_images \
        --width 640 --height 480

Usage (headless / no display):
    python examples/reward/collect_reward_images.py \
        --camera /dev/video0 \
        --output-dir datasets/so101_reward_images \
        --headless

Controls:
    p : capture the current frame and save it to the active folder
    q : toggle active folder between ``success`` and ``failure``
    ESC / e : exit
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rlinf.utils.logging import get_logger

logger = get_logger()


class _FrameBuffer:
    """Thread-safe buffer for the latest captured frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect success/failure images for reward model training."
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="/dev/video0",
        help="Camera device path or index (default: /dev/video0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/so101_reward_images",
        help="Root directory that will contain success/ and failure/ subfolders.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height (default: 480).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Capture FPS (default: 30).",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="FOURCC codec string (default: MJPG).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a GUI preview (useful over SSH or on headless machines).",
    )
    return parser.parse_args()


def _open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    """Open the camera with the requested settings."""
    try:
        camera_index = int(args.camera)
    except ValueError:
        camera_index = args.camera

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(
        f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps "
        f"(requested {args.width}x{args.height} @ {args.fps} fps)"
    )
    return cap


def _draw_overlay(frame: np.ndarray, active_folder: str, counts: dict[str, int]) -> None:
    """Draw the current mode and capture counts on the preview frame."""
    text_lines = [
        f"Target: {active_folder.upper()}",
        f"Success: {counts['success']}  Failure: {counts['failure']}",
        "p = capture  |  q = toggle folder  |  ESC / e = exit",
    ]
    y0 = 30
    dy = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _save_frame(
    frame: np.ndarray,
    output_root: Path,
    active_folder: str,
    counts: dict[str, int],
) -> Path:
    """Save a frame to the active folder and update counts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{active_folder}_{timestamp}.png"
    save_path = output_root / active_folder / filename
    cv2.imwrite(str(save_path), frame)
    counts[active_folder] += 1
    return save_path


def _capture_loop(cap: cv2.VideoCapture, buffer: _FrameBuffer, running: threading.Event) -> None:
    """Continuously grab frames in the background."""
    while running.is_set():
        ret, frame = cap.read()
        if ret and frame is not None:
            buffer.update(frame)
        time.sleep(0.01)


def _enable_raw_terminal() -> Optional[list]:
    """Switch stdin to cbreak mode so single keystrokes are readable."""
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return old_settings
    except Exception as exc:  # pragma: no cover - non-terminal fallback
        logger.warning(f"Could not enable raw terminal mode: {exc}")
        return None


def _restore_terminal(old_settings: Optional[list]) -> None:
    """Restore the previous terminal settings."""
    if old_settings is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not restore terminal settings: {exc}")


def _read_char(timeout: float = 0.05) -> Optional[str]:
    """Read a single character from stdin without blocking."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def _run_headless(
    cap: cv2.VideoCapture,
    output_root: Path,
    counts: dict[str, int],
) -> None:
    """Run the collection loop without a GUI preview."""
    buffer = _FrameBuffer()
    running = threading.Event()
    running.set()

    capture_thread = threading.Thread(
        target=_capture_loop, args=(cap, buffer, running), daemon=True
    )
    capture_thread.start()

    active_folder = "success"
    old_settings = _enable_raw_terminal()
    last_status_time = time.time()

    logger.info(
        "Headless mode started. Controls: p=capture, q=toggle folder, ESC/e=exit"
    )

    try:
        while True:
            char = _read_char(0.05)
            if char is not None:
                if char == "p":
                    frame = buffer.get()
                    if frame is None:
                        logger.warning("No frame captured yet; try again.")
                    else:
                        save_path = _save_frame(frame, output_root, active_folder, counts)
                        logger.info(f"Saved {save_path}")
                elif char == "q":
                    active_folder = "failure" if active_folder == "success" else "success"
                    logger.info(f"Switched target folder to: {active_folder}")
                elif char in ("e", "\x1b", "\x03"):  # e, ESC, Ctrl+C
                    logger.info("Exit requested.")
                    break

            if time.time() - last_status_time >= 2.0:
                logger.info(
                    f"Status: target={active_folder} | "
                    f"success={counts['success']} failure={counts['failure']}"
                )
                last_status_time = time.time()
    finally:
        running.clear()
        capture_thread.join(timeout=1.0)
        _restore_terminal(old_settings)
        cap.release()


def _run_gui(
    cap: cv2.VideoCapture,
    output_root: Path,
    counts: dict[str, int],
) -> bool:
    """Run the collection loop with an OpenCV preview window.

    Returns:
        True if the loop exited normally, False if the display is unavailable.
    """
    active_folder = "success"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera; retrying...")
                continue

            display = frame.copy()
            _draw_overlay(display, active_folder, counts)
            cv2.imshow("Reward image collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                save_path = _save_frame(frame, output_root, active_folder, counts)
                logger.info(f"Saved {save_path}")
            elif key == ord("q"):
                active_folder = "failure" if active_folder == "success" else "success"
                logger.info(f"Switched target folder to: {active_folder}")
            elif key == 27 or key == ord("e"):  # ESC or 'e'
                logger.info("Exit requested.")
                break
    except cv2.error as exc:
        logger.warning(
            f"OpenCV GUI failed (display likely unavailable): {exc}. "
            "Falling back to headless mode."
        )
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def main() -> None:
    """Run the interactive image collection loop."""
    args = parse_args()
    output_root = Path(args.output_dir)
    success_dir = output_root / "success"
    failure_dir = output_root / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_capture(args)

    counts = {
        "success": len(list(success_dir.glob("*"))),
        "failure": len(list(failure_dir.glob("*"))),
    }

    logger.info(
        f"Saving images to {output_root}. "
        "Controls: p=capture, q=toggle folder, ESC/e=exit"
    )

    if args.headless:
        _run_headless(cap, output_root, counts)
    else:
        gui_ok = _run_gui(cap, output_root, counts)
        if not gui_ok:
            # Re-open the camera for headless fallback.
            cap = _open_capture(args)
            _run_headless(cap, output_root, counts)

    logger.info(
        f"Collection finished. Totals: success={counts['success']}, "
        f"failure={counts['failure']}"
    )


if __name__ == "__main__":
    main()
