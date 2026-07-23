# Dobot USB Camera Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Dobot use the already-proven SO101/LeRobot OpenCV USB-camera implementation with the RYS camera at a stable V4L2 path, while preserving the existing Dobot image and threading contracts.

**Architecture:** Add a thin `BaseCamera` adapter around LeRobot's `OpenCVCamera`; do not reimplement V4L2 capture and do not modify SO101. The adapter returns BGR frames because `DobotEnv._get_camera_frames()` already converts BGR to RGB. Add explicit camera resolution/FPS configuration and configure the single camera as the logical `cam_left_wrist` input used by the Dobot checkpoint.

**Tech Stack:** Python 3.11, OpenCV/V4L2, LeRobot `OpenCVCamera`, Gymnasium, OmegaConf/Hydra, pytest.

---

## Confirmed baseline

- Robot workspace: `/home/zylab/project/RLinf`
- Camera device: `/dev/video6`
- Stable device path:
  `/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0`
- `/dev/video7` is the second interface of the same physical camera; do not configure it as a second camera.
- The device advertises MJPEG and YUYV422.
- A direct LeRobot test has already succeeded at `640x480`, 30 FPS, HWC RGB `uint8`.
- SO101 uses `lerobot.common.robot_devices.cameras.opencv.OpenCVCamera`.
- SO101's YAML `fourcc: MJPG` is currently ignored by `SO101Controller`; the installed `OpenCVCameraConfig` has no `fourcc` field. Do not add FOURCC support in this task.
- The remote worktree already contains many modified/untracked Dobot files. Never run `git reset`, `git checkout --`, `git clean`, or `git add -A`. Stage only the exact files owned by this task, and only after the maintainer confirms the current Dobot baseline is ready to commit.

## Non-goals

- Do not change SO101 camera behavior.
- Do not modify `LumosCamera`; it is a special 1280x1280 YU12 backend.
- Do not change OpenPI image preprocessing or model weights.
- Do not introduce depth capture, automatic device discovery, multi-camera synchronization, or FOURCC selection.
- Do not connect, enable, reset, or move the Dobot during camera-only verification.

## Acceptance criteria

1. `create_camera()` accepts `camera_type: opencv` and returns the adapter.
2. The adapter opens the stable RYS path through LeRobot and produces `640x480x3` BGR `uint8` frames.
3. With one camera and `enable_high_camera: false`, Dobot names the frame `cam_left_wrist`.
4. Dobot converts the BGR frame to `224x224x3` RGB exactly once.
5. Train/eval reward configuration references an existing frame key.
6. Existing Dobot unit tests remain green.
7. Camera-only hardware verification passes without initializing the arm.

---

### Task 0: Protect the existing dirty worktree

**Files:** None.

**Step 1: Record the current state**

Run:

```bash
cd /home/zylab/project/RLinf
git status --short
git diff -- rlinf/envs/realworld/common/camera
```

Expected: existing Dobot work is untracked/modified; no USB-camera adapter exists.

**Step 2: Confirm ownership before editing**

Ask the maintainer whether the current Dobot baseline has been saved elsewhere. Do not discard or overwrite any pre-existing edits.

**Step 3: Use an isolated branch if the baseline permits it**

```bash
git switch -c feat/dobot-opencv-usb-camera
```

Expected: a new branch is created without changing working-tree contents. If the maintainer does not want a branch while the baseline is dirty, continue without committing and provide a path-scoped diff at handoff.

---

### Task 1: Write failing tests for the LeRobot adapter

**Files:**

- Create: `tests/unit_tests/test_opencv_usb_camera.py`
- Future create: `rlinf/envs/realworld/common/camera/opencv_camera.py`

**Step 1: Add adapter contract tests**

The tests must use a fake LeRobot camera and must never open real hardware:

```python
import numpy as np
import pytest

import rlinf.envs.realworld.common.camera.opencv_camera as camera_module
from rlinf.envs.realworld.common.camera.base_camera import CameraInfo


class FakeOpenCVCameraConfig:
    def __init__(
        self,
        *,
        camera_index,
        width,
        height,
        fps,
        color_mode,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.color_mode = color_mode


class FakeLeRobotOpenCVCamera:
    instances = []

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.frame = np.full((480, 640, 3), [10, 20, 30], dtype=np.uint8)
        type(self).instances.append(self)

    def connect(self):
        self.is_connected = True

    def read(self):
        return self.frame.copy()

    def disconnect(self):
        self.is_connected = False


@pytest.fixture(autouse=True)
def fake_lerobot_camera(monkeypatch):
    FakeLeRobotOpenCVCamera.instances.clear()
    monkeypatch.setattr(
        camera_module,
        "OpenCVCameraConfig",
        FakeOpenCVCameraConfig,
    )
    monkeypatch.setattr(
        camera_module,
        "LeRobotOpenCVCamera",
        FakeLeRobotOpenCVCamera,
    )


def test_adapter_passes_path_resolution_fps_and_bgr_contract():
    info = CameraInfo(
        name="cam_left_wrist",
        serial_number="/dev/video6",
        camera_type="opencv",
        resolution=(640, 480),
        fps=30,
    )

    camera = camera_module.OpenCVUSBCamera(info)
    device = FakeLeRobotOpenCVCamera.instances[0]

    assert device.is_connected
    assert device.config.camera_index == "/dev/video6"
    assert device.config.width == 640
    assert device.config.height == 480
    assert device.config.fps == 30
    assert device.config.color_mode == "bgr"

    ok, frame = camera._read_frame()
    assert ok is True
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [10, 20, 30]

    camera._close_device()
    assert not device.is_connected


def test_close_is_idempotent_after_disconnect():
    info = CameraInfo(
        name="cam_left_wrist",
        serial_number="/dev/video6",
        camera_type="opencv",
    )
    camera = camera_module.OpenCVUSBCamera(info)
    camera._close_device()
    camera._close_device()
```

**Step 2: Run the test and verify failure**

```bash
cd /home/zylab/project/RLinf
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py -q
```

Expected: FAIL because `opencv_camera.py` or `OpenCVUSBCamera` does not exist.

---

### Task 2: Implement the thin LeRobot-to-BaseCamera adapter

**Files:**

- Create: `rlinf/envs/realworld/common/camera/opencv_camera.py`
- Modify: `rlinf/envs/realworld/common/camera/__init__.py:15-48`
- Test: `tests/unit_tests/test_opencv_usb_camera.py`

**Step 1: Implement the adapter**

Use the existing LeRobot capture implementation and the existing RLinf queue/thread implementation. Do not call `async_read()`: `BaseCamera` already owns the capture thread, and using both would create two competing background threads.

```python
"""Generic USB/V4L2 camera backed by LeRobot's OpenCVCamera."""

from typing import Optional

import numpy as np
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import (
    OpenCVCamera as LeRobotOpenCVCamera,
)

from .base_camera import BaseCamera, CameraInfo


class OpenCVUSBCamera(BaseCamera):
    """Adapt LeRobot OpenCVCamera to RLinf's BaseCamera interface.

    Frames intentionally remain BGR. DobotEnv performs the single BGR-to-RGB
    conversion after crop/resize.
    """

    def __init__(self, camera_info: CameraInfo):
        super().__init__(camera_info)
        config = OpenCVCameraConfig(
            camera_index=camera_info.serial_number,
            width=int(camera_info.resolution[0]),
            height=int(camera_info.resolution[1]),
            fps=int(camera_info.fps),
            color_mode="bgr",
        )
        self._device = LeRobotOpenCVCamera(config)
        self._device.connect()

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        return True, self._device.read()

    def _close_device(self) -> None:
        if self._device.is_connected:
            self._device.disconnect()
```

The imports remain inside the lazily imported backend module, so cloud-only processes do not instantiate or connect the camera.

**Step 2: Register aliases in the camera factory**

In `rlinf/envs/realworld/common/camera/__init__.py`, add:

```python
if camera_type in ("opencv", "usb", "v4l2"):
    from .opencv_camera import OpenCVUSBCamera

    return OpenCVUSBCamera(camera_info)
```

Update the docstring and unsupported-type error to include `opencv`, `usb`, and `v4l2`. Do not import the adapter eagerly at module import time.

**Step 3: Add factory alias tests**

Append to the test file:

```python
@pytest.mark.parametrize("camera_type", ["opencv", "usb", "v4l2"])
def test_factory_accepts_usb_camera_aliases(camera_type):
    from rlinf.envs.realworld.common.camera import create_camera

    camera = create_camera(
        CameraInfo(
            name="cam_left_wrist",
            serial_number="/dev/video6",
            camera_type=camera_type,
            resolution=(640, 480),
            fps=30,
        )
    )
    try:
        assert isinstance(camera, camera_module.OpenCVUSBCamera)
    finally:
        camera._close_device()
```

**Step 4: Run focused tests**

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py -q
```

Expected: all tests PASS without touching `/dev/video6`.

**Step 5: Commit only owned files, if baseline ownership is resolved**

```bash
git add \
  rlinf/envs/realworld/common/camera/opencv_camera.py \
  rlinf/envs/realworld/common/camera/__init__.py \
  tests/unit_tests/test_opencv_usb_camera.py
git commit -m "feat(camera): adapt LeRobot OpenCV camera for RLinf"
```

---

### Task 3: Propagate USB camera resolution and FPS through Dobot configuration

**Files:**

- Modify: `rlinf/scheduler/hardware/robots/dobot.py:157-164`
- Modify: `rlinf/envs/realworld/dobot/dobot_env.py:103-115`
- Modify: `rlinf/envs/realworld/dobot/dobot_env.py:334-346`
- Modify: `rlinf/envs/realworld/dobot/dobot_env.py:812-835`
- Test: `tests/unit_tests/test_opencv_usb_camera.py`

**Step 1: Write a failing propagation test**

Append:

```python
from rlinf.envs.realworld.dobot import dobot_env as dobot_env_module
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig


class FakeRLinfCamera:
    def __init__(self, info):
        self.info = info
        self.opened = False

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False


def test_dobot_passes_camera_settings_and_maps_single_camera_to_main(monkeypatch):
    created = []

    def fake_create_camera(info):
        camera = FakeRLinfCamera(info)
        created.append(camera)
        return camera

    monkeypatch.setattr(dobot_env_module, "create_camera", fake_create_camera)

    env = DobotEnv.__new__(DobotEnv)
    env.config = DobotRobotConfig(
        camera_serials=["/dev/video6"],
        camera_type="opencv",
        camera_resolution=(640, 480),
        camera_fps=30,
        enable_high_camera=False,
        is_dummy=False,
    )

    env._open_cameras()

    assert len(created) == 1
    assert created[0].opened
    assert created[0].info.name == "cam_left_wrist"
    assert created[0].info.serial_number == "/dev/video6"
    assert created[0].info.camera_type == "opencv"
    assert created[0].info.resolution == (640, 480)
    assert created[0].info.fps == 30
```

**Step 2: Verify the test fails**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py::test_dobot_passes_camera_settings_and_maps_single_camera_to_main \
  -q
```

Expected: FAIL because `DobotRobotConfig` lacks `camera_resolution` and `camera_fps`.

**Step 3: Add fields to both Dobot configuration classes**

Add to both `DobotConfig` and `DobotRobotConfig`:

```python
camera_resolution: tuple[int, int] = (640, 480)
"""Requested USB/color stream resolution as ``(width, height)``."""

camera_fps: int = 30
"""Requested color stream frame rate."""
```

Update `camera_type` documentation to list `opencv`, `usb`, and `v4l2`.

**Step 4: Add validation**

In `DobotEnv._validate_config()` validate after all Hydra/hardware overrides:

```python
resolution = tuple(int(v) for v in c.camera_resolution)
if len(resolution) != 2 or any(v <= 0 for v in resolution):
    raise ValueError(
        f"camera_resolution must contain two positive integers, got {c.camera_resolution!r}"
    )
if int(c.camera_fps) <= 0:
    raise ValueError(f"camera_fps must be positive, got {c.camera_fps!r}")
```

Add focused invalid-resolution and invalid-FPS tests. Do not add automatic fallback to another resolution; silent fallback would hide data-distribution changes.

**Step 5: Merge hardware values into environment config**

Add these keys to `DobotEnv._merge_hardware_info()`:

```python
"camera_resolution",
"camera_fps",
```

**Step 6: Pass fields to `CameraInfo`**

Change `_open_cameras()`:

```python
info = CameraInfo(
    name=cam_name,
    serial_number=serial,
    camera_type=camera_type,
    resolution=tuple(int(v) for v in self.config.camera_resolution),
    fps=int(self.config.camera_fps),
)
```

**Step 7: Run focused and existing Dobot tests**

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py \
  tests/unit_tests/test_dobot_reward_and_dummy.py \
  -q
```

Expected: all tests PASS.

**Step 8: Commit only owned paths, if permitted**

```bash
git add \
  rlinf/scheduler/hardware/robots/dobot.py \
  rlinf/envs/realworld/dobot/dobot_env.py \
  tests/unit_tests/test_opencv_usb_camera.py
git commit -m "feat(dobot): propagate USB camera stream settings"
```

---

### Task 4: Lock the BGR-to-RGB contract with a regression test

**Files:**

- Modify: `tests/unit_tests/test_opencv_usb_camera.py`
- Verify only: `rlinf/envs/realworld/dobot/dobot_env.py:842-873`

**Step 1: Add a deterministic color test**

Create a uniform BGR test image with pixel `[10, 20, 30]`. Invoke Dobot's frame processing with a fake camera and assert the policy frame is RGB `[30, 20, 10]`, shape `(224, 224, 3)`, dtype `uint8`.

Use a minimal fake player:

```python
class FakePlayer:
    def put_frame(self, frames):
        self.frames = frames
```

Construct the environment with `DobotEnv.__new__`, set:

```python
env.observation_space = gym.spaces.Dict(
    {
        "frames": gym.spaces.Dict(
            {
                "cam_left_wrist": gym.spaces.Box(
                    0, 255, shape=(224, 224, 3), dtype=np.uint8
                )
            }
        )
    }
)
env.camera_player = FakePlayer()
```

The fake camera must expose `_camera_info.name` and `get_frame()`.

**Step 2: Run the color test**

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py -q
```

Expected: PASS and exactly one channel reversal.

**Step 3: Do not modify production color processing unless the test reveals a real mismatch**

The intended contract is:

```text
LeRobot adapter BGR
    → Dobot center crop/resize
    → Dobot [..., ::-1]
    → OpenPI RGB
```

**Step 4: Commit the test, if permitted**

```bash
git add tests/unit_tests/test_opencv_usb_camera.py
git commit -m "test(dobot): lock USB camera RGB conversion"
```

---

### Task 5: Configure the one-camera Dobot deployment

**Files:**

- Modify: `examples/embodiment/config/dobot_async_ppo_pi05.yaml:35-44`
- Modify: `examples/embodiment/config/dobot_async_ppo_pi05.yaml:104-109`
- Modify: `examples/embodiment/config/dobot_async_ppo_pi05.yaml:131-136`
- Modify: `examples/embodiment/config/env/realworld_dobot.yaml:12`
- Modify: `examples/embodiment/config/env/realworld_dobot.yaml:32-36`

**Step 1: Change robot hardware camera configuration**

Use the stable path:

```yaml
camera_serials:
  - "/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0"
camera_type: "opencv"
camera_resolution: [640, 480]
camera_fps: 30
```

Do not add `/dev/video7`.

**Step 2: Configure a single logical main camera**

In `realworld_dobot.yaml`:

```yaml
main_image_key: cam_left_wrist

init_params:
  camera_serials:
    - "/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0"
  camera_type: "opencv"
  camera_resolution: [640, 480]
  camera_fps: 30
  enable_high_camera: false
```

`cam_left_wrist` is a logical policy slot. This mapping is required because `DobotDataConfig` trained the checkpoint with `observation.images.cam_left_wrist`.

**Step 3: Point human reward at an existing frame**

In both train and eval:

```yaml
reward_image_key: "cam_left_wrist"
```

Leaving `cam_high` would cause a missing-frame-key error at episode end.

**Step 4: Resolve and inspect Hydra configuration without starting workers**

Use the project's supported Hydra print mode if available:

```bash
.venv/bin/python examples/embodiment/train_async.py \
  --config-name dobot_async_ppo_pi05 \
  --cfg job
```

Expected resolved values:

```text
camera_type: opencv
camera_resolution: [640, 480]
camera_fps: 30
enable_high_camera: false
main_image_key: cam_left_wrist
reward_image_key: cam_left_wrist
```

If this entry point does not support `--cfg job`, use a small `OmegaConf` composition test; do not start Ray or the arm merely to inspect configuration.

**Step 5: Commit configuration, if permitted**

```bash
git add \
  examples/embodiment/config/dobot_async_ppo_pi05.yaml \
  examples/embodiment/config/env/realworld_dobot.yaml
git commit -m "config(dobot): use single RYS USB camera"
```

---

### Task 6: Update the Dobot reproduction document

**Files:**

- Modify: `docs/examples/dobot_pi05_ppo.md`

**Step 1: Replace RealSense-only instructions**

Document:

- stable by-id device path;
- the `opencv` camera type;
- `640x480@30`;
- single-camera mapping to `cam_left_wrist`;
- `reward_image_key: cam_left_wrist`;
- `/dev/video7` is not a second physical camera.

**Step 2: Add permission checks**

```bash
ls -l /dev/video6
id -nG
```

The runtime user must have access through the `video` group or an equivalent ACL. Do not recommend `chmod 777`.

**Step 3: Add camera-only verification**

Use the command from Task 7 below. Clearly state that it does not initialize the Dobot.

**Step 4: Commit documentation, if permitted**

```bash
git add docs/examples/dobot_pi05_ppo.md
git commit -m "docs(dobot): document RYS USB camera setup"
```

---

### Task 7: Run camera-only hardware verification

**Files:** None.

**Step 1: Confirm no process owns the camera**

```bash
fuser /dev/video6
```

Expected: no PID output. If occupied, stop and identify the owner; do not kill an unknown process.

**Step 2: Exercise the new adapter without the robot**

```bash
cd /home/zylab/project/RLinf
PYTHONPATH=. .venv/bin/python - <<'PY'
from rlinf.envs.realworld.common.camera import CameraInfo, create_camera

info = CameraInfo(
    name="cam_left_wrist",
    serial_number="/dev/v4l/by-id/usb-RYS_CAMERA071101_2026071101-video-index0",
    camera_type="opencv",
    resolution=(640, 480),
    fps=30,
)
camera = create_camera(info)
try:
    camera.open()
    frame = camera.get_frame(timeout=5)
    print("shape:", frame.shape)
    print("dtype:", frame.dtype)
    print("range:", int(frame.min()), int(frame.max()))
finally:
    camera.close()
PY
```

Expected:

```text
shape: (480, 640, 3)
dtype: uint8
range: 0 255
```

The precise minimum/maximum can vary with the scene, but the frame must not be empty or constant.

**Step 3: Repeat open/close three times**

Run the command three times. Expected: no `device busy`, hanging thread, or disconnect exception.

**Step 4: Stop on any camera failure**

Do not continue to a real environment if:

- the path cannot be opened;
- requested width/height/FPS differs;
- `get_frame()` times out;
- repeated open/close leaks the device;
- colors are visibly swapped.

---

### Task 8: Run regression and handoff checks

**Files:** All task-owned files.

**Step 1: Run formatting/linting used by the repository**

Use the existing project commands. At minimum:

```bash
.venv/bin/ruff check \
  rlinf/envs/realworld/common/camera/opencv_camera.py \
  rlinf/envs/realworld/common/camera/__init__.py \
  rlinf/envs/realworld/dobot/dobot_env.py \
  rlinf/scheduler/hardware/robots/dobot.py \
  tests/unit_tests/test_opencv_usb_camera.py
```

Expected: no new violations.

**Step 2: Run the focused regression suite**

```bash
PYTHONPATH=. .venv/bin/pytest \
  tests/unit_tests/test_opencv_usb_camera.py \
  tests/unit_tests/test_dobot_reward_and_dummy.py \
  tests/unit_tests/test_realworld_install_metadata.py \
  -q
```

Expected: all tests PASS.

**Step 3: Review only scoped changes**

```bash
git diff -- \
  rlinf/envs/realworld/common/camera/opencv_camera.py \
  rlinf/envs/realworld/common/camera/__init__.py \
  rlinf/envs/realworld/dobot/dobot_env.py \
  rlinf/scheduler/hardware/robots/dobot.py \
  examples/embodiment/config/dobot_async_ppo_pi05.yaml \
  examples/embodiment/config/env/realworld_dobot.yaml \
  tests/unit_tests/test_opencv_usb_camera.py \
  docs/examples/dobot_pi05_ppo.md
```

Expected: no SO101, Lumos, model, reward-worker, or robot-control changes.

**Step 4: Produce a handoff report**

Report:

- exact files changed;
- unit-test results;
- camera-only test results;
- resolved device path and actual width/height/FPS;
- confirmation that no arm motion was performed;
- any remaining blocker.

Do not start full real-robot PPO as part of this camera task. Full integration should begin only after maintainer review and supervised robot safety checks.

---

## Review checklist for the maintainer

- [ ] Adapter calls LeRobot `.read()`, not `.async_read()`.
- [ ] Adapter requests `color_mode="bgr"`.
- [ ] Dobot retains its existing single `[..., ::-1]` RGB conversion.
- [ ] Factory imports the adapter lazily.
- [ ] Stable by-id path is configured, not `/dev/video6` alone.
- [ ] Only one camera interface is configured.
- [ ] `enable_high_camera` is false.
- [ ] `main_image_key` and both reward keys are `cam_left_wrist`.
- [ ] No SO101 or Lumos behavior changed.
- [ ] Tests never access real hardware.
- [ ] Hardware smoke test never initializes the arm.
- [ ] Existing dirty-worktree changes were preserved.
