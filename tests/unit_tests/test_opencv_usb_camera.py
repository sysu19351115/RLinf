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
"""Tests for the LeRobot-to-BaseCamera USB camera adapter.

These tests never touch real hardware: LeRobot's ``OpenCVCamera`` and
``OpenCVCameraConfig`` are monkeypatched with in-memory fakes. They also cover
Dobot's propagation of ``camera_resolution`` / ``camera_fps`` and the BGR→RGB
conversion contract.
"""

import numpy as np
import pytest

import rlinf.envs.realworld.common.camera.opencv_camera as camera_module
from rlinf.envs.realworld.common.camera.base_camera import CameraInfo


class FakeOpenCVCameraConfig:
    """In-memory replacement for LeRobot ``OpenCVCameraConfig``."""

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
    """In-memory replacement for LeRobot ``OpenCVCamera``."""

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
    """Replace LeRobot camera symbols so no V4L2 device is opened."""
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


# ---------------------------------------------------------------------------
# Task 1 / Task 2: adapter contract
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 3: Dobot propagation of camera_resolution / camera_fps
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402

from rlinf.envs.realworld.dobot import dobot_env as dobot_env_module  # noqa: E402
from rlinf.envs.realworld.dobot.dobot_env import (  # noqa: E402
    DobotEnv,
    DobotRobotConfig,
)


class FakeRLinfCamera:
    """In-memory replacement for a RLinf ``BaseCamera`` instance."""

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


@pytest.mark.parametrize("bad_resolution", [(0, 480), (640, 0), (640,), (640, 0, 480)])
def test_dobot_rejects_invalid_camera_resolution(bad_resolution):
    env = DobotEnv.__new__(DobotEnv)
    env.config = DobotRobotConfig(
        camera_resolution=bad_resolution,
    )
    with pytest.raises(ValueError, match="camera_resolution"):
        env._validate_config()


def test_dobot_rejects_invalid_camera_fps():
    env = DobotEnv.__new__(DobotEnv)
    env.config = DobotRobotConfig(
        camera_fps=0,
    )
    with pytest.raises(ValueError, match="camera_fps"):
        env._validate_config()


# ---------------------------------------------------------------------------
# Task 4: BGR→RGB conversion contract
# ---------------------------------------------------------------------------


class _FakeCameraInfo:
    def __init__(self, name):
        self.name = name


class FakeBgrCamera:
    """Returns a uniform BGR frame for the RGB-conversion regression test."""

    def __init__(self, name, frame):
        self._camera_info = _FakeCameraInfo(name)
        self._frame = frame

    def get_frame(self, timeout=5):
        return self._frame


class FakePlayer:
    def put_frame(self, frames):
        self.frames = frames


def test_dobot_converts_bgr_to_rgb_exactly_once():
    # Uniform BGR pixel [10, 20, 30] must become RGB [30, 20, 10].
    bgr_frame = np.full((480, 640, 3), [10, 20, 30], dtype=np.uint8)
    camera = FakeBgrCamera("cam_left_wrist", bgr_frame)

    env = DobotEnv.__new__(DobotEnv)
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
    # Bypass _open_cameras: inject the fake camera directly.
    env._cameras = [camera]

    frames = env._get_camera_frames()

    frame = frames["cam_left_wrist"]
    assert frame.shape == (224, 224, 3)
    assert frame.dtype == np.uint8
    # BGR [10, 20, 30] → RGB [30, 20, 10].
    assert frame[0, 0].tolist() == [30, 20, 10]


# ---------------------------------------------------------------------------
# FOURCC / native OpenCV path (1080p MJPG)
# ---------------------------------------------------------------------------


class FakeVideoCapture:
    """Records the order of V4L2 property sets and serves frames."""

    # cv2 property constants used by the adapter.
    CAP_PROP_FOURCC = "FOURCC"
    CAP_PROP_FRAME_WIDTH = "WIDTH"
    CAP_PROP_FRAME_HEIGHT = "HEIGHT"
    CAP_PROP_FPS = "FPS"
    CAP_PROP_BUFFERSIZE = "BUFFERSIZE"

    def __init__(self, frame):
        self._frame = frame
        self.set_order = []  # ordered list of (prop, value)
        self._values = {}
        self._released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_order.append((prop, value))
        self._values[prop] = value
        return True

    def get(self, prop):
        return self._values.get(prop, 0)

    def read(self):
        if self._released:
            return False, None
        return True, self._frame.copy()

    def release(self):
        self._released = True


def test_fourcc_sets_format_before_resolution_and_fps(monkeypatch):
    """Native path must apply FOURCC, then resolution, then FPS."""
    import sys
    import types

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cap = FakeVideoCapture(frame)

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_V4L2 = "V4L2"
    fake_cv2.CAP_PROP_FOURCC = FakeVideoCapture.CAP_PROP_FOURCC
    fake_cv2.CAP_PROP_FRAME_WIDTH = FakeVideoCapture.CAP_PROP_FRAME_WIDTH
    fake_cv2.CAP_PROP_FRAME_HEIGHT = FakeVideoCapture.CAP_PROP_FRAME_HEIGHT
    fake_cv2.CAP_PROP_FPS = FakeVideoCapture.CAP_PROP_FPS
    fake_cv2.CAP_PROP_BUFFERSIZE = FakeVideoCapture.CAP_PROP_BUFFERSIZE
    fake_cv2.VideoCapture = lambda *a, **k: cap
    fake_cv2.VideoWriter_fourcc = lambda *chars: "".join(chars)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    info = CameraInfo(
        name="cam_left_wrist",
        serial_number="/dev/video6",
        camera_type="opencv",
        resolution=(1920, 1080),
        fps=30,
        fourcc="MJPG",
    )
    camera = camera_module.OpenCVUSBCamera(info)

    # The FOURCC must be the FIRST property set (V4L2 needs the format chosen
    # before resolution/FPS so the driver can budget bandwidth).
    props_in_order = [p for p, _ in cap.set_order]
    assert props_in_order[0] == FakeVideoCapture.CAP_PROP_FOURCC
    assert props_in_order.index(FakeVideoCapture.CAP_PROP_FOURCC) < props_in_order.index(
        FakeVideoCapture.CAP_PROP_FRAME_WIDTH
    )
    assert props_in_order.index(FakeVideoCapture.CAP_PROP_FOURCC) < props_in_order.index(
        FakeVideoCapture.CAP_PROP_FPS
    )

    ok, fr = camera._read_frame()
    assert ok is True
    assert fr.shape == (1080, 1920, 3)

    camera._close_device()
    ok2, fr2 = camera._read_frame()
    # After close, the native cap is released → must report no frame gracefully.
    assert ok2 is False and fr2 is None


def test_no_fourcc_uses_lerobot_path():
    """Without fourcc the adapter must instantiate LeRobot OpenCVCamera."""
    info = CameraInfo(
        name="cam_left_wrist",
        serial_number="/dev/video6",
        camera_type="opencv",
        resolution=(640, 480),
        fps=30,
        fourcc=None,
    )
    camera = camera_module.OpenCVUSBCamera(info)
    assert isinstance(camera._device, FakeLeRobotOpenCVCamera)
    assert camera._cap is None
    camera._close_device()
