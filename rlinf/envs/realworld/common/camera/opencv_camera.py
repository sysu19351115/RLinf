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
"""Generic USB/V4L2 camera backed by LeRobot's :class:`OpenCVCamera`.

This is a thin adapter that delegates the actual V4L2 capture to LeRobot's
already-proven ``OpenCVCamera`` (the same backend SO101 uses) and exposes it
through RLinf's :class:`BaseCamera` interface, so the threaded frame queue,
``open``/``close``/``get_frame`` lifecycle and Dobot's BGR contract are reused
without change.

Frames intentionally remain **BGR** ``uint8``: :meth:`DobotEnv._get_camera_frames`
performs the single BGR→RGB conversion (``[..., ::-1]``) after crop/resize.

Only ``read()`` is used (not ``async_read()``): :class:`BaseCamera` already owns
a background capture thread, so calling ``async_read()`` would spawn a second
competing thread.

FOURCC / high-resolution capture
--------------------------------
When ``camera_info.fourcc`` is ``None`` (default), the adapter uses LeRobot's
``OpenCVCamera`` verbatim — identical to SO101, fully backward compatible.

When ``camera_info.fourcc`` is set (e.g. ``"MJPG"``), the adapter opens the
device via a **native OpenCV** path and applies the pixel format *before*
resolution and FPS. This is required for high resolutions such as 1920x1080,
where the V4L2 default (uncompressed YUYV) is bandwidth-limited (≈5 FPS) and
LeRobot's strict FPS check rejects it. With MJPG the RYS camera reaches 1080p
at the requested rate. LeRobot's ``OpenCVCamera`` does not expose a FOURCC
knob and configures FPS *before* it could be overridden, so the native path is
unavoidable here; it mirrors what :class:`LumosCamera` already does.
"""

from typing import Optional

import numpy as np
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import (
    OpenCVCamera as LeRobotOpenCVCamera,
)

from .base_camera import BaseCamera, CameraInfo


class OpenCVUSBCamera(BaseCamera):
    """Adapt LeRobot ``OpenCVCamera`` (or native OpenCV) to RLinf's BaseCamera.

    ``camera_info.serial_number`` is passed straight to the backend as the
    device index, so it may be a ``/dev/videoN`` path, a stable
    ``/dev/v4l/by-id/...`` path, or an integer index.

    Frames are produced in BGR (matching RealSense / ZED / Lumos backends);
    callers convert to RGB exactly once after crop/resize.

    Args:
        camera_info: Descriptor whose optional ``fourcc`` selects the backend:
            ``None`` → LeRobot ``OpenCVCamera`` (default, SO101-compatible);
            a FOURCC like ``"MJPG"`` → native OpenCV path with the format set
            before resolution/FPS (needed for 1080p+ capture).
    """

    def __init__(self, camera_info: CameraInfo):
        super().__init__(camera_info)
        if camera_info.fourcc:
            self._device: Optional[object] = None  # native path uses self._cap
            self._cap = None
            self._cv2 = None
            self._connect_native(camera_info)
        else:
            self._cap = None
            self._cv2 = None
            config = OpenCVCameraConfig(
                camera_index=camera_info.serial_number,
                width=int(camera_info.resolution[0]),
                height=int(camera_info.resolution[1]),
                fps=int(camera_info.fps),
                color_mode="bgr",
            )
            self._device = LeRobotOpenCVCamera(config)
            self._device.connect()

    def _connect_native(self, camera_info: CameraInfo) -> None:
        """Open the device via native OpenCV with FOURCC set first.

        V4L2 requires the pixel format to be chosen *before* resolution/FPS so
        the driver can pick a matching bandwidth budget (e.g. MJPG for 1080p).
        """
        import cv2

        self._cv2 = cv2
        cap = cv2.VideoCapture(camera_info.serial_number, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open USB camera (serial={camera_info.serial_number})."
            )
        fourcc = cv2.VideoWriter_fourcc(*camera_info.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_info.resolution[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_info.resolution[1]))
        cap.set(cv2.CAP_PROP_FPS, int(camera_info.fps))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        # Native OpenCV path.
        if self._cap is not None:
            ok, frame = self._cap.read()
            return bool(ok), frame
        # LeRobot path. Guard against the close race: BaseCamera.close() may
        # disconnect the device before the capture thread finishes its loop, in
        # which case .read() raises — treat that as "no frame" (per the
        # BaseCamera contract) instead of an error.
        device = self._device
        if device is None or not getattr(device, "is_connected", False):
            return False, None
        try:
            return True, device.read()
        except Exception:
            return False, None

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            return
        if getattr(self._device, "is_connected", False):
            self._device.disconnect()
