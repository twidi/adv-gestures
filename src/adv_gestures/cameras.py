from __future__ import annotations

import glob
import re
from contextlib import suppress
from typing import NamedTuple

from linuxpy.video.device import (  # type: ignore[import-untyped]
    BufferType,
    Device,
    PixelFormat,
)


class CameraInfo(NamedTuple):
    device_index: int
    name: str
    height: int
    width: int
    format: PixelFormat

    def __str__(self) -> str:
        return f"[{self.device_index}] {self.name} - {self.width}x{self.height} @ {self.format.name}"


def list_cameras() -> list[CameraInfo]:
    """List all available cameras and return a list of CameraInfo objects."""
    cameras = []
    video_devices = sorted(glob.glob("/dev/video*"))

    for device_path in video_devices:
        with suppress(Exception):  # Skip devices that can't be accessed
            device_index = int(re.search(r"/dev/video(\d+)", device_path).group(1))  # type: ignore[union-attr]  # handled by suppress
            device = Device(device_path)
            device.open()

            # Only list devices that support video capture
            formats = [f for f in device.info.formats if f.type == BufferType.VIDEO_CAPTURE]
            if not formats:
                device.close()
                continue

            # Get current format
            current_format = device.get_format(BufferType.VIDEO_CAPTURE)

            # Skip GREY cameras
            if current_format.pixel_format == PixelFormat.GREY:
                device.close()
                continue

            camera_info = CameraInfo(
                device_index=device_index,
                name=device.info.card,
                height=current_format.height,
                width=current_format.width,
                format=current_format.pixel_format,
            )
            cameras.append(camera_info)
            device.close()

    return cameras
