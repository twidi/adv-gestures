from __future__ import annotations

from enum import IntEnum
from typing import ClassVar, NamedTuple, TypeAlias

from ..mediapipe import NormalizedLandmark


class HandLandmark(IntEnum):
    """MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


LandmarkGroup: TypeAlias = list[HandLandmark]


class LandmarkGroups:
    PALM: ClassVar[LandmarkGroup] = [
        HandLandmark.WRIST,
        HandLandmark.THUMB_CMC,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    ]
    THUMB: ClassVar[LandmarkGroup] = [
        HandLandmark.THUMB_CMC,
        HandLandmark.THUMB_MCP,
        HandLandmark.THUMB_IP,
        HandLandmark.THUMB_TIP,
    ]
    INDEX: ClassVar[LandmarkGroup] = [
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.INDEX_FINGER_PIP,
        HandLandmark.INDEX_FINGER_DIP,
        HandLandmark.INDEX_FINGER_TIP,
    ]
    MIDDLE: ClassVar[LandmarkGroup] = [
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_PIP,
        HandLandmark.MIDDLE_FINGER_DIP,
        HandLandmark.MIDDLE_FINGER_TIP,
    ]
    RING: ClassVar[LandmarkGroup] = [
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.RING_FINGER_PIP,
        HandLandmark.RING_FINGER_DIP,
        HandLandmark.RING_FINGER_TIP,
    ]
    PINKY: ClassVar[LandmarkGroup] = [
        HandLandmark.PINKY_MCP,
        HandLandmark.PINKY_PIP,
        HandLandmark.PINKY_DIP,
        HandLandmark.PINKY_TIP,
    ]


PALM_LANDMARKS = LandmarkGroups.PALM
FINGERS_LANDMARKS = [
    LandmarkGroups.THUMB,
    LandmarkGroups.INDEX,
    LandmarkGroups.MIDDLE,
    LandmarkGroups.RING,
    LandmarkGroups.PINKY,
]


class Landmark(NamedTuple):
    """A landmark with pixel coordinates.

    Attributes:
        x: X coordinate in pixels (0 to width) as integer
        y: Y coordinate in pixels (0 to height) as integer
        x_normalized: Original MediaPipe X coordinate (0 to 1)
        y_normalized: Original MediaPipe Y coordinate (0 to 1)
        z_normalized: Original MediaPipe Z coordinate (depth relative to wrist)
    """

    x: int
    y: int
    x_normalized: float
    y_normalized: float
    z_normalized: float | None

    @classmethod
    def from_normalized(cls, mp_landmark: NormalizedLandmark, width: int, height: int) -> Landmark:
        """Create a Landmark from MediaPipe's NormalizedLandmark.

        Args:
            mp_landmark: MediaPipe landmark with normalized coordinates
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Landmark with pixel coordinates
        """
        return cls(
            x=int(round(mp_landmark.x * width)),
            y=int(round(mp_landmark.y * height)),
            x_normalized=mp_landmark.x,
            y_normalized=mp_landmark.y,
            z_normalized=getattr(mp_landmark, "z", None),
        )
