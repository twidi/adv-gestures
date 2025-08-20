from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar, NamedTuple, TypeAlias

from ..mediapipe import NormalizedLandmark, WorldLandmark


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
        HandLandmark.THUMB_MCP,
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
        x: X coordinate in pixels (0 to width) as integer  => also accessible via `landmark[0]`
        y: Y coordinate in pixels (0 to height) as integer  => also accessible via `landmark[1]`
        x_raw: Original MediaPipe X coordinate (0 to 1)
        y_raw: Original MediaPipe Y coordinate (0 to 1)
        z_raw: Original MediaPipe Z coordinate (depth relative to wrist)
    """

    x: int
    y: int
    x_raw: float
    y_raw: float
    z_raw: float
    x_world: float
    y_world: float
    z_world: float

    @classmethod
    def from_mediapipe(
        cls,
        normalized_landmark: NormalizedLandmark,
        world_landmark: WorldLandmark,
        width: int,
        height: int,
        mirroring: bool,
    ) -> Landmark:
        """Create a Landmark from MediaPipe's landmarks data.

        Args:
            normalized_landmark: MediaPipe landmark with normalized coordinates
            world_landmark: MediaPipe world landmark with depth information
            width: Image width in pixels
            height: Image height in pixels
            mirroring: Whether to mirror the X coordinate (for left-handed users)

        Returns:
            Landmark with pixel coordinates, with x values updated based on mirroring.
        """
        return cls(
            x=int(round((normalized_landmark.x if not mirroring else 1 - normalized_landmark.x) * width)),
            y=int(round(normalized_landmark.y * height)),
            x_raw=(normalized_landmark.x if not mirroring else 1 - normalized_landmark.x),
            y_raw=normalized_landmark.y,
            z_raw=normalized_landmark.z,
            x_world=world_landmark.x if not mirroring else -world_landmark.x,
            y_world=world_landmark.y,
            z_world=world_landmark.z,
        )

    @property
    def xy(self) -> tuple[int, int]:
        """Get the (x, y) coordinates as a tuple."""
        return self.x, self.y

    def to_dict(self) -> dict[str, Any]:
        """Export landmark data as a dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "raw": {
                "x": self.x_raw,
                "y": self.y_raw,
                "z": self.z_raw,
            },
            "world": {
                "x": self.x_world,
                "y": self.y_world,
                "z": self.z_world,
            },
        }
