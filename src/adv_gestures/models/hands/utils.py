from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Handedness(str, Enum):
    """Handedness enum for MediaPipe."""

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def from_data(cls, handedness_str: str) -> Handedness:
        """Convert MediaPipe handedness string to Handedness enum."""
        return cls(handedness_str.lower())

    def __str__(self) -> str:
        """Return the string representation of the handedness."""
        return self.value


class Box(NamedTuple):
    """Represents a bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


class HandsDirectionalRelationship(Enum):
    """Enum representing the directional relationship between two hands."""

    PARALLEL = "parallel"  # Hands pointing in the same direction (less than 1 degree difference)
    INTERSECTING = "intersecting"  # Hand segments cross each other
    CONVERGING = "converging"  # Hands pointing towards each other in front
    LEFT_INTO_RIGHT = "left_into_right"  # Left hand pointing towards right hand
    RIGHT_INTO_LEFT = "right_into_left"  # Right hand pointing towards left hand
    DIVERGING_NORMAL = "diverging_normal"  # V shape with uncrossed wrists
    DIVERGING_CROSSED = "diverging_crossed"  # >< shape with crossed wrists
    DIVERGING_LEFT_BEHIND_RIGHT = "diverging_left_behind_right"  # |_ Left ray passes behind right wrist
    DIVERGING_RIGHT_BEHIND_LEFT = "diverging_right_behind_left"  # _| Right ray passes behind left wrist
