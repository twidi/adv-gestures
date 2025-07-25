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
