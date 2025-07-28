from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class Box:
    """Represents a bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def line_intersections(
        self,
        line_origin: tuple[float, float],
        line_direction: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Find the two intersection points of a line (not ray) with this bounding box.

        Returns a tuple of two points (entry, exit) ordered based on the given direction,
        or None if no valid intersections. The first point is where the line enters
        the box when traveling along line_direction, and the second is where it exits.
        The line extends infinitely in both directions.
        """
        ox, oy = line_origin
        dx, dy = line_direction

        intersections = []

        # Check intersection with each edge
        # Left edge (x = min_x)
        if abs(dx) > 1e-10:
            t = (self.min_x - ox) / dx
            y = oy + t * dy
            if self.min_y <= y <= self.max_y:
                intersections.append((t, (self.min_x, y)))

        # Right edge (x = max_x)
        if abs(dx) > 1e-10:
            t = (self.max_x - ox) / dx
            y = oy + t * dy
            if self.min_y <= y <= self.max_y:
                intersections.append((t, (self.max_x, y)))

        # Bottom edge (y = min_y)
        if abs(dy) > 1e-10:
            t = (self.min_y - oy) / dy
            x = ox + t * dx
            if self.min_x <= x <= self.max_x:
                intersections.append((t, (x, self.min_y)))

        # Top edge (y = max_y)
        if abs(dy) > 1e-10:
            t = (self.max_y - oy) / dy
            x = ox + t * dx
            if self.min_x <= x <= self.max_x:
                intersections.append((t, (x, self.max_y)))

        # Remove duplicates (same point from different edges)
        unique_intersections: list[tuple[float, tuple[float, float]]] = []
        for t, point in intersections:
            is_duplicate = False
            for _, existing_point in unique_intersections:
                if abs(point[0] - existing_point[0]) < 1e-10 and abs(point[1] - existing_point[1]) < 1e-10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append((t, point))

        if len(unique_intersections) != 2:
            return None

        # Sort by t value to get consistent ordering
        unique_intersections.sort(key=lambda x: x[0])

        # Return just the points
        return unique_intersections[0][1], unique_intersections[1][1]


class HandsDirectionalRelationship(Enum):
    """Enum representing the directional relationship between two hands."""

    PARALLEL = "parallel"  # Hands pointing in the same direction (less than 1 degree difference)
    INTERSECTING = "intersecting"  # Hand segments cross each other
    CONVERGING_INSIDE_FRAME = "converging_inside_frame"  # Hands pointing towards each other, intersection in frame
    CONVERGING_OUTSIDE_FRAME = (
        "converging_outside_frame"  # Hands pointing towards each other, intersection outside frame
    )
    LEFT_INTO_RIGHT = "left_into_right"  # Left hand pointing towards right hand
    RIGHT_INTO_LEFT = "right_into_left"  # Right hand pointing towards left hand
    DIVERGING_NORMAL = "diverging_normal"  # V shape with uncrossed wrists
    DIVERGING_CROSSED = "diverging_crossed"  # >< shape with crossed wrists
    DIVERGING_LEFT_BEHIND_RIGHT_INSIDE_FRAME = (
        "diverging_left_behind_right_inside_frame"  # Left ray passes behind right wrist, intersection in frame
    )
    DIVERGING_LEFT_BEHIND_RIGHT_OUTSIDE_FRAME = (
        "diverging_left_behind_right_outside_frame"  # Left ray passes behind right wrist, intersection outside frame
    )
    DIVERGING_RIGHT_BEHIND_LEFT_INSIDE_FRAME = (
        "diverging_right_behind_left_inside_frame"  # Right ray passes behind left wrist, intersection in frame
    )
    DIVERGING_RIGHT_BEHIND_LEFT_OUTSIDE_FRAME = (
        "diverging_right_behind_left_outside_frame"  # Right ray passes behind left wrist, intersection outside frame
    )
