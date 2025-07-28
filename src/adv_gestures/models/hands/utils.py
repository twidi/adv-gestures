from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    @property
    def left(self) -> float:
        """Left edge of the box."""
        return self.min_x

    @property
    def right(self) -> float:
        """Right edge of the box."""
        return self.max_x

    @property
    def top(self) -> float:
        """Top edge of the box."""
        return self.min_y

    @property
    def bottom(self) -> float:
        """Bottom edge of the box."""
        return self.max_y

    def overlaps(self, other: Box) -> bool:
        """Check if this box overlaps with another box."""
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )

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


def oriented_boxes_overlap(
    corners1: Sequence[tuple[float, float]],
    corners2: Sequence[tuple[float, float]],
) -> bool:
    """Check if two oriented rectangles overlap using the Separating Axis Theorem (SAT).

    This function is optimized for rectangles (not general convex polygons) and only
    tests the 4 necessary axes - the perpendiculars to the two unique edge directions
    of each rectangle.

    Args:
        corners1: Four corners of the first rectangle in order (consecutive corners form edges)
        corners2: Four corners of the second rectangle in order

    Returns:
        True if the rectangles overlap, False otherwise
    """
    if len(corners1) != 4 or len(corners2) != 4:
        raise ValueError("Both boxes must have exactly 4 corners")

    # Convert to numpy arrays for easier computation
    box1 = np.array(corners1)
    box2 = np.array(corners2)

    # For rectangles, we only need to test 4 axes:
    # - 2 axes from the first rectangle (perpendicular to its two edge directions)
    # - 2 axes from the second rectangle (perpendicular to its two edge directions)

    axes = []

    # Get the two unique edge directions from box1 (edges 0-1 and 1-2)
    # The other two edges (2-3 and 3-0) are parallel to these
    edge1 = box1[1] - box1[0]
    edge2 = box1[2] - box1[1]

    # Get perpendicular axes (normals) to these edges
    axis1 = np.array([-edge1[1], edge1[0]])
    axis2 = np.array([-edge2[1], edge2[0]])

    # Normalize and add to axes list
    length1 = np.linalg.norm(axis1)
    if length1 > 1e-10:
        axes.append(axis1 / length1)

    length2 = np.linalg.norm(axis2)
    if length2 > 1e-10:
        axes.append(axis2 / length2)

    # Get the two unique edge directions from box2
    edge3 = box2[1] - box2[0]
    edge4 = box2[2] - box2[1]

    # Get perpendicular axes (normals) to these edges
    axis3 = np.array([-edge3[1], edge3[0]])
    axis4 = np.array([-edge4[1], edge4[0]])

    # Normalize and add to axes list
    length3 = np.linalg.norm(axis3)
    if length3 > 1e-10:
        axes.append(axis3 / length3)

    length4 = np.linalg.norm(axis4)
    if length4 > 1e-10:
        axes.append(axis4 / length4)

    # Check each axis for separation
    for axis in axes:
        # Project all vertices of both boxes onto this axis
        projections1 = [np.dot(corner, axis) for corner in box1]
        projections2 = [np.dot(corner, axis) for corner in box2]

        # Find min and max projections for each box
        min1, max1 = min(projections1), max(projections1)
        min2, max2 = min(projections2), max(projections2)

        # Check if there's a gap between the projections
        if max1 < min2 or max2 < min1:
            # Found a separating axis, boxes don't overlap
            return False

    # No separating axis found, boxes must overlap
    return True
