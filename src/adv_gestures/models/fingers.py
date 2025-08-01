# Finger straightness thresholds - can be customized per finger
from __future__ import annotations

from collections import deque
from enum import IntEnum
from functools import cached_property
from math import acos, atan2, degrees, exp, radians, sqrt
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeAlias, TypeVar

import numpy as np

from ..config import BaseFingerConfig, Config, FingerConfig, IndexConfig, ThumbConfig
from ..smoothing import (
    CoordSmoother,
    SmoothedBase,
    SmoothedProperty,
    smoothed_bool,
    smoothed_float,
    smoothed_optional_float,
)
from .landmarks import Landmark
from .utils import Direction

if TYPE_CHECKING:
    from .hands.hand import Hand

FINGER_STRAIGHT_THRESHOLD = 0.90  # Default score above this is considered straight (strict)

FINGER_NEARLY_STRAIGHT_THRESHOLD = 0.70  # Default score above this is considered nearly straight (relaxed)


class FingerIndex(IntEnum):
    """Finger index constants for easier reference."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


FingerConfigType = TypeVar("FingerConfigType", bound=BaseFingerConfig)


class Finger(SmoothedBase, Generic[FingerConfigType]):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "centroid",
        "start_point",
        "end_point",
        "straightness_score",
        "is_straight",
        "is_nearly_straight",
        "straight_direction",
        "straight_direction_angle",
        "is_fully_bent",
        "fold_angle",
        "tip_direction",
        "tip_direction_angle",
        "tip_on_thumb",
        "touching_adjacent_fingers",
    )

    index: ClassVar[FingerIndex]
    adjacent_fingers_indexes: ClassVar[tuple[FingerIndex, ...]] = ()
    start_point_index: ClassVar[int] = 0

    def __init__(self, hand: Hand, config: Config):
        super().__init__()
        self.config = config
        self.finger_config: FingerConfigType = getattr(config.hands, self.index.name.lower())
        self.hand = hand
        self.landmarks: list[Landmark] = []

    def reset(self) -> None:
        """Reset the finger and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        # Clear cached properties
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update(self, landmarks: list[Landmark]) -> None:
        """Update the finger with new landmarks."""
        self.landmarks = landmarks

    def __bool__(self) -> bool:
        """Check if the finger is visible and has landmarks."""
        return len(self.landmarks) > 0

    def is_adjacent_to(self, other: FingerIndex | AnyFinger) -> bool:
        return abs(self.index - (other.index if isinstance(other, Finger) else other)) == 1

    def _calc_centroid(self) -> tuple[float, float]:
        """Calculate the centroid of all finger landmarks."""
        if not self.landmarks:
            return 0.0, 0.0

        centroid_x = sum(landmark.x for landmark in self.landmarks) / len(self.landmarks)
        centroid_y = sum(landmark.y for landmark in self.landmarks) / len(self.landmarks)

        return centroid_x, centroid_y

    centroid = SmoothedProperty(_calc_centroid, CoordSmoother)

    def _calc_straightness_score(self) -> float:
        raise NotImplementedError

    straightness_score = smoothed_float(_calc_straightness_score)

    def _calc_is_straight(self) -> bool:
        """Check if the finger is straight based on the straightness score."""

        # Use the straightness score with strict threshold from finger config
        return self.straightness_score >= self.finger_config.straightness.straight_threshold

    is_straight = smoothed_bool(_calc_is_straight)

    def _calc_is_nearly_straight(self) -> bool:
        """Check if the finger is nearly straight based on the straightness score."""

        # Use the straightness score with relaxed threshold from finger config
        nearly_threshold = self.finger_config.straightness.nearly_straight_threshold
        straight_threshold = self.finger_config.straightness.straight_threshold
        return nearly_threshold <= self.straightness_score < straight_threshold

    is_nearly_straight = smoothed_bool(_calc_is_nearly_straight)

    @property
    def is_nearly_straight_or_straight(self) -> bool:
        """Check if the finger is either nearly straight or straight."""
        return self.is_nearly_straight or self.is_straight

    @property
    def is_not_straight_at_all(self) -> bool:
        return not (self.is_straight or self.is_nearly_straight)

    def _calc_start_point(self) -> tuple[int, int] | None:
        """Get the start point of the finger (base)."""
        if not self.landmarks:
            return None
        return self.landmarks[self.start_point_index].xy

    start_point = SmoothedProperty(_calc_start_point, CoordSmoother)

    def _calc_end_point(self) -> tuple[int, int] | None:
        """Get the end point of the finger (tip)."""
        if not self.landmarks:
            return None
        return self.landmarks[-1].xy

    end_point = SmoothedProperty(_calc_end_point, CoordSmoother)

    def _calc_fold_angle(self) -> float | None:
        """Calculate the fold angle at the PIP joint (angle between MCP->PIP and PIP->TIP vectors).
        Returns angle in degrees. 180 = straight, lower angles = more bent."""
        if len(self.landmarks) < 4:
            return None

        # For thumb, use MCP->IP->TIP instead of MCP->PIP->TIP
        if self.index == FingerIndex.THUMB:
            mcp, pip, tip = self.landmarks[1], self.landmarks[2], self.landmarks[3]
        else:
            # For other fingers: MCP->PIP->TIP
            mcp, pip, tip = self.landmarks[0], self.landmarks[1], self.landmarks[3]

        # Vector PIP->MCP
        v1 = np.array([mcp.x - pip.x, mcp.y - pip.y])
        # Vector PIP->DIP
        v2 = np.array([tip.x - pip.x, tip.y - pip.y])

        # Calculate magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 < 0.001 or mag2 < 0.001:
            return None

        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (mag1 * mag2)

        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_rad = acos(cos_angle)
        angle_deg = degrees(angle_rad)

        return angle_deg

    fold_angle = smoothed_optional_float(_calc_fold_angle)

    def _calc_tip_direction(self) -> tuple[float, float] | None:
        """Calculate the direction of the finger tip using the last two points.
        Returns a normalized vector (dx, dy) pointing from second-to-last to last point.
        Returns None if finger is fully bent."""
        if len(self.landmarks) < 2:
            return None

        # Don't calculate direction if finger is fully bent
        if self.is_fully_bent:
            return None

        # Get the last two points
        second_last = self.landmarks[-2]
        last = self.landmarks[-1]

        # Calculate direction vector
        dx = last.x - second_last.x
        dy = last.y - second_last.y

        # Normalize the vector
        magnitude = sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    tip_direction = SmoothedProperty(_calc_tip_direction, CoordSmoother)

    def _calc_tip_direction_angle(self) -> float | None:
        """Calculate the angle of the tip direction vector in degrees.
        Returns angle in range [-180, 180] where:
        - 0° = pointing right
        - 90° = pointing up
        - 180°/-180° = pointing left
        - -90° = pointing down
        """
        direction = self.tip_direction
        if direction is None:
            return None

        dx, dy = direction
        # Calculate angle in radians and convert to degrees
        angle_rad = atan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
        angle_deg = degrees(angle_rad)
        return angle_deg

    tip_direction_angle = smoothed_optional_float(_calc_tip_direction_angle)

    def _calc_straight_direction(self) -> tuple[float, float] | None:
        """Calculate the overall direction of the finger from base to tip.
        Returns a normalized vector (dx, dy) pointing from first to last point.
        Returns None if finger is not straight."""
        if self.is_not_straight_at_all:
            return None

        if self.end_point is None or self.start_point is None:
            return None

        # Calculate direction vector
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]

        # Normalize the vector
        magnitude = sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    straight_direction = SmoothedProperty(_calc_straight_direction, CoordSmoother)

    def _calc_straight_direction_angle(self) -> float | None:
        """Calculate the angle of the straight direction vector in degrees.
        Returns angle in range [-180, 180] where:
        - 0° = pointing right
        - 90° = pointing up
        - 180°/-180° = pointing left
        - -90° = pointing down
        """
        direction = self.straight_direction
        if direction is None:
            return None

        dx, dy = direction
        # Calculate angle in radians and convert to degrees
        angle_rad = atan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
        angle_deg = degrees(angle_rad)
        return angle_deg

    straight_direction_angle = smoothed_optional_float(_calc_straight_direction_angle)

    def _calc_is_fully_bent(self) -> bool:
        """Check if the finger is fully bent based on fold angle threshold."""

        if self.fold_angle is None:
            return True

        return self.fold_angle < self.finger_config.fully_bent_max_angle_degrees

    is_fully_bent = smoothed_bool(_calc_is_fully_bent)

    @cached_property
    def touching_adjacent_fingers(self) -> list[FingerIndex]:
        return [finger_index for finger_index in self.adjacent_fingers_indexes if self.is_touching(finger_index)]

    def is_touching(self, other: FingerIndex | AnyFinger) -> bool:
        """Check if this finger is touching another finger. Can only be True for adjacent fingers."""

        if not self.is_adjacent_to(other):
            return False

        other_index = other.index if isinstance(other, Finger) else other

        # Order fingers to avoid duplicates (always use lower index first)
        finger1, finger2 = sorted((self.index, other_index))
        key = (finger1, finger2)

        if key == (FingerIndex.THUMB, FingerIndex.INDEX):
            return self.hand.thumb_index_touching
        if key == (FingerIndex.INDEX, FingerIndex.MIDDLE):
            return self.hand.index_middle_touching
        if key == (FingerIndex.MIDDLE, FingerIndex.RING):
            return self.hand.middle_ring_touching
        if key == (FingerIndex.RING, FingerIndex.PINKY):
            return self.hand.ring_pinky_touching

        return False

    def to_dict(self) -> dict[str, Any]:
        """Export finger data as a dictionary."""
        centroid = self.centroid
        start_point = self.start_point
        end_point = self.end_point
        straight_direction = self.straight_direction
        tip_direction = self.tip_direction

        return {
            "type": self.index.name,
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "centroid": {"x": centroid[0], "y": centroid[1]} if centroid else {"x": 0.0, "y": 0.0},
            "start_point": {"x": start_point[0], "y": start_point[1]} if start_point else None,
            "end_point": {"x": end_point[0], "y": end_point[1]} if end_point else None,
            "straightness_score": self.straightness_score,
            "is_straight": self.is_straight,
            "is_nearly_straight": self.is_nearly_straight,
            "is_nearly_straight_or_straight": self.is_nearly_straight_or_straight,
            "is_not_straight_at_all": self.is_not_straight_at_all,
            "straight_direction": (
                {"x": straight_direction[0], "y": straight_direction[1]} if straight_direction else None
            ),
            "straight_direction_angle": self.straight_direction_angle,
            "tip_direction": {"x": tip_direction[0], "y": tip_direction[1]} if tip_direction else None,
            "tip_direction_angle": self.tip_direction_angle,
            "is_fully_bent": self.is_fully_bent,
            "fold_angle": self.fold_angle,
            "touching_adjacent_fingers": [
                FingerIndex(finger_idx).name for finger_idx in self.touching_adjacent_fingers
            ],
        }


class Thumb(Finger[ThumbConfig]):
    index = FingerIndex.THUMB
    adjacent_fingers_indexes = (FingerIndex.INDEX,)
    start_point_index: ClassVar[int] = 1

    def _calc_straightness_score(self) -> float:
        """Basic straightness score calculation for thumb."""
        # Configuration constants from per-finger config
        alignment_threshold = self.finger_config.straightness.alignment_threshold
        max_deviation_for_zero_score = self.finger_config.straightness.max_deviation_for_zero_score
        min_denominator_threshold = 0.001  # Minimum line length to avoid division by zero

        if len(self.landmarks) < 3:
            return 0.0

        x_coords = [landmark.x for landmark in self.landmarks]
        y_coords = [landmark.y for landmark in self.landmarks]

        # Calculate the line from first to last point
        x1, y1 = x_coords[0], y_coords[0]
        x2, y2 = x_coords[-1], y_coords[-1]

        # Calculate thumb length for normalization
        thumb_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if thumb_length < min_denominator_threshold:
            return 0.0

        # Check intermediate points alignment
        max_deviation = 0.0
        for i in range(1, len(self.landmarks) - 1):
            x, y = x_coords[i], y_coords[i]

            # Calculate perpendicular distance from point to line
            # Using formula: |ax + by + c| / sqrt(a^2 + b^2)
            # where line is: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if denominator > min_denominator_threshold:  # Avoid division by zero
                distance = numerator / denominator
                # Normalize the distance by thumb length to get a relative measure
                normalized_distance = distance / thumb_length
                max_deviation = max(max_deviation, normalized_distance)

        # Convert deviation to score
        if max_deviation <= alignment_threshold:
            score = 1.0
        else:
            # Linear decay from 1.0 to 0 as deviation increases
            score = max(0.0, 1.0 - (max_deviation / max_deviation_for_zero_score))

        return score

    straightness_score = smoothed_float(_calc_straightness_score)


class OtherFinger(Finger[FingerConfig]):
    def _calc_straightness_score(self) -> float:
        """Calculate a straightness score from 0 to 1, where 1 is perfectly straight and 0 is not straight."""

        # Configuration constants from per-finger config
        distal_segments_max_ratio = self.finger_config.straightness.distal_segments_max_ratio
        distal_segments_max_ratio_back = self.finger_config.straightness.distal_segments_max_ratio_back
        max_angle_degrees = self.finger_config.straightness.max_angle_degrees

        # Segment ratio scoring parameters
        segment_ratio_score_at_threshold = self.finger_config.straightness.segment_ratio_score_at_threshold
        segment_ratio_decay_rate = self.finger_config.straightness.segment_ratio_decay_rate
        segment_ratio_linear_range = self.finger_config.straightness.segment_ratio_linear_range

        # Angle scoring parameters
        angle_score_linear_range = self.finger_config.straightness.angle_score_linear_range
        angle_score_at_threshold = self.finger_config.straightness.angle_score_at_threshold
        angle_decay_rate = self.finger_config.straightness.angle_decay_rate

        # Score combination weights
        angle_score_weight = self.finger_config.straightness.angle_score_weight
        segment_ratio_weight = self.finger_config.straightness.segment_ratio_weight

        # Numerical thresholds
        min_magnitude_threshold = 0.001  # Minimum vector magnitude to avoid division by zero

        if not self.hand.is_facing_camera:
            distal_segments_max_ratio = distal_segments_max_ratio_back

        if len(self.landmarks) < 3:  # Need at least 3 points to check alignment
            return 0.0

        # Extract x and y coordinates for all finger landmarks
        x_coords = [landmark.x for landmark in self.landmarks]
        y_coords = [landmark.y for landmark in self.landmarks]

        # Calculate segment lengths
        segment_lengths = []
        for i in range(len(self.landmarks) - 1):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
            segment_length = sqrt(dx**2 + dy**2)
            segment_lengths.append(segment_length)

        # For fingers (not thumb), expect 3 segments: MCP-PIP, PIP-DIP, DIP-TIP
        # The MCP-PIP segment is naturally longer, so check proportional consistency
        segment_ratio_score = 1.0
        if len(segment_lengths) == 3:
            mcp_pip, pip_dip, dip_tip = segment_lengths  # proximal, middle, distal segments

            # Check if the distal segments (PIP-DIP and DIP-TIP) are reasonably similar
            if pip_dip > 0 and dip_tip > 0:
                ratio = max(pip_dip, dip_tip) / min(pip_dip, dip_tip)
                if ratio > distal_segments_max_ratio:
                    # Convert ratio to score: at threshold = segment_ratio_score_at_threshold, higher ratios decay
                    excess_ratio = ratio - distal_segments_max_ratio
                    # Use exponential decay for excessive ratios
                    segment_ratio_score = segment_ratio_score_at_threshold * exp(
                        -excess_ratio * segment_ratio_decay_rate
                    )
                else:
                    # Linear interpolation from 1.0 (ratio=1) to (1.0 - segment_ratio_linear_range) at threshold
                    segment_ratio_score = (
                        1.0 - ((ratio - 1.0) / (distal_segments_max_ratio - 1.0)) * segment_ratio_linear_range
                    )

        # Verify segment angles are relatively straight
        # Calculate angles between consecutive segments
        max_found_angle = 0.0
        if len(segment_lengths) >= 2:
            for i in range(len(segment_lengths) - 1):
                # Get vectors for consecutive segments
                p1 = (x_coords[i], y_coords[i])
                p2 = (x_coords[i + 1], y_coords[i + 1])
                p3 = (x_coords[i + 2], y_coords[i + 2])

                # Vector from p1 to p2
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                # Vector from p2 to p3
                v2 = (p3[0] - p2[0], p3[1] - p2[1])

                # Calculate angle between vectors using dot product
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = sqrt(v2[0] ** 2 + v2[1] ** 2)

                if mag1 > min_magnitude_threshold and mag2 > min_magnitude_threshold:  # Avoid division by zero
                    cos_angle = dot_product / (mag1 * mag2)
                    # Clamp to [-1, 1] to handle numerical errors
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = acos(cos_angle)

                    # Convert to degrees for easier interpretation
                    angle_degrees = degrees(angle)
                    max_found_angle = max(max_found_angle, angle_degrees)

        # Convert angle to score: 0° = 1.0, max_angle_degrees = angle_score_at_threshold, higher angles decay to 0
        if max_found_angle <= max_angle_degrees:
            # Perfect to acceptable range: linear from 1.0 to angle_score_at_threshold
            angle_score = 1.0 - (max_found_angle / max_angle_degrees) * angle_score_linear_range
        else:
            # Beyond threshold: exponential decay from angle_score_at_threshold to 0
            excess_angle = max_found_angle - max_angle_degrees
            # Use exponential decay: score decreases rapidly after threshold
            angle_score = angle_score_at_threshold * exp(-excess_angle / angle_decay_rate)

        # Combine both scores (weighted average: angle is more important)
        combined_score = angle_score_weight * angle_score + segment_ratio_weight * segment_ratio_score

        return combined_score

    straightness_score = smoothed_float(_calc_straightness_score)

    def _calc_tip_on_thumb(self) -> bool:
        """Check if this finger tip touches the thumb tip of the same hand."""

        # Get the thumb finger
        thumb = self.hand.thumb

        if not thumb or not thumb.landmarks or not self.landmarks:
            return False

        # Get the tip landmarks (last landmark of each finger)
        thumb_tip = thumb.end_point
        finger_tip = self.end_point

        if thumb_tip is None or finger_tip is None:
            return False

        # Calculate distance (coordinates already include aspect ratio correction)
        dx = finger_tip[0] - thumb_tip[0]
        dy = finger_tip[1] - thumb_tip[1]
        distance = sqrt(dx**2 + dy**2)

        # Compare distance to a relative threshold based on current finger and thumb segment lengths

        segments = []
        for finger in (thumb, self):
            # Calculate segment length
            ip = finger.landmarks[2]
            tip = finger.landmarks[3]
            segment_dx = tip[0] - ip[0]
            segment_dy = tip[1] - ip[1]
            segment_length = sqrt(segment_dx**2 + segment_dy**2)
            segments.append(segment_length)

        segment = max(segments)

        relative_threshold = self.finger_config.thumb_distance_relative_threshold * segment

        one_is_fully_bent = thumb.is_fully_bent or self.is_fully_bent
        if one_is_fully_bent:
            relative_threshold *= 2

        # if self.index == FingerIndex.INDEX:  # and not (distance < relative_threshold):
        #     print(
        #         f"distance={distance:5.2f}, segments=(thumb={segments[0]:5.2f}, index={segments[1]:5.2f}), "
        #         f"segment={segment:5.2f}, "
        #         f"fully_bent={str(one_is_fully_bent):5}, relative_threshold={relative_threshold:5.2f}"
        #         f" -> {distance < relative_threshold}"
        #     )

        return distance < relative_threshold

    tip_on_thumb = smoothed_bool(_calc_tip_on_thumb)

    def to_dict(self) -> dict[str, Any]:
        """Export finger data as a dictionary, including tip_on_thumb."""
        data = super().to_dict()
        data["tip_on_thumb"] = self.tip_on_thumb
        return data


class IndexFinger(OtherFinger):
    index = FingerIndex.INDEX
    adjacent_fingers_indexes = (FingerIndex.THUMB, FingerIndex.MIDDLE)

    def __init__(self, hand: Hand, config: Config):
        super().__init__(hand, config)
        self.finger_config: IndexConfig = config.hands.index
        self._tip_position_history: deque[tuple[float, float, float]] = deque()
        # Direction history for straight_direction (timestamp, direction_x, direction_y, direction_z)
        self._direction_history: deque[tuple[float, float, float, float]] = deque()

    def update(self, landmarks: list[Landmark]) -> None:
        """Update finger with new landmarks and track tip position."""
        super().update(landmarks)

        if landmarks and len(landmarks) > 0:
            tip = landmarks[-1]
            current_time = time()

            # Add new position
            self._tip_position_history.append((current_time, tip.x, tip.y))

            # Remove positions older than 5 seconds
            cutoff_time = current_time - 5.0
            while self._tip_position_history and self._tip_position_history[0][0] < cutoff_time:
                self._tip_position_history.popleft()

        # Update direction history if finger has a straight direction
        direction = self.straight_direction
        if direction is not None:
            current_time = time()
            # Add new direction to history
            self._direction_history.append((current_time, direction[0], direction[1], 0.0))

            # Remove directions older than 5 seconds
            cutoff_time = current_time - 5.0
            while self._direction_history and self._direction_history[0][0] < cutoff_time:
                self._direction_history.popleft()

    def get_tip_median_position(self, since: float) -> tuple[float, float] | None:
        """Calculate the median position of the index tip since the given timestamp.

        Args:
            since: Timestamp to start calculating from

        Returns:
            Tuple of (median_x, median_y) or None if no positions available
        """
        # Get positions from the history that occurred since the given timestamp
        positions_since = [(x, y) for t, x, y in self._tip_position_history if t >= since]

        if not positions_since:
            return None

        # Calculate median x and y separately
        x_values = [x for x, y in positions_since]
        y_values = [y for x, y in positions_since]

        # Sort and find median
        x_values.sort()
        y_values.sort()

        n = len(x_values)
        if n % 2 == 0:
            median_x = (x_values[n // 2 - 1] + x_values[n // 2]) / 2
            median_y = (y_values[n // 2 - 1] + y_values[n // 2]) / 2
        else:
            median_x = x_values[n // 2]
            median_y = y_values[n // 2]

        return (median_x, median_y)

    def detect_direction_changes(
        self,
        duration_window: float = 1.0,
        min_direction_changes: int = 2,
        min_movement_angle: float = 2.5,
        x_tolerance: float = 0.05,
        max_time_since_last_change: float = 0.5,
        require_recent_movement: bool = True,
        recent_movement_window: float = 0.3,
    ) -> tuple[bool, list[tuple[Direction, float]]]:
        """Detect oscillating directional changes in index movement.

        Args:
            duration_window: Time window in seconds to analyze movement
            min_direction_changes: Minimum number of direction changes required
            min_movement_angle: Minimum angle in degrees for significant movement
            x_tolerance: Tolerance zone around x=0 to avoid noise
            max_time_since_last_change: Maximum time since last direction change
            require_recent_movement: Whether to require recent movement
            recent_movement_window: Time window for recent movement check

        Returns:
            Tuple of (has_changes, list of (direction, time_ago) pairs)
        """
        if not self._direction_history:
            return False, []

        current_time = time()
        min_movement_angle_rad = radians(min_movement_angle)

        # Find directions within the duration window
        cutoff_time = current_time - duration_window
        directions_in_window = [(t, x, y) for t, x, y, _ in self._direction_history if t >= cutoff_time]

        if len(directions_in_window) < 3:  # Need at least 3 points to detect oscillation
            return False, []

        # Check if we have data covering sufficient duration (at least 80% of window)
        time_coverage = current_time - directions_in_window[0][0]
        if time_coverage < duration_window * 0.8:
            return False, []

        # Detect direction changes based on X component sign changes
        # Track direction changes with their direction
        direction_changes: list[tuple[float, Direction]] = []
        last_significant_x = None

        for i, (t, x, _y) in enumerate(directions_in_window):
            # Skip values in tolerance zone
            if abs(x) < x_tolerance:
                continue

            current_sign = 1 if x > 0 else -1

            if last_significant_x is not None:
                last_sign = 1 if last_significant_x > 0 else -1
                if current_sign != last_sign:
                    # Direction change detected
                    if i > 0:
                        # Determine which direction we changed TO
                        new_direction = Direction.RIGHT if current_sign > 0 else Direction.LEFT
                        direction_changes.append((t, new_direction))

            last_significant_x = x

        # Check if we have enough direction changes
        if len(direction_changes) < min_direction_changes:
            return False, []

        # Check that the last direction change is recent
        # This ensures we stop detecting when hand stops moving
        if direction_changes and max_time_since_last_change > 0:
            last_change_time = direction_changes[-1][0]
            time_since_last_change = current_time - last_change_time
            if time_since_last_change > max_time_since_last_change:
                return False, []

        # Check for recent movement if required
        if require_recent_movement:
            recent_directions = [
                (t, x, y) for t, x, y in directions_in_window if current_time - t <= recent_movement_window
            ]
            if len(recent_directions) < 2:
                return False, []

        # Verify angle changes are significant enough
        # Check angles between consecutive significant directions
        significant_directions = [(x, y) for _, x, y in directions_in_window if abs(x) >= x_tolerance]

        if len(significant_directions) < 3:
            return False, []

        # Check angle between first and middle, middle and last directions
        angle_verified = False
        for i in range(1, len(significant_directions) - 1):
            prev_x, prev_y = significant_directions[i - 1]
            curr_x, curr_y = significant_directions[i]
            next_x, next_y = significant_directions[i + 1]

            # Calculate angles between vectors
            # Angle between prev→curr and curr→next
            dot1 = prev_x * curr_x + prev_y * curr_y
            dot2 = curr_x * next_x + curr_y * next_y

            # Clamp to avoid numerical errors with acos
            dot1 = max(-1.0, min(1.0, dot1))
            dot2 = max(-1.0, min(1.0, dot2))

            angle1 = acos(dot1)
            angle2 = acos(dot2)

            # At least one angle should be significant
            if angle1 >= min_movement_angle_rad or angle2 >= min_movement_angle_rad:
                angle_verified = True
                break

        if not angle_verified:
            return False, []

        # Convert to list of (direction, time_ago) pairs
        changes_with_time_ago = [(direction, current_time - timestamp) for timestamp, direction in direction_changes]

        return True, changes_with_time_ago


class MiddleFinger(OtherFinger):
    index = FingerIndex.MIDDLE
    adjacent_fingers_indexes = (FingerIndex.INDEX, FingerIndex.RING)


class RingFinger(OtherFinger):
    index = FingerIndex.RING
    adjacent_fingers_indexes = (FingerIndex.MIDDLE, FingerIndex.PINKY)


class PinkyFinger(OtherFinger):
    index = FingerIndex.PINKY
    adjacent_fingers_indexes = (FingerIndex.RING,)


FINGER_CLASS: dict[FingerIndex, type[Finger[Any]]] = {
    FingerIndex.THUMB: Thumb,
    FingerIndex.INDEX: IndexFinger,
    FingerIndex.MIDDLE: MiddleFinger,
    FingerIndex.RING: RingFinger,
    FingerIndex.PINKY: PinkyFinger,
}


AnyFinger: TypeAlias = Thumb | IndexFinger | MiddleFinger | RingFinger | PinkyFinger
