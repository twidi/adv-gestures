# Finger straightness thresholds - can be customized per finger
from __future__ import annotations

from collections import deque
from enum import IntEnum
from functools import cached_property
from math import sqrt
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

if TYPE_CHECKING:
    from .hands import Hand

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
        "is_fully_bent",
        "fold_angle",
        "tip_direction",
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
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

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
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    tip_direction = SmoothedProperty(_calc_tip_direction, CoordSmoother)

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
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    straight_direction = SmoothedProperty(_calc_straight_direction, CoordSmoother)

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
        thumb_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

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

        return float(score)

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
            segment_length = np.sqrt(dx**2 + dy**2)
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
                    segment_ratio_score = segment_ratio_score_at_threshold * np.exp(
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
                mag1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)

                if mag1 > min_magnitude_threshold and mag2 > min_magnitude_threshold:  # Avoid division by zero
                    cos_angle = dot_product / (mag1 * mag2)
                    # Clamp to [-1, 1] to handle numerical errors
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = np.arccos(cos_angle)

                    # Convert to degrees for easier interpretation
                    angle_degrees = np.degrees(angle)
                    max_found_angle = max(max_found_angle, angle_degrees)

        # Convert angle to score: 0° = 1.0, max_angle_degrees = angle_score_at_threshold, higher angles decay to 0
        if max_found_angle <= max_angle_degrees:
            # Perfect to acceptable range: linear from 1.0 to angle_score_at_threshold
            angle_score = 1.0 - (max_found_angle / max_angle_degrees) * angle_score_linear_range
        else:
            # Beyond threshold: exponential decay from angle_score_at_threshold to 0
            excess_angle = max_found_angle - max_angle_degrees
            # Use exponential decay: score decreases rapidly after threshold
            angle_score = angle_score_at_threshold * np.exp(-excess_angle / angle_decay_rate)

        # Combine both scores (weighted average: angle is more important)
        combined_score = angle_score_weight * angle_score + segment_ratio_weight * segment_ratio_score

        return float(combined_score)

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

        # Calculate hand scale using the thumb's IP-TIP segment length
        # This gives us a reference that scales with hand distance from camera
        hand_scale = 1.0

        if len(thumb.landmarks) >= 4:
            # For thumb: IP is at index 2, TIP is at index 3
            ip = thumb.landmarks[2]  # IP
            tip = thumb.landmarks[3]  # TIP
            # Calculate segment length (coordinates already include aspect ratio)
            segment_dx = tip[0] - ip[0]
            segment_dy = tip[1] - ip[1]
            hand_scale = sqrt(segment_dx**2 + segment_dy**2)

        # Touch threshold relative to hand scale
        relative_threshold = self.finger_config.thumb_distance_relative_threshold * hand_scale

        return distance < relative_threshold

    tip_on_thumb = smoothed_bool(_calc_tip_on_thumb)


class IndexFinger(OtherFinger):
    index = FingerIndex.INDEX
    adjacent_fingers_indexes = (FingerIndex.THUMB, FingerIndex.MIDDLE)

    def __init__(self, hand: Hand, config: Config):
        super().__init__(hand, config)
        self.finger_config: IndexConfig = config.hands.index
        self._tip_position_history: deque[tuple[float, float, float]] = deque()

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

    def _calc_is_tip_stable(self) -> bool:
        """Check if the tip has been stable for the configured duration."""
        if not self._tip_position_history:
            return False

        current_time = time()
        duration = self.finger_config.tip_stability.duration
        threshold = self.finger_config.tip_stability.threshold

        # Find positions within the duration window
        cutoff_time = current_time - duration
        positions_in_window = [(t, x, y) for t, x, y in self._tip_position_history if t >= cutoff_time]

        if not positions_in_window or len(positions_in_window) < 2:
            return False

        # Check if we have data covering the full duration
        if current_time - positions_in_window[0][0] < duration * 0.9:  # 90% of duration
            return False

        # Calculate maximum movement in the window
        max_movement = 0.0
        first_x, first_y = positions_in_window[0][1], positions_in_window[0][2]

        for _, x, y in positions_in_window:
            dx = x - first_x
            dy = y - first_y
            movement = sqrt(dx**2 + dy**2)
            max_movement = max(max_movement, movement)

        # Normalize max_movement from pixels to normalized coordinates
        # We need the frame dimensions from the hand's stream_info
        normalized_movement = max_movement
        if self.hand.stream_info:
            # Calculate diagonal to normalize the movement
            diagonal = sqrt(self.hand.stream_info.width**2 + self.hand.stream_info.height**2)
            normalized_movement = max_movement / diagonal

        return normalized_movement < threshold

    is_tip_stable = smoothed_bool(_calc_is_tip_stable)


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
