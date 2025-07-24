# Finger straightness thresholds - can be customized per finger
from __future__ import annotations

from enum import IntEnum
from functools import cached_property
from math import sqrt
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeAlias, TypeVar, cast

import numpy as np

from ..config import BaseFingerConfig, Config, FingerConfig, ThumbConfig
from ..gestures import Gestures
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
        "touches_thumb",
        "touching_adjacent_fingers",
    )

    index: ClassVar[FingerIndex]

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
        other_index = other.index if isinstance(other, Finger) else other
        return abs(self.index - other_index) == 1

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
        # Handle gesture-based shortcuts first
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return False
        elif self.hand.gesture == Gestures.OPEN_PALM:
            if self.index != FingerIndex.THUMB:
                return True
        elif self.hand.gesture == Gestures.POINTING_UP:
            if self.index == FingerIndex.INDEX:
                return True
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            return self.index == FingerIndex.THUMB
        elif self.hand.gesture == Gestures.VICTORY:
            return self.index in (FingerIndex.INDEX, FingerIndex.MIDDLE)

        # Use the straightness score with strict threshold from finger config
        return self.straightness_score >= self.finger_config.straightness.straight_threshold

    is_straight = smoothed_bool(_calc_is_straight)

    def _calc_is_nearly_straight(self) -> bool:
        """Check if the finger is nearly straight based on the straightness score."""
        # Handle gesture-based shortcuts first (same as is_straight for consistency)
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return False
        elif self.hand.gesture == Gestures.OPEN_PALM:
            if self.index != FingerIndex.THUMB:
                return False
        elif self.hand.gesture == Gestures.POINTING_UP:
            if self.index == FingerIndex.INDEX:
                return False
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            if self.index == FingerIndex.THUMB:
                return False
        elif self.hand.gesture == Gestures.VICTORY:
            if self.index in (FingerIndex.INDEX, FingerIndex.MIDDLE):
                return False

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
        return self.landmarks[0].x, self.landmarks[0].y

    start_point = SmoothedProperty(_calc_start_point, CoordSmoother)

    def _calc_end_point(self) -> tuple[int, int] | None:
        """Get the end point of the finger (tip)."""
        if not self.landmarks:
            return None
        return self.landmarks[-1].x, self.landmarks[-1].y

    end_point = SmoothedProperty(_calc_end_point, CoordSmoother)

    def _calc_fold_angle(self) -> float | None:
        """Calculate the fold angle at the PIP joint (angle between MCP->PIP and PIP->TIP vectors).
        Returns angle in degrees. 180 = straight, lower angles = more bent."""
        if len(self.landmarks) < 3:
            return None

        # For thumb, use CMC->IP->TIP instead of MCP->PIP->TIP
        if self.index == FingerIndex.THUMB:
            if len(self.landmarks) <= 3:
                return None
            mcp, pip, tip = self.landmarks[0], self.landmarks[1], self.landmarks[3]
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

        if len(self.landmarks) < 2:
            return None

        # Get the first and last points
        first = self.landmarks[0]
        last = self.landmarks[-1]

        # Calculate direction vector
        dx = last.x - first.x
        dy = last.y - first.y

        # Normalize the vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    straight_direction = SmoothedProperty(_calc_straight_direction, CoordSmoother)

    def _calc_is_fully_bent(self) -> bool:
        """Check if the finger is fully bent based on fold angle threshold."""
        # Check gesture-based shortcuts first
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return True
        elif self.hand.gesture == Gestures.OPEN_PALM:
            return False
        elif self.hand.gesture == Gestures.POINTING_UP:
            return self.index != FingerIndex.INDEX
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            return self.index != FingerIndex.THUMB
        elif self.hand.gesture == Gestures.VICTORY:
            return self.index not in (FingerIndex.INDEX, FingerIndex.MIDDLE)

        # Fall back to fold angle check
        if self.fold_angle is None:
            return True

        return self.fold_angle < self.finger_config.fully_bent_max_angle_degrees

    is_fully_bent = smoothed_bool(_calc_is_fully_bent)

    @cached_property
    def touching_adjacent_fingers(self) -> list[FingerIndex]:
        raise NotImplementedError

    def is_touching(self, other: FingerIndex | AnyFinger) -> bool:
        """Check if this finger is touching another finger."""
        other_index = other.index if isinstance(other, Finger) else other

        if self.index == other_index:
            return True  # A finger is always touching itself

        # Special case: checking if a finger touches the thumb
        if self.index == FingerIndex.THUMB:
            # Get the other finger and check its touches_thumb property
            other_finger = cast(OtherFinger, self.hand.fingers[other_index])
            return other_finger.touches_thumb

        if other_index == FingerIndex.THUMB:
            # This finger checking if it touches thumb
            return cast(OtherFinger, self).touches_thumb

        # For non-thumb combinations, use smoothed properties
        # Order fingers to avoid duplicates (always use lower index first)
        finger1, finger2 = sorted((self.index, other_index))

        if finger1 == FingerIndex.INDEX:
            if finger2 == FingerIndex.MIDDLE:
                return self.hand.index_middle_touching
            if finger2 == FingerIndex.RING:
                return self.hand.index_ring_touching
            if finger2 == FingerIndex.PINKY:
                return self.hand.index_pinky_touching

        if finger1 == FingerIndex.MIDDLE:
            if finger2 == FingerIndex.RING:
                return self.hand.middle_ring_touching
            if finger2 == FingerIndex.PINKY:
                return self.hand.middle_pinky_touching

        if finger1 == FingerIndex.RING:
            if finger2 == FingerIndex.PINKY:
                return self.hand.ring_pinky_touching

        # Should never reach here if all combinations are covered
        return False


class Thumb(Finger[ThumbConfig]):
    index = FingerIndex.THUMB

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

    @cached_property
    def touching_adjacent_fingers(self) -> list[FingerIndex]:
        """Get list of adjacent fingers that this finger is touching."""
        return [FingerIndex.INDEX] if self.hand.index.touches_thumb else []


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

    def _calc_touches_thumb(self) -> bool:
        """Check if this finger tip touches the thumb tip of the same hand."""

        # Get the thumb finger
        thumb = self.hand.thumb

        if not thumb or not thumb.landmarks or not self.landmarks:
            return False

        # Get the tip landmarks (last landmark of each finger)
        thumb_tip = thumb.landmarks[-1]
        finger_tip = self.landmarks[-1]

        # Calculate distance (coordinates already include aspect ratio correction)
        dx = finger_tip.x - thumb_tip.x
        dy = finger_tip.y - thumb_tip.y
        distance = sqrt(dx**2 + dy**2)

        # Calculate hand scale using the current finger's DIP-TIP segment length
        # This gives us a reference that scales with hand distance from camera
        hand_scale = 1.0

        if len(self.landmarks) >= 4:
            dip = self.landmarks[-2]  # DIP
            tip = self.landmarks[-1]  # TIP
            # Calculate segment length (coordinates already include aspect ratio)
            segment_dx = tip.x - dip.x
            segment_dy = tip.y - dip.y
            hand_scale = sqrt(segment_dx**2 + segment_dy**2)

        # Touch threshold relative to hand scale
        relative_threshold = self.finger_config.thumb_distance_relative_threshold * hand_scale

        return distance < relative_threshold

    touches_thumb = smoothed_bool(_calc_touches_thumb)

    @cached_property
    def touching_adjacent_fingers(self) -> list[FingerIndex]:
        """Get list of adjacent fingers that this finger is touching."""

        touching = []

        # Define adjacent fingers based on finger index
        if self.index == FingerIndex.INDEX:
            # Index can only touch middle
            if self.hand.index_middle_touching:
                touching.append(FingerIndex.MIDDLE)

        elif self.index == FingerIndex.MIDDLE:
            # Middle can touch index and ring
            if self.hand.index_middle_touching:
                touching.append(FingerIndex.INDEX)
            if self.hand.middle_ring_touching:
                touching.append(FingerIndex.RING)

        elif self.index == FingerIndex.RING:
            # Ring can touch middle and pinky
            if self.hand.middle_ring_touching:
                touching.append(FingerIndex.MIDDLE)
            if self.hand.ring_pinky_touching:
                touching.append(FingerIndex.PINKY)

        elif self.index == FingerIndex.PINKY:
            # Pinky can only touch ring
            if self.hand.ring_pinky_touching:
                touching.append(FingerIndex.RING)

        return touching


class IndexFinger(OtherFinger):
    index = FingerIndex.INDEX


class MiddleFinger(OtherFinger):
    index = FingerIndex.MIDDLE


class RingFinger(OtherFinger):
    index = FingerIndex.RING


class PinkyFinger(OtherFinger):
    index = FingerIndex.PINKY


FINGER_CLASS: dict[FingerIndex, type[Finger[Any]]] = {
    FingerIndex.THUMB: Thumb,
    FingerIndex.INDEX: IndexFinger,
    FingerIndex.MIDDLE: MiddleFinger,
    FingerIndex.RING: RingFinger,
    FingerIndex.PINKY: PinkyFinger,
}


AnyFinger: TypeAlias = Thumb | IndexFinger | MiddleFinger | RingFinger | PinkyFinger
