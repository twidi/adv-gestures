from __future__ import annotations

from functools import cached_property
from math import sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar

from ...config import Config
from ...gestures import Gestures
from ...smoothing import (
    EnumSmoother,
    GestureWeights,
    MultiGestureSmoother,
    SmoothedBase,
    SmoothedProperty,
    smoothed_bool,
    smoothed_optional_float,
)
from .hand import Hand
from .hands_gestures import TwoHandsGesturesDetector
from .utils import Handedness, HandsDirectionalRelationship

if TYPE_CHECKING:
    from ...recognizer import Recognizer, StreamInfo


class Hands(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "gestures",
        "gestures_durations",
        "hands_distance",
        "hands_are_close",
        "hands_direction_angle_diff",
        "directional_relationship",
    )

    def __init__(self, config: Config) -> None:
        """Initialize both hands."""
        super().__init__()
        self.config = config
        self.stream_info: StreamInfo | None = None
        self.left: Hand = Hand(handedness=Handedness.LEFT, config=config)
        self.right: Hand = Hand(handedness=Handedness.RIGHT, config=config)

        self._raw_gestures: GestureWeights = {}
        self._gestures_start_times: dict[Gestures, float] = {}
        self._last_gestures: set[Gestures] = set()

        self.gestures_detector = TwoHandsGesturesDetector(self)

    def reset(self) -> None:
        """Reset both hands and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        self.left.reset()
        self.right.reset()

        # Clear cached properties by deleting them from __dict__
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update_hands(self, recognizer: Recognizer, stream_info: StreamInfo | None = None) -> None:
        """Update the hands object with new gesture recognition results."""
        # Store the stream info
        self.stream_info = stream_info

        # If the recognizer didn't run yet, do nothing
        if not (result := recognizer.last_result):
            return

        # Reset all hands first
        self.reset()

        if result.hand_landmarks:
            for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                # Get handedness
                handedness = None
                if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
                    handedness = Handedness.from_data(result.handedness[hand_index][0].category_name)

                # Skip if handedness not detected
                if not handedness:
                    continue

                # Get the appropriate hand
                hand = self.left if handedness == Handedness.LEFT else self.right

                # Get default gesture information
                gesture_type = None
                if (
                    hand_landmarks is not None
                    and result.gestures
                    and hand_index < len(result.gestures)
                    and result.gestures[hand_index]
                ):
                    if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.default.disable_all):
                        gesture = result.gestures[hand_index][0]  # Get the top gesture
                        gesture_type = (
                            None
                            if gesture.category_name in (None, "None", "Unknown")
                            else Gestures(gesture.category_name)
                        )

                # Update hand data
                hand.update(
                    default_gesture=gesture_type,
                    all_landmarks=hand_landmarks,
                    stream_info=stream_info,
                )

        for hand in (self.left, self.right):
            if not hand.is_visible:
                hand.update(
                    default_gesture=None,
                    all_landmarks=None,
                    stream_info=stream_info,
                )

        # We do it even if not both hands are visible because we may have smoothed gestures
        gestures: GestureWeights = {}
        if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.two_hands.disable.all):  # type: ignore[attr-defined]
            gestures = self.detect_gestures()
        self._raw_gestures = gestures

    def _calc_gestures(self) -> GestureWeights:
        """Get the custom gestures if detected."""
        current_gestures = set(self._raw_gestures.keys())
        now = time()

        # Detect new gestures
        new_gestures = current_gestures - self._last_gestures
        for gesture in new_gestures:
            self._gestures_start_times[gesture] = now

        # Remove ended gestures
        ended_gestures = self._last_gestures - current_gestures
        for gesture in ended_gestures:
            self._gestures_start_times.pop(gesture, None)

        self._last_gestures = current_gestures
        return self._raw_gestures

    gestures = SmoothedProperty(_calc_gestures, MultiGestureSmoother, default_value={})

    @cached_property
    def gestures_durations(self) -> dict[Gestures, float]:
        """Get the durations for all currently active gestures."""
        now = time()
        return {
            gesture: now - start_time
            for gesture, start_time in self._gestures_start_times.items()
            if gesture in self.gestures
        }

    def is_gesture_disabled(self, gesture: Gestures) -> bool:
        return bool(getattr(self.config.hands.gestures.two_hands.disable, gesture.name))

    def detect_gestures(self) -> GestureWeights:
        """Detect all applicable custom gestures with weights.

        Returns:
            Dictionary of detected gestures with weight 1.0 for each
        """
        return self.gestures_detector.detect()

    def _calc_hands_distance(self) -> float | None:
        """Calculate the distance in pixels between the two hands' palm centroids."""
        if not self.left or not self.right:
            return None

        left_centroid = self.left.palm.centroid
        right_centroid = self.right.palm.centroid

        if not left_centroid or not right_centroid:
            return None

        # Calculate Euclidean distance between centroids
        dx = right_centroid[0] - left_centroid[0]
        dy = right_centroid[1] - left_centroid[1]

        return sqrt(dx * dx + dy * dy)

    hands_distance = smoothed_optional_float(_calc_hands_distance)

    def _calc_hands_are_close(self) -> bool:
        """Check if the two hands are close based on their distance."""
        distance = self.hands_distance
        if distance is None:
            return False

        # Calculate the average distance from wrist to thumb MCP for both hands
        threshold = 0.0
        count = 0

        for hand in (self.left, self.right):
            if hand and hand.wrist_landmark and hand.thumb and len(hand.thumb.landmarks) > 1:
                # Thumb MCP is the second landmark (index 1)
                thumb_mcp = hand.thumb.landmarks[1]
                wrist = hand.wrist_landmark
                dx = thumb_mcp.x - wrist.x
                dy = thumb_mcp.y - wrist.y

                threshold += sqrt(dx * dx + dy * dy)
                count += 1

        if count == 0:
            return False

        # Average threshold
        threshold = threshold / count

        # Hands are close if distance is less than the average wrist->thumb_mcp distance
        return distance < threshold

    hands_are_close = smoothed_bool(_calc_hands_are_close)

    def _calc_hands_direction_angle_diff(self) -> float | None:
        """Calculate the angle difference between the two hands' main directions.
        Returns the angle difference in degrees (0-180), or None if either hand direction is unavailable.
        """
        if not self.left or not self.right:
            return None

        left_angle = self.left.main_direction_angle
        right_angle = self.right.main_direction_angle

        if left_angle is None or right_angle is None:
            return None

        # Calculate angle difference, handling wrap-around at ±180°
        angle_diff = abs(left_angle - right_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    hands_direction_angle_diff = smoothed_optional_float(_calc_hands_direction_angle_diff)

    def _ray_box_intersection(
        self,
        ray_origin: tuple[float, float],
        ray_direction: tuple[float, float],
        box_min: tuple[float, float],
        box_max: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Calculate where a ray exits a box.

        Returns the exit point or None if the ray doesn't intersect the box.
        """
        # Unpack coordinates
        ox, oy = ray_origin
        dx, dy = ray_direction
        min_x, min_y = box_min
        max_x, max_y = box_max

        # Calculate t values for each box boundary
        t_values = []

        # Check intersection with each edge
        # Left edge (x = min_x)
        if dx != 0:
            t = (min_x - ox) / dx
            if t > 0:
                y = oy + t * dy
                if min_y <= y <= max_y:
                    t_values.append((t, min_x, y))

        # Right edge (x = max_x)
        if dx != 0:
            t = (max_x - ox) / dx
            if t > 0:
                y = oy + t * dy
                if min_y <= y <= max_y:
                    t_values.append((t, max_x, y))

        # Bottom edge (y = min_y)
        if dy != 0:
            t = (min_y - oy) / dy
            if t > 0:
                x = ox + t * dx
                if min_x <= x <= max_x:
                    t_values.append((t, x, min_y))

        # Top edge (y = max_y)
        if dy != 0:
            t = (max_y - oy) / dy
            if t > 0:
                x = ox + t * dx
                if min_x <= x <= max_x:
                    t_values.append((t, x, max_y))

        # Return the furthest intersection point (exit point)
        if t_values:
            # Sort by t value and get the last one
            t_values.sort(key=lambda x: x[0])
            _, x, y = t_values[-1]
            return (x, y)

        return None

    def _segments_intersect(
        self, p1: tuple[float, float], q1: tuple[float, float], p2: tuple[float, float], q2: tuple[float, float]
    ) -> bool:
        """Check if two line segments intersect."""

        def ccw(A: tuple[float, float], B: tuple[float, float], C: tuple[float, float]) -> bool:
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p2, q2) != ccw(q1, p2, q2) and ccw(p1, q1, p2) != ccw(p1, q1, q2)

    def _ray_segment_intersect(
        self,
        ray_origin: tuple[float, float],
        ray_direction: tuple[float, float],
        segment_start: tuple[float, float],
        segment_end: tuple[float, float],
    ) -> bool:
        """Check if a ray intersects with a line segment."""
        # Ray: P = ray_origin + t * ray_direction (t > 0)
        # Segment: parametric form from segment_start to segment_end

        ox, oy = ray_origin
        dx, dy = ray_direction
        sx, sy = segment_start
        ex, ey = segment_end

        # Segment direction
        seg_dx = ex - sx
        seg_dy = ey - sy

        # Solve ray_origin + t * ray_direction = segment_start + s * segment_direction
        # Cross product to check if parallel
        cross = dx * seg_dy - dy * seg_dx

        if abs(cross) < 1e-10:
            return False  # Parallel

        # Calculate parameters
        t = ((sx - ox) * seg_dy - (sy - oy) * seg_dx) / cross
        s = ((sx - ox) * dy - (sy - oy) * dx) / cross

        # Check if intersection is valid
        # t > 0 means on the ray (not behind)
        # 0 <= s <= 1 means on the segment
        return t > 0 and 0 <= s <= 1

    def _ray_box_intersect(
        self,
        ray_origin: tuple[float, float],
        ray_direction: tuple[float, float],
        box_min: tuple[float, float],
        box_max: tuple[float, float],
    ) -> bool:
        """Check if a ray passes through a box."""
        ox, oy = ray_origin
        dx, dy = ray_direction
        min_x, min_y = box_min
        max_x, max_y = box_max

        # Calculate t values for entering and exiting the box
        t_min = float("-inf")
        t_max = float("inf")

        # Check X bounds
        if dx != 0:
            t1 = (min_x - ox) / dx
            t2 = (max_x - ox) / dx
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is parallel to X axis
            if ox < min_x or ox > max_x:
                return False

        # Check Y bounds
        if dy != 0:
            t1 = (min_y - oy) / dy
            t2 = (max_y - oy) / dy
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is parallel to Y axis
            if oy < min_y or oy > max_y:
                return False

        # Check if there's a valid intersection
        return t_max >= t_min and t_max > 0

    def _calc_directional_relationship(self) -> HandsDirectionalRelationship | None:
        """Determine the directional relationship between two hands."""
        if not self.left or not self.right:
            return None

        # Get wrist positions and main directions for both hands
        if not (left_wrist := self.left.wrist_landmark):
            return None
        if not (right_wrist := self.right.wrist_landmark):
            return None
        if not (left_direction := self.left.main_direction):
            return None
        if not (right_direction := self.right.main_direction):
            return None
        if not (left_box := self.left.bounding_box):
            return None
        if not (right_box := self.right.bounding_box):
            return None

        # Extract coordinates
        p1_x, p1_y = left_wrist.x, left_wrist.y  # Left hand wrist position
        d1_x, d1_y = left_direction  # Left hand direction vector

        p2_x, p2_y = right_wrist.x, right_wrist.y  # Right hand wrist position
        d2_x, d2_y = right_direction  # Right hand direction vector

        # First check if rays pass through opposite bounding boxes
        if self._ray_box_intersect(
            (p1_x, p1_y), (d1_x, d1_y), (right_box.min_x, right_box.min_y), (right_box.max_x, right_box.max_y)
        ):
            return HandsDirectionalRelationship.LEFT_INTO_RIGHT

        if self._ray_box_intersect(
            (p2_x, p2_y), (d2_x, d2_y), (left_box.min_x, left_box.min_y), (left_box.max_x, left_box.max_y)
        ):
            return HandsDirectionalRelationship.RIGHT_INTO_LEFT

        # Get exit points from bounding boxes
        left_exit = self._ray_box_intersection(
            (p1_x, p1_y), (d1_x, d1_y), (left_box.min_x, left_box.min_y), (left_box.max_x, left_box.max_y)
        )

        right_exit = self._ray_box_intersection(
            (p2_x, p2_y), (d2_x, d2_y), (right_box.min_x, right_box.min_y), (right_box.max_x, right_box.max_y)
        )

        # Check if segments intersect (only if we have exit points)
        if left_exit and right_exit:
            if self._segments_intersect((p1_x, p1_y), left_exit, (p2_x, p2_y), right_exit):
                return HandsDirectionalRelationship.INTERSECTING

        # Check if hands are parallel (less than 1 degree difference)
        left_angle = self.left.main_direction_angle
        right_angle = self.right.main_direction_angle
        if left_angle is not None and right_angle is not None:
            angle_diff = abs(left_angle - right_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff < 1:
                return HandsDirectionalRelationship.PARALLEL

        # Check if lines are mathematically parallel
        cross_product = d1_x * d2_y - d1_y * d2_x
        epsilon = 1e-10
        if abs(cross_product) < epsilon:
            # Lines are parallel but not pointing in the same direction
            # This is a special case of diverging
            return HandsDirectionalRelationship.DIVERGING_NORMAL

        # Calculate ray intersection parameters to determine converging/diverging
        t = ((p2_x - p1_x) * d2_y - (p2_y - p1_y) * d2_x) / cross_product
        s = ((p2_x - p1_x) * d1_y - (p2_y - p1_y) * d1_x) / cross_product

        # Determine the type based on t and s values
        if t > 0 and s > 0:
            return HandsDirectionalRelationship.CONVERGING
        elif t < 0 and s < 0:
            # Both intersect behind - check if wrists are crossed
            if left_angle is not None and right_angle is not None:
                # Normalize angles with +180 and modulo 360
                left_rotated = (left_angle + 180) % 360
                right_rotated = (right_angle + 180) % 360

                if left_rotated < right_rotated:
                    return HandsDirectionalRelationship.DIVERGING_NORMAL
                else:
                    return HandsDirectionalRelationship.DIVERGING_CROSSED
            else:
                # Fallback to position-based check
                if p1_x <= p2_x:
                    return HandsDirectionalRelationship.DIVERGING_NORMAL
                else:
                    return HandsDirectionalRelationship.DIVERGING_CROSSED
        elif t < 0 and s > 0:
            return HandsDirectionalRelationship.DIVERGING_LEFT_BEHIND_RIGHT
        else:  # t > 0 and s < 0
            return HandsDirectionalRelationship.DIVERGING_RIGHT_BEHIND_LEFT

    directional_relationship = SmoothedProperty(
        _calc_directional_relationship, EnumSmoother[HandsDirectionalRelationship | None], default_value=None
    )
