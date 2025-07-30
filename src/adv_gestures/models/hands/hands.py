from __future__ import annotations

from functools import cached_property
from math import inf, sqrt
from time import time
from typing import TYPE_CHECKING, Any, ClassVar

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
from ..utils import Handedness, HandsDirectionalRelationship, oriented_boxes_overlap
from .hand import Hand
from .hands_gestures import TwoHandsGesturesDetector

if TYPE_CHECKING:
    from ...recognizer import Recognizer, StreamInfo


class Hands(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "gestures",
        "gestures_durations",
        "gestures_data",
        "hands_distance",
        "hands_are_close",
        "hands_direction_angle_diff",
        "directional_relationship",
        "bounding_boxes_overlap",
        "oriented_bounding_boxes_overlap",
    )

    def __init__(self, config: Config) -> None:
        """Initialize both hands."""
        super().__init__()
        self.config = config
        self.stream_info: StreamInfo | None = None
        self.left: Hand = Hand(handedness=Handedness.LEFT, hands=self, config=config)
        self.right: Hand = Hand(handedness=Handedness.RIGHT, hands=self, config=config)

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
            if not hand:
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

    @cached_property
    def gestures_data(self) -> dict[Gestures, dict[str, Any]]:
        """Get the data from gesture detectors for all currently active gestures."""
        return {
            gesture: data
            for gesture in self.gestures
            if (data := self.gestures_detector.detectors[gesture].get_data()) is not None
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

    def _calc_directional_relationship(self) -> HandsDirectionalRelationship | None:
        """Determine the directional relationship between two hands using ray alignments."""
        if not self.left or not self.right:
            return None
        if not self.left.wrist_landmark or not self.right.wrist_landmark:
            return None

        if (left_angle := self.left.main_direction_angle) is None:
            return None
        if (right_angle := self.right.main_direction_angle) is None:
            return None

        # Check if hands are parallel first (less than 1 degree difference)
        if self.hands_direction_angle_diff is not None and self.hands_direction_angle_diff < 1:
            return HandsDirectionalRelationship.PARALLEL

        # Get ray alignment values for both hands
        left_align = self.left.other_hand_line_intersection
        right_align = self.right.other_hand_line_intersection
        no_left_align = left_align is None
        no_right_align = right_align is None

        # DIVERGING NORMAL/CROSSED: both None
        if no_left_align and no_right_align:
            # Need to determine normal vs crossed
            # Normalize angles with +180 and modulo 360
            left_rotated = (left_angle + 180) % 360
            right_rotated = (right_angle + 180) % 360

            if left_rotated < right_rotated:
                return HandsDirectionalRelationship.DIVERGING_CROSSED
            else:
                return HandsDirectionalRelationship.DIVERGING_NORMAL

        has_left_align = not no_left_align
        has_right_align = not no_right_align

        left_in_range = has_left_align and 0 <= left_align <= 1  # type: ignore[operator]
        right_in_range = has_right_align and 0 <= right_align <= 1  # type: ignore[operator]

        # CONVERGING cases
        if has_left_align and has_right_align:
            # Both > 1 => converging
            if left_align > 1 and right_align > 1:  # type: ignore[operator]
                # Check if either is infinity
                if left_align == inf or right_align == inf:
                    return HandsDirectionalRelationship.CONVERGING_OUTSIDE_FRAME
                else:
                    return HandsDirectionalRelationship.CONVERGING_INSIDE_FRAME

            # Both in [0, 1] => intersecting
            if left_in_range and right_in_range:
                return HandsDirectionalRelationship.INTERSECTING

        # DIVERGING BEHIND cases
        if no_left_align and has_right_align and right_align < 0:  # type: ignore[operator]
            # Right ray passes behind left
            if right_align == -inf:
                return HandsDirectionalRelationship.DIVERGING_RIGHT_BEHIND_LEFT_OUTSIDE_FRAME
            else:
                return HandsDirectionalRelationship.DIVERGING_RIGHT_BEHIND_LEFT_INSIDE_FRAME

        if no_right_align and has_left_align and left_align < 0:  # type: ignore[operator]
            # Left ray passes behind right
            if left_align == -inf:
                return HandsDirectionalRelationship.DIVERGING_LEFT_BEHIND_RIGHT_OUTSIDE_FRAME
            else:
                return HandsDirectionalRelationship.DIVERGING_LEFT_BEHIND_RIGHT_INSIDE_FRAME

        # INTO cases
        if has_left_align and left_align > 1 and right_in_range:  # type: ignore[operator]
            return HandsDirectionalRelationship.RIGHT_INTO_LEFT

        if has_right_align and right_align > 1 and left_in_range:  # type: ignore[operator]
            return HandsDirectionalRelationship.LEFT_INTO_RIGHT

        # Default fallback
        return None

    directional_relationship = SmoothedProperty(
        _calc_directional_relationship, EnumSmoother[HandsDirectionalRelationship | None], default_value=None
    )

    @cached_property
    def bounding_boxes_overlap(self) -> bool | None:
        """Check if the bounding boxes of both hands overlap.
        Returns True if overlapping, False if both present but not overlapping, None if either hand not visible.
        """
        if not self.left or not self.right:
            return None

        left_box = self.left.bounding_box
        right_box = self.right.bounding_box

        if not left_box or not right_box:
            return None

        return left_box.overlaps(right_box)

    @cached_property
    def oriented_bounding_boxes_overlap(self) -> bool | None:
        """Check if the oriented bounding boxes of both hands overlap.
        Returns True if overlapping, False if both present but not overlapping, None if either hand not visible.
        """
        if not self.left or not self.right:
            return None

        left_corners = self.left.oriented_bounding_box
        right_corners = self.right.oriented_bounding_box

        if not left_corners or not right_corners:
            return None

        return oriented_boxes_overlap(left_corners, right_corners)

    def to_dict(self) -> dict[str, Any]:
        """Export hands data as a dictionary."""
        return {
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "gestures": {gesture.name: weight for gesture, weight in self.gestures.items()},
            "gestures_durations": {gesture.name: duration for gesture, duration in self.gestures_durations.items()},
            "gestures_data": {gesture.name: data for gesture, data in self.gestures_data.items()},
            "hands_distance": self.hands_distance,
            "hands_are_close": self.hands_are_close,
            "hands_direction_angle_diff": self.hands_direction_angle_diff,
            "directional_relationship": self.directional_relationship.name if self.directional_relationship else None,
            "bounding_boxes_overlap": self.bounding_boxes_overlap,
            "oriented_bounding_boxes_overlap": self.oriented_bounding_boxes_overlap,
            "stream_info": self.stream_info.to_dict() if self.stream_info else None,
        }
