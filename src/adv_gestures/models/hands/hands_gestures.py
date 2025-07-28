from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...gestures import TWO_HANDS_GESTURES, Gestures
from ...smoothing import GestureWeights
from .base_gestures import BaseGestureDetector, Range, StatefulMode, direction_matches
from .utils import HandsDirectionalRelationship

if TYPE_CHECKING:
    from .hand import Hand
    from .hands import Hands


class TwoHandsGesturesDetector(BaseGestureDetector["Hands"]):
    gestures_set = TWO_HANDS_GESTURES
    main_directions_range: ClassVar[tuple[Range | None, Range | None] | None] = None
    angle_diff_range: ClassVar[Range | None] = None
    directional_relationships: ClassVar[set[HandsDirectionalRelationship] | None] = None

    def __init__(self, obj: Hands) -> None:
        super().__init__(obj)
        self.hands = obj
        self.left = obj.left
        self.right = obj.right

    def pre_matches(self, detected: GestureWeights) -> bool:
        if not super().pre_matches(detected):
            return False
        if not self.left or not self.right:
            return False

        if self.main_directions_range is not None:
            if not self.hand_matches_direction(self.left, self.main_directions_range[0]):
                return False
            if not self.hand_matches_direction(self.right, self.main_directions_range[1]):
                return False

        if self.angle_diff_range is not None and not direction_matches(
            self.hands.hands_direction_angle_diff, self.angle_diff_range
        ):
            return False

        if self.directional_relationships is not None:
            directional_relationship = self.hands.directional_relationship
            if directional_relationship is None or directional_relationship not in self.directional_relationships:
                return False

        if not self.left_hand_in_good_shape() or not self.right_hand_in_good_shape():
            return False

        return True

    def hand_in_good_shape(self, hand: Hand) -> bool:
        raise NotImplementedError

    def left_hand_in_good_shape(self) -> bool:
        return self.hand_in_good_shape(self.left)

    def right_hand_in_good_shape(self) -> bool:
        return self.hand_in_good_shape(self.right)

    def _matches(self, detected: GestureWeights) -> bool:
        return self.matches(self.hands, self.left, self.right, detected)

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        raise NotImplementedError(f"This method should be implemented in subclass {self.__class__.__name__}.")


class PrayDetector(TwoHandsGesturesDetector):
    gesture = Gestures.PRAY
    main_directions_range = (70, 110), (70, 110)
    angle_diff_range = None, 30
    directional_relationships = {
        HandsDirectionalRelationship.PARALLEL,
        HandsDirectionalRelationship.CONVERGING_INSIDE_FRAME,
        HandsDirectionalRelationship.CONVERGING_OUTSIDE_FRAME,
    }

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return hand.is_showing_side and (
            Gestures.STOP in hand.gestures
            or (hand.index.is_nearly_straight_or_straight and hand.middle.is_nearly_straight_or_straight)
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        return self.hands.hands_are_close


class ClapDetector(TwoHandsGesturesDetector):
    gesture = Gestures.CLAP
    angle_diff_range = None, 30
    directional_relationships = {
        HandsDirectionalRelationship.PARALLEL,
        HandsDirectionalRelationship.CONVERGING_INSIDE_FRAME,
        HandsDirectionalRelationship.CONVERGING_OUTSIDE_FRAME,
    }

    stateful_mode = StatefulMode.POST_DETECTION
    min_gesture_duration = 0.05  # Min duration for valid clap
    max_gesture_duration = 0.5  # Max duration for hands to be joined
    post_detection_duration = 0.2  # Show clap duration after separation

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return (
            Gestures.STOP in hand.gestures
            or Gestures.OPEN_PALM in hand.gestures
            or (hand.index.is_nearly_straight_or_straight and hand.middle.is_nearly_straight_or_straight)
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        """Check if hands are in a position suitable for clapping."""
        return self.hands.hands_are_close


class CrossedFlatDetector(TwoHandsGesturesDetector):
    gesture = Gestures.CROSSED_FLAT
    directional_relationships = {HandsDirectionalRelationship.DIVERGING_CROSSED}

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return (
            hand.index.is_nearly_straight_or_straight
            and hand.middle.is_nearly_straight_or_straight
            and hand.ring.is_nearly_straight_or_straight
            and hand.pinky.is_nearly_straight_or_straight
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        return True


class CrossedFistsDetector(TwoHandsGesturesDetector):
    gesture = Gestures.CROSSED_FISTS
    directional_relationships = {HandsDirectionalRelationship.DIVERGING_CROSSED}

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return (
            hand.index.is_not_straight_at_all
            and hand.middle.is_not_straight_at_all
            and hand.ring.is_not_straight_at_all
            and hand.pinky.is_not_straight_at_all
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        return True


class TimeOutDetector(TwoHandsGesturesDetector):
    gesture = Gestures.TIME_OUT
    angle_diff_range = 70, 110
    directional_relationships = {
        HandsDirectionalRelationship.LEFT_INTO_RIGHT,
        HandsDirectionalRelationship.RIGHT_INTO_LEFT,
    }

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return (
            hand.index.is_nearly_straight_or_straight
            and hand.middle.is_nearly_straight_or_straight
            and hand.ring.is_nearly_straight_or_straight
            and hand.pinky.is_nearly_straight_or_straight
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        # First check if the oriented bounding boxes overlap
        if not hands.bounding_boxes_overlap:
            return False

        # Determine which hand is pointing at the other based on directional relationship
        pointing_hand: Hand | None = None
        if hands.directional_relationship == HandsDirectionalRelationship.LEFT_INTO_RIGHT:
            pointing_hand = left
        elif hands.directional_relationship == HandsDirectionalRelationship.RIGHT_INTO_LEFT:
            pointing_hand = right

        if pointing_hand is None:
            return False

        # Check if the pointing hand's ray hits roughly the middle of the other hand
        intersection = pointing_hand.other_hand_line_intersection
        if intersection is None:
            return False

        # Check if intersection is in the middle part of the other hand
        return 0.4 <= intersection <= 0.8


TwoHandsGesturesDetector._ensure_all_detectors_registered()
