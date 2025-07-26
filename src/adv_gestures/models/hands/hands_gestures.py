from __future__ import annotations

from typing import TYPE_CHECKING

from ...gestures import TWO_HANDS_GESTURES, Gestures
from ...smoothing import GestureWeights
from .base_gestures import (
    BaseGestureDetector,
    DirectionMatcher,
    StatefulMode,
    up_with_tolerance,
)

if TYPE_CHECKING:
    from .hand import Hand
    from .hands import Hands


class TwoHandsGesturesDetector(BaseGestureDetector["Hands"]):
    gestures_set = TWO_HANDS_GESTURES

    def __init__(self, obj: Hands) -> None:
        super().__init__(obj)
        self.hands = obj
        self.left = obj.left
        self.right = obj.right

    def matches_main_direction(self, main_direction_range: DirectionMatcher) -> bool:
        return self.hand_matches_direction(self.left, main_direction_range) and self.hand_matches_direction(
            self.right, main_direction_range
        )

    def pre_matches(self, detected: GestureWeights) -> bool:
        return (
            super().pre_matches(detected)
            and bool(self.left and self.right)
            and self.left_hand_in_good_shape()
            and self.right_hand_in_good_shape()
        )

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
    main_direction_range = up_with_tolerance(20)

    def hand_in_good_shape(self, hand: Hand) -> bool:
        return hand.is_showing_side and (
            Gestures.STOP in hand.gestures
            or (hand.index.is_nearly_straight_or_straight and hand.middle.is_nearly_straight_or_straight)
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        return self.hands.hands_are_close


class ClapDetector(TwoHandsGesturesDetector):
    gesture = Gestures.CLAP
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


TwoHandsGesturesDetector._ensure_all_detectors_registered()
