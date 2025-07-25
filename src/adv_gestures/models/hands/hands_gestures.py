from __future__ import annotations

from math import cos, radians
from typing import TYPE_CHECKING, Any, ClassVar

from ...gestures import TWO_HANDS_GESTURES, Gestures
from ...smoothing import GestureWeights

if TYPE_CHECKING:
    from .hand import Hand
    from .hands import Hands


def up_with_tolerance(angle_deg: float) -> float:
    """Calculate the cosine of an angle in radians for a given degree with a tolerance."""
    return -cos(radians(angle_deg))


class GestureDetector:
    gesture: ClassVar[Gestures]
    hands: Hands

    by_gesture: ClassVar[dict[Gestures, type[GestureDetector]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.gesture in GestureDetector.by_gesture:
            raise ValueError(f"Gesture {cls.gesture} already has a detector")
        GestureDetector.by_gesture[cls.gesture] = cls

    def __init__(self, hands: Hands) -> None:
        super().__init__()
        self.hands = hands
        self.left = hands.left
        self.right = hands.right

    def detect(self, detected: GestureWeights) -> None:
        if self.hands.is_gesture_disabled(self.gesture):
            return
        if not self.left or not self.right:
            return
        if not self.matches(self.hands, self.left, self.right, detected):
            return
        detected[self.gesture] = 1.0

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")


class TwoHandsGesturesDetector:
    def __init__(self, hands: Hands) -> None:
        self.hands = hands
        self.detectors = [detector(hands) for detector in GestureDetector.by_gesture.values()]

    def detect(self) -> GestureWeights:
        detected: GestureWeights = {}
        for detector in self.detectors:
            detector.detect(detected)
        return detected

    @staticmethod
    def _ensure_all_detectors_registered() -> None:
        """Ensure that all custom gestures have detectors registered."""
        registered = set(GestureDetector.by_gesture.keys())
        if registered != TWO_HANDS_GESTURES:
            raise ValueError(
                f"Not all hands gestures have detectors registered. Missing: {TWO_HANDS_GESTURES - registered}"
            )


class PrayDetector(GestureDetector):
    gesture = Gestures.PRAY
    main_direction_range: ClassVar[float] = up_with_tolerance(20)

    @staticmethod
    def hand_in_good_shape(hand: Hand) -> bool:
        return hand.is_showing_side and (
            Gestures.STOP in hand.gestures
            or (hand.index.is_nearly_straight_or_straight and hand.middle.is_nearly_straight_or_straight)
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        return (
            left.main_direction is not None
            and right.main_direction is not None
            and left.main_direction[1] < self.main_direction_range
            and right.main_direction[1] < self.main_direction_range
            and self.hands.hands_are_close
            and self.hand_in_good_shape(left)
            and self.hand_in_good_shape(right)
        )


TwoHandsGesturesDetector._ensure_all_detectors_registered()
