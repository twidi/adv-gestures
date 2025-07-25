from __future__ import annotations

from math import cos, radians
from typing import TYPE_CHECKING, Any, ClassVar

from ...gestures import CUSTOM_GESTURES, Gestures
from ...smoothing import GestureWeights
from ..fingers import IndexFinger, MiddleFinger, PinkyFinger, RingFinger, Thumb

if TYPE_CHECKING:
    from .hand import Hand


def up_with_tolerance(angle_deg: float) -> float:
    """Calculate the cosine of an angle in radians for a given degree with a tolerance."""
    return -cos(radians(angle_deg))


class GestureDetector:
    gesture: ClassVar[Gestures]
    hand: Hand

    by_gesture: ClassVar[dict[Gestures, type[GestureDetector]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.gesture in GestureDetector.by_gesture:
            raise ValueError(f"Gesture {cls.gesture} already has a detector")
        GestureDetector.by_gesture[cls.gesture] = cls

    def __init__(self, hand: Hand) -> None:
        super().__init__()
        self.hand = hand
        self.thumb = hand.thumb
        self.index = hand.index
        self.middle = hand.middle
        self.ring = hand.ring
        self.pinky = hand.pinky

    def detect(self, detected: GestureWeights) -> None:
        if self.hand.is_gesture_disabled(self.gesture):
            return
        if not self.matches(
            self.hand, self.hand.thumb, self.hand.index, self.hand.middle, self.hand.ring, self.hand.pinky, detected
        ):
            return
        detected[self.gesture] = 1.0

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")


class HandGesturesDetector:
    def __init__(self, hand: Hand) -> None:
        self.hand = hand
        self.detectors = [detector(hand) for detector in GestureDetector.by_gesture.values()]

    def detect(self) -> GestureWeights:
        detected: GestureWeights = {}
        for detector in self.detectors:
            detector.detect(detected)
        return detected

    @staticmethod
    def _ensure_all_detectors_registered() -> None:
        """Ensure that all custom gestures have detectors registered."""
        registered = set(GestureDetector.by_gesture.keys())
        if registered != CUSTOM_GESTURES:
            raise ValueError(
                f"Not all custom gestures have detectors registered. Missing: {CUSTOM_GESTURES - registered}"
            )


class MiddleFingerDetector(GestureDetector):
    gesture = Gestures.MIDDLE_FINGER

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            middle.is_straight
            and index.is_not_straight_at_all
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
        )


class VictoryDetector(GestureDetector):
    # (should be detected as default gesture, but it's not always the case)
    gesture = Gestures.VICTORY
    main_direction_range: ClassVar[float] = up_with_tolerance(30)

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            hand.main_direction is not None
            and hand.main_direction[1] < self.main_direction_range
            and index.is_straight
            and middle.is_straight
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
            and thumb.is_fully_bent
            and not index.is_touching(middle)
        )


class SpockDetector(GestureDetector):
    gesture = Gestures.SPOCK
    main_direction_range: ClassVar[float] = up_with_tolerance(20)

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            hand.is_facing_camera
            and hand.main_direction is not None
            and hand.main_direction[1] < self.main_direction_range
            and index.is_straight
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and index.is_touching(middle)
            and ring.is_touching(pinky)
            and not middle.is_touching(ring)
        )


class RockDetector(GestureDetector):
    gesture = Gestures.ROCK

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            index.is_straight
            and pinky.is_straight
            and not thumb.is_straight
            and middle.is_not_straight_at_all
            and ring.is_not_straight_at_all
        )


class OkDetector(GestureDetector):
    gesture = Gestures.OK

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            hand.is_facing_camera
            and index.tip_on_thumb
            and not middle.is_fully_bent
            and not ring.is_fully_bent
            and not pinky.is_fully_bent
        )


class StopDetector(GestureDetector):
    gesture = Gestures.STOP
    main_direction_range: ClassVar[float] = up_with_tolerance(20)

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            hand.is_facing_camera
            and hand.main_direction is not None
            and hand.main_direction[1] < self.main_direction_range
            and index.is_straight
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and hand.all_adjacent_fingers_except_thumb_touching
        )


class PinchDetector(GestureDetector):
    gesture = Gestures.PINCH

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            thumb.is_nearly_straight_or_straight
            and not middle.is_straight
            and not ring.is_straight
            and not pinky.is_straight
        )


class PinchTouchDetector(PinchDetector):
    gesture = Gestures.PINCH_TOUCH

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return super().matches(hand, thumb, index, middle, ring, pinky, detected) and index.tip_on_thumb


class FingerGunDetector(GestureDetector):
    gesture = Gestures.FINGER_GUN
    main_direction_range: ClassVar[float] = up_with_tolerance(90)

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            thumb.straight_direction is not None
            and thumb.straight_direction[1] < self.main_direction_range
            and thumb.is_nearly_straight_or_straight
            and index.is_nearly_straight_or_straight
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
        )


class GunDetector(FingerGunDetector):
    gesture = Gestures.GUN

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            super().matches(hand, thumb, index, middle, ring, pinky, detected)
            and middle.is_nearly_straight_or_straight
            and index.is_touching(middle)
        )


class AirTapDetector(GestureDetector):
    gesture = Gestures.AIR_TAP

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (
            index.is_nearly_straight_or_straight
            and middle.is_not_straight_at_all
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
            and index.is_tip_stable
        )


class WaveDetector(GestureDetector):
    gesture = Gestures.WAVE

    def matches(
        self,
        hand: Hand,
        thumb: Thumb,
        index: IndexFinger,
        middle: MiddleFinger,
        ring: RingFinger,
        pinky: PinkyFinger,
        detected: GestureWeights,
    ) -> bool:
        return (hand.default_gesture == Gestures.OPEN_PALM or Gestures.STOP in detected) and hand.is_waving


HandGesturesDetector._ensure_all_detectors_registered()
