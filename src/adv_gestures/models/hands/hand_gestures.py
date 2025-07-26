from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...gestures import CUSTOM_GESTURES, Gestures
from ...smoothing import GestureWeights
from ..fingers import IndexFinger, MiddleFinger, PinkyFinger, RingFinger, Thumb
from .base_gestures import BaseGestureDetector, DirectionMatcher, up_with_tolerance

if TYPE_CHECKING:
    from .hand import Hand


class HandGesturesDetector(BaseGestureDetector["Hand"]):
    gestures_set = CUSTOM_GESTURES

    facing_camera: ClassVar[bool | None] = None

    def __init__(self, obj: Hand) -> None:
        super().__init__(obj)
        self.hand = obj
        self.thumb = obj.thumb
        self.index = obj.index
        self.middle = obj.middle
        self.ring = obj.ring
        self.pinky = obj.pinky

    def matches_main_direction(self, main_direction_range: DirectionMatcher) -> bool:
        return self.hand_matches_direction(self.hand, main_direction_range)

    def _matches(self, detected: GestureWeights) -> bool:
        if self.facing_camera is not None and self.hand.is_facing_camera != self.facing_camera:
            return False
        return self.matches(
            self.hand, self.hand.thumb, self.hand.index, self.hand.middle, self.hand.ring, self.hand.pinky, detected
        )

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


class MiddleFingerDetector(HandGesturesDetector):
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


class VictoryDetector(HandGesturesDetector):
    # (should be detected as default gesture, but it's not always the case)
    gesture = Gestures.VICTORY
    main_direction_range = up_with_tolerance(30)

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
            and middle.is_straight
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
            and thumb.is_fully_bent
            and not index.is_touching(middle)
        )


class SpockDetector(HandGesturesDetector):
    gesture = Gestures.SPOCK
    main_direction_range = up_with_tolerance(20)
    facing_camera = True

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
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and index.is_touching(middle)
            and ring.is_touching(pinky)
            and not middle.is_touching(ring)
        )


class RockDetector(HandGesturesDetector):
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


class OkDetector(HandGesturesDetector):
    gesture = Gestures.OK
    facing_camera = True

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
        return index.tip_on_thumb and not middle.is_fully_bent and not ring.is_fully_bent and not pinky.is_fully_bent


class StopDetector(HandGesturesDetector):
    gesture = Gestures.STOP
    main_direction_range = up_with_tolerance(20)
    facing_camera = True

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
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and hand.all_adjacent_fingers_except_thumb_touching
        )


class PinchDetector(HandGesturesDetector):
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


class FingerGunDetector(HandGesturesDetector):
    gesture = Gestures.FINGER_GUN
    thumb_direction_range: DirectionMatcher = up_with_tolerance(90)

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
            self.finger_matches_straight_direction(thumb, self.thumb_direction_range)
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


class AirTapDetector(HandGesturesDetector):
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


class WaveDetector(HandGesturesDetector):
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
