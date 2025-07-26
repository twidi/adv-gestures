from __future__ import annotations

from math import acos, radians
from time import time
from typing import TYPE_CHECKING, ClassVar

from ...gestures import CUSTOM_GESTURES, Gestures
from ...smoothing import GestureWeights, SmoothedBase, smoothed_bool
from ..fingers import IndexFinger, MiddleFinger, PinkyFinger, RingFinger, Thumb
from .base_gestures import BaseGestureDetector, DirectionMatcher, up_with_tolerance

if TYPE_CHECKING:
    from .hand import Hand


class HandGesturesDetector(BaseGestureDetector["Hand"], SmoothedBase):
    gestures_set = CUSTOM_GESTURES

    facing_camera: ClassVar[bool | None] = None

    def __init__(self, obj: Hand) -> None:
        BaseGestureDetector.__init__(self, obj)
        SmoothedBase.__init__(self)
        self.hand = obj
        self.thumb = obj.thumb
        self.index = obj.index
        self.middle = obj.middle
        self.ring = obj.ring
        self.pinky = obj.pinky

    def matches_main_direction(self, main_direction_range: DirectionMatcher) -> bool:
        return self.hand_matches_direction(self.hand, main_direction_range)

    def _matches(self, detected: GestureWeights) -> bool:
        self.reset()  # clear cache of smoothed properties
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
    min_angle_deg = 2.5
    min_angle_rad = radians(min_angle_deg)

    def _calc_hand_is_waving(self) -> bool:
        """Check if the hand is performing a waving motion (oscillating left-right movement)."""
        if not self.hand._direction_history:
            return False

        current_time = time()
        duration = 1

        # Find directions within the duration window
        cutoff_time = current_time - duration
        directions_in_window = [(t, x, y) for t, x, y, _ in self.hand._direction_history if t >= cutoff_time]

        if len(directions_in_window) < 3:  # Need at least 3 points to detect oscillation
            return False

        # Check if we have data covering sufficient duration (at least 80% of window)
        time_coverage = current_time - directions_in_window[0][0]
        if time_coverage < duration * 0.8:
            return False

        # Detect direction changes based on X component sign changes
        # We need to detect at least 1.5 oscillations (e.g., left→right→left)
        x_tolerance = 0.05  # Tolerance zone around x=0 to avoid noise

        # Track direction changes
        direction_changes = []
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
                        prev_t = directions_in_window[i - 1][0]
                        direction_changes.append((prev_t, t))

            last_significant_x = x

        # Check if we have at least 2 direction changes (1.5 oscillations)
        if len(direction_changes) < 2:
            return False

        # Check that the last direction change is recent (within last 0.5 seconds)
        # This ensures we stop detecting wave when hand stops moving
        if direction_changes:
            last_change_time = direction_changes[-1][1]
            time_since_last_change = current_time - last_change_time
            if time_since_last_change > 0.5:
                return False

        # Also check that we have recent movement in the last 0.3 seconds
        # to ensure the hand is still actively moving
        recent_directions = [(t, x, y) for t, x, y in directions_in_window if current_time - t <= 0.3]
        if len(recent_directions) < 2:
            return False

        # Verify angle changes are significant enough
        # Check angles between consecutive significant directions

        significant_directions = [(x, y) for _, x, y in directions_in_window if abs(x) >= x_tolerance]

        if len(significant_directions) < 3:
            return False

        # Check angle between first and middle, middle and last directions
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
            if angle1 >= self.min_angle_rad or angle2 >= self.min_angle_rad:
                return True

        return False

    hand_is_waving = smoothed_bool(_calc_hand_is_waving)

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
        return (hand.default_gesture == Gestures.OPEN_PALM or Gestures.STOP in detected) and self.hand_is_waving


HandGesturesDetector._ensure_all_detectors_registered()
