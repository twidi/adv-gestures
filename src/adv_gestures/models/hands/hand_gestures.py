from __future__ import annotations

from math import sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar, cast

from ...gestures import CUSTOM_GESTURES, Gestures
from ...smoothing import GestureWeights, SmoothedBase, smoothed_bool
from ..fingers import IndexFinger, MiddleFinger, PinkyFinger, RingFinger, Thumb
from .base_gestures import (
    BaseGestureDetector,
    DetectionState,
    Range,
    StatefulMode,
)
from .utils import Direction

if TYPE_CHECKING:
    from .hand import Hand


class HandGesturesDetector(BaseGestureDetector["Hand"], SmoothedBase):
    gestures_set = CUSTOM_GESTURES

    facing_camera: ClassVar[bool | None] = None
    showing_side: ClassVar[bool | None] = None
    main_direction_range: ClassVar[Range | None] = None

    def __init__(self, obj: Hand) -> None:
        BaseGestureDetector.__init__(self, obj)
        SmoothedBase.__init__(self)
        self.hand = obj
        self.thumb = obj.thumb
        self.index = obj.index
        self.middle = obj.middle
        self.ring = obj.ring
        self.pinky = obj.pinky

    def pre_matches(self, detected: GestureWeights) -> bool:
        self.reset()  # clear cache of smoothed properties
        if not super().pre_matches(detected):
            return False
        if self.facing_camera is not None and self.hand.is_facing_camera != self.facing_camera:
            return False
        if self.showing_side is not None and self.hand.is_showing_side != self.showing_side:
            return False
        if self.main_direction_range is not None and not self.hand_matches_direction(
            self.hand, self.main_direction_range
        ):
            return False
        return True

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
    main_direction_range = 60, 120

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
    main_direction_range = 70, 110
    facing_camera = True
    showing_side = False

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
    showing_side = False

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
    showing_side = False
    main_direction_range = 0, 180

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
    main_direction_range = 70, 110
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
    thumb_direction_range: Range = 0, 180

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
    showing_side = False

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
    stateful_mode = StatefulMode.POST_DETECTION
    post_detection_duration = 0.5  # Report tap for 0.5s after movement

    def __init__(self, obj: Hand) -> None:
        super().__init__(obj)
        self.config = obj.config.hands.index.tip_stability
        self.min_gesture_duration: float = self.config.min_duration
        self.max_gesture_duration: float = self.config.max_duration
        self.movement_threshold: float = self.config.movement_threshold
        self._last_removed_tip_position: tuple[float, float] | None = None

    def _calc_is_tip_stable(self) -> bool:
        """Check if the index tip has been stable for the configured duration."""
        if not self.index._tip_position_history:
            return False

        current_time = time()
        duration = self.min_gesture_duration

        # Find positions within the duration window
        cutoff_time = current_time - duration
        positions_in_window = [(t, x, y) for t, x, y in self.index._tip_position_history if t >= cutoff_time]

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

        return normalized_movement < self.movement_threshold

    is_tip_stable = smoothed_bool(_calc_is_tip_stable)

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
        self.reset()  # clear cache of is_tip_stable
        return (
            index.is_nearly_straight_or_straight
            and middle.is_not_straight_at_all
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
            and self.is_tip_stable
        )

    def _stateful_start_tracking(self, now: float) -> DetectionState:
        detection = super()._stateful_start_tracking(now)
        detection.data = {"tap_position": self.index.end_point}
        return detection

    def _stateful_update_detections(self, detected: GestureWeights) -> None:
        """Override to store tap position in detection state data."""

        # Track tip position when removing a tracking due to max duration
        now = time()
        for detection in self.tracking_detections:
            if self._stateful_is_expired(detection, now):
                self._last_removed_tip_position = detection.data["tap_position"]  # type: ignore[index]  # data defined in _stateful_start_tracking
                break

        # Effectively update detections
        super()._stateful_update_detections(detected)

        # Update tap position for all tracking states
        if self.index.end_point is not None:
            tap_position = self.index.end_point
            for detection in self.tracking_detections:
                # Only update TRACKING, because POST_DETECTING states keep their last position
                detection.data["tap_position"] = tap_position  # type: ignore[index]  # data defined in _stateful_start_tracking
                break

    def _stateful_can_start_tracking(self) -> bool:
        """Don't start new tracking if we removed one at same position due to max duration."""
        if self._last_removed_tip_position is None:
            return True

        # Check if tip has moved from the last removed position
        if not self.index.landmarks:
            # No current position, allow tracking
            # self._last_removed_tip_position = None
            return True

        tip = self.index.landmarks[-1]
        current_x, current_y = tip.x, tip.y
        last_x, last_y = self._last_removed_tip_position

        # Calculate movement distance (in normalized coordinates)
        dx = current_x - last_x
        dy = current_y - last_y
        movement = sqrt(dx**2 + dy**2)
        diagonal = sqrt(self.hand.stream_info.width**2 + self.hand.stream_info.height**2)  # type: ignore[union-attr]
        normalized_movement = movement / diagonal

        # If finger hasn't moved beyond threshold, prevent new tracking
        if normalized_movement < self.movement_threshold:
            return False

        # Finger moved, allow new tracking and reset
        self._last_removed_tip_position = None
        return True

    @property
    def tip_position(self) -> tuple[int, int] | None:
        """Get the current tap position for visualization."""
        # First, look for POST_DETECTING states (oldest first)

        if post_detecting_states := self.post_detecting_detections:
            # Get the oldest POST_DETECTING state
            oldest = min(post_detecting_states, key=lambda d: d.tracking_start)
            return cast(tuple[int, int], oldest.data["tap_position"])  # type: ignore[index]  # data defined in _stateful_start_tracking

        # Otherwise, look for TRACKING states
        if tracking_states := self.tracking_detections:
            # Get the first tracking state
            return cast(tuple[int, int], tracking_states[0].data["tap_position"])  # type: ignore[index]  # data defined in _stateful_start_tracking

        return None

    @property
    def tap_state(self) -> str | None:
        """Get the current tap state: 'detected' for post-detection, 'detecting' for tracking, None otherwise."""
        # First, look for POST_DETECTING states (priority)
        if self.post_detecting_detections:
            return "detected"

        # Otherwise, look for TRACKING states
        if self.tracking_detections:
            return "detecting"

        return None


class WaveDetector(HandGesturesDetector):
    gesture = Gestures.WAVE

    def _calc_hand_is_waving(self) -> bool:
        """Check if the hand is performing a waving motion (oscillating left-right movement)."""
        has_changes, _ = self.hand.detect_direction_changes(
            duration_window=1.0,
            min_direction_changes=2,
            min_movement_angle=2.5,
        )
        return has_changes

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


class SnapDetector(HandGesturesDetector):
    gesture = Gestures.SNAP
    stateful_mode = StatefulMode.POST_DETECTION
    post_detection_duration = 0.3  # Report snap for 0.3s after detection
    max_transition_time = 0.5  # 500ms max between before and after states

    def __init__(self, obj: Hand) -> None:
        super().__init__(obj)
        self._last_before_state_time: float | None = None

    def _match_before_snap(self) -> bool:
        """Check if hand is in pre-snap position."""
        return (
            self.ring.is_not_straight_at_all
            and self.pinky.is_not_straight_at_all
            and not self.index.is_fully_bent
            and not self.index.is_nearly_straight_or_straight
            and not self.middle.is_fully_bent
            and not self.middle.is_nearly_straight_or_straight
            and self.middle.tip_on_thumb
        )

    def _match_after_snap(self) -> bool:
        """Check if hand is in post-snap position."""
        return (
            self.ring.is_not_straight_at_all
            and self.pinky.is_not_straight_at_all
            and self.middle.is_not_straight_at_all
            and not self.index.is_fully_bent
            and not self.index.is_nearly_straight_or_straight
            and self.index.tip_on_thumb
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
        current_time = time()

        # Check if we're in the before state
        if self._match_before_snap():
            self._last_before_state_time = current_time
            return False  # Not a snap yet

        # Check if we're in the after state
        if self._last_before_state_time is not None and self._match_after_snap():
            # Check if we had a recent before state
            if (
                self._last_before_state_time is not None
                and current_time - self._last_before_state_time <= self.max_transition_time
            ):
                # Reset to prevent repeated detections
                self._last_before_state_time = None
                return True

            self._last_before_state_time = None

        return False


class _SwipeHandDetector(HandGesturesDetector):
    """Base class for swipe hand detectors."""

    stateful_mode = StatefulMode.POST_DETECTION
    post_detection_duration = 0.5  # Report swipe for 0.5s after detection
    direction: Direction | None = None  # Placeholder for direction, to be set in subclasses

    def _calc_is_swiping(self) -> bool:
        """Check if hand is currently performing a swipe motion in the expected direction."""

        if self.direction is None:
            raise ValueError("Swipe direction must be set in subclasses.")

        # For a swipe, we need exactly 1 direction change
        # Use a shorter duration window since swipes are quick
        has_changes, changes = self.hand.detect_direction_changes(
            duration_window=0.8,  # Shorter window for swipes
            min_direction_changes=1,  # Only need 1 change for a swipe
            min_movement_angle=5.0,  # Larger angle for more decisive swipes
            x_tolerance=0.05,
            max_time_since_last_change=0.5,
            require_recent_movement=True,
            recent_movement_window=0.3,
        )

        if not has_changes or not changes:
            return False

        # Check if the most recent change matches our expected direction
        # changes contains (Direction, time_ago) pairs
        most_recent_change = changes[-1]  # Get the most recent change
        direction, time_ago = most_recent_change

        return direction == self.direction and time_ago < 0.5

    is_swiping = smoothed_bool(_calc_is_swiping)

    def hand_in_good_shape(self) -> bool:
        """Check if the 4 fingers (except thumb) are straight or nearly straight."""
        return (
            self.index.is_nearly_straight_or_straight
            and self.middle.is_nearly_straight_or_straight
            and self.ring.is_nearly_straight_or_straight
            and self.pinky.is_nearly_straight_or_straight
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
        # Check basic conditions
        if hand.default_gesture == Gestures.OPEN_PALM or Gestures.STOP in detected:
            return False

        # Check fingers are straight AND swipe motion is detected
        return self.hand_in_good_shape() and self.is_swiping


class SwipeHandToLeftDetector(_SwipeHandDetector):
    gesture = Gestures.SWIPE_HAND_TO_LEFT
    direction = Direction.LEFT


class SwipeHandToRightDetector(_SwipeHandDetector):
    gesture = Gestures.SWIPE_HAND_TO_RIGHT
    direction = Direction.RIGHT


HandGesturesDetector._ensure_all_detectors_registered()
