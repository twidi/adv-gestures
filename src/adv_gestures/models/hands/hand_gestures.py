from __future__ import annotations

from math import sqrt
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid4

from ...gestures import CUSTOM_GESTURES, Gestures
from ...smoothing import GestureWeights, SmoothedBase, smoothed_bool
from ..fingers import IndexFinger, MiddleFinger, PinkyFinger, RingFinger, Thumb
from ..utils import Direction, Handedness, SwipeMode
from .base_gestures import (
    BaseGestureDetector,
    DetectionState,
    Range,
)

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
            middle.is_nearly_straight_or_straight
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
        return (
            index.tip_on_thumb
            and not middle.is_fully_bent
            and not ring.is_fully_bent
            and not pinky.is_fully_bent
            and (thumb_convergence_score := index.thumb_convergence_score) is not None
            and thumb_convergence_score >= 0.7
        )


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

    def get_angle_between_thumb_and_index_tips(self) -> float | None:
        """Calculate the angle between the thumb and index finger tips using their tip_direction_angle.

        Returns:
            - Positive angle (0 to 180) when fingers are diverging (pointing away from each other)
            - Negative angle (-180 to 0) when fingers are converging (pointing towards each other)
            - None if angles are not available.
        """
        thumb = self.hand.thumb
        index = self.hand.index

        thumb_angle = thumb.tip_direction_angle
        index_angle = index.tip_direction_angle

        if thumb_angle is None or index_angle is None:
            return None

        # Calculate the raw angle difference
        angle_diff = index_angle - thumb_angle

        # Normalize to -180 to 180 range
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        if self.hand.handedness == Handedness.RIGHT:
            angle_diff = -angle_diff

        return angle_diff

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
            and not index.is_fully_bent
            and not middle.is_straight
            and not ring.is_straight
            and not pinky.is_straight
            and (angle := self.get_angle_between_thumb_and_index_tips()) is not None
            and -130 <= angle <= 75
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
    post_detection_mode = True
    post_detection_duration = 0.5  # Report tap for 0.5s after movement

    def __init__(self, obj: Hand) -> None:
        super().__init__(obj)
        self.config = obj.config.hands.index.tip_stability
        self.min_gesture_duration: float = self.config.min_duration
        self.max_gesture_duration: float = self.config.max_duration
        self.movement_threshold: float = self.config.movement_threshold
        self.tap_movement_threshold: float = self.config.tap_movement_threshold
        self._last_removed_tip: tuple[float, tuple[float, float]] | None = None

    def get_movement_amplitude(self, since: float) -> float | None:
        median_position = self.index.get_tip_median_position(since)
        if median_position is None:
            return False

        # Get current tip position
        if not self.index.landmarks:
            return False

        tip = self.index.landmarks[-1]
        current_x, current_y = tip.x, tip.y
        median_x, median_y = median_position

        # Calculate distance from median to current position
        dx = current_x - median_x
        dy = current_y - median_y
        distance = sqrt(dx**2 + dy**2)

        # Normalize the distance
        diagonal = sqrt(self.hand.stream_info.width**2 + self.hand.stream_info.height**2)
        return distance / diagonal

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

        is_stable = True
        tracking_detection: DetectionState | None = None
        if tracking_detections := self.tracking_detections:
            tracking_detection = tracking_detections[0]
            is_stable = tracking_detection.data["is_stable"]  # type: ignore[index]  # data defined in _stateful_start_tracking

        cutoff_time = tracking_detection.tracking_start if tracking_detection else time() - self.min_gesture_duration
        if (amplitude := self.get_movement_amplitude(cutoff_time)) is None:
            return False

        if is_stable:
            is_stable = (
                index.is_nearly_straight_or_straight
                and middle.is_not_straight_at_all
                and ring.is_not_straight_at_all
                and pinky.is_not_straight_at_all
                and amplitude < self.movement_threshold
            )
            if not is_stable:
                if tracking_detection is None:
                    return False
                tracking_detection.data["is_stable"] = False  # type: ignore[index]  # data defined in _stateful_start_tracking

        if is_stable:
            return True

        # Not stable, we still return True until we hit a major movement
        return amplitude < self.tap_movement_threshold

    def _stateful_start_tracking(self, now: float) -> DetectionState:
        detection = super()._stateful_start_tracking(now)
        detection.data = {
            "tap_position": self.index.end_point,
            "tap_id": str(uuid4()),
            "is_stable": True,
        }
        return detection

    def _stateful_update_detections(self, detected: GestureWeights) -> None:
        """Override to store tap position in detection state data."""

        # Track tip position when removing a tracking due to max duration
        now = time()
        for detection in self.tracking_detections:
            if self._stateful_is_expired(detection, now):
                self._last_removed_tip = (now, detection.data["tap_position"])  # type: ignore[index]  # data defined in _stateful_start_tracking
                break

        # Effectively update detections
        super()._stateful_update_detections(detected)

        # Update tap position for all tracking detections in the "stable" state
        for detection in self.tracking_detections:
            if detection.data["is_stable"]:  # type: ignore[index]  # data defined in _stateful_start_tracking
                if (median_position := self.index.get_tip_median_position(detection.tracking_start)) is None:
                    continue
                detection.data["tap_position"] = median_position  # type: ignore[index]  # data defined in _stateful_start_tracking

    def _stateful_can_start_tracking(self) -> bool:
        """Don't start new tracking if we removed one at same position due to max duration."""

        if not self.index.landmarks or not self.index.end_point:
            return False

        if self._last_removed_tip is None:
            return True

        # Check if last removed tip position is old enough to allow new tracking at the same position
        now = time()
        if self._last_removed_tip[0] and now - self._last_removed_tip[0] > 1:
            self._last_removed_tip = None
            return True

        tip = self.index.landmarks[-1]
        current_x, current_y = tip.x, tip.y
        last_x, last_y = self._last_removed_tip[1]

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
        self._last_removed_tip = None
        return True

    def get_data(self) -> dict[str, Any] | None:
        result = super().get_data()
        if not result:
            return result
        result.pop("is_stable", None)  # Remove is_stable from data, it is for internal use only
        return result | {
            "max_duration": self.post_detection_duration,
            "elapsed_since_tap": time() - self.post_detecting_detections[0].post_detection_start,  # type: ignore[operator] # post_detection_start is set when setting the state to POST_DETECTING
        }


class PreAirTapDetector(HandGesturesDetector):
    gesture = Gestures.PRE_AIR_TAP

    _air_tap_detector: AirTapDetector | None = None

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
        if not self.air_tap_detector.tracking_detections:
            return False

        # Check if at least one tracking detection has exceeded min_gesture_duration
        now = time()
        for detection in self.air_tap_detector.tracking_detections:
            duration = now - detection.tracking_start
            if duration >= self.air_tap_detector.min_gesture_duration:
                return True

        return False

    @property
    def air_tap_detector(self) -> AirTapDetector:
        if self._air_tap_detector is None:
            self._air_tap_detector = cast(AirTapDetector, self.hand.gestures_detector.detectors[Gestures.AIR_TAP])
        return self._air_tap_detector

    def get_data(self) -> dict[str, Any] | None:
        data = super().get_data()
        if not self.air_tap_detector.tracking_detections:
            return data

        # Get the first tracking detection (should be the oldest)
        detection = self.air_tap_detector.tracking_detections[0]

        # Calculate current duration
        now = time()
        current_duration = now - detection.tracking_start

        # Ensure we have a dict to return
        if data is None:
            data = {}

        # Add duration info
        data["duration"] = max(0.0, current_duration - self.air_tap_detector.min_gesture_duration)
        data["max_duration"] = self.air_tap_detector.max_gesture_duration - self.air_tap_detector.min_gesture_duration

        # Add tap position if available
        if detection.data and "tap_position" in detection.data:
            data["tap_position"] = detection.data["tap_position"]

        return data


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


class NoDetector(HandGesturesDetector):
    gesture = Gestures.NO

    def _calc_index_is_waving(self) -> bool:
        """Check if the index finger is performing a waving motion (oscillating left-right movement)."""
        has_changes, _ = self.index.detect_direction_changes(
            duration_window=1.0,
            min_direction_changes=2,
            min_movement_angle=2.5,
        )
        return has_changes

    index_is_waving = smoothed_bool(_calc_index_is_waving)

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
        return hand.default_gesture == Gestures.POINTING_UP and self.index_is_waving


class SnapDetector(HandGesturesDetector):
    gesture = Gestures.SNAP
    post_detection_mode = True
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
            and not self.middle.tip_on_thumb
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
            #             print(f"""
            # PRESNAP at {current_time=}
            #     {self.ring.is_not_straight_at_all=}
            #     {self.pinky.is_not_straight_at_all=}
            #     {not self.index.is_fully_bent=}
            #     {not self.index.is_nearly_straight_or_straight=}
            #     {not self.middle.is_fully_bent=}
            #     {not self.middle.is_nearly_straight_or_straight=}
            #     {self.middle.tip_on_thumb=}
            # """)
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


class DoubleSnapDetector(HandGesturesDetector):
    gesture = Gestures.DOUBLE_SNAP
    stateful = True
    min_gesture_duration = 0.0  # No minimum duration for first snap
    max_gesture_duration = 1.0  # Maximum 1 second between snaps
    post_detection_mode = True
    post_detection_duration = 0.3  # Report double snap for 0.3s after detection

    def __init__(self, obj: Hand) -> None:
        super().__init__(obj)
        self._snap_detector: SnapDetector | None = None
        self._first_snap_time: float | None = None
        self._waiting_for_second_snap = False
        self._had_gap_after_first_snap = False
        self._last_double_snap_time: float | None = None

    @property
    def snap_detector(self) -> SnapDetector:
        if self._snap_detector is None:
            self._snap_detector = cast(SnapDetector, self.hand.gestures_detector.detectors[Gestures.SNAP])
        return self._snap_detector

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
        # Check if a snap is detected
        snap_detected = Gestures.SNAP in detected
        current_time = time()

        # If we just detected a double snap, wait for the post_detection_duration of the snap
        # before allowing a new detection cycle
        if self._last_double_snap_time:
            if current_time - self._last_double_snap_time < self.snap_detector.post_detection_duration:
                return False
            else:
                # Enough time has passed, reset for new detection
                self._last_double_snap_time = None

        # If we're waiting for second snap
        if self._waiting_for_second_snap:
            # First check for gap after first snap
            if not snap_detected and not self._had_gap_after_first_snap:
                self._had_gap_after_first_snap = True
                return False

            # If we had a gap and now detect a snap, it's the second snap
            if snap_detected and self._had_gap_after_first_snap:
                # Check if we're still within the time window
                if self._first_snap_time and current_time - self._first_snap_time <= self.max_gesture_duration:
                    # Second snap detected within time window!
                    self._waiting_for_second_snap = False
                    self._first_snap_time = None
                    self._had_gap_after_first_snap = False
                    self._last_double_snap_time = current_time
                    return True
                else:
                    # Too late, reset completely
                    self._waiting_for_second_snap = False
                    self._first_snap_time = None
                    self._had_gap_after_first_snap = False
                    # Don't start a new detection cycle in the same frame
                    return False

            # Check if we've exceeded the time window
            if self._first_snap_time and current_time - self._first_snap_time > self.max_gesture_duration:
                # Reset state
                self._waiting_for_second_snap = False
                self._first_snap_time = None
                self._had_gap_after_first_snap = False

        # If first snap detected and we're not waiting
        elif snap_detected and not self._waiting_for_second_snap:
            # Only start tracking if this is truly a new snap (not continuing from a reset)
            # We need to ensure we don't have a snap that was just reset from being too late
            self._first_snap_time = current_time
            self._waiting_for_second_snap = True
            self._had_gap_after_first_snap = False

        return False


class SwipeDetector(HandGesturesDetector):
    """Unified swipe detector that detects swipes in any direction by hand or index finger."""

    gesture = Gestures.SWIPE
    post_detection_mode = True
    post_detection_duration = 0.5  # Report swipe for 0.5s after detection

    def __init__(self, hand: Hand):
        super().__init__(hand)
        self._detected_direction: Direction | None = None

    def _calc_is_swiping(self) -> bool:
        """Check if hand is currently performing a swipe motion and capture direction."""
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
            self._detected_direction = None
            return False

        # Get the most recent change
        # changes contains (Direction, time_ago) pairs
        most_recent_change = changes[-1]  # Get the most recent change
        direction, time_ago = most_recent_change

        if time_ago < 0.5:
            self._detected_direction = direction
            return True

        self._detected_direction = None
        return False

    is_swiping = smoothed_bool(_calc_is_swiping)

    def hand_in_good_shape(self) -> bool:
        """Check if the 4 fingers (except thumb) are straight or nearly straight."""
        return (
            self.index.is_nearly_straight_or_straight
            and self.middle.is_nearly_straight_or_straight
            and self.ring.is_nearly_straight_or_straight
            and self.pinky.is_nearly_straight_or_straight
        )

    def index_only_shape(self) -> bool:
        """Check if only the index finger is straight (for index swipe)."""
        return (
            self.index.is_nearly_straight_or_straight
            and self.middle.is_not_straight_at_all
            and self.ring.is_not_straight_at_all
            and self.pinky.is_not_straight_at_all
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

        # Check if swipe motion is detected
        if not self.is_swiping:
            return False

        # Check fingers for either hand swipe or index swipe
        return self.hand_in_good_shape() or self.index_only_shape()

    def _stateful_start_tracking(self, now: float) -> DetectionState:
        """Override to store the detected swipe direction and type in detection data."""
        detection = super()._stateful_start_tracking(now)
        # Determine swipe mode
        if self.hand_in_good_shape():
            mode = SwipeMode.HAND
        elif self.index_only_shape():
            mode = SwipeMode.INDEX
        else:
            raise ValueError("Swipe detection logic should ensure one of the mode HAND/INDEX is matched.")
        detection.data = {"direction": self._detected_direction, "mode": mode}
        return detection


HandGesturesDetector._ensure_all_detectors_registered()
