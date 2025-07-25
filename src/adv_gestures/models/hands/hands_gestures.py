from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, radians
from time import time
from typing import TYPE_CHECKING, Any, ClassVar

from ...gestures import TWO_HANDS_GESTURES, Gestures
from ...smoothing import GestureWeights

if TYPE_CHECKING:
    from .hand import Hand
    from .hands import Hands


def up_with_tolerance(angle_deg: float) -> float:
    """Calculate the cosine of an angle in radians for a given degree with a tolerance."""
    return -cos(radians(angle_deg))


class State(Enum):
    TRACKING = "tracking"
    POST_DETECTING = "post_detecting"


@dataclass
class DetectionState:
    tracking_start: float
    state: State
    post_detection_start: float | None = None


class GestureDetector:
    gesture: ClassVar[Gestures]
    hands: Hands

    by_gesture: ClassVar[dict[Gestures, type[GestureDetector]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "gesture"):
            return  # maybe a intermediate subclass
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


class StatefulGestureDetector(GestureDetector):
    min_gesture_duration: ClassVar[float | None] = None
    max_gesture_duration: ClassVar[float | None] = None
    post_detection_duration: ClassVar[float] = 0.0
    min_interval_between_detections: ClassVar[float | None] = None

    def __init__(self, hands: Hands) -> None:
        super().__init__(hands)
        self.tracked_states: list[DetectionState] = []
        self._last_valid_detection_time: float | None = None

    def detect(self, detected: GestureWeights) -> None:
        if self.hands.is_gesture_disabled(self.gesture):
            return
        if not self.left or not self.right:
            return

        self._cleanup_detections()
        self._update_detections(detected)

        if self._has_valid_detection():
            detected[self.gesture] = 1.0

    def _cleanup_detections(self) -> None:
        now = time()
        self.tracked_states = [d for d in self.tracked_states if not self._is_expired(d, now)]

    def _is_expired(self, detection: DetectionState, now: float) -> bool:
        if detection.state == State.TRACKING:
            if self.max_gesture_duration is not None:
                return now - detection.tracking_start > self.max_gesture_duration
            return False
        else:  # POST_DETECTING
            if detection.post_detection_start is None:
                return True
            return now - detection.post_detection_start > self.post_detection_duration

    def _update_detections(self, detected: GestureWeights) -> None:
        now = time()
        gesture_found = self.matches(self.hands, self.left, self.right, detected)

        # Update existing tracking detections
        for detection in self.tracked_states[:]:  # Copy list to allow modification during iteration
            if detection.state == State.TRACKING and not gesture_found:
                # Check if duration is valid
                duration = now - detection.tracking_start

                if self.min_gesture_duration is not None and duration < self.min_gesture_duration:
                    # Too short, remove this detection
                    self.tracked_states.remove(detection)
                else:
                    # Valid - move to post-detecting
                    detection.state = State.POST_DETECTING
                    detection.post_detection_start = now
                    self._last_valid_detection_time = now

        # Check if we can start a new detection
        if gesture_found and self._can_start_new_detection():
            # Check if any tracking already exists
            any_tracking_exists = any(d.state == State.TRACKING for d in self.tracked_states)

            if not any_tracking_exists:
                # Start new tracking
                self.tracked_states.append(DetectionState(tracking_start=now, state=State.TRACKING))

    def _has_valid_detection(self) -> bool:
        return any(d.state == State.POST_DETECTING for d in self.tracked_states)

    def _can_start_new_detection(self) -> bool:
        if self.min_interval_between_detections is None:
            return True
        if self._last_valid_detection_time is None:
            return True
        return time() - self._last_valid_detection_time >= self.min_interval_between_detections


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


class ClapDetector(StatefulGestureDetector):
    gesture = Gestures.CLAP
    min_gesture_duration = 0.05  # Min duration for valid clap
    max_gesture_duration = 0.5  # Max duration for hands to be joined
    post_detection_duration = 0.2  # Show clap duration  after separation

    @staticmethod
    def hand_in_good_shape(hand: Hand) -> bool:
        return (
            Gestures.STOP in hand.gestures
            or Gestures.OPEN_PALM in hand.gestures
            or (hand.index.is_nearly_straight_or_straight and hand.middle.is_nearly_straight_or_straight)
        )

    def matches(self, hands: Hands, left: Hand, right: Hand, detected: GestureWeights) -> bool:
        """Check if hands are in a position suitable for clapping."""
        return (
            self.hands.hands_are_close and self.hand_in_good_shape(self.left) and self.hand_in_good_shape(self.right)
        )


TwoHandsGesturesDetector._ensure_all_detectors_registered()
