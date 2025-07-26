from __future__ import annotations

import enum
from dataclasses import dataclass
from math import cos, radians
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

from ...gestures import Gestures
from ...smoothing import GestureWeights

if TYPE_CHECKING:
    from ..fingers import AnyFinger
    from .hand import Hand


DirectionMatcher: TypeAlias = tuple[tuple[float, float] | None, tuple[float, float] | None] | None


def up_with_tolerance(angle_deg: float) -> DirectionMatcher:
    # no check on x, and y must be up with `angle_deg` tolerance
    return None, (-1, -cos(radians(angle_deg)))


def direction_matches(direction: tuple[float, float] | None, direction_range: DirectionMatcher) -> bool:
    if direction_range is None:
        return True
    if direction is None:
        return False
    for axe in (0, 1):  # Check x and y axes
        if (_range := direction_range[axe]) is None:
            continue
        if not _range[0] <= direction[axe] <= _range[1]:
            return False
    return True


class WithGesturesProtocol(Protocol):
    def is_gesture_disabled(self, gesture: Gestures) -> bool: ...


class StatefulDetectionState(enum.Enum):
    TRACKING = "tracking"
    POST_DETECTING = "post_detecting"


@dataclass
class DetectionState:
    tracking_start: float
    state: StatefulDetectionState
    post_detection_start: float | None = None


WithGesturesType = TypeVar("WithGesturesType", bound=WithGesturesProtocol)


class BaseGestureDetector(Generic[WithGesturesType]):

    # To be defined for a baseclass for each "T"
    gestures_set: ClassVar[set[Gestures]]

    # To be defined for each final subclass
    gesture: ClassVar[Gestures]
    stateful: ClassVar[bool] = False
    # For pre-matching checks
    main_direction_range: ClassVar[DirectionMatcher] = None
    # Only for stateful detectors
    min_gesture_duration: ClassVar[float | None] = None
    max_gesture_duration: ClassVar[float | None] = None
    post_detection_duration: ClassVar[float] = 0.0
    min_interval_between_detections: ClassVar[float | None] = None

    # Automatically defined for register classes
    _by_gesture: ClassVar[dict[Gestures, type[BaseGestureDetector[Any]]]]
    _register_classes: ClassVar[set[type[BaseGestureDetector[Any]]]] = set()
    _detectors: list[BaseGestureDetector[WithGesturesType]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "gesture"):
            # Look for the base class to use for registration
            register_class = cls._get_register_class()

            # Check if the gesture is in the set of gestures for the base class
            if cls.gesture not in register_class.gestures_set:
                raise ValueError(
                    f"Gesture {cls.gesture} is not in the set of gestures {register_class.gestures_set} "
                    f"for class {register_class.__name__}."
                )

            # Init the registration if not already done
            if register_class not in BaseGestureDetector._register_classes:
                BaseGestureDetector._register_classes.add(register_class)
                register_class._by_gesture = {}

            # Register the gesture detector class
            if cls.gesture in register_class._by_gesture:
                raise ValueError(f"Gesture {cls.gesture} already has a detector in {register_class.__name__}")

            register_class._by_gesture[cls.gesture] = cls

    @classmethod
    def _get_register_class(cls) -> type[BaseGestureDetector[Any]]:
        """Get the class used for registering gesture detectors."""
        for base in cls.__mro__[1:]:
            if hasattr(base, "gestures_set") and not hasattr(base, "gesture"):
                return base
        raise ValueError(f"Class {cls.__name__} must have a 'gestures_set' attribute defined in a base class.")

    @classmethod
    def _ensure_all_detectors_registered(cls) -> None:
        """Ensure that all custom gestures have detectors registered."""
        if cls not in BaseGestureDetector._register_classes:
            raise ValueError(f"Class {cls.__name__} is not a base class for gesture detection.")
        registered = set(cls._by_gesture.keys())
        if registered != cls.gestures_set:
            raise ValueError(
                f"Not all hands gestures have detectors registered. Missing: {cls.gestures_set - registered}"
            )

    def __init__(self, obj: WithGesturesType) -> None:
        self.obj: WithGesturesType = obj
        if self.__class__ in BaseGestureDetector._register_classes:
            self._detectors = [cls(obj) for cls in self._by_gesture.values()]

        # only used for stateful detectors
        self.tracked_states: list[DetectionState] = []
        self._last_valid_detection_time: float | None = None

    def detect_all(self, detected: GestureWeights) -> None:
        for detector in self._detectors:
            detector.detect(detected)

    def detect(self, detected: GestureWeights | None = None) -> GestureWeights:
        if self._detectors:
            detected = {} if detected is None else detected
            self.detect_all(detected)
            return detected

        assert detected is not None, "Detected must be provided for final detectors."

        if not self.obj.is_gesture_disabled(self.gesture) and self.pre_matches(detected):
            if self.stateful:
                self.detect_stateful(detected)
            else:
                self.detect_stateless(detected)

        return detected

    def detect_stateless(self, detected: GestureWeights) -> None:
        if self._matches(detected):
            detected[self.gesture] = 1.0

    def detect_stateful(self, detected: GestureWeights) -> None:
        self._stateful_cleanup_detections()
        self._stateful_update_detections(detected)

        if self._stateful_has_valid_detection():
            detected[self.gesture] = 1.0

    def _stateful_cleanup_detections(self) -> None:
        now = time()
        self.tracked_states = [d for d in self.tracked_states if not self._stateful_is_expired(d, now)]

    def _stateful_is_expired(self, detection: DetectionState, now: float) -> bool:
        if detection.state == StatefulDetectionState.TRACKING:
            if self.max_gesture_duration is not None:
                return now - detection.tracking_start > self.max_gesture_duration
            return False
        else:  # POST_DETECTING
            if detection.post_detection_start is None:
                return True
            return now - detection.post_detection_start > self.post_detection_duration

    def _stateful_update_detections(self, detected: GestureWeights) -> None:
        now = time()
        gesture_found = self._matches(detected)

        # Update existing tracking detections
        for detection in self.tracked_states[:]:  # Copy list to allow modification during iteration
            if detection.state == StatefulDetectionState.TRACKING and not gesture_found:
                # Check if duration is valid
                duration = now - detection.tracking_start

                if self.min_gesture_duration is not None and duration < self.min_gesture_duration:
                    # Too short, remove this detection
                    self.tracked_states.remove(detection)
                else:
                    # Valid - move to post-detecting
                    detection.state = StatefulDetectionState.POST_DETECTING
                    detection.post_detection_start = now
                    self._last_valid_detection_time = now

        # Check if we can start a new detection
        if gesture_found and self._stateful_can_start_new_detection():
            # Check if any tracking already exists
            any_tracking_exists = any(d.state == StatefulDetectionState.TRACKING for d in self.tracked_states)

            if not any_tracking_exists:
                # Start new tracking
                self.tracked_states.append(DetectionState(tracking_start=now, state=StatefulDetectionState.TRACKING))

    def _stateful_has_valid_detection(self) -> bool:
        return any(d.state == StatefulDetectionState.POST_DETECTING for d in self.tracked_states)

    def _stateful_can_start_new_detection(self) -> bool:
        if self.min_interval_between_detections is None:
            return True
        if self._last_valid_detection_time is None:
            return True
        return time() - self._last_valid_detection_time >= self.min_interval_between_detections

    def _matches_main_direction(self) -> bool:
        if self.main_direction_range is None:
            return True
        return self.matches_main_direction(self.main_direction_range)

    @staticmethod
    def hand_matches_direction(hand: Hand, main_direction_range: DirectionMatcher) -> bool:
        """Check if the hand matches the main direction."""
        return direction_matches(hand.main_direction, main_direction_range)

    @staticmethod
    def finger_matches_straight_direction(finger: AnyFinger, straight_direction_range: DirectionMatcher) -> bool:
        """Check if the finger matches the straight direction."""
        return direction_matches(finger.straight_direction, straight_direction_range)

    @staticmethod
    def finger_matches_tip_direction(finger: AnyFinger, tip_direction_range: DirectionMatcher) -> bool:
        """Check if the finger matches the tip direction."""
        return direction_matches(finger.tip_direction, tip_direction_range)

    def matches_main_direction(self, main_direction_range: DirectionMatcher) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def pre_matches(self, detected: GestureWeights) -> bool:
        """Pre-check before matching the gesture."""
        if self.main_direction_range is not None and not self._matches_main_direction():
            return False
        return True

    def _matches(self, detected: GestureWeights) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")
