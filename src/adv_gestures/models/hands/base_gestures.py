from __future__ import annotations

import enum
from dataclasses import dataclass
from operator import attrgetter
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

from ...gestures import Gestures
from ...smoothing import GestureWeights

if TYPE_CHECKING:
    from ..fingers import AnyFinger
    from .hand import Hand


Range: TypeAlias = tuple[float | None, float | None]


class StatefulMode(enum.Enum):
    POST_DETECTION = "post_detection"  # Send detection after gesture ends (e.g., CLAP)
    CONTINUOUS = "continuous"  # Send detection during gesture (e.g., WAVE)


def direction_matches(angle: float | None, range_: Range | None) -> bool:
    if range_ is None:
        return True
    if angle is None:
        return False
    if range_[0] is not None and angle < range_[0]:
        return False
    if range_[1] is not None and angle > range_[1]:
        return False
    return True


class WithGesturesProtocol(Protocol):
    def is_gesture_disabled(self, gesture: Gestures) -> bool: ...


class StatefulDetectionState(enum.Enum):
    TRACKING = "tracking"
    ACTIVE = "active"  # For CONTINUOUS mode: gesture is active and being reported
    POST_DETECTING = "post_detecting"  # For POST_DETECTION mode: reporting after gesture ends


@dataclass
class DetectionState:
    tracking_start: float
    state: StatefulDetectionState
    post_detection_start: float | None = None
    data: dict[str, Any] | None = None  # Additional data for the detection, if needed

    @property
    def tracking_duration(self) -> float:
        """Duration since tracking started."""
        return time() - self.tracking_start

    @property
    def is_tracking(self) -> bool:
        """Check if the detection is currently tracking."""
        return self.state == StatefulDetectionState.TRACKING

    @property
    def is_active(self) -> bool:
        """Check if the detection is currently active."""
        return self.state == StatefulDetectionState.ACTIVE

    @property
    def is_post_detecting(self) -> bool:
        """Check if the detection is in post-detecting state."""
        return self.state == StatefulDetectionState.POST_DETECTING

    @property
    def post_detection_duration(self) -> float | None:
        """Duration since post-detection started, if applicable."""
        if self.post_detection_start is None:
            return None
        return time() - self.post_detection_start


WithGesturesType = TypeVar("WithGesturesType", bound=WithGesturesProtocol)


class BaseGestureDetector(Generic[WithGesturesType]):

    # To be defined for a baseclass for each "T"
    gestures_set: ClassVar[set[Gestures]]

    # To be defined for each final subclass
    gesture: ClassVar[Gestures]
    stateful_mode: ClassVar[StatefulMode | None] = None  # None = stateless detection
    # Only for stateful detectors
    min_gesture_duration: float | None = None
    max_gesture_duration: float | None = None
    post_detection_duration: ClassVar[float] = 0.0
    min_interval_between_detections: ClassVar[float | None] = None

    # Automatically defined for register classes
    _by_gesture: ClassVar[dict[Gestures, type[BaseGestureDetector[Any]]]]
    _register_classes: ClassVar[set[type[BaseGestureDetector[Any]]]] = set()
    detectors: dict[Gestures, BaseGestureDetector[WithGesturesType]] = {}

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
            if not base.__name__.startswith("_") and hasattr(base, "gestures_set") and not hasattr(base, "gesture"):
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
            self.detectors = {gesture: cls(obj) for gesture, cls in self._by_gesture.items()}

        # only used for stateful detectors
        self.tracked_states: list[DetectionState] = []
        self._last_valid_detection_time: float | None = None

    def detect_all(self, detected: GestureWeights) -> None:
        for detector in self.detectors.values():
            detector.detect(detected)

    def detect(self, detected: GestureWeights | None = None) -> GestureWeights:
        if self.detectors:
            detected = {} if detected is None else detected
            self.detect_all(detected)
            return detected

        assert detected is not None, "Detected must be provided for final detectors."

        if not self.obj.is_gesture_disabled(self.gesture) and self.pre_matches(detected):
            if self.stateful_mode is None:
                self.detect_stateless(detected)
            else:
                self.detect_stateful(detected)

        return detected

    def detect_stateless(self, detected: GestureWeights) -> None:
        if self._matches(detected):
            detected[self.gesture] = 1.0

    def detect_stateful(self, detected: GestureWeights) -> None:
        self._stateful_update_detections(detected)

        if self._stateful_has_valid_detection():
            detected[self.gesture] = 1.0

    def _stateful_is_expired(self, detection: DetectionState, now: float) -> bool:
        if detection.state == StatefulDetectionState.TRACKING:
            if self.max_gesture_duration is not None:
                return now - detection.tracking_start > self.max_gesture_duration
            return False

        elif detection.state == StatefulDetectionState.ACTIVE:
            # CONTINUOUS mode: expire if max_duration exceeded
            if self.max_gesture_duration is not None:
                return now - detection.tracking_start > self.max_gesture_duration
            return False

        else:  # POST_DETECTING
            # POST_DETECTION mode: expire after post_detection_duration
            if detection.post_detection_start is None:
                return True
            return now - detection.post_detection_start > self.post_detection_duration

    def _stateful_update_detections(self, detected: GestureWeights) -> None:
        now = time()
        gesture_found = self._matches(detected)

        # Update existing tracking detections
        for detection in self.tracked_states[:]:  # Copy list to allow modification during iteration
            if self._stateful_is_expired(detection, now):
                self.tracked_states.remove(detection)

            duration = now - detection.tracking_start

            if detection.state == StatefulDetectionState.TRACKING:
                if self.stateful_mode == StatefulMode.CONTINUOUS:
                    # For CONTINUOUS: move to ACTIVE after min_duration while gesture continues
                    if gesture_found and self.min_gesture_duration and duration >= self.min_gesture_duration:
                        detection.state = StatefulDetectionState.ACTIVE
                    elif not gesture_found:
                        # Gesture ended, remove detection
                        self.tracked_states.remove(detection)

                elif self.stateful_mode == StatefulMode.POST_DETECTION:
                    # For POST_DETECTION: move to POST_DETECTING when gesture ends
                    if not gesture_found:
                        if self.min_gesture_duration is not None and duration < self.min_gesture_duration:
                            # Too short, remove this detection
                            self.tracked_states.remove(detection)
                        else:
                            # Valid - move to post-detecting
                            detection.state = StatefulDetectionState.POST_DETECTING
                            detection.post_detection_start = now
                            self._last_valid_detection_time = now

            elif detection.state == StatefulDetectionState.ACTIVE:
                # CONTINUOUS mode: continue while gesture is detected
                if not gesture_found:
                    self.tracked_states.remove(detection)

        # Check if we can start a new detection
        if gesture_found and self._stateful_can_start_new_detection():
            # Check if any tracking or active detection already exists
            any_active_tracking = any(
                d.state in (StatefulDetectionState.TRACKING, StatefulDetectionState.ACTIVE)
                for d in self.tracked_states
            )

            if not any_active_tracking and self._stateful_can_start_tracking():
                self._stateful_start_tracking(now)

    def _stateful_start_tracking(self, now: float) -> DetectionState:
        """Start a new tracking state."""
        detection = DetectionState(tracking_start=now, state=StatefulDetectionState.TRACKING)
        self.tracked_states.append(detection)
        return detection

    def _stateful_has_valid_detection(self) -> bool:
        if self.stateful_mode == StatefulMode.POST_DETECTION:
            return any(d.state == StatefulDetectionState.POST_DETECTING for d in self.tracked_states)
        elif self.stateful_mode == StatefulMode.CONTINUOUS:
            return any(d.state == StatefulDetectionState.ACTIVE for d in self.tracked_states)
        return False

    def _stateful_can_start_new_detection(self) -> bool:
        if self.min_interval_between_detections is None:
            return True
        if self._last_valid_detection_time is None:
            return True
        return time() - self._last_valid_detection_time >= self.min_interval_between_detections

    def _stateful_can_start_tracking(self) -> bool:
        """Check if a new tracking state can be started. Override for custom logic."""
        return True

    @property
    def tracking_detections(self) -> list[DetectionState]:
        """Get all currently tracking detections."""
        return sorted([d for d in self.tracked_states if d.is_tracking], key=attrgetter("tracking_start"))

    @property
    def active_detections(self) -> list[DetectionState]:
        """Get all currently active detections."""
        return [d for d in self.tracked_states if d.is_active]

    @property
    def post_detecting_detections(self) -> list[DetectionState]:
        """Get all currently post-detecting detections, the oldest first."""
        return sorted([d for d in self.tracked_states if d.is_post_detecting], key=attrgetter("post_detection_start"))

    @staticmethod
    def hand_matches_direction(hand: Hand, range_: Range | None) -> bool:
        """Check if the hand matches the main direction."""
        return direction_matches(hand.main_direction_angle, range_)

    @staticmethod
    def finger_matches_straight_direction(finger: AnyFinger, range_: Range | None) -> bool:
        """Check if the finger matches the straight direction."""
        return direction_matches(finger.straight_direction_angle, range_)

    @staticmethod
    def finger_matches_tip_direction(finger: AnyFinger, range_: Range | None) -> bool:
        """Check if the finger matches the tip direction."""
        return direction_matches(finger.tip_direction_angle, range_)

    def pre_matches(self, detected: GestureWeights) -> bool:
        """Pre-check before matching the gesture."""
        return True

    def _matches(self, detected: GestureWeights) -> bool:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_data(self) -> dict[str, Any] | None:
        """Get data from the oldest post-detecting detection."""
        if post_detecting_states := self.post_detecting_detections:
            # Get the oldest POST_DETECTING state
            return post_detecting_states[0].data
        return None
