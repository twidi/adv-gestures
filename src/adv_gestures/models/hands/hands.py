from __future__ import annotations

from functools import cached_property
from math import sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar

from ...config import Config
from ...gestures import Gestures
from ...smoothing import (
    GestureWeights,
    MultiGestureSmoother,
    SmoothedBase,
    SmoothedProperty,
    smoothed_bool,
    smoothed_optional_float,
)
from .hand import Hand
from .hands_gestures import TwoHandsGesturesDetector
from .utils import Handedness

if TYPE_CHECKING:
    from ...recognizer import Recognizer, StreamInfo


class Hands(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "gestures",
        "gestures_durations",
        "hands_distance",
        "hands_are_close",
    )

    def __init__(self, config: Config) -> None:
        """Initialize both hands."""
        super().__init__()
        self.config = config
        self.stream_info: StreamInfo | None = None
        self.left: Hand = Hand(handedness=Handedness.LEFT, config=config)
        self.right: Hand = Hand(handedness=Handedness.RIGHT, config=config)

        self._raw_gestures: GestureWeights = {}
        self._gestures_start_times: dict[Gestures, float] = {}
        self._last_gestures: set[Gestures] = set()

        self.gestures_detector = TwoHandsGesturesDetector(self)

    def reset(self) -> None:
        """Reset both hands and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        self.left.reset()
        self.right.reset()

        # Clear cached properties by deleting them from __dict__
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update_hands(self, recognizer: Recognizer, stream_info: StreamInfo | None = None) -> None:
        """Update the hands object with new gesture recognition results."""
        # Store the stream info
        self.stream_info = stream_info

        # If the recognizer didn't run yet, do nothing
        if not (result := recognizer.last_result):
            return

        # Reset all hands first
        self.reset()

        if not result.hand_landmarks:
            return

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            # Get handedness
            handedness = None
            if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
                handedness = Handedness.from_data(result.handedness[hand_index][0].category_name)

            # Skip if handedness not detected
            if not handedness:
                continue

            # Get the appropriate hand
            hand = self.left if handedness == Handedness.LEFT else self.right

            # Get default gesture information
            gesture_type = None
            if (
                hand_landmarks is not None
                and result.gestures
                and hand_index < len(result.gestures)
                and result.gestures[hand_index]
            ):
                if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.default.disable_all):
                    gesture = result.gestures[hand_index][0]  # Get the top gesture
                    gesture_type = (
                        None
                        if gesture.category_name in (None, "None", "Unknown")
                        else Gestures(gesture.category_name)
                    )

            # Update hand data
            hand.update(
                default_gesture=gesture_type,
                all_landmarks=hand_landmarks,
                stream_info=stream_info,
            )

        gestures: GestureWeights = {}
        if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.two_hands.disable.all):  # type: ignore[attr-defined]
            gestures = self.detect_gestures()
        self._raw_gestures = gestures

    def _calc_gestures(self) -> GestureWeights:
        """Get the custom gestures if detected."""
        current_gestures = set(self._raw_gestures.keys())
        now = time()

        # Detect new gestures
        new_gestures = current_gestures - self._last_gestures
        for gesture in new_gestures:
            self._gestures_start_times[gesture] = now

        # Remove ended gestures
        ended_gestures = self._last_gestures - current_gestures
        for gesture in ended_gestures:
            self._gestures_start_times.pop(gesture, None)

        self._last_gestures = current_gestures
        return self._raw_gestures

    gestures = SmoothedProperty(_calc_gestures, MultiGestureSmoother, default_value={})

    @cached_property
    def gestures_durations(self) -> dict[Gestures, float]:
        """Get the durations for all currently active gestures."""
        now = time()
        return {
            gesture: now - start_time
            for gesture, start_time in self._gestures_start_times.items()
            if gesture in self.gestures
        }

    def is_gesture_disabled(self, gesture: Gestures) -> bool:
        return bool(getattr(self.config.hands.gestures.two_hands.disable, gesture.name))

    def detect_gestures(self) -> GestureWeights:
        """Detect all applicable custom gestures with weights.

        Returns:
            Dictionary of detected gestures with weight 1.0 for each
        """
        return self.gestures_detector.detect()

    def _calc_hands_distance(self) -> float | None:
        """Calculate the distance in pixels between the two hands' palm centroids."""
        if not self.left or not self.right:
            return None

        left_centroid = self.left.palm.centroid
        right_centroid = self.right.palm.centroid

        if not left_centroid or not right_centroid:
            return None

        # Calculate Euclidean distance between centroids
        dx = right_centroid[0] - left_centroid[0]
        dy = right_centroid[1] - left_centroid[1]

        return sqrt(dx * dx + dy * dy)

    hands_distance = smoothed_optional_float(_calc_hands_distance)

    def _calc_hands_are_close(self) -> bool:
        """Check if the two hands are close based on their distance."""
        distance = self.hands_distance
        if distance is None:
            return False

        # Calculate the average distance from wrist to thumb MCP for both hands
        threshold = 0.0
        count = 0

        for hand in (self.left, self.right):
            if hand and hand.wrist_landmark and hand.thumb and len(hand.thumb.landmarks) > 1:
                # Thumb MCP is the second landmark (index 1)
                thumb_mcp = hand.thumb.landmarks[1]
                wrist = hand.wrist_landmark
                dx = thumb_mcp.x - wrist.x
                dy = thumb_mcp.y - wrist.y

                threshold += sqrt(dx * dx + dy * dy)
                count += 1

        if count == 0:
            return False

        # Average threshold
        threshold = threshold / count

        # Hands are close if distance is less than the average wrist->thumb_mcp distance
        return distance < threshold

    hands_are_close = smoothed_bool(_calc_hands_are_close)
