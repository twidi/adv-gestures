from __future__ import annotations

from collections import deque
from functools import cached_property
from math import acos, radians, sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from ...config import Config
from ...gestures import Gestures
from ...smoothing import (
    BoxSmoother,
    CoordSmoother,
    GestureSmoother,
    GestureWeights,
    MultiGestureSmoother,
    SmoothedBase,
    SmoothedProperty,
    smoothed_bool,
    smoothed_optional_float,
)
from ..fingers import (
    AnyFinger,
    Finger,
    FingerIndex,
    IndexFinger,
    MiddleFinger,
    PinkyFinger,
    RingFinger,
    Thumb,
)
from ..landmarks import FINGERS_LANDMARKS, PALM_LANDMARKS, HandLandmark, Landmark
from .hand_gestures import HandGesturesDetector
from .palm import Palm
from .utils import Box, Handedness

if TYPE_CHECKING:
    from ...recognizer import StreamInfo


class Hand(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "is_facing_camera",
        "is_showing_side",
        "main_direction",
        "is_waving",
        "all_adjacent_fingers_touching",
        "all_adjacent_fingers_except_thumb_touching",
        "bounding_box",
        "pinch_box",
        # All adjacent finger pairs
        "thumb_index_spread_angle",
        "thumb_index_touching",
        "index_middle_spread_angle",
        "index_middle_touching",
        "middle_ring_spread_angle",
        "middle_ring_touching",
        "ring_pinky_spread_angle",
        "ring_pinky_touching",
        # gestures
        "default_gesture",
        "custom_gestures",
        "gestures",
        "gestures_durations",
        "custom_gestures_durations",
    )

    def __init__(self, handedness: Handedness, config: Config) -> None:
        super().__init__()
        self.config = config
        self.handedness = handedness

        self.is_visible: bool = False
        self.stream_info: StreamInfo | None = None
        self.palm: Palm = Palm(hand=self, config=config)

        self.thumb = Thumb(hand=self, config=config)
        self.index = IndexFinger(hand=self, config=config)
        self.middle = MiddleFinger(hand=self, config=config)
        self.ring = RingFinger(hand=self, config=config)
        self.pinky = PinkyFinger(hand=self, config=config)
        self.fingers: tuple[AnyFinger, ...] = (
            self.thumb,
            self.index,
            self.middle,
            self.ring,
            self.pinky,
        )

        self.wrist_landmark: Landmark | None = None
        self._raw_default_gesture: Gestures | None = None
        self._raw_custom_gestures: GestureWeights = {}
        self._finger_touch_cache: dict[tuple[FingerIndex, FingerIndex], bool] = {}
        self.all_landmarks: list[Landmark] = []
        self._default_gesture_start_time: float | None = None
        self._last_default_gesture: Gestures | None = None
        self._custom_gestures_start_times: dict[Gestures, float] = {}
        self._gestures_start_times: dict[Gestures, float] = {}
        self._last_custom_gestures: set[Gestures] = set()
        self._last_gestures: set[Gestures] = set()

        # Direction history for wave detection (timestamp, direction_x, direction_y, direction_z)
        self._direction_history: deque[tuple[float, float, float, float]] = deque()

        self.gestures_detector = HandGesturesDetector(self)

    def reset(self) -> None:
        """Reset the hand state and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        # Clear cached properties by deleting them from __dict__
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

        self.is_visible = False

        # Clear finger touch cache
        self._finger_touch_cache.clear()

        # Reset palm
        if self.palm:
            self.palm.reset()

        # Reset each finger
        for finger in self.fingers:
            finger.reset()

    def update(
        self,
        default_gesture: Gestures | None,
        all_landmarks: list[Landmark] | None = None,
        stream_info: StreamInfo | None = None,
    ) -> None:
        """Update the hand with new data from MediaPipe.

        Note: custom_gesture is not passed here because it needs to be calculated
        after the hand data is updated. Use update_custom_gesture() for that.
        """
        self._raw_default_gesture = default_gesture
        self.stream_info = stream_info
        if all_landmarks is None:
            return

        self.is_visible = True
        self.wrist_landmark = all_landmarks[HandLandmark.WRIST]
        self.all_landmarks = all_landmarks

        # Update palm
        self.palm.update([self.all_landmarks[idx] for idx in PALM_LANDMARKS])

        # Update fingers
        for finger_idx, finger in enumerate(self.fingers):
            finger_landmarks = [self.all_landmarks[idx] for idx in FINGERS_LANDMARKS[finger_idx]]
            finger.update(landmarks=finger_landmarks)

        # Detect custom gestures (always detect them, regardless of default gesture)
        custom_gestures: GestureWeights = {}
        if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.custom.disable.all):  # type: ignore[attr-defined]
            custom_gestures = self.detect_gestures()
        self._raw_custom_gestures = custom_gestures

        # Update direction history if hand is visible and has direction
        direction = self.main_direction
        if direction is not None:
            current_time = time()
            # Add new direction to history
            self._direction_history.append((current_time, direction[0], direction[1], 0.0))

            # Remove directions older than 5 seconds
            cutoff_time = current_time - 5.0
            while self._direction_history and self._direction_history[0][0] < cutoff_time:
                self._direction_history.popleft()

    def __bool__(self) -> bool:
        """Check if the hand is visible and has a valid handedness."""
        return self.is_visible

    def _calc_is_facing_camera(self) -> bool:
        """Determine if the hand is showing its palm or back to the camera using cross product method."""
        if (
            not self.handedness
            or not self.palm
            or len(self.palm.landmarks) < len(PALM_LANDMARKS)
            or not self.wrist_landmark
        ):
            return False

        # Get key landmarks for cross product calculation
        wrist = self.wrist_landmark
        thumb_mcp = self.palm.landmarks[1]
        pinky_mcp = self.palm.landmarks[-1]

        # Create vectors from wrist to thumb MCP and wrist to pinky MCP
        # Note: x_normalized and y_normalized are MediaPipe normalized coordinates in the range [0,1]
        # and z_normalized represents depth
        vec1 = np.array(
            [
                thumb_mcp.x_normalized - wrist.x_normalized,
                thumb_mcp.y_normalized - wrist.y_normalized,
                (
                    (thumb_mcp.z_normalized - wrist.z_normalized)
                    if thumb_mcp.z_normalized is not None and wrist.z_normalized is not None
                    else 0
                ),
            ]
        )

        vec2 = np.array(
            [
                pinky_mcp.x_normalized - wrist.x_normalized,
                pinky_mcp.y_normalized - wrist.y_normalized,
                (
                    (pinky_mcp.z_normalized - wrist.z_normalized)
                    if pinky_mcp.z_normalized is not None and wrist.z_normalized is not None
                    else 0
                ),
            ]
        )

        # Calculate cross product to get normal vector
        normal = np.cross(vec1, vec2)

        # The z-component of the normal indicates orientation
        # For right hand: negative z = palm facing camera
        # For left hand: positive z = palm facing camera
        return cast(bool, normal[2] < 0 if self.handedness == Handedness.RIGHT else normal[2] > 0)

    is_facing_camera = smoothed_bool(_calc_is_facing_camera)

    def _calc_is_showing_side(self) -> bool:
        """Check if hand is showing its side (perpendicular to camera)."""
        if not self.stream_info:
            return False

        # Get centroids of all fingers except thumb (indices 1-4)
        centroids = []
        for i in range(1, 5):  # index, middle, ring, pinky
            centroid = self.fingers[i].centroid
            if not centroid:
                return False
            centroids.append(centroid)

        # Calculate mean position
        mean_x = sum(c[0] for c in centroids) / 4
        mean_y = sum(c[1] for c in centroids) / 4

        # Calculate variance (sum of squared distances from mean)
        variance_x = sum((c[0] - mean_x) ** 2 for c in centroids) / 4
        variance_y = sum((c[1] - mean_y) ** 2 for c in centroids) / 4
        total_variance = variance_x + variance_y

        # Normalize by frame dimensions (to make threshold independent of distance)
        normalized_variance = total_variance / (self.stream_info.width**2)

        # Threshold (empirically determined)
        threshold = 0.00025

        return normalized_variance < threshold

    is_showing_side = smoothed_bool(_calc_is_showing_side)

    def _calc_main_direction(self) -> tuple[float, float] | None:
        """Calculate the main direction of the hand in the x/y plane.
        Returns a normalized vector (dx, dy) pointing from wrist to middle finger centroid."""
        if not self.wrist_landmark or not self.fingers or len(self.fingers) <= FingerIndex.MIDDLE:
            return None

        # Get middle finger centroid position
        # fingers list is ordered by FingerIndex, so middle finger is at index 2
        middle_finger = self.fingers[FingerIndex.MIDDLE]
        if not middle_finger.landmarks:
            return None

        centroid = middle_finger.centroid
        if not centroid:
            return None

        middle_centroid_x, middle_centroid_y = centroid

        # Calculate direction vector from wrist to middle finger centroid
        dx = middle_centroid_x - self.wrist_landmark.x
        dy = middle_centroid_y - self.wrist_landmark.y

        # Normalize the vector
        magnitude = sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    main_direction = SmoothedProperty(_calc_main_direction, CoordSmoother)

    def _calc_is_waving(self) -> bool:
        """Check if the hand is waving (oscillating left-right movement)."""
        if not self._direction_history:
            return False

        current_time = time()
        duration = 1  # Check for wave pattern in last 2 seconds

        # Find directions within the duration window
        cutoff_time = current_time - duration
        directions_in_window = [(t, x, y) for t, x, y, _ in self._direction_history if t >= cutoff_time]

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

        # Verify angle changes are significant enough (30 degrees)
        # Check angles between consecutive significant directions
        min_angle_deg = 2.5
        min_angle_rad = radians(min_angle_deg)

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
            if angle1 >= min_angle_rad or angle2 >= min_angle_rad:
                return True

        return False

    is_waving = smoothed_bool(_calc_is_waving)

    def get_finger_spread_angle(
        self, finger1: FingerIndex | AnyFinger, finger2: FingerIndex | AnyFinger
    ) -> float | None:
        """Calculate the spread angle between two fingers.

        Returns:
            - Positive angle: fingers are spreading apart
            - Zero: fingers are parallel
            - Negative angle: fingers are crossing/overlapping
            - None: if angle cannot be calculated
        """
        if isinstance(finger1, Finger):
            finger1 = finger1.index
        if isinstance(finger2, Finger):
            finger2 = finger2.index

        # Get the finger objects
        finger1_obj = self.fingers[finger1]
        finger2_obj = self.fingers[finger2]

        # We only calculate angles for fingers that are at least nearly straight
        if finger1_obj.is_not_straight_at_all or finger2_obj.is_not_straight_at_all:
            return None

        # Get directions
        dir1 = finger1_obj.straight_direction
        dir2 = finger2_obj.straight_direction

        if dir1 is None or dir2 is None:
            return None

        # Get base positions
        base1 = finger1_obj.start_point
        base2 = finger2_obj.start_point

        if not base1 or not base2:
            return None

        # Calculate angle between directions using dot product
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        # Clamp to [-1, 1] to handle numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))

        # Calculate if fingers diverge from a common origin (forming a V)
        # Using parametric line equations to find intersection
        # Line 1: base1 + t1 * dir1
        # Line 2: base2 + t2 * dir2

        # Cross product between finger directions
        cross_product = dir1[0] * dir2[1] - dir1[1] * dir2[0]

        # Calculate angle in degrees (always positive from arccos)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # If directions are parallel (cross product ≈ 0), angle is 0
        if abs(cross_product) < 0.001:
            return 0.0

        # Calculate intersection point of the two finger lines
        # We solve: base1 + t1 * dir1 = base2 + t2 * dir2
        dx = base2[0] - base1[0]
        dy = base2[1] - base1[1]

        # Calculate t1 (parameter for line 1)
        t1 = (dx * dir2[1] - dy * dir2[0]) / cross_product

        # If t1 < 0, intersection is behind the base (fingers form a V, spreading)
        # If t1 > 0, intersection is ahead of the base (fingers converging)
        return float(angle_deg if t1 < 0 else -angle_deg)

    def are_fingers_touching(self, finger1: FingerIndex | AnyFinger, finger2: FingerIndex | AnyFinger) -> bool:
        """Check if two fingers are touching, computing and caching the result if needed."""

        if isinstance(finger1, Finger):
            finger1 = finger1.index
        if isinstance(finger2, Finger):
            finger2 = finger2.index

        finger1, finger2 = sorted((finger1, finger2))

        # Check if already computed
        key = (finger1, finger2)
        if key in self._finger_touch_cache:
            return self._finger_touch_cache[key]

        # Check if fingers are adjacent and get their max angle threshold
        max_angle: float | None = None
        angle_deg: float | None = None
        if key == (FingerIndex.THUMB, FingerIndex.INDEX):
            max_angle = self.config.hands.adjacent_fingers.thumb_index_max_angle_degrees
            angle_deg = self.thumb_index_spread_angle
        elif key == (FingerIndex.INDEX, FingerIndex.MIDDLE):
            max_angle = self.config.hands.adjacent_fingers.index_middle_max_angle_degrees
            angle_deg = self.index_middle_spread_angle
        elif key == (FingerIndex.MIDDLE, FingerIndex.RING):
            max_angle = self.config.hands.adjacent_fingers.middle_ring_max_angle_degrees
            angle_deg = self.middle_ring_spread_angle
        elif key == (FingerIndex.RING, FingerIndex.PINKY):
            max_angle = self.config.hands.adjacent_fingers.ring_pinky_max_angle_degrees
            angle_deg = self.ring_pinky_spread_angle

        if angle_deg is None or max_angle is None:
            self._finger_touch_cache[key] = False
            self._finger_touch_cache[(finger2, finger1)] = False
            return False

        # Check if angle is within the threshold
        # Fingers are touching if spread angle is less than or equal to threshold
        # (negative angles mean crossing, which definitely counts as touching)
        result = angle_deg <= max_angle

        # Cache and return result
        self._finger_touch_cache[key] = result
        self._finger_touch_cache[(finger2, finger1)] = result

        return bool(result)

    def _calc_all_adjacent_fingers_touching(self) -> bool:
        """Check if all adjacent fingers are touching each other."""
        return self.all_adjacent_fingers_except_thumb_touching and self.are_fingers_touching(
            FingerIndex.THUMB, FingerIndex.INDEX
        )

    all_adjacent_fingers_touching = smoothed_bool(_calc_all_adjacent_fingers_touching)

    def _calc_all_adjacent_fingers_except_thumb_touching(self) -> bool:
        """Check if all adjacent fingers are touching each other."""
        return all(
            self.are_fingers_touching(finger1, finger2)
            for finger1, finger2 in (
                (FingerIndex.INDEX, FingerIndex.MIDDLE),
                (FingerIndex.MIDDLE, FingerIndex.RING),
                (FingerIndex.RING, FingerIndex.PINKY),
            )
        )

    all_adjacent_fingers_except_thumb_touching = smoothed_bool(_calc_all_adjacent_fingers_except_thumb_touching)

    def _calc_thumb_index_spread_angle(self) -> float | None:
        """Calculate the spread angle between thumb and index fingers."""
        return self.get_finger_spread_angle(FingerIndex.THUMB, FingerIndex.INDEX)

    thumb_index_spread_angle = smoothed_optional_float(_calc_thumb_index_spread_angle)

    def _calc_thumb_index_touching(self) -> bool:
        """Check if index and middle fingers are touching."""
        return self.are_fingers_touching(FingerIndex.THUMB, FingerIndex.INDEX)

    thumb_index_touching = smoothed_bool(_calc_thumb_index_touching)

    def _calc_index_middle_spread_angle(self) -> float | None:
        """Calculate the spread angle between index and middle fingers."""
        return self.get_finger_spread_angle(FingerIndex.INDEX, FingerIndex.MIDDLE)

    index_middle_spread_angle = smoothed_optional_float(_calc_index_middle_spread_angle)

    def _calc_index_middle_touching(self) -> bool:
        """Check if index and middle fingers are touching."""
        return self.are_fingers_touching(FingerIndex.INDEX, FingerIndex.MIDDLE)

    index_middle_touching = smoothed_bool(_calc_index_middle_touching)

    def _calc_middle_ring_spread_angle(self) -> float | None:
        """Calculate the spread angle between middle and ring fingers."""
        return self.get_finger_spread_angle(FingerIndex.MIDDLE, FingerIndex.RING)

    middle_ring_spread_angle = smoothed_optional_float(_calc_middle_ring_spread_angle)

    def _calc_middle_ring_touching(self) -> bool:
        """Check if middle and ring fingers are touching."""
        return self.are_fingers_touching(FingerIndex.MIDDLE, FingerIndex.RING)

    middle_ring_touching = smoothed_bool(_calc_middle_ring_touching)

    def _calc_ring_pinky_spread_angle(self) -> float | None:
        """Calculate the spread angle between ring and pinky fingers."""
        return self.get_finger_spread_angle(FingerIndex.RING, FingerIndex.PINKY)

    ring_pinky_spread_angle = smoothed_optional_float(_calc_ring_pinky_spread_angle)

    def _calc_ring_pinky_touching(self) -> bool:
        """Check if ring and pinky fingers are touching."""
        return self.are_fingers_touching(FingerIndex.RING, FingerIndex.PINKY)

    ring_pinky_touching = smoothed_bool(_calc_ring_pinky_touching)

    def _calc_bounding_box(self) -> Box | None:
        """Calculate the bounding box of the hand."""
        if not self.all_landmarks:
            return None

        x_coords = [landmark.x for landmark in self.all_landmarks]
        y_coords = [landmark.y for landmark in self.all_landmarks]

        return Box(min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    bounding_box = SmoothedProperty(_calc_bounding_box, BoxSmoother)

    def _calc_pinch_box(self) -> Box | None:
        """Calculate the bounding box around the pinch gesture fingertips.
        Returns None if not pinching."""
        # Check if PINCH or PINCH_TOUCH is in the detected gestures
        if Gestures.PINCH not in self.gestures and Gestures.PINCH_TOUCH not in self.gestures:
            # If no pinch gesture detected, return None
            return None

        # Get thumb and index finger tips
        thumb_tip = self.fingers[FingerIndex.THUMB].end_point
        index_tip = self.fingers[FingerIndex.INDEX].end_point

        if not thumb_tip or not index_tip:
            return None

        # Calculate bounding box
        min_x = min(thumb_tip[0], index_tip[0])
        max_x = max(thumb_tip[0], index_tip[0])
        min_y = min(thumb_tip[1], index_tip[1])
        max_y = max(thumb_tip[1], index_tip[1])

        # Add padding based on frame dimensions and orientation
        if self.stream_info:
            if self.stream_info.width > self.stream_info.height:
                # Landscape: 2% width, 1% height
                padding_x = self.stream_info.width * 0.02
                padding_y = self.stream_info.height * 0.01
            elif self.stream_info.width < self.stream_info.height:
                # Portrait: 1% width, 2% height
                padding_x = self.stream_info.width * 0.01
                padding_y = self.stream_info.height * 0.02
            else:
                # Square: 1.5% for both
                padding_x = self.stream_info.width * 0.015
                padding_y = self.stream_info.height * 0.015

            min_x -= padding_x
            max_x += padding_x
            min_y -= padding_y
            max_y += padding_y

            # Ensure the box stays within frame boundaries
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(self.stream_info.width, max_x)
            max_y = min(self.stream_info.height, max_y)

        return Box(min_x, min_y, max_x, max_y)

    pinch_box = SmoothedProperty(_calc_pinch_box, BoxSmoother)

    def _calc_default_gesture(self) -> Gestures | None:
        """Get the default gesture from MediaPipe."""
        # Track gesture changes for duration calculation
        current = self._raw_default_gesture
        if current != self._last_default_gesture:
            self._default_gesture_start_time = time()
            self._last_default_gesture = current
        return current

    default_gesture = SmoothedProperty(_calc_default_gesture, GestureSmoother, default_value=None)

    @property
    def default_gesture_duration(self) -> float:
        """Get the duration in seconds that the current default gesture has been active."""
        if self._default_gesture_start_time is None:
            return 0.0
        return time() - self._default_gesture_start_time

    def _calc_custom_gestures(self) -> GestureWeights:
        """Get the custom gestures if detected."""
        current_gestures = set(self._raw_custom_gestures.keys())
        now = time()

        # Detect new gestures
        new_gestures = current_gestures - self._last_custom_gestures
        for gesture in new_gestures:
            self._custom_gestures_start_times[gesture] = now

        # Remove ended gestures
        ended_gestures = self._last_custom_gestures - current_gestures
        for gesture in ended_gestures:
            self._custom_gestures_start_times.pop(gesture, None)

        self._last_custom_gestures = current_gestures
        return self._raw_custom_gestures

    custom_gestures = SmoothedProperty(_calc_custom_gestures, MultiGestureSmoother, default_value={})

    def _calc_gestures(self) -> GestureWeights:
        """Get the final gestures (custom + default)."""
        result = self._raw_custom_gestures.copy()

        # Add the default gesture if present
        if self._raw_default_gesture:
            # If no custom gestures or if default not already in custom
            if not result or self._raw_default_gesture not in result:
                result[self._raw_default_gesture] = 1.0

        # Track gesture changes
        current_gestures = set(result.keys())
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
        return result

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

    @cached_property
    def custom_gestures_durations(self) -> dict[Gestures, float]:
        """Get the durations for all currently active custom gestures."""
        now = time()
        return {
            gesture: now - start_time
            for gesture, start_time in self._custom_gestures_start_times.items()
            if gesture in self.custom_gestures
        }

    def is_gesture_disabled(self, gesture: Gestures) -> bool:
        return bool(getattr(self.config.hands.gestures.custom.disable, gesture.name))

    def detect_gestures(self) -> GestureWeights:
        """Detect all applicable custom gestures with weights.

        Returns:
            Dictionary of detected gestures with weight 1.0 for each
        """
        return self.gestures_detector.detect()
