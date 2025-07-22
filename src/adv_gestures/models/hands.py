from __future__ import annotations

from enum import Enum
from math import sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

import numpy as np

from ..smoothing import (
    BoxSmoother,
    CoordSmoother,
    GestureSmoother,
    SmoothedBase,
    SmoothedProperty,
    smoothed_bool,
)
from .fingers import ADJACENT_FINGER_MAX_ANGLES, Finger, FingerIndex
from .gestures import OVERRIDABLE_DEFAULT_GESTURES, Gestures
from .landmarks import FINGERS_LANDMARKS, PALM_LANDMARKS, HandLandmark, Landmark

if TYPE_CHECKING:
    from ..recognizer import Recognizer


class Handedness(str, Enum):
    """Handedness enum for MediaPipe."""

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def from_data(cls, handedness_str: str) -> Handedness:
        """Convert MediaPipe handedness string to Handedness enum."""
        return cls(handedness_str.lower())

    def __str__(self) -> str:
        """Return the string representation of the handedness."""
        return self.value


class Box(NamedTuple):
    """Represents a bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


class Hands:

    def __init__(self) -> None:
        """Initialize both hands."""
        self.left: Hand = Hand(handedness=Handedness.LEFT)
        self.right: Hand = Hand(handedness=Handedness.RIGHT)

    def reset(self) -> None:
        """Reset both hands and clear all cached properties."""
        self.left.reset()
        self.right.reset()

    def update_hands(self, recognizer: Recognizer, width: int, height: int) -> None:
        """Update the hands object with new gesture recognition results."""

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
            if result.gestures and hand_index < len(result.gestures) and result.gestures[hand_index]:
                gesture = result.gestures[hand_index][0]  # Get the top gesture
                gesture_type = (
                    None if gesture.category_name in (None, "None", "Unknown") else Gestures(gesture.category_name)
                )

            # Update hand data
            hand.update(
                is_visible=True,
                wrist_landmark=hand_landmarks[HandLandmark.WRIST],
                default_gesture=gesture_type,
                all_landmarks=hand_landmarks,
            )

            # Update palm
            hand.palm.update([hand_landmarks[idx] for idx in PALM_LANDMARKS])

            # Update fingers
            for finger_idx, finger in enumerate(hand.fingers):
                finger_landmarks = [hand_landmarks[idx] for idx in FINGERS_LANDMARKS[finger_idx]]
                finger.update(landmarks=finger_landmarks)

            # Detect custom gestures if no default gesture was detected
            hand.update_custom_gesture(
                hand.detect_gesture()
                if gesture_type is None or gesture_type in OVERRIDABLE_DEFAULT_GESTURES
                else None
            )


class Hand(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "is_facing_camera",
        "main_direction",
        "all_fingers_touching",
        "bounding_box",
        "pinch_box",
        # All finger pairs (excluding thumb combinations which use touches_thumb)
        "index_middle_touching",
        "index_ring_touching",
        "index_pinky_touching",
        "middle_ring_touching",
        "middle_pinky_touching",
        "ring_pinky_touching",
    )

    def __init__(self, handedness: Handedness) -> None:
        super().__init__()
        self.handedness = handedness

        self.is_visible: bool = False
        self.palm: Palm = Palm(hand=self)
        self.fingers: list[Finger] = [Finger(index=FingerIndex(finger_idx), hand=self) for finger_idx in FingerIndex]
        self.wrist_landmark: Landmark | None = None
        self._raw_default_gesture: Gestures | None = None
        self._raw_custom_gesture: Gestures | None = None
        self._finger_touch_cache: dict[tuple[FingerIndex, FingerIndex], bool] = {}
        self.all_landmarks: list[Landmark] = []
        self._default_gesture_start_time: float | None = None
        self._custom_gesture_start_time: float | None = None
        self._gesture_start_time: float | None = None
        self._last_default_gesture: Gestures | None = None
        self._last_custom_gesture: Gestures | None = None
        self._last_gesture: Gestures | None = None

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
        is_visible: bool,
        wrist_landmark: Landmark | None,
        default_gesture: Gestures | None,
        all_landmarks: list[Landmark] | None = None,
    ) -> None:
        """Update the hand with new data from MediaPipe.

        Note: custom_gesture is not passed here because it needs to be calculated
        after the hand data is updated. Use update_custom_gesture() for that.
        """
        self.is_visible = is_visible
        self.wrist_landmark = wrist_landmark
        self._raw_default_gesture = default_gesture
        if all_landmarks is not None:
            self.all_landmarks = all_landmarks

    def update_custom_gesture(self, custom_gesture: Gestures | None) -> None:
        """Update the custom gesture after detection."""
        self._raw_custom_gesture = custom_gesture

    def __bool__(self) -> bool:
        """Check if the hand is visible and has a valid handedness."""
        return self.is_visible

    def _calc_is_facing_camera(self) -> bool:
        """Determine if the hand is showing its palm or back to the camera using cross product method."""
        if not self.handedness or not self.palm or len(self.palm.landmarks) < 6 or not self.wrist_landmark:
            return False

        # Get key landmarks for cross product calculation
        wrist = self.wrist_landmark
        thumb_mcp = self.palm.landmarks[1]  # THUMB_CMC
        pinky_mcp = self.palm.landmarks[5]  # PINKY_MCP

        # Create vectors from wrist to thumb MCP and wrist to pinky MCP
        # Note: MediaPipe uses normalized coordinates where x,y are in [0,1] and z represents depth
        vec1 = np.array(
            [
                thumb_mcp.x - wrist.x,
                thumb_mcp.y - wrist.y,
                (
                    (thumb_mcp.z_normalized - wrist.z_normalized)
                    if thumb_mcp.z_normalized is not None and wrist.z_normalized is not None
                    else 0
                ),
            ]
        )

        vec2 = np.array(
            [
                pinky_mcp.x - wrist.x,
                pinky_mcp.y - wrist.y,
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

    def _calc_main_direction(self) -> tuple[float, float] | None:
        """Calculate the main direction of the hand in the x/y plane.
        Returns a normalized vector (dx, dy) pointing from wrist to middle finger centroid.
        Note: This returns the direction in normalized coordinate space (0-1)."""
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

    def are_fingers_touching(self, finger1: FingerIndex | Finger, finger2: FingerIndex | Finger) -> bool:
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
        max_angle = None
        if (finger1, finger2) in ADJACENT_FINGER_MAX_ANGLES:
            max_angle = ADJACENT_FINGER_MAX_ANGLES[(finger1, finger2)]
        elif (finger2, finger1) in ADJACENT_FINGER_MAX_ANGLES:
            max_angle = ADJACENT_FINGER_MAX_ANGLES[(finger2, finger1)]

        if max_angle is None:
            self._finger_touch_cache[key] = False
            self._finger_touch_cache[(finger2, finger1)] = False
            return False

        # Get the finger objects directly by index
        finger1_obj = self.fingers[finger1]
        finger2_obj = self.fingers[finger2]

        # Check if both fingers are straight
        if finger1_obj.is_not_straight_at_all or finger2_obj.is_not_straight_at_all:
            self._finger_touch_cache[key] = False
            self._finger_touch_cache[(finger2, finger1)] = False
            return False

        # Check if directions are similar
        dir1 = finger1_obj.straight_direction
        dir2 = finger2_obj.straight_direction

        if dir1 is None or dir2 is None:
            self._finger_touch_cache[key] = False
            self._finger_touch_cache[(finger2, finger1)] = False
            return False

        # Calculate angle between directions using dot product
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        # Clamp to [-1, 1] to handle numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))

        # Calculate angle in degrees
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # Check if angle is within the threshold
        result = angle_deg <= max_angle

        # If not parallel enough, check if fingers are converging
        if not result and angle_deg < 5:
            # Get finger start and end points
            start1 = finger1_obj.start_point
            end1 = finger1_obj.end_point
            start2 = finger2_obj.start_point
            end2 = finger2_obj.end_point

            if start1 and end1 and start2 and end2:
                # Check if the lines are converging (intersecting ahead of the fingertips)
                # Using parametric line equations: P = P0 + t * d
                # Line 1: P1 = start1 + t1 * dir1
                # Line 2: P2 = start2 + t2 * dir2

                # Calculate the intersection point parameter t
                # We need to solve: start1 + t1 * dir1 = start2 + t2 * dir2
                dx1, dy1 = dir1
                dx2, dy2 = dir2
                x1, y1 = start1
                x2, y2 = start2

                # Determinant of the direction matrix
                det = dx1 * dy2 - dy1 * dx2

                if abs(det) > 0.001:  # Lines are not parallel
                    # Solve for t1 and t2 (in normalized direction units)
                    t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / det
                    t2 = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / det

                    # Calculate actual finger lengths
                    finger1_length = sqrt((end1[0] - start1[0]) ** 2 + (end1[1] - start1[1]) ** 2)
                    finger2_length = sqrt((end2[0] - start2[0]) ** 2 + (end2[1] - start2[1]) ** 2)

                    # Normalize t values by finger lengths to get position along finger
                    # t_normalized = 0 at base, = 1 at tip, > 1 beyond tip
                    t1_normalized = t1 / finger1_length if finger1_length > 0 else 0
                    t2_normalized = t2 / finger2_length if finger2_length > 0 else 0

                    # Check if intersection is ahead of both fingertips
                    result = t1_normalized > 1.0 and t2_normalized > 1.0

        # Cache and return result
        self._finger_touch_cache[key] = result
        self._finger_touch_cache[(finger2, finger1)] = result

        return bool(result)

    def _calc_all_fingers_touching(self) -> bool:
        """Check if all adjacent fingers are touching each other."""
        for finger_pair in ADJACENT_FINGER_MAX_ANGLES:
            finger1, finger2 = finger_pair
            if not self.are_fingers_touching(finger1, finger2):
                return False
        return True

    all_fingers_touching = smoothed_bool(_calc_all_fingers_touching)

    def _calc_index_middle_touching(self) -> bool:
        """Check if index and middle fingers are touching."""
        return self.are_fingers_touching(FingerIndex.INDEX, FingerIndex.MIDDLE)

    index_middle_touching = smoothed_bool(_calc_index_middle_touching)

    def _calc_middle_ring_touching(self) -> bool:
        """Check if middle and ring fingers are touching."""
        return self.are_fingers_touching(FingerIndex.MIDDLE, FingerIndex.RING)

    middle_ring_touching = smoothed_bool(_calc_middle_ring_touching)

    def _calc_ring_pinky_touching(self) -> bool:
        """Check if ring and pinky fingers are touching."""
        return self.are_fingers_touching(FingerIndex.RING, FingerIndex.PINKY)

    ring_pinky_touching = smoothed_bool(_calc_ring_pinky_touching)

    # Non-adjacent finger combinations
    def _calc_index_ring_touching(self) -> bool:
        """Check if index and ring fingers are touching."""
        return self.are_fingers_touching(FingerIndex.INDEX, FingerIndex.RING)

    index_ring_touching = smoothed_bool(_calc_index_ring_touching)

    def _calc_index_pinky_touching(self) -> bool:
        """Check if index and pinky fingers are touching."""
        return self.are_fingers_touching(FingerIndex.INDEX, FingerIndex.PINKY)

    index_pinky_touching = smoothed_bool(_calc_index_pinky_touching)

    def _calc_middle_pinky_touching(self) -> bool:
        """Check if middle and pinky fingers are touching."""
        return self.are_fingers_touching(FingerIndex.MIDDLE, FingerIndex.PINKY)

    middle_pinky_touching = smoothed_bool(_calc_middle_pinky_touching)

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
        if self.gesture not in (Gestures.PINCH, Gestures.PINCH_TOUCH):
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

    def _calc_custom_gesture(self) -> Gestures | None:
        """Get the custom gesture if detected."""
        # Track gesture changes for duration calculation
        current = self._raw_custom_gesture
        if current != self._last_custom_gesture:
            self._custom_gesture_start_time = time()
            self._last_custom_gesture = current
        return current

    custom_gesture = SmoothedProperty(_calc_custom_gesture, GestureSmoother, default_value=None)

    @property
    def custom_gesture_duration(self) -> float:
        """Get the duration in seconds that the current custom gesture has been active."""
        if self._custom_gesture_start_time is None:
            return 0.0
        return time() - self._custom_gesture_start_time

    @property
    def _raw_gesture(self) -> Gestures | None:
        """Get the final raw gesture (custom if detected, otherwise default)."""
        return self._raw_custom_gesture if self._raw_custom_gesture else self._raw_default_gesture

    def _calc_gesture(self) -> Gestures | None:
        """Get the final gesture for smoothing."""
        # Track gesture changes for duration calculation
        current = self._raw_gesture
        if current != self._last_gesture:
            self._gesture_start_time = time()
            self._last_gesture = current
        return current

    gesture = SmoothedProperty(_calc_gesture, GestureSmoother, default_value=None)

    @property
    def gesture_duration(self) -> float:
        """Get the duration in seconds that the current gesture has been active."""
        if self._gesture_start_time is None:
            return 0.0
        return time() - self._gesture_start_time

    def detect_gesture(self) -> Gestures | None:
        """Detect custom gestures.

        Returns:
            The detected gesture or None if no custom gesture is detected
        """

        thumb, index, middle, ring, pinky = self.fingers

        # Check all custom gestures
        for gesture in Gestures:

            # Check for Middle Finger gesture
            if gesture == Gestures.MIDDLE_FINGER:
                # Middle finger is straight while index, ring, and pinky are not
                if (
                    middle.is_straight
                    and index.is_not_straight_at_all
                    and ring.is_not_straight_at_all
                    and pinky.is_not_straight_at_all
                ):
                    return gesture

            elif gesture == Gestures.VICTORY:
                # Index and middle fingers are straight, others are not
                # (should be detected by default, but it's not always the case)
                if (
                    index.is_straight
                    and middle.is_straight
                    and ring.is_not_straight_at_all
                    and pinky.is_not_straight_at_all
                    and thumb.is_fully_bent
                    and not index.is_touching(middle)
                ):
                    return gesture

            elif gesture == Gestures.SPOCK:
                # Index + middle together, ring+pinky together, forming a V.
                # All four fingers must be straight, hand must be facing camera, thumb must be fully bent
                if (
                    self.is_facing_camera
                    and thumb.is_fully_bent
                    and index.is_straight
                    and middle.is_straight
                    and ring.is_straight
                    and pinky.is_straight
                    and index.is_touching(middle)
                    and ring.is_touching(pinky)
                    and not middle.is_touching(ring)
                ):
                    return gesture

            elif gesture == Gestures.ROCK:
                # Index and pinky are straight, others are not. Hand must not be facing camera.
                if (
                    not self.is_facing_camera
                    and index.is_straight
                    and pinky.is_straight
                    and not thumb.is_straight
                    and middle.is_not_straight_at_all
                    and ring.is_not_straight_at_all
                ):
                    return gesture

            elif gesture == Gestures.OK:
                # Index is touching thumb, others fingers are straight. Hand must be facing camera.
                if (
                    self.is_facing_camera
                    and index.touches_thumb
                    and middle.is_nearly_straight_or_straight
                    and ring.is_nearly_straight_or_straight
                    and pinky.is_nearly_straight_or_straight
                ):
                    return gesture

            elif gesture == Gestures.STOP:
                # All fingers are straight and touching each others. Thumb is ignored. Hand must be facing camera.
                if (
                    self.is_facing_camera
                    and index.is_straight
                    and middle.is_straight
                    and ring.is_straight
                    and pinky.is_straight
                    and index.is_touching(middle)
                    and middle.is_touching(ring)
                    and ring.is_touching(pinky)
                ):
                    return gesture

            elif gesture in (Gestures.PINCH, Gestures.PINCH_TOUCH):
                # Thumb is straight, fingers except index are not straight. Camera facing.
                if (
                    self.is_facing_camera
                    and thumb.is_nearly_straight_or_straight
                    and not middle.is_straight
                    and not ring.is_straight
                    and not pinky.is_straight
                    # and not index.is_fully_bent
                ):
                    return Gestures.PINCH_TOUCH if index.touches_thumb else Gestures.PINCH

            elif gesture == Gestures.GUN:
                # Thumb is straight or nearly, index and middle are straight and touching, ring and pinky are not.
                if (
                    thumb.is_nearly_straight_or_straight
                    and index.is_nearly_straight_or_straight
                    and middle.is_nearly_straight_or_straight
                    and index.is_touching(middle)
                    and ring.is_not_straight_at_all
                    and pinky.is_not_straight_at_all
                ):
                    return gesture

            elif gesture == Gestures.FINGER_GUN:
                # Same as gun but without the middle finger
                if (
                    thumb.is_nearly_straight_or_straight
                    and index.is_nearly_straight_or_straight
                    and middle.is_not_straight_at_all
                    and ring.is_not_straight_at_all
                    and pinky.is_not_straight_at_all
                ):
                    return gesture

        return None


class Palm(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = ("centroid",)

    def __init__(self, hand: Hand) -> None:
        super().__init__()
        self.hand = hand
        self.landmarks: list[Landmark] = []

    def reset(self) -> None:
        """Reset the palm and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        # Clear cached properties
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update(self, landmarks: list[Landmark]) -> None:
        """Update the palm with new landmarks."""
        self.landmarks = landmarks

    def _calc_centroid(self) -> tuple[float, float]:
        """Calculate the centroid of palm landmarks."""
        if not self.landmarks:
            return 0.0, 0.0

        centroid_x = sum(landmark.x for landmark in self.landmarks) / len(self.landmarks)
        centroid_y = sum(landmark.y for landmark in self.landmarks) / len(self.landmarks)

        return centroid_x, centroid_y

    centroid = SmoothedProperty(_calc_centroid, CoordSmoother)
