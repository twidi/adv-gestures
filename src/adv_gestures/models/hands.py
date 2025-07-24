from __future__ import annotations

from enum import Enum
from functools import cached_property
from math import sqrt
from time import time
from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

import numpy as np

from ..config import Config
from ..gestures import OVERRIDABLE_DEFAULT_GESTURES, Gestures
from ..smoothing import (
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
from .fingers import (
    AnyFinger,
    Finger,
    FingerIndex,
    IndexFinger,
    MiddleFinger,
    PinkyFinger,
    RingFinger,
    Thumb,
)
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

    def __init__(self, config: Config) -> None:
        """Initialize both hands."""
        self.config = config
        self.left: Hand = Hand(handedness=Handedness.LEFT, config=config)
        self.right: Hand = Hand(handedness=Handedness.RIGHT, config=config)

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
                if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.default.disable_all):
                    gesture = result.gestures[hand_index][0]  # Get the top gesture
                    gesture_type = (
                        None
                        if gesture.category_name in (None, "None", "Unknown")
                        else Gestures(gesture.category_name)
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
            custom_gestures: GestureWeights = {}
            if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.custom.disable.all):  # type: ignore[attr-defined]
                if gesture_type is None or gesture_type in OVERRIDABLE_DEFAULT_GESTURES:
                    custom_gestures = hand.detect_gestures()
            hand.update_custom_gestures(custom_gestures)


class Hand(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "is_facing_camera",
        "main_direction",
        "all_fingers_touching",
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
        # gestures durations
        "gestures_durations",
        "custom_gestures_durations",
    )

    def __init__(self, handedness: Handedness, config: Config) -> None:
        super().__init__()
        self.config = config
        self.handedness = handedness

        self.is_visible: bool = False
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

    def update_custom_gestures(self, custom_gestures: GestureWeights) -> None:
        """Update the custom gestures after detection."""
        self._raw_custom_gestures = custom_gestures

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
        detected: GestureWeights = {}

        thumb, index, middle, ring, pinky = self.thumb, self.index, self.middle, self.ring, self.pinky

        # Check for Middle Finger gesture
        # Middle finger is straight while index, ring, and pinky are not
        if (
            not self.is_gesture_disabled(Gestures.MIDDLE_FINGER)
            and middle.is_straight
            and index.is_not_straight_at_all
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
        ):
            detected[Gestures.MIDDLE_FINGER] = 1.0

        # Check for Victory gesture
        # Index and middle fingers are straight, others are not
        # (should be detected by default, but it's not always the case)
        if (
            not self.is_gesture_disabled(Gestures.VICTORY)
            and index.is_straight
            and middle.is_straight
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
            and thumb.is_fully_bent
            and not index.is_touching(middle)
        ):
            detected[Gestures.VICTORY] = 1.0

        # Check for Spock gesture
        # Index + middle together, ring+pinky together, forming a V.
        # All four fingers must be straight, hand must be facing camera, thumb must be fully bent
        if (
            not self.is_gesture_disabled(Gestures.SPOCK)
            and self.is_facing_camera
            and thumb.is_fully_bent
            and index.is_straight
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and index.is_touching(middle)
            and ring.is_touching(pinky)
            and not middle.is_touching(ring)
        ):
            detected[Gestures.SPOCK] = 1.0

        # Check for Rock gesture
        # Index and pinky are straight, others are not. Hand must not be facing camera.
        if (
            not self.is_gesture_disabled(Gestures.ROCK)
            and not self.is_facing_camera
            and index.is_straight
            and pinky.is_straight
            and not thumb.is_straight
            and middle.is_not_straight_at_all
            and ring.is_not_straight_at_all
        ):
            detected[Gestures.ROCK] = 1.0

        # Check for OK gesture
        # Index is touching thumb, others fingers are straight. Hand must be facing camera.
        if (
            not self.is_gesture_disabled(Gestures.OK)
            and self.is_facing_camera
            and index.tip_on_thumb
            and middle.is_nearly_straight_or_straight
            and ring.is_nearly_straight_or_straight
            and pinky.is_nearly_straight_or_straight
        ):
            detected[Gestures.OK] = 1.0

        # Check for Stop gesture
        # All fingers are straight and touching each others. Thumb is ignored. Hand must be facing camera.
        if (
            not self.is_gesture_disabled(Gestures.STOP)
            and self.is_facing_camera
            and index.is_straight
            and middle.is_straight
            and ring.is_straight
            and pinky.is_straight
            and index.is_touching(middle)
            and middle.is_touching(ring)
            and ring.is_touching(pinky)
        ):
            detected[Gestures.STOP] = 1.0

        # Check for Pinch and Pinch Touch gestures
        # Thumb is straight, fingers except index are not straight. Camera facing.
        pinch_enabled = not self.is_gesture_disabled(Gestures.PINCH)
        pinch_touch_enabled = not self.is_gesture_disabled(Gestures.PINCH_TOUCH)
        if (
            (pinch_enabled or pinch_touch_enabled)
            and self.is_facing_camera
            and thumb.is_nearly_straight_or_straight
            and not middle.is_straight
            and not ring.is_straight
            and not pinky.is_straight
        ):
            if pinch_enabled:
                detected[Gestures.PINCH] = 1.0
            if pinch_touch_enabled and index.tip_on_thumb:
                detected[Gestures.PINCH_TOUCH] = 1.0

        # Check for Gun and Finger Gun gestures
        # Thumb is straight or nearly, index and middle are straight and touching, ring and pinky are not.
        gun_enabled = not self.is_gesture_disabled(Gestures.GUN)
        finger_gun_enabled = not self.is_gesture_disabled(Gestures.FINGER_GUN)
        if (
            (gun_enabled or finger_gun_enabled)
            and thumb.is_nearly_straight_or_straight
            and index.is_nearly_straight_or_straight
            and ring.is_not_straight_at_all
            and pinky.is_not_straight_at_all
        ):
            if finger_gun_enabled:
                detected[Gestures.FINGER_GUN] = 1.0
            if gun_enabled and middle.is_nearly_straight_or_straight and index.is_touching(middle):
                detected[Gestures.GUN] = 1.0

        return detected


class Palm(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = ("centroid",)

    def __init__(self, hand: Hand, config: Config) -> None:
        super().__init__()
        self.config = config
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
