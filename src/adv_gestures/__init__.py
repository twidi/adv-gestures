#!/usr/bin/env python3
"""List all available cameras with their paths and names using linuxpy.video."""

from __future__ import annotations

import glob
import re
import time
from collections.abc import Callable
from contextlib import suppress
from enum import Enum, IntEnum
from functools import cached_property
from math import sqrt
from typing import ClassVar, NamedTuple, TypeAlias, cast

import numpy as np
import typer
from mediapipe.tasks.python.components.containers import (  # type: ignore[import-untyped]
    NormalizedLandmark,
)

try:
    from linuxpy.video.device import (  # type: ignore[import-untyped]
        BufferType,
        Device,
        PixelFormat,
    )
except ImportError:
    print("Error: linuxpy is not installed. Install it with: pip install linuxpy")
    exit(1)

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    print("Error: opencv-python is not installed. Install it with: pip install opencv-python")
    exit(1)

try:
    import mediapipe as mp  # type: ignore[import-untyped]
    from mediapipe.tasks import python  # type: ignore[import-untyped]
    from mediapipe.tasks.python import vision  # type: ignore[import-untyped]
except ImportError:
    print("Error: mediapipe is not installed. Install it with: pip install mediapipe")
    exit(1)

ImageArray: TypeAlias = cv2.typing.MatLike  # Type alias for images (numpy arrays)

MIRROR_OUTPUT = True
DESIRED_SIZE = 1280  # size of the max dimension of the camera

# Finger straightness thresholds - can be customized per finger
FINGER_STRAIGHT_THRESHOLD = 0.90  # Default score above this is considered straight (strict)
FINGER_NEARLY_STRAIGHT_THRESHOLD = 0.70  # Default score above this is considered nearly straight (relaxed)

# Global variable to store the latest gesture result
latest_gesture_result: vision.GestureRecognizerResult | None = None


class Gestures(str, Enum):
    # Those ones are created by MediaPipe
    CLOSED_FIST = "Closed_Fist"
    OPEN_PALM = "Open_Palm"
    POINTING_UP = "Pointing_Up"
    THUMB_DOWN = "Thumb_Down"
    THUMB_UP = "Thumb_Up"
    VICTORY = "Victory"
    LOVE = "ILoveYou"
    # Those are the ones we detect
    MIDDLE_FINGER = "Middle_Finger"
    SPOCK = "Spock"
    ROCK = "Rock"
    OK = "OK"
    STOP = "Stop"
    PINCH = "Pinch"
    GUN = "Gun"
    FINGER_GUN = "Finger_Gun"  # Gun without the middle finger


DEFAULT_GESTURES = {
    Gestures.CLOSED_FIST,
    Gestures.OPEN_PALM,
    Gestures.POINTING_UP,
    Gestures.THUMB_DOWN,
    Gestures.THUMB_UP,
    Gestures.VICTORY,
    Gestures.LOVE,
}
OVERRIDABLE_DEFAULT_GESTURES = {Gestures.OPEN_PALM}


class HandLandmark(IntEnum):
    """MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


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


class FingerIndex(IntEnum):
    """Finger index constants for easier reference."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


LandmarkGroup: TypeAlias = list[HandLandmark]


class LandmarkGroups:
    PALM: ClassVar[LandmarkGroup] = [
        HandLandmark.WRIST,
        HandLandmark.THUMB_CMC,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    ]
    THUMB: ClassVar[LandmarkGroup] = [
        HandLandmark.THUMB_CMC,
        HandLandmark.THUMB_MCP,
        HandLandmark.THUMB_IP,
        HandLandmark.THUMB_TIP,
    ]
    INDEX: ClassVar[LandmarkGroup] = [
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.INDEX_FINGER_PIP,
        HandLandmark.INDEX_FINGER_DIP,
        HandLandmark.INDEX_FINGER_TIP,
    ]
    MIDDLE: ClassVar[LandmarkGroup] = [
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_PIP,
        HandLandmark.MIDDLE_FINGER_DIP,
        HandLandmark.MIDDLE_FINGER_TIP,
    ]
    RING: ClassVar[LandmarkGroup] = [
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.RING_FINGER_PIP,
        HandLandmark.RING_FINGER_DIP,
        HandLandmark.RING_FINGER_TIP,
    ]
    PINKY: ClassVar[LandmarkGroup] = [
        HandLandmark.PINKY_MCP,
        HandLandmark.PINKY_PIP,
        HandLandmark.PINKY_DIP,
        HandLandmark.PINKY_TIP,
    ]


PALM_LANDMARKS = LandmarkGroups.PALM
FINGERS_LANDMARKS = [
    LandmarkGroups.THUMB,
    LandmarkGroups.INDEX,
    LandmarkGroups.MIDDLE,
    LandmarkGroups.RING,
    LandmarkGroups.PINKY,
]

# Adjacent finger pairs with their maximum angle thresholds for touching detection
# Each pair maps to a maximum angle in degrees
ADJACENT_FINGER_MAX_ANGLES = {
    (FingerIndex.INDEX, FingerIndex.MIDDLE): 2.5,
    (FingerIndex.MIDDLE, FingerIndex.RING): 1.5,
    (FingerIndex.RING, FingerIndex.PINKY): 2.0,
}

# Per-finger straightness configuration
# Each finger can have custom thresholds and parameters
FINGER_STRAIGHTNESS_CONFIG_DEFAULT = {
    "straight_threshold": 0.85,
    "nearly_straight_threshold": 0.65,
    # Parameters for advanced straightness calculation
    "distal_segments_max_ratio": 1.5,
    "distal_segments_max_ratio_back": 1.5,
    "max_angle_degrees": 10,
    "segment_ratio_score_at_threshold": 0.7,
    "segment_ratio_decay_rate": 2.0,
    "segment_ratio_linear_range": 0.15,
    "angle_score_linear_range": 0.15,
    "angle_score_at_threshold": 0.85,
    "angle_decay_rate": 20.0,
    "angle_score_weight": 0.7,
    "segment_ratio_weight": 0.3,
}
FINGER_STRAIGHTNESS_CONFIG = {
    FingerIndex.THUMB: {
        "straight_threshold": 0.90,
        "nearly_straight_threshold": 0.70,
        # Parameters for thumb's basic straightness calculation
        "alignment_threshold": 0.01,
        "max_deviation_for_zero_score": 0.1,
    },
    FingerIndex.INDEX: FINGER_STRAIGHTNESS_CONFIG_DEFAULT | {},
    FingerIndex.MIDDLE: FINGER_STRAIGHTNESS_CONFIG_DEFAULT | {},
    FingerIndex.RING: FINGER_STRAIGHTNESS_CONFIG_DEFAULT | {},
    FingerIndex.PINKY: FINGER_STRAIGHTNESS_CONFIG_DEFAULT | {},
}

# Colors for drawing fingers (BGR format for OpenCV)
FINGER_COLORS = [
    (255, 0, 0),  # Blue - THUMB
    (0, 255, 0),  # Green - INDEX
    (0, 255, 255),  # Yellow - MIDDLE
    (255, 0, 255),  # Magenta - RING
    (255, 255, 0),  # Cyan - PINKY
]


class Hands:

    def __init__(self) -> None:
        """Initialize both hands."""
        self.left: Hand = Hand(handedness=Handedness.LEFT)
        self.right: Hand = Hand(handedness=Handedness.RIGHT)

    def preview_on_image(self, image: ImageArray) -> ImageArray:
        """Draw both hands on the image."""
        image = self.left.preview_on_image(image)
        image = self.right.preview_on_image(image)
        return image

    def reset(self) -> None:
        """Reset both hands and clear all cached properties."""
        self.left.reset()
        self.right.reset()


class Hand:
    _cached_props: ClassVar[tuple[str, ...]] = ("is_facing_camera", "main_direction", "all_fingers_touching")

    def __init__(self, handedness: Handedness) -> None:
        self.handedness = handedness

        self.is_visible: bool = False
        self.palm: Palm = Palm(hand=self)
        self.fingers: list[Finger] = [Finger(index=FingerIndex(finger_idx), hand=self) for finger_idx in FingerIndex]
        self.wrist_landmark: NormalizedLandmark | None = None
        self.gesture: Gestures | None = None
        self.is_default_gesture: bool = False
        self._finger_touch_cache: dict[tuple[FingerIndex, FingerIndex], bool] = {}

    def reset(self) -> None:
        """Reset the hand state and clear all cached properties."""
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
        wrist_landmark: NormalizedLandmark | None,
        gesture: Gestures | None,
        is_default_gesture: bool,
    ) -> None:
        """Update the hand with new data."""
        self.is_visible = is_visible
        self.wrist_landmark = wrist_landmark
        self.gesture = gesture
        self.is_default_gesture = is_default_gesture

    def __bool__(self) -> bool:
        """Check if the hand is visible and has a valid handedness."""
        return self.is_visible

    @cached_property
    def is_facing_camera(self) -> bool:
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
            [thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y, thumb_mcp.z - wrist.z if hasattr(thumb_mcp, "z") else 0]
        )

        vec2 = np.array(
            [pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z if hasattr(pinky_mcp, "z") else 0]
        )

        # Calculate cross product to get normal vector
        normal = np.cross(vec1, vec2)

        # The z-component of the normal indicates orientation
        # For right hand: negative z = palm facing camera
        # For left hand: positive z = palm facing camera
        return cast(bool, normal[2] < 0 if self.handedness == Handedness.RIGHT else normal[2] > 0)

    @cached_property
    def main_direction(self) -> tuple[float, float] | None:
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

        middle_centroid_x, middle_centroid_y = middle_finger.centroid

        # Calculate direction vector from wrist to middle finger centroid
        dx = middle_centroid_x - self.wrist_landmark.x
        dy = middle_centroid_y - self.wrist_landmark.y

        # Normalize the vector
        magnitude = sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    def are_fingers_touching(self, finger1: FingerIndex | Finger, finger2: FingerIndex | Finger) -> bool:
        """Check if two fingers are touching, computing and caching the result if needed."""

        if isinstance(finger1, Finger):
            finger1 = finger1.index
        if isinstance(finger2, Finger):
            finger2 = finger2.index

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

    @cached_property
    def all_fingers_touching(self) -> bool:
        """Check if all adjacent fingers are touching each other."""
        for finger_pair in ADJACENT_FINGER_MAX_ANGLES:
            finger1, finger2 = finger_pair
            if not self.are_fingers_touching(finger1, finger2):
                return False
        return True

    def preview_on_image(self, image: ImageArray) -> ImageArray:
        """Draw the hand on the image."""
        if not self:
            return image

        # Draw wrist landmark
        if self.wrist_landmark:
            height, width = image.shape[:2]
            wrist_x = int(self.wrist_landmark.x * width)
            wrist_y = int(self.wrist_landmark.y * height)
            cv2.circle(image, (wrist_x, wrist_y), 5, (255, 255, 255), -1)

        # Draw palm
        if self.palm:
            image = self.palm.preview_on_image(image, self.is_facing_camera)

        # Draw fingers
        for finger in self.fingers:
            image = finger.preview_on_image(image)

        # Draw main direction arrow
        if self.main_direction and self.wrist_landmark:
            height, width = image.shape[:2]

            # Start point at wrist
            start_x = int(self.wrist_landmark.x * width)
            start_y = int(self.wrist_landmark.y * height)

            # The main_direction is in normalized space, so we need to convert it to pixel space
            # accounting for the aspect ratio
            dx_norm, dy_norm = self.main_direction

            # Convert normalized direction to pixel direction
            dx_pixel = dx_norm * width
            dy_pixel = dy_norm * height

            # Re-normalize in pixel space
            magnitude_pixel = np.sqrt(dx_pixel**2 + dy_pixel**2)
            if magnitude_pixel > 0:
                dx_pixel_norm = dx_pixel / magnitude_pixel
                dy_pixel_norm = dy_pixel / magnitude_pixel

                # Calculate end point
                arrow_length = 70  # pixels
                end_x = int(start_x + dx_pixel_norm * arrow_length)
                end_y = int(start_y + dy_pixel_norm * arrow_length)

                # Draw arrow (cyan color)
                cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (255, 255, 0), 3, tipLength=0.3)

        return image

    def detect_gesture(self) -> Gestures | None:
        """Detect custom gestures.

        Args:
            hand: The hand to analyze

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

            elif gesture == Gestures.PINCH:
                # Thumb is straight, fingers except index are not straight. Camera facing.
                if (
                    self.is_facing_camera
                    and thumb.is_straight
                    and not middle.is_straight
                    and not ring.is_straight
                    and not pinky.is_straight
                ):
                    return gesture

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


class Palm:
    _cached_props: ClassVar[tuple[str, ...]] = ("centroid",)

    def __init__(self, hand: Hand) -> None:
        self.hand = hand
        self.landmarks: list[NormalizedLandmark] = []

    def reset(self) -> None:
        """Reset the palm and clear all cached properties."""
        # Clear cached properties
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update(self, landmarks: list[NormalizedLandmark]) -> None:
        """Update the palm with new landmarks."""
        self.landmarks = landmarks

    @cached_property
    def centroid(self) -> tuple[float, float]:
        """Calculate the centroid of palm landmarks."""
        if not self.landmarks:
            return 0.0, 0.0

        centroid_x = sum(landmark.x for landmark in self.landmarks) / len(self.landmarks)
        centroid_y = sum(landmark.y for landmark in self.landmarks) / len(self.landmarks)

        return centroid_x, centroid_y

    def preview_on_image(self, image: ImageArray, is_facing_camera: bool = False) -> ImageArray:
        """Draw the palm center on the image."""
        height, width = image.shape[:2]

        palm_x, palm_y = self.centroid

        # Convert to pixel coordinates
        palm_center_x = int(palm_x * width)
        palm_center_y = int(palm_y * height)

        # Determine palm color based on facing
        palm_color = (0, 255, 0) if is_facing_camera else (255, 0, 0)

        # Draw palm center with color indicating facing
        cv2.circle(image, (palm_center_x, palm_center_y), 5, palm_color, -1)

        return image


class Finger:
    _cached_props: ClassVar[tuple[str, ...]] = (
        "centroid",
        "start_point",
        "end_point",
        "straightness_score",
        "is_straight",
        "is_nearly_straight",
        "straight_direction",
        "is_fully_bent",
        "fold_angle",
        "tip_direction",
        "touches_thumb",
        "touching_fingers",
    )

    def __init__(self, index: FingerIndex, hand: Hand):
        self.index = index
        self.hand = hand
        self.landmarks: list[NormalizedLandmark] = []

    def reset(self) -> None:
        """Reset the finger and clear all cached properties."""
        # Clear cached properties
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update(self, landmarks: list[NormalizedLandmark]) -> None:
        """Update the finger with new landmarks."""
        self.landmarks = landmarks

    def __bool__(self) -> bool:
        """Check if the finger is visible and has landmarks."""
        return len(self.landmarks) > 0

    @cached_property
    def centroid(self) -> tuple[float, float]:
        """Calculate the centroid of all finger landmarks."""
        if not self.landmarks:
            return 0.0, 0.0

        centroid_x = sum(landmark.x for landmark in self.landmarks) / len(self.landmarks)
        centroid_y = sum(landmark.y for landmark in self.landmarks) / len(self.landmarks)

        return centroid_x, centroid_y

    @cached_property
    def straightness_score(self) -> float:
        """Calculate a straightness score from 0 to 1, where 1 is perfectly straight and 0 is not straight."""
        # Get configuration for this finger
        config = FINGER_STRAIGHTNESS_CONFIG[self.index]

        # Configuration constants from per-finger config
        distal_segments_max_ratio = config.get("distal_segments_max_ratio", 1.5)
        distal_segments_max_ratio_back = config.get("distal_segments_max_ratio_back", 1.5)
        max_angle_degrees = config.get("max_angle_degrees", 15)

        # Segment ratio scoring parameters
        segment_ratio_score_at_threshold = config.get("segment_ratio_score_at_threshold", 0.7)
        segment_ratio_decay_rate = config.get("segment_ratio_decay_rate", 2.0)
        segment_ratio_linear_range = config.get("segment_ratio_linear_range", 0.15)

        # Angle scoring parameters
        angle_score_linear_range = config.get("angle_score_linear_range", 0.15)
        angle_score_at_threshold = config.get("angle_score_at_threshold", 0.85)
        angle_decay_rate = config.get("angle_decay_rate", 20.0)

        # Score combination weights
        angle_score_weight = config.get("angle_score_weight", 0.7)
        segment_ratio_weight = config.get("segment_ratio_weight", 0.3)

        # Numerical thresholds
        min_magnitude_threshold = 0.001  # Minimum vector magnitude to avoid division by zero

        if not self.hand.is_facing_camera:
            distal_segments_max_ratio = distal_segments_max_ratio_back

        if len(self.landmarks) < 3:  # Need at least 3 points to check alignment
            return 0.0

        # Skip thumb - will be handled separately later
        if self.index == FingerIndex.THUMB:
            return self._straightness_score_basic()

        # Extract x and y coordinates for all finger landmarks
        x_coords = [landmark.x for landmark in self.landmarks]
        y_coords = [landmark.y for landmark in self.landmarks]

        # Calculate segment lengths
        segment_lengths = []
        for i in range(len(self.landmarks) - 1):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
            segment_length = np.sqrt(dx**2 + dy**2)
            segment_lengths.append(segment_length)

        # For fingers (not thumb), expect 3 segments: MCP-PIP, PIP-DIP, DIP-TIP
        # The MCP-PIP segment is naturally longer, so check proportional consistency
        segment_ratio_score = 1.0
        if len(segment_lengths) == 3:
            mcp_pip, pip_dip, dip_tip = segment_lengths  # proximal, middle, distal segments

            # Check if the distal segments (PIP-DIP and DIP-TIP) are reasonably similar
            if pip_dip > 0 and dip_tip > 0:
                ratio = max(pip_dip, dip_tip) / min(pip_dip, dip_tip)
                if ratio > distal_segments_max_ratio:
                    # Convert ratio to score: at threshold = segment_ratio_score_at_threshold, higher ratios decay
                    excess_ratio = ratio - distal_segments_max_ratio
                    # Use exponential decay for excessive ratios
                    segment_ratio_score = segment_ratio_score_at_threshold * np.exp(
                        -excess_ratio * segment_ratio_decay_rate
                    )
                else:
                    # Linear interpolation from 1.0 (ratio=1) to (1.0 - segment_ratio_linear_range) at threshold
                    segment_ratio_score = (
                        1.0 - ((ratio - 1.0) / (distal_segments_max_ratio - 1.0)) * segment_ratio_linear_range
                    )

        # Verify segment angles are relatively straight
        # Calculate angles between consecutive segments
        max_found_angle = 0.0
        if len(segment_lengths) >= 2:
            for i in range(len(segment_lengths) - 1):
                # Get vectors for consecutive segments
                p1 = (x_coords[i], y_coords[i])
                p2 = (x_coords[i + 1], y_coords[i + 1])
                p3 = (x_coords[i + 2], y_coords[i + 2])

                # Vector from p1 to p2
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                # Vector from p2 to p3
                v2 = (p3[0] - p2[0], p3[1] - p2[1])

                # Calculate angle between vectors using dot product
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)

                if mag1 > min_magnitude_threshold and mag2 > min_magnitude_threshold:  # Avoid division by zero
                    cos_angle = dot_product / (mag1 * mag2)
                    # Clamp to [-1, 1] to handle numerical errors
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = np.arccos(cos_angle)

                    # Convert to degrees for easier interpretation
                    angle_degrees = np.degrees(angle)
                    max_found_angle = max(max_found_angle, angle_degrees)

        # Convert angle to score: 0° = 1.0, max_angle_degrees = angle_score_at_threshold, higher angles decay to 0
        if max_found_angle <= max_angle_degrees:
            # Perfect to acceptable range: linear from 1.0 to angle_score_at_threshold
            angle_score = 1.0 - (max_found_angle / max_angle_degrees) * angle_score_linear_range
        else:
            # Beyond threshold: exponential decay from angle_score_at_threshold to 0
            excess_angle = max_found_angle - max_angle_degrees
            # Use exponential decay: score decreases rapidly after threshold
            angle_score = angle_score_at_threshold * np.exp(-excess_angle / angle_decay_rate)

        # Combine both scores (weighted average: angle is more important)
        combined_score = angle_score_weight * angle_score + segment_ratio_weight * segment_ratio_score

        return float(combined_score)

    def _straightness_score_basic(self) -> float:
        """Basic straightness score calculation for thumb or fallback."""
        # Get configuration for this finger
        config = FINGER_STRAIGHTNESS_CONFIG[self.index]

        # Configuration constants from per-finger config
        alignment_threshold = config.get("alignment_threshold", 0.01)
        max_deviation_for_zero_score = config.get("max_deviation_for_zero_score", 0.1)
        min_denominator_threshold = 0.001  # Minimum line length to avoid division by zero

        if len(self.landmarks) < 3:
            return 0.0

        x_coords = [landmark.x for landmark in self.landmarks]
        y_coords = [landmark.y for landmark in self.landmarks]

        # Calculate the line from first to last point
        x1, y1 = x_coords[0], y_coords[0]
        x2, y2 = x_coords[-1], y_coords[-1]

        # Check intermediate points alignment
        max_deviation = 0.0
        for i in range(1, len(self.landmarks) - 1):
            x, y = x_coords[i], y_coords[i]

            # Calculate perpendicular distance from point to line
            # Using formula: |ax + by + c| / sqrt(a^2 + b^2)
            # where line is: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if denominator > min_denominator_threshold:  # Avoid division by zero
                distance = numerator / denominator
                max_deviation = max(max_deviation, distance)

        # Convert deviation to score
        if max_deviation <= alignment_threshold:
            score = 1.0
        else:
            # Linear decay from 1.0 to 0 as deviation increases
            score = max(0.0, 1.0 - (max_deviation / max_deviation_for_zero_score))

        return float(score)

    @cached_property
    def is_straight(self) -> bool:
        """Check if the finger is straight based on the straightness score."""
        # Handle gesture-based shortcuts first
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return False
        elif self.hand.gesture == Gestures.OPEN_PALM:
            if self.index != FingerIndex.THUMB:
                return True
        elif self.hand.gesture == Gestures.POINTING_UP:
            return self.index == FingerIndex.INDEX
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            return self.index == FingerIndex.THUMB
        elif self.hand.gesture == Gestures.VICTORY:
            return self.index in (FingerIndex.INDEX, FingerIndex.MIDDLE)

        # Use the straightness score with strict threshold from finger config
        config = FINGER_STRAIGHTNESS_CONFIG[self.index]
        return self.straightness_score >= config.get("straight_threshold", FINGER_STRAIGHT_THRESHOLD)

    @cached_property
    def is_nearly_straight(self) -> bool:
        """Check if the finger is nearly straight based on the straightness score."""
        # Handle gesture-based shortcuts first (same as is_straight for consistency)
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return False
        elif self.hand.gesture == Gestures.OPEN_PALM:
            if self.index != FingerIndex.THUMB:
                return False
        elif self.hand.gesture == Gestures.POINTING_UP:
            if self.index == FingerIndex.INDEX:
                return False
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            if self.index == FingerIndex.THUMB:
                return False
        elif self.hand.gesture == Gestures.VICTORY:
            if self.index in (FingerIndex.INDEX, FingerIndex.MIDDLE):
                return False

        # Use the straightness score with relaxed threshold from finger config
        config = FINGER_STRAIGHTNESS_CONFIG[self.index]
        nearly_threshold = config.get("nearly_straight_threshold", FINGER_NEARLY_STRAIGHT_THRESHOLD)
        straight_threshold = config.get("straight_threshold", FINGER_STRAIGHT_THRESHOLD)
        return nearly_threshold <= self.straightness_score < straight_threshold

    @property
    def is_nearly_straight_or_straight(self) -> bool:
        """Check if the finger is either nearly straight or straight."""
        return self.is_nearly_straight or self.is_straight

    @property
    def is_not_straight_at_all(self) -> bool:
        return not (self.is_straight or self.is_nearly_straight)

    def _is_straight_basic(self) -> bool:
        """Basic straightness check for thumb or fallback."""
        alignment_threshold = 0.01

        if len(self.landmarks) < 3:
            return False

        x_coords = [landmark.x for landmark in self.landmarks]
        y_coords = [landmark.y for landmark in self.landmarks]

        x1, y1 = x_coords[0], y_coords[0]
        x2, y2 = x_coords[-1], y_coords[-1]

        finger_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if finger_length < 0.05:
            return False

        # Check alignment only
        distances = []
        for i in range(1, len(self.landmarks) - 1):
            x, y = x_coords[i], y_coords[i]
            if x2 != x1 or y2 != y1:
                dist = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt(
                    (y2 - y1) ** 2 + (x2 - x1) ** 2
                )
                distances.append(dist)

        max_distance = max(distances) if distances else 0
        return max_distance < alignment_threshold

    @cached_property
    def start_point(self) -> tuple[float, float] | None:
        """Get the start point of the finger (base)."""
        if not self.landmarks:
            return None
        return self.landmarks[0].x, self.landmarks[0].y

    @cached_property
    def end_point(self) -> tuple[float, float] | None:
        """Get the end point of the finger (tip)."""
        if not self.landmarks:
            return None
        return self.landmarks[-1].x, self.landmarks[-1].y

    @cached_property
    def fold_angle(self) -> float | None:
        """Calculate the fold angle at the PIP joint (angle between MCP->PIP and PIP->TIP vectors).
        Returns angle in degrees. 180 = straight, lower angles = more bent."""
        if len(self.landmarks) < 3:
            return None

        # For thumb, use CMC->IP->TIP instead of MCP->PIP->TIP
        if self.index == FingerIndex.THUMB:
            if len(self.landmarks) <= 3:
                return None
            mcp, pip, tip = self.landmarks[0], self.landmarks[1], self.landmarks[3]
        else:
            # For other fingers: MCP->PIP->TIP
            mcp, pip, tip = self.landmarks[0], self.landmarks[1], self.landmarks[3]

        # Vector PIP->MCP
        v1 = np.array([mcp.x - pip.x, mcp.y - pip.y])
        # Vector PIP->DIP
        v2 = np.array([tip.x - pip.x, tip.y - pip.y])

        # Calculate magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 < 0.001 or mag2 < 0.001:
            return None

        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (mag1 * mag2)

        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    @cached_property
    def tip_direction(self) -> tuple[float, float] | None:
        """Calculate the direction of the finger tip using the last two points.
        Returns a normalized vector (dx, dy) pointing from second-to-last to last point.
        Returns None if finger is fully bent."""
        if len(self.landmarks) < 2:
            return None

        # Don't calculate direction if finger is fully bent
        if self.is_fully_bent:
            return None

        # Get the last two points
        second_last = self.landmarks[-2]
        last = self.landmarks[-1]

        # Calculate direction vector
        dx = last.x - second_last.x
        dy = last.y - second_last.y

        # Normalize the vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    @cached_property
    def straight_direction(self) -> tuple[float, float] | None:
        """Calculate the overall direction of the finger from base to tip.
        Returns a normalized vector (dx, dy) pointing from first to last point.
        Returns None if finger is not straight."""
        if not self.is_straight:
            return None

        if len(self.landmarks) < 2:
            return None

        # Get the first and last points
        first = self.landmarks[0]
        last = self.landmarks[-1]

        # Calculate direction vector
        dx = last.x - first.x
        dy = last.y - first.y

        # Normalize the vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 0.001:  # Avoid division by zero
            return None

        return dx / magnitude, dy / magnitude

    @cached_property
    def is_fully_bent(self) -> bool:
        """Check if the finger is fully bent based on fold angle threshold."""
        if self.hand.gesture == Gestures.CLOSED_FIST:
            return True
        elif self.hand.gesture == Gestures.OPEN_PALM:
            return False
        elif self.hand.gesture == Gestures.POINTING_UP:
            return self.index != FingerIndex.INDEX
        elif self.hand.gesture in (Gestures.THUMB_UP, Gestures.THUMB_DOWN):
            return self.index != FingerIndex.THUMB
        elif self.hand.gesture == Gestures.VICTORY:
            return self.index not in (FingerIndex.INDEX, FingerIndex.MIDDLE)

        if self.fold_angle is None:
            return True

        return self.fold_angle < (150 if self.index == FingerIndex.THUMB else 60.0)

    @cached_property
    def touches_thumb(self) -> bool:
        """Check if this finger tip touches the thumb tip of the same hand."""
        # Skip if this finger is the thumb itself
        if self.index == FingerIndex.THUMB:
            return False

        if self.hand.gesture is not None:
            return False

        if self.is_fully_bent:
            return False

        # Need a reference to the hand to access the thumb
        if not self.hand or not self.hand.fingers:
            return False

        # Get the thumb finger
        thumb = None
        for finger in self.hand.fingers:
            if finger.index == FingerIndex.THUMB:
                thumb = finger
                break

        if not thumb or not thumb.landmarks or not self.landmarks:
            return False

        # Get the tip landmarks (last landmark of each finger)
        thumb_tip = thumb.landmarks[-1]
        finger_tip = self.landmarks[-1]

        # Calculate 3D spatial distance between tips
        dx = finger_tip.x - thumb_tip.x
        dy = finger_tip.y - thumb_tip.y
        dz = (finger_tip.z - thumb_tip.z) if hasattr(finger_tip, "z") and hasattr(thumb_tip, "z") else 0

        distance = sqrt(dx**2 + dy**2 + dz**2)

        # Touch threshold (in normalized coordinates)
        touch_threshold = 0.05  # Adjust based on testing

        return distance < touch_threshold

    @cached_property
    def touching_fingers(self) -> list[FingerIndex]:
        """Get list of adjacent fingers that this finger is touching.
        Returns empty list for thumb, up to 1 for index/pinky, up to 2 for middle/ring."""
        # Thumb never touches other fingers in this context
        if self.index == FingerIndex.THUMB:
            return []

        # Need reference to hand
        if not self.hand:
            return []

        touching = []

        # Define adjacent fingers based on finger index
        if self.index == FingerIndex.INDEX:
            # Index can only touch middle
            if self.hand.are_fingers_touching(self.index, FingerIndex.MIDDLE):
                touching.append(FingerIndex.MIDDLE)

        elif self.index == FingerIndex.MIDDLE:
            # Middle can touch index and ring
            if self.hand.are_fingers_touching(self.index, FingerIndex.INDEX):
                touching.append(FingerIndex.INDEX)
            if self.hand.are_fingers_touching(self.index, FingerIndex.RING):
                touching.append(FingerIndex.RING)

        elif self.index == FingerIndex.RING:
            # Ring can touch middle and pinky
            if self.hand.are_fingers_touching(self.index, FingerIndex.MIDDLE):
                touching.append(FingerIndex.MIDDLE)
            if self.hand.are_fingers_touching(self.index, FingerIndex.PINKY):
                touching.append(FingerIndex.PINKY)

        elif self.index == FingerIndex.PINKY:
            # Pinky can only touch ring
            if self.hand.are_fingers_touching(self.index, FingerIndex.RING):
                touching.append(FingerIndex.RING)

        return touching

    def is_touching(self, other: FingerIndex | Finger) -> bool:
        """Check if this finger is touching another finger."""
        return self.hand.are_fingers_touching(self.index, other.index if isinstance(other, Finger) else other)

    def preview_on_image(self, image: ImageArray) -> ImageArray:
        """Draw the finger on the image."""
        height, width = image.shape[:2]

        # Draw all landmarks
        for landmark in self.landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 3, FINGER_COLORS[self.index], -1)

        # Draw colored line for straight or nearly straight finger
        if (self.is_straight or self.is_nearly_straight) and self.start_point and self.end_point:
            start_x = int(self.start_point[0] * width)
            start_y = int(self.start_point[1] * height)
            end_x = int(self.end_point[0] * width)
            end_y = int(self.end_point[1] * height)

            color = FINGER_COLORS[self.index]

            if self.is_straight:
                # Solid line for straight fingers
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 3)
            else:
                # Dashed line for nearly straight fingers
                # Calculate total length and create dashed pattern
                dx = end_x - start_x
                dy = end_y - start_y
                length = np.sqrt(dx**2 + dy**2)

                if length > 0:
                    # Normalize direction
                    dx_norm = dx / length
                    dy_norm = dy / length

                    # Draw dashed line with 10 pixel segments and 5 pixel gaps
                    dash_length = 10
                    gap_length = 5
                    total_pattern = dash_length + gap_length

                    current_pos = 0
                    while current_pos < length:
                        # Calculate dash start and end
                        dash_start = current_pos
                        dash_end = min(current_pos + dash_length, length)

                        # Convert to pixel coordinates
                        x1 = int(start_x + dx_norm * dash_start)
                        y1 = int(start_y + dy_norm * dash_start)
                        x2 = int(start_x + dx_norm * dash_end)
                        y2 = int(start_y + dy_norm * dash_end)

                        # Draw dash segment
                        cv2.line(image, (x1, y1), (x2, y2), color, 3)

                        # Move to next segment
                        current_pos += total_pattern

        # Draw finger direction arrow
        if self.tip_direction and self.end_point:
            # Get fingertip position
            tip_x = int(self.end_point[0] * width)
            tip_y = int(self.end_point[1] * height)

            # Convert normalized direction to pixel space
            dx_norm, dy_norm = self.tip_direction
            dx_pixel = dx_norm * width
            dy_pixel = dy_norm * height

            # Re-normalize in pixel space
            magnitude_pixel = np.sqrt(dx_pixel**2 + dy_pixel**2)
            if magnitude_pixel > 0:
                dx_pixel_norm = dx_pixel / magnitude_pixel
                dy_pixel_norm = dy_pixel / magnitude_pixel

                # Calculate arrow end point
                arrow_length = 30  # Small arrow
                arrow_end_x = int(tip_x + dx_pixel_norm * arrow_length)
                arrow_end_y = int(tip_y + dy_pixel_norm * arrow_length)

                # Draw small arrow in white
                cv2.arrowedLine(image, (tip_x, tip_y), (arrow_end_x, arrow_end_y), (255, 255, 255), 2, tipLength=0.4)

        # Draw red circle if this finger touches the thumb
        if self.touches_thumb and self.end_point and self.hand:
            # Get thumb finger
            thumb = None
            for finger in self.hand.fingers:
                if finger.index == FingerIndex.THUMB:
                    thumb = finger
                    break

            if thumb and thumb.end_point:
                # Calculate middle point between both tips
                finger_tip_x = self.end_point[0]
                finger_tip_y = self.end_point[1]
                thumb_tip_x = thumb.end_point[0]
                thumb_tip_y = thumb.end_point[1]

                middle_x = (finger_tip_x + thumb_tip_x) / 2
                middle_y = (finger_tip_y + thumb_tip_y) / 2

                # Convert to pixel coordinates
                pixel_x = int(middle_x * width)
                pixel_y = int(middle_y * height)

                # Draw small red filled circle
                cv2.circle(image, (pixel_x, pixel_y), 8, (0, 0, 255), -1)

        # Draw indicators for touching adjacent fingers
        if self.touching_fingers and self.end_point and self.hand:
            for touching_index in self.touching_fingers:
                # Find the touching finger
                touching_finger = None
                for finger in self.hand.fingers:
                    if finger.index == touching_index:
                        touching_finger = finger
                        break

                if touching_finger and touching_finger.end_point:
                    # Draw a line between the fingertips
                    my_tip_x = int(self.end_point[0] * width)
                    my_tip_y = int(self.end_point[1] * height)
                    other_tip_x = int(touching_finger.end_point[0] * width)
                    other_tip_y = int(touching_finger.end_point[1] * height)

                    # Draw a thick cyan line between touching fingers
                    cv2.line(image, (my_tip_x, my_tip_y), (other_tip_x, other_tip_y), (255, 255, 0), 4)

                    # Draw small circles at connection points
                    cv2.circle(image, (my_tip_x, my_tip_y), 5, (255, 255, 0), -1)
                    cv2.circle(image, (other_tip_x, other_tip_y), 5, (255, 255, 0), -1)

        return image


def create_gesture_recognizer(
    callback_function: Callable[[vision.GestureRecognizerResult, mp.Image, int], None],
) -> vision.GestureRecognizer:
    """Create and return a MediaPipe gesture recognizer configured for live stream mode."""
    import os
    import urllib.request

    model_path = "gesture_recognizer.task"
    model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Downloading...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Successfully downloaded model to '{model_path}'")
        except Exception as download_error:
            print(f"Failed to download model: {download_error}")
            raise RuntimeError(f"Could not download model from {model_url}: {download_error}") from download_error

    base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=callback_function,
    )
    return vision.GestureRecognizer.create_from_options(options)


def on_gesture_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """Callback function for gesture recognition results."""
    global latest_gesture_result
    latest_gesture_result = result


def update_hands_from_results(hands: Hands, result: vision.GestureRecognizerResult | None) -> None:
    """Update the hands object with new gesture recognition results."""

    # Reset all hands first
    hands.reset()

    if not result or not result.hand_landmarks:
        return

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        # Get handedness
        handedness = None
        if result.handedness and i < len(result.handedness) and result.handedness[i]:
            handedness = Handedness.from_data(result.handedness[i][0].category_name)

        # Skip if handedness not detected
        if not handedness:
            continue

        # Get the appropriate hand
        hand = hands.left if handedness == Handedness.LEFT else hands.right

        # Get default gesture information
        gesture_type = None
        if result.gestures and i < len(result.gestures) and result.gestures[i]:
            gesture = result.gestures[i][0]  # Get the top gesture
            gesture_type = (
                None if gesture.category_name in (None, "None", "Unknown") else Gestures(gesture.category_name)
            )

        # Update hand data
        hand.update(
            is_visible=True,
            wrist_landmark=hand_landmarks[HandLandmark.WRIST],
            gesture=gesture_type,
            is_default_gesture=gesture_type is not None and gesture_type in DEFAULT_GESTURES,
        )

        # Update palm
        hand.palm.update([hand_landmarks[idx] for idx in PALM_LANDMARKS])

        # Update fingers
        for finger_idx, finger in enumerate(hand.fingers):
            finger_landmarks = [hand_landmarks[idx] for idx in FINGERS_LANDMARKS[finger_idx]]
            finger.update(landmarks=finger_landmarks)

        # Detect custom gestures if no default gesture was detected
        if gesture_type is None or gesture_type in OVERRIDABLE_DEFAULT_GESTURES:
            custom_gesture = hand.detect_gesture()
            if custom_gesture:
                hand.gesture = custom_gesture
                hand.is_default_gesture = False


class CameraInfo(NamedTuple):
    device_index: int
    name: str
    height: int
    width: int
    format: PixelFormat

    def __str__(self) -> str:
        return f"[{self.device_index}] {self.name} - {self.width}x{self.height} @ {self.format.name}"


def list_cameras() -> list[CameraInfo]:
    """List all available cameras and return a list of CameraInfo objects."""
    cameras = []
    video_devices = sorted(glob.glob("/dev/video*"))

    for device_path in video_devices:
        with suppress(Exception):  # Skip devices that can't be accessed
            device_index = int(re.search(r"/dev/video(\d+)", device_path).group(1))  # type: ignore[union-attr]  # handled by suppress
            device = Device(device_path)
            device.open()

            # Only list devices that support video capture
            formats = [f for f in device.info.formats if f.type == BufferType.VIDEO_CAPTURE]
            if not formats:
                device.close()
                continue

            # Get current format
            current_format = device.get_format(BufferType.VIDEO_CAPTURE)

            # Skip GREY cameras
            if current_format.pixel_format == PixelFormat.GREY:
                device.close()
                continue

            camera_info = CameraInfo(
                device_index=device_index,
                name=device.info.card,
                height=current_format.height,
                width=current_format.width,
                format=current_format.pixel_format,
            )
            cameras.append(camera_info)
            device.close()

    return cameras


def pick_camera(filter_name: str | None = None) -> CameraInfo | None:
    """List cameras and let user pick one. Returns selected CameraInfo or None.

    Args:
        filter_name: Optional string to filter cameras by name (case insensitive)
    """
    cameras = list_cameras()

    if not cameras:
        print("No cameras found!")
        return None

    # Filter cameras if a filter name is provided
    if filter_name:
        filter_lower = filter_name.lower()
        filtered_cameras = [cam for cam in cameras if filter_lower in cam.name.lower()]

        if not filtered_cameras:
            print(f"No cameras found matching '{filter_name}'")
            return None

        if len(filtered_cameras) == 1:
            # Auto-select if only one match
            selected = filtered_cameras[0]
            print(f"Auto-selected camera: {selected}")
            return selected

        # Multiple matches, show filtered list
        cameras = filtered_cameras
        print(f"Cameras matching '{filter_name}':")
    else:
        # If no filter and only one camera available, auto-select it
        if len(cameras) == 1:
            selected = cameras[0]
            print(f"Auto-selected camera: {selected}")
            return selected

        print("Available cameras:")

    cam_dict = {}
    for cam in cameras:
        print(f"  {cam}")
        cam_dict[cam.device_index] = cam

    valid_indices = sorted(cam_dict.keys())

    while True:
        try:
            choice = input(f"\nSelect camera ({', '.join(map(str, valid_indices))} or q to quit): ")
            if choice.lower() == "q":
                return None
            idx = int(choice)
            if idx in cam_dict:
                return cam_dict[idx]
            else:
                print(f"Invalid choice. Please enter one of: {', '.join(map(str, valid_indices))}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


def preview_hands_info(hands: Hands, fps: float, frame: ImageArray) -> ImageArray:
    """Draw hands preview on frame and return the modified frame."""
    frame = hands.preview_on_image(frame)

    if MIRROR_OUTPUT:
        frame = cv2.flip(frame, 1)

    # Draw FPS in top-left corner
    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # First, prepare all text to determine footer height
    texts = []
    line_height = 40
    padding = 15

    # Count visible hands to calculate positions from bottom
    visible_hands = []
    for hand in [hands.left, hands.right]:
        if hand:
            visible_hands.append(hand)

    # Calculate text positions from bottom up
    frame_height = frame.shape[0]
    for i, hand in enumerate(visible_hands):
        handedness_str = hand.handedness.name if hand.handedness else "Unknown"
        facing_str = "PALM" if hand.is_facing_camera else "BACK"

        text = f"{handedness_str} hand showing {facing_str}"

        # Add gesture information if available
        if hand.gesture:
            text += f" - Gesture: {hand.gesture} ({'detected' if hand.is_default_gesture else 'custom'})"

        # Position from bottom: padding + line_height * (total_hands - current_index)
        y_pos = frame_height - padding - (line_height * (len(visible_hands) - i - 1))
        texts.append((text, y_pos))

    # Add semi-transparent black footer based on actual text height
    if texts:
        footer_height = line_height * len(visible_hands) + padding
        footer_y_start = frame_height - footer_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, footer_y_start), (frame.shape[1], frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw text with anti-aliasing
        for text, y_pos in texts:
            position = (10, y_pos)
            # Draw with LINE_AA for anti-aliasing
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def print_hands_info(hands: Hands, fps: float) -> None:
    """Print important features of hands/fingers to console."""
    # Print FPS first if available
    if fps > 0:
        print(f"\rFPS: {fps:.1f}", end="")

    if not hands.left and not hands.right:
        if fps <= 0:  # Only print if FPS wasn't already printed
            print("\rNo hands detected", end="")
        return

    for hand in [hands.left, hands.right]:
        if not hand:
            continue

        handedness = hand.handedness.name if hand.handedness else "Unknown"
        facing = "PALM" if hand.is_facing_camera else "BACK"
        gesture = hand.gesture if hand.gesture else "None"

        print(f"\n{handedness} Hand - {facing} - Gesture: {gesture}")

        if hand.main_direction:
            direction = f"({hand.main_direction[0]:.2f}, {hand.main_direction[1]:.2f})"
            print(f"  Main direction: {direction}")

        if hand.all_fingers_touching:
            print("  All adjacent fingers touching!")

        for finger in hand.fingers:
            finger_name = finger.index.name
            status = []

            if finger.is_straight:
                status.append("straight")
                if finger.straight_direction:
                    dir_str = f"({finger.straight_direction[0]:.2f}, {finger.straight_direction[1]:.2f})"
                    status.append(f"dir:{dir_str}")

            if finger.is_fully_bent:
                status.append("bent")

            if finger.touches_thumb:
                status.append("touches_thumb")

            if finger.touching_fingers:
                touching_names = [f.name for f in finger.touching_fingers]
                status.append(f"touching:{','.join(touching_names)}")

            if finger.fold_angle is not None:
                status.append(f"angle:{finger.fold_angle:.0f}°")

            if finger.tip_direction:
                dx, dy = finger.tip_direction
                tip_angle = np.degrees(np.arctan2(dy, dx))
                status.append(f"tip_angle:{tip_angle:.0f}°")

            status_str = ", ".join(status) if status else "neutral"
            print(f"    {finger_name}: {status_str}")


def init_camera_capture(
    camera_info: CameraInfo, show_preview: bool = True
) -> tuple[cv2.VideoCapture | None, str | None]:
    """Initialize camera capture and set resolution."""
    cap = cv2.VideoCapture(camera_info.device_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_info.device_index}")
        return None, None

    # Calculate dimensions based on DESIRED_SIZE while maintaining aspect ratio
    aspect_ratio = camera_info.width / camera_info.height
    if camera_info.width > camera_info.height:
        width = DESIRED_SIZE
        height = int(DESIRED_SIZE / aspect_ratio)
    else:
        height = DESIRED_SIZE
        width = int(DESIRED_SIZE * aspect_ratio)

    # Set resolution based on calculated dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))  # Use MJPEG for better performance
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Camera {camera_info.name} opened successfully at {width}x{height} with FPS: {cap_fps:.2f}")

    window_name = None
    if show_preview:
        window_name = f"Camera Preview - {camera_info.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"Showing preview for {camera_info.name}")

    return cap, window_name


def run(camera_info: CameraInfo, show_preview: bool = True) -> None:
    """Show a live preview of the selected camera with gesture recognition."""
    global latest_gesture_result

    # Initialize global hands instance
    hands = Hands()

    cap, window_name = init_camera_capture(camera_info, show_preview)
    if cap is None:
        return

    print("Loading gesture recognizer model...")

    try:
        # Create gesture recognizer
        recognizer = create_gesture_recognizer(on_gesture_result)
        print("Gesture recognizer loaded successfully")
        if show_preview:
            print("Press 'q' or ESC to quit")
    except Exception as e:
        print(f"\nError loading gesture recognizer: {e}")
        print("\nAborting - gesture recognition is required for this application.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Initialize timestamp and FPS tracking
    start_time = time.time()
    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Process frame with gesture recognizer if available
        if recognizer:
            # Convert frame to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate timestamp in milliseconds
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Perform gesture recognition
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Update hands object with latest results
            update_hands_from_results(hands, latest_gesture_result)
            if show_preview:
                frame = preview_hands_info(hands, fps, frame)
            else:
                print_hands_info(hands, fps)

        if show_preview:
            cv2.imshow(cast(str, window_name), frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - fps_timer >= 1.0:  # Update FPS every second
            fps = frame_count / (current_time - fps_timer)
            frame_count = 0
            fps_timer = current_time

    cap.release()

    if show_preview:
        cv2.destroyAllWindows()

    if recognizer:
        recognizer.close()


def check_camera(camera_info: CameraInfo, show_preview: bool = True) -> None:
    """Check camera functionality without gesture recognition."""
    cap, window_name = init_camera_capture(camera_info, show_preview)
    if cap is None:
        return

    if not show_preview:
        print("Camera check completed successfully.")
        cap.release()
        return

    print("Press 'q' or ESC to quit")

    # Initialize FPS tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - fps_timer >= 1.0:  # Update FPS every second
            fps = frame_count / (current_time - fps_timer)
            frame_count = 0
            fps_timer = current_time

        # Add FPS text to frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(cast(str, window_name), frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


app = typer.Typer()


@app.command()
def run_gestures(
    filter_name: str = typer.Argument(None, help="Optional camera name filter (case insensitive)"),
    preview: bool = typer.Option(False, "--preview", help="Show visual preview window"),
    check: bool = typer.Option(False, "--check", help="Only check camera without gesture recognition"),
) -> None:
    """List and preview cameras with optional gesture recognition."""
    selected = pick_camera(filter_name)

    if selected:
        print(f"\nSelected: {selected}")
        if check:
            check_camera(selected, show_preview=preview)
        else:
            run(selected, show_preview=preview)
    else:
        print("\nNo camera selected.")


def main() -> None:
    """Entry point for the application."""
    app()


if __name__ == "__main__":
    main()
