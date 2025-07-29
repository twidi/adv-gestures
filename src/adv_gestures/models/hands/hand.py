from __future__ import annotations

from collections import deque
from functools import cached_property
from math import acos, atan2, degrees, inf, radians, sqrt
from time import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from ...config import Config
from ...gestures import DEFAULT_GESTURES, Gestures
from ...smoothing import (
    GESTURE_SMOOTHING_WINDOW,
    BoxSmoother,
    CoordSmoother,
    EnumSmoother,
    GestureWeights,
    ManyCoordsSmoother,
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
from .utils import Box, Handedness, SwipeDirection

if TYPE_CHECKING:
    from ...recognizer import StreamInfo
    from .hands import Hands


class Hand(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = (
        "is_facing_camera",
        "is_showing_side",
        "main_direction",
        "all_adjacent_fingers_touching",
        "all_adjacent_fingers_except_thumb_touching",
        "bounding_box",
        "oriented_bounding_box",
        "pinch_box",
        "main_direction_angle",
        "bounding_box_direction_line_points",
        "frame_direction_line_points",
        "other_hand_line_intersection",
        "other_hand_line_intersection_normalized",
        "other_hand_line_intersection_absolute",
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
        "gestures_data",
    )

    def __init__(self, handedness: Handedness, hands: Hands, config: Config) -> None:
        super().__init__()
        self.hands = hands
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

    @cached_property  # will never change after initialization
    def other_hand(self) -> Hand:
        """Get the other hand (left or right) based on handedness."""
        return self.hands.left if self.handedness == Handedness.RIGHT else self.hands.right

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
            for finger in self.fingers:
                finger.update(landmarks=[])
            self._raw_custom_gestures = {}
            self._direction_history.clear()
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
            not self.is_visible
            or not self.handedness
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
        if not self.stream_info or not self.is_visible or not self.palm:
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

    def _calc_main_direction_angle(self) -> float | None:
        """Calculate the angle of the main direction vector in degrees.
        Returns angle in range [-180, 180] where:
        - 0° = pointing right
        - 90° = pointing up
        - 180°/-180° = pointing left
        - -90° = pointing down
        """
        direction = self.main_direction
        if direction is None:
            return None

        dx, dy = direction
        # Calculate angle in radians and convert to degrees
        angle_rad = atan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
        angle_deg = degrees(angle_rad)
        return angle_deg

    main_direction_angle = smoothed_optional_float(_calc_main_direction_angle)

    def detect_direction_changes(
        self,
        duration_window: float = 1.0,
        min_direction_changes: int = 2,
        min_movement_angle: float = 2.5,
        x_tolerance: float = 0.05,
        max_time_since_last_change: float = 0.5,
        require_recent_movement: bool = True,
        recent_movement_window: float = 0.3,
    ) -> tuple[bool, list[tuple[SwipeDirection, float]]]:
        """Detect oscillating directional changes in hand movement.

        Args:
            duration_window: Time window in seconds to analyze movement
            min_direction_changes: Minimum number of direction changes required
            min_movement_angle: Minimum angle in degrees for significant movement
            x_tolerance: Tolerance zone around x=0 to avoid noise
            max_time_since_last_change: Maximum time since last direction change
            require_recent_movement: Whether to require recent movement
            recent_movement_window: Time window for recent movement check

        Returns:
            Tuple of (has_changes, list of (direction, time_ago) pairs)
        """
        if not self._direction_history:
            return False, []

        current_time = time()
        min_movement_angle_rad = radians(min_movement_angle)

        # Find directions within the duration window
        cutoff_time = current_time - duration_window
        directions_in_window = [(t, x, y) for t, x, y, _ in self._direction_history if t >= cutoff_time]

        if len(directions_in_window) < 3:  # Need at least 3 points to detect oscillation
            return False, []

        # Check if we have data covering sufficient duration (at least 80% of window)
        time_coverage = current_time - directions_in_window[0][0]
        if time_coverage < duration_window * 0.8:
            return False, []

        # Detect direction changes based on X component sign changes
        # Track direction changes with their direction
        direction_changes: list[tuple[float, SwipeDirection]] = []
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
                        # Determine which direction we changed TO
                        new_direction = SwipeDirection.RIGHT if current_sign > 0 else SwipeDirection.LEFT
                        direction_changes.append((t, new_direction))

            last_significant_x = x

        # Check if we have enough direction changes
        if len(direction_changes) < min_direction_changes:
            return False, []

        # Check that the last direction change is recent
        # This ensures we stop detecting when hand stops moving
        if direction_changes and max_time_since_last_change > 0:
            last_change_time = direction_changes[-1][0]
            time_since_last_change = current_time - last_change_time
            if time_since_last_change > max_time_since_last_change:
                return False, []

        # Check for recent movement if required
        if require_recent_movement:
            recent_directions = [
                (t, x, y) for t, x, y in directions_in_window if current_time - t <= recent_movement_window
            ]
            if len(recent_directions) < 2:
                return False, []

        # Verify angle changes are significant enough
        # Check angles between consecutive significant directions
        significant_directions = [(x, y) for _, x, y in directions_in_window if abs(x) >= x_tolerance]

        if len(significant_directions) < 3:
            return False, []

        # Check angle between first and middle, middle and last directions
        angle_verified = False
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
            if angle1 >= min_movement_angle_rad or angle2 >= min_movement_angle_rad:
                angle_verified = True
                break

        if not angle_verified:
            return False, []

        # Convert to list of (direction, time_ago) pairs
        changes_with_time_ago = [(direction, current_time - timestamp) for timestamp, direction in direction_changes]

        return True, changes_with_time_ago

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
        angle_rad = acos(dot_product)
        angle_deg = degrees(angle_rad)

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
        return angle_deg if t1 < 0 else -angle_deg

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

    def _calc_oriented_bounding_box_corners(self) -> tuple[tuple[float, float], ...] | None:
        """Calculate the four corners of the oriented bounding box.
        The box has one side perpendicular to main_direction and one side parallel to it.
        Returns tuple of 4 points (top-left, top-right, bottom-right, bottom-left) in pixel coordinates."""
        if not self.all_landmarks or not self.main_direction:
            return None

        # Get all landmark positions
        points = [(lm.x, lm.y) for lm in self.all_landmarks]

        # Get the main direction vector (already normalized)
        dir_x, dir_y = self.main_direction

        # Calculate perpendicular vector (rotate 90 degrees counter-clockwise)
        perp_x = -dir_y
        perp_y = dir_x

        # Project all points onto the two axes (parallel and perpendicular to main direction)
        parallel_coords = []
        perp_coords = []

        for x, y in points:
            # Project onto parallel axis (main direction)
            parallel_proj = x * dir_x + y * dir_y
            parallel_coords.append(parallel_proj)

            # Project onto perpendicular axis
            perp_proj = x * perp_x + y * perp_y
            perp_coords.append(perp_proj)

        # Find min/max projections
        min_parallel = min(parallel_coords)
        max_parallel = max(parallel_coords)
        min_perp = min(perp_coords)
        max_perp = max(perp_coords)

        # Build corners in the oriented coordinate system, then convert back to pixel coordinates
        # The four corners in terms of the parallel/perpendicular axes
        corners_local = [
            (min_parallel, min_perp),  # bottom-left in oriented space
            (max_parallel, min_perp),  # bottom-right in oriented space
            (max_parallel, max_perp),  # top-right in oriented space
            (min_parallel, max_perp),  # top-left in oriented space
        ]

        # Convert back to pixel coordinates
        corners = []
        for parallel, perp in corners_local:
            # Reconstruct the point from its projections
            px = parallel * dir_x + perp * perp_x
            py = parallel * dir_y + perp * perp_y
            corners.append((px, py))

        return tuple(corners)

    oriented_bounding_box = SmoothedProperty(_calc_oriented_bounding_box_corners, ManyCoordsSmoother, nb_coords=4)

    @cached_property
    def bounding_box_direction_line_points(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Get the entry and exit points of the hand's direction line through its bounding box."""
        if not self.wrist_landmark or not self.main_direction or not self.bounding_box:
            return None

        return self.bounding_box.line_intersections(
            (self.wrist_landmark.x, self.wrist_landmark.y), self.main_direction
        )

    @cached_property
    def frame_direction_line_points(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Get the entry and exit points of the hand's direction line through the frame boundaries."""
        if not self.wrist_landmark or not self.main_direction or not self.stream_info:
            return None

        # Create a Box representing the frame boundaries
        frame_box = Box(0, 0, self.stream_info.width, self.stream_info.height)

        return frame_box.line_intersections((self.wrist_landmark.x, self.wrist_landmark.y), self.main_direction)

    def _calc_other_hand_line_intersection(self) -> float | None:
        """Calculate where this hand's ray aligns with the other hand's direction line.

        Returns:
        - < 0: pointing behind the other hand
        - [0, 1]: pointing through the other hand's bounding box (0=entry, 1=exit)
        - > 1: pointing beyond the other hand
        - -inf: intersection would be negative but is outside frame
        - +inf: intersection would be positive but is outside frame
        - None: if rays don't intersect (parallel lines)
        """
        # Get other hand
        other = self.other_hand
        if not other:
            return None

        # Get required data for both hands
        my_wrist = self.wrist_landmark
        my_direction = self.main_direction

        if not my_wrist or not my_direction:
            return None

        # Get other hand's line intersection points with its bounding box
        other_line_points = other.bounding_box_direction_line_points
        if not other_line_points:
            return None

        entry_point, exit_point = other_line_points

        # Now we need to find where my ray intersects the other hand's line
        # This is a ray-line intersection problem

        # Other hand's line: P = entry_point + s * (exit_point - entry_point)
        # My ray: Q = my_wrist + t * my_direction (t > 0)

        # Vector along other hand's line
        line_dx = exit_point[0] - entry_point[0]
        line_dy = exit_point[1] - entry_point[1]

        # Solve for intersection
        px, py = my_wrist.x, my_wrist.y
        pdx, pdy = my_direction

        ex, ey = entry_point

        # Cross product to check if parallel
        cross = pdx * line_dy - pdy * line_dx

        if abs(cross) < 1e-10:
            # Lines are parallel, no intersection
            return None

        # Calculate parameters
        t = ((ex - px) * line_dy - (ey - py) * line_dx) / cross
        s = ((ex - px) * pdy - (ey - py) * pdx) / cross

        # Check if intersection is in positive direction (t > 0)
        if t <= 0:
            # Ray points backward
            return None

        # s is already the normalized position on the other hand's line:
        # s = 0 means at entry point
        # s = 1 means at exit point
        # s < 0 means before entry
        # s > 1 means after exit

        # Check if intersection point is within frame bounds
        if self.stream_info:
            intersection_x = px + t * pdx
            intersection_y = py + t * pdy

            if not (0 <= intersection_x <= self.stream_info.width and 0 <= intersection_y <= self.stream_info.height):
                # Intersection is outside frame
                # Return +inf or -inf depending on whether s would be positive or negative
                return inf if s >= 0 else -inf

        return s

    other_hand_line_intersection = smoothed_optional_float(_calc_other_hand_line_intersection)

    @cached_property
    def other_hand_line_intersection_normalized(self) -> float | None:
        """Normalized version of other_hand_line_intersection.

        Returns:
        - None, +inf, -inf: kept as is
        - Other values: normalized to [0, 1] based on frame_direction_line_points
        """
        # Get the raw intersection value (p_B in the formula)
        s = self.other_hand_line_intersection

        # Keep special values as is
        if s is None or s == inf or s == -inf:
            return s

        # Get the other hand's frame and bounding box line points
        if not (other := self.other_hand):
            return None
        if not (other_frame_points := other.frame_direction_line_points):
            return None
        if not (other_bbox_points := other.bounding_box_direction_line_points):
            return None

        # Unpack points - already ordered (entry, exit)
        frame_entry, frame_exit = other_frame_points
        bbox_entry, bbox_exit = other_bbox_points

        # Calculate distances for the formula
        # F = distance between frame_entry and frame_exit
        F = sqrt((frame_exit[0] - frame_entry[0]) ** 2 + (frame_exit[1] - frame_entry[1]) ** 2)

        # B = distance between bbox_entry and bbox_exit
        B = sqrt((bbox_exit[0] - bbox_entry[0]) ** 2 + (bbox_exit[1] - bbox_entry[1]) ** 2)

        # d_B = distance between frame_entry and bbox_entry
        d_B = sqrt((bbox_entry[0] - frame_entry[0]) ** 2 + (bbox_entry[1] - frame_entry[1]) ** 2)

        # Apply the formula: p_F = (d_B + p_B × B) / F
        # where p_B is s (the intersection position on bbox line)
        p_F = (d_B + s * B) / F

        return p_F

    @cached_property
    def other_hand_line_intersection_absolute(self) -> tuple[float, float] | None:
        """Absolute coordinates of the intersection point.

        Returns None if no valid intersection, otherwise (x, y) coordinates.
        """
        # Get the normalized intersection value
        p_f = self.other_hand_line_intersection_normalized

        # Can't calculate absolute position for special values
        if p_f is None or p_f == inf or p_f == -inf:
            return None

        # Get the other hand's frame line points
        if not (other := self.other_hand):
            return None
        if not (other_frame_points := other.frame_direction_line_points):
            return None

        # Unpack points
        frame_entry, frame_exit = other_frame_points

        # Calculate absolute coordinates using linear interpolation
        x = frame_entry[0] + p_f * (frame_exit[0] - frame_entry[0])
        y = frame_entry[1] + p_f * (frame_exit[1] - frame_entry[1])

        return x, y

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

    default_gesture = SmoothedProperty(
        _calc_default_gesture, EnumSmoother[Gestures | None], window=GESTURE_SMOOTHING_WINDOW, default_value=None
    )

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
    def gestures_data(self) -> dict[Gestures, dict[str, Any]]:
        """Get the data from gesture detectors for all currently active gestures."""
        return {
            gesture: data
            for gesture in self.gestures
            if gesture not in DEFAULT_GESTURES
            and (data := self.gestures_detector.detectors[gesture].get_data()) is not None
        }

    def is_gesture_disabled(self, gesture: Gestures) -> bool:
        return bool(getattr(self.config.hands.gestures.custom.disable, gesture.name))

    def detect_gestures(self) -> GestureWeights:
        """Detect all applicable custom gestures with weights.

        Returns:
            Dictionary of detected gestures with weight 1.0 for each
        """
        return self.gestures_detector.detect()
