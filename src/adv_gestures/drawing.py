from math import sqrt
from typing import TYPE_CHECKING, TypeAlias, cast

import cv2  # type: ignore[import-untyped]

from .gestures import Gestures
from .models.fingers import AnyFinger, FingerIndex, Thumb
from .models.hands import Box, Hand, Hands, HandsDirectionalRelationship, Palm
from .models.hands.hand_gestures import AirTapDetector

if TYPE_CHECKING:
    from .recognizer import StreamInfo

# Colors for drawing fingers (BGR format for OpenCV)
FINGER_COLORS = [
    (255, 0, 0),  # Blue - THUMB
    (0, 255, 0),  # Green - INDEX
    (0, 255, 255),  # Yellow - MIDDLE
    (255, 0, 255),  # Magenta - RING
    (255, 255, 0),  # Cyan - PINKY
]

DIRECTIONS_INDICATORS = {
    HandsDirectionalRelationship.PARALLEL: " ||",
    HandsDirectionalRelationship.CONVERGING: " ^",
    HandsDirectionalRelationship.INTERSECTING: " x",
    HandsDirectionalRelationship.LEFT_INTO_RIGHT: " -|",
    HandsDirectionalRelationship.RIGHT_INTO_LEFT: " |-",
    HandsDirectionalRelationship.DIVERGING_NORMAL: " v",
    HandsDirectionalRelationship.DIVERGING_CROSSED: " ><",
    HandsDirectionalRelationship.DIVERGING_LEFT_BEHIND_RIGHT: " |_",
    HandsDirectionalRelationship.DIVERGING_RIGHT_BEHIND_LEFT: " _|",
}

OpenCVImage: TypeAlias = cv2.typing.MatLike  # Type alias for images (numpy arrays)


def draw_dotted_box(
    box: Box,
    image: OpenCVImage,
    color: tuple[int, int, int] = (180, 180, 180),
    thickness: int = 1,
    dot_length: int = 5,
    gap_length: int = 5,
) -> OpenCVImage:
    """Draw the box as a dotted rectangle on the image.

    Args:
        image: The image to draw on
        color: BGR color tuple
        thickness: Line thickness
        dot_length: Length of each dot
        gap_length: Length of gap between dots

    Returns:
        The image with the box drawn
    """
    height, width = image.shape[:2]
    x1 = int(round(box.min_x))
    y1 = int(round(box.min_y))
    x2 = int(round(box.max_x))
    y2 = int(round(box.max_y))

    # Top edge
    x = x1
    while x < x2:
        cv2.line(image, (x, y1), (min(x + dot_length, x2), y1), color, thickness)
        x += dot_length + gap_length

    # Bottom edge
    x = x1
    while x < x2:
        cv2.line(image, (x, y2), (min(x + dot_length, x2), y2), color, thickness)
        x += dot_length + gap_length

    # Left edge
    y = y1
    while y < y2:
        cv2.line(image, (x1, y), (x1, min(y + dot_length, y2)), color, thickness)
        y += dot_length + gap_length

    # Right edge
    y = y1
    while y < y2:
        cv2.line(image, (x2, y), (x2, min(y + dot_length, y2)), color, thickness)
        y += dot_length + gap_length

    return image


def draw_hands_marks(hands: Hands, image: OpenCVImage) -> OpenCVImage:
    """Draw both hands on the image."""
    image = draw_hand_marks(hands.left, image)
    image = draw_hand_marks(hands.right, image)
    return image


def draw_hand_marks(hand: Hand, image: OpenCVImage) -> OpenCVImage:
    """Draw the hand on the image."""
    if not hand:
        return image

    # Draw wrist landmark
    if hand.wrist_landmark:
        wrist_x = int(round(hand.wrist_landmark.x))
        wrist_y = int(round(hand.wrist_landmark.y))
        cv2.circle(image, (wrist_x, wrist_y), 5, (255, 255, 255), -1)

    # Draw palm
    if hand.palm:
        image = draw_palm_marks(hand.palm, image)

    # Draw fingers
    for finger in hand.fingers:
        image = draw_finger_marks(finger, image)

    # Draw main direction arrow
    if hand.main_direction and hand.wrist_landmark:
        # Start point at wrist
        start_x = hand.wrist_landmark.x
        start_y = hand.wrist_landmark.y

        # The main_direction is already a normalized vector in pixel space
        dx_norm, dy_norm = hand.main_direction

        # Calculate end point
        arrow_length = 70  # pixels
        end_x = int(start_x + dx_norm * arrow_length)
        end_y = int(start_y + dy_norm * arrow_length)

        # Draw arrow (cyan color)
        cv2.arrowedLine(
            image, (int(round(start_x)), int(round(start_y))), (end_x, end_y), (255, 255, 0), 3, tipLength=0.3
        )

        # Continue the ray to the edge of the frame
        frame_height, frame_width = image.shape[:2]

        # Calculate where the ray intersects with frame boundaries
        # We need to find the furthest intersection point
        t_values = []

        # Check intersection with left edge (x = 0)
        if dx_norm < 0:
            t = -start_x / dx_norm
            y = start_y + t * dy_norm
            if 0 <= y <= frame_height:
                t_values.append((t, 0, int(y)))

        # Check intersection with right edge (x = frame_width)
        if dx_norm > 0:
            t = (frame_width - start_x) / dx_norm
            y = start_y + t * dy_norm
            if 0 <= y <= frame_height:
                t_values.append((t, frame_width - 1, int(y)))

        # Check intersection with top edge (y = 0)
        if dy_norm < 0:
            t = -start_y / dy_norm
            x = start_x + t * dx_norm
            if 0 <= x <= frame_width:
                t_values.append((t, int(x), 0))

        # Check intersection with bottom edge (y = frame_height)
        if dy_norm > 0:
            t = (frame_height - start_y) / dy_norm
            x = start_x + t * dx_norm
            if 0 <= x <= frame_width:
                t_values.append((t, int(x), frame_height - 1))

        # Draw the extended line if we found an intersection
        if t_values:
            # Get the closest intersection (smallest positive t)
            t_values.sort(key=lambda x: x[0])
            _, edge_x, edge_y = t_values[0]

            # Draw line from arrow end to edge
            cv2.line(image, (end_x, end_y), (edge_x, edge_y), (255, 255, 0), 1)

    # Draw bounding box with dotted lines
    if hand.bounding_box:
        image = draw_dotted_box(hand.bounding_box, image)

    # Draw pinch box with dotted lines in different color
    if hand.pinch_box:
        # Red for PINCH_TOUCH, yellow for regular PINCH
        if Gestures.PINCH_TOUCH in hand.gestures:
            color = (0, 0, 255)  # Red for pinch touch
        else:
            color = (0, 255, 255)  # Yellow for regular pinch

        image = draw_dotted_box(
            hand.pinch_box,
            image,
            color=color,
            thickness=2,
        )

    # Draw air tap indicator
    air_tap_detector = cast(AirTapDetector, hand.gestures_detector.detectors[Gestures.AIR_TAP])
    if (air_tap_state := air_tap_detector.tap_state) is not None:
        # if tip_state is not None, tip_position is set
        air_tap_x, air_tap_y = cast(tuple[float, float], air_tap_detector.tip_position)

        # Calculate circle radius based on frame width (approximately 2% of width)
        frame_width = image.shape[1]
        radius = int(frame_width * 0.02)

        # Choose color based on tap state
        if air_tap_state == "detected":
            color = (255, 0, 255)  # Magenta for detected (post-detection)
        else:  # "detecting"
            color = (255, 100, 0)  # Blue for detecting (tracking)

        # Draw circle (unfilled, just the border)
        cv2.circle(image, (int(air_tap_x), int(air_tap_y)), radius, color, 2)

    return image


def draw_palm_marks(palm: Palm, image: OpenCVImage) -> OpenCVImage:
    """Draw the palm center on the image."""
    centroid = palm.centroid
    if not centroid:
        return image

    palm_x, palm_y = centroid

    # Already in pixel coordinates
    palm_center_x = int(round(palm_x))
    palm_center_y = int(round(palm_y))

    # Determine palm color based on facing
    palm_color = (0, 255, 0) if palm.hand.is_facing_camera else (255, 0, 0)

    # Draw palm center with color indicating facing
    cv2.circle(image, (palm_center_x, palm_center_y), 5, palm_color, -1)

    return image


def draw_finger_marks(finger: AnyFinger, image: OpenCVImage) -> OpenCVImage:
    """Draw the finger on the image."""
    # Draw all landmarks
    for landmark in finger.landmarks:
        cv2.circle(image, landmark.xy, 3, FINGER_COLORS[finger.index], -1)

    # Draw cross at finger centroid
    if finger.centroid:
        centroid_x, centroid_y = finger.centroid
        cx = int(round(centroid_x))
        cy = int(round(centroid_y))
        color = FINGER_COLORS[finger.index]
        # Cross size matching landmark size
        cross_size = 5
        cv2.line(image, (cx - cross_size, cy), (cx + cross_size, cy), color, 1)
        cv2.line(image, (cx, cy - cross_size), (cx, cy + cross_size), color, 1)

    if not finger.start_point or not finger.end_point:
        # If no start or end point, we cannot draw the finger line
        return image

    # Draw colored line for straight or nearly straight finger
    if finger.is_straight or finger.is_nearly_straight:
        start_x = finger.start_point[0]
        start_y = finger.start_point[1]
        end_x = finger.end_point[0]
        end_y = finger.end_point[1]

        color = FINGER_COLORS[finger.index]

        if finger.is_straight:
            cv2.line(
                image, (int(round(start_x)), int(round(start_y))), (int(round(end_x)), int(round(end_y))), color, 3
            )
        else:
            # Dashed line for nearly straight fingers
            # Calculate total length and create dashed pattern
            dx = end_x - start_x
            dy = end_y - start_y
            length = sqrt(dx**2 + dy**2)

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
    if finger.tip_direction:
        # Get fingertip position
        tip_x = finger.end_point[0]
        tip_y = finger.end_point[1]

        # The tip_direction is already a normalized vector in pixel space
        dx_norm, dy_norm = finger.tip_direction

        # Calculate arrow end point
        arrow_length = 30  # Small arrow
        arrow_end_x = int(tip_x + dx_norm * arrow_length)
        arrow_end_y = int(tip_y + dy_norm * arrow_length)

        # Draw small arrow in white
        cv2.arrowedLine(
            image,
            (int(round(tip_x)), int(round(tip_y))),
            (arrow_end_x, arrow_end_y),
            (255, 255, 255),
            2,
            tipLength=0.4,
        )

    # Draw red circle if this finger touches the thumb
    if not isinstance(finger, Thumb) and finger.tip_on_thumb:
        # Get thumb finger
        thumb = None
        for other_finger in finger.hand.fingers:
            if other_finger.index == FingerIndex.THUMB:
                thumb = other_finger
                break

        if thumb and thumb.end_point:
            # Calculate middle point between both tips
            finger_tip_x = finger.end_point[0]
            finger_tip_y = finger.end_point[1]
            thumb_tip_x = thumb.end_point[0]
            thumb_tip_y = thumb.end_point[1]

            middle_x = (finger_tip_x + thumb_tip_x) / 2
            middle_y = (finger_tip_y + thumb_tip_y) / 2

            # Convert to integer pixel coordinates
            pixel_x = int(round(middle_x))
            pixel_y = int(round(middle_y))

            # Draw small red filled circle
            cv2.circle(image, (pixel_x, pixel_y), 8, (0, 0, 255), -1)

    # Draw indicators for touching adjacent fingers
    if finger.touching_adjacent_fingers:
        for touching_index in finger.touching_adjacent_fingers:
            # Find the touching finger
            touching_finger = None
            for other_finger in finger.hand.fingers:
                if other_finger.index == touching_index:
                    touching_finger = other_finger
                    break

            if touching_finger and touching_finger.end_point:
                # Draw a line between the fingertips
                my_tip_x = int(round(finger.end_point[0]))
                my_tip_y = int(round(finger.end_point[1]))
                other_tip_x = int(round(touching_finger.end_point[0]))
                other_tip_y = int(round(touching_finger.end_point[1]))

                # Draw a thick cyan line between touching fingers
                cv2.line(image, (my_tip_x, my_tip_y), (other_tip_x, other_tip_y), (255, 255, 0), 4)

                # Draw small circles at connection points
                cv2.circle(image, (my_tip_x, my_tip_y), 5, (255, 255, 0), -1)
                cv2.circle(image, (other_tip_x, other_tip_y), 5, (255, 255, 0), -1)

    return image


def draw_hands_marks_and_info(hands: Hands, stream_info: "StreamInfo", frame: OpenCVImage) -> OpenCVImage:
    """Draw hands preview on frame and return the modified frame."""
    if stream_info.mirroring:
        frame = cv2.flip(frame, 1)

    frame = draw_hands_marks(hands, frame)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Draw top banner with FPS and latency info
    header_height = 30
    padding = 10

    # Add semi-transparent black header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, header_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Display FPS and latency on same line
    fps_text = f"FPS: frames: {stream_info.frames_fps:.1f}, recognition: {stream_info.recognition_fps:.1f}"
    latency_ms = stream_info.latency * 1000
    latency_text = f"Latency: {latency_ms:.1f}ms"
    metrics_text = f"{fps_text}  |  {latency_text}"

    cv2.putText(
        frame,
        metrics_text,
        (padding, header_height - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Prepare text lines for each hand
    line_height = 20  # Reduced line height for more compact display
    gesture_header_height = 25  # Height for the hand header
    gesture_padding = 10

    # Build text lines for left and right hands separately
    left_lines: list[tuple[str, str]] = []
    right_lines: list[tuple[str, str]] = []

    for hand, lines in [(hands.left, left_lines), (hands.right, right_lines)]:
        # if not hand:
        #     continue
        handedness_str = hand.handedness.name
        if not hand.is_visible:
            # If hand is not visible, show "No hand detected" message
            lines.append((f"{handedness_str} hand not detected", "header"))
            continue
        if hand.is_showing_side:
            facing_str = "SIDE"
        else:
            facing_str = "PALM" if hand.is_facing_camera else "BACK"

        # Hand header line
        angle_str = f" ({hand.main_direction_angle:.0f}deg)" if hand.main_direction_angle is not None else ""
        header_text = f"{handedness_str} hand showing {facing_str}{angle_str}"
        lines.append((header_text, "header"))

        # Show all active gestures with weights and durations
        if hand.gestures:
            gestures_list = sorted(hand.gestures.items(), key=lambda x: x[1], reverse=True)
            durations = hand.gestures_durations

            for gesture, weight in gestures_list:
                gesture_text = f"{gesture.name}: {weight:.2f}"
                if gesture in durations:
                    gesture_text += f" ({durations[gesture]:.1f}s)"

                # Add source indicator (custom/default)
                source_indicators = []
                if gesture in hand.custom_gestures:
                    source_indicators.append("custom")
                if gesture == hand.default_gesture:
                    source_indicators.append("default")
                if source_indicators:
                    gesture_text += f" [{'/'.join(source_indicators)}]"

                lines.append((gesture_text, "gesture"))
        else:
            lines.append(("No gestures detected", "gesture"))

    # Process two-hands info to display in center column (if both hands are visible)
    center_lines: list[tuple[str, str]] = []
    if hands.left and hands.right:
        # Header for two-hands info with angle difference
        angle_diff_str = (
            f" (diff {hands.hands_direction_angle_diff:.0f}deg)"
            if hands.hands_direction_angle_diff is not None
            else ""
        )
        # Add direction indicator based on hands directions
        direction_indicator = (
            DIRECTIONS_INDICATORS.get(hands.directional_relationship, "") if hands.directional_relationship else ""
        )
        center_lines.append((f"TWO HANDS{angle_diff_str}{direction_indicator}", "header"))

        # Show two-hands gestures if any
        if hands.gestures:
            # Get two-hands gestures with their durations
            gestures_list = sorted(hands.gestures.items(), key=lambda x: x[1], reverse=True)
            durations = hands.gestures_durations

            for gesture, weight in gestures_list:
                gesture_text = f"{gesture.name}: {weight:.2f}"
                if gesture in durations:
                    gesture_text += f" ({durations[gesture]:.1f}s)"
                center_lines.append((gesture_text, "gesture"))
        else:
            center_lines.append(("No two-hands gestures", "gesture"))

    # Calculate footer dimensions based on max lines
    if left_lines or right_lines or center_lines:
        # Calculate height needed for each side
        left_height = 0
        right_height = 0
        center_height = 0

        has_gestures = False
        for _, line_type in left_lines:
            if line_type == "header":
                left_height += gesture_header_height
            elif line_type == "gesture":
                has_gestures = True
                left_height += line_height
        if has_gestures:
            left_height += gesture_padding

        has_gestures = False
        for _, line_type in right_lines:
            if line_type == "header":
                right_height += gesture_header_height
            elif line_type == "gesture":
                has_gestures = True
                right_height += line_height
        if has_gestures:
            right_height += gesture_padding

        has_gestures = False
        for _, line_type in center_lines:
            if line_type == "header":
                center_height += gesture_header_height
            elif line_type == "gesture":
                has_gestures = True
                center_height += line_height
        if has_gestures:
            center_height += gesture_padding

        # Use the maximum height
        footer_height = max(left_height, right_height, center_height)
        footer_y_start = frame_height - footer_height

        # Add semi-transparent black footer
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, footer_y_start), (frame_width, frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)  # Slightly darker overlay

        for lines, color in (
            (left_lines, (255, 255, 255)),
            (right_lines, (255, 255, 255)),
            (center_lines, (255, 255, 0)),
        ):
            # if not lines:
            #     continue
            y_pos = footer_y_start
            center_x = frame_width // 2
            for text, line_type in lines:
                if line_type == "header":
                    if lines is left_lines:
                        x_pos = 10
                    else:
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        x_pos = (
                            (frame_width - text_size[0] - 10)
                            if lines is right_lines
                            else (center_x - text_size[0] // 2)
                        )
                    # Header with larger font
                    cv2.putText(
                        frame,
                        text,
                        (x_pos, y_pos + gesture_header_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_pos += gesture_header_height
                elif line_type == "gesture":
                    # Gesture info with smaller font
                    if lines is left_lines:
                        x_pos = 10
                    else:
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        x_pos = (
                            (frame_width - text_size[0] - 10)
                            if lines is right_lines
                            else (center_x - text_size[0] // 2)
                        )
                    cv2.putText(
                        frame,
                        text,
                        (x_pos, y_pos + line_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_pos += line_height

    return frame
