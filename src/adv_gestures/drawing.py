from math import sqrt
from typing import TYPE_CHECKING, TypeAlias

import cv2  # type: ignore[import-untyped]

from .models import Box, Finger, FingerIndex, Hand, Hands, Palm

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
    x1, y1, x2, y2 = box.to_pixels(width, height)

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
        height, width = image.shape[:2]
        wrist_x = int(hand.wrist_landmark.x * width)
        wrist_y = int(hand.wrist_landmark.y * height)
        cv2.circle(image, (wrist_x, wrist_y), 5, (255, 255, 255), -1)

    # Draw palm
    if hand.palm:
        image = draw_palm_marks(hand.palm, image)

    # Draw fingers
    for finger in hand.fingers:
        image = draw_finger_marks(finger, image)

    # Draw main direction arrow
    if hand.main_direction and hand.wrist_landmark:
        height, width = image.shape[:2]

        # Start point at wrist
        start_x = int(hand.wrist_landmark.x * width)
        start_y = int(hand.wrist_landmark.y * height)

        # The main_direction is in normalized space, so we need to convert it to pixel space
        # accounting for the aspect ratio
        dx_norm, dy_norm = hand.main_direction

        # Convert normalized direction to pixel direction
        dx_pixel = dx_norm * width
        dy_pixel = dy_norm * height

        # Re-normalize in pixel space
        magnitude_pixel = sqrt(dx_pixel**2 + dy_pixel**2)
        if magnitude_pixel > 0:
            dx_pixel_norm = dx_pixel / magnitude_pixel
            dy_pixel_norm = dy_pixel / magnitude_pixel

            # Calculate end point
            arrow_length = 70  # pixels
            end_x = int(start_x + dx_pixel_norm * arrow_length)
            end_y = int(start_y + dy_pixel_norm * arrow_length)

            # Draw arrow (cyan color)
            cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (255, 255, 0), 3, tipLength=0.3)

    # Draw bounding box with dotted lines
    if hand.bounding_box:
        image = draw_dotted_box(hand.bounding_box, image)

    # Draw pinch box with dotted lines in different color
    if hand.pinch_box:
        # Check if thumb and index are touching
        thumb, index, *_ = hand.fingers
        if index.touches_thumb:
            color = (0, 0, 255)  # Red when touching
        else:
            color = (0, 255, 255)  # Yellow when not touching

        image = draw_dotted_box(
            hand.pinch_box,
            image,
            color=color,
            thickness=2,
        )

    return image


def draw_palm_marks(palm: Palm, image: OpenCVImage) -> OpenCVImage:
    """Draw the palm center on the image."""
    height, width = image.shape[:2]

    centroid = palm.centroid
    if not centroid:
        return image

    palm_x, palm_y = centroid

    # Convert to pixel coordinates
    palm_center_x = int(palm_x * width)
    palm_center_y = int(palm_y * height)

    # Determine palm color based on facing
    palm_color = (0, 255, 0) if palm.hand.is_facing_camera else (255, 0, 0)

    # Draw palm center with color indicating facing
    cv2.circle(image, (palm_center_x, palm_center_y), 5, palm_color, -1)

    return image


def draw_finger_marks(finger: Finger, image: OpenCVImage) -> OpenCVImage:
    """Draw the finger on the image."""
    height, width = image.shape[:2]

    # Draw all landmarks
    for landmark in finger.landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 3, FINGER_COLORS[finger.index], -1)

    if not finger.start_point or not finger.end_point:
        # If no start or end point, we cannot draw the finger line
        return image

    # Draw colored line for straight or nearly straight finger
    if finger.is_straight or finger.is_nearly_straight:
        start_x = int(finger.start_point[0] * width)
        start_y = int(finger.start_point[1] * height)
        end_x = int(finger.end_point[0] * width)
        end_y = int(finger.end_point[1] * height)

        color = FINGER_COLORS[finger.index]

        if finger.is_straight:
            # Solid line for straight fingers
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 3)
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
        tip_x = int(finger.end_point[0] * width)
        tip_y = int(finger.end_point[1] * height)

        # Convert normalized direction to pixel space
        dx_norm, dy_norm = finger.tip_direction
        dx_pixel = dx_norm * width
        dy_pixel = dy_norm * height

        # Re-normalize in pixel space
        magnitude_pixel = sqrt(dx_pixel**2 + dy_pixel**2)
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
    if finger.touches_thumb:
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

            # Convert to pixel coordinates
            pixel_x = int(middle_x * width)
            pixel_y = int(middle_y * height)

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
                my_tip_x = int(finger.end_point[0] * width)
                my_tip_y = int(finger.end_point[1] * height)
                other_tip_x = int(touching_finger.end_point[0] * width)
                other_tip_y = int(touching_finger.end_point[1] * height)

                # Draw a thick cyan line between touching fingers
                cv2.line(image, (my_tip_x, my_tip_y), (other_tip_x, other_tip_y), (255, 255, 0), 4)

                # Draw small circles at connection points
                cv2.circle(image, (my_tip_x, my_tip_y), 5, (255, 255, 0), -1)
                cv2.circle(image, (other_tip_x, other_tip_y), 5, (255, 255, 0), -1)

    return image


def draw_hands_marks_and_info(
    hands: Hands, stream_info: "StreamInfo", frame: OpenCVImage, mirror_hands: bool = False
) -> OpenCVImage:
    """Draw hands preview on frame and return the modified frame."""
    frame = draw_hands_marks(hands, frame)

    if mirror_hands:
        frame = cv2.flip(frame, 1)

    # Draw metrics in top-left corner
    y_pos = 30

    # Display FPS in single line
    fps_text = f"FPS: frames: {stream_info.frames_fps:.1f} recognition: {stream_info.recognition_fps:.1f}"
    cv2.putText(frame, fps_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    y_pos += 30

    # Display latency
    latency_ms = stream_info.latency * 1000
    latency_text = f"Latency: {latency_ms:.1f}ms"
    cv2.putText(frame, latency_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

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
            # Show the final gesture (smoothed)
            text += f" - Gesture: {hand.gesture}"

            # Add gesture duration
            if hand.gesture_duration > 0:
                text += f" ({hand.gesture_duration:.1f}s)"

            # Show custom and default gestures with durations
            details = []

            # Custom gesture info
            if hand.custom_gesture:
                custom_text = f"custom: {hand.custom_gesture}"
                if hand.custom_gesture_duration > 0:
                    custom_text += f" ({hand.custom_gesture_duration:.1f}s)"
                details.append(custom_text)

            # Default gesture info
            if hand.default_gesture:
                default_text = f"default: {hand.default_gesture}"
                if hand.default_gesture_duration > 0:
                    default_text += f" ({hand.default_gesture_duration:.1f}s)"
                details.append(default_text)

            if details:
                text += f" | {' | '.join(details)}"

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
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
