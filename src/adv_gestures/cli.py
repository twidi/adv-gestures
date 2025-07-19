#!/usr/bin/env python3

"""The CLI is for development and debugging purpose."""

from __future__ import annotations

import os
import sys
import time
from typing import cast

import cv2  # type: ignore[import-untyped]
import numpy as np
import typer

from .cameras import CameraInfo, list_cameras
from .drawing import draw_hands_marks_and_info
from .models import Hands
from .recognizer import Recognizer, StreamInfo

MIRROR_OUTPUT = True
DESIRED_SIZE = 1280  # size of the max dimension of the camera


def pick_camera(filter_name: str | None = None) -> CameraInfo | None:
    """List cameras and let user pick one. Returns selected CameraInfo or None.

    Args:
        filter_name: Optional string to filter cameras by name (case insensitive)
    """
    cameras = list_cameras()

    if not cameras:
        print("No cameras found!", file=sys.stderr)
        return None

    # Filter cameras if a filter name is provided
    if filter_name:
        filter_lower = filter_name.lower()
        filtered_cameras = [cam for cam in cameras if filter_lower in cam.name.lower()]

        if not filtered_cameras:
            print(f"No cameras found matching '{filter_name}'", file=sys.stderr)
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
                print(f"Invalid choice. Please enter one of: {', '.join(map(str, valid_indices))}", file=sys.stderr)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'", file=sys.stderr)


def print_hands_info(hands: Hands, stream_info: StreamInfo) -> None:
    """Print important features of hands/fingers to console."""
    # Print metrics first if available
    metrics_info = []
    if stream_info.frames_fps > 0:
        metrics_info.append(f"Frames FPS: {stream_info.frames_fps:.1f}")
    if stream_info.recognition_fps > 0:
        metrics_info.append(f"Recognition FPS: {stream_info.recognition_fps:.1f}")

    # Add latency in milliseconds
    latency_ms = stream_info.latency * 1000
    metrics_info.append(f"Latency: {latency_ms:.1f}ms")

    if metrics_info:
        print(f"\r{' | '.join(metrics_info)}", end="")

    if not hands.left and not hands.right:
        if not metrics_info:  # Only print if metrics weren't already printed
            print("\rNo hands detected", end="")
        return

    for hand in [hands.left, hands.right]:
        if not hand:
            continue

        handedness = hand.handedness.name if hand.handedness else "Unknown"
        facing = "PALM" if hand.is_facing_camera else "BACK"
        gesture = hand.gesture if hand.gesture else "None"

        # Add gesture duration to the display
        gesture_display = f"{gesture}"
        if hand.gesture_duration > 0:
            gesture_display += f" ({hand.gesture_duration:.1f}s)"

        print(f"\n{handedness} Hand - {facing} - Gesture: {gesture_display}")

        # Show custom and default gestures with durations if different from main gesture
        gesture_details = []
        if hand.custom_gesture and hand.custom_gesture != hand.gesture:
            custom_text = f"custom: {hand.custom_gesture}"
            if hand.custom_gesture_duration > 0:
                custom_text += f" ({hand.custom_gesture_duration:.1f}s)"
            gesture_details.append(custom_text)

        if hand.default_gesture and hand.default_gesture != hand.gesture:
            default_text = f"default: {hand.default_gesture}"
            if hand.default_gesture_duration > 0:
                default_text += f" ({hand.default_gesture_duration:.1f}s)"
            gesture_details.append(default_text)

        if gesture_details:
            print(f"  Gesture details: {' | '.join(gesture_details)}")

        if hand.main_direction:
            direction = f"({hand.main_direction[0]:.2f}, {hand.main_direction[1]:.2f})"
            print(f"  Main direction: {direction}")

        if hand.all_fingers_touching:
            print("  All adjacent fingers touching!")

        for finger in hand.fingers:
            finger_name = finger.index.name
            status = []

            if finger.is_nearly_straight_or_straight:
                status.append("straight" if finger.is_straight else "nearly_straight")
                if finger.straight_direction:
                    dir_str = f"({finger.straight_direction[0]:.2f}, {finger.straight_direction[1]:.2f})"
                    status.append(f"dir:{dir_str}")

            if finger.is_fully_bent:
                status.append("bent")

            if finger.touches_thumb:
                status.append("touches_thumb")

            if finger.touching_adjacent_fingers:
                touching_names = [f.name for f in finger.touching_adjacent_fingers]
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
        print(f"Error: Could not open camera {camera_info.device_index}", file=sys.stderr)
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


def run_gestures(camera_info: CameraInfo, show_preview: bool = True) -> None:
    """Show a live preview of the selected camera with gesture recognition."""

    # Initialize global hands instance
    hands = Hands()

    cap, window_name = init_camera_capture(camera_info, show_preview)
    if cap is None:
        return

    print("Loading gesture recognizer model...")

    try:
        # Create gesture recognizer with context manager
        with Recognizer(
            os.getenv("GESTURE_RECOGNIZER_MODEL_PATH", "").strip() or "gesture_recognizer.task"
        ) as recognizer:
            print("Gesture recognizer loaded successfully")
            if show_preview:
                print("Press 'q' or ESC to quit")

            for frame, stream_info, _ in recognizer.handle_opencv_capture(cap, hands):
                if show_preview:
                    frame = draw_hands_marks_and_info(hands, stream_info, frame, MIRROR_OUTPUT)
                else:
                    print_hands_info(hands, stream_info)

                if show_preview:
                    cv2.imshow(cast(str, window_name), frame)

                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' or ESC
                        break

                    # Check if window was closed
                    try:
                        if cv2.getWindowProperty(cast(str, window_name), cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        # Window was closed
                        break
    except Exception as e:
        print(f"\nError loading gesture recognizer: {e}", file=sys.stderr)
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


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
            print("Error: Failed to capture frame", file=sys.stderr)
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

        # Check if window was closed
        try:
            if cv2.getWindowProperty(cast(str, window_name), cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # Window was closed
            break

    cap.release()
    cv2.destroyAllWindows()


app = typer.Typer()


@app.command()
def run(
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
            run_gestures(selected, show_preview=preview)
    else:
        print("\nNo camera selected.")


def main() -> None:
    """Entry point for the application."""
    app()


if __name__ == "__main__":
    main()
