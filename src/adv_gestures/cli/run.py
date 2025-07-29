from __future__ import annotations

import os
import sys
from math import atan2, degrees
from pathlib import Path
from typing import cast

import cv2  # type: ignore[import-untyped]
import typer

from ..cameras import CameraInfo
from ..config import Config
from ..drawing import draw_hands_marks_and_info
from ..models import Hands, Thumb
from ..recognizer import Recognizer, StreamInfo
from .common import (
    DEFAULT_USER_CONFIG_PATH,
    app,
    init_camera_capture,
    pick_camera,
)


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
        angle = f" ({hand.main_direction_angle:.0f}°)" if hand.main_direction_angle is not None else ""

        print(f"\n{handedness} Hand - {facing}{angle}")

        # Show all active gestures with weights and durations
        if hand.gestures:
            print("  Active gestures:")
            gestures_list = sorted(hand.gestures.items(), key=lambda x: x[1], reverse=True)
            durations = hand.gestures_durations

            for gesture, weight in gestures_list:
                gesture_line = f"    {gesture.name}: weight={weight:.2f}"

                # Add duration if available
                if gesture in durations:
                    gesture_line += f", duration={durations[gesture]:.1f}s"

                # Add source indicators
                sources = []
                if gesture in hand.custom_gestures:
                    sources.append("custom")
                if gesture == hand.default_gesture:
                    sources.append("default")
                if sources:
                    gesture_line += f" [{'/'.join(sources)}]"

                print(gesture_line)
        else:
            print("  No gestures detected")

        if hand.main_direction:
            direction = f"({hand.main_direction[0]:.2f}, {hand.main_direction[1]:.2f})"
            print(f"  Main direction: {direction}")

        if hand.all_adjacent_fingers_touching:
            print("  All adjacent fingers touching!")
        elif hand.all_adjacent_fingers_except_thumb_touching:
            print("  All adjacent fingers except thumb touching!")

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

            if not isinstance(finger, Thumb) and finger.tip_on_thumb:
                status.append("tip_on_thumb")

            if finger.touching_adjacent_fingers:
                touching_names = [f.name for f in finger.touching_adjacent_fingers]
                status.append(f"touching:{','.join(touching_names)}")

            if finger.fold_angle is not None:
                status.append(f"angle:{finger.fold_angle:.0f}°")

            if finger.tip_direction:
                dx, dy = finger.tip_direction
                tip_angle = degrees(atan2(-dy, dx))  # Negative dy because y increases downward in image coordinates
                status.append(f"tip_angle:{tip_angle:.0f}°")

            status_str = ", ".join(status) if status else "neutral"
            print(f"    {finger_name}: {status_str}")

    # Display two-hands gestures
    if hands.gestures:
        print("\nTwo-Hands Gestures:")
        gestures_list = sorted(hands.gestures.items(), key=lambda x: x[1], reverse=True)
        durations = hands.gestures_durations

        for gesture, weight in gestures_list:
            gesture_line = f"  {gesture.name}: weight={weight:.2f}"
            if gesture in durations:
                gesture_line += f", duration={durations[gesture]:.1f}s"
            print(gesture_line)

    # Display hands distance info
    if hands.hands_distance is not None:
        print(f"\nHands distance: {hands.hands_distance:.0f}px")
        if hands.hands_are_close:
            print("  Hands are close together!")


def run_gestures(
    camera_info: CameraInfo,
    show_preview: bool,
    config: Config,
    mirror: bool,
    desired_size: int,
) -> None:
    """Show a live preview of the selected camera with gesture recognition."""

    # Initialize global hands instance
    hands = Hands(config=config)

    cap, window_name = init_camera_capture(camera_info, show_preview, desired_size)
    if cap is None:
        return

    print("Loading gesture recognizer model...")

    try:
        # Create gesture recognizer with context manager
        with Recognizer(
            os.getenv("GESTURE_RECOGNIZER_MODEL_PATH", "").strip() or "gesture_recognizer.task", mirroring=mirror
        ) as recognizer:
            print("Gesture recognizer loaded successfully")
            if show_preview:
                print("Press 'q' or ESC to quit")

            for frame, stream_info, _ in recognizer.handle_opencv_capture(cap, hands):
                if show_preview:
                    frame = draw_hands_marks_and_info(hands, stream_info, frame)
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
        raise
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


@app.callback(invoke_without_command=True)
def run_gestures_cmd(
    ctx: typer.Context,
    camera: str | None = typer.Option(None, "--camera", "--cam", help="Camera name filter (case insensitive)"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show visual preview window"),
    mirror: bool | None = typer.Option(None, "--mirror/--no-mirror", help="Mirror the video output"),
    size: int | None = typer.Option(None, "--size", "-s", help="Maximum dimension of the camera capture"),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help=f"Path to config file. Default: {DEFAULT_USER_CONFIG_PATH}"
    ),
) -> None:
    """Run gesture recognition on selected camera.

    The default config location is platform-specific and will be shown if the config file is not found.
    """
    # If a subcommand is being invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    # Load configuration
    config = Config.load(config_path)

    # Use config values as defaults, but CLI options take precedence
    final_camera = camera if camera is not None else config.cli.camera
    final_mirror = mirror if mirror is not None else config.cli.mirror
    final_size = size if size is not None else config.cli.size

    selected = pick_camera(final_camera)

    if selected:
        print(f"\nSelected: {selected}")
        run_gestures(selected, show_preview=preview, config=config, mirror=final_mirror, desired_size=final_size)
    else:
        print("\nNo camera selected.")
