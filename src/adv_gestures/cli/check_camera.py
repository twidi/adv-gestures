from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import cast

import cv2  # type: ignore[import-untyped]
import typer

from ..cameras import CameraInfo
from ..config import Config
from .common import (
    DEFAULT_USER_CONFIG_PATH,
    app,
    determine_mirror_mode,
    init_camera_capture,
    pick_camera,
)


def check_camera(
    camera_info: CameraInfo,
    show_preview: bool,
    mirror: bool,
    desired_size: int,
) -> None:
    """Check camera functionality without gesture recognition."""
    cap, window_name = init_camera_capture(camera_info, show_preview, desired_size)
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

        # Mirror frame if requested
        if mirror:
            frame = cv2.flip(frame, 1)

        # Draw top banner with FPS info
        frame_width = frame.shape[1]
        header_height = 30
        padding = 10

        # Add semi-transparent black header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, header_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Display FPS
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (padding, header_height - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

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


@app.command(name="check-camera")
def check_camera_cmd(
    camera: str | None = typer.Option(None, "--camera", "--cam", help="Camera name filter (case insensitive)"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show visual preview window"),
    mirror: bool = typer.Option(False, "--mirror", help="Force mirror mode (overrides environment variable)"),
    no_mirror: bool = typer.Option(
        False, "--no-mirror", help="Force no mirror mode (overrides environment variable)"
    ),
    size: int | None = typer.Option(None, "--size", "-s", help="Maximum dimension of the camera capture"),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help=f"Path to config file. Default: {DEFAULT_USER_CONFIG_PATH}"
    ),
) -> None:
    """Check camera functionality without gesture recognition."""
    # Load configuration
    config = Config.load(config_path)

    # Determine mirror mode (now with config)
    use_mirror = determine_mirror_mode(mirror, no_mirror, config)

    # Use config values as defaults, but CLI options take precedence
    final_camera = camera if camera is not None else config.cli.camera
    final_size = size if size is not None else config.cli.size

    selected = pick_camera(final_camera)

    if selected:
        print(f"\nSelected: {selected}")
        check_camera(selected, show_preview=preview, mirror=use_mirror, desired_size=final_size)
    else:
        print("\nNo camera selected.")
