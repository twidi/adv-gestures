from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import cv2  # type: ignore[import-untyped]

from ..cameras import CameraInfo
from ..config import Config
from ..models import Hands
from ..models.landmarks import HandLandmark
from ..recognizer import Recognizer
from . import options
from .common import (
    app,
    determine_gpu_usage,
    determine_mirror_mode,
    init_camera_capture,
    pick_camera,
)

# Define hand connections (skeleton structure)
HAND_CONNECTIONS = [
    # Thumb connections
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    # Index finger connections
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    # Middle finger connections
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    # Ring finger connections
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    # Pinky connections
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    # Palm connections
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.WRIST),
]


def visualize_depth(
    camera_info: CameraInfo,
    show_preview: bool,
    config: Config,
    mirror: bool,
    desired_size: int,
    use_gpu: bool = True,
) -> None:
    """Visualize hand landmarks with depth information."""
    # Initialize global hands instance
    hands = Hands(config=config)

    cap, window_name = init_camera_capture(camera_info, show_preview, desired_size)
    if cap is None:
        return

    print("Loading gesture recognizer model...")

    try:
        # Create gesture recognizer with context manager
        with Recognizer(
            os.getenv("GESTURE_RECOGNIZER_MODEL_PATH", "").strip() or "gesture_recognizer.task",
            use_gpu=use_gpu,
            mirroring=mirror,
        ) as recognizer:
            print("Gesture recognizer loaded successfully")
            if show_preview:
                print("Press 'q' or ESC to quit")
                print("Visualizing hand depth - darker points are further away")

            for frame, stream_info, _ in recognizer.handle_opencv_capture(cap, hands):
                if show_preview:
                    # Draw depth visualization
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    frame = draw_depth_visualization(frame, hands)

                    # Add FPS info
                    if stream_info.frames_fps > 0:
                        fps_text = f"FPS: {stream_info.frames_fps:.1f}"
                        cv2.putText(
                            frame,
                            fps_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
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
    except Exception as e:
        print(f"\nError loading gesture recognizer: {e}", file=sys.stderr)
        raise
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


def draw_depth_visualization(frame: Any, hands: Hands) -> Any:
    """Draw hand landmarks with depth-based coloring and connections."""
    for hand in [hands.left, hands.right]:
        if not hand or not hand.all_landmarks:
            continue

        # Get all z values to normalize depth
        z_values = [lm.z_raw for lm in hand.all_landmarks if lm.z_raw is not None]
        if not z_values:
            continue

        z_min = min(z_values)
        z_max = max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        # Draw connections first (behind landmarks)
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_lm = hand.all_landmarks[start_idx]
            end_lm = hand.all_landmarks[end_idx]

            if start_lm.z_raw is not None and end_lm.z_raw is not None:
                # Calculate average depth for connection color
                avg_z = (start_lm.z_raw + end_lm.z_raw) / 2
                normalized_z = (avg_z - z_min) / z_range if z_range > 0 else 0.5

                # Color from white (close) to dark gray (far)
                conn_intensity = int(255 * (1 - normalized_z * 0.8))  # Keep minimum brightness at 51
                color = (conn_intensity, conn_intensity, conn_intensity)

                # Draw connection line
                cv2.line(frame, (start_lm.x, start_lm.y), (end_lm.x, end_lm.y), color, 2, cv2.LINE_AA)

        # Draw landmarks with depth-based coloring
        for idx, landmark in enumerate(hand.all_landmarks):
            # Normalize z value (0 = closest, 1 = farthest)
            normalized_z = (landmark.z_raw - z_min) / z_range if z_range > 0 else 0.5

            # Calculate color based on depth
            # Close = bright cyan, Far = dark blue
            if normalized_z < 0.5:
                # Close to camera: cyan to white
                point_intensity = 1 - normalized_z * 2  # 1 to 0 as depth increases
                color = (
                    int(255 * (1 - point_intensity * 0.5)),  # B: 255 to 127
                    int(255 * (1 - point_intensity * 0.5)),  # G: 255 to 127
                    255,  # R: always 255
                )
            else:
                # Far from camera: blue to dark blue
                point_intensity = (normalized_z - 0.5) * 2  # 0 to 1 as depth increases
                color = (
                    int(127 * (1 - point_intensity * 0.7)),  # B: 127 to 38
                    int(127 * (1 - point_intensity)),  # G: 127 to 0
                    int(127 * (1 - point_intensity)),  # R: 127 to 0
                )

            # Draw landmark circle
            radius = (
                8
                if idx
                in [
                    HandLandmark.WRIST,
                    HandLandmark.THUMB_TIP,
                    HandLandmark.INDEX_FINGER_TIP,
                    HandLandmark.MIDDLE_FINGER_TIP,
                    HandLandmark.RING_FINGER_TIP,
                    HandLandmark.PINKY_TIP,
                ]
                else 6
            )
            cv2.circle(frame, (landmark.x, landmark.y), radius, color, -1, cv2.LINE_AA)

            # Add white outline for better visibility
            cv2.circle(frame, (landmark.x, landmark.y), radius, (255, 255, 255), 1, cv2.LINE_AA)

            # Add depth value text for fingertips and wrist
            if idx in [
                HandLandmark.WRIST,
                HandLandmark.THUMB_TIP,
                HandLandmark.INDEX_FINGER_TIP,
                HandLandmark.MIDDLE_FINGER_TIP,
                HandLandmark.RING_FINGER_TIP,
                HandLandmark.PINKY_TIP,
            ]:
                depth_text = f"{landmark.z_raw:.4f}"
                cv2.putText(
                    frame,
                    depth_text,
                    (landmark.x + 10, landmark.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return frame


@app.command(name="depth-viz")
def depth_viz_cmd(
    camera: str | None = options.camera,
    preview: bool = options.preview,
    mirror: bool | None = options.mirror,
    size: int | None = options.size,
    config_path: Path | None = options.config,  # noqa: B008
    gpu: bool | None = options.gpu,
) -> None:
    """Visualize hand landmarks with depth information from Z-axis."""
    # Load configuration
    config = Config.load(config_path)

    # Determine GPU usage (now with config)
    use_gpu = determine_gpu_usage(gpu, config)

    # Determine mirror mode (now with config)
    use_mirror = determine_mirror_mode(mirror, config)

    # Use config values as defaults, but CLI options take precedence
    final_camera = camera if camera is not None else config.cli.camera
    final_size = size if size is not None else config.cli.size

    selected = pick_camera(final_camera)

    if selected:
        print(f"\nSelected: {selected}")
        visualize_depth(
            selected,
            show_preview=preview,
            config=config,
            mirror=use_mirror,
            desired_size=final_size,
            use_gpu=use_gpu,
        )
    else:
        print("\nNo camera selected.")
