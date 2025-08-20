from __future__ import annotations

import os
import sys

import cv2  # type: ignore[import-untyped]
import typer

from ..cameras import CameraInfo, list_cameras
from ..config import Config

app = typer.Typer()

DEFAULT_USER_CONFIG_PATH = Config.get_user_path()


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


def init_camera_capture(
    camera_info: CameraInfo, show_preview: bool, desired_size: int
) -> tuple[cv2.VideoCapture | None, str | None]:
    """Initialize camera capture and set resolution."""
    cap = cv2.VideoCapture(camera_info.device_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_info.device_index}", file=sys.stderr)
        return None, None

    # Calculate dimensions based on desired_size while maintaining aspect ratio
    aspect_ratio = camera_info.width / camera_info.height
    if camera_info.width > camera_info.height:
        width = desired_size
        height = int(desired_size / aspect_ratio)
    else:
        height = desired_size
        width = int(desired_size * aspect_ratio)

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


def determine_gpu_usage(gpu: bool, no_gpu: bool, config: Config | None = None) -> bool:
    """Determine whether to use GPU based on CLI arguments, environment variable, and config.

    Priority order:
    1. CLI arguments (--gpu / --no-gpu)
    2. Environment variable (GESTURE_RECOGNIZER_USE_GPU)
    3. Config file (config.cli.use_gpu)
    4. Default (True)

    Args:
        gpu: Whether --gpu flag was specified
        no_gpu: Whether --no-gpu flag was specified
        config: Optional Config object with CLI settings

    Returns:
        bool: Whether to use GPU

    Raises:
        typer.Exit: If both --gpu and --no-gpu are specified
    """
    # Check for conflicting arguments
    if gpu and no_gpu:
        print("Error: Cannot specify both --gpu and --no-gpu")
        raise typer.Exit(1)

    # Priority 1: CLI arguments
    if gpu:
        use_gpu = True
    elif no_gpu:
        use_gpu = False
    else:
        # Priority 2: Environment variable
        env_gpu = os.getenv("GESTURE_RECOGNIZER_USE_GPU", "").strip().lower()
        if env_gpu in ("false", "0", "no"):
            use_gpu = False
        elif env_gpu in ("true", "1", "yes"):
            use_gpu = True
        # Priority 3: Config file
        elif config is not None:
            use_gpu = config.cli.use_gpu
        # Priority 4: Default
        else:
            use_gpu = False

    # Print status message
    if use_gpu:
        print("Using GPU acceleration (may fall back to CPU if GPU is unavailable)")
    else:
        print("Using CPU processing")

    return use_gpu


def determine_mirror_mode(mirror: bool, no_mirror: bool, config: Config | None = None) -> bool:
    """Determine whether to use mirror mode based on CLI arguments, environment variable, and config.

    Priority order:
    1. CLI arguments (--mirror / --no-mirror)
    2. Environment variable (GESTURE_RECOGNIZER_MIRROR)
    3. Config file (config.cli.mirror)
    4. Default (True)

    Args:
        mirror: Whether --mirror flag was specified
        no_mirror: Whether --no-mirror flag was specified
        config: Optional Config object with CLI settings

    Returns:
        bool: Whether to use mirror mode

    Raises:
        typer.Exit: If both --mirror and --no-mirror are specified
    """
    # Check for conflicting arguments
    if mirror and no_mirror:
        print("Error: Cannot specify both --mirror and --no-mirror")
        raise typer.Exit(1)

    # Priority 1: CLI arguments
    if mirror:
        use_mirror = True
    elif no_mirror:
        use_mirror = False
    else:
        # Priority 2: Environment variable
        env_mirror = os.getenv("GESTURE_RECOGNIZER_MIRROR", "").strip().lower()
        if env_mirror in ("false", "0", "no"):
            use_mirror = False
        elif env_mirror in ("true", "1", "yes"):
            use_mirror = True
        # Priority 3: Config file
        elif config is not None:
            use_mirror = config.cli.mirror
        # Priority 4: Default
        else:
            use_mirror = True

    # Print status message
    if use_mirror:
        print("Mirror mode enabled (video output will be horizontally flipped)")
    else:
        print("Mirror mode disabled")

    return use_mirror
