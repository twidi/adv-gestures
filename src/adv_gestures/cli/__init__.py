#!/usr/bin/env python3

"""The CLI is for development and debugging purpose."""


from .check_camera import check_camera_cmd  # noqa: F401
from .common import app
from .depth_viz import depth_viz_cmd  # noqa: F401
from .playground import playground_cmd  # noqa: F401
from .run import run_gestures_cmd  # noqa: F401
from .tweak import tweak_cmd  # noqa: F401


def main() -> None:
    """Entry point for the application."""
    app()


if __name__ == "__main__":
    main()
