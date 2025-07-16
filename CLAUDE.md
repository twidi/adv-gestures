# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time hand gesture recognition Python package called "adv-gestures". It uses MediaPipe for hand tracking and OpenCV for video capture/display to detect and analyze hand gestures from camera input.

## Installation and Running

The project is a proper Python package that should be installed first:

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

After installation, run the application using the `adv-gestures` command:

```bash
# Basic usage - auto-select camera
adv-gestures

# Filter cameras by name
adv-gestures "webcam"

# Preview mode (show available cameras)
adv-gestures --preview

# Check mode (run without displaying video)
adv-gestures --check
```

## Dependencies

The project uses `pyproject.toml` to manage dependencies. Key requirements:
- Python >= 3.11
- typer, linuxpy, opencv-python, mediapipe, numpy, typing-extensions

Development dependencies include mypy, black, isort, and ruff for code quality.

**Important**: The MediaPipe model file `gesture_recognizer.task` will be automatically downloaded on first run if not present.

## Architecture

The codebase is structured as a Python package in `src/adv_gestures/__init__.py` with these key components:

- **Camera Management**: Uses `linuxpy` to enumerate and select Linux camera devices
- **Hand Tracking**: MediaPipe-based detection of hand landmarks for both hands
- **Gesture Recognition**: 
  - Built-in MediaPipe gestures (Closed Fist, Open Palm, Pointing Up, Thumb Up/Down, Victory, ILoveYou)
  - Custom gestures detected through finger analysis (Middle Finger, Spock, Rock, OK, Stop, Pinch, Gun, Finger Gun)
- **Data Classes**:
  - `Hands`: Container for both hands
  - `Hand`: Individual hand with palm, fingers, and gesture state
  - `Palm`: Palm landmarks and orientation analysis
  - `Finger`: Finger tracking with straightness, touching, and direction analysis

## Key Features to Understand

1. **Finger Straightness Detection**: Each finger has configurable straightness thresholds (recently updated to per-finger values)
2. **Touch Detection**: Detects when fingers touch each other or when thumb touches other fingers
3. **Custom Gesture Logic**: Located in `Hand._analyze_gesture()` method, uses finger states and positions
4. **Real-time Visualization**: Overlays hand landmarks, finger states, and detected gestures on video feed

## Development Notes

- No formal testing framework is set up - test by running the application
- The project has a Makefile with linting and formatting tools
- The project uses type hints throughout - maintain type annotations when adding code
- Camera handling is Linux-specific due to `linuxpy` dependency
- FPS is set to 30 and codec to MJPG for optimal performance

## Code Quality

**Important**: At the end of each conversation, run `make pretty lint` to ensure the code stays clean and properly formatted.

## Type Checking Guidelines

- Never tell mypy to ignore a file. Try to fix mypy errors when running mypy.
- At maximum, use `# type: ignore[reason]` for specific lines that cannot be resolved

## Documentation Maintenance

**Important**: When making significant changes to the project:
- Update README.md to reflect new features, usage instructions, or dependencies
- Update CLAUDE.md to document new architectural decisions, development patterns, or important notes for future development
- Keep both files accurate and in sync with the current state of the project

## Common Tasks

### Adding New Gestures
1. Define gesture logic in `Hand._analyze_gesture()` method
2. Use existing finger state properties: `is_straight`, `is_nearly_straight`, `touches()`, `thumb_touches()`
3. Add the gesture name to be displayed when detected

### Debugging Gesture Detection
- Use the visual overlay to see finger states and landmarks
- Check the `is_straight` and `is_nearly_straight` properties for finger positions
- The straightness thresholds can be adjusted in the `Finger` class initialization

### Camera Issues
- Use `--preview` flag to list available cameras
- Filter cameras by name if multiple are present
- The application requires Linux due to camera enumeration method

## Project Guidelines

- All code and comments or any text (readme, etc) must be in English
