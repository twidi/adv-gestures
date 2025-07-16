# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time hand gesture recognition application built in Python. It uses MediaPipe for hand tracking and OpenCV for video capture/display to detect and analyze hand gestures from camera input.

## Running the Application

The application is a single Python file that can be run directly:

```bash
# Basic usage - auto-select camera
python gestures.py

# Filter cameras by name
python gestures.py "webcam"

# Preview mode (show available cameras)
python gestures.py --preview

# Check mode (run without displaying video)
python gestures.py --check
```

## Dependencies

Install required packages:
```bash
pip install typer linuxpy opencv-python mediapipe numpy typing-extensions
```

**Important**: You also need the MediaPipe model file `gesture_recognizer.task` in the project root. This file is not included in the repository and must be downloaded separately.

## Architecture

The codebase is structured as a single-file application (`gestures.py`) with these key components:

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
- No linting configuration exists - maintain consistent code style with existing code
- The project uses type hints throughout - maintain type annotations when adding code
- Camera handling is Linux-specific due to `linuxpy` dependency
- FPS is set to 30 and codec to MJPG for optimal performance

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
