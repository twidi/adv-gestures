# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python 3.11+ hand gesture recognition application that uses MediaPipe and OpenCV for real-time hand tracking and gesture detection. The project is packaged as `adv-gestures` and provides both a library and CLI tool.

## Development Commands

### Installation
- `make install` - Install the project with dependencies
- `make dev` - Install in development mode with all dev dependencies

### Code Quality (IMPORTANT: Run these before completing any task)
- `make pretty lint` - Run ALL prettifiers et linters 

### Running the Application (Development/testing interface)
- `adv-gestures` - Run gesture recognition (default command, requires camera access)
- `adv-gestures --camera "name"` - Run gesture recognition with specific camera (or `--cam`)
- `adv-gestures check-camera` - Check camera functionality without gesture recognition
- `adv-gestures check-camera --camera "name"` - Check specific camera
- Both commands support `--config`, `--mirror`, `--size` options
- The app will prompt for camera selection if multiple cameras are available when --camera is not specified

### Cleaning
- `make clean` - Clean build artifacts
- `make full-clean` - Deep clean including all caches

## Architecture Overview

### Core Components

1. **Recognizer** (`src/adv_gestures/recognizer.py`): Central class wrapping MediaPipe's gesture recognition. Handles:
   - Async frame processing with callbacks
   - Result storage and hand model updates
   - Performance metrics (FPS, latency)

2. **Models** (`src/adv_gestures/models/`): Type-safe data structures
   - `fingers.py`: Finger representations with landmark positions
   - `landmarks.py`: Hand landmark definitions
   - `hands/` package: Refactored hand models
     - `base_gestures.py`: Unified base class for all gesture detectors (stateless and stateful)
     - `hand.py`: Single hand representation with finger tracking
     - `hand_gestures.py`: Single hand gesture detectors (AIR_TAP, WAVE, etc.)
     - `hands.py`: Collection of hands
     - `hands_gestures.py`: Two-hands gesture detectors (PRAY, CLAP)
     - `palm.py`: Palm-related functionality
     - `utils.py`: Hand utility functions

3. **Gestures** (`src/adv_gestures/gestures.py`): Gesture definitions
   - Gesture enums (both MediaPipe built-in and custom)
   - DEFAULT_GESTURES and CUSTOM_GESTURES definitions
   - TWO_HANDS_GESTURES for gestures requiring both hands
   - Gesture detection thresholds and configurations

4. **Smoothing System** (`src/adv_gestures/smoothing.py`): Sophisticated EMA-based smoothing
   - Decorators: `@smoothed_bool`, `@smoothed_float`, `@smoothed_optional_float`, `@smoothed_coord`
   - Multiple smoother types for different data
   - Configurable time windows and smoothing factors

5. **CLI** (`src/adv_gestures/cli/`): Development/testing interface organized into modules
   - `run.py`: Main gesture recognition runner with real-time visualization
   - `check_camera.py`: Camera checking and preview functionality
   - `tweak.py`: Configuration tweaking interface
   - `common.py`: Shared CLI utilities (camera selection, config loading)
   - Config file support via `--config` option

6. **Configuration** (`src/adv_gestures/config.py`): Pydantic-based configuration
   - JSON-based configuration files
   - Default config location: user config directory via platformdirs
   - Nested configuration for hands, fingers, and gesture detection thresholds
   - CLI configuration options: camera filter, mirror mode, and capture size

7. **MediaPipe Wrapper** (`src/adv_gestures/mediapipe.py`): Centralized MediaPipe imports
   - Single import point for all MediaPipe components
   - Simplifies imports across the codebase

8. **Camera Management** (`src/adv_gestures/cameras.py`): Camera-related functionality
   - Camera enumeration and selection
   - Linux-specific camera handling via `linuxpy`

### Key Patterns

- **Streaming Architecture**: Async callbacks for real-time processing
- **Property-based Smoothing**: Decorators that provide both `.raw` and smoothed values
- **Type Safety**: Extensive use of type hints, generics, and protocols
- **Modular Gestures**: Easy extension with custom gesture definitions

### Important Implementation Notes

1. **Coordinate Systems**: 
   - MediaPipe provides normalized coordinates (0-1)
   - Drawing functions convert to pixel coordinates
   - Bounding boxes use pixel coordinates

2. **Gesture Detection Flow**:
   - MediaPipe detects basic gestures (single gesture)
   - Custom gestures detected via gesture detector classes (multiple simultaneous)
   - All detected gestures are tracked with individual stability weights and durations
   - Override mechanism for incorrect MediaPipe detections
   - Each gesture smoothed independently for stability
   - Gesture weights normalized with strongest gesture at 1.0
   - Gesture detectors organized as classes inheriting from `BaseGestureDetector`
   - Two-hands gestures require both hands to be detected and properly positioned
   - Stateful gestures (e.g., CLAP) controlled by `stateful` class attribute
   - BaseGestureDetector provides unified functionality:
     - Automatic registration of gesture detectors via `__init_subclass__`
     - Pre-matching checks with `main_direction_range`
     - Stateful gesture support with `min_gesture_duration`, `max_gesture_duration`, `post_detection_duration`
     - Direction matching utilities for hands and fingers
   - HandGesturesDetector extends BaseGestureDetector for single-hand gestures
   - TwoHandsGesturesDetector extends BaseGestureDetector for two-hands gestures

3. **Performance Considerations**:
   - Smoothed properties cached per frame
   - Lazy initialization of smoothers
   - Optional GPU acceleration via MediaPipe

4. **Linux-specific Features**: Camera handling is in `cameras.py` module

## Code Style Requirements

- **Type Hints**: ALWAYS use type hints (project uses `mypy --strict`). Use `list` and not `List`. Same for `dict`, `set`, etc.
- **Formatting**: Code must pass `black` and `isort` formatting (`make pretty` command)
- **Linting**: Must pass `ruff` checks (`make lint` command)
- **No Comments**: Do not add useless comments. Especially no comments explaining what changed.
- **Follow Existing Patterns**: Match the style of surrounding code

## Claude Guidance
- When working with README.md and CLAUDE.md files, the content should reflect the CURRENT state of the project, not a narrative of changes. 
  - Avoid phrases like "We added this" or "This was refactored"
  - Focus on describing the project's current structure, features, and capabilities as they exist RIGHT NOW

## Technical Considerations
- On z-axis coordinate handling:
  - We cannot reliably use the z-axis because MediaPipe landmark z-values are not consistently reliable