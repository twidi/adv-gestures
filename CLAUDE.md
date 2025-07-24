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
   - `hands.py`: Hand class with finger tracking and gesture detection
   - `fingers.py`: Finger representations with landmark positions
   - `gestures.py`: Gesture enums (both MediaPipe built-in and custom)
   - `landmarks.py`: Hand landmark definitions

3. **Smoothing System** (`src/adv_gestures/smoothing.py`): Sophisticated EMA-based smoothing
   - Decorators: `@smoothed_bool`, `@smoothed_float`, `@smoothed_optional_float`, `@smoothed_coord`
   - Multiple smoother types for different data
   - Configurable time windows and smoothing factors

4. **CLI** (`src/adv_gestures/cli.py`): Development/testing interface
   - Camera selection and preview
   - Real-time visualization with metrics
   - Debug drawing overlays
   - Config file support via `--config` option

5. **Configuration** (`src/adv_gestures/config.py`): Pydantic-based configuration
   - JSON-based configuration files
   - Default config location: user config directory via platformdirs
   - Nested configuration for hands, fingers, and gesture detection thresholds
   - CLI configuration options: camera filter, mirror mode, and capture size

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
   - MediaPipe detects basic gestures
   - Custom gestures detected via finger positions
   - Override mechanism for incorrect MediaPipe detections
   - All values smoothed for stability

3. **Performance Considerations**:
   - Smoothed properties cached per frame
   - Lazy initialization of smoothers
   - Optional GPU acceleration via MediaPipe

4. **Linux-specific Features**: Uses `linuxpy` for camera enumeration

## Code Style Requirements

- **Type Hints**: ALWAYS use type hints (project uses `mypy --strict`). Use `list` and not `List`. Same for `dict`, `set`, etc.
- **Formatting**: Code must pass `black` and `isort` formatting (`make pretty` command)
- **Linting**: Must pass `ruff` checks (`make lint` command)
- **No Comments**: Do not add useless comments. Especially no comments explaining what changed.
- **Follow Existing Patterns**: Match the style of surrounding code
