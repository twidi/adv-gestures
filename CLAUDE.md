# CLAUDE.md

AI assistant instructions for working with the adv-gestures codebase.

## Project Overview

Python 3.11+ hand gesture recognition application using MediaPipe and OpenCV for real-time hand tracking and gesture detection. The project is packaged as `adv-gestures` and provides both a library and CLI tool.

For detailed documentation, architecture, and usage examples, refer to [README.md](README.md).

## Development Workflow

### Essential Commands

```bash
# Installation (uses uv for fast dependency management)
make dev       # Development mode with all dependencies
make install   # Production dependencies only

# Code quality - MUST run before completing any Python task
make pretty lint  # Format and lint (Python changes only)

# Cleaning
make clean       # Clean build artifacts
make full-clean  # Deep clean including caches
```

### Quick Testing Commands
```bash
adv-gestures                    # Run gesture recognition
adv-gestures check-camera       # Test camera without recognition
adv-gestures tweak             # Interactive config editor
adv-gestures playground --open  # Web interface with apps
```

## Code Requirements

### Type Hints (MANDATORY)
- **Always** use type hints (project uses `mypy --strict`)
- Use `list` not `List`, `dict` not `Dict`, etc.
- All `to_dict()` methods must return `dict[str, Any]`

### Code Style
- No useless comments, especially not "what changed" comments
- Follow existing patterns in surrounding code
- Never import inside functions (except for circular imports)
- Run `make pretty lint` before completing tasks (Python changes only)

## Architecture Summary

### Core Components
- **Recognizer**: MediaPipe wrapper with async processing
- **Models**: Type-safe hand/finger/landmark representations
- **Gesture Detectors**: Classes inheriting from `BaseGestureDetector`
- **Smoothing**: Time-based EMA decorators (`.raw` and smoothed values)
- **CLI**: Commands for running, testing, tweaking, playground
- **Playground**: WebRTC + SSE web interface with multiple apps

### Gesture Detection System
- All detectors inherit from `BaseGestureDetector`
- Auto-registration via `__init_subclass__`
- Multiple simultaneous gestures with normalized weights (strongest = 1.0)
- Stateful gestures use duration controls
- Two-hands gestures require both hands detected

See README.md for full architecture details.

## Special Instructions

### Voice Dictation Corrections
The user may use voice dictation with these common mistakes:
- "cloud"/"cloude" → CLAUDE.md file
- "RTAP" → AIR_TAP gesture  
- "type" (about fingers) → finger tip
- "trou" (French) → `true` boolean
- camelCase/spaces in names → convert to snake_case (Python) or camelCase (JS)

### to_dict Methods
When modifying model properties:
- **MUST** update corresponding `to_dict()` method
- Points use `{x, y}` objects, not tuples/arrays
- Enums sent as uppercase strings (e.g., `THUMB`, `INDEX`)
- Check existing `to_dict()` methods for exact structure

### Playground Applications
New app requirements:
- Single JavaScript file in `src/adv_gestures/cli/playground/static/apps/`
- Import in `static/index.html`
- Extend from `_base.js` base class

### Font Awesome Icons
- **NEVER** invent SVG paths
- Get actual SVG from: `https://site-assets.fontawesome.com/releases/v7.0.0/svgs-full/regular/{icon-name}.svg`. FILES ARE **NEVER** TRUNCATED. IF YOU THINK THEY ARE, GET THE WHOLE SVG AND YOU'LL SEE.
- Ask user if unsure which icon to use

### JavaScript Best Practices
- Use `classList` for class manipulation, not style attributes
- Use CSS custom properties for numeric values
- Pure JavaScript only (no build step)

## Technical Constraints

### Z-axis Coordinates
Cannot reliably use MediaPipe z-values - they are inconsistent

### Math Operations  
Use `math` module, not NumPy (unless working with NumPy arrays):
```python
from math import degrees, atan2, acos  # NOT numpy
```

### Documentation Language
- Everything in English only
- Describe CURRENT state, not changes history
- Avoid "We added", "This was refactored", etc.

## Playground Data Format

Server sends `hands.to_dict()` via SSE:
- Points: `{x: float, y: float}` objects
- Bounding boxes: `{top_left: {x, y}, bottom_right: {x, y}}`
- Finger names: uppercase strings (`THUMB`, `INDEX`, etc.)
- Gestures: dict with weights and durations
- All processing happens client-side

See individual model classes for exact `to_dict()` structure.
