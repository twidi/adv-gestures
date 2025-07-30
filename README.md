# adv-gestures

A Python library and CLI tool for real-time hand gesture recognition using MediaPipe and OpenCV.

## Features

- Real-time hand tracking and gesture detection
- Built-in gesture recognition (peace, thumbs up, etc.)
- Custom gesture detection system
- Sophisticated smoothing for stable detection
- Performance metrics (FPS, latency monitoring)
- Multi-camera support with automatic enumeration
- Type-safe implementation with strict typing

## Installation

### From PyPI (when available)
```bash
pip install adv-gestures
```

### Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd gestures

# Install in development mode
make dev
```

### Playground Installation (Optional)
The web-based playground requires additional dependencies:
```bash
# Install with playground support
pip install "adv-gestures[playground]"
```

## Quick Start

### CLI Usage (Development/testing interface)
```bash
# Run gesture recognition with preview (default)
adv-gestures

# Run without preview window
adv-gestures --no-preview

# Run with JSON output (outputs hands.to_dict() as JSON for each frame)
adv-gestures --json

# Run with JSON output and no preview (JSON only)
adv-gestures --no-preview --json

# Run with custom config file
adv-gestures --config /path/to/config.json

# Run with specific camera
adv-gestures --camera "webcam"
# or short form
adv-gestures --cam "webcam"

# Run with mirror mode
adv-gestures --mirror

# Run with custom size (max dimension)
adv-gestures --size 1920
# or short form
adv-gestures -s 800

# Combine multiple options
adv-gestures --camera "webcam" --mirror --size 1600 --json

# Check camera functionality without gesture recognition
adv-gestures check-camera

# Check specific camera
adv-gestures check-camera --camera "webcam"

# Check camera with preview disabled
adv-gestures check-camera --no-preview

# Check with mirror and custom size
adv-gestures check-camera --mirror --size 1920

# Check with custom config file
adv-gestures check-camera --config /path/to/config.json

# The tool will prompt for camera selection if multiple cameras are available when --camera is not specified
```

### Playground (Web Interface)
The library includes a web-based playground for real-time gesture recognition:

```bash
# Start the playground server
adv-gestures playground

# Start and open in browser
adv-gestures playground --open

# Custom host and port
adv-gestures playground --host 0.0.0.0 --port 8080
```

The playground provides:
- Real-time hand tracking visualization
- Camera selection interface
- Mirror mode toggle (persisted)
- Debug overlays for hand bounding boxes
- Live gesture detection logs
- WebRTC video streaming with SSE data channel

Note: Playground dependencies must be installed separately with `pip install "adv-gestures[playground]"`

### Configuration File

The application supports a JSON configuration file that can include CLI defaults:

```json
{
  "cli": {
    "camera": "webcam",
    "mirror": true,
    "size": 1920
  },
  "hands": {
    // ... hand configuration options
  }
}
```

CLI options always take precedence over configuration file values. The default config location is platform-specific and will be shown if the config file is not found.

### Library Usage
```python
import cv2
from adv_gestures import Recognizer, Hands, Gestures, Config

# Initialize recognizer with model path using context manager
with Recognizer("gesture_recognizer.task") as recognizer:
    hands = Hands(config=Config())  # Use default config
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    try:
        # Process frames
        for frame, stream_info, result in recognizer.handle_opencv_capture(cap, hands):
            # Check detected gestures
            for hand in hands:
                # Access all detected gestures with stability weights
                for gesture, weight in hand.gestures.items():
                    duration = hand.gestures_durations.get(gesture, 0)
                    print(f"{gesture}: weight={weight:.2f}, duration={duration:.1f}s")

                if Gestures.VICTORY in hand.gestures:
                    print("Victory/Peace sign detected!")
                elif Gestures.THUMB_UP in hand.gestures:
                    print("Thumbs up!")
            
            # Display frame (optional)
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
```

## Supported Gestures

### Built-in MediaPipe Gestures
- `CLOSED_FIST` - Closed hand
- `OPEN_PALM` - Open hand facing camera
- `POINTING_UP` - Index finger pointing up
- `THUMB_DOWN` - Thumb pointing down
- `THUMB_UP` - Thumb pointing up
- `VICTORY` - Peace/Victory sign
- `LOVE` - I Love You sign

### Custom Gestures
- `MIDDLE_FINGER` - Middle finger gesture
- `SPOCK` - Vulcan salute
- `ROCK` - Rock gesture (index and pinky up)
- `OK` - OK sign
- `STOP` - Stop gesture
- `PINCH` - Thumb and index finger pinching
- `PINCH_TOUCH` - Pinch + index and thumb touching
- `GUN` - Gun gesture
- `FINGER_GUN` - Finger gun (without middle finger)
- `AIR_TAP` - Index finger held straight and still for a certain delay ("tap_position" available in gesture data)
- `PRE_AIR_TAP` - Index finger held straight and still before an air tap ("tap_position" available in gesture data)
- `WAVE` - Open palm waving left-right motion
- `SNAP` - Finger snapping
- `SWIPE` - Swipe in any direction by hand or index finger ("direction" (left/right) and "mode" (hand/index) available in gesture data)
- `NO` - Index finger waving left-right with other fingers bent

### Two-Hands Gestures
- `PRAY` - Both hands in a prayer position, palms together, fingers pointing up
- `CLAP` - Hands joined briefly (less than 1 second) then separated
- `CROSSED_FLAT` - Both hands crossed with fingers straight
- `CROSSED_FISTS` - Both hands crossed with fingers in fist position
- `TIME_OUT` - Two hands forming a T shape, perpendicular
- `FRAME` - Photo frame with thumbs and index fingers forming L shapes

## Multiple Gesture Detection

The library supports detecting multiple gestures simultaneously. This is useful when gestures overlap or when you want to track gesture combinations.

- Each detected gesture has a weight between 0.0 and 1.0
- Weights are normalized so the strongest gesture always has weight 1.0
- Weights indicate stability over time
- Custom gestures are always detected regardless of default gesture detection
- Two-hands gestures (like PRAY) require both hands to be detected

## Data Export

All tracking data can be easily exported as dictionaries for integration with other systems:

```python
# Get all hand tracking data in a single call
data = hands.to_dict()

# This returns a complete hierarchy including:
# - Both hands (left/right) with visibility status
# - All landmarks with pixel and normalized coordinates
# - Finger positions, angles, and touch states
# - Detected gestures with weights and durations
# - Palm centroids and hand orientations
# - Bounding boxes and directional information
# - Stream information (dimensions, FPS)
```

The `to_dict()` method is available on all model classes (Hands, Hand, Finger, Palm, Landmark, etc.) and returns data suitable for JSON serialization.

### JSON Output Mode

You can use the `--json` flag with the CLI to output tracking data as JSON:

```bash
# Output JSON with preview window
adv-gestures --json

# Output JSON only (no preview, no other console output)
adv-gestures --no-preview --json
```

This outputs one JSON object per frame, containing the complete hand tracking data from `hands.to_dict()`. This is useful for:
- Piping data to other programs
- Recording sessions for later analysis
- Real-time integration with external systems

## Architecture

### Core Components

- **Recognizer**: Central processing engine with async frame handling
- **Hand Model**: Tracks hand state, fingers, and detected gestures
- **Smoothing System**: EMA-based smoothing for stable detection
- **CLI Interface**: Development and testing tool with visualization

### Key Features

- **Streaming Architecture**: Async callbacks for real-time processing
- **Property-based Smoothing**: Automatic smoothing with raw value access
- **Type Safety**: Full type hints with mypy strict mode
- **Extensible Design**: Easy to add custom gestures

## Development

### Setup Development Environment
```bash
make install  # Basic installation
make dev      # Development installation with all tools
```

### Code Quality
```bash
make pretty lint  # Format and lint code
```

### Project Structure
```
adv-gestures/
├── src/adv_gestures/
│   ├── __init__.py      # Package initialization
│   ├── __main__.py      # Entry point for python -m adv_gestures
│   ├── cameras.py       # Camera enumeration and management
│   ├── config.py        # Configuration management
│   ├── drawing.py       # Visualization and drawing utilities
│   ├── gestures.py      # Gesture definitions and enums
│   ├── mediapipe.py     # MediaPipe wrapper and imports
│   ├── recognizer.py    # Core recognition engine
│   ├── smoothing.py     # Smoothing decorators
│   ├── cli/             # CLI interface modules
│   │   ├── __init__.py  # CLI package initialization
│   │   ├── check_camera.py  # Camera checking functionality
│   │   ├── common.py    # Shared CLI utilities
│   │   ├── run.py       # Main gesture recognition runner
│   │   ├── tweak.py     # Configuration tweaking interface
│   │   └── playground/  # Playground web server
│   │       ├── __init__.py  # Playground command
│   │       ├── server.py    # WebRTC/SSE server
│   │       └── static/      # Client-side files
│   │           ├── index.html
│   │           ├── main.js
│   │           └── styles.css
│   └── models/          # Data models
│       ├── __init__.py  # Models package initialization
│       ├── fingers.py   # Finger tracking
│       ├── landmarks.py # MediaPipe landmarks
│       ├── utils.py     # Model utility functions
│       └── hands/       # Hand models package
│           ├── __init__.py  # Hands package initialization
│           ├── hand.py      # Single hand representation
│           ├── base_gestures.py  # Base classes for gesture detectors
│           ├── hand_gestures.py  # Single hand gesture detectors
│           ├── hands.py     # Hands collection
│           ├── hands_gestures.py # Two-hands gesture detectors
│           └── palm.py      # Palm-related functionality
├── Makefile            # Development commands
└── pyproject.toml      # Project configuration
```

## Requirements

- Python 3.11+
- Camera/webcam for real-time detection
- Linux (for camera enumeration features)

## Performance

The system includes built-in performance monitoring:
- Real-time FPS tracking
- Processing latency measurement
- Configurable smoothing parameters for accuracy vs responsiveness trade-off

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure code passes all quality checks:
```bash
make pretty lint
```
