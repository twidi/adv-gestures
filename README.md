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

## Quick Start

### CLI Usage (Development/testing interface)
```bash
# Run gesture recognition with preview (default)
adv-gestures

# Run without preview window
adv-gestures --no-preview

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
adv-gestures --camera "webcam" --mirror --size 1600

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
                if hand.gesture == Gestures.VICTORY:
                    print("Victory/Peace sign detected!")
                elif hand.gesture == Gestures.THUMB_UP:
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
│   ├── cameras.py       # Camera enumeration and management
│   ├── cli.py           # CLI application
│   ├── config.py        # Configuration management
│   ├── drawing.py       # Visualization and drawing utilities
│   ├── mediapipe.py     # MediaPipe wrapper and utilities
│   ├── recognizer.py    # Core recognition engine
│   ├── smoothing.py     # Smoothing decorators
│   └── models/          # Data models
│       ├── __init__.py  # Models package initialization
│       ├── hands.py     # Hand representation
│       ├── fingers.py   # Finger tracking
│       ├── gestures.py  # Gesture definitions
│       └── landmarks.py # MediaPipe landmarks
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
