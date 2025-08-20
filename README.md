# adv-gestures

A Python library and CLI tool for real-time hand gesture recognition using MediaPipe and OpenCV.

## Features

- **Real-time hand tracking** with MediaPipe-based detection
- **30+ gestures supported** including single-hand, two-hands, and stateful gestures
- **Sophisticated smoothing system** for stable detection with time-based EMA
- **Web-based playground** with interactive applications (Pong, Theremin, Drawing)
- **Performance optimized** with GPU acceleration support and FPS monitoring
- **Multi-camera support** with automatic enumeration and selection
- **JSON export** for integration with external systems
- **Type-safe implementation** with strict typing (mypy --strict)

## Installation

### From PyPI
```bash
pip install adv-gestures
```

### With Playground Support
```bash
# Includes web server and WebRTC dependencies
pip install "adv-gestures[playground]"
```

### Development Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd gestures

# Install in development mode with all dependencies (uses uv internally)
make dev
```

## Quick Start

### CLI Commands

#### Main Gesture Recognition
```bash
# Run with default settings (preview enabled)
adv-gestures

# All available options:
adv-gestures --camera "webcam"     # or --cam, -cm (filter cameras by name)
adv-gestures --preview             # or -p (show preview window, default)
adv-gestures --no-preview          # or -np (disable preview window)
adv-gestures --mirror              # or -m (mirror video horizontally)
adv-gestures --no-mirror           # or -nm (disable mirroring)
adv-gestures --size 1920           # or -s (max capture dimension)
adv-gestures --config /path/to/config.json  # or -c (custom config file)
adv-gestures --gpu                 # or -g (enable GPU acceleration)
adv-gestures --no-gpu              # or -ng (disable GPU, use CPU)
adv-gestures --json                # Output tracking data as JSON

# Common combinations:
adv-gestures --no-preview --json   # JSON output only, no GUI
adv-gestures --camera "webcam" --mirror --size 1600 --gpu
```

#### Camera Testing
```bash
# Check camera functionality (without gesture recognition)
adv-gestures check-camera

# All available options:
adv-gestures check-camera --camera "webcam"  # or --cam, -cm
adv-gestures check-camera --preview          # or -p (default)
adv-gestures check-camera --no-preview       # or -np
adv-gestures check-camera --mirror           # or -m
adv-gestures check-camera --no-mirror        # or -nm
adv-gestures check-camera --size 1920        # or -s
adv-gestures check-camera --config /path/to/config.json  # or -c
```

#### Interactive Configuration Tweaking
```bash
# Launch interactive configuration editor
adv-gestures tweak

# All available options:
adv-gestures tweak --camera "webcam"  # or --cam, -cm
adv-gestures tweak --mirror           # or -m
adv-gestures tweak --no-mirror        # or -nm
adv-gestures tweak --size 1920        # or -s
adv-gestures tweak --config /path/to/config.json  # or -c
adv-gestures tweak --gpu              # or -g
adv-gestures tweak --no-gpu           # or -ng
```

#### Depth Visualization
```bash
# Visualize depth information from hand tracking
adv-gestures depth-viz

# All available options:
adv-gestures depth-viz --camera "webcam"  # or --cam, -cm
adv-gestures depth-viz --mirror           # or -m
adv-gestures depth-viz --no-mirror        # or -nm
adv-gestures depth-viz --size 1920        # or -s
adv-gestures depth-viz --config /path/to/config.json  # or -c
adv-gestures depth-viz --gpu              # or -g
adv-gestures depth-viz --no-gpu           # or -ng
```

The CLI will prompt for camera selection if multiple cameras are available and `--camera` is not specified.

### Web Playground

The playground provides an interactive web interface with multiple applications:

```bash
# Start the playground server
adv-gestures playground

# All available options:
adv-gestures playground --host 127.0.0.1  # Server host (default: 127.0.0.1)
adv-gestures playground --port 9810       # Server port (default: 9810)
adv-gestures playground --open            # Open browser automatically
adv-gestures playground --mirror          # or -m
adv-gestures playground --no-mirror      # or -nm
adv-gestures playground --gpu            # or -g
adv-gestures playground --no-gpu         # or -ng
adv-gestures playground --config /path/to/config.json  # or -c

# Common usage:
adv-gestures playground --open
adv-gestures playground --host 0.0.0.0 --port 8080  # For network access
```

#### Available Applications

- **Default**: Basic hand tracking visualization with gesture detection
- **Debug**: Advanced debugging overlays with landmark details
- **Drawing**: Draw in the air using finger gestures
- **Pong**: Play pong using hand movements
- **Theremin**: Create music by moving your hands
- **3D Viewer**: Visualize hand landmarks in 3D space

#### Playground Features

- WebRTC video streaming for low latency
- Server-Sent Events (SSE) for real-time data
- Multi-user support with session isolation
- Camera selection and mirror mode
- No build step required (pure JavaScript)

**Note**: Requires `pip install "adv-gestures[playground]"`

### Configuration

#### Configuration File

**⚠️ WARNING**: It is strongly recommended to use the `adv-gestures tweak` command to modify configuration rather than editing the JSON file manually. The tweak command provides a safe, interactive interface with real-time preview.

JSON configuration with nested settings for all aspects of gesture detection:

```json
{
  "cli": {
    "camera": "webcam",
    "mirror": true,  // Default is true, mirrors video horizontally
    "size": 1920,
    "gpu": false     // Default is false, uses CPU for processing
  },
  "hands": {
    "smoothing_window": 0.5,
    "fingers": {
      "angle_smoothing_window": 0.3
    }
  },
  "gestures": {
    "default": {
      "enabled": true,
      "min_detection_confidence": 0.7
    },
    "custom": {
      "air_tap": {
        "enabled": true,
        "hold_duration": 0.5
      }
    }
  }
}
```

#### Environment Variables

```bash
# Model path (default: gesture_recognizer.task)
export GESTURE_RECOGNIZER_MODEL_PATH="/path/to/model.task"

# GPU acceleration (true/false, default: false)
export GESTURE_RECOGNIZER_USE_GPU="true"

# Mirror mode (true/false, default: true)
export GESTURE_RECOGNIZER_MIRROR="true"
```

**Priority**: CLI options > Environment variables > Config file > Defaults

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

### MediaPipe Built-in Gestures (7)
- `CLOSED_FIST` - Closed hand
- `OPEN_PALM` - Open hand facing camera  
- `POINTING_UP` - Index finger pointing up
- `THUMB_DOWN` - Thumb pointing down
- `THUMB_UP` - Thumb pointing up
- `VICTORY` - Peace/Victory sign
- `LOVE` - I Love You sign

### Custom Single-Hand Gestures (17)

#### Static Gestures
- `MIDDLE_FINGER` - Middle finger extended
- `SPOCK` - Vulcan salute
- `ROCK` - Rock gesture (index and pinky up)
- `OK` - OK sign (thumb and index circle)
- `STOP` - Stop gesture (open palm facing forward)
- `PINCH` - Thumb and index pinching
- `PINCH_TOUCH` - Pinch with fingers touching
- `GUN` - Gun gesture with thumb up
- `FINGER_GUN` - Finger gun (index extended)
- `NO` - Index finger wagging side to side

#### Stateful Gestures
- `AIR_TAP` - Index held still (includes `tap_position`)
- `PRE_AIR_TAP` - Pre-tap detection phase
- `WAVE` - Open palm waving motion
- `SNAP` - Finger snap detection
- `DOUBLE_SNAP` - Two snaps within 1 second (exit gesture in playground)
- `SWIPE` - Directional swipe (includes `direction` and `mode`)

### Two-Hands Gestures (6)
- `PRAY` - Prayer position with palms together
- `CLAP` - Clapping motion (stateful)
- `CROSSED_FLAT` - Crossed hands with open palms
- `CROSSED_FISTS` - Crossed hands with closed fists
- `TIME_OUT` - T-shape timeout gesture
- `FRAME` - Photo frame with L-shaped fingers

## Gesture Detection System

### Multiple Simultaneous Detection

The system can detect multiple gestures at once with weighted confidence:

- Each gesture has a weight (0.0 to 1.0) indicating detection strength
- Weights are normalized with the strongest at 1.0
- Time-based smoothing ensures stable detection
- Custom gestures can override incorrect MediaPipe detections

### Gesture Detector Architecture

All gestures are implemented as detector classes inheriting from `BaseGestureDetector`:

```python
# Access detected gestures
for hand in hands:
    for gesture, weight in hand.gestures.items():
        duration = hand.gestures_durations[gesture]
        data = hand.gestures_data.get(gesture, {})
```

**Detector Features**:
- Automatic registration via metaclass
- Pre-matching with `main_direction_range`
- Stateful support with duration controls
- Custom data in `gestures_data` dict

### Stateful vs Static Gestures

**Static**: Detected based on current hand pose
- Examples: THUMB_UP, OK, SPOCK

**Stateful**: Require temporal tracking
- Examples: WAVE (motion), SNAP (before/after), CLAP (contact/release)
- Controlled by `min_gesture_duration`, `max_gesture_duration`, `post_detection_duration`

## Data Export

All tracking data can be easily exported as dictionaries for integration with other systems:

```python
# Get all hand tracking data in a single call
data = hands.to_dict()

# This returns a complete hierarchy including:
# - Both hands (left/right) with visibility status
# - All landmarks with pixel, raw and world coordinates
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

- **Recognizer** (`recognizer.py`): MediaPipe wrapper with async processing
- **Models** (`models/`): Type-safe data structures
  - `Hand`: Single hand with fingers and landmarks
  - `Hands`: Collection managing both hands
  - `Finger`: Individual finger tracking
  - `Palm`: Palm position and orientation
  - `Landmark`: Enhanced landmark model with world coordinates
- **Gesture Detectors** (`models/hands/`): 
  - `BaseGestureDetector`: Abstract base for all detectors
  - `HandGesturesDetector`: Single-hand gestures
  - `TwoHandsGesturesDetector`: Two-hands gestures
- **Smoothing** (`smoothing.py`): Time-based EMA decorators
- **CLI** (`cli/`): Command-line interface modules
- **Playground** (`cli/playground/`): Web server and applications

### Design Principles

- **Type Safety**: Strict typing with mypy --strict
- **Property-based Smoothing**: Decorators provide `.raw` and smoothed values
- **Streaming Architecture**: Async callbacks for real-time processing  
- **Modular Gestures**: Easy extension via detector classes
- **Zero Configuration**: Works out of the box with sensible defaults

## Development

### Setup Development Environment

The project uses [uv](https://github.com/astral-sh/uv) for dependency management, which is significantly faster than pip.

```bash
# Install with development dependencies (uses uv)
make dev

# Install only core dependencies (uses uv)
make install

# Format and lint (REQUIRED before committing)
make pretty lint

# Clean build artifacts
make clean

# Deep clean including caches
make full-clean
```

### Adding Custom Gestures

1. Create a detector class inheriting from `HandGesturesDetector`
2. Implement `detect()` method returning confidence (0.0-1.0)
3. Set class attributes: `gesture`, `stateful`, timing parameters
4. Class auto-registers on definition

Example:
```python
class MyGestureDetector(HandGesturesDetector):
    gesture = Gestures.MY_GESTURE
    main_direction_range = (80, 100)  # Optional pre-filter
    
    def detect(self, hand: Hand) -> float:
        # Return 0.0-1.0 based on hand pose
        return confidence
```

### Project Structure
```
adv-gestures/
├── src/adv_gestures/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   ├── cameras.py           # Camera management (Linux)
│   ├── config.py            # Pydantic configuration
│   ├── drawing.py           # OpenCV visualization
│   ├── gestures.py          # Gesture enums and sets
│   ├── mediapipe.py         # MediaPipe imports
│   ├── recognizer.py        # Core recognition engine
│   ├── smoothing.py         # EMA smoothing decorators
│   ├── cli/
│   │   ├── __init__.py      # CLI app definition
│   │   ├── check_camera.py  # Camera testing command
│   │   ├── common.py        # Shared utilities
│   │   ├── options.py       # CLI option definitions
│   │   ├── run.py           # Main recognition command
│   │   ├── tweak.py         # Interactive config editor
│   │   ├── depth_viz.py     # Depth visualization command
│   │   └── playground/
│   │       ├── __init__.py  # Playground command
│   │       ├── server.py    # aiohttp + WebRTC server
│   │       └── static/
│   │           ├── index.html           # Main page
│   │           ├── main.js              # Core client
│   │           ├── styles.css           # Styling
│   │           ├── application-manager.js # App loader
│   │           └── apps/
│   │               ├── _base.js        # Base app class
│   │               ├── default.js      # Default viz
│   │               ├── debug.js        # Debug overlays
│   │               ├── drawing.js      # Air drawing
│   │               ├── pong.js         # Pong game
│   │               ├── theremin.js     # Music app
│   │               └── 3d_viewer.js    # 3D visualization
│   └── models/
│       ├── __init__.py      # Model exports
│       ├── fingers.py       # Finger model
│       ├── landmarks.py     # Landmark definitions
│       ├── utils.py         # Utilities
│       └── hands/
│           ├── __init__.py           # Package exports
│           ├── base_gestures.py      # Detector base
│           ├── hand.py               # Hand model
│           ├── hand_gestures.py      # Single detectors
│           ├── hands.py              # Hands collection
│           ├── hands_gestures.py     # Two-hands detectors
│           └── palm.py               # Palm model
├── CLAUDE.md               # AI assistant instructions
├── Makefile                # Development tasks
├── pyproject.toml          # Package configuration
└── README.md               # This file
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
