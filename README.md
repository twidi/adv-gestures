# Advanced Gestures (adv-gestures)

A real-time hand gesture recognition Python package using MediaPipe and OpenCV. Detects both built-in MediaPipe gestures and custom gestures through advanced finger state analysis.

## Features

- **Real-time hand tracking** using MediaPipe
- **Dual hand support** - tracks both left and right hands simultaneously
- **Built-in gestures**: Closed Fist, Open Palm, Pointing Up, Thumb Up/Down, Victory, ILoveYou
- **Custom gestures**: Middle Finger, Spock, Rock, OK, Stop, Pinch, Gun, Finger Gun
- **Advanced finger analysis**: Straightness detection, touch detection, and direction tracking
- **Camera auto-selection** with filtering capabilities
- **Visual overlay** showing hand landmarks and detected gestures

## Requirements

- Python >= 3.11
- Linux (required for camera enumeration via `linuxpy`)
- Webcam or camera device

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/gestures.git
cd gestures

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

The MediaPipe model file will be automatically downloaded on first run.

## Usage

After installation, use the `adv-gestures` command:

### Basic Usage
```bash
# Auto-select camera and start gesture recognition
adv-gestures
```

### Camera Selection
```bash
# Filter cameras by name (case-insensitive)
adv-gestures "webcam"

# Preview available cameras without starting recognition
adv-gestures --preview
```

### Headless Mode
```bash
# Run gesture recognition without displaying video (for testing)
adv-gestures --check
```

## Controls

- **ESC**: Exit the application
- The application displays:
  - Hand landmarks (dots and connections)
  - Finger states (straight/bent indicators)
  - Detected gestures above each hand
  - FPS counter in the top-left corner

## Architecture

The package is structured as a Python module in `src/adv_gestures/__init__.py` with these key components:

- **Camera Management**: Automatic camera enumeration and selection
- **Hand Detection**: MediaPipe-based landmark detection
- **Gesture Analysis**: Combination of MediaPipe's built-in recognition and custom finger state analysis
- **Visualization**: Real-time overlay of detection results

## Custom Gesture Detection

Custom gestures are detected by analyzing:
- Individual finger straightness (with per-finger thresholds)
- Finger-to-finger touch detection
- Thumb-to-finger touch detection
- Relative finger positions and directions

## Performance

- Optimized for 30 FPS with MJPG codec
- Real-time processing with minimal latency
- Efficient finger state calculations

## Development

### Code Quality
```bash
# Format code and run linters
make pretty lint
```

### Adding New Gestures
1. Edit the `Hand.detect_gesture()` method in `src/adv_gestures/__init__.py`
2. Use existing finger state properties:
   - `finger.is_straight`
   - `finger.is_nearly_straight`
   - `finger.is_touching(other_finger)`
   - `finger.touches_thumb`
3. Add your gesture logic and return the gesture name

### Project Structure
```
gestures/
├── src/
│   └── adv_gestures/
│       └── __init__.py  # Main package code
├── pyproject.toml       # Package configuration
├── Makefile            # Build and lint commands
├── CLAUDE.md           # AI assistant instructions
├── README.md           # This file
└── gesture_recognizer.task  # MediaPipe model (auto-downloaded)
```

## Troubleshooting

### Camera Not Found
- Use `--preview` to list available cameras
- Ensure your camera is connected and recognized by Linux
- Try filtering by camera name if multiple devices exist

### Gesture Not Detected
- Ensure good lighting conditions
- Keep hand within camera frame
- Check that fingers are clearly visible
- Adjust straightness thresholds if needed

### Performance Issues
- Close other camera-using applications
- Ensure system has sufficient resources
- Check that FPS is stable around 30

## License

[Your license here]

## Contributing

[Your contributing guidelines here]