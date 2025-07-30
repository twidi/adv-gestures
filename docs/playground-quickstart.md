# Gesture Recognition Playground - Quick Start

The Gesture Recognition Playground is a web-based environment for real-time hand gesture recognition.

## Installation

First, install the playground dependencies:

```bash
pip install "adv-gestures[playground]"
```

## Running the Playground

Start the playground server:

```bash
adv-gestures playground --open
```

Options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 9810)
- `--open`: Open browser automatically after starting

## Using the Playground

1. **Camera Access**: When you first open the playground, it will request camera access. Grant permission to enable gesture recognition.

2. **Camera Selection**: If you have multiple cameras, you can select which one to use. The selection is saved for future sessions.

3. **Controls**:
   - **Mirror Toggle (M)**: Toggle mirror mode (default: ON)
   - **Debug Toggle (D)**: Show/hide debug overlays like hand bounding boxes (default: OFF)
   - **Logs Panel**: View real-time logs including gesture detection events

4. **Gesture Detection**: The playground detects and logs all hand gestures in real-time. When gestures are added or removed, they appear in the logs panel.

## Technical Details

- **Protocol**: WebRTC for video streaming, Server-Sent Events (SSE) for data
- **Data Format**: Raw `hands.to_dict()` output from the gesture recognition library
- **Performance**: Best-effort for local demo usage

## Troubleshooting

- **Camera not working**: Ensure you've granted camera permissions in your browser
- **Connection errors**: Check that no other application is using port 9810
- **Page reload required**: Changing mirror mode requires a page reload to apply the setting