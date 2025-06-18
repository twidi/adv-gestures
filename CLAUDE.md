# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based gesture recognition application that uses MediaPipe for hand tracking and gesture detection. The main application (`gestures.py`) captures video from cameras and provides real-time hand gesture recognition with visual feedback.

The repository is managed with Git version control.

### Key Components

- **Hand Detection & Tracking**: Uses MediaPipe's gesture recognizer with a pre-trained model (`gesture_recognizer.task`)
- **Camera Management**: Supports multiple cameras via linuxpy.video for device enumeration and OpenCV for capture
- **Visual Feedback**: Real-time overlay showing hand landmarks, palm/back detection, finger directions, and gesture classifications
- **Data Structures**: Comprehensive hand modeling with `Hand`, `Palm`, `Finger` classes and geometric calculations

### Architecture

The application follows a modular design:
- MediaPipe integration for ML-based gesture recognition
- Structured data classes for hand anatomy representation
- Camera abstraction layer for device management
- Real-time visualization pipeline with OpenCV

## Dependencies

The application requires these Python packages:
- `linuxpy` - Linux camera device access
- `opencv-python` - Video capture and display
- `mediapipe` - Hand tracking and gesture recognition
- `numpy` - Numerical computations

Install with: `pip install linuxpy opencv-python mediapipe numpy`

## Running the Application

Main command: `python gestures.py [camera_filter]`

The application will:
1. List available cameras (optionally filtered by name)
2. Allow selection of a camera
3. Load the MediaPipe gesture recognition model
4. Start real-time gesture detection

The gesture recognizer model file `gesture_recognizer.task` must be present in the working directory.

## Development Notes

- All development work is contained in the single file `gestures.py`
- Testing requires human interaction in front of a camera - Claude Code cannot automatically verify if gesture recognition updates work correctly
- Changes should be tested manually by running the application and performing gestures

## Available Tools

- **ast-grep** (v0.38.5) - Available at `/home/twidi/.npm-global/bin/ast-grep` for structural code search and refactoring
  - Use for finding code patterns with AST-aware matching
  - Supports Python and many other languages
  - More precise than regex-based grep for code patterns

## Model Requirements

The application expects a MediaPipe gesture recognition model file at `gesture_recognizer.task`. If missing, download from:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task