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

## Hand Detection Capabilities

The program provides comprehensive hand analysis and can detect:

### Hand-Level Features

- **Handedness Detection**: Distinguishes between left and right hands
- **Hand Orientation**: Determines if palm or back of hand is facing the camera using cross-product analysis
- **Hand Direction**: Calculates main hand direction vector from wrist to middle finger centroid
- **Wrist Position**: Tracks wrist landmark as the base reference point

### Palm Detection

- **Palm Landmarks**: Tracks 6 key palm points (wrist, thumb CMC, and all finger MCP joints)
- **Palm Centroid**: Calculates the geometric center of the palm region
- **Visual Indicators**: Green circle for palm facing camera, red for back of hand

### Finger Analysis (Per Finger)

Each finger (thumb, index, middle, ring, pinky) provides detailed analysis:

#### Geometric Properties
- **Finger Landmarks**: All joint positions (MCP, PIP, DIP, TIP for fingers; CMC, MCP, IP, TIP for thumb)
- **Finger Centroid**: Geometric center of all finger landmarks
- **Start/End Points**: Base (MCP) and tip positions
- **Finger Direction**: Normalized direction vector from base to tip

#### State Detection
- **Straightness**: Advanced algorithm analyzing joint alignment and segment proportions
- **Bend State**: Determines if finger is fully bent based on fold angles
- **Fold Angle**: Calculates bend angle at PIP joint (180° = straight, lower = more bent)
- **Tip Direction**: Direction vector of fingertip based on last two landmarks

#### Interaction Detection
- **Thumb Touch**: Detects when any finger tip touches the thumb tip (3D distance analysis)
- **Adjacent Finger Touch**: Identifies when neighboring fingers are touching using direction similarity
- **All Fingers Touch**: Checks if all adjacent finger pairs are simultaneously touching

### Gesture Recognition

#### Built-in MediaPipe Gestures
- Closed Fist
- Open Palm  
- Pointing Up
- Thumb Up/Down
- Victory (Peace sign)
- ILoveYou (Rock and roll sign)

#### Custom Gesture Analysis
Beyond MediaPipe's built-in gestures, the program analyzes:
- Individual finger states in any combination
- Complex multi-finger interactions
- Hand orientation relative to camera
- Precise geometric relationships between fingers

### Visual Feedback System

The program, when called with `--preview`, provides rich visual overlays:
- **Hand Landmarks**: All 21 hand landmarks with anatomically correct positioning
- **Finger Lines**: Colored lines for straight fingers (blue=thumb, green=index, yellow=middle, magenta=ring, cyan=pinky)
- **Direction Arrows**: White arrows showing individual finger tip directions
- **Main Direction**: Cyan arrow from wrist showing overall hand direction
- **Touch Indicators**: Red circles for thumb-finger contact, cyan lines for adjacent finger contact
- **Palm/Back Indicator**: Green (palm) or red (back) circle at palm center

### Technical Implementation

- **Real-time Processing**: All calculations performed in real-time during video capture
- **Caching**: Expensive calculations cached using Python's `@cached_property` decorator
- **Coordinate System**: Uses MediaPipe's normalized coordinates (0-1 range)
- **3D Analysis**: Incorporates depth information (z-coordinate) where available
- **Threshold-based Detection**: Configurable thresholds for touch detection, straightness, etc.

### Data Structure

The program models hands hierarchically:
- `Hands` → contains left/right `Hand` objects
- `Hand` → contains `Palm` and list of `Finger` objects
- `Finger` → contains landmarks and computed properties
- All objects provide `preview_on_image()` methods for visual debugging

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
