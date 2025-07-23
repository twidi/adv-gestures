"""Advanced hand gesture recognition library using MediaPipe and OpenCV."""

from .config import Config
from .gestures import Gestures
from .models import Box, Finger, FingerIndex, Hand, Hands, Palm
from .recognizer import Recognizer, StreamInfo

__all__ = [
    # Core classes
    "Recognizer",
    "StreamInfo",
    # Models
    "Hand",
    "Hands",
    "Finger",
    "FingerIndex",
    "Palm",
    "Box",
    # Gesture models
    "Gestures",
    # Configuration
    "Config",
]
