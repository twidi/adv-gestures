from __future__ import annotations

from enum import Enum


class Gestures(str, Enum):
    # Those ones are created by MediaPipe
    CLOSED_FIST = "Closed_Fist"
    OPEN_PALM = "Open_Palm"
    POINTING_UP = "Pointing_Up"
    THUMB_DOWN = "Thumb_Down"
    THUMB_UP = "Thumb_Up"
    VICTORY = "Victory"
    LOVE = "ILoveYou"
    # Those are the ones we detect
    MIDDLE_FINGER = "Middle_Finger"
    SPOCK = "Spock"
    ROCK = "Rock"
    OK = "OK"
    STOP = "Stop"
    PINCH = "Pinch"
    GUN = "Gun"
    FINGER_GUN = "Finger_Gun"  # Gun without the middle finger


DEFAULT_GESTURES = {
    Gestures.CLOSED_FIST,
    Gestures.OPEN_PALM,
    Gestures.POINTING_UP,
    Gestures.THUMB_DOWN,
    Gestures.THUMB_UP,
    Gestures.VICTORY,
    Gestures.LOVE,
}
OVERRIDABLE_DEFAULT_GESTURES = {Gestures.OPEN_PALM}
