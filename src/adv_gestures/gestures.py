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
    PINCH_TOUCH = "Pinch_Touch"  # Pinch with thumb and index touching
    GUN = "Gun"
    FINGER_GUN = "Finger_Gun"  # Gun without the middle finger
    AIR_TAP = "Air_Tap"  # Index finger held straight and still for 2 seconds
    WAVE = "Wave"  # Open palm waving left-right motion
    SNAP = "Snap"  # Finger snap detected by before/after states

    # Hands gestures (gestures implying both hands)
    PRAY = "Pray"  # Both hands in a prayer position, palms together, fingers pointing up
    CLAP = "Clap"  # Hands joined briefly (less than 1 second) then separated


DEFAULT_GESTURES: set[Gestures] = {
    Gestures.CLOSED_FIST,
    Gestures.OPEN_PALM,
    Gestures.POINTING_UP,
    Gestures.THUMB_DOWN,
    Gestures.THUMB_UP,
    Gestures.VICTORY,
    Gestures.LOVE,
}
TWO_HANDS_GESTURES: set[Gestures] = {Gestures.PRAY, Gestures.CLAP}

OVERRIDABLE_DEFAULT_GESTURES: set[Gestures] = {Gestures.VICTORY}

CUSTOM_GESTURES: set[Gestures] = {
    gesture for gesture in Gestures if gesture not in DEFAULT_GESTURES and gesture not in TWO_HANDS_GESTURES
} | OVERRIDABLE_DEFAULT_GESTURES
