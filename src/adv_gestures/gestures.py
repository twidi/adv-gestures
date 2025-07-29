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
    AIR_TAP = "Air_Tap"  # Index finger held straight and still for short delay
    PRE_AIR_TAP = "Pre_Air_Tap"  # Index finger held straight and still before an air tap
    WAVE = "Wave"  # Open palm waving left-right motion
    SNAP = "Snap"  # Finger snap detected by before/after states
    SWIPE = "Swipe"  # Swipe in any direction by hand or index finger (direction and type stored in detection data)
    NO = "No"  # Index finger waving left-right with other fingers bent

    # Hands gestures (gestures implying both hands)
    PRAY = "Pray"  # Both hands in a prayer position, palms together, fingers pointing up
    CLAP = "Clap"  # Hands joined briefly (less than 1 second) then separated
    CROSSED_FLAT = "Crossed_Flat"  # Both hands crossed with fingers straight
    CROSSED_FISTS = "Crossed_Fists"  # Both hands crossed with fingers in fist position
    TIME_OUT = "Time_Out"  # Two hands forming a T shape, perpendicular


DEFAULT_GESTURES: set[Gestures] = {
    Gestures.CLOSED_FIST,
    Gestures.OPEN_PALM,
    Gestures.POINTING_UP,
    Gestures.THUMB_DOWN,
    Gestures.THUMB_UP,
    Gestures.VICTORY,
    Gestures.LOVE,
}
TWO_HANDS_GESTURES: set[Gestures] = {
    Gestures.PRAY,
    Gestures.CLAP,
    Gestures.CROSSED_FLAT,
    Gestures.CROSSED_FISTS,
    Gestures.TIME_OUT,
}

OVERRIDABLE_DEFAULT_GESTURES: set[Gestures] = {Gestures.VICTORY}

CUSTOM_GESTURES: set[Gestures] = {
    gesture for gesture in Gestures if gesture not in DEFAULT_GESTURES and gesture not in TWO_HANDS_GESTURES
} | OVERRIDABLE_DEFAULT_GESTURES
