from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import Config
from ...gestures import Gestures
from .hand import Hand
from .utils import Handedness

if TYPE_CHECKING:
    from ...recognizer import Recognizer, StreamInfo


class Hands:

    def __init__(self, config: Config) -> None:
        """Initialize both hands."""
        self.config = config
        self.stream_info: StreamInfo | None = None
        self.left: Hand = Hand(handedness=Handedness.LEFT, config=config)
        self.right: Hand = Hand(handedness=Handedness.RIGHT, config=config)

    def reset(self) -> None:
        """Reset both hands and clear all cached properties."""
        self.left.reset()
        self.right.reset()

    def update_hands(self, recognizer: Recognizer, stream_info: StreamInfo | None = None) -> None:
        """Update the hands object with new gesture recognition results."""
        # Store the stream info
        self.stream_info = stream_info

        # If the recognizer didn't run yet, do nothing
        if not (result := recognizer.last_result):
            return

        # Reset all hands first
        self.reset()

        if not result.hand_landmarks:
            return

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            # Get handedness
            handedness = None
            if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
                handedness = Handedness.from_data(result.handedness[hand_index][0].category_name)

            # Skip if handedness not detected
            if not handedness:
                continue

            # Get the appropriate hand
            hand = self.left if handedness == Handedness.LEFT else self.right

            # Get default gesture information
            gesture_type = None
            if (
                hand_landmarks is not None
                and result.gestures
                and hand_index < len(result.gestures)
                and result.gestures[hand_index]
            ):
                if not (self.config.hands.gestures.disable_all or self.config.hands.gestures.default.disable_all):
                    gesture = result.gestures[hand_index][0]  # Get the top gesture
                    gesture_type = (
                        None
                        if gesture.category_name in (None, "None", "Unknown")
                        else Gestures(gesture.category_name)
                    )

            # Update hand data
            hand.update(
                default_gesture=gesture_type,
                all_landmarks=hand_landmarks,
                stream_info=stream_info,
            )
