from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...config import Config
from ...smoothing import CoordSmoother, SmoothedBase, SmoothedProperty
from ..landmarks import Landmark

if TYPE_CHECKING:
    from .hand import Hand


class Palm(SmoothedBase):
    _cached_props: ClassVar[tuple[str, ...]] = ("centroid",)

    def __init__(self, hand: Hand, config: Config) -> None:
        super().__init__()
        self.config = config
        self.hand = hand
        self.landmarks: list[Landmark] = []

    def reset(self) -> None:
        """Reset the palm and clear all cached properties."""
        # Call parent class reset for smoothed properties
        super().reset()

        # Clear cached properties
        for prop in self._cached_props:
            self.__dict__.pop(prop, None)

    def update(self, landmarks: list[Landmark]) -> None:
        """Update the palm with new landmarks."""
        self.landmarks = landmarks

    def _calc_centroid(self) -> tuple[float, float]:
        """Calculate the centroid of palm landmarks."""
        if not self.landmarks:
            return 0.0, 0.0

        centroid_x = sum(landmark.x for landmark in self.landmarks) / len(self.landmarks)
        centroid_y = sum(landmark.y for landmark in self.landmarks) / len(self.landmarks)

        return centroid_x, centroid_y

    centroid = SmoothedProperty(_calc_centroid, CoordSmoother)
