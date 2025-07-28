"""Smoothing utilities for gesture detection values."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast, overload

from .gestures import Gestures

if TYPE_CHECKING:
    from .models.hands.utils import Box

# Smoothing configuration constants
SMOOTHING_WINDOW = 0.1  # Window for all smoothing operations (in seconds)
SMOOTHING_EMA_WEIGHT = 0.3  # Weight for new values in exponential moving average (0-1)
GESTURE_SMOOTHING_WINDOW = 0.3  # Window for gesture smoothing operations (in seconds)

T = TypeVar("T")


@dataclass
class TimedValue(Generic[T]):
    """Value with timestamp."""

    value: T
    timestamp: float  # time.time()


NumberSmootherType = TypeVar("NumberSmootherType", float, float | None)


class _NumberSmoother(Generic[NumberSmootherType]):
    """Smooths a numeric value over a time window."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
        default_value: NumberSmootherType = 0.0,
    ):
        self.window = window
        self.ema_alpha = ema_alpha
        self.history: deque[TimedValue[NumberSmootherType | None]] = deque()
        self._last_raw: NumberSmootherType | None = default_value
        self.default_value: NumberSmootherType = default_value

    def update(self, value: NumberSmootherType | None) -> NumberSmootherType:
        """Update with new value and return smoothed result."""
        now = time()
        self._last_raw = value
        self.history.append(TimedValue(value, now))

        # Clean old values outside the window
        cutoff = now - self.window
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

        return self._compute_smoothed()

    def _compute_smoothed(self) -> NumberSmootherType:
        """Compute smoothed value using exponential moving average."""
        if not self.history:
            return self.default_value

        # Find first non-None value to initialize
        result: NumberSmootherType | None = None
        start_idx = 0
        for i, tv in enumerate(self.history):
            if tv.value is not None:
                result = tv.value
                start_idx = i
                break

        if result is None:
            # If all values are None, return default value
            return self.default_value

        # Exponential moving average starting from the next value after initialization
        for tv in list(self.history)[start_idx + 1 :]:
            if tv.value is None:
                continue
            result = self.ema_alpha * tv.value + (1 - self.ema_alpha) * result
        return result

    @property
    def raw(self) -> float | None:
        """Get the last raw (unsmoothed) value."""
        return self._last_raw


class NumberSmoother(_NumberSmoother[float]):
    pass


class OptionalNumberSmoother(_NumberSmoother[float | None]):
    """Smooths a numeric value that can be None."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
        default_value: float | None = None,
    ):
        self.none_smoother = BooleanSmoother(window=window, ema_alpha=ema_alpha)
        super().__init__(window=window, ema_alpha=ema_alpha, default_value=default_value)

    def update(self, value: float | None) -> float | None:
        smoothed_to_none = self.none_smoother.update(value is None)
        smoothed_value = super().update(value)
        return None if smoothed_to_none else smoothed_value


class CoordSmoother:
    """Smooths a 2D/3D coordinate."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
    ):
        self.smoothers = [OptionalNumberSmoother(window, ema_alpha) for _ in range(2)]
        self._last_raw: tuple[float, float] | None = None

    def update(self, coord: tuple[float, float] | None) -> tuple[float, float] | None:
        """Update with new coordinate and return smoothed result."""

        if coord is not None and len(coord) != 2:
            raise ValueError(f"Expected 2 dimensions, got {len(coord)}")

        self._last_raw = coord

        # Update each dimension
        results = tuple(
            s.update(v) for s, v in zip(self.smoothers, (None, None) if coord is None else coord, strict=True)
        )

        return None if None in results else cast(tuple[float, float], results)

    @property
    def raw(self) -> tuple[float, float] | None:
        """Get the last raw (unsmoothed) coordinate."""
        return self._last_raw


class ManyCoordsSmoother:
    """Smooths a tuple of many coordinates (n tuples of two floats each)."""

    def __init__(
        self,
        nb_coords: int,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
    ):
        self.nb_coords = nb_coords
        self.smoothers = [CoordSmoother(window, ema_alpha) for _ in range(self.nb_coords)]
        self._last_raw: tuple[tuple[float, float], ...] | None = None

    def update(self, coords: tuple[tuple[float, float], ...] | None) -> tuple[tuple[float, float], ...] | None:
        """Update with new coordinates and return smoothed result."""
        if coords is not None and len(coords) != self.nb_coords:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")

        self._last_raw = coords

        results = tuple(
            s.update(v)
            for s, v in zip(self.smoothers, (None,) * self.nb_coords if coords is None else coords, strict=True)
        )

        return None if None in results else cast(tuple[tuple[float, float], ...], results)

    @property
    def raw(self) -> tuple[tuple[float, float], ...] | None:
        """Get the last raw (unsmoothed) coordinates."""
        return self._last_raw


class BoxSmoother:
    """Smooths a bounding box."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
    ):
        self.min_smoother = CoordSmoother(window, ema_alpha)
        self.max_smoother = CoordSmoother(window, ema_alpha)
        self._last_raw: Box | None = None

    def update(self, box: "Box | None") -> "Box | None":
        """Update with new bounding box and return smoothed result."""
        # Import here to avoid circular import
        from .models import Box

        self._last_raw = box

        min_smoothed = self.min_smoother.update(None if box is None else (box.min_x, box.min_y))
        max_smoothed = self.max_smoother.update(None if box is None else (box.max_x, box.max_y))

        if min_smoothed is None or max_smoothed is None:
            # If either smoothed value is None, return None
            return None

        return Box(
            min_x=min_smoothed[0],
            min_y=min_smoothed[1],
            max_x=max_smoothed[0],
            max_y=max_smoothed[1],
        )

    @property
    def raw(self) -> "Box | None":
        """Get the last raw (unsmoothed) bounding box."""
        return self._last_raw


class BooleanSmoother:
    """Smooths boolean values using exponential moving average."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
    ):
        # Use NumberSmoother internally
        self.smoother = NumberSmoother(0.0, window, ema_alpha)
        self._last_raw = False

    def update(self, value: bool) -> bool:
        """Update with new boolean value and return smoothed result."""
        self._last_raw = value

        # Convert bool to float (True=1.0, False=0.0)
        numeric_result = self.smoother.update(float(value))

        # Convert back to bool (threshold at 0.5)
        return numeric_result >= 0.5 if numeric_result is not None else False

    @property
    def raw(self) -> bool:
        """Get the last raw (unsmoothed) value."""
        return self._last_raw


class EnumSmoother(Generic[T]):
    """Smooths enum values transitions using exponentially weighted voting."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
        default_value: T | None = None,
    ):
        self.window = window
        self.ema_alpha = ema_alpha
        self.history: deque[TimedValue[T]] = deque()
        self.current_value = default_value
        self._last_raw = default_value

    def update(self, value: T) -> T | None:
        """Update with new value and return smoothed result."""
        now = time()
        self._last_raw = value
        self.history.append(TimedValue(value, now))

        # Clean old values outside the window
        cutoff = now - self.window
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

        if not self.history:
            return self.current_value

        # Calculate exponentially weighted votes
        weights_by_value: dict[T, float] = {}

        # Iterate from newest to oldest (reversed)
        for k, tv in enumerate(reversed(self.history)):
            weight = (1 - self.ema_alpha) ** k
            if tv.value not in weights_by_value:
                weights_by_value[tv.value] = 0.0
            weights_by_value[tv.value] += weight

        # Find the value with maximum weight
        if weights_by_value:
            self.current_value = max(weights_by_value, key=weights_by_value.get)  # type: ignore[arg-type]

        return self.current_value

    @property
    def raw(self) -> T | None:
        """Get the last raw (unsmoothed) value."""
        return self._last_raw


# Type alias for gesture weights
GestureWeights = dict[Gestures, float]


class MultiGestureSmoother:
    """Smooths multiple gestures with confidence weights using number smoothers."""

    def __init__(
        self,
        window: float = GESTURE_SMOOTHING_WINDOW,
        ema_alpha: float = SMOOTHING_EMA_WEIGHT,
        default_value: Gestures | None = None,
    ):
        self.window = window
        self.ema_alpha = ema_alpha
        self.smoothers: dict[Gestures, NumberSmoother] = {}
        self._last_raw: GestureWeights = {}

    def update(self, gesture_weights: GestureWeights) -> GestureWeights:
        """Update with new gesture weights and return smoothed result."""
        from .gestures import Gestures

        self._last_raw = gesture_weights.copy()

        # Create smoothers for new gestures on demand
        for gesture in Gestures:
            if gesture not in self.smoothers:
                self.smoothers[gesture] = NumberSmoother(
                    window=self.window, ema_alpha=self.ema_alpha, default_value=0.0
                )

        # Update each smoother and collect results
        smoothed_weights = {}
        max_weight = 0.0

        for gesture, smoother in self.smoothers.items():
            # Weight is 1.0 if detected, 0.0 otherwise
            weight = gesture_weights.get(gesture, 0.0)
            smoother.update(weight)

            # Calculate weight as sum of values in history
            history_sum = sum(tv.value for tv in smoother.history if tv.value is not None)

            # Include any gesture with weight > 0
            if history_sum > 0:
                smoothed_weights[gesture] = history_sum
                max_weight = max(max_weight, history_sum)

        # Normalize weights to [0, 1] based on max weight
        if max_weight > 0:
            smoothed_weights = {g: w / max_weight for g, w in smoothed_weights.items()}

        return smoothed_weights

    @property
    def raw(self) -> GestureWeights:
        """Get the last raw (unsmoothed) gesture weights."""
        return self._last_raw


class SmoothedBase:
    """Base class for objects with smoothed properties."""

    def __init__(self) -> None:
        """Initialize the smoothed properties tracker."""
        self._smoothed_properties: set[str] = set()

    @property
    def raw(self) -> Any:
        """Access raw (unsmoothed) values."""

        class RawAccessor:
            def __init__(self, parent: Any) -> None:
                self._parent = parent

            def __getattr__(self, name: str) -> Any:
                # Try to get the raw value from smoother
                smoother_attr = f"__{name}_smoother"
                if hasattr(self._parent, smoother_attr):
                    smoother = getattr(self._parent, smoother_attr)
                    return smoother.raw
                # Fallback to direct attribute if it exists
                if hasattr(self._parent, name):
                    return getattr(self._parent, name)
                raise AttributeError(f"'{type(self._parent).__name__}' object has no attribute '{name}'")

        return RawAccessor(self)

    def reset(self) -> None:
        """Reset all cached smoothed properties."""
        for prop_name in self._smoothed_properties:
            cached_attr = f"__{prop_name}_cached"
            if hasattr(self, cached_attr):
                delattr(self, cached_attr)


class Smoother(Protocol[T]):
    """Protocol for smoother classes."""

    def update(self, value: T) -> T:
        """Update with new value and return smoothed result."""
        ...

    @property
    def raw(self) -> T:
        """Get the last raw value."""
        ...


class SmoothedProperty(Generic[T]):
    """Type-safe descriptor for smoothed properties."""

    def __init__(
        self,
        func: Callable[[Any], T],
        smoother_class: type[Smoother[T]],
        cached: bool = True,
        **smoother_kwargs: Any,
    ):
        self.func = func
        self.smoother_class = smoother_class
        self.cached = cached
        self.smoother_kwargs = smoother_kwargs
        self.name = func.__name__
        self.smoother_attr = f"__{self.name}_smoother"
        self.cached_attr = f"__{self.name}_cached"

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Called when the descriptor is assigned to a class."""
        self.name = name
        self.smoother_attr = f"__{name}_smoother"
        self.cached_attr = f"__{name}_cached"

    @overload
    def __get__(self, obj: None, objtype: type[Any]) -> "SmoothedProperty[T]": ...

    @overload
    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> T: ...

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> T | "SmoothedProperty[T]":
        """Get the smoothed value."""
        if obj is None:
            return self

        # Register this property on first use
        if hasattr(obj, "_smoothed_properties"):
            obj._smoothed_properties.add(self.name)

        # Check cache first if caching enabled
        if self.cached and hasattr(obj, self.cached_attr):
            return getattr(obj, self.cached_attr)  # type: ignore[no-any-return]

        # Create smoother on first use
        if not hasattr(obj, self.smoother_attr):
            setattr(obj, self.smoother_attr, self.smoother_class(**self.smoother_kwargs))

        # Compute raw value
        raw_value = self.func(obj)

        # Update smoother and get smoothed value
        smoother: Smoother[T] = getattr(obj, self.smoother_attr)
        smoothed_value = smoother.update(raw_value)

        # Cache if enabled
        if self.cached:
            setattr(obj, self.cached_attr, smoothed_value)

        return smoothed_value


# Helper functions for creating smoothed properties with proper typing
def smoothed_bool(
    func: Callable[[Any], bool],
    cached: bool = True,
    **kwargs: Any,
) -> SmoothedProperty[bool]:
    """Create a smoothed boolean property."""
    return SmoothedProperty(func, BooleanSmoother, cached, **kwargs)


def smoothed_float(
    func: Callable[[Any], float],
    cached: bool = True,
    default_value: float = 0.0,
    **kwargs: Any,
) -> SmoothedProperty[float]:
    """Create a smoothed float property."""
    return SmoothedProperty(func, NumberSmoother, cached, default_value=default_value, **kwargs)  # type: ignore[arg-type]


def smoothed_optional_float(
    func: Callable[[Any], float | None],
    cached: bool = True,
    default_value: float | None = None,
    **kwargs: Any,
) -> SmoothedProperty[float | None]:
    """Create a smoothed optional float property."""
    return SmoothedProperty(func, OptionalNumberSmoother, cached, default_value=default_value, **kwargs)  # type: ignore[arg-type]
