"""Smoothing utilities for gesture detection values."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import time
from typing import Any, Generic, Literal, Protocol, TypeVar, overload

# Smoothing configuration constants (in seconds)
SMOOTHING_WINDOW = 0.15  # Window for numeric values
BOOLEAN_DEBOUNCE = 0.1  # Debouncing for booleans
GESTURE_CONFIDENCE_WINDOW = 0.5  # Window for gesture transitions

SmoothingMethod = Literal["average", "ema", "median"]
T = TypeVar("T")
P = TypeVar("P")


@dataclass
class TimedValue(Generic[T]):
    """Value with timestamp."""

    value: T
    timestamp: float  # time.time()


class NumberSmoother:
    """Smooths a numeric value over a time window."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        method: SmoothingMethod = "ema",
        ema_alpha: float = 0.3,
    ):
        self.window = window
        self.method = method
        self.ema_alpha = ema_alpha
        self.history: deque[TimedValue[float]] = deque()
        self._last_raw: float | None = 0.0

    def update(self, value: float | None) -> float | None:
        """Update with new value and return smoothed result."""
        if value is None:
            self._last_raw = None
            return None

        now = time()
        self._last_raw = value
        self.history.append(TimedValue(value, now))

        # Clean old values outside the window
        cutoff = now - self.window
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

        return self._compute_smoothed()

    def _compute_smoothed(self) -> float:
        """Compute smoothed value based on selected method."""
        if not self.history:
            return 0.0

        if self.method == "average":
            return sum(tv.value for tv in self.history) / len(self.history)
        elif self.method == "ema":
            # Exponential moving average
            result = self.history[0].value
            for tv in list(self.history)[1:]:
                result = self.ema_alpha * tv.value + (1 - self.ema_alpha) * result
            return result
        else:  # median
            values = sorted(tv.value for tv in self.history)
            return values[len(values) // 2]

    @property
    def raw(self) -> float | None:
        """Get the last raw (unsmoothed) value."""
        return self._last_raw


class CoordSmoother:
    """Smooths a 2D/3D coordinate."""

    def __init__(
        self,
        dimensions: int = 2,
        window: float = SMOOTHING_WINDOW,
        method: SmoothingMethod = "ema",
        ema_alpha: float = 0.3,
    ):
        self.dimensions = dimensions
        self.smoothers = [NumberSmoother(window, method, ema_alpha) for _ in range(dimensions)]
        self._last_raw: tuple[float, ...] | None = (0.0,) * dimensions

    def update(self, coord: tuple[float, ...] | None) -> tuple[float, ...] | None:
        """Update with new coordinate and return smoothed result."""
        if coord is None:
            self._last_raw = None
            return None

        if len(coord) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(coord)}")

        self._last_raw = coord

        # Update each dimension
        results = []
        for s, v in zip(self.smoothers, coord, strict=True):
            result = s.update(v)
            if result is None:
                return None
            results.append(result)

        return tuple(results)

    @property
    def raw(self) -> tuple[float, ...] | None:
        """Get the last raw (unsmoothed) coordinate."""
        return self._last_raw


class BoxSmoother:
    """Smooths a bounding box."""

    def __init__(
        self,
        window: float = SMOOTHING_WINDOW,
        method: SmoothingMethod = "ema",
        ema_alpha: float = 0.3,
    ):
        self.min_smoother = CoordSmoother(2, window, method, ema_alpha)
        self.max_smoother = CoordSmoother(2, window, method, ema_alpha)
        self._last_raw: Any = None  # Will be Box type from __init__.py

    def update(self, box: Any) -> Any:
        """Update with new bounding box and return smoothed result."""
        # Import here to avoid circular import
        from . import Box

        self._last_raw = box

        # Handle None case
        if box is None:
            return None

        min_smoothed = self.min_smoother.update((box.min_x, box.min_y))
        max_smoothed = self.max_smoother.update((box.max_x, box.max_y))

        # Both should not be None since we passed non-None values
        assert min_smoothed is not None
        assert max_smoothed is not None

        return Box(
            min_x=min_smoothed[0],
            min_y=min_smoothed[1],
            max_x=max_smoothed[0],
            max_y=max_smoothed[1],
        )

    @property
    def raw(self) -> Any:
        """Get the last raw (unsmoothed) bounding box."""
        return self._last_raw


class BooleanSmoother:
    """Smooths boolean values with time-based debouncing."""

    def __init__(self, debounce: float = BOOLEAN_DEBOUNCE):
        self.debounce = debounce
        self.current_state = False
        self.last_change_time = 0.0
        self.pending_state: bool | None = None
        self._last_raw = False

    def update(self, value: bool) -> bool:
        """Update with new boolean value and return debounced result."""
        now = time()
        self._last_raw = value

        if value != self.current_state:
            # Value changed from current state
            if self.pending_state != value:
                # New pending change
                self.pending_state = value
                self.last_change_time = now
            elif (now - self.last_change_time) >= self.debounce:
                # Pending change has been stable long enough
                self.current_state = value
                self.pending_state = None
        else:
            # Value matches current state, cancel any pending change
            self.pending_state = None

        return self.current_state

    @property
    def raw(self) -> bool:
        """Get the last raw (undebounced) value."""
        return self._last_raw


class GestureSmoother(Generic[T]):
    """Smooths gesture transitions with majority voting and confidence threshold."""

    def __init__(
        self,
        window: float = GESTURE_CONFIDENCE_WINDOW,
        confidence_threshold: float = 0.6,
        default_value: T | None = None,
    ):
        self.window = window
        self.confidence_threshold = confidence_threshold
        self.history: deque[TimedValue[T]] = deque()
        self.current_gesture = default_value
        self._last_raw = default_value

    def update(self, gesture: T) -> T | None:
        """Update with new gesture and return smoothed result."""
        now = time()
        self._last_raw = gesture
        self.history.append(TimedValue(gesture, now))

        # Clean old values outside the window
        cutoff = now - self.window
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

        if not self.history:
            return self.current_gesture

        # Count occurrences of each gesture
        gesture_counts: dict[T, int] = {}
        for tv in self.history:
            gesture_counts[tv.value] = gesture_counts.get(tv.value, 0) + 1

        # Find the most common gesture
        if gesture_counts:
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            # Only update if it meets the confidence threshold
            if most_common[1] / len(self.history) >= self.confidence_threshold:
                self.current_gesture = most_common[0]

        return self.current_gesture

    @property
    def raw(self) -> T | None:
        """Get the last raw (unsmoothed) gesture."""
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
    **kwargs: Any,
) -> SmoothedProperty[float]:
    """Create a smoothed float property."""
    return SmoothedProperty(func, NumberSmoother, cached, **kwargs)  # type: ignore[arg-type]
