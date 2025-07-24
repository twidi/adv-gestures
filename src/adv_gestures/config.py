import sys
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import BaseModel, Field, create_model

from .gestures import CUSTOM_GESTURES


class BaseFingerStraightnessConfig(BaseModel):
    straight_threshold: float = Field(0.85, description="Minimum score for a finger to be considered straight")
    nearly_straight_threshold: float = Field(
        0.65, description="Minimum score for a finger to be considered nearly straight"
    )


class FingerStraightnessConfig(BaseFingerStraightnessConfig):
    distal_segments_max_ratio: float = Field(
        1.5, description="Max ratio between distal finger segments when facing camera"
    )
    distal_segments_max_ratio_back: float = Field(
        1.5, description="Max ratio between distal finger segments when facing away"
    )
    max_angle_degrees: float = Field(10.0, description="Maximum angle (degrees) for perfect straightness score")
    segment_ratio_score_at_threshold: float = Field(0.7, description="Score when segment ratio equals max_ratio")
    segment_ratio_decay_rate: float = Field(2.0, description="Exponential decay rate for segment ratio scoring")
    segment_ratio_linear_range: float = Field(0.15, description="Linear interpolation range for segment ratio")
    angle_score_linear_range: float = Field(0.15, description="Linear interpolation range for angle scoring")
    angle_score_at_threshold: float = Field(0.85, description="Score when angle equals max_angle_degrees")
    angle_decay_rate: float = Field(20.0, description="Exponential decay rate for angle scoring")
    angle_score_weight: float = Field(0.7, description="Weight for angle score in final calculation")
    segment_ratio_weight: float = Field(0.3, description="Weight for segment ratio score in final calculation")


class ThumbStraightnessConfig(BaseFingerStraightnessConfig):
    straight_threshold: float = Field(
        0.6, description=BaseFingerStraightnessConfig.model_fields["straight_threshold"].description
    )
    nearly_straight_threshold: float = Field(
        0.3, description=BaseFingerStraightnessConfig.model_fields["nearly_straight_threshold"].description
    )
    alignment_threshold: float = Field(0.02, description="Minimum cross product magnitude for perfect alignment")
    max_deviation_for_zero_score: float = Field(0.15, description="Deviation magnitude that results in zero score")


class BaseFingerConfig(BaseModel):
    fully_bent_max_angle_degrees: float = Field(
        60.0, description="Max angle (degrees) for a finger to be considered fully bent"
    )
    straightness: BaseFingerStraightnessConfig


class FingerConfig(BaseFingerConfig):
    straightness: FingerStraightnessConfig = Field(
        default_factory=lambda: FingerStraightnessConfig(),
        description="Configuration for finger straightness detection",
    )
    thumb_distance_relative_threshold: float = Field(
        0.7, description="Relative distance threshold for thumb touch detection"
    )


class ThumbConfig(BaseFingerConfig):
    fully_bent_max_angle_degrees: float = Field(
        150.0, description=BaseFingerConfig.model_fields["fully_bent_max_angle_degrees"].description
    )
    straightness: ThumbStraightnessConfig = Field(
        default_factory=lambda: ThumbStraightnessConfig(),
        description="Configuration for thumb straightness detection",
    )


class AdjacentFingerConfig(BaseModel):
    thumb_index_max_angle_degrees: float = Field(
        10.0, description="Max angle (degrees) between thumb and index fingers for touching"
    )
    index_middle_max_angle_degrees: float = Field(
        3.0, description="Max angle (degrees) between index and middle fingers for touching"
    )
    middle_ring_max_angle_degrees: float = Field(
        3.0, description="Max angle (degrees) between middle and ring fingers for touching"
    )
    ring_pinky_max_angle_degrees: float = Field(
        3.0, description="Max angle (degrees) between ring and pinky fingers for touching"
    )


class DefaultGesturesConfig(BaseModel):
    disable_all: bool = Field(False, description="Disable all default MediaPipe gestures")


# Create fields for CustomGesturesDisableConfig dynamically
_custom_gesture_fields: dict[str, Any] = {"all": (bool, Field(False, description="Disable all custom gestures"))}

# Add a field for each custom gesture, sorted alphabetically
for _gesture in sorted(CUSTOM_GESTURES, key=lambda g: g.name):
    _field_name = _gesture.name  # This will be MIDDLE_FINGER, SPOCK, etc.
    _custom_gesture_fields[_field_name] = (bool, Field(False, description=f"Disable {_gesture.value} gesture"))

CustomGesturesDisableConfig = create_model("CustomGesturesDisableConfig", **_custom_gesture_fields)


class CustomGesturesConfig(BaseModel):
    disable: CustomGesturesDisableConfig = Field(  # type: ignore[valid-type]
        default_factory=lambda: CustomGesturesDisableConfig(),
        description="Disable configuration for custom gestures",
    )


class GesturesConfig(BaseModel):
    disable_all: bool = Field(False, description="Disable all gestures (both default and custom)")
    default: DefaultGesturesConfig = Field(
        default_factory=lambda: DefaultGesturesConfig(),
        description="Configuration for default MediaPipe gestures",
    )
    custom: CustomGesturesConfig = Field(
        default_factory=lambda: CustomGesturesConfig(),
        description="Configuration for custom gestures",
    )


class HandsConfig(BaseModel):
    thumb: ThumbConfig = Field(
        default_factory=lambda: ThumbConfig(),
        description="Configuration for thumb detection",
    )
    index: FingerConfig = Field(default_factory=lambda: FingerConfig(), description="Configuration for index finger")
    middle: FingerConfig = Field(
        default_factory=lambda: FingerConfig(), description="Configuration for middle finger"
    )
    ring: FingerConfig = Field(default_factory=lambda: FingerConfig(), description="Configuration for ring finger")
    pinky: FingerConfig = Field(default_factory=lambda: FingerConfig(), description="Configuration for pinky finger")
    adjacent_fingers: AdjacentFingerConfig = Field(
        default_factory=lambda: AdjacentFingerConfig(),
        description="Configuration for adjacent finger touch detection",
    )
    gestures: GesturesConfig = Field(
        default_factory=lambda: GesturesConfig(),
        description="Configuration for gesture detection",
    )


class CLIConfig(BaseModel):
    """Configuration for CLI settings."""

    camera: str | None = Field(None, description="Camera name filter for auto-selection")
    mirror: bool = Field(False, description="Mirror the video output horizontally")
    size: int = Field(1280, description="Maximum dimension for camera capture resolution")


class Config(BaseModel):
    hands: HandsConfig = Field(default_factory=lambda: HandsConfig(), description="Hand detection configuration")
    cli: CLIConfig = Field(default_factory=lambda: CLIConfig(), description="CLI configuration")

    @classmethod
    def get_user_path(cls) -> Path:
        app_name = "adv-gestures"
        config_dir = Path(platformdirs.user_config_dir(app_name))
        return config_dir / "config.json"

    @classmethod
    def validate_path(cls, path: Path | str | None) -> Path:
        if path is None:
            path = cls.get_user_path()
        elif isinstance(path, str):
            path = Path(path)

        return path.resolve()

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = cls.validate_path(path)

        if not path.exists():
            # If the config file does not exist, return a default config
            print(f"Config file {path} does not exist. Returning default config.")
            return cls()

        if path.exists() and not path.is_file():
            raise ValueError(f"Path {path} exists and is not a file.")

        try:
            return cls.model_validate_json(path.read_text(), strict=True)
        except Exception as e:
            print(f"Error loading config from {path}: {e}", file=sys.stderr)
            print("Returning default config.", file=sys.stderr)
            return cls()

    def save(self, path: Path | str | None = None) -> None:
        path = self.validate_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not path.is_file():
            raise ValueError(f"Path {path} exists and is not a file.")

        try:
            path.write_text(self.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error saving config to {path}: {e}", file=sys.stderr)
            raise e
