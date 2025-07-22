import sys
from pathlib import Path

import platformdirs
from pydantic import BaseModel


class BaseFingerStraightnessConfig(BaseModel):
    straight_threshold: float = 0.85
    nearly_straight_threshold: float = 0.65


class FingerStraightnessConfig(BaseFingerStraightnessConfig):
    distal_segments_max_ratio: float = 1.5
    distal_segments_max_ratio_back: float = 1.5
    max_angle_degrees: float = 10.0
    segment_ratio_score_at_threshold: float = 0.7
    segment_ratio_decay_rate: float = 2.0
    segment_ratio_linear_range: float = 0.15
    angle_score_linear_range: float = 0.15
    angle_score_at_threshold: float = 0.85
    angle_decay_rate: float = 20.0
    angle_score_weight: float = 0.7
    segment_ratio_weight: float = 0.3


class ThumbStraightnessConfig(BaseFingerStraightnessConfig):
    alignment_threshold: float = 0.01
    max_deviation_for_zero_score: float = 0.1


class BaseFingerConfig(BaseModel):
    fully_bent_max_angle_degrees: float = 30.0
    straightness: BaseFingerStraightnessConfig


class FingerConfig(BaseFingerConfig):
    straightness: FingerStraightnessConfig = FingerStraightnessConfig()
    thumb_distance_relative_threshold: float = 1.2


class ThumbConfig(BaseFingerConfig):
    straightness: ThumbStraightnessConfig = ThumbStraightnessConfig()


class AdjacentFingerConfig(BaseModel):
    index_middle_max_angle_degrees: float = 3.0
    middle_ring_max_angle_degrees: float = 1.5
    ring_pinky_max_angle_degrees: float = 2.0


class HandsConfig(BaseModel):
    thumb: ThumbConfig = ThumbConfig(fully_bent_max_angle_degrees=150)
    index: FingerConfig = FingerConfig()
    middle: FingerConfig = FingerConfig()
    ring: FingerConfig = FingerConfig()
    pinky: FingerConfig = FingerConfig()
    adjacent_fingers: AdjacentFingerConfig = AdjacentFingerConfig()


class Config(BaseModel):
    hands: HandsConfig = HandsConfig()

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
