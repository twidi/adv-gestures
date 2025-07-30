from __future__ import annotations

import os
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, TypeAlias, TypeVar

import cv2

from .mediapipe import (
    BaseOptions,
    Category,
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult,
    RunningMode,
    mp,
)
from .models.hands import Hands
from .models.landmarks import Landmark

OpenCVImage: TypeAlias = cv2.typing.MatLike  # Type alias for images (numpy arrays)


@dataclass
class RecognizerResult:
    gestures: list[list[Category]]
    handedness: list[list[Category]]
    hand_landmarks: list[list[Landmark]]

    timestamp: float  # Timestamp of the result
    recognized_image: mp.Image | None = None  # The image that was recognized


class Recognizer:
    model_url: ClassVar[str] = (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    )

    def __init__(self, model_path: str, use_gpu: bool = True, mirroring: bool = False) -> None:
        self.last_result: RecognizerResult | None = None

        self.check_model(model_path)

        self.mirroring = mirroring

        self.recognizer: GestureRecognizer = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path,
                    delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU,
                ),
                running_mode=RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self.save_result,
            )
        )

    def check_model(self, model_path: str) -> None:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found. Downloading...")
            try:
                urllib.request.urlretrieve(self.model_url, model_path)
                print(f"Successfully downloaded model to '{model_path}'")
            except Exception as exc:
                print(f"Failed to download model: {exc}", file=sys.stderr)
                raise RuntimeError(f"Could not download model from {self.model_url}: {exc}") from exc

    @staticmethod
    def convert_image_from_opencv(frame: OpenCVImage) -> mp.Image:
        # Convert frame to RGB (opencv BGR not supported by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def recognize_image_from_opencv(self, frame: OpenCVImage, timestamp: float) -> mp.Image:
        return self.recognize_image(self.convert_image_from_opencv(frame), timestamp)

    def recognize_image(self, image: mp.Image, timestamp: float) -> mp.Image:
        self.recognizer.recognize_async(image, int(timestamp * 1000))  # Convert seconds to milliseconds
        return image

    def save_result(self, result: GestureRecognizerResult, input_image: mp.Image, timestamp_ms: int) -> None:
        """Save the latest gesture recognition result."""
        self.last_result = RecognizerResult(
            gestures=result.gestures,
            handedness=result.handedness,
            hand_landmarks=[  # Convert MediaPipe landmarks to our Landmark model
                [
                    Landmark.from_normalized(landmark, input_image.width, input_image.height, self.mirroring)
                    for landmark in landmarks
                ]
                for landmarks in result.hand_landmarks
            ],
            timestamp=timestamp_ms / 1000,  # Convert milliseconds to seconds
            recognized_image=input_image,  # Store the recognized image
        )

    def close(self) -> None:
        """Close the recognizer and release resources."""
        if self.recognizer:
            self.recognizer.close()
            self.recognizer = None

    def __enter__(self) -> Recognizer:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Exit the context manager and clean up resources."""
        self.close()

    def _handle_frames_generic(
        self,
        frames: Iterator[T],
        hands: Hands,
        recognize_fn: Callable[[T, float], mp.Image],
    ) -> Iterator[tuple[T, StreamInfo, RecognizerResult]]:
        """Generic frame handling logic for both mp.Image and OpenCVImage frames."""
        start_time = time.perf_counter()
        last_recognized_timestamp: float = -1
        frames_count = 0
        recognized_frames_count = 0

        for frame in frames:
            frames_count += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            # Call recognize function and get MediaPipe image
            mp_image = recognize_fn(frame, elapsed_time)

            if self.last_result is None:
                continue
            if self.last_result.timestamp == last_recognized_timestamp:
                continue

            # Calculate metrics first
            recognized_frames_count += 1
            iterator_fps = frames_count / elapsed_time if elapsed_time > 0 else 0
            recognition_fps = recognized_frames_count / elapsed_time if elapsed_time > 0 else 0
            latency = (current_time - start_time) - self.last_result.timestamp
            last_recognized_timestamp = self.last_result.timestamp

            # Create stream info before updating hands
            stream_info = StreamInfo(
                frames_count=frames_count,
                recognized_frames_count=recognized_frames_count,
                frames_fps=iterator_fps,
                recognition_fps=recognition_fps,
                latency=latency,
                height=mp_image.height,
                width=mp_image.width,
                mirroring=self.mirroring,
            )

            # Update hands with stream info
            hands.update_hands(self, stream_info)

            yield frame, stream_info, self.last_result

    def handle_frames(self, frames: FramesIterator, hands: Hands) -> ResultsGenerator:
        """Read frames from the provided frames provider."""
        return self._handle_frames_generic(frames, hands, self.recognize_image)

    def handle_frames_from_opencv(
        self, frames: OpenCVFramesIterator, hands: Hands
    ) -> ResultsGeneratorWithOpenCVFrame:
        """Read frames from the provided OpenCV frames provider."""
        return self._handle_frames_generic(frames, hands, self.recognize_image_from_opencv)

    def handle_opencv_capture(self, cap: cv2.VideoCapture, hands: Hands) -> ResultsGeneratorWithOpenCVFrame:
        """Read frames from an OpenCV VideoCapture object."""

        def frames_provider() -> OpenCVFramesIterator:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame

        return self.handle_frames_from_opencv(frames_provider(), hands)


class StreamInfo(NamedTuple):
    frames_count: int  # Total number of frames from iterator
    recognized_frames_count: int  # Number of frames that were recognized
    frames_fps: float  # FPS of the frame iterator
    recognition_fps: float  # FPS of recognition (based on update_hands calls)
    latency: float  # Time since last result (current time - last result timestamp)
    width: int  # Width of the image
    height: int  # Height of the image
    mirroring: bool = False  # Whether the recognition results are for a mirrored output

    def to_dict(self) -> dict[str, Any]:
        """Export stream info as a dictionary with all fields."""
        return {
            "frames_count": self.frames_count,
            "recognized_frames_count": self.recognized_frames_count,
            "frames_fps": self.frames_fps,
            "recognition_fps": self.recognition_fps,
            "latency": self.latency,
            "width": self.width,
            "height": self.height,
            "mirroring": self.mirroring,
        }


T = TypeVar("T")
ResultsGenerator: TypeAlias = Iterator[tuple[mp.Image, StreamInfo, RecognizerResult]]
ResultsGeneratorWithOpenCVFrame: TypeAlias = Iterator[tuple[OpenCVImage, StreamInfo, RecognizerResult]]
FramesIterator: TypeAlias = Iterator[mp.Image]
OpenCVFramesIterator: TypeAlias = Iterator[OpenCVImage]
