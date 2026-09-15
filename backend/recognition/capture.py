"""Live webcam capture and sign segmentation (T1.18).

`extract_pose_frames` was written in T1.3 for exactly this and had never been
called: Sign->Speech only ever ran on files, so the direction that defines the
product had no live path at all.

**Segmentation is motion-based, not landmark-based, and that is the load-bearing
decision.** Deciding "is the signer signing?" from hand landmarks means running
MediaPipe Holistic on every frame to find out whether to run it — about 1.7s per
clip here, nowhere near a 33ms budget. Frame differencing over a downscaled
greyscale image costs well under a millisecond, and MediaPipe then runs once, on
the completed segment. The cost of being wrong is small in one direction and
large in the other: a segment that starts slightly early carries a few still
frames, which `active_segment` trims anyway, while a segmenter that cannot keep
up drops the sign entirely.

The thresholds are in normalised motion units so they do not depend on the
camera's resolution, and the segmenter is a pure function of a motion signal so
it can be tested without a webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Sequence

import numpy as np


class State(Enum):
    IDLE = "idle"
    SIGNING = "signing"


@dataclass
class SegmenterConfig:
    """Tuned for signing, not for general motion detection.

    `start_frames` is short because a sign begins abruptly; `stop_frames` is
    several times longer because signs contain held positions, and cutting at the
    first still frame would split one sign into two.
    """

    motion_threshold: float = 0.012     # normalised mean absolute frame difference
    start_frames: int = 3               # ~0.1s of motion to open a segment
    stop_frames: int = 12               # ~0.4s of stillness to close one
    min_frames: int = 10                # anything shorter is a twitch
    max_frames: int = 150               # ~5s, a hard stop so one never runs away


@dataclass
class SignSegmenter:
    """Turns a stream of motion values into sign start/end events.

    Deliberately holds no frames itself: the caller owns the buffer, so this can
    be driven from a test with a synthetic signal.
    """

    config: SegmenterConfig = field(default_factory=SegmenterConfig)
    state: State = State.IDLE
    _above: int = 0
    _below: int = 0
    _length: int = 0

    def reset(self) -> None:
        self.state = State.IDLE
        self._above = self._below = self._length = 0

    def update(self, motion: float) -> str | None:
        """Feed one frame's motion. Returns 'start', 'end', or None."""
        moving = motion >= self.config.motion_threshold

        if self.state is State.IDLE:
            self._above = self._above + 1 if moving else 0
            if self._above >= self.config.start_frames:
                self.state = State.SIGNING
                self._below = 0
                self._length = self._above
                return "start"
            return None

        self._length += 1
        if self._length >= self.config.max_frames:
            self.reset()
            return "end"
        self._below = 0 if moving else self._below + 1
        if self._below >= self.config.stop_frames:
            long_enough = self._length >= self.config.min_frames
            self.reset()
            return "end" if long_enough else None
        return None


def frame_motion(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Normalised mean absolute difference between two greyscale frames."""
    if previous is None:
        return 0.0
    return float(np.abs(current.astype(np.int16) - previous.astype(np.int16)).mean() / 255.0)


def downscale_grey(frame: np.ndarray, width: int = 160) -> np.ndarray:
    """Cheap greyscale downscale by striding — no OpenCV resize needed."""
    step = max(1, frame.shape[1] // width)
    small = frame[::step, ::step]
    return small.mean(axis=2).astype(np.uint8) if small.ndim == 3 else small


class CaptureError(RuntimeError):
    """The camera could not be opened or read."""


def open_camera(index: int = 0, width: int = 1280, height: int = 720):
    import cv2

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise CaptureError(
            f"could not open camera {index}. On macOS the terminal needs camera "
            f"permission: System Settings > Privacy & Security > Camera."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def segments_from_camera(
    *,
    camera_index: int = 0,
    fps: float = 30.0,
    config: SegmenterConfig | None = None,
    on_state: Callable[[State, float], None] | None = None,
    max_segments: int | None = None,
) -> Iterator[tuple[list[np.ndarray], float, int, int]]:
    """Yield `(rgb_frames, fps, width, height)` for each detected sign.

    Frames are RGB, which is what `extract_pose_frames` expects; OpenCV hands
    back BGR and converting at the boundary keeps that detail here rather than
    in the caller.
    """
    import cv2

    capture = open_camera(camera_index)
    segmenter = SignSegmenter(config or SegmenterConfig())
    buffer: list[np.ndarray] = []
    previous_grey: np.ndarray | None = None
    produced = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise CaptureError("camera stopped returning frames")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            grey = downscale_grey(rgb)
            motion = frame_motion(previous_grey, grey)
            previous_grey = grey

            event = segmenter.update(motion)
            if segmenter.state is State.SIGNING or event == "end":
                buffer.append(rgb)
            if on_state is not None:
                on_state(segmenter.state, motion)

            if event == "start":
                buffer = [rgb]
            elif event == "end":
                if len(buffer) >= (config or SegmenterConfig()).min_frames:
                    height, width = rgb.shape[:2]
                    yield buffer, fps, width, height
                    produced += 1
                    if max_segments is not None and produced >= max_segments:
                        return
                buffer = []
    finally:
        capture.release()


def pose_from_segment(frames: Sequence[np.ndarray], fps: float, width: int, height: int):
    from backend.recognition.extract import extract_pose_frames

    return extract_pose_frames(frames, fps=fps, width=width, height=height)
