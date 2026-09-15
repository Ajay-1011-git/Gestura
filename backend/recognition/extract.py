"""Pose extraction wrappers over `pose-format` (T1.3).

Two entry points, matching the two ways signs reach the system:

- :func:`extract_pose_file` / :func:`extract_directory` — offline, file-based.
  Turns the curated vocabulary's source clips into ``.pose`` files once, so
  recognition at runtime compares against pre-extracted data rather than
  re-extracting reference signs on every call (TRD §7).
- :func:`extract_pose_frames` — live, in-memory. Turns a buffered webcam frame
  window into a ``Pose`` without touching disk.

Environment constraint, verified 2026-09-15: ``pose-format`` requires
``mediapipe<0.10.30`` (the legacy ``mediapipe.python.solutions`` API, removed in
1.x), and legacy MediaPipe ships no Python 3.13 wheels. This package therefore
requires **Python 3.12** with ``mediapipe==0.10.21``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pose_format import Pose
from pose_format.bin.pose_estimation import pose_video
from pose_format.utils.holistic import load_holistic

POSE_FORMAT = "mediapipe"
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# MediaPipe Holistic emits 576 keypoints per frame, 468 of them face mesh.
# Stage 1 recognition uses only upper body and hands, and keeping the face mesh
# makes each file ~7.7x larger (698KB vs 91KB measured). Face landmarks are
# dropped on write; re-extract without this filter if Stage 3's optional
# lip-sync layer is ever pursued.
RECOGNITION_COMPONENTS = [
    "POSE_LANDMARKS",
    "LEFT_HAND_LANDMARKS",
    "RIGHT_HAND_LANDMARKS",
]


class ExtractionError(RuntimeError):
    """Pose extraction failed for a specific input."""


def extract_pose_file(
    video_path: Path,
    output_path: Path,
    *,
    model_complexity: int = 1,
    progress: bool = False,
    components: Sequence[str] | None = RECOGNITION_COMPONENTS,
) -> Path:
    """Extract one video file to a ``.pose`` file. Returns the output path.

    Pass ``components=None`` to keep every MediaPipe Holistic component.
    """
    if not video_path.is_file():
        raise ExtractionError(f"input video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pose_video(
            str(video_path),
            str(output_path),
            POSE_FORMAT,
            additional_config={"model_complexity": model_complexity},
            progress=progress,
        )
        if components is not None:
            trimmed = load_pose(output_path).get_components(list(components))
            with output_path.open("wb") as handle:
                trimmed.write(handle)
    except Exception as exc:
        raise ExtractionError(f"extraction failed for {video_path}: {exc}") from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ExtractionError(f"extraction produced no output for {video_path}")
    return output_path


def extract_directory(
    video_dir: Path,
    output_dir: Path,
    *,
    skip_existing: bool = True,
    model_complexity: int = 1,
    components: Sequence[str] | None = RECOGNITION_COMPONENTS,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Extract every video under ``video_dir`` into ``output_dir``.

    Mirrors the ``<gloss_id>/<clip>.mp4`` layout into ``<gloss_id>/<clip>.pose``.
    Returns ``(written, failures)``. Failures are collected rather than raised so
    one unreadable clip does not abandon a long batch — the caller decides what
    an acceptable failure rate is.
    """
    videos = sorted(
        p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES
    )
    written: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for video in videos:
        target = output_dir / video.relative_to(video_dir).with_suffix(".pose")
        if skip_existing and target.is_file() and target.stat().st_size > 0:
            written.append(target)
            continue
        try:
            written.append(
                extract_pose_file(
                    video,
                    target,
                    model_complexity=model_complexity,
                    components=components,
                )
            )
        except ExtractionError as exc:
            failures.append((video, str(exc)))

    return written, failures


def extract_pose_frames(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    width: int,
    height: int,
) -> Pose:
    """Extract a buffered window of RGB webcam frames into a `Pose`.

    The live-stream counterpart to :func:`extract_pose_file`, for the
    Sign->Speech direction where frames never reach disk.
    """
    if not frames:
        raise ExtractionError("no frames supplied")
    return load_holistic(
        list(frames), fps=fps, width=width, height=height, progress=False
    )


def load_pose(path: Path) -> Pose:
    """Read a ``.pose`` file written by any of the extractors above."""
    if not path.is_file():
        raise ExtractionError(f"pose file not found: {path}")
    with path.open("rb") as handle:
        return Pose.read(handle.read())


def pose_shape(pose: Pose) -> tuple[int, int, int]:
    """``(frames, people, keypoints)`` for a loaded pose — used by T1.4."""
    data = pose.body.data
    return int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
