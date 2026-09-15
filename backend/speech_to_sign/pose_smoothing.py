"""Speech->Sign: pose sequence smoothing before retargeting (T1.10).

**Which path was taken, and why.** The task nominates
`sign-language-processing/fluent-pose-synthesis`, with an explicit allowance to
fall back to simpler interpolation if its current API does not fit the
timeline. Checked on 2026-09-15: the repo is real and maintained (MIT, pushed
2026-07-23) but **ships no pretrained checkpoints**. It is a training project —
using it means downloading the DGS Corpus through TFDS and training a diffusion
model on German Sign Language, then hoping it transfers to ISL. That is not a
hackathon-timeline dependency, so the documented fallback is taken.

**What is already smoothed, so this does not duplicate it.**
`spoken-to-signed-translation` applies a Butterworth filter (``filtfilt`` over
non-face components) while concatenating looked-up poses, so T1.9's output
arrives already smoothed *between* signs. This module handles what that does not:
gaps where MediaPipe lost a landmark mid-sign, and resampling to the frame rate
the renderer actually runs at.

Interpolation here fills gaps **only between known observations**, and never
extrapolates past the ends of a gap. Inventing landmark positions where none
were observed would put motion on the avatar that the signer never produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pose_format import Pose

MAX_GAP_FRAMES = 6          # ~200ms at 30fps; longer gaps stay missing
MOVING_AVERAGE_WINDOW = 3   # odd, small — enough to take jitter off, not motion


class SmoothingError(RuntimeError):
    """A pose sequence could not be smoothed."""


@dataclass(frozen=True)
class SmoothingReport:
    frames_in: int
    frames_out: int
    fps_in: float
    fps_out: float
    gaps_filled: int
    points_still_missing: int


def _confidence_mask(pose: Pose) -> np.ndarray:
    return np.asarray(pose.body.confidence)[:, 0, :] > 0


def fill_gaps(pose: Pose, *, max_gap: int = MAX_GAP_FRAMES) -> tuple[Pose, int, int]:
    """Linearly interpolate short dropouts in landmark tracks.

    Only gaps bounded by real observations on both sides are filled, and only
    when shorter than ``max_gap``. A landmark missing at the very start or end
    of a clip stays missing.
    """
    data = np.array(pose.body.data, dtype=np.float64)
    mask = _confidence_mask(pose)
    frames, keypoints = mask.shape
    filled = 0

    for point in range(keypoints):
        present = np.flatnonzero(mask[:, point])
        if present.size < 2:
            continue
        for left, right in zip(present[:-1], present[1:]):
            gap = int(right - left) - 1
            if 0 < gap <= max_gap:
                for axis in range(data.shape[3]):
                    data[left + 1 : right, 0, point, axis] = np.linspace(
                        data[left, 0, point, axis],
                        data[right, 0, point, axis],
                        gap + 2,
                    )[1:-1]
                mask[left + 1 : right, point] = True
                filled += gap

    smoothed = pose.copy()
    smoothed.body.data = np.ma.array(data, mask=~np.repeat(
        mask[:, None, :, None], data.shape[3], axis=3
    ))
    smoothed.body.confidence = mask[:, None, :].astype(np.float32)
    return smoothed, filled, int((~mask).sum())


def moving_average(pose: Pose, *, window: int = MOVING_AVERAGE_WINDOW) -> Pose:
    """Take frame-to-frame jitter off the tracks without flattening real motion."""
    if window < 2:
        return pose
    data = np.array(pose.body.data, dtype=np.float64)
    mask = _confidence_mask(pose)
    kernel = np.ones(window) / window
    pad = window // 2

    for point in range(data.shape[2]):
        if not mask[:, point].any():
            continue
        for axis in range(data.shape[3]):
            track = data[:, 0, point, axis]
            padded = np.pad(track, pad, mode="edge")
            data[:, 0, point, axis] = np.convolve(padded, kernel, mode="valid")[: len(track)]

    out = pose.copy()
    out.body.data = np.ma.array(data, mask=~np.repeat(
        mask[:, None, :, None], data.shape[3], axis=3
    ))
    return out


def resample(pose: Pose, target_fps: float) -> Pose:
    """Resample to the frame rate the renderer runs at."""
    source_fps = float(pose.body.fps) or 30.0
    if abs(source_fps - target_fps) < 1e-6:
        return pose

    data = np.array(pose.body.data, dtype=np.float64)
    frames = data.shape[0]
    count = max(2, int(round(frames * target_fps / source_fps)))
    index = np.linspace(0, frames - 1, count)

    resampled = np.empty((count,) + data.shape[1:], dtype=np.float64)
    for point in range(data.shape[2]):
        for axis in range(data.shape[3]):
            resampled[:, 0, point, axis] = np.interp(
                index, np.arange(frames), data[:, 0, point, axis]
            )

    confidence = np.asarray(pose.body.confidence)[:, 0, :]
    new_confidence = np.stack(
        [np.interp(index, np.arange(frames), confidence[:, p]) for p in range(confidence.shape[1])],
        axis=1,
    )
    keep = new_confidence > 0.5

    out = pose.copy()
    out.body.data = np.ma.array(resampled, mask=~np.repeat(
        keep[:, None, :, None], data.shape[3], axis=3
    ))
    out.body.confidence = keep[:, None, :].astype(np.float32)
    out.body.fps = target_fps
    return out


def smooth(
    pose: Pose,
    *,
    target_fps: float | None = None,
    window: int = MOVING_AVERAGE_WINDOW,
    max_gap: int = MAX_GAP_FRAMES,
) -> tuple[Pose, SmoothingReport]:
    """Fill short gaps, de-jitter, and optionally resample. Returns the report
    so the decision log can show what was actually done (FR-15)."""
    if pose.body.data.shape[0] == 0:
        raise SmoothingError("empty pose sequence")

    frames_in = int(pose.body.data.shape[0])
    fps_in = float(pose.body.fps) or 30.0

    result, filled, missing = fill_gaps(pose, max_gap=max_gap)
    result = moving_average(result, window=window)
    if target_fps is not None:
        result = resample(result, target_fps)

    return result, SmoothingReport(
        frames_in=frames_in,
        frames_out=int(result.body.data.shape[0]),
        fps_in=fps_in,
        fps_out=float(result.body.fps),
        gaps_filled=filled,
        points_still_missing=missing,
    )
