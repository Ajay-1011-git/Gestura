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
from pathlib import Path

import numpy as np
from pose_format import Pose

from backend.recognition.extract import load_pose

MAX_GAP_FRAMES = 6          # ~200ms at 30fps; longer gaps stay missing
MOVING_AVERAGE_WINDOW = 3   # odd, small — enough to take jitter off, not motion
MIN_SEGMENT_FRAMES = 8      # below this, use the whole clip rather than a sliver


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


def component_slices(pose: Pose) -> dict[str, slice]:
    """Map component name -> keypoint slice.

    Necessary because keypoint offsets are **not** fixed across this project:
    recognition poses are trimmed to 75 keypoints while lexicon poses carry the
    full 576. Hardcoding offsets silently reads the face mesh as hands on the
    other layout, which is a bug that produces plausible-looking wrong answers
    rather than an error.
    """
    slices: dict[str, slice] = {}
    offset = 0
    for component in pose.header.components:
        slices[component.name] = slice(offset, offset + len(component.points))
        offset += len(component.points)
    return slices


def hand_present(pose: Pose) -> np.ndarray:
    """Per-frame mask of whether either hand is tracked, by component name."""
    slices = component_slices(pose)
    conf = np.asarray(pose.body.confidence)[:, 0, :]
    mask = np.zeros(conf.shape[0], dtype=bool)
    for name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
        if name in slices:
            mask |= conf[:, slices[name]].sum(axis=1) > 0
    return mask


def build_avatar_sequence(
    glosses: "list[str]",
    lexicon_dir: "Path",
    *,
    transition_s: float = 0.30,
    hold_s: float = 0.35,
) -> "tuple[Pose, list[dict]]":
    """Concatenate lexicon poses for the avatar, **preserving real 3D depth**.

    Deliberately bypasses `spoken-to-signed-translation`'s own concatenation.
    That pipeline normalises and rescales for 2D stick-figure rendering, which
    flattens Z to a constant (measured: every landmark at Z≈256 in a 512px
    frame) and drops hand tracking to 31% of frames. Both are fatal for driving
    a 3D rig — a solver handed a degenerate depth plane returns the same
    rotation every frame, and absent hands mean no finger articulation at all.

    Reading the lexicon `.pose` files directly keeps MediaPipe's real output:
    POSE_LANDMARKS Z spanning ~1.2 normalised units, and hands tracked in
    93-100% of signing frames.

    Each sign is trimmed to its active signing segment, held briefly so it reads
    as a discrete sign rather than a blur, and linked to the next by an
    interpolated transition.
    """
    import csv as _csv

    from backend.recognition.classifier import active_segment

    index = {}
    with (lexicon_dir / "index.csv").open() as handle:
        for row in _csv.DictReader(handle):
            index[row["glosses"].upper()] = row["path"]

    clips: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    timeline: list[dict] = []
    header = None
    fps = 30.0
    frame_cursor = 0

    for gloss in glosses:
        path = index.get(gloss.upper())
        if path is None:
            continue
        pose = load_pose(lexicon_dir / path)
        if header is None:
            header = pose.header
            fps = float(pose.body.fps) or 30.0

        # Tight gap tolerance: at max_gap=1 every lexicon sign has 1.00 hand
        # coverage inside the window, so the avatar is never driven from frames
        # where the hands were never tracked.
        start, end = active_segment(hand_present(pose), max_gap=1)
        if end - start < MIN_SEGMENT_FRAMES:
            start, end = 0, pose.body.data.shape[0]

        data = np.array(pose.body.data[start:end], dtype=np.float64)
        conf = np.asarray(pose.body.confidence)[start:end]

        hold_frames = int(hold_s * fps)
        if hold_frames > 0 and len(data):
            data = np.concatenate([data, np.repeat(data[-1:], hold_frames, axis=0)])
            conf = np.concatenate([conf, np.repeat(conf[-1:], hold_frames, axis=0)])

        if clips:
            steps = max(1, int(transition_s * fps))
            a, b = clips[-1][-1], data[0]
            blend = np.stack([a + (b - a) * ((i + 1) / (steps + 1)) for i in range(steps)])
            gate = np.minimum(confidences[-1][-1], conf[0])
            clips.append(blend)
            confidences.append(np.repeat(gate[None], steps, axis=0))
            frame_cursor += steps

        timeline.append({
            "gloss": gloss.upper(),
            "start_frame": frame_cursor,
            "end_frame": frame_cursor + len(data),
            "start_s": round(frame_cursor / fps, 3),
            "duration_s": round(len(data) / fps, 3),
        })
        frame_cursor += len(data)
        clips.append(data)
        confidences.append(conf)

    if not clips:
        raise SmoothingError(f"no lexicon entries matched {glosses}")

    combined = Pose(header, load_pose(lexicon_dir / index[timeline[0]["gloss"]]).body)
    combined.body.data = np.ma.array(np.concatenate(clips))
    combined.body.confidence = np.concatenate(confidences)
    combined.body.fps = fps
    return combined, timeline


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
