"""Curated-vocabulary sign classifier (T1.4).

Approach, per the task's preference for a simple low-risk method over adapting
an external recognition codebase: normalize landmarks, trim each clip to its
active signing segment, resample to a fixed length, and classify by DTW
nearest-neighbour against the reference clips extracted in T1.3.

Confidence is a real calibrated value derived from the margin between the best
and second-best class distances, never a fixed placeholder (FR-2), and is mapped
onto the shared :class:`CoverageStatus` buckets so both directions speak one
status vocabulary (contracts §B.2).

Feature layout matches AI4Bharat's INCLUDE keypoint models — pose[0:25] plus
both 21-point hands, x/y only, 67 keypoints = 134 dims per frame — so a
pretrained INCLUDE backbone can be warm-started against these features later
without changing the extraction path.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pose_format import Pose

from backend.contracts import CoverageStatus, Direction, Segment
from backend.recognition.extract import load_pose

# Component offsets inside the trimmed .pose written by extract.py:
# POSE_LANDMARKS 0-32, LEFT_HAND_LANDMARKS 33-53, RIGHT_HAND_LANDMARKS 54-74.
POSE_SLICE = slice(0, 25)      # upper body only, matching MediaPipe 0.8's upper_body_only
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

RESAMPLE_FRAMES = 32
MIN_ACTIVE_FRAMES = 8
MAX_DROPOUT_GAP = 5

# Calibrated against a real stratified held-out run (47 clips, 16 glosses,
# 278 templates), not guessed. Measured margin distribution: correct
# predictions mean 0.283, incorrect mean 0.102 with a maximum of 0.212.
#
#   threshold   precision   recall of correct
#      0.22       1.000          0.47
#      0.20       0.950          0.56
#      0.10       0.806          0.74
#
# 0.22 is chosen because it is the lowest threshold at which precision is
# perfect on held-out data: nothing labelled LEXICON_HIT was actually wrong.
# The project's governing principle is that uncertainty must never be smoothed
# into confident-looking output (TRD §2), so precision is worth more here than
# recall — a segment below the bar escalates to clarification, which is a
# designed behaviour rather than a failure.
LEXICON_HIT_THRESHOLD = 0.22
LANGUAGE_BACKUP_THRESHOLD = 0.10


class ClassifierError(RuntimeError):
    """Classification could not be performed on the given input."""


def active_segment(mask: np.ndarray, max_gap: int = MAX_DROPOUT_GAP) -> tuple[int, int]:
    """Longest run of hand-present frames, tolerating short dropouts.

    Dictionary and corpus clips both begin and end with the signer's hands at
    rest, which is not signal. Measuring detection across a whole clip therefore
    understates quality badly; the signing itself is one contiguous run.
    """
    text = "".join("1" if bit else "0" for bit in mask)
    best = (0, 0, 0)
    for match in re.finditer(r"1(?:[01]{0,%d}1)*" % max_gap, text):
        span = match.end() - match.start()
        if span > best[0]:
            best = (span, match.start(), match.end())
    return best[1], best[2]


def _hand_present(pose: Pose) -> np.ndarray:
    conf = pose.body.confidence
    left = conf[:, 0, LEFT_HAND_SLICE].sum(axis=1) > 0
    right = conf[:, 0, RIGHT_HAND_SLICE].sum(axis=1) > 0
    return np.asarray(left | right)


def pose_features(pose: Pose, *, resample: int = RESAMPLE_FRAMES) -> np.ndarray:
    """Turn a `Pose` into a normalized ``(resample, 134)`` feature array."""
    start, end = active_segment(_hand_present(pose))
    if end - start < MIN_ACTIVE_FRAMES:
        raise ClassifierError(
            f"no usable signing segment (found {end - start} frames, "
            f"need {MIN_ACTIVE_FRAMES})"
        )

    data = np.asarray(pose.body.data[start:end, 0, :, :2], dtype=np.float64)
    conf = np.asarray(pose.body.confidence[start:end, 0, :])

    body = data[:, POSE_SLICE, :]
    left = data[:, LEFT_HAND_SLICE, :]
    right = data[:, RIGHT_HAND_SLICE, :]

    # A hand MediaPipe did not find lands at the origin, which reads as a huge
    # jump to the wrist's actual position. Collapse it onto the wrist instead so
    # a missing hand is a degenerate point, not a spurious trajectory.
    for hand, wrist, sl in (
        (left, LEFT_WRIST, LEFT_HAND_SLICE),
        (right, RIGHT_WRIST, RIGHT_HAND_SLICE),
    ):
        missing = conf[:, sl].sum(axis=1) == 0
        if missing.any():
            hand[missing] = body[missing, wrist : wrist + 1, :]

    frames = np.concatenate([body, left, right], axis=1)  # (T, 67, 2)

    # Shoulder-centred, shoulder-width-scaled: cancels camera distance and
    # framing, which differ wildly between the corpora feeding this vocabulary.
    centre = (body[:, LEFT_SHOULDER, :] + body[:, RIGHT_SHOULDER, :]) / 2.0
    width = np.linalg.norm(
        body[:, LEFT_SHOULDER, :] - body[:, RIGHT_SHOULDER, :], axis=1
    )
    width = np.where(width < 1e-6, 1.0, width)
    frames = (frames - centre[:, None, :]) / width[:, None, None]

    flat = frames.reshape(frames.shape[0], -1)  # (T, 134)
    idx = np.linspace(0, flat.shape[0] - 1, resample)
    return np.stack(
        [np.interp(idx, np.arange(flat.shape[0]), flat[:, d]) for d in range(flat.shape[1])],
        axis=1,
    )


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping distance between two ``(T, D)`` feature arrays."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return float(acc[n, m] / (n + m))


@dataclass
class Prediction:
    gloss_id: str
    confidence: float
    coverage_status: CoverageStatus
    runner_up: str
    best_distance: float
    runner_up_distance: float


def confidence_to_coverage(
    confidence: float,
    *,
    lexicon_hit: float = LEXICON_HIT_THRESHOLD,
    language_backup: float = LANGUAGE_BACKUP_THRESHOLD,
) -> CoverageStatus:
    """Map a recogniser's confidence onto the shared four-tier status.

    The thresholds are arguments because they are a property of the classifier,
    not of the status vocabulary. DTW's confidence is a normalised distance
    margin and the learned recogniser's is a softmax margin; the two are on
    different scales even though both run 0-1, and calibrating one against the
    other's numbers would silently change how often the waterfall escalates.
    """
    if confidence >= lexicon_hit:
        return CoverageStatus.LEXICON_HIT
    if confidence >= language_backup:
        return CoverageStatus.LANGUAGE_BACKUP
    return CoverageStatus.UNMATCHED


class SignClassifier:
    """DTW nearest-neighbour over the curated vocabulary's reference clips."""

    def __init__(self) -> None:
        self._templates: list[tuple[str, np.ndarray]] = []

    @property
    def vocabulary(self) -> list[str]:
        return sorted({gloss for gloss, _ in self._templates})

    def add_template(self, gloss_id: str, features: np.ndarray) -> None:
        self._templates.append((gloss_id, features))

    def fit_directory(
        self, vocab_dir: Path, *, exclude: Iterable[Path] = ()
    ) -> tuple[int, list[tuple[Path, str]]]:
        """Load every ``<gloss_id>/*.pose`` under ``vocab_dir`` as a template."""
        skip = {p.resolve() for p in exclude}
        failures: list[tuple[Path, str]] = []
        for gloss_dir in sorted(p for p in vocab_dir.iterdir() if p.is_dir()):
            if gloss_dir.name == "raw_video":
                continue
            for path in sorted(gloss_dir.glob("*.pose")):
                if path.resolve() in skip:
                    continue
                try:
                    self.add_template(gloss_dir.name, pose_features(load_pose(path)))
                except (ClassifierError, Exception) as exc:
                    failures.append((path, str(exc)[:120]))
        return len(self._templates), failures

    def classify_features(self, features: np.ndarray) -> Prediction:
        if not self._templates:
            raise ClassifierError("classifier has no templates — call fit_directory first")

        per_class: dict[str, float] = {}
        for gloss, template in self._templates:
            distance = dtw_distance(features, template)
            if distance < per_class.get(gloss, np.inf):
                per_class[gloss] = distance

        ranked = sorted(per_class.items(), key=lambda kv: kv[1])
        (best_gloss, best_d) = ranked[0]
        (second_gloss, second_d) = ranked[1] if len(ranked) > 1 else (best_gloss, best_d)

        # Calibrated on the real separation between the top two classes: a clear
        # winner scores high, a near-tie scores low. This is what makes demo
        # scene 2's ambiguity genuine rather than staged.
        confidence = 0.0 if second_d <= 0 else float((second_d - best_d) / second_d)
        confidence = max(0.0, min(1.0, confidence))

        return Prediction(
            gloss_id=best_gloss,
            confidence=confidence,
            coverage_status=confidence_to_coverage(confidence),
            runner_up=second_gloss,
            best_distance=best_d,
            runner_up_distance=second_d,
        )

    def classify(self, pose: Pose) -> Segment:
        """Classify a pose sequence into a `Segment` for the escalation waterfall."""
        prediction = self.classify_features(pose_features(pose))
        return Segment(
            id=str(uuid.uuid4()),
            direction=Direction.SIGN_TO_SPEECH,
            raw_input=prediction.gloss_id,
            confidence=prediction.confidence,
            coverage_status=prediction.coverage_status,
            timestamp=time.time(),
        )
