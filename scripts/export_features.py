#!/usr/bin/env python3
"""Export the vocabulary as a features archive for training elsewhere.

    .venv/bin/python scripts/export_features.py

Writes `data/models/features.npz`: every clip already run through
`pose_features`, the exact function the runtime classifier uses.

Exporting features rather than clips is what makes training portable. A notebook
given this file needs numpy and torch and nothing else — no `pose-format`, no
MediaPipe (which pins `mediapipe<0.10.30` and Python 3.12, and is the most
fragile dependency in the project), and 11MB instead of 190MB of video. It also
removes a whole class of bug: features computed in two places drift, and a model
trained on a reimplementation of this function would score well in the notebook
and badly at runtime, with nothing to indicate why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition.classifier import pose_features  # noqa: E402
from backend.recognition.extract import load_pose  # noqa: E402


def main() -> int:
    vocab = ROOT / "data" / "vocab"
    out = ROOT / "data" / "models" / "features.npz"
    classes = sorted(p.name for p in vocab.iterdir() if p.is_dir() and p.name != "raw_video")

    X, y, names = [], [], []
    skipped = []
    for index, gloss in enumerate(classes):
        for path in sorted((vocab / gloss).glob("*.pose")):
            try:
                X.append(pose_features(load_pose(path)).astype(np.float32))
            except Exception as exc:
                skipped.append((path.name, str(exc)[:60]))
                continue
            y.append(index)
            names.append(path.name)

    X = np.stack(X)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=np.array(y), classes=np.array(classes), names=np.array(names)
    )

    counts = np.bincount(np.array(y), minlength=len(classes))
    print(f"{len(classes)} classes, {len(X)} clips -> {X.shape}")
    print(f"written to {out.relative_to(ROOT)} ({out.stat().st_size/1e6:.1f} MB)")
    thin = [f"{classes[i]}({c})" for i, c in enumerate(counts) if c < 5]
    if thin:
        print(f"thin classes: {', '.join(thin)}")
    if skipped:
        print(f"skipped {len(skipped)} unusable clips, e.g. {skipped[0][0]}: {skipped[0][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
