#!/usr/bin/env python3
"""Rebuild the demo avatar sequence and its timeline from the lexicon.

The two files this writes drive the Stage 1 demo page, and until now nothing in
the repository produced them — they were assembled ad hoc, and had drifted from
what the code actually does. The committed pair carried a 234-frame timeline
against a 281-frame pose, so every sign's span landed about 20% early; the
sequence declared 30fps while its timings had been computed at the lexicon's 25;
and 47 frames past the last span belonged to no sign at all. Those are the
symptoms of a demo asset that no build step owns.

    .venv/bin/python scripts/build_demo_sequence.py
    .venv/bin/python scripts/build_demo_sequence.py --glosses HELLO YOU SIT PLEASE

Writes `assets/demo_sequence.pose` and `assets/demo_timeline.json`, and prints
what it did so the result can be checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.speech_to_sign.pose_smoothing import (  # noqa: E402
    build_avatar_sequence, component_slices, smooth,
)

# The sentence the Stage 1 demo page signs. Kept here rather than in the page so
# the assets and the glosses they were built from stay together.
DEMO_GLOSSES = ["HELLO", "YOU", "SIT", "PLEASE"]
RENDER_FPS = 30.0


def hand_coverage(pose) -> dict[str, float]:
    """Fraction of frames each hand is tracked — the figure that decides whether
    the avatar has fingers to drive."""
    slices = component_slices(pose)
    conf = np.asarray(pose.body.confidence)[:, 0, :]
    out = {}
    for name, label in (("LEFT_HAND_LANDMARKS", "left"), ("RIGHT_HAND_LANDMARKS", "right")):
        if name in slices:
            out[label] = float((conf[:, slices[name]] > 0).any(axis=1).mean())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glosses", nargs="+", default=DEMO_GLOSSES)
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "lexicon")
    parser.add_argument("--out", type=Path, default=ROOT / "assets")
    parser.add_argument("--fps", type=float, default=RENDER_FPS)
    args = parser.parse_args()

    pose, timeline = build_avatar_sequence(
        args.glosses, args.lexicon, target_fps=args.fps,
    )
    # Fill the short mid-sign gaps MediaPipe leaves; no resampling here, the
    # assembly already did it alongside the timeline.
    pose, report = smooth(pose)

    frames = int(pose.body.data.shape[0])
    coverage = hand_coverage(pose)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "demo_sequence.pose").open("wb") as handle:
        pose.write(handle)
    (args.out / "demo_timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")

    print(f"glosses     {' '.join(args.glosses)}")
    print(f"sequence    {frames} frames @ {pose.body.fps:g}fps "
          f"({frames / pose.body.fps:.2f}s)")
    print(f"smoothing   {report.gaps_filled} gaps filled, "
          f"{report.points_still_missing} points still missing")
    print(f"hands       " + ", ".join(f"{k} {v:.0%}" for k, v in coverage.items()))
    print("timeline")
    for entry in timeline:
        print(f"  {entry['gloss']:<10} frames {entry['start_frame']:>3}-{entry['end_frame']:<3}"
              f"  {entry['start_s']:>5.2f}s +{entry['duration_s']:.2f}s")

    covered = sum(e["end_frame"] - e["start_frame"] for e in timeline)
    last = timeline[-1]["end_frame"]
    print(f"coverage    {covered}/{frames} frames inside a sign; "
          f"{frames - last} after the last one")
    if last > frames:
        print("MISMATCH    timeline runs past the end of the pose", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
