#!/usr/bin/env python3
"""Build the ISL fingerspelling handshape library (T2.6).

Stage 1 shipped `FINGERSPELLING` as a defined-but-unreachable coverage tier:
`spoken-to-signed-translation` bundles no ISL lexicon (`ins` is absent, `ise` is
Italian), so every out-of-vocabulary term fell straight to `UNMATCHED`. This
script produces the 26 `.pose` assets that make the tier real.

**Source, and why the planning docs' exclusion no longer holds.** Stage 2's
plan ruled out every known ISL fingerspelling dataset, including
`kirandevraj/ISL-Fingerspelling`, on the grounds that it is "unaligned
news-broadcast video with no extracted poses and no confirmed license". Checked
directly against the dataset on 2026-09-15, two of those three claims are out
of date:

- It **is** aligned, to the frame, per letter. `letter_annotations.csv` carries
  14,685 letter instances with `start_frame`/`end_frame` against each clip.
- Its license **is** stated: CC-BY-NC-4.0. Non-commercial — fine for this
  project's research/demo use, and recorded in the generated manifest so the
  constraint travels with the assets rather than living in a build script.

What remains true is that it ships no poses. That is what this script does, via
T1.3's existing extraction pipeline, unmodified — the same path `build_lexicon`
already uses for the 16-sign render lexicon, with the full MediaPipe component
set retained because the downstream smoothing step indexes face landmarks.

**Why one signer.** All three signers in the corpus cover all 26 letters;
`signer_00` has the most instances. Taking every letter from one signer keeps
hand proportions and signing-space placement consistent across a spelled word,
which matters when the letters are concatenated into a sequence the avatar
plays back-to-back.

**Why the middle of each span.** These are letters cut from *continuous*
fingerspelling, so each span's edges are coarticulation — the hand travelling
from the previous letter and towards the next. The central portion is the
actual held handshape. Trimming to it is what makes an arbitrary concatenation
look like spelling rather than a stutter.

    .venv/bin/python scripts/build_fingerspelling.py --plan
    .venv/bin/python scripts/build_fingerspelling.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = "kirandevraj/ISL-Fingerspelling"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
LICENSE = "CC-BY-NC-4.0"
PAPER = "Kirandevraj et al., ACL WSLP 2025 (arXiv:2407.05404)"

PREFERRED_SIGNER = "signer_00"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# A span shorter than this is a letter rushed through in connected spelling —
# too few frames to read as a distinct handshape once played back.
MIN_SPAN_FRAMES = 6
# Keep the central portion of each span; the rest is travel to and from the
# neighbouring letters.
HOLD_FRACTION = 0.6
MIN_HOLD_FRAMES = 4

OUT_DIR = ROOT / "data" / "fingerspelling"
CACHE = ROOT / "data" / "fingerspelling" / ".source_video"


@dataclass
class Candidate:
    letter: str
    uid: str
    video_id: str
    start_frame: int
    end_frame: int
    fps: float

    @property
    def span(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def hold(self) -> tuple[int, int]:
        """The central held portion of the span, in frames."""
        keep = max(int(round(self.span * HOLD_FRACTION)), MIN_HOLD_FRAMES)
        keep = min(keep, self.span)
        pad = (self.span - keep) // 2
        return self.start_frame + pad, self.start_frame + pad + keep


def _download(remote: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RESOLVE + remote, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def _read_csv(name: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(RESOLVE + name, timeout=120) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode())))


def choose_exemplars(signer: str = PREFERRED_SIGNER) -> dict[str, Candidate]:
    """Pick the clearest instance of each letter: longest held span, one signer."""
    rows = _read_csv("letter_timestamps.csv")
    by_letter: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        letter = row["letter"].strip().lower()
        if letter not in ALPHABET or row["signer"] != signer:
            continue
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if end - start < MIN_SPAN_FRAMES:
            continue
        by_letter[letter].append(
            Candidate(letter, row["uid"], row["video_id"], start, end, float(row["fps"]))
        )

    chosen: dict[str, Candidate] = {}
    for letter in ALPHABET:
        options = by_letter.get(letter, [])
        if not options:
            continue
        # Longest span = the most deliberately held handshape.
        chosen[letter] = max(options, key=lambda c: c.span)
    return chosen


def build(chosen: dict[str, Candidate], *, out_dir: Path = OUT_DIR) -> list[dict]:
    from pose_format import Pose

    from backend.recognition.extract import extract_pose_file, load_pose

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # One extraction per source clip, however many letters come from it.
    by_uid: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in chosen.values():
        by_uid[candidate.uid].append(candidate)

    for index, (uid, candidates) in enumerate(sorted(by_uid.items()), start=1):
        letters = "".join(sorted(c.letter for c in candidates))
        print(f"[{index}/{len(by_uid)}] {uid} -> {letters}")
        video = _download(f"videos/{uid}.mp4", CACHE / f"{uid}.mp4")
        pose_path = CACHE / f"{uid}.pose"
        if not pose_path.exists():
            # T1.3's pipeline, called not modified. components=None keeps the
            # full MediaPipe set, matching what build_lexicon() writes.
            extract_pose_file(video, pose_path, components=None)
        source = load_pose(pose_path)

        for candidate in sorted(candidates, key=lambda c: c.letter):
            start, end = candidate.hold
            end = min(end, int(source.body.data.shape[0]))
            if end - start < MIN_HOLD_FRAMES:
                print(f"    skip {candidate.letter!r}: only {end-start} usable frames")
                continue

            clipped = Pose(
                header=source.header,
                body=source.body[start:end],
            )
            destination = out_dir / f"{candidate.letter}.pose"
            with destination.open("wb") as handle:
                clipped.write(handle)

            records.append({
                "letter": candidate.letter,
                "frames": end - start,
                "fps": candidate.fps,
                "duration_s": round((end - start) / candidate.fps, 3),
                "source_uid": uid,
                "source_video_id": candidate.video_id,
                "source_span": [candidate.start_frame, candidate.end_frame],
                "hold_span": [start, end],
                "file": destination.name,
            })
            print(f"    {candidate.letter}  frames {start}-{end} "
                  f"({(end-start)/candidate.fps:.2f}s) -> {destination.name}")
    return records


def write_manifest(records: list[dict], *, out_dir: Path = OUT_DIR) -> Path:
    """Write the human-readable manifest, same pattern as T1.2's vocabulary one."""
    covered = {r["letter"] for r in records}
    missing = [c for c in ALPHABET if c not in covered]
    total = sum(r["duration_s"] for r in records)

    lines = [
        "# ISL fingerspelling handshape library",
        "",
        "Generated by `scripts/build_fingerspelling.py`. Reviewed by hand before use —",
        "regenerating this file does not substitute for looking at the poses.",
        "",
        "## Provenance",
        "",
        f"- **Source:** `{DATASET}` on Hugging Face — {PAPER}",
        f"- **License:** {LICENSE}. **Non-commercial.** This constraint travels with",
        "  these assets: the library may be used for this project's research and demo",
        "  use, and may not be used commercially without separate permission.",
        f"- **Signer:** `{PREFERRED_SIGNER}`, one signer for all letters, so hand",
        "  proportions and signing-space placement stay consistent across a spelled word.",
        "- **Extraction:** T1.3's `extract.py` pipeline unmodified, full MediaPipe",
        "  component set retained (the downstream smoothing step indexes face landmarks).",
        "",
        "## What these clips are, and are not",
        "",
        "These are letters cut from **continuous** fingerspelling, not isolated poses",
        "recorded letter by letter. Each entry is the central held portion of its span —",
        f"{int(HOLD_FRACTION*100)}% of the annotated letter, with the coarticulated travel to and from",
        "the neighbouring letters trimmed off both ends.",
        "",
        "That has one real consequence worth stating: a letter's held shape here is the",
        "shape it takes *in the middle of a word*, which for some letters is not identical",
        "to its citation form signed alone. The trade is deliberate — concatenating these",
        "produces spelling that moves like spelling, where citation forms concatenated",
        "back-to-back read as a sequence of unrelated stills.",
        "",
        "ISL's manual alphabet is two-handed, and several letters are genuinely dynamic",
        "rather than static. Both properties survive here because each entry is a frame",
        "sequence, not a single pose — the same way the rest of the vocabulary handles",
        "motion.",
        "",
        "## Coverage",
        "",
        f"- Letters: **{len(covered)}/26**" + (f" — missing: {', '.join(missing)}" if missing else " — complete"),
        f"- Total held material: {total:.2f}s across {len(records)} clips",
        "",
        "| Letter | File | Frames | Duration | Source clip | Span in source |",
        "|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda r: r["letter"]):
        lines.append(
            f"| {record['letter'].upper()} | `{record['file']}` | {record['frames']} | "
            f"{record['duration_s']:.2f}s | `{record['source_uid']}` | "
            f"{record['source_span'][0]}–{record['source_span'][1]} |"
        )
    lines.append("")

    path = out_dir / "manifest.md"
    path.write_text("\n".join(lines))
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signer", default=PREFERRED_SIGNER)
    parser.add_argument("--plan", action="store_true", help="choose exemplars and stop")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    chosen = choose_exemplars(args.signer)
    print(f"chose {len(chosen)}/26 letters from {args.signer}, "
          f"{len({c.uid for c in chosen.values()})} source clip(s)")
    for letter in ALPHABET:
        candidate = chosen.get(letter)
        if candidate is None:
            print(f"  {letter}: NO CANDIDATE")
        else:
            start, end = candidate.hold
            print(f"  {letter}: {candidate.uid} span {candidate.span}f -> hold {end-start}f")
    if args.plan:
        return 0

    records = build(chosen, out_dir=args.out)
    manifest = write_manifest(records, out_dir=args.out)
    print(f"\n{len(records)} handshapes written; manifest at {manifest.relative_to(ROOT)}")
    return 0 if len(records) == 26 else 1


if __name__ == "__main__":
    raise SystemExit(main())
