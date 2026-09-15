#!/usr/bin/env python3
"""Download ISL sign clips and extract them into the curated vocabulary.

Nothing in the repository fetched the corpus: `data/vocab/raw_video/` and
`data/lexicon_src/` are gitignored and were populated by hand, so the vocabulary
could not be rebuilt or extended from a clean checkout. This is that step.

    .venv/bin/python scripts/fetch_vocabulary.py --list
    .venv/bin/python scripts/fetch_vocabulary.py --all
    .venv/bin/python scripts/fetch_vocabulary.py friend school market

Source: `vidit031/isl-isolated-40words` on Hugging Face — ungated, 642 H.264
clips over 40 glosses, aggregated from ISL500, INCLUDE, CISLR and the ISLRTC
dictionary, with per-clip provenance in the filename. That provenance matters
downstream: clips from different upstream corpora are framed differently, and
the recognition model is sensitive to it.

Videos land in `data/vocab/raw_video/<GLOSS>/` and poses in
`data/vocab/<GLOSS>/`, matching the layout `build_lexicon()` already expects.
Poses are extracted without face landmarks, as recognition requires; the lexicon
builder re-extracts its own chosen clip with the full component set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = "vidit031/isl-isolated-40words"
API = f"https://huggingface.co/api/datasets/{DATASET}"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"


def gloss_dir_name(gloss: str) -> str:
    """`thank_you` -> `THANK-YOU`, matching the directories already on disk."""
    return gloss.strip().upper().replace("_", "-").replace(" ", "-")


def catalogue() -> dict[str, list[str]]:
    with urllib.request.urlopen(API, timeout=60) as response:
        meta = json.load(response)
    clips: dict[str, list[str]] = {}
    for sibling in meta["siblings"]:
        name = sibling["rfilename"]
        if name.endswith(".mp4") and "/" in name:
            clips.setdefault(name.split("/")[0], []).append(name)
    return {k: sorted(v) for k, v in sorted(clips.items())}


def fetch(remote: str, destination: Path, retries: int = 3) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(RESOLVE + remote, destination)
            return True
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries - 1:
                raise SystemExit(f"failed to download {remote}: {exc}")
            time.sleep(2 * (attempt + 1))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("glosses", nargs="*", help="dataset gloss names, e.g. friend school")
    parser.add_argument("--all", action="store_true", help="every gloss not already present")
    parser.add_argument("--list", action="store_true", help="show the catalogue and stop")
    parser.add_argument("--no-extract", action="store_true", help="download only")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab")
    args = parser.parse_args()

    clips = catalogue()
    present = {p.name for p in args.vocab.iterdir() if p.is_dir() and p.name != "raw_video"}

    if args.list:
        print(f"{DATASET}: {sum(len(v) for v in clips.values())} clips, {len(clips)} glosses\n")
        for gloss, files in clips.items():
            mark = "have" if gloss_dir_name(gloss) in present else "    "
            print(f"  {mark}  {gloss:<12} {len(files):>3} clips")
        return 0

    if args.all:
        wanted = [g for g in clips if gloss_dir_name(g) not in present]
    else:
        wanted = [g for g in args.glosses if g in clips]
        unknown = [g for g in args.glosses if g not in clips]
        if unknown:
            raise SystemExit(f"not in the dataset: {unknown}. Try --list.")
    if not wanted:
        print("nothing to do — every requested gloss is already extracted.")
        return 0

    total = sum(len(clips[g]) for g in wanted)
    print(f"{len(wanted)} gloss(es), {total} clips: {', '.join(wanted)}\n")

    extract = None
    if not args.no_extract:
        from backend.recognition.extract import extract_pose_file
        extract = extract_pose_file

    downloaded = extracted = skipped = failed = 0
    started = time.time()
    for gloss in wanted:
        directory = gloss_dir_name(gloss)
        for remote in clips[gloss]:
            stem = Path(remote).stem
            video = args.vocab / "raw_video" / directory / f"{stem}.mp4"
            pose = args.vocab / directory / f"{stem}.pose"
            downloaded += fetch(remote, video)
            if extract is None or pose.exists():
                skipped += pose.exists()
                continue
            pose.parent.mkdir(parents=True, exist_ok=True)
            try:
                extract(video, pose)
                extracted += 1
            except Exception as exc:                      # a bad clip is not fatal
                failed += 1
                print(f"  ! {stem}: {str(exc)[:80]}")
        print(f"  {directory:<12} {len(clips[gloss]):>3} clips  "
              f"({time.time() - started:.0f}s elapsed)")

    print(f"\ndownloaded {downloaded}, extracted {extracted}, already present {skipped}, "
          f"failed {failed}, in {time.time() - started:.0f}s")
    print("next: .venv/bin/python -c \"from pathlib import Path; "
          "from backend.speech_to_sign.gloss_lookup import build_lexicon; "
          "build_lexicon(Path('data/vocab'), Path('data/lexicon'))\" to extend the lexicon")
    return 1 if failed and not extracted else 0


if __name__ == "__main__":
    raise SystemExit(main())
