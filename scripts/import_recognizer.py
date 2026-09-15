#!/usr/bin/env python3
"""Validate a recogniser trained elsewhere, score it, then install it.

    .venv/bin/python scripts/import_recognizer.py ~/Downloads/recognizer.pt

A checkpoint that loads is not the same as a checkpoint that works. This one
checks the class list against the vocabulary actually on disk, confirms the
weights fit the architecture the runtime uses, and scores it on held-out clips
before copying it into place — so a notebook that silently trained on a
different feature layout, a stale vocabulary, or a reimplemented
`pose_features` is caught here rather than during a demo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition.classifier import SignClassifier, pose_features  # noqa: E402
from backend.recognition.extract import load_pose  # noqa: E402
from backend.recognition.neural import NeuralSignClassifier, SignNet  # noqa: E402
from scripts.train_recognizer import held_out, score  # noqa: E402

DEST = ROOT / "data" / "models" / "recognizer.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true",
                        help="install even if it does not beat the DTW baseline")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"no such file: {args.checkpoint}")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key in ("classes", "state"):
        if key not in blob:
            raise SystemExit(
                f"checkpoint has no {key!r} — expected "
                '{"classes": [...], "state": state_dict}'
            )
    trained = list(blob["classes"])

    on_disk = sorted(
        p.name for p in args.vocab.iterdir() if p.is_dir() and p.name != "raw_video"
    )
    if trained != on_disk:
        missing = sorted(set(on_disk) - set(trained))
        extra = sorted(set(trained) - set(on_disk))
        print(f"class list differs from {args.vocab.relative_to(ROOT)}:")
        if missing:
            print(f"  not in the checkpoint: {', '.join(missing)}")
        if extra:
            print(f"  not in the vocabulary: {', '.join(extra)}")
        raise SystemExit("refusing to install — the model would predict labels that "
                         "do not correspond to this vocabulary")

    model = SignNet(len(trained))
    try:
        model.load_state_dict(blob["state"])
    except RuntimeError as exc:
        raise SystemExit(f"weights do not fit SignNet — was it trained from a different "
                         f"architecture?\n{exc}")

    classifier = NeuralSignClassifier(trained)
    classifier.model = model.to(classifier.device).eval()
    print(f"loaded: {len(trained)} classes, "
          f"{sum(p.numel() for p in model.parameters())/1e3:.0f}k parameters\n")

    # Score it the same way the training script does, on the same split.
    classes, train, test = held_out(args.vocab, args.seed)
    hit, total, right, wrong = score(classifier.classify_features, test)

    dtw = SignClassifier()
    for gloss in classes:
        for path in train[gloss]:
            try:
                dtw.add_template(gloss, pose_features(load_pose(path)))
            except Exception:
                pass
    dhit, dtotal, dright, dwrong = score(dtw.classify_features, test)

    print(f"held-out {total} clips")
    print(f"  imported model   top-1 {hit/total:.1%}   "
          f"above 0.22: {right} right / {wrong} wrong")
    print(f"  DTW baseline     top-1 {dhit/dtotal:.1%}   "
          f"above 0.22: {dright} right / {dwrong} wrong")

    # A caveat worth printing rather than hiding: if the notebook held out the
    # same clips, they were in its training set here, and this is optimistic.
    print("\nnote: if the notebook used the same split, these test clips were in its\n"
          "      training data and this number is optimistic. Re-run the notebook's\n"
          "      own evaluation for the honest figure.")

    if hit <= dhit and not args.force:
        print(f"\nnot installing — it does not beat DTW. Pass --force to install anyway.")
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.resolve() == DEST.resolve():
        print(f"\nalready installed at {DEST.relative_to(ROOT)}")
        return 0
    shutil.copy2(args.checkpoint, DEST)
    print(f"\ninstalled to {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
