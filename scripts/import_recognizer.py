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

FEATURES = ROOT / "data" / "models" / "features.npz"


def notebook_split(seed: int):
    """Reproduce the split from `features.npz`, which is what a notebook trains on.

    Deriving it from the directory listing instead is subtly wrong, and was: one
    clip in the corpus has no usable signing segment, so `features.npz` holds 641
    rows where the disk holds 642. That one missing row shifts the shared random
    stream for every class after it alphabetically, and four clips the notebook
    had trained on landed in the test set here — reporting 78.8% where the honest
    figure was 76.1%. Splitting over the same array the model was trained on
    removes the discrepancy rather than documenting it.
    """
    blob = np.load(FEATURES, allow_pickle=True)
    y, names = blob["y"], blob["names"]
    classes = [str(c) for c in blob["classes"]]
    rng = np.random.RandomState(seed)
    test = []
    for index in range(len(classes)):
        rows = np.where(y == index)[0]
        n_test = 2 if len(rows) >= 5 else (1 if len(rows) >= 3 else 0)
        test += list(rng.permutation(rows)[:n_test])
    return blob, classes, np.array(sorted(test))

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

    # Scored on the split derived from features.npz — the same rows the notebook
    # held out — so the number here is the one the notebook should have reported.
    blob, feature_classes, test_rows = notebook_split(args.seed)
    X, y = blob["X"], blob["y"]
    train_rows = np.array([i for i in range(len(y)) if i not in set(test_rows)])

    hit = sum(
        classifier.classify_features(X[i]).gloss_id == feature_classes[y[i]]
        for i in test_rows
    )
    kept = wrong_kept = 0
    for i in test_rows:
        prediction = classifier.classify_features(X[i])
        if prediction.coverage_status.value == "lexicon_hit":
            kept += 1
            wrong_kept += prediction.gloss_id != feature_classes[y[i]]

    dtw = SignClassifier()
    for i in train_rows:
        dtw.add_template(feature_classes[y[i]], X[i])
    dhit = sum(
        dtw.classify_features(X[i]).gloss_id == feature_classes[y[i]] for i in test_rows
    )

    total = len(test_rows)
    print(f"held-out {total} clips (the split features.npz implies, seed {args.seed})")
    print(f"  imported model   top-1 {hit/total:.1%}   "
          f"reported LEXICON_HIT on {kept}, {kept-wrong_kept} right "
          f"({(kept-wrong_kept)/max(kept,1):.0%} precision)")
    print(f"  DTW baseline     top-1 {dhit/total:.1%}")

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
