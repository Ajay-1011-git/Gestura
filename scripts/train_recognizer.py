#!/usr/bin/env python3
"""Train the sign recogniser and report how it compares to the DTW baseline.

    .venv/bin/python scripts/train_recognizer.py                 # train and save
    .venv/bin/python scripts/train_recognizer.py --evaluate      # held-out comparison

Training takes about a minute on an M-series Mac and needs no GPU and no
downloaded checkpoint. `--evaluate` holds clips out of *both* classifiers and
scores them on the same split, because the only number worth quoting is the one
measured against what the project already had.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition.classifier import SignClassifier, pose_features  # noqa: E402
from backend.recognition.extract import load_pose  # noqa: E402
from backend.recognition.neural import NeuralSignClassifier  # noqa: E402

DEFAULT_MODEL = ROOT / "data" / "models" / "recognizer.pt"


def held_out(vocab: Path, seed: int) -> tuple[list[str], dict, dict]:
    """Two clips per class where the class can spare them, one where it can't."""
    classes = sorted(p.name for p in vocab.iterdir() if p.is_dir() and p.name != "raw_video")
    rng = np.random.RandomState(seed)
    train, test = {}, {}
    for gloss in classes:
        clips = sorted((vocab / gloss).glob("*.pose"))
        n_test = 2 if len(clips) >= 5 else (1 if len(clips) >= 3 else 0)
        order = rng.permutation(len(clips))
        test[gloss] = [clips[i] for i in order[:n_test]]
        train[gloss] = [clips[i] for i in order[n_test:]]
    return classes, train, test


def score(predict, test: dict) -> tuple[int, int, int, int]:
    hit = total = confident_right = confident_wrong = 0
    for gloss, paths in test.items():
        for path in paths:
            try:
                prediction = predict(pose_features(load_pose(path)))
            except Exception:
                continue
            total += 1
            correct = prediction.gloss_id == gloss
            hit += correct
            if prediction.confidence >= 0.22:
                confident_right += correct
                confident_wrong += not correct
    return hit, total, confident_right, confident_wrong


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", action="store_true",
                        help="hold clips out and compare against the DTW baseline")
    args = parser.parse_args()

    if not args.evaluate:
        model = NeuralSignClassifier()
        report = model.fit_directory(args.vocab, epochs=args.epochs, progress=True)
        model.save(args.out)
        print(f"\n{len(report.classes)} classes, {report.train_clips} train / "
              f"{report.val_clips} val clips")
        print(f"validation accuracy {report.val_accuracy:.1%} in {report.seconds:.0f}s")
        print(f"saved to {args.out.relative_to(ROOT)}")
        return 0

    classes, train, test = held_out(args.vocab, args.seed)
    n_test = sum(len(v) for v in test.values())
    thin = [c for c in classes if not test[c]]
    print(f"{len(classes)} classes | {sum(len(v) for v in train.values())} train "
          f"| {n_test} test")
    if thin:
        print(f"too few clips to hold any out: {', '.join(thin)}")

    excluded = [p for paths in test.values() for p in paths]

    print("\nDTW template matching:")
    started = time.time()
    dtw = SignClassifier()
    for gloss in classes:
        for path in train[gloss]:
            try:
                dtw.add_template(gloss, pose_features(load_pose(path)))
            except Exception:
                pass
    hit, total, right, wrong = score(dtw.classify_features, test)
    print(f"  top-1 {hit}/{total} = {hit/total:.1%}   above 0.22: {right} right / {wrong} wrong"
          f"   ({time.time()-started:.0f}s)")

    print("\nlearned recogniser:")
    started = time.time()
    neural = NeuralSignClassifier()
    report = neural.fit_directory(args.vocab, exclude=excluded, epochs=args.epochs,
                                  seed=args.seed)
    nhit, ntotal, nright, nwrong = score(neural.classify_features, test)
    print(f"  top-1 {nhit}/{ntotal} = {nhit/ntotal:.1%}   above 0.22: {nright} right / "
          f"{nwrong} wrong   ({time.time()-started:.0f}s, val {report.val_accuracy:.1%})")

    delta = nhit / ntotal - hit / total
    print(f"\ndifference: {delta:+.1%} top-1 in favour of "
          f"{'the learned model' if delta > 0 else 'DTW'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
