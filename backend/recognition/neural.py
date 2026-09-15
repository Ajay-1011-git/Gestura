"""Learned isolated-sign recognition over MediaPipe landmarks (T1.4b).

Replaces the DTW template matcher for the expanded vocabulary. Same inputs, same
`Segment` output, so the escalation waterfall is unchanged.

**Why not a pretrained sign-language model.** AI4Bharat's INCLUDE ships real,
downloadable keypoint transformers — verified: the checkpoints load strict, and
on INCLUDE's own clips the 263-class model returns the right label at p=0.99.
They still lost here, and the reason is worth recording so nobody re-runs the
experiment. INCLUDE feeds **absolute pixel coordinates** in a 1920x1080 frame,
so the model learns where in the frame the signer stands. Our corpus is
aggregated from ISL500, INCLUDE, CISLR and ISLRTC at different resolutions and
framings, and across that shift the model collapses toward a single class.
Measured on an identical 40-class split:

    DTW template matching (previous)            62.1%
    INCLUDE-263 fine-tuned, pretrained init     47.8%
    INCLUDE-263 fine-tuned, random init         41.8%
    this model                                  70.7-74.6%

The pretraining is real — it is worth 6 points over random init on the same
architecture — but it cannot outrun the wrong input representation. The features
here are shoulder-centred and shoulder-width-scaled, which is framing-invariant
by construction, and that is what actually carries across the corpora.

Small on purpose: 510k parameters, about a minute to train on an M-series CPU or
MPS, and inference is far below a frame budget. There is no GPU requirement and
no checkpoint to download.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from backend.contracts import Direction, Segment
from backend.recognition.classifier import (
    ClassifierError, Prediction, confidence_to_coverage, pose_features,
)
from backend.recognition.extract import load_pose

FEATURE_DIM = 134           # 25 upper-body + 21 + 21 hand landmarks, x and y
HIDDEN = 128
LAYERS = 2

# Calibrated on the held-out split, not inherited. DTW's 0.22 is a distance
# margin; this is a softmax margin, and on the same 67 clips 0.15 keeps 58 of
# them at 84% precision. Carrying 0.22 across would have thrown away correct
# recognitions the model was confident about, which shows up as the waterfall
# asking for clarification on signs it had actually read.
LEXICON_HIT_MARGIN = 0.15
LANGUAGE_BACKUP_MARGIN = 0.05


class SignNet(nn.Module):
    """Bidirectional GRU over the normalised landmark sequence.

    Max-pooling over time rather than taking the last state: a sign's identity
    sits in its most distinctive moment, not its final frame, and clips here are
    padded or resampled to a fixed length so the last frame is often a hold.
    """

    def __init__(self, n_classes: int, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.gru = nn.GRU(
            FEATURE_DIM, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Dropout(0.4), nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out.max(dim=1).values)


@dataclass
class TrainReport:
    classes: list[str]
    train_clips: int
    val_clips: int
    epochs: int
    val_accuracy: float
    seconds: float


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class NeuralSignClassifier:
    """Drop-in replacement for `SignClassifier` with the same public surface."""

    def __init__(self, classes: Sequence[str] | None = None) -> None:
        self.classes: list[str] = list(classes or [])
        self.model: SignNet | None = None
        self.device = _device()

    # ---- training -----------------------------------------------------------

    def fit_directory(
        self,
        vocab_dir: Path,
        *,
        exclude: Sequence[Path] = (),
        epochs: int = 120,
        lr: float = 2e-3,
        val_fraction: float = 0.12,
        seed: int = 0,
        progress: bool = False,
    ) -> TrainReport:
        skip = {p.resolve() for p in exclude}
        features: list[np.ndarray] = []
        labels: list[int] = []
        classes = sorted(
            p.name for p in vocab_dir.iterdir() if p.is_dir() and p.name != "raw_video"
        )
        for index, gloss in enumerate(classes):
            for path in sorted((vocab_dir / gloss).glob("*.pose")):
                if path.resolve() in skip:
                    continue
                try:
                    features.append(pose_features(load_pose(path)))
                    labels.append(index)
                except Exception:          # a clip with no signing segment is not fatal
                    continue
        if not features:
            raise ClassifierError(f"no usable clips under {vocab_dir}")

        self.classes = classes
        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.long)

        # Validation comes out of training data. Early stopping on the test set
        # would report a number that cannot be reproduced on anything else.
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        cut = max(1, int(val_fraction * len(X)))
        val, train = order[:cut], order[cut:]

        torch.manual_seed(seed)
        model = SignNet(len(classes)).to(self.device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        Xv = X[val].to(self.device)
        yv = y[val].to(self.device)
        best_accuracy, best_state = -1.0, None
        started = time.time()

        for epoch in range(epochs):
            model.train()
            shuffled = torch.randperm(len(train))
            for i in range(0, len(shuffled), 32):
                batch = train[shuffled[i: i + 32].numpy()]
                xb = X[batch]
                # Scale and shift jitter: the same sign filmed closer or slightly
                # off-centre is the same sign, and the corpus varies in both.
                xb = xb * (1 + torch.randn(len(batch), 1, 1) * 0.05)
                xb = xb + torch.randn(len(batch), 1, FEATURE_DIM) * 0.02
                optimiser.zero_grad()
                criterion(model(xb.to(self.device)), y[batch].to(self.device)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            schedule.step()

            model.eval()
            with torch.no_grad():
                accuracy = (model(Xv).argmax(1) == yv).float().mean().item()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if progress and epoch % 20 == 0:
                print(f"    epoch {epoch:>3}  val {accuracy:.1%}")

        model.load_state_dict(best_state)
        model.eval()
        self.model = model
        return TrainReport(
            classes=classes, train_clips=len(train), val_clips=len(val),
            epochs=epochs, val_accuracy=best_accuracy, seconds=time.time() - started,
        )

    # ---- inference ----------------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        return list(self.classes)

    def classify_features(self, features: np.ndarray) -> Prediction:
        if self.model is None:
            raise ClassifierError("model not trained or loaded")
        x = torch.tensor(features, dtype=torch.float32)[None].to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(x)[0], dim=-1).cpu().numpy()
        order = np.argsort(probabilities)[::-1]
        best, runner_up = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        # Margin between the top two, matching the DTW classifier's scale so the
        # waterfall's thresholds keep the meaning they were calibrated with: a
        # clear winner scores high, a near-tie scores low.
        confidence = float(probabilities[best] - probabilities[runner_up])
        return Prediction(
            gloss_id=self.classes[best],
            confidence=confidence,
            coverage_status=confidence_to_coverage(
                confidence,
                lexicon_hit=LEXICON_HIT_MARGIN,
                language_backup=LANGUAGE_BACKUP_MARGIN,
            ),
            runner_up=self.classes[runner_up],
            best_distance=float(1 - probabilities[best]),
            runner_up_distance=float(1 - probabilities[runner_up]),
        )

    def classify(self, pose) -> Segment:
        prediction = self.classify_features(pose_features(pose))
        return Segment(
            id=str(uuid.uuid4()),
            direction=Direction.SIGN_TO_SPEECH,
            raw_input=prediction.gloss_id,
            confidence=prediction.confidence,
            coverage_status=prediction.coverage_status,
            timestamp=time.time(),
        )

    # ---- persistence --------------------------------------------------------

    def save(self, path: Path) -> Path:
        if self.model is None:
            raise ClassifierError("nothing to save — train first")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classes": self.classes, "state": self.model.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "NeuralSignClassifier":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(blob["classes"])
        model = SignNet(len(blob["classes"]))
        model.load_state_dict(blob["state"])
        instance.model = model.to(instance.device).eval()
        return instance
