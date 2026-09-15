"""The runtime loop that makes the components a system (T1.17).

Every Stage 1 task built one piece and verified it alone. Nothing joined them:
there was no path from a webcam frame to spoken English, or from a transcript to
a moving avatar, outside the test harnesses. §E items 2, 3 and 10 all assume this
file exists, and until now it did not.

**The rule this enforces.** Both directions go through the escalation waterfall
before anything reaches an output device. A recognition is never spoken and a
gloss is never rendered on the strength of its own confidence alone — the
waterfall decides translate, clarify or refuse, and only TRANSLATE produces
output. That ordering is the whole point of the architecture: the failure this
system is built to avoid is confidently saying something the signer did not sign.

Output goes through the collision manager for the same reason. Two directions
sharing one room will otherwise talk over each other, and the avatar has no way
to know someone started speaking mid-sign.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from backend.collision.state_machine import CollisionManager, HoldDecision
from backend.contracts import CoverageStatus, Direction, Segment
from backend.recognition.classifier import SignClassifier, pose_features
from backend.recognition.extract import load_pose
from backend.speech_to_sign.gloss_lookup import GlossLookup
from backend.speech_to_sign.pose_smoothing import build_avatar_sequence, smooth
from backend.speech_to_sign.reasoning import GlossError, to_gloss
from backend.sign_to_speech.reasoning import ReasoningError, reconstruct_sentence
from backend.waterfall.decision_log import DecisionLog
from backend.waterfall.escalation import Action, EscalationWaterfall

ROOT = Path(__file__).resolve().parent.parent
RECOGNIZER = ROOT / "data" / "models" / "recognizer.pt"
LEXICON = ROOT / "data" / "lexicon"
VOCAB = ROOT / "data" / "vocab"
RENDER_FPS = 30.0

# Spoken equivalent of the panel's uncertainty label. Short on purpose: it has to
# survive being heard once, in a room, without turning every hedged sentence into
# a paragraph.
UNCERTAIN_PREFIX = "I'm not certain, but I think they signed:"


@dataclass
class Outcome:
    """What the system did, and why — the same fields the panel renders."""

    direction: Direction
    action: Action
    segment: Segment
    detail: str = ""
    spoken: str | None = None
    gloss: str | None = None
    coverage: tuple = ()
    pose_frames: int = 0
    timeline: list = field(default_factory=list)
    question: str | None = None
    uncertain: bool = False


def _load_recognizer(path: Path, vocab: Path):
    """The learned recogniser if one is installed, DTW otherwise.

    Falling back rather than failing is deliberate: a checkout without the
    trained checkpoint still runs, just less well, and the decision log says
    which one answered.
    """
    try:
        from backend.recognition.neural import NeuralSignClassifier

        if path.is_file():
            return NeuralSignClassifier.load(path), "learned"
    except Exception:
        pass
    classifier = SignClassifier()
    classifier.fit_directory(vocab)
    return classifier, "dtw"


class Interpreter:
    """Both directions, sharing one waterfall, one collision manager, one log."""

    def __init__(
        self,
        *,
        recognizer_path: Path = RECOGNIZER,
        vocab_dir: Path = VOCAB,
        lexicon_dir: Path = LEXICON,
        speak: Callable[[str], object] | None = None,
        render: Callable[[object, list], object] | None = None,
        log: DecisionLog | None = None,
    ) -> None:
        self.log = log or DecisionLog()
        self.waterfall = EscalationWaterfall(on_log=self.log.add)
        self.collision = CollisionManager()
        self.lookup = GlossLookup(lexicon_dir)
        self.recognizer, self.recognizer_kind = _load_recognizer(recognizer_path, vocab_dir)
        # Injected so the loop can be exercised without a virtual microphone or a
        # browser attached; the real sinks are wired in by scripts/run_interpreter.py.
        self._speak = speak
        self._render = render

    # ---- Sign -> Speech ------------------------------------------------------

    def sign_to_speech(self, pose) -> Outcome:
        segment = self.recognizer.classify(pose)

        # An UNMATCHED recognition is the bottom of the same four-tier status the
        # Speech->Sign direction uses, and it has to mean the same thing in both:
        # there is nothing here worth saying. Without this the first low-confidence
        # segment is spoken before hysteresis can fire — live, someone simply
        # moving in frame produced "I think they signed: School." at confidence
        # 0.03. Passing it as unmatched makes the waterfall refuse instead, which
        # is FR-12 applied to the direction it was not originally written for.
        unrecognised = segment.coverage_status is CoverageStatus.UNMATCHED
        decision = self.waterfall.process(
            segment,
            candidates=self._candidates(pose, segment),
            unmatched=[segment.raw_input] if unrecognised else [],
        )

        if decision.action is not Action.TRANSLATE:
            question = None
            if unrecognised:
                # Refusing silently would leave the signer waiting. Say so.
                question = "I didn't catch that sign — could you sign it again?"
            elif decision.action is Action.CLARIFY and len(decision.candidates) >= 2:
                question = (f"Did you sign {decision.candidates[0]} or "
                            f"{decision.candidates[1]}?")
            elif decision.action is Action.CLARIFY:
                question = "Could you sign that again?"
            if question and self._speak:
                self._speak(question)
            return Outcome(Direction.SIGN_TO_SPEECH, decision.action, segment,
                           detail=decision.reason, question=question)

        try:
            reconstruction = reconstruct_sentence(segment.raw_input, segment.confidence)
        except ReasoningError as exc:
            return Outcome(Direction.SIGN_TO_SPEECH, Action.REFUSE, segment,
                           detail=f"reasoning failed: {exc}")

        # §4.1's third rung: the waterfall can translate a low-confidence
        # recognition rather than stall on every bad frame, but only "with a
        # visible uncertain label". Speaking it flat would drop the label and
        # state a guess as fact, which is the exact failure this system exists to
        # avoid — a 0.04-confidence read reaching the room as "Sorry."
        uncertain = bool(decision.uncertain_terms)
        sentence = reconstruction.sentence
        spoken = f"{UNCERTAIN_PREFIX} {sentence}" if uncertain else sentence
        if self._speak:
            self._speak(spoken)
        return Outcome(Direction.SIGN_TO_SPEECH, Action.TRANSLATE, segment,
                       detail=decision.reason, spoken=spoken, uncertain=uncertain)

    def _candidates(self, pose, segment: Segment) -> list[str]:
        """Top two labels, so a clarification can name real alternatives."""
        try:
            prediction = self.recognizer.classify_features(pose_features(pose))
            return [prediction.gloss_id, prediction.runner_up]
        except Exception:
            return [segment.raw_input]

    # ---- Speech -> Sign ------------------------------------------------------

    def speech_to_sign(self, transcript: str) -> Outcome:
        segment = Segment(
            id=str(uuid.uuid4()), direction=Direction.SPEECH_TO_SIGN,
            raw_input=transcript, confidence=1.0,
            coverage_status=None, timestamp=time.time(),
        )
        try:
            result = to_gloss(transcript, vocabulary=self.lookup.glosses)
        except GlossError as exc:
            return Outcome(Direction.SPEECH_TO_SIGN, Action.REFUSE, segment,
                           detail=f"gloss failed: {exc}")

        coverage = self.lookup.coverage_for(list(result.tokens))
        unmatched = [c.gloss for c in coverage if c.status is CoverageStatus.UNMATCHED]
        renderable = [c.gloss for c in coverage if c.status is CoverageStatus.LEXICON_HIT]
        segment.raw_input = result.gloss
        segment.coverage_status = (
            CoverageStatus.UNMATCHED if unmatched else CoverageStatus.LEXICON_HIT
        )

        decision = self.waterfall.process(segment, unmatched=unmatched)
        if decision.action is not Action.TRANSLATE or not renderable:
            # Refusing is the correct outcome for a term with no validated sign;
            # rendering the rest would present a partial sentence as a whole one.
            return Outcome(Direction.SPEECH_TO_SIGN, decision.action, segment,
                           detail=decision.reason or "nothing renderable",
                           gloss=result.gloss, coverage=coverage)

        pose, timeline = build_avatar_sequence(
            renderable, self.lookup.lexicon_dir, target_fps=RENDER_FPS
        )
        pose, _ = smooth(pose)
        frames = int(pose.body.data.shape[0])
        duration = frames / RENDER_FPS

        self.collision.submit(result.gloss, duration_s=duration)
        if self._render:
            self._render(pose, timeline)
        return Outcome(Direction.SPEECH_TO_SIGN, Action.TRANSLATE, segment,
                       detail=decision.reason, gloss=result.gloss, coverage=coverage,
                       pose_frames=frames, timeline=timeline)

    # ---- shared -------------------------------------------------------------

    def observe_audio(self, voice_present: bool) -> HoldDecision:
        """Feed one VAD frame. Own TTS is discarded inside the manager (FR-13)."""
        return self.collision.observe_audio(voice_present)

    def trace(self, limit: int | None = None) -> str:
        return self.log.render(limit=limit)


def interpreter_from_paths(**kwargs) -> Interpreter:
    return Interpreter(**kwargs)
