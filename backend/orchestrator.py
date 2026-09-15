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
from backend.contracts import (
    CoverageStatus, DecisionLogEntry, Direction, DomainContext, Segment,
)
from backend.recognition.classifier import SignClassifier, pose_features
from backend.recognition.extract import load_pose
from backend.resilience import safe_mode as safe_mode_module
from backend.resilience.safe_mode import SafeMode
from backend.speech_to_sign import fingerspell
from backend.speech_to_sign.gloss_lookup import GlossLookup
from backend.speech_to_sign.pose_smoothing import build_avatar_sequence, smooth
from backend.speech_to_sign.reasoning import GlossError, to_gloss
from backend.sign_to_speech.reasoning import ReasoningError, reconstruct_sentence
from backend.waterfall.decision_log import DecisionLog
from backend.waterfall.domain_glossary import DomainGlossary, DomainGlossaryError
from backend.waterfall.escalation import Action, EscalationWaterfall
from backend.waterfall.session_glossary import SessionGlossary

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
    # Stage 2. `unrendered` carries FR-21/NFR-9's label: set whenever safe mode
    # suppressed the avatar work this outcome would otherwise have produced, so
    # a degraded result can never be presented identically to a confident one.
    unrendered: str | None = None
    fingerspelled: tuple[str, ...] = ()


def _resolve_domain(domain: DomainContext | str | None) -> DomainContext:
    """The room's domain, supplied by a human at session start — never inferred.

    Falls back to the DOMAIN_GLOSSARY_DEFAULT env var, then to GENERAL. An
    unrecognised value raises rather than silently defaulting: quietly running a
    medical deployment against the general glossary because someone typed
    "medicine" is precisely the wrong-domain resolution FR-20 forbids.
    """
    import os

    if domain is None:
        domain = os.environ.get("DOMAIN_GLOSSARY_DEFAULT", "").strip() or "general"
    if isinstance(domain, DomainContext):
        return domain
    try:
        return DomainContext(str(domain).strip().lower())
    except ValueError:
        valid = ", ".join(d.value for d in DomainContext)
        raise ValueError(f"unknown domain {domain!r} — expected one of: {valid}") from None


def _load_domain_glossary(domain: DomainContext, log: DecisionLog) -> DomainGlossary | None:
    """Load the selected domain's manifest, or run without one if it is absent.

    A missing manifest is not fatal — the rung simply stays a passthrough and
    the ladder continues to LLM reasoning, which is what Stage 1 did. It is
    logged, though: silently running without the domain vocabulary someone
    selected would be a worse failure than not offering domains at all.
    """
    try:
        return DomainGlossary(domain, on_log=log.add)
    except DomainGlossaryError as exc:
        log.add(DecisionLogEntry(
            timestamp=time.time(), stage="OBSERVE", segment_id="domain_glossary",
            detail=f"no usable {domain.value} glossary ({exc}) — running without it",
        ))
        return None


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
        domain: DomainContext | str | None = None,
        safe_mode: SafeMode | None = None,
    ) -> None:
        self.log = log or DecisionLog()
        self.collision = CollisionManager()
        self.lookup = GlossLookup(lexicon_dir)
        self.recognizer, self.recognizer_kind = _load_recognizer(recognizer_path, vocab_dir)

        # ---- Stage 2 ---------------------------------------------------------
        # One session glossary per call (T2.1, FR-18): constructed here, dropped
        # with the Interpreter, never written anywhere.
        self.session_glossary = SessionGlossary(on_log=self.log.add)
        self.domain = _resolve_domain(domain)
        self.domain_glossary = _load_domain_glossary(self.domain, self.log)
        # T2.3. Registered process-wide so the Stage 1 call sites several layers
        # down can read `safe_mode.is_active()` without a signature change.
        self.safe_mode = safe_mode or SafeMode(
            on_log=self.log.add, on_hold_message=self._hold_message
        )
        safe_mode_module.register(self.safe_mode)

        self.waterfall = EscalationWaterfall(
            on_log=self.log.add,
            session_glossary=self.session_glossary,
            domain_glossary=self.domain_glossary,
        )
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
        # T2.1/T2.2: the glossaries are checked *before* the reasoning call, not
        # after. A cache consulted after the tokens have already been spent
        # saves nothing — and protecting the real 8,000 tokens/min ceiling is
        # the entire justification for having them (NFR-11, TNFR-8).
        gloss_text = self._glossary_lookup(transcript)
        if gloss_text is not None:
            result = _GlossaryResult(gloss_text)
        else:
            try:
                result = to_gloss(transcript, vocabulary=self.lookup.glosses)
            except GlossError as exc:
                return Outcome(Direction.SPEECH_TO_SIGN, Action.REFUSE, segment,
                               detail=f"gloss failed: {exc}")
            # Record what reasoning worked out, so the next occurrence in this
            # call is free. Only a real resolution is recorded, never a guess.
            self.session_glossary.record(transcript, result.gloss)

        coverage = self.lookup.coverage_for(list(result.tokens))
        unmatched = [c.gloss for c in coverage if c.status is CoverageStatus.UNMATCHED]
        renderable = [c.gloss for c in coverage if c.status is CoverageStatus.LEXICON_HIT]
        spellable = [c.gloss for c in coverage if c.status is CoverageStatus.FINGERSPELLING]
        segment.raw_input = result.gloss
        # The segment's own status is the weakest tier any of its tokens reached.
        # Stage 1 only had two outcomes here; with T2.6 a sentence can be part
        # signed and part spelled, and calling that a clean LEXICON_HIT would
        # tell the panel the whole thing was signed when some of it was spelled
        # out — a small lie in exactly the place the four-tier vocabulary exists
        # to prevent one (FR-28, NFR-12).
        if unmatched:
            segment.coverage_status = CoverageStatus.UNMATCHED
        elif spellable:
            segment.coverage_status = CoverageStatus.FINGERSPELLING
        else:
            segment.coverage_status = CoverageStatus.LEXICON_HIT

        decision = self.waterfall.process(segment, unmatched=unmatched)
        if decision.action is not Action.TRANSLATE or not (renderable or spellable):
            # Refusing is the correct outcome for a term with no validated sign;
            # rendering the rest would present a partial sentence as a whole one.
            return Outcome(Direction.SPEECH_TO_SIGN, decision.action, segment,
                           detail=decision.reason or "nothing renderable",
                           gloss=result.gloss, coverage=coverage)

        # T2.3/FR-21: while degraded, no *new* avatar animation is started. The
        # gloss still reaches the participant as text, carrying the unrendered
        # label so it can never be mistaken for a rendered result (NFR-9).
        if self.safe_mode.is_active:
            self.log.add(DecisionLogEntry(
                timestamp=time.time(), stage="ACTION", segment_id=segment.id,
                detail=f"safe mode: avatar animation suppressed for {result.gloss!r}; "
                       f"shown as text with the unrendered label",
            ))
            return Outcome(Direction.SPEECH_TO_SIGN, Action.TRANSLATE, segment,
                           detail=decision.reason, gloss=result.gloss,
                           coverage=coverage,
                           unrendered=self.safe_mode.unrendered_label)

        pose, timeline = build_avatar_sequence(
            renderable, self.lookup.lexicon_dir, target_fps=RENDER_FPS
        ) if renderable else (None, [])

        # T2.6: terms with no sign are spelled rather than dropped.
        #
        # Known limitation, stated rather than hidden: spelled terms are appended
        # after *all* the signed content, not interleaved at their gloss
        # position. "HELLO AJAY" comes out right because the spelled term is
        # already last; "AJAY HELLO" would come out reordered. Interleaving
        # needs build_avatar_sequence to accept a mixed signed/spelled timeline,
        # and that function is Stage 1's — out of scope here (§B.3). The gloss
        # text and the coverage report both carry the true order, so the panel
        # shows what was meant even when the avatar's ordering is coarse.
        spelled = self._fingerspell_tokens(spellable, segment)
        pose = _join_poses(pose, [s.pose for s in spelled if s.pose is not None])
        if pose is None:
            return Outcome(Direction.SPEECH_TO_SIGN, Action.REFUSE, segment,
                           detail="nothing renderable or spellable",
                           gloss=result.gloss, coverage=coverage)

        pose, _ = smooth(pose)
        frames = int(pose.body.data.shape[0])
        duration = frames / RENDER_FPS

        self.collision.submit(result.gloss, duration_s=duration)
        if self._render:
            self._render(pose, timeline)
        return Outcome(Direction.SPEECH_TO_SIGN, Action.TRANSLATE, segment,
                       detail=decision.reason, gloss=result.gloss, coverage=coverage,
                       pose_frames=frames, timeline=timeline,
                       fingerspelled=tuple(s.term for s in spelled if s.pose is not None))

    # ---- Stage 2 helpers ----------------------------------------------------

    def _glossary_lookup(self, transcript: str) -> str | None:
        """Session glossary, then domain glossary. None means neither had it.

        Order matters: the session glossary holds what *this conversation* has
        already settled, which is more specific than the room's curated default
        and must win over it.
        """
        hit = self.session_glossary.resolve(transcript)
        if hit is not None:
            return hit.resolution
        if self.domain_glossary is not None:
            hit = self.domain_glossary.resolve(transcript)
            if hit is not None:
                # Promote a domain hit into the session glossary so the rest of
                # the call skips even the file lookup.
                self.session_glossary.record(transcript, hit.resolution)
                return hit.resolution
        return None

    def _fingerspell_tokens(self, tokens: list[str], segment: Segment) -> list:
        """Spell each out-of-vocabulary token, logging what was spelled (FR-28)."""
        spelled = []
        for token in tokens:
            try:
                result = fingerspell.spell(token)
            except fingerspell.FingerspellError as exc:
                self.log.add(DecisionLogEntry(
                    timestamp=time.time(), stage="DECIDE", segment_id=segment.id,
                    detail=f"cannot fingerspell {token!r}: {exc}",
                ))
                continue
            self.log.add(DecisionLogEntry(
                timestamp=time.time(), stage="ACTION", segment_id=segment.id,
                detail=f"FINGERSPELLING — {fingerspell.describe(result)}",
            ))
            spelled.append(result)
        return spelled

    def _hold_message(self, message: str) -> None:
        """Speak safe mode's hold message (FR-21), through the normal TTS sink."""
        if self._speak:
            self._speak(message)

    def close(self) -> None:
        """End the call. FR-18's discard, made an explicit act rather than a hope."""
        self.session_glossary.discard()
        safe_mode_module.register(None)

    def __enter__(self) -> "Interpreter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- shared -------------------------------------------------------------

    def observe_audio(self, voice_present: bool) -> HoldDecision:
        """Feed one VAD frame. Own TTS is discarded inside the manager (FR-13)."""
        return self.collision.observe_audio(voice_present)

    def trace(self, limit: int | None = None) -> str:
        return self.log.render(limit=limit)


@dataclass(frozen=True)
class _GlossaryResult:
    """A glossary hit, shaped like `to_gloss`'s result so the caller is uniform.

    Deliberately not a subclass of `GlossResult`: this did not come from a model
    and carries no latency or model attribution to report. Sharing the two
    fields the caller reads is the whole contract.
    """

    gloss: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(t for t in self.gloss.split() if t)


def _join_poses(first, rest: list):
    """Concatenate a signed sequence with any fingerspelled ones after it."""
    parts = [p for p in ([first] if first is not None else []) + list(rest) if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody

    data = np.ma.array(np.concatenate([np.asarray(p.body.data) for p in parts], axis=0))
    confidence = np.concatenate([np.asarray(p.body.confidence) for p in parts], axis=0)
    return Pose(
        header=parts[0].header,
        body=NumPyPoseBody(fps=float(parts[0].body.fps), data=data, confidence=confidence),
    )


def interpreter_from_paths(**kwargs) -> Interpreter:
    return Interpreter(**kwargs)
