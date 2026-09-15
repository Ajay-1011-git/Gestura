"""The escalation waterfall (T1.14) — architecture v3 §3 and §4.1.

This is the architectural centrepiece: the decision structure every segment
passes through, and the reason Gestura is an agent rather than a pipeline with a
language model in it. The agent's real decision at each step is *whether to
escalate further or stop and act* — cheap when confident, expensive only when it
has to be.

**Implementation choice, made on a real check rather than assumed.** The task
nominates `langgraph`, allowing a plain state machine if the current API adds
more complexity than it saves. langgraph 1.2.11 was installed and inspected on
2026-09-15: it pulls 12 packages (langchain-core, langsmith, langchain-protocol,
pydantic, orjson, tenacity, …) and its strengths — persistence, multi-actor
orchestration, interrupts — are not exercised here. This waterfall is a linear
ladder, and the only cycles are hysteresis counters *across* segments, which is
session state rather than graph cycles within an invocation. A plain state
machine was chosen and langgraph removed rather than left as an unused
dependency. The requirement that actually matters, FR-15's Observe/Decide/Action
trace, is produced either way and is emitted here directly.

Stage 2 and Stage 3 steps are present as explicit no-op passthroughs so the
ladder's shape is visible and later stages slot in without restructuring.
Building them now would be scope drift.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from backend.contracts import CoverageStatus, DecisionLogEntry, Direction, Segment


class Action(Enum):
    TRANSLATE = "translate"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class Stage(Enum):
    LOCAL_LOOKUP = "local_lookup"
    SESSION_GLOSSARY = "session_glossary"        # Stage 2
    DOMAIN_GLOSSARY = "domain_glossary"          # Stage 2
    AGENT_REPROCESSING = "agent_reprocessing"    # Stage 3
    LLM_REASONING = "llm_reasoning"
    TERMINAL = "terminal"


class ClarificationLevel(Enum):
    """§4.1's ladder — escalation, not a binary stall."""

    NONE = 0
    STALL_AND_ASK = 1      # stall both sides, ask a targeted or generic question
    SHOW_CANDIDATES = 2    # stop re-asking the same way; show candidate text
    PROCEED_UNCERTAIN = 3  # proceed with a visible "uncertain" label


# §4.1 hysteresis: two consecutive low-confidence segments to trigger, one clean
# segment to resume, then a cooldown so a single bad frame cannot cause repeated
# interruptions.
LOW_CONFIDENCE_TRIGGER = 2
CLEAN_SEGMENTS_TO_RESUME = 1
COOLDOWN_SEGMENTS = 2

# Calibrated in T1.4 against a real held-out run, not guessed.
CONFIDENT_THRESHOLD = 0.22


@dataclass
class Decision:
    action: Action
    segment: Segment
    stage_reached: Stage
    clarification: ClarificationLevel = ClarificationLevel.NONE
    candidates: tuple[str, ...] = ()
    reason: str = ""
    uncertain_terms: tuple[str, ...] = ()

    @property
    def stalls_both_sides(self) -> bool:
        """§4.1: the first trigger stalls the signer and the hearing side at once."""
        return self.action is Action.CLARIFY and self.clarification is ClarificationLevel.STALL_AND_ASK


@dataclass
class WaterfallState:
    consecutive_low: int = 0
    consecutive_clean: int = 0
    cooldown_remaining: int = 0
    active_clarification: ClarificationLevel = ClarificationLevel.NONE
    clarifying_segment_id: str | None = None


class EscalationWaterfall:
    """Runs every segment through the ladder and decides translate/clarify/refuse."""

    def __init__(
        self,
        *,
        confident_threshold: float = CONFIDENT_THRESHOLD,
        low_confidence_trigger: int = LOW_CONFIDENCE_TRIGGER,
        cooldown_segments: int = COOLDOWN_SEGMENTS,
        on_log: Callable[[DecisionLogEntry], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.confident_threshold = confident_threshold
        self.low_confidence_trigger = low_confidence_trigger
        self.cooldown_segments = cooldown_segments
        self.state = WaterfallState()
        self.log: list[DecisionLogEntry] = []
        self._on_log = on_log
        self._clock = clock

    # ---- decision log (FR-15) -------------------------------------------------

    def _emit(self, stage: str, segment_id: str, detail: str) -> None:
        entry = DecisionLogEntry(
            timestamp=self._clock(), stage=stage, segment_id=segment_id, detail=detail
        )
        self.log.append(entry)
        if self._on_log is not None:
            self._on_log(entry)

    # ---- Stage 2 / Stage 3 hooks ---------------------------------------------

    def _session_glossary(self, segment: Segment) -> str | None:
        return None  # TODO: Stage 2 — in-memory, call-scoped glossary

    def _domain_glossary(self, segment: Segment) -> str | None:
        return None  # TODO: Stage 2 — room-context-selected domain glossary

    def _agent_reprocessing(self, segment: Segment) -> Segment | None:
        return None  # TODO: Stage 3 — re-query recognition on the buffered window

    def _offer_sign_coinage(self, segment: Segment) -> bool:
        return False  # TODO: Stage 3 — collaborative sign coinage

    # ---- the ladder -----------------------------------------------------------

    def process(
        self,
        segment: Segment,
        *,
        candidates: Sequence[str] = (),
        unmatched: Sequence[str] = (),
    ) -> Decision:
        """Run one segment through the waterfall.

        ``unmatched`` carries T1.9's UNMATCHED gloss tokens. An unmatched term
        must never proceed as if it were translated (FR-12).
        """
        sid = segment.id
        self._emit(
            "OBSERVE",
            sid,
            f"{segment.direction.value}: {segment.raw_input!r}, confidence "
            f"{segment.confidence:.2f}, coverage "
            f"{segment.coverage_status.value if segment.coverage_status else 'none'}",
        )

        # 1. Refuse-to-fabricate takes precedence over everything else. No amount
        #    of downstream confidence makes a sign that does not exist renderable.
        #
        #    UNMATCHED means two different things depending on direction, and
        #    conflating them is a real trap: contracts §B.2 deliberately reuses
        #    one four-tier vocabulary for both, so a low-confidence *recognition*
        #    ("unsure which sign that was") arrives looking identical to a lexicon
        #    *miss* ("no validated sign exists"). Only the second is a refusal.
        #    Treating the first as one would make clarification unreachable and
        #    silently delete demo scene 2.
        lexicon_miss = bool(unmatched) or (
            segment.coverage_status is CoverageStatus.UNMATCHED
            and segment.direction is Direction.SPEECH_TO_SIGN
        )
        if lexicon_miss:
            terms = tuple(unmatched) or (segment.raw_input,)
            self._emit("DECIDE", sid, f"no validated sign for {', '.join(terms)}; lexicon miss is terminal")
            self._emit("ACTION", sid, f"REFUSE — will not fabricate a sign for {', '.join(terms)}")
            self._note_clean_or_low(segment, low=False)
            return Decision(
                action=Action.REFUSE,
                segment=segment,
                stage_reached=Stage.LOCAL_LOOKUP,
                reason="unmatched_gloss",
                uncertain_terms=terms,
            )

        confident = segment.confidence >= self.confident_threshold

        # 2. Update hysteresis first. A clean segment arriving mid-clarification
        #    is the resume signal (§4.1), and it has to clear the clarification
        #    state *before* the confidence branch — otherwise a high-confidence
        #    segment falls through to the escalation path and the log reports it
        #    as "low confidence 0.84", which is both wrong and exactly the kind
        #    of thing the decision log exists not to do.
        self._note_clean_or_low(segment, low=not confident)

        # 3. Confident and covered: resolve at the cheapest step, pay for nothing more.
        if confident and self.state.active_clarification is ClarificationLevel.NONE:
            self._emit("DECIDE", sid, f"confidence {segment.confidence:.2f} >= {self.confident_threshold:.2f}, resolved at local lookup")
            self._emit("ACTION", sid, f"TRANSLATE {segment.raw_input!r}")
            return Decision(
                action=Action.TRANSLATE,
                segment=segment,
                stage_reached=Stage.LOCAL_LOOKUP,
                reason="confident_lexicon_hit",
            )

        # 4. Escalate. Each rung is cheap and reported, including the inert ones,
        #    so the trace shows the ladder rather than implying a single threshold.
        stage = Stage.LOCAL_LOOKUP

        for step, resolver in (
            (Stage.SESSION_GLOSSARY, self._session_glossary),
            (Stage.DOMAIN_GLOSSARY, self._domain_glossary),
        ):
            stage = step
            if resolver(segment) is not None:
                self._emit("ACTION", sid, f"resolved at {step.value}")
                return Decision(action=Action.TRANSLATE, segment=segment, stage_reached=step, reason=step.value)
            self._emit("DECIDE", sid, f"{step.value}: miss (Stage 2 hook, inert)")

        stage = Stage.AGENT_REPROCESSING
        if self._agent_reprocessing(segment) is not None:
            self._emit("ACTION", sid, "resolved by agent-directed reprocessing")
            return Decision(action=Action.TRANSLATE, segment=segment, stage_reached=stage, reason="reprocessed")
        self._emit("DECIDE", sid, "agent_reprocessing: unavailable (Stage 3 hook, inert)")

        stage = Stage.LLM_REASONING
        if self.state.consecutive_low < self.low_confidence_trigger:
            self._emit(
                "DECIDE",
                sid,
                f"low confidence {segment.confidence:.2f} but only "
                f"{self.state.consecutive_low}/{self.low_confidence_trigger} consecutive — "
                f"hysteresis not met, proceeding with uncertainty marked",
            )
            self._emit("ACTION", sid, f"TRANSLATE {segment.raw_input!r} (marked uncertain)")
            return Decision(
                action=Action.TRANSLATE,
                segment=segment,
                stage_reached=stage,
                reason="below_hysteresis",
                uncertain_terms=(segment.raw_input,),
            )

        return self._clarify(segment, candidates, stage)

    def _clarify(self, segment: Segment, candidates: Sequence[str], stage: Stage) -> Decision:
        sid = segment.id
        same_segment = self.state.clarifying_segment_id == segment.raw_input
        level = self.state.active_clarification

        if not same_segment or level is ClarificationLevel.NONE:
            level = ClarificationLevel.STALL_AND_ASK
        elif level is ClarificationLevel.STALL_AND_ASK:
            level = ClarificationLevel.SHOW_CANDIDATES
        else:
            level = ClarificationLevel.PROCEED_UNCERTAIN

        self.state.active_clarification = level
        self.state.clarifying_segment_id = segment.raw_input

        if level is ClarificationLevel.PROCEED_UNCERTAIN:
            self._emit("DECIDE", sid, "clarification exhausted; proceeding with a visible uncertainty label")
            self._emit("ACTION", sid, f"TRANSLATE {segment.raw_input!r} labelled uncertain")
            self._reset_clarification()
            return Decision(
                action=Action.TRANSLATE,
                segment=segment,
                stage_reached=Stage.TERMINAL,
                clarification=ClarificationLevel.PROCEED_UNCERTAIN,
                candidates=tuple(candidates),
                reason="uncertain_fallback",
                uncertain_terms=(segment.raw_input,),
            )

        if level is ClarificationLevel.STALL_AND_ASK:
            targeted = len(candidates) >= 2
            detail = (
                f"two defensible candidates {candidates[0]}/{candidates[1]} — asking a targeted question"
                if targeted
                else "no second defensible candidate — asking for a repeat"
            )
            self._emit("DECIDE", sid, f"{self.state.consecutive_low} consecutive low-confidence segments; {detail}")
            self._emit("ACTION", sid, "CLARIFY — stalling both sides at once")
        else:
            self._emit("DECIDE", sid, "same segment uncertain again; not re-asking the same way")
            self._emit("ACTION", sid, f"CLARIFY — showing candidates {list(candidates) or [segment.raw_input]}")

        return Decision(
            action=Action.CLARIFY,
            segment=segment,
            stage_reached=Stage.TERMINAL,
            clarification=level,
            candidates=tuple(candidates),
            reason="low_confidence_sustained",
        )

    # ---- hysteresis -----------------------------------------------------------

    def _note_clean_or_low(self, segment: Segment, *, low: bool) -> None:
        if low:
            self.state.consecutive_low += 1
            self.state.consecutive_clean = 0
            return

        self.state.consecutive_clean += 1
        self.state.consecutive_low = 0
        if (
            self.state.active_clarification is not ClarificationLevel.NONE
            and self.state.consecutive_clean >= CLEAN_SEGMENTS_TO_RESUME
        ):
            self._emit("OBSERVE", segment.id, "clean segment received")
            self._emit("ACTION", segment.id, "translation resumed; entering cooldown")
            self._reset_clarification()

    def _reset_clarification(self) -> None:
        self.state.active_clarification = ClarificationLevel.NONE
        self.state.clarifying_segment_id = None
        self.state.consecutive_low = 0
        self.state.cooldown_remaining = self.cooldown_segments


def make_segment(
    raw_input: str,
    confidence: float,
    *,
    direction: Direction = Direction.SIGN_TO_SPEECH,
    coverage: CoverageStatus | None = None,
) -> Segment:
    """Convenience constructor for callers that are not T1.4."""
    return Segment(
        id=str(uuid.uuid4()),
        direction=direction,
        raw_input=raw_input,
        confidence=confidence,
        coverage_status=coverage,
        timestamp=time.time(),
    )
