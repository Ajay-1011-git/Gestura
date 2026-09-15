"""Decision log — the Observe/Decide/Action transparency record (T1.15).

Architecture v3 §9. A live, human-readable record of what Gestura observed,
decided and did, shown during the interaction rather than buried in a file.

Two rules from §9 that are enforced here rather than left to callers:

- **Structured decision factors only, never raw model chain-of-thought.** The
  log shows *why* in terms of confidence, coverage and candidate disagreement.
  Pasting a model's reasoning would be both unreadable and a different kind of
  claim about what the system "thought".
- **The manual override control is always present**, in every state. §9 notes
  that visibly bounded autonomy reads as a strength; it is also FR-16, so the
  override is part of the log's own state rather than a separate widget that
  could be absent when it matters.

The waterfall (T1.14) already emits `DecisionLogEntry` rows as it runs, and the
collision manager (T1.13) emits equivalent transitions. This module consumes
those rather than instrumenting the pipeline a second time — TRD §9's explicit
instruction not to build parallel logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Iterator

from backend.collision.state_machine import Transition
from backend.contracts import CoverageStatus, DecisionLogEntry

VALID_STAGES = ("OBSERVE", "DECIDE", "ACTION")

# §B.2's four-tier status vocabulary, surfaced with the colours architecture v3
# assigns so TNFR-5 holds: anything other than a direct hit must be visibly
# distinguishable in the log, not just recorded.
COVERAGE_MARKS = {
    CoverageStatus.LEXICON_HIT: ("green", "●"),
    CoverageStatus.LANGUAGE_BACKUP: ("yellow", "◐"),
    CoverageStatus.FINGERSPELLING: ("orange", "◑"),
    CoverageStatus.UNMATCHED: ("red", "○"),
}


def monotonic_to_wall(monotonic_timestamp: float) -> float:
    """Convert a `time.monotonic()` reading to a wall-clock epoch timestamp."""
    return monotonic_timestamp + (time.time() - time.monotonic())


class OverrideState(str):
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class DecisionLog:
    """Collects Observe/Decide/Action rows and renders them for the panel."""

    max_entries: int = 500
    entries: list[DecisionLogEntry] = field(default_factory=list)
    _override: str = OverrideState.RUNNING
    _subscribers: list[Callable[[DecisionLogEntry], None]] = field(default_factory=list)

    # ---- manual override (FR-16) ---------------------------------------------

    @property
    def override_state(self) -> str:
        return self._override

    @property
    def paused(self) -> bool:
        return self._override == OverrideState.PAUSED

    def pause(self, *, by: str = "supervisor") -> None:
        self._override = OverrideState.PAUSED
        self._record(DecisionLogEntry(time.time(), "ACTION", "override", f"paused by {by}"))

    def resume(self, *, by: str = "supervisor") -> None:
        self._override = OverrideState.RUNNING
        self._record(DecisionLogEntry(time.time(), "ACTION", "override", f"resumed by {by}"))

    # ---- ingestion ------------------------------------------------------------

    def subscribe(self, callback: Callable[[DecisionLogEntry], None]) -> None:
        self._subscribers.append(callback)

    def _record(self, entry: DecisionLogEntry) -> None:
        if entry.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {entry.stage!r}")
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            del self.entries[: len(self.entries) - self.max_entries]
        for callback in self._subscribers:
            callback(entry)

    def add(self, entry: DecisionLogEntry) -> None:
        """Ingest a row emitted by the waterfall (T1.14)."""
        self._record(entry)

    def add_collision(self, transition: Transition) -> None:
        """Ingest a collision-manager transition (T1.13) as an Observe/Action row.

        A hold or release is something Gestura *did* to its own output, so it
        belongs in the same trace as the waterfall's decisions rather than a
        separate stream the supervisor has to correlate by eye.

        `Transition.at` comes from ``time.monotonic()`` — correct for measuring
        durations, but not an epoch, so rendering it directly put collision rows
        at 05:30:00 next to correctly-timed waterfall rows. It is converted to
        wall clock here, at the boundary, rather than changing the collision
        manager's clock: monotonic time is the right choice there precisely
        because it cannot jump backwards mid-hold.
        """
        stage = "ACTION" if transition.decision.name in ("HOLD", "RELEASE") else "OBSERVE"
        self._record(
            DecisionLogEntry(
                timestamp=monotonic_to_wall(transition.at),
                stage=stage,
                segment_id="collision",
                detail=f"[{transition.avatar.value}/{transition.audio.value}] "
                f"{transition.decision.name}: {transition.detail}",
            )
        )

    # ---- rendering ------------------------------------------------------------

    def render(self, *, limit: int | None = None, clock_format: str = "%H:%M:%S") -> str:
        """Render as the §9 panel: ``HH:MM:SS  STAGE  detail``."""
        rows = self.entries[-limit:] if limit else self.entries
        lines = [
            f"{datetime.fromtimestamp(e.timestamp).strftime(clock_format)}  "
            f"{e.stage:<7}  {e.detail}"
            for e in rows
        ]
        lines.append(
            f"[ override: {self._override.upper()} — "
            f"{'RESUME' if self.paused else 'PAUSE'} always available ]"
        )
        return "\n".join(lines)

    def to_json(self, *, limit: int | None = None) -> str:
        """Serialize for the frontend panel (T1.15's UI half)."""
        rows = self.entries[-limit:] if limit else self.entries
        return json.dumps(
            {
                "override": self._override,
                "entries": [
                    {
                        "timestamp": e.timestamp,
                        "stage": e.stage,
                        "segment_id": e.segment_id,
                        "detail": e.detail,
                    }
                    for e in rows
                ],
            },
            indent=2,
        )

    def for_segment(self, segment_id: str) -> list[DecisionLogEntry]:
        return [e for e in self.entries if e.segment_id == segment_id]


def coverage_mark(status: CoverageStatus | None) -> tuple[str, str]:
    """Colour and glyph for a coverage status (TNFR-5).

    Every status other than a direct lexicon hit must be visibly distinct in the
    log — a fingerspelling fallback or an unmatched term must never look like a
    confident match.
    """
    if status is None:
        return ("grey", "·")
    return COVERAGE_MARKS[status]
