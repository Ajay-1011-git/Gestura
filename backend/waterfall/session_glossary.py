"""Session-scope glossary — a term resolved once stays resolved (T2.1).

The user story this exists for is small and concrete: a Deaf signer should not
have to clarify the same word twice in one conversation. The first time a term
reaches LLM reasoning, what came back is recorded here; every later occurrence
in the same call is answered from memory.

**Why that is a resilience feature and not a cache.** Stage 1's live run found
the real Groq ceiling is 8,000 *tokens* per minute, not the request count — so
the thing that actually ends a demo is a long conversation repeatedly spending
reasoning tokens on words it has already worked out. Every hit here is one Groq
call that never happens (NFR-11, TNFR-8). That is also why the lookup happens
*before* the LLM step in the waterfall rather than after it: a cache consulted
after the call has already been made saves nothing.

**Scope is a hard requirement, not a default.** FR-18 says this holds no state
beyond the current call. There is deliberately no path to disk, no shared store
and no cross-session reuse — a term resolved in a medical appointment must not
leak into the next person's conversation, and a glossary that outlived its call
would be exactly the kind of quiet data retention this project has committed
not to do. :meth:`discard` is called by `Interpreter` at call end, and the
instance is dropped with the session either way.
"""

from __future__ import annotations

from typing import Callable

from backend.contracts import CoverageStatus, DecisionLogEntry, GlossaryHit

SOURCE = "session"


def _key(term: str) -> str:
    """Terms match case- and whitespace-insensitively.

    Gloss arrives uppercase (``HOSPITAL``) and transcripts arrive as typed, so
    an exact-match dict would miss the very repeat this module exists to catch.
    """
    return " ".join(term.strip().lower().split())


class SessionGlossary:
    """In-memory, call-scoped record of terms already resolved this session."""

    def __init__(self, *, on_log: Callable[[DecisionLogEntry], None] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        import time

        self._entries: dict[str, GlossaryHit] = {}
        self._on_log = on_log
        self._clock = clock or time.time
        self.hits = 0
        self.misses = 0

    # ---- decision log (FR-24, reusing FR-15's one path) ----------------------

    def _emit(self, stage: str, term: str, detail: str) -> None:
        if self._on_log is None:
            return
        self._on_log(
            DecisionLogEntry(
                timestamp=self._clock(), stage=stage,
                segment_id=f"session_glossary:{_key(term)}", detail=detail,
            )
        )

    # ---- lookup and record ---------------------------------------------------

    def resolve(self, term: str) -> GlossaryHit | None:
        """Return the recorded resolution for ``term``, or None on a miss.

        A miss returns None and logs nothing beyond the miss itself — it must
        fall through to the next waterfall rung completely unchanged, never
        substituting a near-match (the same rule FR-20 states for the domain
        glossary, applied here for the same reason).
        """
        key = _key(term)
        if not key:
            return None

        hit = self._entries.get(key)
        if hit is None:
            self.misses += 1
            return None

        self.hits += 1
        self._emit(
            "DECIDE", term,
            f"session glossary hit: {term!r} already resolved this call as "
            f"{hit.resolution!r} — no LLM call needed",
        )
        return hit

    def record(self, term: str, resolution: str,
               *, coverage_status: CoverageStatus = CoverageStatus.LEXICON_HIT) -> GlossaryHit:
        """Record how ``term`` was resolved, so the next occurrence is free.

        Called once the waterfall has actually resolved a term — recording a
        guess would turn one uncertain reading into a confident one repeated
        for the rest of the call, which is the failure mode the whole project
        is built to avoid.
        """
        key = _key(term)
        if not key:
            raise ValueError("cannot record an empty term")
        if not resolution.strip():
            raise ValueError(f"cannot record an empty resolution for {term!r}")

        hit = GlossaryHit(
            term=term.strip(), resolution=resolution.strip(),
            source=SOURCE, coverage_status=coverage_status,
        )
        self._entries[key] = hit
        self._emit(
            "ACTION", term,
            f"session glossary: recorded {term!r} -> {hit.resolution!r} "
            f"({coverage_status.value}) for the rest of this call",
        )
        return hit

    # ---- lifetime ------------------------------------------------------------

    def discard(self) -> int:
        """Drop everything. FR-18's "discarded, not persisted", made explicit."""
        count = len(self._entries)
        self._entries.clear()
        if count:
            self._emit("ACTION", "session",
                       f"session glossary discarded at call end: {count} term(s) forgotten")
        return count

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and _key(term) in self._entries

    @property
    def terms(self) -> list[str]:
        return sorted(hit.term for hit in self._entries.values())
