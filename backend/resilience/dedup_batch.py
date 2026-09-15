"""Duplicate-ack suppression and batching for the STT chunk stream (T2.5).

Conversation is full of "okay. okay. right, okay" — and every one of those
becomes a gloss call, a lexicon lookup and an avatar animation that says
nothing. Collapsing them is worth real Groq tokens against the 8,000/min ceiling
Stage 1 found, and it spares the signer watching the avatar nod four times.

**This is not summarization, and the distinction is the whole design.** FR-26
is a hard requirement: names, numbers and questions must survive verbatim.
Deciding that something is "not substantive enough to keep" is exactly the
judgement this module must never make. So it does the opposite of what a
summarizer does — it works from a closed, explicit list of pure
acknowledgment tokens, and *anything* not fully composed of those passes
through untouched.

**The failure direction is chosen, not accidental.** When the classifier is not
certain a chunk is pure filler, it passes it through unmerged. Letting one
redundant "okay" through costs a wasted animation; dropping "no" from "no, the
seven o'clock one" costs the conversation its meaning. That asymmetry is the
same one refuse-to-fabricate applies elsewhere: err toward preserving content
over tidying it.

**Why here and not later.** Filler only exists as filler at the level of
adjacent STT chunks. One rung downstream the segments have been merged into
meaning and the adjacency that identified them is gone — re-deriving it would
mean reconstructing information that is free at this point in the pipeline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

# Pure acknowledgments: tokens that carry no propositional content on their own.
# Closed by design — this list is the entire basis on which anything is ever
# collapsed, so it is short, reviewable, and grows only by deliberate edit.
#
# Note what is deliberately ABSENT: "yes" and "no" are never here. They are
# answers, they are in the render lexicon, and collapsing a repeated "no" would
# destroy the clearest signal a conversation has.
FILLER_TOKENS = frozenset({
    "okay", "ok", "mhm", "mhmm", "uh", "um", "erm", "ah", "oh", "hmm", "hm",
    "right", "sure", "yeah", "yep", "uh-huh", "mm", "mmm", "alright",
    "gotcha", "i see", "got it", "makes sense", "fair enough",
})

# A chunk longer than this is never treated as filler regardless of content —
# length itself is evidence that something is being said.
MAX_FILLER_WORDS = 3

# Adjacent chunks closer together than this are candidates for batching.
DEFAULT_BATCH_WINDOW_S = 1.5

# Short non-filler chunks this close together are joined into one utterance
# rather than sent as two. Never applied across a sentence-ending boundary.
MAX_BATCH_WORDS = 8

_WORD = re.compile(r"[a-z0-9'-]+")
_SENTENCE_END = re.compile(r"[.!?]\s*$")
_HAS_DIGIT = re.compile(r"\d")
# A capitalised word that is not sentence-initial is treated as a proper noun.
_INTERIOR_CAPITAL = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+")


@dataclass(frozen=True)
class Chunk:
    """One STT result, as it comes off the chunker.

    ``meta`` carries whatever the caller needs to reconstruct its own record for
    this chunk — the live loop puts the source `Transcript` here so a chunk
    released later still reports the real duration, latency and model rather
    than whichever transcript happened to be in scope at release time.
    """

    text: str
    timestamp: float = 0.0
    meta: object | None = None

    @property
    def words(self) -> list[str]:
        return _WORD.findall(self.text.lower())


@dataclass
class BatchStats:
    """What the filter actually did — surfaced so the log can say so."""

    seen: int = 0
    suppressed: int = 0
    merged: int = 0
    emitted: int = 0
    suppressed_texts: list[str] = field(default_factory=list)


def is_substantive(text: str) -> bool:
    """Does this chunk carry content that must survive verbatim (FR-26)?

    Deliberately over-inclusive. Every one of these checks exists to *prevent*
    a merge, and a false "yes" costs nothing but a passed-through chunk.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True                       # a direct question
    if _HAS_DIGIT.search(stripped):
        return True                       # any number
    if _INTERIOR_CAPITAL.search(stripped):
        return True                       # a probable proper noun
    words = _WORD.findall(stripped.lower())
    if len(words) > MAX_FILLER_WORDS:
        return True
    return not is_pure_filler(stripped)


def is_pure_filler(text: str) -> bool:
    """True only when *every* token is a known acknowledgment.

    The conservative half of the design: one unrecognised word anywhere in the
    chunk makes the whole chunk content. There is no scoring and no threshold,
    because a confidence score here would be a licence to guess.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped or _HAS_DIGIT.search(stripped):
        return False

    lowered = stripped.lower().strip(" .,!;:")
    if lowered in FILLER_TOKENS:
        return True

    words = _WORD.findall(lowered)
    if not words or len(words) > MAX_FILLER_WORDS:
        return False

    # Multi-word forms like "i see" and "got it" are single entries in the set,
    # so check the joined form before falling back to per-token membership.
    if " ".join(words) in FILLER_TOKENS:
        return True
    return all(w in FILLER_TOKENS for w in words)


def _mergeable(previous: Chunk, current: Chunk, window_s: float) -> bool:
    """Should two adjacent non-filler chunks be sent as one utterance?

    Only when both are short, neither ends a sentence, and they arrived close
    together — i.e. when the chunker split mid-thought on a breath rather than
    at a real boundary. Anything carrying a name, number or question is excluded
    before this is reached.
    """
    if current.timestamp and previous.timestamp:
        if current.timestamp - previous.timestamp > window_s:
            return False
    if _SENTENCE_END.search(previous.text):
        return False
    if len(previous.words) + len(current.words) > MAX_BATCH_WORDS:
        return False
    if "?" in previous.text or "?" in current.text:
        return False
    if _HAS_DIGIT.search(previous.text) or _HAS_DIGIT.search(current.text):
        return False
    if _INTERIOR_CAPITAL.search(previous.text) or _INTERIOR_CAPITAL.search(current.text):
        return False
    return True


class DedupBatcher:
    """Stateful stream transform: collapses filler runs, joins split thoughts.

    Ordering is preserved and nothing is reordered — a chunk is either emitted
    in place, merged into the immediately preceding one, or (filler only)
    dropped as a repeat.
    """

    def __init__(
        self,
        *,
        batch_window_s: float = DEFAULT_BATCH_WINDOW_S,
        on_note: Callable[[str], None] | None = None,
    ) -> None:
        self.batch_window_s = batch_window_s
        self._on_note = on_note
        self._last_filler: str | None = None
        self._pending: Chunk | None = None
        self.stats = BatchStats()

    def _note(self, message: str) -> None:
        if self._on_note is not None:
            self._on_note(message)

    def push(self, chunk: Chunk) -> list[Chunk]:
        """Feed one chunk; get back whatever is ready to go downstream."""
        self.stats.seen += 1
        out: list[Chunk] = []

        if is_pure_filler(chunk.text):
            key = " ".join(chunk.words)
            if self._last_filler == key:
                # A repeat of the acknowledgment we just let through. This is
                # the only case where anything is ever dropped.
                self.stats.suppressed += 1
                self.stats.suppressed_texts.append(chunk.text)
                self._note(f"suppressed repeated acknowledgment {chunk.text!r}")
                return out
            self._last_filler = key
            out.extend(self._flush_pending())
            out.append(chunk)
            self.stats.emitted += 1
            return out

        self._last_filler = None

        if self._pending is not None and _mergeable(self._pending, chunk, self.batch_window_s):
            merged = Chunk(
                text=f"{self._pending.text.rstrip()} {chunk.text.lstrip()}",
                timestamp=self._pending.timestamp,
            )
            self.stats.merged += 1
            self._note(f"batched {self._pending.text!r} + {chunk.text!r}")
            self._pending = merged
            return out

        out.extend(self._flush_pending())
        # Hold a short, unterminated, content-free-of-landmines chunk back one
        # step in case the next one continues it; send anything else straight on.
        if (
            not _SENTENCE_END.search(chunk.text)
            and len(chunk.words) <= MAX_BATCH_WORDS
            and "?" not in chunk.text
            and not _HAS_DIGIT.search(chunk.text)
            and not _INTERIOR_CAPITAL.search(chunk.text)
        ):
            self._pending = chunk
        else:
            out.append(chunk)
            self.stats.emitted += 1
        return out

    def _flush_pending(self) -> list[Chunk]:
        if self._pending is None:
            return []
        pending, self._pending = self._pending, None
        self.stats.emitted += 1
        return [pending]

    def due(self, now: float) -> list[Chunk]:
        """Release a held chunk once its batch window has passed.

        A chunk is held back only to see whether the *next* one continues it.
        Without this, "I need" waits for whatever is said next — which in a real
        conversation might be a minute away, or never, if the call ends there.
        Both outcomes are wrong: the first delays output indefinitely, and the
        second drops substantive content, which FR-26 forbids outright.

        The live loop calls this on every audio frame, so the wait is bounded by
        the batch window rather than by the speaker.
        """
        if self._pending is None:
            return []
        if self._pending.timestamp and now - self._pending.timestamp <= self.batch_window_s:
            return []
        return self._flush_pending()

    def flush(self) -> list[Chunk]:
        """Emit anything still held back. Always call at end of stream."""
        return self._flush_pending()


def filter_stream(
    chunks: Iterable[Chunk | str],
    *,
    batch_window_s: float = DEFAULT_BATCH_WINDOW_S,
    on_note: Callable[[str], None] | None = None,
) -> Iterator[Chunk]:
    """Stream transform over chunks, preserving order (TRD §5's `dedup_batch.filter`)."""
    batcher = DedupBatcher(batch_window_s=batch_window_s, on_note=on_note)
    for raw in chunks:
        chunk = raw if isinstance(raw, Chunk) else Chunk(text=raw, timestamp=time.time())
        yield from batcher.push(chunk)
    yield from batcher.flush()


def filter_texts(texts: Sequence[str], **kwargs) -> list[str]:
    """Convenience wrapper for tests and the scripted verification."""
    return [c.text for c in filter_stream(
        [Chunk(text=t, timestamp=float(i) * 0.5) for i, t in enumerate(texts)], **kwargs
    )]
