"""Domain glossary — curated, human-selected vocabulary per room (T2.2).

A hospital front desk and a workshop bench do not share a vocabulary, and a
general-purpose reasoning pass is measurably worse at both than a narrow one is
at either. This is the mechanism behind G-7's "domain-appropriate vocabulary"
and, with it, G-4's "narrow, supervised pilot for one context at a time": the
operator says which room Setu is standing in, and Setu prefers that room's
terms.

**Selected by a human, never inferred.** `DomainContext` comes from a flag at
session start. Inferring it from what people are saying would be automatic
topic-shift detection, which this project has permanently excluded — and a
wrongly-inferred domain is worse than none, because it would quietly prefer the
wrong sense of a word without anyone having chosen that.

**A miss must stay a miss (FR-20).** The one genuinely dangerous failure here
is substituting a near-match from the wrong domain — rendering the medical
sense of a word in a technical conversation, or vice versa. Lookup is therefore
exact-match only on curated surface forms; anything else falls through to the
next waterfall rung completely unchanged, with no `GlossaryHit` fabricated for
it. Fuzzy matching is not a future improvement here, it is the bug.

**Content is a curation task, not a build step.** `data/glossary/*.md` are
small, human-reviewed manifests in the same shape as T1.2's vocabulary
manifest. They are read, never generated — an auto-built glossary would be an
LLM's guess about a domain wearing the costume of curated ground truth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

from backend.contracts import CoverageStatus, DecisionLogEntry, DomainContext, GlossaryHit

SOURCE = "domain"
ROOT = Path(__file__).resolve().parent.parent.parent
GLOSSARY_DIR = ROOT / "data" / "glossary"

# `| term | gloss | note |` rows, skipping the header and the `|---|` rule.
_ROW = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")
_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


class DomainGlossaryError(RuntimeError):
    """A glossary manifest is missing or malformed."""


def _key(term: str) -> str:
    return " ".join(term.strip().lower().split())


def parse_manifest(path: Path) -> dict[str, tuple[str, str]]:
    """Parse one `data/glossary/<domain>.md` into ``{key: (gloss, note)}``.

    Validated at the boundary rather than trusted: a glossary file is external
    input to this process exactly like a model response is, and a row whose
    gloss is empty would otherwise resolve a term to nothing and read
    downstream as a successful hit.
    """
    if not path.is_file():
        raise DomainGlossaryError(f"no glossary manifest at {path}")

    entries: dict[str, tuple[str, str]] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        match = _ROW.match(line)
        if not match or _RULE.match(line):
            continue
        cells = [c.strip() for c in match.group("cells").split("|")]
        if len(cells) < 2:
            continue
        term, gloss = cells[0], cells[1]
        note = cells[2] if len(cells) > 2 else ""
        if term.lower() in ("term", "spoken term", "word"):
            continue  # header row
        if not term or not gloss:
            raise DomainGlossaryError(
                f"{path.name}:{lineno}: row has an empty term or gloss — "
                f"a blank resolution would read downstream as a real hit"
            )
        entries[_key(term)] = (gloss.strip(), note)
    if not entries:
        raise DomainGlossaryError(f"{path} parsed to zero entries")
    return entries


class DomainGlossary:
    """Exact-match lookup against one selected domain's curated manifest."""

    def __init__(
        self,
        domain: DomainContext = DomainContext.GENERAL,
        *,
        glossary_dir: Path = GLOSSARY_DIR,
        on_log: Callable[[DecisionLogEntry], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.domain = domain
        self.glossary_dir = glossary_dir
        self._on_log = on_log
        self._clock = clock or time.time
        self._entries = parse_manifest(glossary_dir / f"{domain.value}.md")
        self.hits = 0
        self.misses = 0

    # ---- decision log --------------------------------------------------------

    def _emit(self, stage: str, term: str, detail: str) -> None:
        if self._on_log is None:
            return
        self._on_log(
            DecisionLogEntry(
                timestamp=self._clock(), stage=stage,
                segment_id=f"domain_glossary:{_key(term)}", detail=detail,
            )
        )

    # ---- lookup --------------------------------------------------------------

    def resolve(self, term: str, domain: DomainContext | None = None) -> GlossaryHit | None:
        """Exact-match ``term`` in the selected domain, or None for a clean miss.

        ``domain`` may override the session's selection for a single lookup; it
        reloads that domain's manifest, so the caller gets the real answer for
        the domain it asked about rather than a stale one.
        """
        if domain is not None and domain is not self.domain:
            return DomainGlossary(
                domain, glossary_dir=self.glossary_dir,
                on_log=self._on_log, clock=self._clock,
            ).resolve(term)

        key = _key(term)
        if not key:
            return None

        found = self._entries.get(key)
        if found is None:
            self.misses += 1
            self._emit(
                "DECIDE", term,
                f"domain glossary ({self.domain.value}): miss for {term!r} — "
                f"falling through unchanged, no substitution",
            )
            return None

        gloss, note = found
        self.hits += 1
        self._emit(
            "DECIDE", term,
            f"domain glossary ({self.domain.value}): hit {term!r} -> {gloss!r}"
            + (f" [{note}]" if note else "") + " — no LLM call needed",
        )
        return GlossaryHit(
            term=term.strip(), resolution=gloss, source=SOURCE,
            coverage_status=CoverageStatus.LEXICON_HIT,
        )

    # ---- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and _key(term) in self._entries

    @property
    def terms(self) -> list[str]:
        return sorted(self._entries)


def available_domains(glossary_dir: Path = GLOSSARY_DIR) -> list[DomainContext]:
    """Domains that actually have a manifest on disk."""
    return [d for d in DomainContext if (glossary_dir / f"{d.value}.md").is_file()]
