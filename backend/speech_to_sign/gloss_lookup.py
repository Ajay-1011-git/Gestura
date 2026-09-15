"""Speech->Sign: gloss to pose lookup with coverage reporting (T1.9).

Wraps `spoken-to-signed-translation`'s lexicon lookup, pointed at an ISL lexicon
built from the curated vocabulary's extracted ``.pose`` files.

Refusing to fabricate is a property this dependency already has rather than
something layered on afterwards: lookup misses are reported as a coverage
category instead of being silently dropped or approximated, which is what FR-7,
FR-8 and FR-12 require.

Two facts verified against the installed package on 2026-09-15, both of which
the build instructions flagged as unknown:

- ``signed_language`` is **not** validated against an ISO allowlist.
  ``gloss_to_pose/languages.py`` holds only a two-entry backup map, so ``ins``
  works as a free-form selector.
- **No ISL fingerspelling lexicon exists.** The package bundles
  ase/asq/bzs/cse/csq/eso/gsg/gss/ise/jos/lls/mfs/psr/sgg/ssp/svk/swl/tsm/ukl —
  ``ise`` is Italian, not Indian, and ``ins`` is absent. An out-of-vocabulary
  term therefore resolves ``UNMATCHED`` rather than ``FINGERSPELLING``.
  See :data:`FINGERSPELLING_STATUS` for why that is left as-is for Stage 1.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

from pose_format import Pose
from spoken_to_signed.gloss_to_pose import gloss_to_pose
from spoken_to_signed.gloss_to_pose.lookup import CSVPoseLookup
from spoken_to_signed.gloss_to_pose.lookup.lookup import CoverageType
from spoken_to_signed.text_to_gloss.types import GlossItem

from backend.contracts import CoverageStatus
from backend.recognition.classifier import active_segment, _hand_present
from backend.recognition.extract import extract_pose_file, load_pose
from backend.speech_to_sign import fingerspell

SPOKEN_LANGUAGE = "en"
SIGNED_LANGUAGE = "ins"  # ISO 639-3 for Indian Sign Language; free-form here
INDEX_COLUMNS = [
    "path", "spoken_language", "signed_language", "start", "end",
    "segment_start", "segment_end", "words", "glosses", "priority",
]

# Stage 2 (T2.6) made this tier reachable. Stage 1 shipped no ISL manual
# alphabet and said so here: the candidate source it named
# (Hemg/Indian_sign_language_dataset) is 128px hand crops with no torso, and the
# avatar needs body landmarks to place hands in signing space. That assessment
# was correct for that dataset and still is.
#
# What changed is the source. `kirandevraj/ISL-Fingerspelling` carries
# full-body signer video with per-letter frame alignment and a stated
# CC-BY-NC-4.0 license, so the handshapes could be extracted through T1.3's
# pipeline with real body landmarks intact. See
# `scripts/build_fingerspelling.py` and `data/fingerspelling/manifest.md`.
#
# The refusal rule is unchanged: an out-of-vocabulary term is spelled, never
# assigned an invented sign, and spelled output stays tagged FINGERSPELLING so
# the coverage report still distinguishes it from a real lexicon hit.
FINGERSPELLING_STATUS = "isl_manual_alphabet"

COVERAGE_MAP = {
    CoverageType.LEXICON: CoverageStatus.LEXICON_HIT,
    CoverageType.LANGUAGE_BACKUP: CoverageStatus.LANGUAGE_BACKUP,
    CoverageType.FINGERSPELLING_BACKUP: CoverageStatus.FINGERSPELLING,
    CoverageType.UNMATCHED: CoverageStatus.UNMATCHED,
}


class LookupError_(RuntimeError):
    """Gloss could not be resolved to a pose sequence."""


def _is_spellable(token: str) -> bool:
    """Does this token contain at least one letter the manual alphabet covers?

    Two ways to fail this. A token that is entirely digits or punctuation has no
    form in a 26-letter alphabet. And an ordinary English word that simply lacks
    a pose in this deployment's 16-sign lexicon is a lexicon gap (R-10), not a
    spelling case — spelling it out would disguise the gap as a success. Both
    stay UNMATCHED, which is the honest, logged refusal Stage 1 already gave.
    """
    return (
        any(c in fingerspell.ALPHABET for c in token.lower())
        and fingerspell.should_spell(token)
    )


@dataclass(frozen=True)
class TokenCoverage:
    gloss: str
    status: CoverageStatus


@dataclass(frozen=True)
class LookupResult:
    pose: Pose | None
    coverage: tuple[TokenCoverage, ...]

    @property
    def unmatched(self) -> tuple[str, ...]:
        return tuple(c.gloss for c in self.coverage if c.status is CoverageStatus.UNMATCHED)

    @property
    def fully_covered(self) -> bool:
        return all(c.status is CoverageStatus.LEXICON_HIT for c in self.coverage)


def _score(path: Path) -> tuple[int, int]:
    """Rank a clip by how much clean signing it contains."""
    pose = load_pose(path)
    mask = _hand_present(pose)
    start, end = active_segment(mask)
    return end - start, int(mask.sum())


def build_lexicon(
    vocab_dir: Path,
    lexicon_dir: Path,
    *,
    raw_video_dir: Path | None = None,
) -> list[tuple[str, Path, int, int]]:
    """Build the `spoken-to-signed-translation` lexicon from extracted vocabulary.

    Picks the single best clip per gloss — the one with the longest clean
    signing segment — and records that segment's millisecond span in
    ``index.csv`` so lookup renders the sign rather than the resting frames
    around it.

    **This is not how the shipped lexicon was built, and running it over
    `data/vocab/` would downgrade it.** The entries on disk come from the ISLRTC
    dictionary at 1920x1080 (`data/lexicon_src/`), measuring 95px of palm;
    `data/vocab/raw_video/` holds the aggregated corpus at 854x480 and 78px. The
    avatar's finger solving reads that difference directly. Point this at ISLRTC
    source video, or accept that every existing entry is replaced with a
    lower-resolution one to gain the new glosses.

    The chosen clip is re-extracted from source video with **all** MediaPipe
    components retained. T1.3 drops FACE_LANDMARKS from recognition poses to cut
    file size roughly 7.7x, but this package's smoothing step indexes that
    component directly and raises without it. Recognition keeps the compact
    format; only these 16 lexicon entries carry the full set.
    """
    target = lexicon_dir / SIGNED_LANGUAGE
    target.mkdir(parents=True, exist_ok=True)
    raw_video_dir = raw_video_dir or (vocab_dir / "raw_video")
    rows: list[tuple[str, Path, int, int]] = []

    for gloss_dir in sorted(p for p in vocab_dir.iterdir() if p.is_dir()):
        if gloss_dir.name == "raw_video":
            continue
        clips = sorted(gloss_dir.glob("*.pose"))
        if not clips:
            continue
        best = max(clips, key=_score)

        pose = load_pose(best)
        start, end = active_segment(_hand_present(pose))
        fps = float(pose.body.fps) or 30.0
        segment_start, segment_end = int(start / fps * 1000), int(end / fps * 1000)

        destination = target / f"{gloss_dir.name.lower()}.pose"
        source_video = next(
            (v for v in (raw_video_dir / gloss_dir.name).glob(f"{best.stem}.*")
             if v.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")),
            None,
        )
        if source_video is None:
            raise LookupError_(
                f"source video for {best.name} not found under {raw_video_dir} — "
                f"lexicon poses must be re-extracted with face landmarks, which "
                f"the recognition poses deliberately omit"
            )
        extract_pose_file(source_video, destination, components=None)
        rows.append((gloss_dir.name, destination, segment_start, segment_end))

    index = lexicon_dir / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        for gloss, destination, segment_start, segment_end in rows:
            writer.writerow([
                f"{SIGNED_LANGUAGE}/{destination.name}",
                SPOKEN_LANGUAGE, SIGNED_LANGUAGE, 0, 0,
                segment_start, segment_end,
                gloss.lower().replace("-", " "), gloss, 0,
            ])
    return rows


class GlossLookup:
    """Resolves gloss tokens to a pose sequence, reporting coverage per token."""

    def __init__(self, lexicon_dir: Path) -> None:
        if not (lexicon_dir / "index.csv").is_file():
            raise LookupError_(
                f"no index.csv in {lexicon_dir} — run build_lexicon() first"
            )
        self.lexicon_dir = lexicon_dir
        self._lookup = CSVPoseLookup(directory=str(lexicon_dir))

    @property
    def vocabulary(self) -> list[str]:
        """The spoken-language *words* the lexicon covers, e.g. ``thank you``."""
        words = self._lookup.words_index.get(SPOKEN_LANGUAGE, {}).get(SIGNED_LANGUAGE, {})
        return sorted(words)

    @property
    def glosses(self) -> list[str]:
        """The *gloss* tokens the lexicon covers, e.g. ``THANK-YOU``.

        Not the same list as :attr:`vocabulary`, and the difference matters to
        anything generating gloss. That one returns the index's spoken words —
        lowercase, and space-separated where a sign covers two words — so a
        multi-word sign arrives as ``thank you``. Handed to a model as the set of
        available signs, it teaches exactly the wrong lesson and comes back as
        two tokens, ``THANK YOU``, neither of which resolves. `coverage_for`
        already normalises across the two spellings on the way back in; this is
        the same correspondence, exposed for the way out.
        """
        with (self.lexicon_dir / "index.csv").open() as handle:
            return sorted({
                row["glosses"].strip().upper()
                for row in csv.DictReader(handle)
                if row.get("glosses", "").strip()
            })

    def coverage_for(self, tokens: list[str]) -> tuple[TokenCoverage, ...]:
        """Per-token coverage without building the pose (FR-8).

        Stage 2 (T2.6) added the third outcome. A token with no lexicon match
        and no language-backup match is no longer automatically UNMATCHED: if
        the ISL manual alphabet can spell it, it is reported FINGERSPELLING and
        the caller renders the spelling. UNMATCHED now means what it always
        should have — nothing here can be signed *or* spelled (FR-27, FR-28).
        """
        known = {w.lower() for w in self.vocabulary}
        spellable = fingerspell.is_available()
        statuses = []
        for token in tokens:
            if token.lower().replace("-", " ") in known or token.lower() in known:
                statuses.append(TokenCoverage(token, CoverageStatus.LEXICON_HIT))
                continue
            if spellable and _is_spellable(token):
                statuses.append(TokenCoverage(token, CoverageStatus.FINGERSPELLING))
                continue
            statuses.append(TokenCoverage(token, CoverageStatus.UNMATCHED))
        return tuple(statuses)

    def fingerspell(self, token: str) -> "fingerspell.SpellResult":
        """Spell one out-of-vocabulary token. Only valid after a real miss.

        Guarded rather than trusting the caller: spelling a token the lexicon
        can actually sign would silently downgrade a real sign to its letters.
        """
        known = {w.lower() for w in self.vocabulary}
        if token.lower().replace("-", " ") in known or token.lower() in known:
            raise LookupError_(
                f"{token!r} has a validated sign — refusing to fingerspell it instead"
            )
        return fingerspell.spell(token)

    def lookup(self, gloss: str) -> LookupResult:
        """Resolve a gloss string to a pose sequence plus per-token coverage.

        An unmatched token never silently disappears: it is reported so the
        waterfall can refuse rather than render a partial sentence as if it were
        complete (FR-12).
        """
        tokens = [t for t in gloss.split() if t]
        if not tokens:
            raise LookupError_("empty gloss supplied")

        coverage = self.coverage_for(tokens)
        renderable = [c.gloss for c in coverage if c.status is CoverageStatus.LEXICON_HIT]
        if not renderable:
            return LookupResult(pose=None, coverage=coverage)

        items = [
            GlossItem(word=t.lower().replace("-", " "), gloss=t.lower())
            for t in renderable
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = gloss_to_pose(
                    items, self._lookup, SPOKEN_LANGUAGE, SIGNED_LANGUAGE
                )
        except Exception as exc:
            raise LookupError_(f"pose lookup failed for {gloss!r}: {exc}") from exc

        return LookupResult(pose=result.pose, coverage=coverage)
