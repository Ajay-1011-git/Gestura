"""Fingerspelling — spelling a term the lexicon cannot sign (T2.6).

This is the rung Stage 1 defined but could not reach. `CoverageStatus` has
carried a `FINGERSPELLING` tier since the Stage 1 contracts were written, but
`spoken-to-signed-translation` bundles no ISL lexicon (`ins` is absent; `ise`
is Italian), so nothing could ever produce that status and every
out-of-vocabulary term fell straight to `UNMATCHED`. The handshape library
built by `scripts/build_fingerspelling.py` closes that gap.

**Where this sits in the ladder.** It is the last step before refusal, not an
alternative to the lexicon. `gloss_lookup.py` calls it only after its own
lookup reports no lexicon hit *and* no language-backup hit. A term that has a
real sign must always get the real sign — spelling out a word the deployment
can actually sign would be a downgrade dressed up as a feature.

**Why spelling is not fabrication.** The project's governing rule is that no
stage may silently convert uncertainty into confident-looking output, and
fingerspelling does not break it: spelling a name letter by letter is what a
human interpreter does with a name, and it claims nothing beyond the letters
themselves. What would break the rule is inventing a *sign* for an unknown
term, which is why that remains refused. The output is tagged `FINGERSPELLING`,
never folded into `LEXICON_HIT`, so the coverage report still says plainly that
this term was spelled rather than signed (FR-28, NFR-12).

**What cannot be spelled is still refused.** Digits and punctuation have no
entry in a 26-letter manual alphabet. A term containing them is reported as a
partial result with the unspellable characters named, and the caller decides —
it is not silently dropped to make the output look complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from pose_format import Pose

from backend.contracts import CoverageStatus

ROOT = Path(__file__).resolve().parent.parent.parent
FINGERSPELLING_DIR = ROOT / "data" / "fingerspelling"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# Curated list of ordinary English words with established ISL signs. A miss on
# one of these is a gap in the 16-sign render lexicon (R-10), not a word that
# wants spelling — see the file's own header for the full argument.
DO_NOT_SPELL = FINGERSPELLING_DIR / "do-not-spell.md"

# Held between letters so a spelled word reads as discrete letters rather than
# one continuous smear. Matches the inter-sign pause `build_avatar_sequence`
# already uses between lexicon entries.
INTER_LETTER_HOLD_FRAMES = 2

MAX_SPELLABLE_CHARS = 32


class FingerspellError(RuntimeError):
    """A term could not be fingerspelled."""


@dataclass(frozen=True)
class SpelledLetter:
    letter: str
    frames: int
    source: Path


@dataclass
class SpellResult:
    """What spelling a term produced, and what it could not cover."""

    term: str
    pose: Pose | None
    letters: tuple[SpelledLetter, ...] = ()
    unspellable: tuple[str, ...] = ()
    coverage_status: CoverageStatus = CoverageStatus.FINGERSPELLING

    @property
    def frames(self) -> int:
        return int(self.pose.body.data.shape[0]) if self.pose is not None else 0

    @property
    def complete(self) -> bool:
        """True when every character of the term was actually spelled."""
        return self.pose is not None and not self.unspellable

    @property
    def spelled(self) -> str:
        return "".join(letter.letter for letter in self.letters)


@lru_cache(maxsize=1)
def available_letters(directory: Path = FINGERSPELLING_DIR) -> frozenset[str]:
    """Which letters actually have a `.pose` on disk — checked, never assumed."""
    if not directory.is_dir():
        return frozenset()
    return frozenset(
        path.stem.lower() for path in directory.glob("*.pose")
        if len(path.stem) == 1 and path.stem.lower() in ALPHABET
    )


@lru_cache(maxsize=1)
def unspellable_words(path: Path = DO_NOT_SPELL) -> frozenset[str]:
    """Words that must surface as UNMATCHED rather than be spelled out."""
    if not path.is_file():
        return frozenset()
    words: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cell = stripped.strip("|").strip()
        if not cell or set(cell) <= set("-: ") or cell.lower() == "word":
            continue
        words.add(cell.lower())
    return frozenset(words)


def should_spell(term: str, path: Path = DO_NOT_SPELL) -> bool:
    """Does this term warrant fingerspelling, as opposed to an honest refusal?

    Fingerspelling is for proper nouns, acronyms and terms with genuinely no
    sign. Spelling an ordinary word because *this deployment* happens to lack
    its pose would present a lexicon gap as a successful translation — and
    fifteen handshapes for "I need" is not a translation a Deaf viewer can read.
    """
    cleaned = term.strip().lower().replace("-", " ")
    if not cleaned:
        return False
    stop = unspellable_words(path)
    # A multi-word gloss is spellable only if some part of it warrants it.
    return any(
        word not in stop for word in cleaned.split() if any(c in ALPHABET for c in word)
    )


def is_available(directory: Path = FINGERSPELLING_DIR) -> bool:
    """Whether the library is complete enough to promise spelling at all.

    A partial library is worse than none: it would spell some terms and refuse
    others for reasons no participant could see. All 26 or nothing.
    """
    return len(available_letters(directory)) == 26


@lru_cache(maxsize=64)
def _load_letter(letter: str, directory: Path = FINGERSPELLING_DIR) -> Pose:
    path = directory / f"{letter}.pose"
    if not path.is_file():
        raise FingerspellError(f"no handshape for {letter!r} at {path}")
    with path.open("rb") as handle:
        return Pose.read(handle.read())


def spell(
    term: str,
    *,
    directory: Path = FINGERSPELLING_DIR,
    inter_letter_hold: int = INTER_LETTER_HOLD_FRAMES,
) -> SpellResult:
    """Spell ``term`` letter by letter into one concatenated pose sequence.

    Returns a `SpellResult` whose `pose` is None when nothing at all could be
    spelled — the caller then refuses, exactly as it did before this module
    existed. A partial result keeps `unspellable` populated rather than quietly
    presenting an incomplete spelling as a whole one.
    """
    cleaned = term.strip()
    if not cleaned:
        raise FingerspellError("empty term supplied")
    if len(cleaned) > MAX_SPELLABLE_CHARS:
        raise FingerspellError(
            f"refusing to fingerspell {len(cleaned)} characters — "
            f"{term!r} is longer than the {MAX_SPELLABLE_CHARS}-character limit"
        )

    letters: list[SpelledLetter] = []
    unspellable: list[str] = []
    chunks: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    header = None

    for character in cleaned.lower():
        if character in (" ", "-", "'", "."):
            continue  # word separators carry no handshape; not an error
        if character not in ALPHABET:
            unspellable.append(character)
            continue
        try:
            pose = _load_letter(character, directory)
        except FingerspellError:
            unspellable.append(character)
            continue

        if header is None:
            header = pose.header
        data = np.asarray(pose.body.data)
        confidence = np.asarray(pose.body.confidence)
        chunks.append(data)
        confidences.append(confidence)

        if inter_letter_hold > 0:
            # Repeat the final frame rather than inserting a rest pose: the hand
            # settles on the letter instead of returning to neutral between
            # every character, which is how spelling actually looks.
            chunks.append(np.repeat(data[-1:], inter_letter_hold, axis=0))
            confidences.append(np.repeat(confidence[-1:], inter_letter_hold, axis=0))

        letters.append(
            SpelledLetter(letter=character, frames=int(data.shape[0]),
                          source=directory / f"{character}.pose")
        )

    if not chunks or header is None:
        return SpellResult(
            term=cleaned, pose=None, letters=(), unspellable=tuple(unspellable),
            coverage_status=CoverageStatus.UNMATCHED,
        )

    from pose_format.numpy import NumPyPoseBody

    body = NumPyPoseBody(
        fps=float(_load_letter(letters[0].letter, directory).body.fps),
        data=np.ma.array(np.concatenate(chunks, axis=0)),
        confidence=np.concatenate(confidences, axis=0),
    )
    return SpellResult(
        term=cleaned,
        pose=Pose(header=header, body=body),
        letters=tuple(letters),
        unspellable=tuple(unspellable),
        coverage_status=CoverageStatus.FINGERSPELLING,
    )


def describe(result: SpellResult) -> str:
    """One line for the decision log — what was spelled, and what was not."""
    if result.pose is None:
        return f"{result.term!r}: nothing spellable in the ISL manual alphabet"
    spelled = "-".join(letter.letter.upper() for letter in result.letters)
    note = (f"; {len(result.unspellable)} character(s) unspellable: "
            f"{''.join(result.unspellable)}" if result.unspellable else "")
    return (f"{result.term!r} fingerspelled as {spelled} "
            f"({result.frames} frames){note}")
