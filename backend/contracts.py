"""Canonical data contracts for Setu Stage 1 (build instructions §B.2).

This file is imported everywhere else and must stay dependency-free of the rest
of the app. Do not redefine any of these types in another module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Direction(Enum):
    SIGN_TO_SPEECH = "sign_to_speech"
    SPEECH_TO_SIGN = "speech_to_sign"


class CoverageStatus(Enum):
    """Mirrors spoken-to-signed-translation's --coverage-info categories.
    Do not invent a different status set — reuse this one everywhere a
    confidence/coverage tier needs to be represented, including the
    Sign->Speech recognition confidence tiering (map recognition confidence
    onto these same four buckets for one consistent status vocabulary
    across both directions)."""

    LEXICON_HIT = "lexicon_hit"          # green
    LANGUAGE_BACKUP = "language_backup"  # yellow
    FINGERSPELLING = "fingerspelling"    # orange
    UNMATCHED = "unmatched"              # red


@dataclass
class Segment:
    id: str
    direction: Direction
    raw_input: str            # transcript (speech->sign) or recognized gloss text (sign->speech)
    confidence: float         # 0.0-1.0
    coverage_status: Optional[CoverageStatus]
    timestamp: float


@dataclass
class DecisionLogEntry:
    timestamp: float
    stage: str          # "OBSERVE" | "DECIDE" | "ACTION"
    segment_id: str
    detail: str          # human-readable, e.g. "candidate disagreement: HIGH"


# --- Stage 2 (build instructions §B.2) ---------------------------------------
#
# Appended, not woven in: everything above is Stage 1's and is byte-identical to
# its Stage 1 form. These three types extend that contract for the resilience
# layer and reuse `CoverageStatus` above rather than introducing a second status
# vocabulary for glossary hits or fingerspelled output (NFR-12).


class DomainContext(Enum):
    """Which curated glossary a session is running against.

    Human-supplied at session start, never inferred — automatic topic-shift
    detection is permanently out of scope for this project, and guessing the
    room's domain would be exactly that by another name.
    """

    GENERAL = "general"
    MEDICAL = "medical"
    TECHNICAL = "technical"


@dataclass
class GlossaryHit:
    term: str
    resolution: str                   # resolved gloss text or pose reference
    source: str                       # "session" | "domain"
    coverage_status: CoverageStatus   # reuse Stage 1's enum — do not define a second one


@dataclass
class SafeModeEvent:
    timestamp: float
    triggered_by: str   # "latency" | "error" | "queue_age"
    entering: bool      # True = entering safe mode, False = exiting
