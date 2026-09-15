"""Sign->Speech: Groq TTS synthesis and virtual-microphone output (T1.6).

Carries the reconstructed sentence from T1.5 out to the call as audio (FR-4).

**The `SETU_TTS_ACTIVE` flag is the load-bearing part of this module, not the
synthesis.** Architecture v3 §8.3 names self-TTS pickup as the single most
likely live-demo failure: Gestura's own speech reaches its own microphone, its
VAD reports voice activity, and the collision manager concludes the hearing
participant has started talking. Every source consulted during planning flagged
it independently. The fix is a boolean, not signal processing — while Gestura is
speaking, overlap events are its own output and must not be treated as external
speech (FR-13).

The flag is exported here and consumed by T1.7's ``should_listen`` gate and
T1.13's collision state machine. It is deliberately module-level and
process-wide: there is exactly one audio output path, so there is exactly one
speaking state.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from groq import APIConnectionError, APIStatusError, GroqError, RateLimitError

from backend.config import TTS_MODEL, get_client
from backend.meeting_bridge.virtualcam import BridgeError, VirtualMicrophone, decode_wav

# Verified against the live API on 2026-09-15 by submitting an invalid voice and
# reading the rejection: [autumn diana hannah austin daniel troy].
AVAILABLE_VOICES = ("autumn", "diana", "hannah", "austin", "daniel", "troy")
DEFAULT_VOICE = "autumn"
RESPONSE_FORMAT = "wav"

# Clause boundaries for streaming. Synthesizing clause-by-clause rather than
# waiting for a whole response is what makes NFR-1's low-single-digit
# time-to-first-audible-output reachable (TRD §7).
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+")


class TTSError(RuntimeError):
    """Speech synthesis or playback failed."""


class _SpeakingState:
    """Process-wide `SETU_TTS_ACTIVE` flag (architecture v3 §8.3).

    Reference-counted so overlapping clause playback cannot clear the flag while
    a later clause is still speaking — a plain boolean would unset on the first
    clause's completion and re-expose the false-overlap bug this exists to
    prevent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._since: float | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._depth > 0

    @property
    def speaking_for(self) -> float:
        with self._lock:
            return 0.0 if self._since is None else time.monotonic() - self._since

    def __enter__(self) -> "_SpeakingState":
        with self._lock:
            if self._depth == 0:
                self._since = time.monotonic()
            self._depth += 1
        return self

    def __exit__(self, *exc_info: object) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self._since = None


SETU_TTS_ACTIVE = _SpeakingState()


def is_self_speaking() -> bool:
    """True while Gestura's own TTS is playing.

    T1.7 passes this as ``should_listen`` (inverted) and T1.13 consults it
    before treating detected voice activity as an external speaker.
    """
    return SETU_TTS_ACTIVE.active


@dataclass(frozen=True)
class Utterance:
    text: str
    audio: bytes
    sample_rate: int
    duration_s: float
    latency_s: float
    voice: str


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence into clauses for streaming synthesis."""
    parts = [part.strip() for part in _CLAUSE_SPLIT.split(sentence.strip()) if part.strip()]
    return parts or ([sentence.strip()] if sentence.strip() else [])


def synthesize(text: str, *, voice: str = DEFAULT_VOICE) -> Utterance:
    """Synthesize one piece of text via Groq TTS."""
    text = text.strip()
    if not text:
        raise TTSError("empty text supplied to synthesize")
    if voice not in AVAILABLE_VOICES:
        raise TTSError(f"voice {voice!r} not in {AVAILABLE_VOICES}")

    started = time.monotonic()
    try:
        response = get_client().audio.speech.create(
            input=text, model=TTS_MODEL, voice=voice, response_format=RESPONSE_FORMAT
        )
    except RateLimitError as exc:
        raise TTSError(
            f"Groq TTS rate limit reached — Stage 1 has no local fallback by "
            f"design (TRD §8): {exc}"
        ) from exc
    except (APIConnectionError, APIStatusError, GroqError) as exc:
        raise TTSError(f"Groq TTS failed for {text!r}: {exc}") from exc

    audio = response.read() if hasattr(response, "read") else response.content
    if not audio:
        raise TTSError(f"Groq TTS returned no audio for {text!r}")

    samples, sample_rate = decode_wav(audio)
    return Utterance(
        text=text,
        audio=audio,
        sample_rate=sample_rate,
        duration_s=len(samples) / sample_rate,
        latency_s=time.monotonic() - started,
        voice=voice,
    )


def stream_utterances(sentence: str, *, voice: str = DEFAULT_VOICE) -> Iterator[Utterance]:
    """Yield synthesized clauses as they become ready, rather than after the whole
    sentence is done (NFR-1)."""
    for clause in split_clauses(sentence):
        yield synthesize(clause, voice=voice)


def speak(
    sentence: str,
    *,
    voice: str = DEFAULT_VOICE,
    microphone: VirtualMicrophone | None = None,
) -> list[Utterance]:
    """Synthesize and play a sentence to the virtual microphone.

    Holds `SETU_TTS_ACTIVE` for the entire span — including the gaps between
    clauses — so a pause mid-sentence is never mistaken for the floor being free.
    """
    try:
        sink = microphone or VirtualMicrophone()
    except BridgeError as exc:
        raise TTSError(f"cannot speak without a virtual microphone: {exc}") from exc

    spoken: list[Utterance] = []
    with SETU_TTS_ACTIVE:
        for utterance in stream_utterances(sentence, voice=voice):
            sink.play_wav(utterance.audio, blocking=True)
            spoken.append(utterance)
    return spoken
