"""Rendering collision management (T1.13), per architecture v3 §8.1-8.4.

Two state machines run in parallel — one for what the avatar is rendering, one
for what the local audio path contains — and a hold fires only when both agree
that Gestura's output and a real incoming utterance would otherwise collide.

**Naming matters here and is not cosmetic.** This is *rendering collision*
management, not floor management. Gestura observes only its own local audio
path; it cannot know who holds the conferencing platform's speaking floor, and
NFR-4 forbids claiming otherwise. Nothing in this module infers a speaker.

Three rules that are easy to get subtly wrong, so they are enforced explicitly:

- A hold needs *sustained* voice, not one frame. A cough, a keyboard click or a
  door closing must not stall the avatar (§8.2).
- Release happens at a clip boundary, never mid-gesture. Interrupting a sign
  partway through produces a different sign, or nonsense.
- Gestura's own TTS is never external speech. That is the failure mode §8.3
  singles out, and it is suppressed by consulting `SETU_TTS_ACTIVE` rather than
  by trying to separate sources acoustically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from backend.sign_to_speech.tts_output import is_self_speaking


class AvatarState(Enum):
    IDLE = "idle"
    SIGNING = "signing"
    DRAINING = "draining"


class AudioState(Enum):
    SILENCE = "silence"
    VOICE_DETECTED = "voice_detected"
    SPEAKING = "speaking"


class HoldDecision(Enum):
    CONTINUE = "continue"
    HOLD = "hold"
    RELEASE = "release"


# §8.2: sustained voice activity, not a single VAD frame.
VOICE_DEBOUNCE_S = 0.35          # within the 250-400ms window the spec gives
SILENCE_RELEASE_S = 0.40
MAX_HOLD_S = 8.0                 # never hold indefinitely; surface the backlog

# §8.2 priority surfacing: these must not sit in a queue behind routine content.
PRIORITY_PATTERNS = ("?", "name", "number", "emergency", "pain", "allergic", "doctor")


@dataclass
class QueuedItem:
    """One unit of avatar output waiting to render."""

    text: str
    duration_s: float
    priority: bool = False
    queued_at: float = field(default_factory=time.monotonic)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.queued_at


def is_priority(text: str) -> bool:
    """Whether held content should surface sooner than routine content (§8.2)."""
    lowered = text.lower()
    return any(p in lowered for p in PRIORITY_PATTERNS) or any(c.isdigit() for c in lowered)


@dataclass
class Transition:
    at: float
    avatar: AvatarState
    audio: AudioState
    decision: HoldDecision
    detail: str


class CollisionManager:
    """Coordinates avatar rendering against detected local audio activity."""

    def __init__(
        self,
        *,
        voice_debounce_s: float = VOICE_DEBOUNCE_S,
        silence_release_s: float = SILENCE_RELEASE_S,
        max_hold_s: float = MAX_HOLD_S,
        self_speaking: Callable[[], bool] = is_self_speaking,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.voice_debounce_s = voice_debounce_s
        self.silence_release_s = silence_release_s
        self.max_hold_s = max_hold_s
        self._self_speaking = self_speaking
        self._clock = clock

        self.avatar = AvatarState.IDLE
        self.audio = AudioState.SILENCE
        self.transitions: list[Transition] = []

        self._queue: list[QueuedItem] = []
        self._current: QueuedItem | None = None
        self._clip_ends_at: float | None = None
        self._voice_since: float | None = None
        self._silence_since: float | None = None
        self._held_since: float | None = None
        self._suppressed_frames = 0

    @property
    def holding(self) -> bool:
        return self._held_since is not None

    @property
    def suppressed_self_frames(self) -> int:
        """How many voice frames were attributed to Gestura's own TTS (§8.3)."""
        return self._suppressed_frames

    def _record(self, decision: HoldDecision, detail: str) -> None:
        self.transitions.append(
            Transition(self._clock(), self.avatar, self.audio, decision, detail)
        )

    def submit(self, text: str, duration_s: float) -> QueuedItem:
        """Queue avatar output. Priority items jump ahead of routine ones."""
        item = QueuedItem(text=text, duration_s=duration_s, priority=is_priority(text))
        if item.priority:
            routine = next(
                (i for i, q in enumerate(self._queue) if not q.priority), len(self._queue)
            )
            self._queue.insert(routine, item)
        else:
            self._queue.append(item)
        self._record(HoldDecision.CONTINUE, f"queued {'priority' if item.priority else 'routine'}: {text!r}")
        return item

    def observe_audio(self, voice_present: bool) -> HoldDecision:
        """Feed one VAD frame. Returns the current hold decision.

        A voice frame arriving while Gestura is speaking is Gestura's own output
        and is discarded before any state change — FR-13, and the single most
        likely live-demo failure per §8.3.
        """
        now = self._clock()

        if voice_present and self._self_speaking():
            self._suppressed_frames += 1
            self._record(
                HoldDecision.CONTINUE,
                "voice frame attributed to own TTS (SETU_TTS_ACTIVE), not external speech",
            )
            return HoldDecision.HOLD if self.holding else HoldDecision.CONTINUE

        if voice_present:
            self._silence_since = None
            if self._voice_since is None:
                self._voice_since = now
                self.audio = AudioState.VOICE_DETECTED
                self._record(HoldDecision.CONTINUE, "voice detected, awaiting debounce")
            elif now - self._voice_since >= self.voice_debounce_s:
                if self.audio is not AudioState.SPEAKING:
                    self.audio = AudioState.SPEAKING
                    self._record(
                        HoldDecision.CONTINUE,
                        f"voice sustained past {self.voice_debounce_s:.2f}s debounce",
                    )
        else:
            self._voice_since = None
            if self._silence_since is None:
                self._silence_since = now
            elif now - self._silence_since >= self.silence_release_s:
                if self.audio is not AudioState.SILENCE:
                    self.audio = AudioState.SILENCE
                    self._record(HoldDecision.CONTINUE, "audio returned to silence")

        return self._evaluate()

    def _evaluate(self) -> HoldDecision:
        now = self._clock()
        avatar_busy = self.avatar in (AvatarState.SIGNING, AvatarState.DRAINING) or bool(self._queue)

        if self.audio is AudioState.SPEAKING and avatar_busy and not self.holding:
            self._held_since = now
            self._record(HoldDecision.HOLD, "overlap: real speech while avatar output pending")
            return HoldDecision.HOLD

        if self.holding:
            if now - self._held_since >= self.max_hold_s:
                self._held_since = None
                self._record(
                    HoldDecision.RELEASE,
                    f"max hold {self.max_hold_s:.0f}s exceeded — surfacing backlog rather than holding indefinitely",
                )
                return HoldDecision.RELEASE
            if self.audio is AudioState.SILENCE:
                self._held_since = None
                self._record(HoldDecision.RELEASE, "incoming speech ended")
                return HoldDecision.RELEASE
            return HoldDecision.HOLD

        return HoldDecision.CONTINUE

    def tick(self) -> QueuedItem | None:
        """Advance rendering. Returns the clip that just started, if any.

        Never interrupts a clip in flight: a hold only prevents the *next* clip
        from starting, so release always lands on a clip boundary (§8.2).
        """
        now = self._clock()

        if self._current is not None:
            if self._clip_ends_at is not None and now < self._clip_ends_at:
                return None
            finished, self._current, self._clip_ends_at = self._current, None, None
            self.avatar = AvatarState.DRAINING if self._queue else AvatarState.IDLE
            self._record(HoldDecision.CONTINUE, f"clip boundary reached after {finished.text!r}")

        if self.holding:
            return None

        if not self._queue:
            if self.avatar is not AvatarState.IDLE:
                self.avatar = AvatarState.IDLE
                self._record(HoldDecision.CONTINUE, "queue empty, avatar idle")
            return None

        item = self._queue.pop(0)
        self._current = item
        self._clip_ends_at = now + item.duration_s
        self.avatar = AvatarState.SIGNING
        self._record(
            HoldDecision.CONTINUE,
            f"started {'priority ' if item.priority else ''}clip {item.text!r} "
            f"(waited {item.age_s:.2f}s)",
        )
        return item
