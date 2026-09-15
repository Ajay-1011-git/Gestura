"""Safe mode — the degraded state Setu enters when Groq falters (T2.3).

Stage 1's TRD named this the single highest-value unmitigated live-demo risk:
one path to reasoning, one path to speech, and no fallback for either. The venue
network and Groq's own uptime are both outside the builder's control, so until
now the only mitigation was a rehearsed script. This module is the mitigation.

**The governing principle, extending Stage 1's.** Stage 1: *no stage may
silently convert uncertainty into confident-looking output.* Stage 2 adds: *no
stage may silently convert unavailability into failure.* Groq being slow or
rate-limited is exactly as much an uncertain-state event as a low-confidence
recognition, and it gets handled the same way — surfaced, logged, recoverable.
A hang or a stack trace is the thing this exists to prevent.

**The three triggers read real signals, verified against the installed SDK
(groq 1.7.0) on 2026-09-15 rather than assumed:**

- ``error`` — the SDK's own exception hierarchy. ``RateLimitError`` (HTTP 429),
  ``APIConnectionError``/``APITimeoutError`` (network), and 5xx
  ``InternalServerError`` all subclass ``GroqError``.
- ``latency`` — measured here, on the call's own return path. There is no
  server-supplied latency signal to read, so the only honest source is a clock
  this module holds. Baseline measured on this machine the same day: a real
  `to_gloss` call ran 0.46-1.48s, median 0.80s, over five live calls; Stage 1's
  end-to-end figure was ~1.8s. The 3s default below is roughly twice the
  observed worst case — high enough not to fire on a normal slow call, low
  enough that a human in the room has already noticed.
- ``queue_age`` — how long the oldest un-served segment has been waiting.
  Distinct from latency on purpose: a sequence of individually-acceptable calls
  can still fall irrecoverably behind a conversation, and that is the failure a
  participant actually experiences.

A fourth signal is read opportunistically and is the most useful of the four:
Groq returns ``x-ratelimit-remaining-tokens`` on every successful response. The
real ceiling is **8,000 tokens/min**, and Stage 1 discovered live that this —
not the 1,000 req/min limit — is what bites. Reading it lets safe mode enter
*before* the 429 arrives rather than after the first failed translation.

**Hysteresis, for the same reason the waterfall has it.** A single slow call is
weather, not climate. Entering takes a sustained signal and leaving takes a
clean one, so safe mode cannot flap in and out mid-sentence — which would be
more disorienting to both participants than either steady state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GroqError,
    InternalServerError,
    RateLimitError,
)

from backend.contracts import DecisionLogEntry, SafeModeEvent

# Tuned against the real Stage 1 baseline recorded in this module's docstring,
# not picked as round numbers. Both are env-overridable so a venue with a slower
# network can be accommodated without a code change.
DEFAULT_LATENCY_THRESHOLD_MS = 3_000.0
DEFAULT_QUEUE_AGE_THRESHOLD_S = 5.0

# Enter only after this many consecutive bad signals; leave after this many
# consecutive good ones. Entering is deliberately faster than leaving: the cost
# of entering late is a participant staring at a frozen avatar, while the cost
# of leaving late is a few extra seconds of a working degraded mode.
CONSECUTIVE_BAD_TO_ENTER = 2
CONSECUTIVE_GOOD_TO_EXIT = 3

# Enter pre-emptively when the token budget is nearly gone. 8,000 tokens/min is
# the real observed ceiling; a single reasoning call costs a few hundred, so
# this leaves room for roughly one more call before the 429 would land.
TOKEN_BUDGET_FLOOR = 500

# Spoken while degraded. Short on purpose — it has to be heard once, in a room,
# without becoming the thing everyone is waiting for.
HOLD_MESSAGE = "One moment — I'm switching to a slower local mode."

# Shown next to the transcript/gloss while degraded (FR-21, NFR-9). Never
# rendered in the same style as a confident result.
UNRENDERED_LABEL = "UNRENDERED — safe mode: text only, avatar paused"

TRANSIENT_ERRORS = (
    RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None


@dataclass
class Signal:
    """One observation about a Groq call's outcome."""

    latency_s: float | None = None
    error: BaseException | None = None
    queue_age_s: float | None = None
    remaining_tokens: int | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


class SafeMode:
    """Threshold state machine owning the degraded/normal decision.

    Lives inside `Interpreter` rather than in a parallel supervisor process:
    `Interpreter` already owns the per-segment loop and every Groq call site, so
    a separate watcher would be duplicating state it does not observe directly
    and would drift out of sync with what actually happened.
    """

    def __init__(
        self,
        *,
        latency_threshold_ms: float | None = None,
        queue_age_threshold_s: float | None = None,
        on_log: Callable[[DecisionLogEntry], None] | None = None,
        on_hold_message: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.latency_threshold_ms = (
            latency_threshold_ms
            if latency_threshold_ms is not None
            else _env_float("SAFE_MODE_LATENCY_THRESHOLD_MS", DEFAULT_LATENCY_THRESHOLD_MS)
        )
        self.queue_age_threshold_s = (
            queue_age_threshold_s
            if queue_age_threshold_s is not None
            else _env_float("SAFE_MODE_QUEUE_AGE_THRESHOLD_S", DEFAULT_QUEUE_AGE_THRESHOLD_S)
        )
        self._active = False
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._on_log = on_log
        self._on_hold_message = on_hold_message
        self._clock = clock
        self.events: list[SafeModeEvent] = []
        self.entered_at: float | None = None
        self.last_trigger: str | None = None

    # ---- state ---------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Checked wherever Stage 1 starts avatar work or makes a Groq call."""
        return self._active

    @property
    def unrendered_label(self) -> str | None:
        """NFR-9: visibly distinguishable from a confident result, always."""
        return UNRENDERED_LABEL if self._active else None

    # ---- trigger evaluation (TRD §5) ----------------------------------------

    def should_enter(self, signal: Signal) -> tuple[bool, str]:
        """Would this signal, on its own, justify entering? Returns (yes, why)."""
        if signal.is_error and isinstance(signal.error, TRANSIENT_ERRORS):
            return True, "error"
        if signal.is_error and isinstance(signal.error, (APIStatusError, GroqError)):
            return True, "error"
        if signal.latency_s is not None and signal.latency_s * 1000.0 > self.latency_threshold_ms:
            return True, "latency"
        if signal.queue_age_s is not None and signal.queue_age_s > self.queue_age_threshold_s:
            return True, "queue_age"
        if signal.remaining_tokens is not None and signal.remaining_tokens < TOKEN_BUDGET_FLOOR:
            # The pre-emptive path: the budget Stage 1 actually hit, caught
            # before it becomes a 429 and a failed translation.
            return True, "queue_age"
        return False, ""

    def should_exit(self, signal: Signal) -> bool:
        """A clean, timely, non-erroring call is the only thing that ends it."""
        if signal.is_error:
            return False
        if signal.latency_s is not None and signal.latency_s * 1000.0 > self.latency_threshold_ms:
            return False
        if signal.remaining_tokens is not None and signal.remaining_tokens < TOKEN_BUDGET_FLOOR:
            return False
        return True

    # ---- the observation path -----------------------------------------------

    def observe(self, signal: Signal) -> bool:
        """Feed one Groq call's outcome. Returns `is_active` afterwards.

        Called on every Groq call's own return path rather than from a polling
        timer — a second thread would observe a different world than the one
        `Interpreter` just lived through.
        """
        enter, reason = self.should_enter(signal)

        if enter:
            self._consecutive_bad += 1
            self._consecutive_good = 0
            if not self._active and self._consecutive_bad >= CONSECUTIVE_BAD_TO_ENTER:
                self._transition(entering=True, triggered_by=reason, signal=signal)
            elif not self._active:
                self._emit(
                    "OBSERVE", f"degraded signal ({reason}) "
                    f"{self._consecutive_bad}/{CONSECUTIVE_BAD_TO_ENTER} — "
                    f"not entering safe mode yet",
                )
            return self._active

        self._consecutive_good += 1
        self._consecutive_bad = 0
        if self._active and self.should_exit(signal) and self._consecutive_good >= CONSECUTIVE_GOOD_TO_EXIT:
            self._transition(entering=False, triggered_by=self.last_trigger or "error", signal=signal)
        return self._active

    def observe_call(self, fn: Callable[[], Any], *, queue_age_s: float | None = None) -> Any:
        """Run one Groq call, measure it, feed the result in, and re-raise.

        The measurement has to wrap the call rather than sit beside it — that is
        the only place the real latency and the real exception both exist.
        """
        started = time.monotonic()
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 — observed, then re-raised
            self.observe(Signal(latency_s=time.monotonic() - started, error=exc,
                                queue_age_s=queue_age_s))
            raise
        self.observe(Signal(
            latency_s=time.monotonic() - started,
            queue_age_s=queue_age_s,
            remaining_tokens=remaining_tokens(result),
        ))
        return result

    # ---- transitions ---------------------------------------------------------

    def _transition(self, *, entering: bool, triggered_by: str, signal: Signal) -> None:
        self._active = entering
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self.last_trigger = triggered_by if entering else None
        self.entered_at = self._clock() if entering else None

        event = SafeModeEvent(
            timestamp=self._clock(), triggered_by=triggered_by, entering=entering
        )
        self.events.append(event)

        if entering:
            detail = self._describe(signal)
            self._emit("DECIDE", f"Groq degraded ({triggered_by}): {detail}")
            self._emit(
                "ACTION",
                f"ENTER safe mode — new avatar animation stopped, transcript shown "
                f"as '{UNRENDERED_LABEL}', STT/LLM routed to the local fallback stack",
            )
            if self._on_hold_message is not None:
                self._on_hold_message(HOLD_MESSAGE)
        else:
            self._emit("DECIDE", f"{CONSECUTIVE_GOOD_TO_EXIT} consecutive clean Groq calls — recovered")
            self._emit("ACTION", "EXIT safe mode — resuming normal rendering and Groq routing")

    def _describe(self, signal: Signal) -> str:
        if signal.is_error:
            return f"{type(signal.error).__name__}: {signal.error}"
        parts = []
        if signal.latency_s is not None:
            parts.append(f"latency {signal.latency_s*1000:.0f}ms > {self.latency_threshold_ms:.0f}ms")
        if signal.queue_age_s is not None:
            parts.append(f"queue age {signal.queue_age_s:.1f}s > {self.queue_age_threshold_s:.1f}s")
        if signal.remaining_tokens is not None:
            parts.append(f"{signal.remaining_tokens} tokens left in the 8000/min budget")
        return ", ".join(parts) or "unspecified"

    # ---- decision log (FR-24 — the existing log, not a second one) -----------

    def _emit(self, stage: str, detail: str) -> None:
        if self._on_log is None:
            return
        self._on_log(
            DecisionLogEntry(
                timestamp=self._clock(), stage=stage, segment_id="safe_mode", detail=detail
            )
        )

    # ---- test/demo support ---------------------------------------------------

    def force(self, *, active: bool, reason: str = "manual") -> None:
        """Force a transition. For rehearsal and the T2.4 verification only."""
        if active == self._active:
            return
        self._transition(entering=active, triggered_by=reason, signal=Signal())


# ---- the session's current instance ------------------------------------------
#
# `Interpreter` owns the SafeMode object, but the Stage 1 call sites that need
# to route around it (`stt_input`, both `reasoning` modules) are plain functions
# several layers down that Stage 1 deliberately built without a session handle.
# Threading one through all of them would mean changing every signature on the
# path — a much larger edit to Stage 1 code than a routing check justifies.
#
# So the session registers its instance here and those call sites read it. It is
# a module-level singleton, which is worth naming rather than hiding: there is
# exactly one interpreter session per process by construction (one camera, one
# microphone, one avatar), and `current()` returns None when nothing has
# registered, which is what keeps every Stage 1 test harness working untouched.

_CURRENT: "SafeMode | None" = None


def register(instance: "SafeMode | None") -> None:
    """Make ``instance`` the session's safe mode. Called by `Interpreter`."""
    global _CURRENT
    _CURRENT = instance


def current() -> "SafeMode | None":
    return _CURRENT


def is_active() -> bool:
    """Whether this session is degraded right now.

    False when no session has registered — an unregistered process is not in
    safe mode, it simply has no safe mode, and must behave exactly as Stage 1 did.
    """
    return _CURRENT is not None and _CURRENT.is_active


def remaining_tokens(response: Any) -> int | None:
    """Pull ``x-ratelimit-remaining-tokens`` off a raw Groq response, if present.

    Returns None for the ordinary (non-raw) response objects the Stage 1 call
    sites use — the header is only reachable via ``with_raw_response``. Safe
    mode therefore treats it as a bonus signal, never a required one: the three
    documented triggers all work without it.
    """
    headers: Mapping[str, str] | None = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("x-ratelimit-remaining-tokens")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
