"""Local LLM reasoning fallback, routed to while safe mode is active (T2.4).

Ollama, running a model on the target machine. Free, offline, and — the point —
independent of the thing that just failed. A fallback that shares a dependency
with the primary is not a fallback; that is why Groq's own catalog, on a second
key or otherwise, cannot serve this role. A venue network drop or a Groq-side
outage takes out both at once, and safe mode would become a label with nothing
behind it.

**API shape verified against the installed client (ollama 0.6.2, 2026-09-15)**
rather than assumed: `ollama.chat(model, messages, *, think, options, format)`
returns a `ChatResponse` whose text is at `response.message.content`.

**This stage's version of a bug Stage 1 already paid for, and the fix that
actually worked.** Stage 1 discovered live that Groq's `gpt-oss` models return
empty content unless `reasoning_effort` is passed, because hidden reasoning
eats the completion budget. Qwen3 is also a reasoning model and fails the same
way from the other direction: measured here, `think=False` did *not* stop it
reasoning — it emitted 2,060 characters of "First, the user asked..." as
visible content and hit the token ceiling mid-thought, so the gloss validator
rejected it.

Prompting harder is not the fix; a local 4B model will not reliably obey "output
only gloss tokens" the way a hosted 20B does. Constraining the *output shape*
is. Every call passes a one-field JSON schema through Ollama's `format`
parameter, so the model must emit `{"answer": "..."}` and cannot preface it
with its reasoning. The same prompt that produced 2,060 characters of rambling
returns `HOSPITAL WHERE` in 0.46s under the schema — faster than the Groq call
it stands in for. `think=False` is still passed, as belt and braces.

**Drop-in shape.** :func:`complete` returns a plain `str`, which is what both
Stage 1 reasoning modules do with `choice.message.content` before validating
it. The call sites keep their own validators — a local model's output gets
exactly the same boundary check the remote one's does, because it is no more
trustworthy for being local.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# Chosen by real benchmarking on the target machine, not by reputation — see
# `scripts/benchmark_fallback.py` and `data/out/fallback_benchmark.json`
# (TNFR-7). Overridable via OLLAMA_MODEL.
DEFAULT_MODEL = "qwen3:4b"

MAX_OUTPUT_CHARS = 2_000

# Deterministic, matching the Groq call sites: a demo cannot rehearse against
# output that changes between runs, and there is no creative latitude wanted in
# either gloss generation or sentence reconstruction.
OPTIONS = {"temperature": 0.0, "num_predict": 512}

# One field, type string. This is what stops a reasoning model narrating its way
# through the completion budget — see the module docstring.
ANSWER_KEY = "answer"
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {ANSWER_KEY: {"type": "string"}},
    "required": [ANSWER_KEY],
}


class LlmFallbackError(RuntimeError):
    """The local LLM fallback could not produce a completion."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    latency_s: float


def _model() -> str:
    return os.environ.get("OLLAMA_MODEL", "").strip() or DEFAULT_MODEL


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LlmFallbackError(
            "the ollama client is not installed — `uv pip install ollama`. Safe "
            "mode falls back to a disclosed pause without it (TRD §8)."
        ) from exc
    return ollama


def complete_verbose(
    prompt: str, *, system: str | None = None, model: str | None = None
) -> Completion:
    """One local completion, with the timing the benchmark and log both want."""
    if not prompt.strip():
        raise LlmFallbackError("empty prompt supplied to the local LLM fallback")

    name = model or _model()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    started = time.monotonic()
    try:
        response = _client().chat(
            model=name,
            messages=messages,
            think=False,
            format=ANSWER_SCHEMA,  # see module docstring — this is what makes it usable
            options=OPTIONS,
        )
    except LlmFallbackError:
        raise
    except Exception as exc:
        raise LlmFallbackError(
            f"local LLM call failed on {name!r}: {exc}. Is `ollama serve` running "
            f"and has `ollama pull {name}` been run?"
        ) from exc
    latency = time.monotonic() - started

    raw = ((getattr(response, "message", None) and response.message.content) or "").strip()
    if not raw:
        raise LlmFallbackError(f"local model {name!r} returned empty content")
    if len(raw) > MAX_OUTPUT_CHARS:
        raise LlmFallbackError(
            f"local model {name!r} returned {len(raw)} chars — refusing a runaway completion"
        )

    # Validated at the boundary: the schema is enforced by Ollama, but a
    # malformed body would otherwise reach the Stage 1 validators looking like
    # a real answer.
    try:
        text = str(json.loads(raw)[ANSWER_KEY]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise LlmFallbackError(
            f"local model {name!r} did not honour the response schema: {raw[:200]!r}"
        ) from exc
    if not text:
        raise LlmFallbackError(f"local model {name!r} returned an empty answer field")

    return Completion(text=text, model=f"ollama/{name}", latency_s=latency)


def complete(prompt: str, *, system: str | None = None, model: str | None = None) -> str:
    """Drop-in for the Groq call sites: prompt in, completion text out."""
    return complete_verbose(prompt, system=system, model=model).text


def is_available(model: str | None = None) -> bool:
    """Whether the local LLM path can actually run — checked, not assumed.

    Verifies the daemon answers *and* that the named model is pulled. Promising
    a fallback that turns out not to be installed would be the second silent
    failure underneath the first that TRD §8 explicitly warns against.
    """
    name = model or _model()
    try:
        listed = _client().list()
    except Exception:
        return False
    models = getattr(listed, "models", None) or []
    available = {getattr(m, "model", "") for m in models}
    return name in available or any(a.split(":")[0] == name.split(":")[0] for a in available)
