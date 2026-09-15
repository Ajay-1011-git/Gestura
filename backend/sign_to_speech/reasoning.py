"""Sign->Speech: reconstruct a fluent English sentence from recognized gloss (T1.5).

ISL gloss drops the copula, articles and inflection that English requires, so
this step is a reasoning call rather than a word-for-word mapping (FR-3). It
must add grammar without adding *content* — inventing information the signer did
not sign would be exactly the fabrication the project's governing principle
forbids (TRD §2).

``confidence`` is accepted but deliberately does not change the phrasing. Hedging
language belongs to the escalation waterfall (T1.14), which owns the decision to
translate, clarify or refuse; this module's only job is faithful reconstruction.
The parameter exists as the hook point for Stage 3's FAST/CAREFUL model policy —
that split is not implemented here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from groq import APIConnectionError, APIStatusError, GroqError, RateLimitError

from backend.config import FAST_MODEL, REASONING_EFFORT, get_client

# Headroom for hidden reasoning, not for the answer. gpt-oss spends 138-184
# tokens reasoning at reasoning_effort="low" even for a one-sentence reply; a
# 200-token budget truncated intermittently and returned empty content with
# finish_reason="length". Found while verifying T1.8, fixed in both directions.
MAX_COMPLETION_TOKENS = 512
MAX_SENTENCE_CHARS = 300

SYSTEM_PROMPT = """You convert Indian Sign Language gloss into fluent English.

Gloss is written in capitals and omits articles, the verb "to be", and \
inflection. Restore only what English grammar requires.

Rules:
- Output exactly one sentence. No preamble, no explanation, no quotation marks.
- Add ONLY grammatical words (articles, copula, inflection, prepositions).
- Never add facts, names, numbers, places or details absent from the gloss.
- If the gloss is a single word that is already a complete utterance \
(a greeting, "yes", "no"), return it as natural English without inventing \
a sentence around it.
- Preserve the speaker's person. ME/I is first person, YOU is second person.
- ISL places question words at the end of a clause; English places them first. \
Reorder accordingly."""


class ReasoningError(RuntimeError):
    """Sentence reconstruction failed."""


@dataclass(frozen=True)
class Reconstruction:
    sentence: str
    gloss: str
    model: str
    latency_s: float


def _validate(text: str, gloss: str) -> str:
    """Validate the model's output at the boundary before trusting it."""
    sentence = text.strip().strip('"').strip()
    if not sentence:
        raise ReasoningError(f"model returned empty output for gloss {gloss!r}")
    if len(sentence) > MAX_SENTENCE_CHARS:
        raise ReasoningError(
            f"model returned {len(sentence)} chars for gloss {gloss!r}, "
            f"expected a single sentence under {MAX_SENTENCE_CHARS}"
        )
    return sentence


def reconstruct_sentence(
    gloss: str,
    confidence: float,
    *,
    model: str = FAST_MODEL,
) -> Reconstruction:
    """Turn recognized gloss into one fluent English sentence.

    ``confidence`` is the recognition confidence from T1.4. It is the Stage 3
    FAST/CAREFUL hook point and does not affect phrasing — see module docstring.
    """
    gloss = gloss.strip()
    if not gloss:
        raise ReasoningError("empty gloss supplied")
    if not 0.0 <= confidence <= 1.0:
        raise ReasoningError(f"confidence {confidence} outside 0.0-1.0")

    # Stage 2 (T2.4) routing check: while degraded, reason locally instead.
    # The local model's output gets the same `_validate` boundary check the
    # remote one's does — being local makes it no more trustworthy.
    from backend.resilience import safe_mode

    if safe_mode.is_active():
        from backend.resilience.llm_fallback import LlmFallbackError, complete_verbose

        try:
            completion = complete_verbose(f"Gloss: {gloss}", system=SYSTEM_PROMPT)
        except LlmFallbackError as exc:
            raise ReasoningError(f"local reasoning fallback failed for {gloss!r}: {exc}") from exc
        return Reconstruction(
            sentence=_validate(completion.text, gloss),
            gloss=gloss,
            model=completion.model,
            latency_s=completion.latency_s,
        )

    started = time.monotonic()
    try:
        response = get_client().chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Gloss: {gloss}"},
            ],
            model=model,
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.2,
        )
    except RateLimitError as exc:
        raise ReasoningError(
            f"Groq rate limit hit reconstructing {gloss!r} — Stage 1 has no local "
            f"fallback by design (TRD §8); safe-mode degradation is Stage 2: {exc}"
        ) from exc
    except (APIConnectionError, APIStatusError, GroqError) as exc:
        raise ReasoningError(f"Groq call failed for gloss {gloss!r}: {exc}") from exc

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ReasoningError(
            f"model hit the {MAX_COMPLETION_TOKENS}-token budget before emitting "
            f"a sentence for gloss {gloss!r} — hidden reasoning consumed the completion"
        )

    return Reconstruction(
        sentence=_validate(choice.message.content or "", gloss),
        gloss=gloss,
        model=response.model,
        latency_s=time.monotonic() - started,
    )
