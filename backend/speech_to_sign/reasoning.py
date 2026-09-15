"""Speech->Sign: convert an English transcript into ISL gloss (T1.8).

The reverse of :mod:`backend.sign_to_speech.reasoning`, and the harder
direction. English is SVO with articles and a copula; ISL is broadly SOV, drops
articles and "to be", puts time at the start of a clause and question words at
the end. This is a reasoning step (FR-6), not a lookup.

**Stated limitation, per Risk R-2.** There is no ISL grammar verifier available
in this timeline, so what this module produces is gloss that looks correct to a
hearing engineer. That is a materially weaker claim than "grammatically correct
ISL" and should be stated that way to evaluators rather than discovered by them.

The gloss this emits is consumed by T1.9's lexicon lookup, which reports a
`CoverageStatus` per token — so a hallucinated gloss term does not silently
become a rendered sign; it surfaces as FINGERSPELLING or UNMATCHED.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

from groq import APIConnectionError, APIStatusError, GroqError, RateLimitError

from backend.config import FAST_MODEL, REASONING_EFFORT, get_client

# Headroom for hidden reasoning, not for the answer. The gloss itself is a
# handful of tokens, but gpt-oss spends 138-184 tokens reasoning at
# reasoning_effort="low"; a 200-token budget truncated intermittently and
# surfaced as empty content with finish_reason="length".
MAX_COMPLETION_TOKENS = 512
MAX_GLOSS_TOKENS = 24
GLOSS_TOKEN_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]*$")

SYSTEM_PROMPT = """You convert English into Indian Sign Language gloss.

Gloss is the written notation for signs, not English. Rules:
- Output ONLY gloss tokens in CAPITALS, separated by single spaces.
- No punctuation, no explanation, no quotation marks, no lowercase.
- Drop articles (a, an, the) and forms of "to be" (is, am, are, was, were).
- Drop inflection: signs are uninflected. WALKED becomes WALK.
- Word order is broadly Subject-Object-Verb, not Subject-Verb-Object.
- Put time markers FIRST: "You are going today" becomes TODAY YOU GO.
- Put question words LAST: "Where is the hospital" becomes HOSPITAL WHERE.
- Negation follows the verb: "He is not going" becomes HE GO NOT.
- Use a hyphen only inside a single multi-word sign, e.g. THANK-YOU.
- Gloss ONLY what the sentence says. Do not add a subject the sentence does not
have: "Please sit down" has no subject, so it is PLEASE SIT, not PLEASE ME SIT.
- Do not drop content either. Every meaningful word in the sentence must appear.
- Never invent a sign for a proper noun. Keep names and places as-is in \
capitals so they can be fingerspelled downstream."""

# Appended when the caller knows which signs are actually available.
#
# This is preference, deliberately not a constraint. Restricting the model to the
# curated list would make it reach for the nearest available sign whenever a
# concept has none — substituting SIT for "lie down", or dropping the term
# entirely — which is precisely the fabrication FR-12 exists to prevent. A term
# with no validated sign has to survive this step intact so that T1.9's coverage
# report can mark it UNMATCHED and the waterfall can refuse it. What the list
# does fix is the model reaching for a *different* sign than the one the lexicon
# holds for the same idea.
VOCABULARY_PROMPT = """
Signs available in this deployment's lexicon:
{vocabulary}

Prefer these exact tokens when one of them genuinely expresses part of the
sentence. If part of the sentence has no sign in that list, still emit the
natural gloss token for it — do NOT substitute a listed sign that means
something else, and do NOT silently omit it. Something downstream reports
unavailable terms; guessing hides them."""


class GlossError(RuntimeError):
    """Gloss generation failed."""


@dataclass(frozen=True)
class GlossResult:
    gloss: str
    tokens: tuple[str, ...]
    transcript: str
    model: str
    latency_s: float


def _validate(text: str, transcript: str) -> tuple[str, tuple[str, ...]]:
    """Validate gloss at the boundary before T1.9 is asked to look it up."""
    cleaned = text.strip().strip('"').replace("\n", " ")
    cleaned = re.sub(r"[.,!?;:]", " ", cleaned)
    tokens = tuple(t for t in cleaned.split() if t)

    if not tokens:
        raise GlossError(f"model returned no gloss tokens for {transcript!r}")
    if len(tokens) > MAX_GLOSS_TOKENS:
        raise GlossError(
            f"model returned {len(tokens)} gloss tokens for {transcript!r}, "
            f"over the {MAX_GLOSS_TOKENS} limit"
        )
    malformed = [t for t in tokens if not GLOSS_TOKEN_PATTERN.match(t)]
    if malformed:
        raise GlossError(
            f"model returned non-gloss tokens {malformed} for {transcript!r} — "
            f"gloss must be uppercase tokens only"
        )
    return " ".join(tokens), tokens


def to_gloss(
    transcript: str,
    *,
    model: str = FAST_MODEL,
    vocabulary: Sequence[str] | None = None,
) -> GlossResult:
    """Convert one English transcript into ISL gloss notation.

    ``vocabulary`` is the caller's list of signs that actually have a validated
    pose — pass ``GlossLookup.glosses``, which is the gloss spelling
    (``THANK-YOU``); ``GlossLookup.vocabulary`` is the spoken wording
    (``thank you``) and glossing from it produces two tokens that resolve to
    nothing. It is guidance, not a filter: see
    ``VOCABULARY_PROMPT``. Left out, the model glosses from general knowledge and
    is measurably worse at it, inventing tokens the deployment cannot render and
    dropping ones it can.
    """
    transcript = transcript.strip()
    if not transcript:
        raise GlossError("empty transcript supplied")

    prompt = SYSTEM_PROMPT
    if vocabulary:
        # Comma-separated: a space-separated list cannot distinguish one
        # two-word sign from two one-word signs.
        prompt += VOCABULARY_PROMPT.format(
            vocabulary=", ".join(sorted({v.strip().upper() for v in vocabulary}))
        )

    # Stage 2 (T2.4) routing check: while degraded, gloss locally instead. The
    # vocabulary prompt goes to the local model too — a fallback that glosses
    # from general knowledge would invent tokens this deployment cannot render,
    # which is the exact failure VOCABULARY_PROMPT exists to prevent.
    from backend.resilience import safe_mode

    if safe_mode.is_active():
        from backend.resilience.llm_fallback import LlmFallbackError, complete_verbose

        try:
            completion = complete_verbose(transcript, system=prompt)
        except LlmFallbackError as exc:
            raise GlossError(f"local gloss fallback failed for {transcript!r}: {exc}") from exc
        gloss, tokens = _validate(completion.text, transcript)
        return GlossResult(
            gloss=gloss, tokens=tokens, transcript=transcript,
            model=completion.model, latency_s=completion.latency_s,
        )

    started = time.monotonic()
    try:
        response = get_client().chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            model=model,
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            # Deterministic. The same transcript returned 'HELLO ME SIT DOWN' and
            # 'SIT DOWN' on consecutive calls at 0.2 — a demo cannot rehearse
            # against output that changes between runs, and there is no creative
            # latitude wanted here anyway.
            temperature=0.0,
        )
    except RateLimitError as exc:
        raise GlossError(
            f"Groq rate limit hit glossing {transcript!r} — Stage 1 has no local "
            f"fallback by design (TRD §8): {exc}"
        ) from exc
    except (APIConnectionError, APIStatusError, GroqError) as exc:
        raise GlossError(f"Groq call failed for {transcript!r}: {exc}") from exc

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise GlossError(
            f"model hit the {MAX_COMPLETION_TOKENS}-token budget before emitting "
            f"gloss for {transcript!r} — hidden reasoning consumed the completion"
        )

    gloss, tokens = _validate(choice.message.content or "", transcript)
    return GlossResult(
        gloss=gloss,
        tokens=tokens,
        transcript=transcript,
        model=response.model,
        latency_s=time.monotonic() - started,
    )


def in_vocabulary(tokens: Sequence[str], vocabulary: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split gloss tokens into those the curated vocabulary covers and those it does not.

    Cheap pre-check before T1.9's lookup, so the waterfall can see immediately
    which tokens have no validated sign (FR-1, FR-12).
    """
    known = {v.upper() for v in vocabulary}
    covered = [t for t in tokens if t in known]
    missing = [t for t in tokens if t not in known]
    return covered, missing
