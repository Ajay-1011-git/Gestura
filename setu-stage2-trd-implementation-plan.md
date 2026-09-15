# Technical Requirements Document (TRD) — Implementation Plan
## Setu — Stage 2: Resilience & Adaptive Behaviors

**Implementation target:** extension of the existing hybrid Python backend + browser-based Three.js frontend from Stage 1 — no new deployment surface, no new runtime target.

**Relationship to other documents:** this TRD explains the *why* behind Stage 2's technical decisions; task-by-task execution lives in `setu-stage2-build-instructions.md` (task IDs `T2.0`–`T2.6`), which this document cross-references. Functional requirements (`FR-n`, `NFR-n`) are defined in `setu-stage2-prd.md`, continuing Stage 1's numbering. If this document conflicts with `setu-final-architecture-v3.md`, the architecture document wins. If it conflicts with anything already built and verified in Stage 1, the as-built Stage 1 code is ground truth — this document does not get to silently redesign something Stage 1 already shipped and tested.

---

## 1. Purpose and Scope

This document specifies how Stage 2 is technically built: the stack additions, the reasoning behind each, the component architecture as it now extends Stage 1's, the new data contracts, and the reliability approach for the specific failure mode this stage exists to close. Scope is exactly Stage 2 as defined in the PRD §4.1 — session/domain glossary, safe-mode degradation, the local fallback stack, duplicate-ack batching, and the ISL fingerspelling handshape library. Stage 3 technical planning exists separately, at lower detail, in `setu-stage3-plan.md` (unchanged by this document).

## 2. System Context

Stage 1 established four compute/interaction domains (capture & recognition, reasoning, rendering, coordination) with `Interpreter` (`backend/orchestrator.py`, built as T1.17) as the single integration point running the per-segment loop for both directions. Stage 2 does not add a fifth domain — it adds a **resilience layer** that sits across the existing reasoning and coordination domains, and it fills in two waterfall hooks (`session_glossary`, `domain_glossary`) that Stage 1 deliberately left as no-op passthroughs.

**The single governing principle for this stage**, extending Stage 1's "no stage may silently convert uncertainty into confident-looking output": **no stage may silently convert unavailability into failure.** When Groq is slow, erroring, or rate-limited, that is exactly as much an uncertain-state event as a low-confidence recognition — and it must be surfaced and handled the same way: visibly, logged, and recoverable, never a silent hang or crash. Safe mode is this principle's direct implementation.

## 3. Technology Stack — Selection and Justification

| Layer | Choice | Justification |
|---|---|---|
| Session glossary | In-memory Python dict, scoped to one `Interpreter` call instance | No persistence is required or wanted (PRD FR-18) — a dict avoids adding a cache dependency for a lookup structure this small and this short-lived. |
| Domain glossary | Curated Markdown manifest files per domain (general/medical/technical), selected by a human-supplied flag at session start | Consistent with the project's explicit, standing rejection of Calendar/Docs OAuth for context (architecture v3 §4.4) — a human-supplied flag is the cheapest correct mechanism, and it reuses the exact manifest-file pattern already proven and human-reviewed in T1.2's vocabulary manifest, rather than inventing a second content format. |
| Local STT fallback | `QuentinFuxa/WhisperLiveKit`, MLX backend | Confirmed in prior research to have a native Apple Silicon backend — a direct hit for the target M5 hardware; free either way, so activating it during safe mode costs nothing Groq wasn't already going to cost. |
| Local LLM fallback | Ollama, specific model deferred to real benchmarking | Free, local, and the natural pairing with WhisperLiveKit for a fully offline-capable fallback path; model choice is deliberately not pinned here — Stage 1's own hardware caveat (architecture v3 §11.6) already flagged that nothing on this machine has been independently benchmarked, and guessing a model now would repeat that same mistake instead of closing it. |
| Safe-mode trigger detection | A threshold-based state machine living inside `Interpreter` (`orchestrator.py`), watching Groq call latency, error rate, and queue age | Reuses the "Interpreter as the one integration point" pattern Stage 1 already established via T1.17/T1.18 — a separate parallel supervisor process would duplicate state `Interpreter` already owns and risk drifting out of sync with it. |
| Duplicate-ack suppression | A short lookback filter inside the existing Speech→Sign STT chunk stream (extends T1.7's `stt_input.py`) | Filler patterns ("okay, okay") only exist at the level of adjacent STT chunks — this is the one point in the pipeline that still sees them as separate segments, before anything downstream merges them into meaning. Filtering later would mean re-deriving segment adjacency that already exists here for free. |
| Fingerspelling asset storage | `.pose` files via `pose-format`, using T1.3's existing extraction pipeline unmodified | Checked this session: no existing ISL fingerspelling dataset is directly usable as pose/motion data. `RealSign62/RealSign-Indian-Sign-Language-Dataset`, the IEEE DataPort ISL fingerspelling set, and `ayeshatasnim-h/Indian-Sign-Language-dataset` are all static labeled images built for image-classifier training, not motion sequences a rigged avatar can play back. A 2025 continuous ISL fingerspelling corpus (Kirandevraj et al., ACL WSLP) exists and is real, but it's unaligned news-broadcast video with no extracted poses and no confirmed license for this use — not a drop-in asset either. Recording the ~26 handshapes personally and extracting them through the exact `video_to_pose` pipeline already proven in T1.3 avoids introducing a second, differently-shaped content pipeline for one small asset set. |

## 4. Component Architecture

```
                          ┌───────────────────────────────────┐
                          │  Interpreter (backend/orchestrator.py)│◄── T1.17, unchanged as the
                          │  — owns the per-segment loop for both   │    one integration point;
                          │    directions; Stage 2 hooks into it,     │    every Stage 2 piece below
                          │    it does not get a parallel owner        │    integrates through it
                          └───────────────┬─────────────────────┘
                                          │
              ┌────────────────────────────┼─────────────────────────────┐
              │                                                            │
   ┌──────────▼───────────┐                                    ┌──────────▼────────────┐
   │ Safe-Mode State Machine │  on trigger, flips routing for   │ Escalation Waterfall      │
   │ (NEW — T2.3)              │─────  both directions ────────►│ (extends T1.14)              │
   │ watches: Groq latency,     │                                │  local lookup                   │
   │ error rate, queue age       │                                │  → SESSION GLOSSARY (T2.1, NEW)   │
   └──────────┬──────────────────┘                                │  → DOMAIN GLOSSARY (T2.2, NEW)      │
              │                                                    │  → agent-directed reprocessing        │
              │ while active, routes STT/LLM calls to:              │    (still inert — Stage 3)              │
              ▼                                                     │  → LLM reasoning                          │
   ┌────────────────────────────┐                                 │  → clarify / refuse /                       │
   │ Local Fallback Stack (T2.4)   │                                │    FINGERSPELL (T2.6, NEW —                  │
   │  STT: WhisperLiveKit (MLX)      │                              │    was a dead end into UNMATCHED)              │
   │  LLM: Ollama                      │                            └──────────────────────────────────────────────┘
   └────────────────────────────┘

   ┌───────────────────────────────────┐
   │ Duplicate-Ack Suppression & Batching  │◄── sits inside T1.7's STT chunk stream
   │ (NEW — T2.5)                              │    (backend/speech_to_sign/stt_input.py),
   └───────────────────────────────────┘     before segments ever reach the waterfall above
```

Everything Stage 1 already built (recognition/classifier, pose extraction, TTS output, Kalidokit retargeting, the collision state machine, the decision log, the OBS bridge) is unchanged and not pictured again here — Stage 2 adds to the diagram in architecture v3 §2, it does not replace any part of it.

## 5. API Contracts

**Internal module boundaries** (extends Stage 1's §5, same style — no public REST surface in Stage 2 either):

- `session_glossary.resolve(term: str) -> Optional[GlossaryHit]`
- `session_glossary.record(term: str, resolution: str) -> None`
- `domain_glossary.resolve(term: str, domain: DomainContext) -> Optional[GlossaryHit]`
- `safe_mode.is_active -> bool` (property, checked at every point Stage 1 starts new avatar work or makes a Groq call)
- `safe_mode.should_enter(signal) -> bool` / `safe_mode.should_exit(signal) -> bool`
- `stt_fallback.transcribe(audio_chunk) -> str` — must return the exact same shape T1.7's Groq Whisper call already returns, so callers don't need to branch on which path served them
- `llm_fallback.complete(prompt: str) -> str` — same drop-in shape requirement, matching T1.5/T1.8's existing Groq calls
- `dedup_batch.filter(chunk_stream) -> chunk_stream` — a stream transform, not a one-shot call; must preserve chunk ordering
- `fingerspell.spell(term: str) -> PoseSequence` — returns a `.pose`-compatible sequence via `pose-format`, called from `gloss_lookup.py` only when T1.9's own lookup reports no lexicon/language-backup match

**External API touchpoints requiring in-session verification before implementation** (not independently confirmed to method-signature level in this planning session):
- WhisperLiveKit's current CLI/Python API and backend flag names (observed evolving fast in prior research).
- Ollama's current Python client/API shape.
- Whatever real signal the currently-installed Groq Python SDK actually exposes for latency/error/rate-limit state — the safe-mode trigger must read a real signal, not an assumed exception type or header name.

## 6. Data Models

New for this stage — extends, does not replace, Stage 1's `Segment` / `CoverageStatus` / `DecisionLogEntry` (`backend/contracts.py`), which are unchanged and imported as-is.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from backend.contracts import CoverageStatus  # reused, not redefined

class DomainContext(Enum):
    GENERAL = "general"
    MEDICAL = "medical"
    TECHNICAL = "technical"

@dataclass
class GlossaryHit:
    term: str
    resolution: str                 # resolved gloss text or pose reference
    source: str                     # "session" | "domain"
    coverage_status: CoverageStatus  # reuse Stage 1's enum — do not define a second one

@dataclass
class SafeModeEvent:
    timestamp: float
    triggered_by: str      # "latency" | "error" | "queue_age"
    entering: bool          # True = entering safe mode, False = exiting
```

## 7. Performance Engineering

- **Session and domain glossary lookups happen before any LLM call**, not after — this is what actually reduces Groq token usage on repeated or predictable terms, directly protecting the real 8,000 tokens/min limit Stage 1 discovered live (T1.1's amendment record), rather than a theoretical optimization.
- **Duplicate-ack suppression happens at the STT-chunk level**, before segments reach the waterfall or any LLM call — filler is filtered before it costs a token, not generated and then discarded after the fact.
- **Safe-mode trigger checks happen on every Groq call's own return path** (latency measured, errors caught), not via a separate polling loop — avoids a second timer/thread whose state could drift out of sync with what `Interpreter` actually observed.
- **Fingerspelling reuses pre-extracted `.pose` assets**, exactly like the Stage 1 vocabulary — no runtime extraction, consistent with Stage 1's decision (T1.3) to do all pose extraction offline, once.

## 8. Reliability and Bug-Prevention Strategy

- Safe mode's entry and exit must be verified against a **real induced failure** (a deliberately invalid API key for one call, or a real rate-limit hit), not a described or mocked behavior — the same discipline Stage 1's T1.13 Day-1 acceptance test already established for collision handling.
- **The fallback can itself fail** — this is a stated property, not a gap to apologize for. If WhisperLiveKit/Ollama also turn out to be too slow on the real machine (genuinely unbenchmarked as of this document — architecture v3 §11.6), safe mode's own fallback path needs a final disclosed-pause behavior rather than a second silent failure underneath the first one. Demos should remain scripted around this until the local stack is actually benchmarked, exactly as Stage 1's TRD already stated for Groq itself.
- Duplicate-ack suppression must **never operate on unclassified content** — if the filler-classifier isn't confidently sure a chunk is pure acknowledgment, it passes through unmerged. Erring toward preserving content over deduplicating it is the correct failure direction here, the same way refuse-to-fabricate errs toward declining over guessing.
- **Live-demo dependency plan, updated from Stage 1's TRD §8**: Stage 1 named Groq's uptime/rate-limits as the single most likely live-demo failure point with zero fallback. Stage 2 directly retires that specific named risk — but only partially, and only once the local fallback stack's real latency is actually measured (§7's TNFR-7 below). Until that benchmark exists, treat the risk as reduced, not eliminated, and keep demos scripted rather than fully open-ended.
- Every safe-mode transition and every glossary hit/miss must produce a real `DecisionLogEntry` through the existing decision log (T1.15) — no console-only debug logging standing in as a substitute for the log the Deaf participant and any supervisor actually see.

## 9. Non-Functional Technical Requirements

Numbering continues from Stage 1's TNFR-5.

- **TNFR-6**: Safe-mode entry/exit verified via a real induced Groq failure, not a simulated code path alone — pass/fail judged on real observed behavior, once T2.3 is complete.
- **TNFR-7**: Local fallback STT/LLM latency, measured on the actual target machine, must be recorded and disclosed before Stage 2's acceptance checklist is signed off — not assumed from the components' general reputation.
- **TNFR-8**: Session and domain glossary lookups must complete with zero Groq requests for the repeated/matched case — verified by confirming a real logged Groq call count of zero for a repeated-term test case.
- **TNFR-9**: Duplicate-ack batching verified against a real transcript containing both filler and substantive content — zero substantive tokens (names, numbers, questions) dropped, confirmed by diffing real input against real output.
- **TNFR-10**: Fingerspelled output uses only the T2.6 handshape library's own `.pose` assets and is tagged `FINGERSPELLING` — it must never silently fall back to `UNMATCHED` when a valid letter-by-letter spelling was actually possible.
