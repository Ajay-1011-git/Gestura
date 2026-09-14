# Technical Requirements Document (TRD) — Implementation Plan
## Setu — Stage 1

**Implementation target:** hybrid application — Python backend (recognition, reasoning orchestration, meeting bridge) + browser-based Three.js frontend (avatar rendering), joining ordinary video calls as a virtual participant.

**Relationship to other documents:** this TRD explains the *why* behind Stage 1's technical decisions. It does not repeat the task-by-task execution steps — those live in `setu-stage1-build-instructions.md` (task IDs `T1.0`–`T1.16`), which this document cross-references throughout. Functional requirements referenced below (`FR-n`, `NFR-n`) are defined in `setu-stage1-prd.md`. If this document conflicts with `setu-final-architecture-v3.md`, the architecture document wins, consistent with the alignment rule already stated in the build instructions.

---

## 1. Purpose and Scope

This document specifies how Stage 1 is technically built: the stack, the reasoning behind each choice, the component architecture, the data contracts, and the performance/reliability approach. Scope is exactly Stage 1 as defined in the PRD §4.1 — curated-vocabulary, English-only, bidirectional interpretation with the Tier-0 agentic behaviors (escalation, refusal, collision management, decision log). Stage 2 and 3 technical planning exist separately, at lower detail, in `setu-stage2-plan.md` and `setu-stage3-plan.md`.

## 2. System Context

Four compute/interaction domains:

- **Capture & recognition** (Python): webcam/microphone input, pose extraction, and the curated-vocabulary sign classifier.
- **Reasoning** (external API): Groq-hosted LLMs handle both directions' gloss ⇄ sentence reconstruction.
- **Rendering** (browser/Three.js): avatar loading, bone mapping, and Kalidokit-driven retargeting.
- **Coordination** (Python): the escalation waterfall, the rendering collision manager, and the decision log sit above the other three domains and gate what each is allowed to do and when.

**The single governing architectural principle for this system**: no stage of the pipeline is permitted to silently convert uncertainty into confident-looking output. Every point where the system doesn't have solid evidence — a recognition below threshold, a gloss with no lexicon match, a possible audio collision — must be surfaced as a visible, logged decision, not absorbed silently. This is what makes Setu an agent rather than a pipeline with a language model in it (PRD G-3, NFR-3), and it is the one principle every other technical decision in this document should be checked against.

## 3. Technology Stack — Selection and Justification

| Layer | Choice | Justification |
|---|---|---|
| LLM reasoning | Groq API — `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` | Free tier with real, usable rate limits (≈30 req/min) removes the per-minute API cost risk named in the project's original constraints (PRD §9); two model sizes on one key allow fast-path/careful-path selection without adding a second vendor. |
| Speech-to-text | Groq-hosted Whisper (`whisper-large-v3-turbo`) | Same free-tier account as reasoning, avoiding a second integration surface; chunking on natural pauses (FR-5) keeps usage within the separately-tracked STT rate limit (≈20 req/min). |
| Text-to-speech | Groq TTS (`playai-tts` / `canopylabs/orpheus-v1-english`) | English-only requirement (PRD §4.1) is fully covered by Groq's free tier, avoiding a self-hosted TTS dependency for Stage 1 entirely. |
| Pose extraction & storage | `pose-format` (`sign-language-processing/pose`) | Removes the need to hand-write MediaPipe extraction and invent a bespoke storage format; its `.pose` format has both Python and JS readers, so the same files are usable by both the backend and, later, the frontend without a translation layer. |
| Gloss → pose lookup | `spoken-to-signed-translation` | Provides fingerspelling fallback (FR-7) and a four-tier coverage classification (FR-8) as existing, working behavior rather than something built from a blank page — directly serves the governing principle in §2: refusing to fabricate is a property the dependency already has, not something layered on afterward. |
| Pose smoothing | `fluent-pose-synthesis` | Purpose-built for exactly this step (concatenated discrete poses → fluent motion); avoids writing custom interpolation logic whose quality would be uncertain without dedicated tuning time. |
| Avatar retargeting | Kalidokit + Three.js + `@pixiv/three-vrm` | Mature, VTuber-ecosystem-proven kinematics solver; matches the already-Blender-verified rig's bone structure closely enough that only manual name-mapping (not a new solver) is required. |
| Meeting bridge | OBS Virtual Camera + `pyvirtualcam` | Satisfies NFR-6 (platform-agnostic) without any platform SDK approval process — the alternative (native Zoom/Meet integration) was evaluated and rejected specifically because it's incompatible with a fast, unapproved hackathon build (PRD §4.2). |
| Escalation loop orchestration | `langgraph`, or a plain state machine if `langgraph`'s current API doesn't fit cleanly | `langgraph` natively supports the branching/cycle/interrupt shape the escalation waterfall needs (FR-10, FR-11); a plain state machine is an acceptable fallback specifically because the requirement that matters (FR-15's Observe/Decide/Action log) is achievable either way — the orchestration library is a convenience, not a load-bearing decision. |

## 4. Component Architecture

```
                        ┌─────────────────────────────┐
                        │   Escalation Waterfall        │◄── governs both directions
                        │   (FR-10, FR-11, FR-12)        │
                        └───────────┬───────────────────┘
                                    │ gates output of both pipelines below
        ┌───────────────────────────┼───────────────────────────┐
        │                                                        │
┌───────▼─────────┐                                   ┌──────────▼────────┐
│ Sign → Speech     │                                   │ Speech → Sign       │
│                   │                                   │                     │
│ Webcam            │                                   │ Microphone          │
│  → pose-format     │                                   │  → Groq Whisper      │
│    extraction       │                                   │  → Groq LLM (gloss)   │
│  → curated classifier│                                   │  → spoken-to-signed-   │
│  → Groq LLM (sentence)│                                   │    translation lookup  │
│  → Groq TTS          │                                   │  → fluent-pose-synthesis│
└───────┬───────────┘                                   │  → Kalidokit retarget    │
        │                                                └──────────┬────────────┘
        ▼                                                            ▼
┌────────────────────┐                              ┌──────────────────────────┐
│ Virtual microphone   │                              │ Three.js avatar render     │
│ (pyvirtualcam)         │                              │  → Virtual webcam            │
└────────────────────┘                              │    (pyvirtualcam)               │
                                                        └──────────────────────────┘
                        ┌─────────────────────────────┐
                        │  Rendering Collision Manager   │◄── gates avatar/TTS timing
                        │  (FR-13, FR-14, §8.3 self-TTS   │    against detected audio
                        │  suppression)                    │    state
                        └─────────────────────────────┘
                        ┌─────────────────────────────┐
                        │  Decision Log (FR-15, FR-16)   │◄── observes all of the above
                        └─────────────────────────────┘
```

## 5. API Contracts

**Internal module boundaries** (the actual "API contracts" for this system, since it has no public REST surface in Stage 1):

- `recognition.classify(pose_sequence) -> Segment` — see Data Models §6.
- `waterfall.process(segment: Segment) -> Action` where `Action` is one of `TRANSLATE`, `CLARIFY`, `REFUSE`, each carrying the data needed for the receiving pipeline to act.
- `gloss_lookup.lookup(gloss_text: str) -> (pose_path: str, coverage: CoverageStatus)`.
- `collision_manager.request_hold() -> bool` / `collision_manager.release()` — called by the avatar-rendering pipeline before starting or continuing playback.

**External API touchpoints** (verify current shape before implementation — not independently confirmed to method-signature level in this session):
- Groq chat completion call (reasoning steps).
- Groq audio transcription call (STT).
- Groq TTS call (speech synthesis).

**Confirmed CLI-level contracts** (verified in prior research, safe to build against as documented):
- `spoken-to-signed-translation`: `text_to_gloss_to_pose --text <input> --glosser <simple|spacylemma|rules|nmt> --lexicon <path> --spoken-language <code> --signed-language <code> --pose <output>.pose`, plus `--coverage-info` / `--coverage-stats <file.json>`.
- `pose-format`: `video_to_pose --format mediapipe -i <input>.mp4 -o <output>.pose`.

## 6. Data Models

Canonical — copied verbatim from `setu-stage1-build-instructions.md` §B.2. This copy and that one must never diverge; if a change is needed, update both.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Direction(Enum):
    SIGN_TO_SPEECH = "sign_to_speech"
    SPEECH_TO_SIGN = "speech_to_sign"

class CoverageStatus(Enum):
    LEXICON_HIT = "lexicon_hit"
    LANGUAGE_BACKUP = "language_backup"
    FINGERSPELLING = "fingerspelling"
    UNMATCHED = "unmatched"

@dataclass
class Segment:
    id: str
    direction: Direction
    raw_input: str
    confidence: float
    coverage_status: Optional[CoverageStatus]
    timestamp: float

@dataclass
class DecisionLogEntry:
    timestamp: float
    stage: str  # "OBSERVE" | "DECIDE" | "ACTION"
    segment_id: str
    detail: str
```

## 7. Performance Engineering

- **Chunk speech on natural pauses, never per-word or fixed windows** — directly protects the Groq STT free-tier limit (NFR-2) rather than treating the quota as unlimited.
- **Extract and store the curated vocabulary's pose data once, offline, not on every recognition call** — recognition at runtime compares against pre-extracted reference data (T1.3, T1.4), not live re-extraction of reference signs.
- **Stream TTS output clause-by-clause rather than waiting for a full response** — this is what makes NFR-1's low-single-digit-second time-to-first-audible-output achievable; waiting for a complete LLM response before synthesizing speech would push this well past that target.
- **Resolve the escalation waterfall's cheapest step first** (local lookup) before any LLM call — an LLM round-trip is only paid for segments that actually need it, keeping both latency and rate-limit usage down on the common case.

## 8. Reliability and Bug-Prevention Strategy

- Type safety on all module boundaries listed in §5.
- Every external input (STT transcript, recognized gloss, Groq API response) validated at the boundary before being trusted downstream — no implicit trust in well-formed API output.
- Structured, typed error handling on every Groq API call — never a bare catch-and-continue, since a swallowed API error is indistinguishable from a real translation to anything downstream.
- **Stated limitation, not a gap to apologize for**: Stage 1 has no local fallback path. If Groq is unavailable or rate-limited, Stage 1 degrades to a hard failure, not a graceful one — this is why safe-mode degradation is explicitly Stage 2's first priority (PRD §4.2) rather than deferred indefinitely. Demos run against Stage 1 alone should stay scripted, not open-ended, until Stage 2 exists.

**Live-demo dependency plan** (the highest-value part of this section, per the project's own prior emphasis on rehearsal): the two dependencies fully outside the builder's control during a live demo are the venue's network (affecting every Groq call) and Groq's own uptime/rate-limit enforcement. Neither has a fallback in Stage 1 — this is a disclosed constraint, not an oversight, and it is the primary reason the demo script (architecture v3 §10) is scripted and rehearsed rather than improvised: a scripted demo can be rehearsed against the actual venue network in advance; an open-ended one cannot. The rendering collision manager's self-TTS suppression (§8.3 of the architecture document, FR-13) is the other dependency treated as a first-class reliability risk — it must pass its own explicit day-1 acceptance test (T1.13) before any further collision-management work proceeds, precisely because every source consulted during this project's planning independently flagged it as the most likely live-demo failure mode.

## 9. Non-Functional Technical Requirements

- **TNFR-1**: Time-to-first-audible-output (Sign → Speech direction) must be measured, not assumed, once T1.6 is complete — target is low single digits of seconds per NFR-1, achieved via streaming (§7).
- **TNFR-2**: STT call volume during any rehearsed demo run must stay under Groq's free-tier per-minute and per-day caps, verified by counting real calls during rehearsal, not estimated.
- **TNFR-3**: Zero false collision holds triggered by Setu's own TTS output, verified by the explicit day-1 acceptance test (T1.13) before the feature is considered complete.
- **TNFR-4**: The avatar must load and stand correctly in an isolated Three.js scene (T1.11) before any retargeting logic is connected — this ordering is a technical requirement, not just a suggested sequence, because it separates export/loading bugs from retargeting bugs during debugging.
- **TNFR-5**: Every `CoverageStatus` other than `LEXICON_HIT` must be visibly represented in the decision log — a fingerspelling fallback or an unmatched term must never be indistinguishable, in the log, from a confident direct match.
