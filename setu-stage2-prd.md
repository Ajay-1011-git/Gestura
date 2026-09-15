# Product Requirements Document (PRD)
## Setu — Stage 2: Resilience & Adaptive Behaviors

**Document status:** Draft
**Track / context:** Inclusive Innovation: Access Without Limits — `hack '26` — Track 4: Agents for Everyday Life
**Aligned goals/standards:** Same as Stage 1 — accessibility/disability-inclusion focus per the hackathon track's own framing; no formal external standard adopted.

---

## 1. Problem Statement

Stage 1 proved the core pipeline works: a real live run measured ~1.8s end-to-end on the Sign→Speech direction, and all 21 scripted acceptance items passed. But that same live run also proved two things architecture v3 only predicted in the abstract:

First, Setu currently has exactly one path to speech and one path to reasoning — Groq — with no fallback of any kind. The Stage 1 TRD named this as the single highest-value unmitigated live-demo risk (§8, "Live-demo dependency plan"): the venue network and Groq's own uptime/rate-limit enforcement are both fully outside the builder's control, and until now Setu had no way to survive either going wrong except a scripted, rehearsed demo. That is a real constraint on what kind of demo — and what kind of pilot — Setu can honestly be.

Second, Stage 1's live run surfaced a defect that was really a symptom of a bigger gap: Sign→Speech let a low-confidence, effectively-unmatched segment through as if it were a hedge rather than a refusal, because the four-tier coverage vocabulary (`LEXICON_HIT` / `LANGUAGE_BACKUP` / `FINGERSPELLING` / `UNMATCHED`) that both directions are supposed to share wasn't actually applied consistently. That specific bug is fixed. But it exposed that one whole tier — `FINGERSPELLING` — is currently unreachable for ISL at all: `spoken-to-signed-translation`'s bundled lexicons don't include one for ISL (`ins` is absent; `ise` is Italian), and ISL's two-handed manual alphabet has no drop-in substitute. Every out-of-vocabulary term currently either gets a lexicon/language-backup hit or falls straight to `UNMATCHED` — the fingerspelling fallback FR-7 and FR-8 already promise doesn't exist yet.

Stage 2 closes both gaps: it gives Setu a real degraded-but-functioning path when its one external dependency falters, and it gives Setu the context-aware, session-coherent, and fingerspelling-capable behavior that the "agent, not a pipeline" pitch (G-3) actually requires in a sustained conversation, not just a single scripted exchange.

## 2. Goals and Objectives

Stage 1's goals (G-1 through G-5) are unchanged and still govern the whole product. Stage 2 adds:

| Goal ID | Description |
|---|---|
| G-6 | Keep Setu functioning, in a disclosed degraded mode, when its single external dependency (Groq) is slow, erroring, or rate-limited — rather than a hard failure with no fallback. |
| G-7 | Make Setu's translation behavior context-aware (domain-appropriate vocabulary) and session-coherent (a term resolved once stays resolved for the rest of the call) — a concrete step toward the "narrow, supervised pilot for one context at a time" positioning in G-4. |
| G-8 | Make the `FINGERSPELLING` coverage tier — defined in Stage 1's data contract but unreachable for ISL in practice — a real, working path, so refuse-to-fabricate degrades gracefully to fingerspelling before it degrades to a flat refusal. |

## 3. Stakeholders and Target Users

Unchanged from Stage 1 (Deaf/hard-of-hearing signer, hearing conversation partner, host site staff, Deaf-community/accessibility reviewers) with one addition relevant to this phase specifically:

| Stakeholder | Relationship to the system |
|---|---|
| Hackathon judges / live-demo evaluators | Directly affected by Stage 2's scope — this is the phase that determines whether a real induced failure during Q&A degrades gracefully or ends the demo. |

**New user stories for this phase:**

- As a **Deaf signer**, I want a term I've already clarified once in this call to stay resolved for the rest of the conversation, so I'm not asked to repeat myself for the same word twice.
- As **host site staff** in a medical context, I want Setu to recognize domain-specific vocabulary correctly more often than a general-purpose pass would, so it's actually usable at the desk it's deployed at.
- As a **hearing conversation partner**, I want Setu to keep working, visibly in a slower or simplified mode, if its cloud connection has a bad moment — not to just stop.
- As a **hackathon judge**, I want to be able to ask a real, unscripted follow-up question without the whole system being one bad network moment away from silently failing.

## 4. Scope

### 4.1 In scope — Stage 2 (this phase)

- **Session-scope glossary**: an in-memory record of terms resolved during the current call, checked before falling through to the domain glossary or LLM reasoning; discarded at call end, no persistence.
- **Domain glossary with visible hit/miss**: a small, curated, human-selected glossary (general / medical / technical), reporting hit/miss through the same four-tier `CoverageStatus` vocabulary Stage 1 already uses — not a second status system.
- **Safe-mode degradation**: on a Groq latency spike, API error, or queue-age threshold being crossed, Setu stops starting new avatar animations, shows the latest transcript/gloss with a visibly "unrendered" label, plays a concise TTS hold message, and resumes automatically once Groq recovers.
- **Local fallback stack**: `QuentinFuxa/WhisperLiveKit` (MLX backend) as the STT fallback and Ollama as the LLM-reasoning fallback, both activated only while safe mode is active.
- **Duplicate-ack suppression and batching**: adjacent short utterances combined, repeated filler ("okay, okay") suppressed — explicitly never at the cost of dropping or compressing substantive content, names, numbers, or questions.
- **ISL fingerspelling handshape library**: a curated, personally-recorded set of ISL manual-alphabet handshapes (~26), extracted and stored via the same `pose-format` pipeline already proven in Stage 1, making the `FINGERSPELLING` coverage tier reachable for the first time.

### 4.2 Out of scope

**Deferred to Stage 3 (unchanged from Stage 1's own sequencing decision):** agent-directed reprocessing, collaborative sign coinage, session policy handshake, asymmetric transparency, consented data export, any compositional pose generation, optional Tamil support, optional facial lip-sync, shared bidirectional discourse state, the medical/legal boundary gate.

**Permanently out of scope, carried forward without exception:** any official video-platform SDK/API integration, broad Calendar/Docs/codebase OAuth, automatic topic-shift detection, real speaker diarization or platform-floor claims, CV-based signer emotion/feedback detection, neural/photorealistic avatar rendering, any claim of general ISL fluency or substitution for a certified interpreter.

**New explicit exclusion, named because Stage 1's live run surfaced it as tempting to fold in — do not:** expanding or reconciling the curated recognition/rendering vocabulary itself. Stage 1's live run found the Sign→Speech recognizer covers 40 words while Speech→Sign rendering covers 16, and that the rendering lexicon (ISLRTC, 1920×1080) and the recognition training corpus (854×480) are different sources at different resolutions — meaning a careless change to `build_lexicon()`'s data pointer could silently downgrade the avatar lexicon. This is a real, named risk (§10, R-10), but it is a **content/vocabulary problem**, not a resilience/architecture problem, and fixing it is not what Stage 2 is for. No Stage 2 task touches `data/vocab/`, `build_lexicon()`, or the lexicon resolution pipeline.

## 5. Functional Requirements

Numbering continues from Stage 1's FR-16.

**Session and domain glossary**
- FR-17: The system shall resolve a term via the session glossary, without a new LLM reasoning call, if that term was already resolved earlier in the same call.
- FR-18: The session glossary shall hold no state beyond the current call — it must be discarded, not persisted, when the call ends.
- FR-19: The system shall accept a human-supplied domain context (general / medical / technical) at session start and shall check the corresponding domain glossary before falling through to LLM reasoning.
- FR-20: A domain glossary miss shall fall through to the existing waterfall unchanged — it must never substitute a wrong-domain resolution instead of a real miss.

**Safe mode and local fallback**
- FR-21: On a Groq API error, a latency spike, or a queue-age threshold being crossed, the system shall enter safe mode: stop starting new avatar animations, display the latest transcript/gloss with a visible "unrendered" label, and play a concise TTS hold message.
- FR-22: While safe mode is active, STT and LLM-reasoning calls shall route to the local fallback stack (WhisperLiveKit, Ollama) instead of Groq.
- FR-23: The system shall exit safe mode automatically once Groq calls succeed again, without requiring a manual restart.
- FR-24: Every safe-mode entry and exit shall be logged as a decision-log event, in the same Observe/Decide/Action form as any other agentic decision (reuses FR-15).

**Batching**
- FR-25: The system shall combine adjacent short utterances and suppress repeated filler acknowledgments in the Speech→Sign STT chunk stream.
- FR-26: Batching shall never drop, merge, or compress substantive content — names, numbers, and direct questions must survive verbatim regardless of adjacent filler.

**Fingerspelling**
- FR-27: A term with no lexicon match and no language-backup match shall be spelled letter-by-letter via the ISL fingerspelling handshape library before the system falls back to `UNMATCHED`.
- FR-28: Fingerspelled output shall be tagged `FINGERSPELLING` in the coverage report — never silently folded into `LEXICON_HIT` or left as `UNMATCHED` when a real fingerspelled result exists.

## 6. Non-Functional Requirements

Numbering continues from Stage 1's NFR-7.

| ID | Category | Requirement |
|---|---|---|
| NFR-8 | Reliability | Safe mode must be entered and exited automatically, with no operator action required, verified against a real induced failure — not a described behavior. |
| NFR-9 | Honesty of output | The "unrendered" label shown during safe mode must be visibly distinguishable from normal rendered output at all times — never presented identically to a confident result. |
| NFR-10 | Cost / Compatibility | The local fallback stack (WhisperLiveKit, Ollama) must run on the same target hardware already used for Stage 1, introducing no new paid dependency. |
| NFR-11 | Performance | Session and domain glossary lookups must resolve without an LLM round trip — the entire justification for adding them is reducing Groq call volume on repeated or predictable terms. |
| NFR-12 | Reliability | Fingerspelling substitution must reuse the existing `CoverageStatus` vocabulary exactly, with no second, parallel status system introduced for it. |

## 7. Success Metrics

| Metric | Target |
|---|---|
| A real induced Groq outage/rate-limit during a scripted demo run | Safe mode triggers automatically and the demo continues in degraded mode without operator intervention, in 100% of induced-failure test runs |
| Local fallback STT/LLM latency on the actual target machine | Measured and disclosed before Stage 2 is declared demo-stable — never assumed |
| A term repeated twice within one call | Second occurrence resolved with zero additional Groq reasoning calls, confirmed by call-count logging |
| A deliberately out-of-vocabulary, out-of-domain-glossary term | Resolved via real fingerspelling and tagged `FINGERSPELLING`, in 100% of test terms — never silently folded into another status |
| A scripted utterance stream mixing filler and substantive content | Filler collapses; every name, number, and question survives unchanged, verified by diffing input against output |

## 8. Assumptions

- Stage 1's §E Final Acceptance checklist is fully green, including the one item that required a human's live eyes on a real camera run, before Stage 2 build work begins.
- `Interpreter` (`backend/orchestrator.py`, built in Stage 1 as T1.17) is the single integration point every Stage 2 component hooks into — this is now an established fact of the codebase, not a plan.
- No component of the local fallback stack (WhisperLiveKit's MLX backend, Ollama's local latency) has been independently benchmarked on the real target machine — carried forward from architecture v3 §11.6, still unresolved as of this document.
- Room/domain context is supplied by a human at session start via a simple selector, not inferred automatically — consistent with the project's already-stated rejection of automatic topic-shift detection.
- The ISL fingerspelling handshape library will be personally recorded and human-reviewed, the same way the Stage 1 vocabulary was (T1.2) — no existing ISL fingerspelling dataset was found usable as pose/motion data as-is (verified this session; see the technical implementation plan §3/§6).

## 9. Dependencies

| Dependency | Nature |
|---|---|
| Groq API | Unchanged from Stage 1 — still the primary path for LLM reasoning, STT, and TTS |
| `QuentinFuxa/WhisperLiveKit` | Open-source library — local STT fallback, MLX backend for the target Apple Silicon machine |
| Ollama | Local LLM runtime — reasoning fallback; specific model not yet chosen, pending real benchmarking |
| `sign-language-processing/pose` (`pose-format`) | Reused, unchanged — storage format for the new fingerspelling handshape library |
| `spoken-to-signed-translation`'s `CoverageStatus` vocabulary | Reused, unchanged — the domain glossary and fingerspelling both report through this same four-tier vocabulary |
| The target MacBook Pro M5 | Same hardware dependency named in Stage 1 — still not independently benchmarked for this stage's local-fallback components |

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| R-8: Local fallback models (WhisperLiveKit, Ollama) may be too slow on the real target machine to make safe mode meaningfully better than a hard failure. | Benchmark on the real machine before finalizing safe-mode timing thresholds (NFR-10); if too slow, safe mode degrades to a disclosed pause rather than a fallback-routed continuation — an acceptable fallback-of-the-fallback, stated honestly rather than hidden. |
| R-9: Batching logic could accidentally merge or drop substantive content (names, numbers, questions) along with filler. | FR-26 as a hard requirement, with an explicit test scenario that interleaves filler and substance in the same utterance stream (see build task T2.5). |
| R-10: The 40-word/16-word recognition-vs-rendering vocabulary asymmetry and the 1920×1080/854×480 resolution mismatch (surfaced during Stage 1's live run) could silently degrade avatar output if anything in Stage 2 touches `build_lexicon()` or the lexicon pointer. | Explicitly excluded from Stage 2 scope (§4.2); no Stage 2 task touches `data/vocab/`, `build_lexicon()`, or the lexicon resolution pipeline. Named here so it isn't forgotten before a dedicated content pass addresses it. |
| R-11: `WhisperLiveKit`'s CLI/flags were observed evolving fast in prior research (qwen3-streaming, causal mode, and voxtral-mlx were all found as current options in one session). | Re-verify current usage in-session before building against it, not assumed frozen since the last check (see build task T2.4). |
| R-12: The Groq model catalog is not stable — Stage 1 discovered live that `llama-3.1-8b-instant`/`llama-3.3-70b-versatile` had been replaced by the `gpt-oss` pair mid-build, requiring an explicit `reasoning_effort` parameter that wasn't previously needed. | Stage 2's Ollama fallback design must not assume Groq's current model IDs stay fixed either; re-verify at build time with the same discipline Stage 1's T1.1 amendment already established. |

## 11. Acceptance Criteria

1. A repeated term within one call resolves via the session glossary, with zero additional Groq reasoning calls for the repeat (FR-17, FR-18) — see build task T2.1.
2. A room-context-selected domain glossary reports a real hit/miss through the existing `CoverageStatus` vocabulary (FR-19, FR-20) — see build task T2.2.
3. A real induced Groq failure (a deliberately invalid call, or a real rate-limit hit) triggers safe mode automatically: new avatar animation stops, an "unrendered" label appears, a hold message plays, and STT/LLM calls route to the local fallback stack (FR-21, FR-22) — see build tasks T2.3, T2.4.
4. Safe mode exits automatically on real recovery, without manual restart (FR-23) — see build task T2.3.
5. A scripted utterance stream with interleaved filler and substantive content (a name, a number, a direct question) batches the filler and preserves the substance verbatim (FR-25, FR-26) — see build task T2.5.
6. A deliberately out-of-vocabulary, out-of-domain-glossary term is fingerspelled via the new ISL handshape library and tagged `FINGERSPELLING`, not `UNMATCHED` (FR-27, FR-28) — see build task T2.6.
7. Every event in criteria 1–6 above appears correctly in the existing decision log, with no second, parallel logging path introduced (FR-24).
8. No Stage 2 task touched `data/vocab/`, `build_lexicon()`, the lexicon resolution pipeline, `backend/recognition/classifier.py`, `backend/recognition/capture.py`, or any file outside its listed scope.
9. Local fallback (WhisperLiveKit, Ollama) latency on the real target machine is measured and disclosed before Stage 2 is declared demo-stable — not assumed (NFR-10).

## 12. Glossary

New terms for this phase (see Stage 1 PRD §12 for ISL, gloss, fingerspelling, escalation waterfall, coverage status, and decision log, all unchanged):

- **Safe mode** — a degraded operating state Setu enters automatically when Groq is slow, erroring, or rate-limited, in which STT/LLM calls route to a local fallback stack and new avatar animation is paused.
- **Local fallback stack** — the specific local, offline-capable components (WhisperLiveKit for STT, Ollama for LLM reasoning) safe mode routes to.
- **Session glossary** — an in-memory record of terms already resolved during the current call, scoped to that call only.
- **Domain glossary** — a small, curated, human-selected vocabulary (general/medical/technical) checked before falling through to LLM reasoning.
- **Fingerspelling handshape library** — the curated set of recorded ISL manual-alphabet poses used to spell out-of-vocabulary terms letter by letter.
