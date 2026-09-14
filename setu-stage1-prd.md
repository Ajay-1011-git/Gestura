# Product Requirements Document (PRD)
## Setu

**Document status:** Draft
**Track / context:** Inclusive Innovation: Access Without Limits — `hack '26` — Track 4: Agents for Everyday Life
**Aligned goals/standards:** Accessibility and disability-inclusion focus, consistent with the hackathon track's own framing; no formal external standard (e.g. SDG target) has been adopted for this project and none should be claimed without deliberately choosing one.

---

## 1. Problem Statement

India has an estimated 18 million Deaf and hard-of-hearing people, served by fewer than 300 certified Indian Sign Language (ISL) interpreters. Existing solutions — certified human interpreters, scheduled interpreting services — cover planned, high-stakes situations (legal proceedings, major medical appointments) but do not cover the much larger volume of everyday, unscheduled interactions: a hospital reception desk, a classroom question, a government office call. In those moments, no interpreter is available, and the Deaf or hard-of-hearing person is left to write notes, lip-read, or go without.

Setu targets specifically this everyday, unscheduled gap — not the high-stakes settings where certified human interpreters remain necessary and irreplaceable.

## 2. Goals and Objectives

| Goal ID | Description |
|---|---|
| G-1 | Enable real-time, bidirectional interpretation between ISL and spoken English inside an ordinary video call, with no dedicated interpreter present. |
| G-2 | Require no integration with, or approval from, any video-calling platform — Setu must work identically across Meet, Zoom, and Teams via a virtual camera/microphone. |
| G-3 | Behave as a genuine agent under uncertainty — able to pause, ask, refuse, or defer rather than silently guess — rather than a fixed translation pipeline with a language model bolted on. |
| G-4 | Be honest, in the product itself and in how it's described, about being a narrow supervised pilot rather than a general replacement for certified ISL interpreters. |
| G-5 | Run entirely on free-tier and self-hostable infrastructure, with no recurring cost that blocks a hackathon team or an early pilot deployment from operating it. |

## 3. Stakeholders and Target Users

| Stakeholder | Relationship to the system |
|---|---|
| Deaf or hard-of-hearing signer | Primary user; the person Setu exists to serve |
| Hearing conversation partner | Secondary user; interacts with Setu's spoken-language output and, when needed, its clarification requests |
| Host site staff (e.g. hospital desk, classroom, government office) | Deploys and supervises Setu for a specific narrow context; responsible for it remaining a supervised pilot, not an unsupervised replacement for a human interpreter |
| Deaf community / accessibility reviewers | Not a user of a single session, but a required stakeholder for any real vocabulary or gloss validation beyond the hackathon's curated demo scope (per G-4 and Risk R-3 below) |

**User stories:**

- As a **Deaf signer**, I want to walk up to an ordinary hospital desk and be understood without pre-booking an interpreter, so that I can access routine services on the same timeline as anyone else.
- As a **Deaf signer**, I want the system to ask me to repeat or clarify a sign it isn't confident about, rather than have it guess and put words in my mouth, so that I can trust what's actually being said on my behalf.
- As a **hearing conversation partner**, I want to know when Setu is uncertain or catching up, so that I don't mistakenly think the conversation has stalled or that I'm being ignored.
- As **host site staff**, I want Setu to clearly signal when it's operating outside its validated vocabulary or context, so that I know when to step in or escalate to a human interpreter instead of relying on it.
- As a **hearing conversation partner in a hurry**, I want the interaction to feel close to real-time — this is in tension with the **Deaf signer's** need for the system to slow down and ask when uncertain (G-3); Setu resolves this tension by making uncertainty visible and brief rather than either ignoring it (fast but untrustworthy) or stopping the conversation entirely (trustworthy but unusably slow).

## 4. Scope

### 4.1 In scope — Stage 1 (this phase)

- Bidirectional interpretation between a **curated, rehearsed ISL vocabulary** and spoken **English only** (Tamil and any broader vocabulary are later-phase, see 4.2).
- Real-time streaming pipeline in both directions (ISL sign recognition → fluent English speech; English speech → ISL gloss → 3D avatar rendering).
- The escalation waterfall: local lookup → session/domain glossary hooks (present but inert this phase) → LLM reasoning → clarification or refusal.
- Refuse-to-fabricate behavior: the system must never invent a sign or a translation it has no basis for.
- Rendering collision management: prevents Setu's own generated speech and the avatar's signing output from being confused with, or talking over, real participant audio.
- A live decision log, visible during the interaction, showing what the system observed, decided, and did.
- Virtual camera/microphone integration (OBS), platform-agnostic across Meet/Zoom/Teams.
- Manual override/pause control, always available to a human supervisor.

### 4.2 Out of scope

**Deferred to Stage 2 (not a rejection, a sequencing decision):** session-scope glossary, safe-mode degradation with local fallback models, duplicate-utterance batching, a room-context-selected domain glossary.

**Deferred to Stage 3 (not a rejection, a sequencing decision):** agent-directed reprocessing, collaborative sign coinage, session policy handshake, asymmetric transparency between Deaf and hearing participants, consented data export, any compositional pose generation, optional Tamil support, optional facial lip-sync.

**Permanently out of scope, evaluated and explicitly rejected — do not reintroduce in any later phase without a new decision:**
- Any official video-platform SDK or API integration (Zoom SDK, Google Meet Media API) — evaluated and rejected due to developer-preview/app-review friction incompatible with a live demo or a fast pilot.
- Broad Calendar/Docs/codebase OAuth ingestion for context.
- Automatic topic-shift detection.
- Real speaker diarization or claims about who holds the platform's speaking floor — Setu can only observe its own local audio path, never the platform's true state.
- CV-based signer emotion/feedback detection.
- Neural/photorealistic avatar rendering of any kind.
- Any claim, implicit or explicit, that Setu provides general ISL fluency, or that it is a substitute for a certified interpreter in legal or medical settings.

## 5. Functional Requirements

**Recognition (Sign → Speech input)**
- FR-1: The system shall recognize signs only from an explicitly curated, human-reviewed vocabulary list, and shall never claim recognition confidence for a sign outside that list.
- FR-2: The system shall attach a calibrated confidence value to every recognition result, not a fixed placeholder.

**Sign → Speech output**
- FR-3: The system shall reconstruct a fluent English sentence from a recognized gloss using an LLM reasoning step, not a literal word-for-word mapping.
- FR-4: The system shall stream synthesized speech output to a virtual microphone such that it is indistinguishable, to the receiving call application, from a live microphone.

**Speech → Sign output**
- FR-5: The system shall transcribe spoken English in natural chunks bounded by speech pauses, not fixed time windows or per-word calls.
- FR-6: The system shall convert transcribed speech into ISL gloss via an LLM reasoning step.
- FR-7: The system shall look up gloss terms against a curated pose lexicon and shall fingerspell, not fabricate, terms missing from that lexicon.
- FR-8: The system shall report, for every translated segment, which of a defined set of coverage categories (direct match / language-backup match / fingerspelling fallback / unmatched) applied.
- FR-9: The system shall render the resulting pose sequence on a rigged 3D avatar via real-time skeletal retargeting, not pre-rendered or neurally synthesized video.

**Agentic behavior**
- FR-10: The system shall escalate through a defined sequence (lookup → glossary hooks → LLM reasoning → clarification/refusal) rather than resolving uncertainty at a single fixed threshold.
- FR-11: On sustained low confidence, the system shall pause output to both the signer and the hearing participant and issue a clarification request, resuming automatically on a clear follow-up.
- FR-12: When no validated translation exists for a segment, the system shall visibly refuse to guess rather than produce an unmarked, confident-looking output.
- FR-13: The system shall detect when its own synthesized output could be mistaken for external speech and shall not trigger a false collision hold as a result.
- FR-14: The system shall hold new avatar output when a real overlapping utterance is detected, releasing only at a safe animation boundary, never mid-gesture.
- FR-15: The system shall log, in a human-readable form, what it observed, decided, and did for every escalation, clarification, or refusal event.
- FR-16: A human supervisor shall be able to pause or override the system at any time, with that control visibly present throughout the interaction.

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Time-to-first-audible-output for the Sign → Speech direction should target low single-digit seconds, achieved via streaming rather than waiting for full-response completion. |
| NFR-2 | Reliability | Speech chunking must be designed against the actual free-tier rate limits of the LLM/STT providers in use, not an assumed unlimited quota. |
| NFR-3 | Honesty of output | Every piece of output with a confidence or coverage value below full certainty must be presented to the user as uncertain — visibly, not just logged internally — never silently smoothed into confident-looking output. |
| NFR-4 | Honesty of output | The system must never claim, in its UI or logged behavior, to know information it structurally cannot have (e.g. which human participant is speaking on the platform's side, as opposed to what audio is present in Setu's own input). |
| NFR-5 | Accessibility | Vocabulary and translation decisions affecting the Deaf user's actual communication must be reviewed by a human before being trusted in any live interaction — this is a process requirement on the team, not only a system property. |
| NFR-6 | Compatibility | The system must function identically regardless of which video-calling application (Meet, Zoom, Teams) is in use, via the virtual camera/microphone bridge, with no platform-specific code path. |
| NFR-7 | Cost | The system must run within free-tier limits of all external APIs used, with no component requiring a paid tier to function at Stage 1 scope. |

## 7. Success Metrics

| Metric | Target |
|---|---|
| End-to-end demo scenes (normal exchange, ambiguity/clarification, refusal, collision handling) run live without manual intervention | All scenes complete successfully in sequence |
| False collision holds triggered by Setu's own synthesized output | Zero, across a full rehearsed demo run |
| Recognition attempted outside the curated vocabulary | Never silently succeeds — always resolves to fingerspelling, refusal, or clarification, never a fabricated confident answer |
| Groq free-tier rate limit breaches during a rehearsed demo run | Zero |
| Judge-facing claims made about the system | 100% consistent with the "claims to make vs. avoid" table in the technical architecture document — no overclaiming under direct questioning |

## 8. Assumptions

- A small, fixed, human-curated ISL vocabulary will be defined and rehearsed before any recognition work begins — this is treated as a prerequisite, not a parallel task.
- The builder has access to a Groq API free-tier account, an OBS installation, and the previously Blender-verified avatar asset.
- The builder has access to the specific MacBook Pro M5 referenced throughout the technical planning; no component's real-world latency on that machine has been independently benchmarked as of this document.
- Judges/evaluators will assess the system primarily via a live, scripted demo rather than unscripted open-ended use.
- No Deaf-community vocabulary validation process is available within the hackathon timeline; this is treated as a named, disclosed limitation (NFR-5, Risk R-3), not something the Stage 1 build attempts to solve.

## 9. Dependencies

| Dependency | Nature |
|---|---|
| Groq API (LLM reasoning, STT, TTS) | External service, free tier, rate-limited |
| `pose-format` (`sign-language-processing/pose`) | Open-source library — pose extraction and storage format |
| `spoken-to-signed-translation` (`sign-language-processing`) | Open-source library — gloss-to-pose lookup, fingerspelling fallback, coverage reporting |
| `fluent-pose-synthesis` (`sign-language-processing`) | Open-source library — pose sequence smoothing |
| Kalidokit + Three.js + `@pixiv/three-vrm` | Open-source libraries — avatar retargeting and rendering |
| OBS Studio + `pyvirtualcam` | External application + library — virtual camera/microphone bridge |
| The previously Blender-verified Sketchfab CC BY 4.0 avatar asset | Project-specific asset, license already confirmed |
| A human-curated ISL vocabulary list | Project-specific content, not an external dependency, but a hard prerequisite (Assumptions, §8) |

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| R-1: ISL recognition is the single highest-risk component — general ISL recognition is closer to a research problem than integration work within a hackathon timeline. | Explicitly scope Stage 1 to a small, rehearsed, curated vocabulary (FR-1) rather than attempting general recognition; state this limitation plainly to evaluators rather than let it be discovered. |
| R-2: Speech→gloss reconstruction may produce fluent-looking output that is not grammatically correct ISL, with no verification loop available in the hackathon timeline. | Disclose this limitation directly (NFR-5, Assumptions §8); do not claim linguistic correctness beyond what's actually been reviewed. |
| R-3: No Deaf-community involvement is available for vocabulary/gloss validation during the hackathon. | State this as a named, disclosed gap consistent with the project's broader honesty-about-deployability positioning, not a silently accepted risk. |
| R-4: Setu's own synthesized speech output could be mistaken by its own collision-detection logic for external speech, causing false holds. | FR-13 as a hard functional requirement, with an explicit day-1 acceptance test before further collision-management work proceeds (see the technical implementation plan). |
| R-5: Free-tier API rate limits (Groq LLM/STT) could be exceeded during a live, unscripted demo Q&A. | Chunk on natural pauses, not fixed windows or per-word calls (FR-5, NFR-2); local fallback is explicitly deferred to Stage 2, so Stage 1 demos should stay scripted rather than open-ended until that exists. |
| R-6: Hardware-specific performance claims (e.g. avatar rendering, any local processing) have not been benchmarked on the actual target machine. | Treat all such figures as unverified until tested (Assumptions §8); do not commit to demo timing that depends on an unconfirmed number. |
| R-7: Overclaiming in front of judges (e.g. implying general ISL support or platform-level speaker awareness) damages credibility more than a narrower, accurate claim would. | NFR-4 as a hard requirement; maintain the claims-to-make-vs-avoid table as a living reference checked before any public-facing description of the system. |

## 11. Acceptance Criteria

1. A real recognized sign from the curated vocabulary (FR-1, FR-2) produces a real, audible, fluent English sentence through the virtual microphone (FR-3, FR-4) — see build task T1.4–T1.6.
2. A real spoken English sentence produces real ISL gloss, a real looked-up and smoothed pose sequence, and visible avatar motion (FR-5–FR-9) — see build tasks T1.7–T1.12.
3. A deliberately out-of-vocabulary term triggers a real, visible fingerspelling fallback with a correctly reported coverage category, never a silent fabrication (FR-7, FR-8, FR-12) — see build task T1.9.
4. A deliberately low-confidence recognition triggers the full escalation and clarification sequence, pausing both sides and resuming correctly on a clean follow-up (FR-10, FR-11) — see build task T1.14.
5. Setu's own synthesized speech does not trigger a false collision hold, verified via the day-1 acceptance test (FR-13) — see build task T1.13.
6. A real overlapping utterance produces a correct, boundary-respecting hold and resume (FR-14) — see build task T1.13.
7. The decision log correctly and legibly records every event in criteria 3–6 above, in Observe/Decide/Action form, with the manual override control visibly present throughout (FR-15, FR-16) — see build task T1.15.
8. All four demo scenes (normal exchange, ambiguity/clarification, refusal, collision handling) can be run live, in sequence, without manual intervention beyond the scripted inputs.
9. No claim made in any public-facing description of the Stage 1 system contradicts the out-of-scope list (§4.2) or the claims-to-avoid table in the technical architecture document.

## 12. Glossary

- **ISL** — Indian Sign Language.
- **Gloss** — a written representation of a sign-language utterance using spoken-language words as labels, not a literal translation.
- **Fingerspelling** — spelling out a word letter-by-letter in sign, used as a fallback when no dedicated sign is known or validated.
- **Escalation waterfall** — the sequence of increasingly effortful steps (lookup → glossary → reasoning → clarification/refusal) the system passes through before acting on an uncertain input.
- **Coverage status** — a per-segment classification of how a translation was resolved: direct lexicon match, language-backup match, fingerspelling fallback, or unmatched.
- **Rendering collision** — a situation where Setu's own generated output (speech or avatar signing) could be, or is, rendered simultaneously with or mistaken for real participant input.
- **Decision log** — the live, human-readable record of what the system observed, decided, and did, shown during an interaction.
