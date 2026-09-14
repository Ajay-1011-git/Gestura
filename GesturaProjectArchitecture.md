# Setu — Final Architecture (v3) — Repo-Grounded Build Reference
**Track:** Inclusive Innovation: Access Without Limits — `hack '26` — **Track 4: Agents for Everyday Life**

This is v3. It carries forward everything in v2 in full — nothing removed for simplicity — and adds verified, concrete repo references so implementation (including via Claude Code) needs less open-ended reasoning about *what* to build and more direct wiring of *what already exists*. Every place something is meant to be taken or referenced from an external repo is marked **REFERENCE:** or **USE:** explicitly. Where something was checked and rejected, it's marked **DO NOT USE:** with the reason.

**Scope change from v2:** Malayalam is dropped. Tamil is optional. English is mandatory. This simplifies Section 6.3 below.

---

## 0. One-Line Pitch

> Setu doesn't just translate speech and signs — it decides when to translate, when to wait, when to ask, and when to admit it can't. It's an ISL/English(/Tamil) interpretation agent that joins any video call as a virtual participant, with no platform integration and no app to install.

---

## 1. Problem & Positioning

### 1.1 The problem

India has an estimated 18 million Deaf and hard-of-hearing people, served by fewer than 300 certified ISL interpreters. Setu targets the everyday, unscheduled gap — a hospital desk, a classroom, a government office call — not the high-stakes legal/medical settings where certified human interpreters remain essential.

### 1.2 Honest positioning (unchanged — this is a strength, keep it)

Setu is a real, fundable pilot for **one narrow context at a time**, not a general replacement for India's ~300 certified interpreters. Named blockers, stated plainly:
- ISL vocabulary and dataset scarcity across the field.
- Liability in high-stakes legal/medical settings (explicitly out of scope).
- Mandatory Deaf-community involvement in vocabulary and validation, not just after-the-fact testing.
- The OBS/virtual-cam integration is a hackathon-appropriate hack, not production infrastructure.
- Ongoing per-minute API cost (mitigated but not eliminated by the free stack below).
- Biometric/video data privacy for real hospital and government interactions.

### 1.3 Novelty position — say this honestly, not more

The base concept ("AI avatar interprets sign language live in video calls") is **not novel** — it's an active, funded category (Sorenson Communications' AST proofs-of-concept, Kara Technologies' NZSL avatars, multiple VRS patents, and a peer-reviewed 2026 XR framework all do versions of this for ASL/NZSL). Do not claim to be first. What's real:

| Layer | Honest novelty | Why |
|---|---|---|
| Core concept (avatar interprets sign in live calls) | Low | Crowded, funded, patented category |
| Regional execution (ISL + English/optional Tamil, zero platform integration, free-stack build) | Meaningful | No comparable production system targets this language pair or this no-SDK architecture |
| Agentic decision layer (refuse-to-fabricate, escalation ladder, collision management, decision log) | Highest | No precedent found in Sorenson/Kara/VRS patents/academic work — those systems fix errors after the fact via human trainers; Setu declares uncertainty live |
| Novel mods (Section 5): sign coinage, agent-directed reprocessing | Highest | Genuinely unrepresented in current prior art |

**Target pitch line for judges:** *"First ISL system doing this with zero platform integration, and the first in this space framing the AI as something that can refuse, wait, and ask — rather than always producing an answer."* Defensible under a follow-up question. Do not claim more than this.

---

## 2. System Architecture — End to End

### 2.1 Sign → Speech

```
Webcam (Deaf signer)
  → Pose extraction [REFERENCE: pose-format's `video_to_pose` — see 6.4]
  → Recognition model (INCLUDE / ISLTranslate-trained)
  → [AGENT DECISION LOOP — see Section 3]
  → LLM reasoning (gloss → fluent sentence)
  → Streaming TTS (Groq PlayAI/Orpheus for English; AI4Bharat indic-parler-tts self-hosted, if Tamil pursued)
  → Virtual microphone (OBS)
  → Any video call app
```

### 2.2 Speech → Sign

```
Microphone (hearing speaker)
  → STT (Groq Whisper large-v3-turbo; WhisperLiveKit local as safe-mode fallback — see 6.2)
  → [AGENT DECISION LOOP — see Section 3]
  → LLM reasoning (sentence → ISL gloss)
  → Gloss → pose lookup [REFERENCE: spoken-to-signed-translation's lookup +
     fingerspelling-fallback + coverage-report pattern — see 6.8. STOP at the
     .pose output — do NOT use its optional pose-to-video neural rendering step,
     see DO NOT USE note in 6.8]
  → Pose smoothing [REFERENCE: fluent-pose-synthesis — see 6.8]
  → Kalidokit retargeting
  → Rigged 3D avatar (Three.js)
  → Optional: lip-sync/facial layer [REFERENCE: TalkingHead — see 6.5, conditional]
  → Virtual webcam (OBS)
  → Any video call app
```

### 2.3 Coordination layer (unchanged from v2 — revision to original Decision 7)

The original decision log describes these as "two independent, always-on loops." **This is revised**: each direction still runs continuously and chunks on natural pauses, but a **rendering collision manager** (Section 8) sits above both, coordinating when each side's output is allowed to render. They are independent in that neither blocks on the other's full completion — they are *not* independent in that they no longer render blindly over each other.

---

## 3. The Core Agent Decision Loop (Escalation Waterfall)

This is the architectural centerpiece — not a feature list, the actual decision structure every segment passes through. Both directions run the same shape of loop.

```
1. Local pose/gloss lookup
        ↓ miss or low confidence
2. Session glossary (in-memory, this call only)
        ↓ miss
3. Domain glossary (small, validated, loaded from room context hint)
        ↓ ambiguous or still low confidence
4. Agent-directed reprocessing  ← NOVEL MOD, see 5.1
   (re-run recognition on the buffered landmark window with a
   relaxed threshold, before giving up)
        ↓ still uncertain
5. LLM reasoning (propose 1-2 plausible candidates)
        ↓ no defensible candidate exists
6. Clarification (escalation ladder — see 4.1) or
   Refuse-to-fabricate + fallback (fingerspell/text) or
   Sign coinage offer if speech→sign and no validated pose exists at all
```

**REFERENCE for step 6's status classification**: `spoken-to-signed-translation`'s `--coverage-info` flag already implements a four-way classification per token — lexicon hit / language-backup hit / fingerspelling fallback / unmatched — color-coded in its CLI output. That four-way split maps almost directly onto this waterfall's own escalation levels. Use it as the concrete schema for what "confidence tier" means at each step, rather than inventing a new one.

**REFERENCE for implementing this loop itself**: `langchain-ai/langgraph` natively supports branching, cycles, persistence, and human-in-the-loop interrupts — the exact shape this waterfall needs. Build the loop as a LangGraph graph rather than a hand-rolled state machine; check current API syntax against their docs at implementation time, since the framework moves fast.

The agent's real decision at every step is **whether to escalate further or stop and act** — cheap when confident, expensive only when it has to be. This is the answer to "why is this agentic and not just a pipeline with an LLM in it."

---

## 4. Agentic Behaviors — Final Build List

### 4.1 Tier 0 — Must build, this is the demo

**Escalating clarification ladder** (multi-signal, with hysteresis)
- Signals combined, not a single threshold: recognition confidence, top-k candidate disagreement, glossary presence, hand/landmark visibility quality.
- Hysteresis: require **two** consecutive low-confidence segments to trigger; **one** clean segment or an explicit repeat to resume. Cooldown so one bad frame can't cause repeated interruptions.
- Escalation, not a binary stall:
  1. First trigger → stall **both sides at once** ("one moment, clarifying a term" to the hearing side; "did you mean X or Y?" to the signer, if two defensible candidates exist — otherwise a generic repeat request).
  2. Second trigger on the same segment → show candidate text/fingerspelling instead of re-asking the same way.
  3. Final fallback → proceed with a visible "uncertain: [term]" label rather than pretending certainty.

**Refuse-to-fabricate on unsupported signs**
- If the pose library has no validated sequence for a required concept, do not invent an animation. Fall back to fingerspelling/text (this is a **built-in default behavior** in `spoken-to-signed-translation` — see 6.8, don't reimplement), or trigger sign coinage (5.2) if in a signer-facing context. Log the refusal explicitly — this is the single strongest agentic moment in the whole project, don't bury it.

**Rendering collision management** (renamed from "floor management" — it does not know who has the platform's speaking floor, only what's in its own audio path)
- Full spec in Section 8. Reuses the existing FIFO queue and TTS/mic path. Never touches real participant mics.

### 4.2 Tier 1 — Build if Tier 0 is stable

- **Session-scope glossary**: plain in-memory dict, resolved terms reused for the rest of the call, discarded after. No persistence, no cross-session memory.
- **Safe-mode degradation**: on latency spike, API error, or queue age crossing a threshold — stop starting new avatar animations, show the latest transcript/gloss with an "unrendered" label, use a concise TTS hold message, resume when recovered. **This is not optional decoration** — see Section 11 on why the free-tier rate limits make this a real requirement, not a nice-to-have. **REFERENCE**: the local fallback path now has concrete tooling — see 6.2/6.7, WhisperLiveKit (MLX) + Ollama.
- **Duplicate-ack suppression + batching**: combine adjacent short utterances, suppress repeated "okay, okay" filler. Explicitly **not** full summarization — never compress or drop substantive content, names, numbers, or questions.

### 4.3 Tier 2 — Nice to have / novelty mods

See Section 5 in full. Summary:
- Agent-directed reprocessing (architectural, highest novelty value)
- Collaborative sign coinage (UX, highest novelty value)
- Consent-based policy handshake at session start
- Asymmetric transparency (decision log detail to Deaf user, simplified version to hearing side)
- Consented, local-only data export/flywheel
- Start-of-room context hint (topic/pasted agenda — not Calendar/Docs OAuth)
- Boundary gate for medical/legal advice ("I can help with logistics, not medical/legal advice")
- Shared bidirectional discourse state (build only if time remains — hard to demo cleanly even though it's not hard to build)
- Compositional pose primitives — narrow, additive only (Section 5.3)
- Optional lip-sync/facial expressiveness layer — see 6.5, conditional on blend-shape check

### 4.4 Cut — do not attempt in hackathon scope

Consistent across every source consulted (three independent brainstorm docs, Gemini's suggestions after correction, our own review, and the repo research session):
- True speaker diarization / platform-level "who has the floor"
- CV-based signer feedback detection (head shakes, frustration, etc. — use explicit buttons instead)
- Broad Calendar/Docs/codebase OAuth ingestion
- Automatic topic-shift detection (use a manual context switch instead)
- Live automatic English/Tamil code-switching detection (choose target language at room start)
- Cross-session personalization / persistent user memory
- **Neural/photoreal avatar rendering** — confirmed again this round: `spoken-to-signed-translation`'s optional `pose-to-video` step uses pix2pix + an upscaler for exactly this, and it will not run real-time on the M5. Stop the pipeline at the `.pose` output (see 2.2, 6.8) and use the existing Kalidokit/Three.js renderer instead.
- General compositional sign generation (only the narrow, hard-coded version — see 5.3)

---

## 5. Novel Architecture & UX Mods

Ranked by actual novelty contribution and checked against feasibility. None of these replace a working component — all are additive to the pipeline in Section 2.

### 5.1 Agent-directed reprocessing (headline architectural mod)

**What it is:** the reasoning step gets a real tool-call back into perception. On low confidence, before escalating to human clarification, it can request a second pass — re-run recognition on the already-buffered landmark window with a relaxed confidence threshold or a longer temporal window.

**Why it's the most novel single piece in the project:** every incumbent system checked (Sorenson, Kara, VRS patents, academic frameworks) is a fixed pipeline — recognize once, translate, and fix errors later via human review logs. None of them have the model actively directing its own sensing in the moment. This is genuine tool-use in service of resolving the agent's own uncertainty.

**Build this way, not the other way:** implement the **scoped version only** — re-query the recognition model against data already computed and buffered (cheap: a short rolling window of raw landmark frames, exposed as a callable function via structured tool-calling, which the Groq-hosted models support natively). **Do not** attempt the full version (re-cropping raw video to a tighter hand region and re-running detection) — that's 1–2 days of real CV work for a path that only fires on the already-rare low-confidence branch. The scoped version is 4–8 hours and tells the identical story to a judge.

**Slots into the waterfall** as step 4 in Section 3 — additive, conditional, doesn't touch the fast default path.

### 5.2 Collaborative sign coinage (headline UX mod)

**What it is:** when the escalation waterfall bottoms out with no validated pose and refusal is the only option, don't just fall back to fingerspelling and stop — invite the Deaf user to construct a sign with Setu right there. Real sign languages gain vocabulary this way, through classifier constructions and compounds coined in the moment. Setu proposes a candidate (or offers a blank), the user confirms or edits via the existing webcam→pose-extraction→Kalidokit capture path (same infrastructure as the live pipeline, not new), and it's saved into the session glossary.

**REFERENCE**: save the coined sign as a `.pose` file using the `pose-format` toolkit (see 6.4) — it's the same format the rest of the pipeline already reads and writes, so the session glossary doesn't need its own bespoke storage.

**Why it matters:** every incumbent treats sign vocabulary as a closed, company-maintained asset corrected after the fact by trainers. Setu extending it collaboratively, live, with the person it's actually for, is a different relationship to the vocabulary gap — not a bigger version of refusal, a different response to it.

**Feasibility:** medium-high, ~3–6 hours, conditional on the recording pipeline already being stable (it should be, since it reuses the sign→speech capture path).

### 5.3 Other mods, in priority order

| Mod | What it does | Feasibility | Effort |
|---|---|---|---|
| Policy handshake | One question at session start ("cautious or casual today?") calibrates the escalation ladder's thresholds for that session | Very high | 1–2 hrs |
| Asymmetric transparency | Full decision log to the Deaf participant; short plain-language version ("one moment, clarifying") to the hearing side | High | ~half day |
| Data flywheel / export | Confirmed corrections + coined signs exportable via explicit opt-in toggle; default retention off | Very high | 1–2 hrs |
| Shared bidirectional discourse state | One rolling context object both directions read/write, so references resolve across sides | High to build, low demo-visibility | ~4–6 hrs |
| Compositional pose primitives | **Narrow only**: hard-code exactly one modifier (e.g. movement amplitude for "big/small") on exactly one library entry, as a single demo beat — not a general system | Low-medium for general version; high for the single hard-coded instance | 1–2 days general / a few hrs for one instance |

**Explicitly rejected:** a parallel race between a fast heuristic lookup and the LLM reasoning call (instead of the sequential waterfall). This is a latency optimization dressed as architecture — adds race-condition risk and burns free-tier rate-limit budget twice as fast for a win that isn't really about novelty. Skip it.

---

## 6. Technology Stack — Entirely Free, Now with Explicit Repo References

### 6.1 LLM reasoning — Groq API (free tier, no credit card)

Unchanged from v2.

- Models available free: **Llama 3.1 8B Instant** (fast/low-latency), **Llama 3.3 70B Versatile** (higher quality), Llama 4 Scout, Qwen3 32B, gpt-oss-20b/120b.
- Free tier: rate-limited, not credit-limited — roughly **30 requests/minute**, up to **~14,400 requests/day**, with per-model token caps (e.g. ~6,000–12,000 tokens/minute depending on model size).
- **Bonus architectural opportunity**: since Groq gives real latency/quality tiers on one key for free, the previously-deferred "FAST vs CAREFUL" adaptive policy idea is cheap to justify — Llama 3.1 8B for casual chit-chat, Llama 3.3 70B for ambiguous/domain-heavy segments.

### 6.2 Speech-to-Text — Groq Whisper primary, WhisperLiveKit for local fallback

**DO NOT build on `ufal/whisper_streaming`** — its own README now points to `QuentinFuxa/WhisperLiveKit` as the actively maintained successor, built by the same contributor. It's effectively superseded.

- **Primary**: Groq Whisper (large-v3-turbo). Free-tier limits: roughly **20 requests/minute, 2,000/day**, 25MB file cap. **Design constraint**: chunk on natural speech pauses (per original Decision 7) — per-word calls will burn the daily cap fast.
- **USE for local safe-mode fallback (updated from v2's generic "Ollama" mention)**: `QuentinFuxa/WhisperLiveKit` — self-hosted, built-in VAD, multi-user, and specifically has a **native MLX backend for Apple Silicon**, a direct match for the MacBook Pro M5. This replaces the vague "small local model" language from v2 with a concrete, purpose-built tool. Verify actual latency on your specific M5 unit before relying on it in the demo (not independently benchmarked here).

### 6.3 Text-to-Speech — simplified: Tamil optional, no Malayalam

- **English (mandatory)**: Groq TTS — PlayAI (`playai-tts`) or Canopy Labs' Orpheus (`canopylabs/orpheus-v1-english`), same free-tier key. Supports English and Arabic.
- **Tamil (optional, only if pursued)**: **USE `ai4bharat/indic-parler-tts`**, self-hosted, Apache 2.0. Confirmed to cover **21 languages including both Tamil and English in the same model** — no reference audio needed, voice described in natural language (age/gender/style). Self-host via MPS on the M5.
  - **DO NOT use `AI4Bharat/IndicF5` for this**: confirmed it covers Tamil but **not English**, and it's reference-audio-based (voice cloning) — needs a per-utterance reference clip plus real consent/licensing handling for whatever voice is cloned. More setup for no benefit here.
  - **DO NOT use `AI4Bharat/Indic-TTS`** (the older project) as a dependency — it's a training/config repo, not a ready inference package.
- **Optional lip-sync layer, conditional**: see 6.5.

### 6.4 Recognition & pose extraction

- **Datasets**: INCLUDE (isolated-sign) and ISLTranslate (continuous, ships with pre-extracted MediaPipe poses). Unchanged.
- **USE for pose extraction and storage, instead of hand-rolling MediaPipe boilerplate**: `sign-language-processing/pose` (the `pose-format` PyPI package). Confirmed real and maintained (112 stars, MIT, pushed July 2026). Provides:
  - A `.pose` file format with **both Python and JavaScript readers/writers** — meaning pose data can be read browser-side too, not just in a Python backend.
  - A ready CLI: `video_to_pose --format mediapipe -i example.mp4 -o example.pose`, plus a batch `videos_to_poses` variant.
  - **Use `.pose` as the storage format for the entire validated sign library** (Decision 5's "recorded pose-sequence lookup"), not a bespoke format — it's the same format the coinage flow (5.2) and the gloss lookup (6.8) already produce and consume.
- **Bonus, not in the original repo list, found during research**: `sign-language-processing/mediapipe-hand-crop-fix` — addresses a known, real MediaPipe Holistic hand-detection accuracy problem that directly affects downstream ISL recognition quality. Worth a look specifically because hand-crop accuracy is a likely source of recognition errors in this exact pipeline.
- **Individual ISL baseline repos** (`Sooryak12/Indian-Sign-Language-Recognition`, `Exploration-Lab/ISLTranslate`, `Swaroop-Srisailam/Continuous-Indian-Sign-Language-Recognition`, `aju22/Real-Time-ISL-Translation`): plausible starting points per their own descriptions, but **not independently verified to the same depth** as the org-level tools above (these are individual repos, not an active maintained org). Spot-check last-commit date and open issues before betting build time on any one of them — this matters more than usual given recognition is the project's single highest-risk component (Section 11.1).
- **`sign-language-processing/detection-train`** (signing/non-signing gate): real, but the org's own listing tags it **"(OLD)"** — treat as a reference for the concept, not a current dependency without checking its state yourself.
- **Could not verify**: "sign/translate" and "sign/inference" as pasted did not resolve to any real repo. Most likely a mis-copied reference to the broader `sign-language-processing` org already covered here — don't build against them as if they're specific real dependencies.

### 6.5 Avatar & retargeting

- **Character**: the already-verified Sketchfab CC BY 4.0 rig — Mixamo/Unity-style bone naming, finger-level bones confirmed in Blender. See Section 7 for the export checklist. Unchanged.
- **Retargeting**: Kalidokit + three-vrm, same solver drives both the live webcam feed (sign→speech direction) and replayed pre-recorded pose sequences (speech→sign direction). Unchanged, confirmed still current.
- **REFERENCE (pattern only, not a drop-in)**: `europanite/webcam_to_avatar` demonstrates the webcam→MediaPipe→Kalidokit→avatar wiring end to end, but it's built around a native `.vrm` file with built-in bone-mapping metadata. **Our asset is a plain glTF with Mixamo-style names, not a `.vrm`** — this repo shows the pattern, it does not hand you a working loader. The manual bone-mapping approach already planned (following the SysMocap precedent, per original Decision 5) still stands and is still necessary.
- **Optional, conditional — lip-sync/facial polish**: `met4citizen/TalkingHead`. Real, actively maintained, MIT-licensed, and — notably — built specifically for **Mixamo-compatible rigs**, which matches our asset better than most Kalidokit demos (which lean VRM). **Before spending any time on this: open the mesh in Blender and confirm it has ARKit and Oculus viseme blend shapes** — TalkingHead's lip-sync requires that specific naming convention, and "has facial shape keys" (per the original resource table) is not the same guarantee. If it matches: this is legitimate demo polish (mouth moves when the avatar speaks), layered on top of Kalidokit, never a replacement for it — Kalidokit still does all hand/finger retargeting. If it doesn't match, skip without regret; it was never load-bearing.
- **Unverified bonus, flagged not vetted**: `met4citizen/HeadTTS` — a free, local (browser or Node), Kokoro-based TTS with viseme timing output. Could complement the lip-sync layer above if pursued. Not independently checked beyond a surface mention — verify before relying on it.

### 6.6 Meeting integration

- OBS Virtual Camera / v4l2loopback for the virtual mic+cam bridge, `letmaik/pyvirtualcam` for sending frames to it from Python, `obsproject/obs-websocket` + `aatikturk/obsws-python` if programmatic OBS scene/source control is needed. All standard, low-risk, confirmed still current. No SDK approval, no platform enrollment, works identically across Meet, Zoom, Teams. Unchanged from original Decision 2.

### 6.7 Local fallback / safe mode

- **STT fallback**: WhisperLiveKit via MLX (see 6.2) — updated from v2's vague mention.
- **LLM fallback**: a small local model via **Ollama** on the MacBook Pro M5, triggered by the same safe-mode thresholds (queue age, API error, 429 response) described in 4.2. Active fallback, not passive insurance — a direct response to the real free-tier rate limits, not a hypothetical.
- **TTS fallback, if Tamil is pursued**: `indic-parler-tts` self-hosted is already the primary Tamil path (6.3), so no separate fallback needed there.

### 6.8 Gloss → Pose Pipeline — the biggest work-reduction find this round

**USE**: `sign-language-processing/spoken-to-signed-translation` (mirror of `ZurichNLP/spoken-to-signed-translation`; MIT, 98 stars, pushed July 2026) as the architectural basis for the entire text→gloss→pose-lookup stage of the speech→sign direction. Concretely, it already gives you, working:

- Multiple glosser backends (`simple`, `spacylemma`, `rules`, `nmt`) for text→gloss.
- Lexicon-based lookup converting glosses into pose sequences.
- **Fingerspelling fallback for lexicon misses, on by default** — this is the refuse-to-fabricate fallback from 4.1, already implemented, not something to build from scratch.
- **`--coverage-info` / `--coverage-stats`**: a per-token status report (lexicon hit / language-backup hit / fingerspelling fallback / unmatched) — use this directly as the confidence/status signal for the escalation waterfall (Section 3).

**What you still have to bring**: your own curated ISL lexicon entries (the pose library is not ISL by default — confirmed, it ships with Swiss/German sign language lexicons). The architecture is reusable; the vocabulary is yours to build regardless, per Section 11.1.

**Then**: run the resulting pose sequence through `sign-language-processing/fluent-pose-synthesis` (13 stars, MIT, pushed July 2026, confirmed real) to smooth the concatenated poses before handing off to Kalidokit.

**DO NOT USE**: the pipeline's optional final step, `text_to_gloss_to_pose_to_video`, which depends on the separate `pose-to-video` package using **pix2pix + an upscaler** for neural photorealistic video rendering. This is GPU-heavy, not real-time, directly contradicts the original Decision 3 (rejected neural avatar rendering in favor of a rigged 3D skeletal avatar), and will not run real-time on the M5. **Stop at the `.pose` file output** (`text_to_gloss_to_pose`) and feed that directly into the existing Kalidokit/Three.js renderer instead. This is a correction from the prior research-session turn, not a new finding — flagging it here so it's impossible to miss when building.

**Also confirmed present in the same org, correctly deprioritized**: the `signwriting-*` cluster (`signwriting-translation`, `signwriting-animation`, `signwriting-clip`, `signwriting-evaluation`) is a parallel representation system (SignWriting notation), architecturally distinct from the gloss+pose-lookup approach above. `signwriting-animation` is additionally marked **archived** (unmaintained) on GitHub. Don't build on any of these for Setu's core pipeline.

---

## 7. Avatar Pipeline — Blender-to-Three.js Export Checklist

Unchanged from v2. Applies to the already-verified Sketchfab rig (Decision 5). No new asset needed — this is export mechanics, not a re-verification.

1. **Export as glTF Binary (`.glb`)**, not separate `.gltf` + textures. One file, embedded textures, nothing to leave behind when moving into the project.
2. **Rest pose, not a posed frame.** Confirm the armature export option is set to Rest Position, not Pose Position. Get this wrong and the model looks completely normal standing still, then twists or "swims" the moment Kalidokit starts driving it — the most likely silent failure.
3. **Confirm T-pose vs A-pose** on the armature (arms horizontal = T-pose, ~45° down = A-pose). Given Mixamo-style naming, it's likely T-pose by default, but confirm — Kalidokit's retargeting math assumes a specific reference pose.
4. **Apply all transforms** (`Ctrl+A` → All Transforms) on both armature and mesh before export, so scale/rotation isn't hiding in the object transform.
5. **Isolated load test before wiring into Kalidokit.** Load the raw `.glb` into a bare-minimum Three.js scene — loader and camera only, nothing else — and confirm it stands upright and faces the right way. This separates export problems from retargeting problems; debugging both at once burns hours you don't have.
6. **If pursuing the optional TalkingHead lip-sync layer (6.5)**: while in Blender for the checklist above, also check the mesh's shape keys for ARKit/Oculus viseme naming. Same file, same session, no extra round trip.

---

## 8. Rendering Collision Management — Full Spec

Unchanged from v2. Renamed deliberately from "floor management": Setu detects overlap in **its own local audio path**, not the platform's true speaking-floor state. Never claim otherwise to judges.

### 8.1 State machine

```
IDLE → SIGNING (queue starts playback)
SIGNING → DRAINING (no new items arrive, current clip finishes)
DRAINING → IDLE (playback reaches a safe boundary)
```

Parallel audio state:
```
SILENCE → VOICE_DETECTED → SPEAKING → SILENCE
```

### 8.2 Trigger logic

- Hold fires only when **both** are true: avatar has active, uncompleted output, AND new speech persists past a debounce window (~250–400ms sustained voice activity, confirmed by an STT partial where possible — not a single VAD frame, to avoid coughs/clicks/keyboard noise).
- Release on a **safe clip boundary**, never mid-gesture — never interrupt a sign mid-handshape, only hold *new* incoming work.
- **Max hold timeout**: after N seconds, don't hold indefinitely — surface the backlog or ask permission to continue.
- **Priority surfacing**: if held speech contains a name, number, direct question, or safety-relevant term, don't silently queue it for long — surface sooner than routine content.

### 8.3 The critical failure mode — self-TTS suppression

Every source consulted independently flagged this as the most likely live-demo failure: Setu's own TTS output gets picked up by its own VAD, and the system thinks the hearing participant is speaking (false overlap).

**Fix**: maintain an explicit `SETU_TTS_ACTIVE` flag. While true, ignore or explicitly mark overlap events as local output, not external speech. Do not attempt real-time source separation or speaker diarization to solve this — unnecessary scope for a problem that a boolean flag solves.

### 8.4 Day-1 acceptance test

Before investing further build time in this feature, verify: can remote participant audio reach Setu's STT, and can Setu's generated TTS reach the meeting, without Setu's own TTS being mistaken for remote speech? If yes, proceed. If the audio routing itself is unreliable, do not spend hackathon time inventing more sophisticated floor detection on top of a broken foundation — fix routing first, or fall back to a controlled/rehearsed audio setup for the demo and say so honestly.

**Optional reference**: `sign-language-processing/segmentation` (sentence/sign-level pose segmentation) could inform natural chunk boundaries on the *signing* side, mirroring what VAD does for speech — not required, but worth a look if this layer needs refinement.

---

## 9. Decision Log / Transparency UI

Unchanged from v2, implementation note added. A live panel next to the latency/queue dashboard, structured as **Observe → Decide → Action**, not raw model output:

```
11:42:01  OBSERVE  ISL candidate: BANK, confidence: 0.41
11:42:01  DECIDE   candidate disagreement: HIGH, context ambiguity: HIGH
11:42:01  ACTION   translation paused, clarification requested
11:42:04  OBSERVE  clarification received
11:42:04  ACTION   translation resumed
```

**REFERENCE**: if building this loop with `langgraph` (see Section 3), its graph execution naturally produces this Observe/Decide/Action trace — don't build separate logging on top, surface the graph's own state transitions.

Show structured decision factors, never raw chain-of-thought. Per Section 5.3's asymmetric transparency mod: full detail to the Deaf participant, a short plain-language line to the hearing side. Keep a visible manual override/pause control at all times — this demonstrates that autonomy is bounded, which reads as a strength to judges, not a limitation.

---

## 10. Demo Script (~3 minutes)

Unchanged from v2.

1. **0:00–0:30 — Normal exchange.** Ordinary sentence both directions. Log shows `[confidence: 0.9+] → translate`. Move fast, don't linger.
2. **0:30–1:00 — Ambiguity → clarification.** Deliberately trigger low confidence. Setu stalls both sides, asks a targeted two-choice question, resumes on answer.
3. **1:00–1:25 — Glossary hit.** Pre-loaded room context ("medical desk") resolves a domain term through the waterfall; log shows which step resolved it.
4. **1:25–1:50 — Unsupported sign, refuse to fabricate.** No validated pose exists. Log shows the refusal and fallback (or the sign coinage flow, if stable). This is the strongest 25 seconds — don't rush it.
5. **1:50–2:25 — Rendering collision.** Someone talks while the avatar is mid-sign. Setu announces a hold, queues, catches up on completion. State indicator visible throughout.
6. **2:25–2:45 — Safe mode** *(only if confident it's stable)*. Inject a simulated delay/fault; Setu visibly drops to the local WhisperLiveKit/Ollama fallback rather than pretending nothing happened.
7. **Close** on the pitch line from Section 0.

Keep the manual override visible the entire time.

---

## 11. Risks & Honest Feasibility

No minimizing — this section exists to protect the project, not flatter it. Unchanged from v2, with one addition at the end.

### 11.1 The real constraint: ISL recognition

This is the load-bearing risk of the entire project, and no amount of agentic framing on top changes that. Getting a recognizer that works live and robustly across lighting/camera angle/signer variation, in hackathon time, is closer to a research problem than integration work. **Realistic outcome: a curated, rehearsed vocabulary** — signs personally trained and repeatedly tested — not general ISL recognition. State this plainly to judges rather than let it be discovered: *"recognition is a validated, curated vocabulary, not general ISL."*

**The single highest-leverage protective decision for the whole project: fix the exact curated vocabulary before building anything else**, since this determines whether the live demo works at all — more than any agentic feature or any repo integration does.

### 11.2 Gloss quality has no verification loop

Speech→ISL gloss via LLM reasoning will produce fluent-*looking* output. Whether it's grammatically correct ISL requires an ISL-fluent reviewer, which the original Decision 9 already names as a real requirement. Absent that review before demo day, what's actually being shown is "output that looks right to a hearing engineer" — a materially weaker claim. Say this upfront in Q&A rather than hope it doesn't come up.

### 11.3 What's genuinely low-risk

STT (Groq Whisper), LLM reasoning (Groq Llama), English TTS (Groq PlayAI/Orpheus), the OBS virtual-cam bridge — all mature, well-documented, low integration risk. Avatar retargeting (Kalidokit + three-vrm + the verified rig) is real work but **de-risked** work — already verified in Blender, following a proven precedent (SysMocap). This is the part of the project most likely to work exactly as designed. The `pose-format` and `spoken-to-signed-translation` integrations (6.4, 6.8) further de-risk the gloss/pose-lookup layer specifically, since that logic no longer needs to be built from a blank page.

### 11.4 Historically underestimated

Streaming end-to-end under 2 seconds is achievable but needs careful chunking — sloppy chunking will hit the real Groq free-tier rate limits mid-demo, not just in theory. Rendering collision management (Section 8) is the piece **every independent source flagged** as the most likely live-demo failure — budget real rehearsal time in the actual venue with actual room acoustics, not an afternoon in a quiet room.

### 11.5 Tamil, if pursued, is optional scope — treat it that way

Since Tamil is optional (not required, unlike v2's Malayalam which was positioned as a committed deliverable), it's now legitimate to scope it as a stretch goal with no cost to the core pitch if it's cut. Self-hosting `indic-parler-tts` on the M5 via MPS is plausible (see 6.3) but is still a separate dependency/install/latency-tuning surface. If time is tight, dropping it entirely is a clean, low-cost decision — unlike v2, where Malayalam being a named requirement made cutting it feel like a concession. It no longer is one.

### 11.6 Hardware-specific claims not independently verified

Flagged honestly, per the repo-research session: the "sub-second latency on MPS" figure for `indic-parler-tts`, and any implicit assumption that WhisperLiveKit's MLX backend or Ollama's local models run comfortably on this specific MacBook Pro M5 configuration, are **not independently benchmarked in this document**. Run a standalone timing test on the actual machine before building demo timing around any of them.

---

## 12. Build Priority & Time Budget

Unchanged in structure from v2, with repo integrations slotted into the relevant steps.

```
P0 — must work, everything else depends on this
  1. Fix the curated sign vocabulary (do this FIRST, before any code)
  2. Both translation directions reliable, on the curated vocabulary
     [use pose-format's video_to_pose for extraction — 6.4]
  3. Streaming reliable end to end
  4. Escalation ladder → clarification (4.1)
     [use spoken-to-signed-translation's coverage-info schema — 6.8]
  5. Refuse-to-fabricate (4.1)
     [use spoken-to-signed-translation's fingerspelling fallback — 6.8]
  6. Rendering collision management + self-TTS suppression (Section 8)
  7. Decision log panel (Section 9)
     [consider langgraph for the loop itself — Section 3]
  8. Avatar export + isolated load test (Section 7)

P1 — makes it feel agentic, build if P0 is stable
  9. Session-scope glossary
  10. Safe-mode degradation + WhisperLiveKit (MLX) + Ollama local fallback (6.2, 6.7)
  11. Duplicate-ack suppression / batching
  12. Domain glossary with visible hit/miss

P2 — the novelty layer, build only once P0/P1 are demo-stable
  13. Agent-directed reprocessing, scoped version (5.1)
  14. Collaborative sign coinage (5.2) — store as .pose via pose-format
  15. Policy handshake, asymmetric transparency, data export (5.3 — cheap, ~half day combined)
  16. One hard-coded compositional pose modifier, as a single demo beat only

Only if P0/P1/P2 are all stable with time to spare
  17. Shared bidirectional discourse state
  18. Tamil path via indic-parler-tts (optional, no longer core — see 11.5)
  19. Boundary gate for medical/legal advice
  20. Optional lip-sync layer via TalkingHead, conditional on blend-shape check (6.5)
```

---

## 13. Judge Q&A — Claims to Make vs. Avoid

Unchanged from v2.

| Don't say | Say instead |
|---|---|
| "Setu knows who is speaking." | "Setu detects conversational overlap from the audio available to it and coordinates its own output." |
| "Setu understands the entire meeting." | "Setu maintains a lightweight session context relevant to the current interaction." |
| "Setu can translate any ISL sentence." | "Setu is a supervised pilot constrained by its validated recognition and pose vocabulary." |
| "There are no ISL datasets." | "ISL resources are substantially more constrained than mainstream speech-language resources, particularly for continuous contextual translation." |
| "We built the first AI sign language interpreter." | "First ISL system with zero platform integration, and the first to frame the AI as something that can refuse, wait, and ask." |

---

## 14. Changelog

### v2 → v3 (this version, repo-research session)
- **Malayalam dropped, Tamil made optional, English confirmed mandatory** — simplifies Section 6.3 and removes Malayalam as a committed deliverable risk (see 11.5).
- **STT fallback made concrete**: `ufal/whisper_streaming` confirmed superseded by `QuentinFuxa/WhisperLiveKit`; local safe-mode fallback now specifically WhisperLiveKit via its MLX backend, matching the M5 hardware, replacing v2's vague "small local model" language.
- **TTS for Tamil made concrete**: `indic-parler-tts` recommended specifically (covers Tamil + English in one self-hosted model); `IndicF5` and `Indic-TTS` explicitly ruled out with reasons (no English / training-only repo, respectively).
- **Pose extraction and storage now has a named dependency**: `sign-language-processing/pose` (`pose-format`), replacing bespoke MediaPipe boilerplate and a from-scratch storage format.
- **Gloss→pose lookup, fingerspelling fallback, and coverage reporting now have a named architectural basis**: `spoken-to-signed-translation`, explicitly stopping before its optional neural video-rendering step (see correction below).
- **Correction, not a new finding**: confirmed and flagged clearly that `spoken-to-signed-translation`'s `pose-to-video` step (pix2pix + upscaler) must not be used — conflicts with original Decision 3 and won't run real-time on the M5. Pipeline explicitly stops at the `.pose` file output.
- **Pose smoothing now has a named dependency**: `fluent-pose-synthesis`, confirmed still active.
- **Escalation loop implementation recommendation added**: `langgraph`, for its native branching/cycles/human-in-the-loop support.
- **Avatar polish option added, conditional**: `TalkingHead` for lip-sync, gated on a blend-shape verification step not yet performed.
- **Individual ISL baseline repos flagged** as not independently verified to the same depth as the org-level tools, with a recommendation to spot-check before relying on them.
- **Two ambiguous entries from the original repo list** ("sign/translate", "sign/inference") could not be resolved to real repos — flagged rather than guessed at.
- **Hardware-specific latency claims flagged as unverified** on the actual M5 unit (11.6).

### v1 → v2 (prior session)
- **Decision 6** (MacBook M5 as "reliability insurance"): upgraded to an active, threshold-triggered safe-mode fallback, justified by real, confirmed Groq free-tier rate limits.
- **Decision 7** ("two independent, always-on loops"): revised — coordinated by the rendering collision manager (Section 8), not fully blind to each other.
- **Decision 6**'s "fast/turbo-tier TTS voice" (originally unspecified, implicitly paid): replaced with the free stack in Section 6.3.
- **LLM reasoning**: Claude Haiku 4.5 → Groq-hosted Llama models.
- **Decisions 1–5, 8, 9**: unchanged, still correct as originally reasoned.
