# Setu — Stage 1: Core Pipeline & Tier-0 Agentic Behaviors — Complete Build Instructions
### Drift-proof, hallucination-resistant prompts aligned to `setu-final-architecture-v3.md`

> **Purpose.** This builds the demo-critical skeleton of Setu: curated-vocabulary
> ISL recognition, both translation directions running end to end, the escalation
> waterfall, refuse-to-fabricate, rendering collision management, and the live
> decision log. This is everything tagged P0 in architecture v3 §12 — the minimum
> that must work for Setu to be demoable at all.
>
> **Alignment guarantee.** If anything in this document conflicts with
> `setu-final-architecture-v3.md`, the architecture doc wins. If a conflict is
> found, stop and flag it rather than silently resolving it either way.
>
> **Solo build note.** This is a single-person, sequential build — not split
> across parallel agents. Tasks are still scoped to explicit file boundaries so
> that later stages (2 and 3) can be added without needing to re-touch or
> refactor Stage 1's files.
>
> **What Stage 1 does NOT include** (deferred to later stages, do not build here):
> safe-mode degradation, local fallback models (WhisperLiveKit/Ollama), session
> glossary, domain glossary, duplicate-ack batching, agent-directed reprocessing,
> sign coinage, policy handshake, asymmetric transparency, data export, Tamil,
> optional lip-sync. Building any of these now is scope drift — leave hooks
> where noted, don't implement.

---

# §A. Operating Contract — paste into CLAUDE.md (or equivalent) first

**What this module is.** Setu's core interpretation pipeline: a Deaf signer's
webcam feed is recognized against a small curated ISL vocabulary and spoken
aloud in English; a hearing speaker's voice is recognized, turned into ISL
gloss, looked up against a pose library, and rendered on a 3D avatar. Both
directions are coordinated by an agentic escalation loop that can pause,
clarify, or refuse rather than guess, and a collision manager that prevents
the two directions from rendering over each other. Everything happens
English-only, on a curated sign vocabulary, for this stage.

**GROUND TRUTH — do not silently change these:**

- **LLM reasoning:** Groq API, free tier. Use `openai/gpt-oss-20b` for
  low-latency segments and `openai/gpt-oss-120b` for higher-quality/complex
  segments. This replaced an earlier plan to use Claude Haiku 4.5 — that
  decision is superseded, do not reintroduce it.
  **AMENDED 2026-09-15 (T1.1):** this block originally pinned
  `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`. Both were verified
  absent from the live Groq catalog via `client.models.list()` on that date and
  replaced with the gpt-oss pair, which preserves the same fast-path/careful-path
  split on a single key. **These are reasoning models** — pass
  `reasoning_effort="low"` on every chat call or they spend the entire
  completion budget on hidden reasoning and return empty content.
  Real observed limits are 1,000 req/min and **8,000 tokens/min** — TPM, not
  RPM, is the binding constraint that NFR-2's chunking design must protect.
- **STT:** Groq-hosted Whisper (`whisper-large-v3-turbo`), chunked on natural
  speech pauses, never per-word (free-tier cap is real — architecture v3 §6.2,
  roughly 20 req/min, 2,000/day).
- **TTS (English only, this stage):** Groq TTS,
  `canopylabs/orpheus-v1-english`. Tamil is out of scope for Stage 1 entirely.
  **AMENDED 2026-09-15 (T1.1):** `playai-tts` was also listed here but is no
  longer in the live catalog; `canopylabs/orpheus-v1-english` is present and is
  now the only English TTS option for this stage.
- **Pose extraction and storage:** the `pose-format` package
  (`sign-language-processing/pose` on GitHub, confirmed 112 stars/MIT/actively
  maintained). Install with `pip install pose-format`. Use its `.pose` file
  format for **all** stored sign data in this project — the curated vocabulary
  library, extracted webcam landmarks, and any pose sequence produced anywhere
  in the pipeline. Do not invent a second storage format.
- **Gloss → pose lookup, fingerspelling fallback, coverage reporting:** the
  `spoken-to-signed-translation` pipeline
  (`sign-language-processing/spoken-to-signed-translation`, MIT, confirmed
  98 stars, actively maintained — mirror of `ZurichNLP/spoken-to-signed-translation`).
  Use its `text_to_gloss_to_pose` command and `--coverage-info` /
  `--coverage-stats` flags. Its bundled lexicons are not ISL — you will build
  and supply your own lexicon directory pointing at the curated vocabulary's
  `.pose` files (see T1.2, T1.9).
  **HARD RULE — DO NOT USE `text_to_gloss_to_pose_to_video` or the
  `pose-to-video` package.** That step renders neural photorealistic video via
  pix2pix + an upscaler. It directly contradicts the project's avatar decision
  (a rigged 3D skeletal avatar, not neural video) and will not run real-time on
  the target laptop. Stop at the `.pose` file output, full stop.
- **Avatar rig:** an already-Blender-verified Sketchfab CC BY 4.0 model,
  Mixamo/Unity-style bone naming, finger-level bones, exported as `.glb`
  following architecture v3 §7's checklist (rest pose, T-pose confirmed,
  transforms applied). **This is a plain glTF, not a native `.vrm` file** — it
  has no built-in VRMHumanoid bone-mapping metadata. Bone names must be mapped
  to Kalidokit's output manually. Do not assume a VRM-style auto-mapping will
  work.
- **Retargeting:** `kalidokit` (npm), driving Three.js + `@pixiv/three-vrm`.
  Kalidokit is used **only** in the Speech→Sign direction, to animate the
  avatar from looked-up pose sequences. It is **not** used in the Sign→Speech
  direction — that direction only extracts landmarks for recognition input, it
  does not render an avatar.
- **Meeting bridge:** OBS Virtual Camera (built into OBS Studio) +
  `pyvirtualcam` (`letmaik/pyvirtualcam`) for sending rendered frames to it
  from Python. No Zoom/Meet SDK, no platform API — that was evaluated and
  explicitly rejected. Do not reintroduce it.
- **Escalation loop:** recommended implementation is `langchain-ai/langgraph`
  for its native branching/cycles/human-in-the-loop support. If, after
  verifying its current API (see T1.14), it adds more overhead than value for
  this scope, a hand-rolled state machine is an acceptable substitute — but the
  Observe → Decide → Action log shape (architecture v3 §9) must be preserved
  either way.
- **Explicitly ruled out — do not reintroduce under any task:** neural/photoreal
  avatar rendering in any form, Zoom/Meet SDK or Google Meet Media API
  integration, broad Calendar/Docs/codebase OAuth ingestion, automatic
  topic-shift detection, real speaker diarization, CV-based emotion/feedback
  detection.

**ANTI-HALLUCINATION RULES:**

1. Several external libraries in this build were verified in a prior research
   session and their real usage is stated above/in the tasks below
   (`pose-format`'s CLI, `spoken-to-signed-translation`'s CLI flags,
   Kalidokit's import shape, WhisperLiveKit's general CLI shape). Others —
   **Groq's exact current Python SDK method signatures, `fluent-pose-synthesis`'s
   exact CLI/API, and the exact language-subtag code (if any) `spoken-to-signed-translation`
   expects for ISL** — were **not** independently verified. Before writing code
   against any of these, search for and confirm the real current API/usage in
   this session. If verification isn't possible, stop and ask rather than
   guess at a plausible-looking signature.
2. Never assume a package version, CLI flag, or function signature — confirm
   against the installed package's actual `--help` output, docstrings, or
   current docs before use.
3. Import the `.pose` format, the gloss/coverage status categories, and the
   decision-log entry schema (§B.2) from their one canonical definition. Never
   redefine a second, slightly different version in a different file.
4. If a requirement below is ambiguous, ask one clarifying question rather
   than assume — and if you must proceed without an answer, state the
   assumption explicitly in the commit message.
5. Never fabricate VERIFY output. Run the real command, paste the real
   result.

**ANTI-DRIFT RULES:**

1. Only touch the files listed in a task's "Files you may touch."
2. Don't refactor unrelated code or add features not in this document, even
   if they'd be easy to add while you're in that file.
3. Keep the schemas in §B.2 byte-aligned with this document — field names and
   types must match exactly wherever they're used.
4. Don't start building anything listed under "What Stage 1 does NOT include"
   above, even partially, even as a stub with more than a `# TODO: Stage 2/3`
   comment.

**QUALITY GATES** (must hold at the end of every task):

- Type safety on all function signatures (Python type hints / TypeScript
  types — match whatever the file's existing language is).
- Every external input (STT transcript, recognized gloss, API response) is
  validated at the boundary before being trusted downstream.
- No secrets (API keys) committed in code — environment variables only, see
  §B.1.
- Every component that calls an external API (Groq, in particular) has a
  typed, structured error path — not a bare `except: pass`.

**WORKING METHOD** (every task):

1. State a short plan before touching multi-file tasks; wait for confirmation
   if the plan touches more than 2 files.
2. Implement.
3. Run the real VERIFY command(s) for that task and paste the real output.
4. Commit with message format: `[T1.n] <short description>`.

**DEFINITION OF DONE** (every task):

- Runs cleanly with no errors.
- Typechecks (if applicable to the language).
- VERIFY passes with pasted real output, not a description of expected
  behavior.
- Only the files listed in "Files you may touch" were touched.
- Any shared contract (§B.2) used by the task is unchanged from its canonical
  definition.

---

# §B. Canonical Specifications

## B.1 Environment variables

```bash
# .env — fill in real values, never commit this file
GROQ_API_KEY=              # Groq free-tier key — https://console.groq.com
                            # Verify current signup/key process before assuming a flow.
OBS_VIRTUALCAM_DEVICE=      # OS-specific virtual camera device name/index,
                            # confirm the actual name after OBS Virtual Camera is started
CURATED_VOCAB_PATH=./data/vocab/       # directory of curated .pose entries, see T1.2
LEXICON_PATH=./data/lexicon/           # spoken-to-signed-translation lexicon dir, see T1.9
AVATAR_GLB_PATH=./assets/avatar.glb    # exported per architecture v3 §7
```

## B.2 Data contracts

These are canonical. Every task below that touches them must match exactly.

```python
# segment.py — a single chunk of speech or signing passed through the waterfall
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Direction(Enum):
    SIGN_TO_SPEECH = "sign_to_speech"
    SPEECH_TO_SIGN = "speech_to_sign"

class CoverageStatus(Enum):
    """Mirrors spoken-to-signed-translation's --coverage-info categories.
    Do not invent a different status set — reuse this one everywhere a
    confidence/coverage tier needs to be represented, including the
    Sign->Speech recognition confidence tiering (map recognition confidence
    onto these same four buckets for one consistent status vocabulary
    across both directions)."""
    LEXICON_HIT = "lexicon_hit"          # green
    LANGUAGE_BACKUP = "language_backup"  # yellow
    FINGERSPELLING = "fingerspelling"    # orange
    UNMATCHED = "unmatched"              # red

@dataclass
class Segment:
    id: str
    direction: Direction
    raw_input: str            # transcript (speech->sign) or recognized gloss text (sign->speech)
    confidence: float         # 0.0-1.0
    coverage_status: Optional[CoverageStatus]
    timestamp: float

# decision_log_entry.py — one row in the Observe/Decide/Action panel (architecture v3 §9)
@dataclass
class DecisionLogEntry:
    timestamp: float
    stage: str          # "OBSERVE" | "DECIDE" | "ACTION"
    segment_id: str
    detail: str          # human-readable, e.g. "candidate disagreement: HIGH"
```

```
# .pose file format: DO NOT define a custom schema for this — it is entirely
# owned by the pose-format package (sign-language-processing/pose). Read and
# write it only via that package's own Python API. Verify the exact read/write
# function names against the installed package before using them (see T1.3).
```

## B.3 File/folder structure — target end state after Stage 1

```
setu/
  .env
  data/
    vocab/                  # curated ISL vocabulary, .pose files — T1.2, T1.3
    lexicon/                # spoken-to-signed-translation lexicon dir — T1.9
  assets/
    avatar.glb               # T1.11
  backend/
    contracts.py             # §B.2 — canonical, T1.0
    recognition/
      extract.py             # pose-format video_to_pose wrapper — T1.3
      classifier.py           # curated-vocabulary classifier — T1.4
    sign_to_speech/
      reasoning.py             # Groq LLM gloss->sentence — T1.5
      tts_output.py             # Groq TTS + virtual mic — T1.6
    speech_to_sign/
      stt_input.py               # Groq Whisper, chunked — T1.7
      reasoning.py                # Groq LLM sentence->gloss — T1.8
      gloss_lookup.py             # spoken-to-signed-translation wrapper — T1.9
      pose_smoothing.py           # fluent-pose-synthesis wrapper — T1.10
    waterfall/
      escalation.py             # T1.14
      decision_log.py             # T1.15
    collision/
      state_machine.py           # T1.13
    meeting_bridge/
      virtualcam.py               # pyvirtualcam wrapper — T1.16
  frontend/
    avatar/
      loader.ts                  # glTF load + bone mapping — T1.11
      retarget.ts                 # Kalidokit wiring — T1.12
    decision_log_panel/
      panel.ts                    # T1.15 UI half
```

---

# §C. Tasks

## T1.0 · Project scaffold and canonical contracts — `backend/contracts.py` · P0 · depends: none

> **PROMPT**
> Goal: set up the project structure per §B.3 and write `contracts.py`
> containing exactly the dataclasses/enums in §B.2, verbatim.
> Files you may touch: create the directory tree in §B.3; write `backend/contracts.py`, `.env.example`.
> Requirements: no logic beyond the data contracts themselves — this file is
> imported everywhere else, it must stay dependency-free of the rest of the app.
> **VERIFY:** `python -c "from backend.contracts import Segment, Direction, CoverageStatus, DecisionLogEntry; print('ok')"` — paste real output.

## T1.1 · Environment and Groq client setup — `backend/config.py` · P0 · depends: T1.0

> **PROMPT**
> Goal: load environment variables from §B.1 and construct a Groq API client.
> Files you may touch: `backend/config.py`, `.env.example`.
> Requirements: **before writing any Groq client code, search for and confirm
> Groq's actual current Python SDK — package name, client construction, and
> call signature — in this session. Do not assume an OpenAI-compatible shape
> without confirming it.** Fail loudly and clearly if `GROQ_API_KEY` is unset.
> **VERIFY:** run a minimal real completion call against Groq with a trivial
> prompt (e.g. "say hello") and paste the real response text.

## T1.2 · Curated ISL vocabulary definition — `data/vocab/vocab_manifest.md` · P0 · depends: none

> **PROMPT**
> Goal: this is the single highest-leverage task in the whole project
> (architecture v3 §11.1) — define the exact, finite list of ISL signs Setu
> will recognize and produce for the Stage 1 demo. This is a content task, not
> a code task.
> Files you may touch: `data/vocab/vocab_manifest.md` only.
> Requirements: list every sign as `gloss_id | English meaning | notes`.
> Keep the list small enough to be fully rehearsed and validated by a human —
> tens of entries, not hundreds, for a hackathon timeline. This manifest is
> the source of truth every other recognition/lookup task in this stage reads
> from. Do not proceed to T1.3/T1.4/T1.9 until this file exists and is
> reviewed by the project owner — this is a human decision, not something to
> auto-generate.
> **VERIFY:** the file exists, is non-empty, and every row has all three
> fields — paste the file's contents.

## T1.3 · Pose extraction pipeline — `backend/recognition/extract.py` · P0 · depends: T1.0, T1.2

> **PROMPT**
> Goal: wrap `pose-format`'s extraction tooling to turn a recorded video of
> each vocabulary entry (or a live webcam stream) into a `.pose` file.
> Files you may touch: `backend/recognition/extract.py`.
> Requirements: install with `pip install pose-format`. Confirmed CLI usage
> from this session's research: `video_to_pose --format mediapipe -i example.mp4
> -o example.pose` for a single file, `videos_to_poses --format mediapipe
> --directory /path/to/videos` for a batch. **Before using either the CLI or
> any Python-level API of this package for live-stream (not file-based)
> extraction, verify the package's actual Python API for that use case in this
> session** — the confirmed usage above is file-based; live webcam extraction
> may need a different entry point that wasn't independently checked. Store
> resulting `.pose` files under `data/vocab/` per T1.2's manifest, one per
> entry (plus repeated takes if useful for classifier training in T1.4).
> **VERIFY:** run extraction on one real recorded sample video and confirm a
> valid `.pose` file is produced — paste the command and file size/output.

## T1.4 · Curated-vocabulary recognition classifier — `backend/recognition/classifier.py` · P0 · depends: T1.3

> **PROMPT**
> Goal: classify a live/extracted `.pose` sequence against the closed
> vocabulary set from T1.2, returning a `Segment` with `confidence` filled in.
> Files you may touch: `backend/recognition/classifier.py`.
> Requirements: given the small, closed vocabulary, prefer a simple, low-risk
> approach over adapting a full external recognition codebase — e.g. a
> nearest-neighbor / DTW distance comparison against the recorded reference
> `.pose` sequences from T1.3, or a small classifier trained only on this
> vocabulary. The individually-listed baseline repos considered during
> research (`Sooryak12/Indian-Sign-Language-Recognition`,
> `Exploration-Lab/ISLTranslate`, `Swaroop-Srisailam/Continuous-Indian-Sign-Language-Recognition`,
> `aju22/Real-Time-ISL-Translation`) were **not independently verified for
> current activity/quality** — treat any of them as an optional reference only
> if the simple approach above proves insufficient, and check each repo's
> actual current state before depending on it. Output confidence must be a
> real calibrated value (e.g. normalized inverse distance), not a fixed
> placeholder.
> **VERIFY:** run classification against a held-out real recording of a known
> vocabulary sign and paste the real predicted `gloss_id` + confidence value.

## T1.5 · Sign→Speech: gloss-to-sentence reasoning — `backend/sign_to_speech/reasoning.py` · P0 · depends: T1.1, T1.4

> **PROMPT**
> Goal: given a recognized gloss (from T1.4) and its confidence, call Groq to
> reconstruct a fluent English sentence.
> Files you may touch: `backend/sign_to_speech/reasoning.py`.
> Requirements: use `llama-3.1-8b-instant` by default; this is a hook point
> for Stage 3's FAST/CAREFUL policy split — do not implement that split now,
> just call the one model. Confirm the real Groq call signature per T1.1
> before writing this.
> **VERIFY:** pass a real recognized gloss through and paste the real returned
> sentence.

## T1.6 · Sign→Speech: streaming TTS and virtual mic output — `backend/sign_to_speech/tts_output.py` · P0 · depends: T1.5, T1.16

> **PROMPT**
> Goal: stream the reconstructed sentence (T1.5) through Groq TTS
> (`playai-tts` or `canopylabs/orpheus-v1-english`) and output the audio to
> the virtual microphone set up in T1.16.
> Files you may touch: `backend/sign_to_speech/tts_output.py`.
> Requirements: **verify Groq's current TTS API call shape and audio format
> in this session before writing this** — not independently confirmed beyond
> the model names existing on the free tier. Set the `SETU_TTS_ACTIVE` flag
> (see T1.13) for the duration of playback — this is required for T1.13's
> self-TTS suppression to work, do not skip it.
> **VERIFY:** run a real sentence through and confirm audible output reaches
> the virtual mic device — paste the command and a description of the real
> observed output (waveform length, device confirmation, etc.).

## T1.7 · Speech→Sign: chunked STT — `backend/speech_to_sign/stt_input.py` · P0 · depends: T1.1

> **PROMPT**
> Goal: capture microphone audio, chunk on natural speech pauses (not fixed
> windows, not per-word), and transcribe via Groq Whisper
> (`whisper-large-v3-turbo`).
> Files you may touch: `backend/speech_to_sign/stt_input.py`.
> Requirements: verify Groq's current audio-transcription API call shape in
> this session before writing it. Respect the free-tier limits noted in §A —
> design the chunking so a real conversation doesn't exceed ~20 req/min.
> **VERIFY:** speak a real test sentence, paste the real returned transcript.

## T1.8 · Speech→Sign: sentence-to-gloss reasoning — `backend/speech_to_sign/reasoning.py` · P0 · depends: T1.7

> **PROMPT**
> Goal: given a transcript (T1.7), call Groq to produce ISL gloss notation.
> Files you may touch: `backend/speech_to_sign/reasoning.py`.
> Requirements: same model/verification discipline as T1.5.
> **VERIFY:** pass a real transcript through, paste the real returned gloss.

## T1.9 · Speech→Sign: gloss-to-pose lookup with fallback and coverage — `backend/speech_to_sign/gloss_lookup.py` · P0 · depends: T1.2, T1.8

> **PROMPT**
> Goal: wrap `spoken-to-signed-translation`'s lookup pipeline, pointed at a
> lexicon directory built from the T1.2 curated vocabulary's `.pose` files
> (T1.3), to convert gloss into a pose sequence with fingerspelling fallback
> and coverage reporting.
> Files you may touch: `backend/speech_to_sign/gloss_lookup.py`,
> `data/lexicon/` (lexicon directory structure only, built from T1.2's `.pose`
> files).
> Requirements: install per `pip install git+https://github.com/sign-language-processing/spoken-to-signed-translation.git`
> (or the `ZurichNLP` origin if the mirror is unavailable — verify which is
> current). Confirmed CLI usage from this session's research:
> `text_to_gloss_to_pose --text <input> --glosser <simple|spacylemma|rules|nmt>
> --lexicon <path_to_directory> --spoken-language <code> --signed-language <code>
> --pose <output>.pose`, plus `--coverage-info` (terminal) or
> `--coverage-stats <file.json>`. **The `--signed-language` code expected for
> ISL was not verified in this session — before using it, check the package's
> source to see whether this parameter is a real ISO/IANA language-subtag
> validated against a list, or simply a folder-name selector for the lexicon
> directory. Do not assume a specific ISL language code exists in their
> validated list without checking; if it's a free-form selector, any
> consistent string (e.g. "isl") is fine.** Map the returned coverage category
> onto this project's `CoverageStatus` enum (§B.2) exactly — lexicon hit →
> `LEXICON_HIT`, language backup → `LANGUAGE_BACKUP`, fingerspelling →
> `FINGERSPELLING`, unmatched → `UNMATCHED`.
> **HARD RULE:** call only `text_to_gloss_to_pose`, never
> `text_to_gloss_to_pose_to_video`. Do not install the `pose-to-video`
> package's video-rendering extras at all.
> **VERIFY:** run a real gloss (including at least one term deliberately
> outside the curated vocabulary, to exercise the fingerspelling fallback)
> through and paste the real `.pose` output confirmation and the real
> `--coverage-info` output.

## T1.10 · Pose sequence smoothing — `backend/speech_to_sign/pose_smoothing.py` · P0 · depends: T1.9

> **PROMPT**
> Goal: smooth the concatenated pose sequence from T1.9 before handing off to
> retargeting.
> Files you may touch: `backend/speech_to_sign/pose_smoothing.py`.
> Requirements: use `fluent-pose-synthesis` (`sign-language-processing/fluent-pose-synthesis`).
> **This package's exact installation and call API were not independently
> verified in this session beyond its stated purpose ("given a sequence of
> individual poses, creates a fluent pose sequence with good intonation and
> prosody") — search for and confirm its real current usage before writing
> this task's code.** If, after checking, its current API doesn't cleanly fit
> a hackathon timeline, a minimal linear-interpolation smoothing between pose
> frames is an acceptable fallback for Stage 1 — note in the commit message
> which path was taken and why.
> **VERIFY:** run the real smoothing step on T1.9's output and paste a
> before/after frame-count or a description of the real transformation
> applied.

## T1.11 · Avatar loading and bone mapping — `frontend/avatar/loader.ts` · P0 · depends: none (parallel to backend tasks)

> **PROMPT**
> Goal: load the exported `.glb` avatar (already produced per architecture v3
> §7's Blender checklist) into a bare Three.js scene and manually map its
> Mixamo/Unity-style bone names to the naming Kalidokit expects.
> Files you may touch: `frontend/avatar/loader.ts`.
> Requirements: use `GLTFLoader`. **Do not assume a VRM-style automatic
> humanoid bone mapping applies** — this asset is not a `.vrm` file. Reference
> `europanite/webcam_to_avatar` only as a pattern for the general
> webcam→landmarks→Kalidokit→avatar wiring shape, not as a drop-in loader — it
> targets a native `.vrm` file, which this asset is not. Confirm this task's
> isolated load first (loader + camera only, nothing else) before any
> retargeting logic touches it, per architecture v3 §7 step 5.
> **VERIFY:** load the real `.glb` in an isolated scene, confirm it stands
> upright and faces the correct direction — paste a description of the real
> rendered result (or a screenshot reference).

## T1.12 · Kalidokit retargeting — `frontend/avatar/retarget.ts` · P0 · depends: T1.11

> **PROMPT**
> Goal: play back a `.pose` sequence (from T1.10) by driving the avatar's
> bones (T1.11) through Kalidokit's solver, frame by frame.
> Files you may touch: `frontend/avatar/retarget.ts`.
> Requirements: `npm install kalidokit`. Confirmed import shape from this
> session's research: `import { Face, Pose, Hand } from "kalidokit"`, with
> `Kalidokit.Pose.solve(...)` and `Kalidokit.Hand.solve(...)` taking landmark
> arrays and returning rotation/blendshape values to apply to the mapped
> bones (T1.11). This direction is **playback of pre-looked-up pose
> sequences**, not live webcam retargeting — do not wire a live camera feed
> into this task.
> **VERIFY:** play back one real `.pose` sequence from T1.10 and confirm the
> avatar's bones visibly move through the expected motion — paste a
> description of the real observed playback.

## T1.13 · Rendering collision state machine and self-TTS suppression — `backend/collision/state_machine.py` · P0 · depends: T1.6

> **PROMPT**
> Goal: implement the IDLE/SIGNING/DRAINING state machine and the parallel
> SILENCE/VOICE_DETECTED/SPEAKING audio state exactly as specified in
> architecture v3 §8.1–8.2.
> Files you may touch: `backend/collision/state_machine.py`.
> Requirements: hold new avatar work only when both (a) avatar has active,
> uncompleted output and (b) new speech persists past a 250–400ms debounce
> window; release only at a safe clip boundary, never mid-gesture; enforce a
> max hold timeout; surface names/numbers/direct questions sooner than routine
> content when held. **Implement the `SETU_TTS_ACTIVE` flag exactly as
> described in v3 §8.3** — while true, ignore or explicitly mark overlap
> events as local output rather than external speech. T1.6 must set this flag;
> confirm that wiring is actually in place, don't just assume it from that
> task's description. Before any further refinement of this feature, run the
> Day-1 acceptance test from v3 §8.4 first (can remote audio reach STT, can
> generated TTS reach the meeting, without self-confusion) and report the real
> result.
> **VERIFY:** simulate one real overlap scenario (trigger TTS output, then
> real or simulated incoming speech during playback) and confirm the state
> machine holds correctly and does not falsely trigger on its own TTS — paste
> the real state transition log.

## T1.14 · Escalation waterfall — `backend/waterfall/escalation.py` · P0 · depends: T1.4, T1.9

> **PROMPT**
> Goal: implement the escalation loop from architecture v3 §3 — local
> lookup → session glossary (leave as a no-op passthrough hook for Stage 2,
> do not implement it now) → domain glossary (same, no-op hook for Stage 2) →
> agent-directed reprocessing (no-op hook for Stage 3) → LLM reasoning →
> clarification / refuse-to-fabricate / sign-coinage-offer (leave sign
> coinage as a no-op hook for Stage 3, implement clarification and
> refuse-to-fabricate fully now).
> Files you may touch: `backend/waterfall/escalation.py`.
> Requirements: **before implementing, verify `langgraph`'s current API in
> this session** (`langchain-ai/langgraph`) — confirm whether its current
> branching/cycle/interrupt primitives cleanly fit this shape for a hackathon
> timeline. If they do, build the waterfall as a LangGraph graph. If, after
> checking, the current API adds more complexity than it saves for this
> scope, build a plain Python state machine instead — either is acceptable,
> but the choice must be based on a real check of the current API, not
> assumed. Implement the escalation ladder exactly as described in
> architecture v3 §4.1: hysteresis (2 consecutive low-confidence segments to
> trigger, 1 clean segment or explicit repeat to resume, cooldown after),
> stall both sides simultaneously on first trigger, show candidate/fingerspelling
> on second trigger, proceed with an "uncertain" label as final fallback.
> Refuse-to-fabricate must check T1.9's `CoverageStatus` — an `UNMATCHED`
> result must never silently proceed as if translated.
> **VERIFY:** run one real segment through each of the three terminal paths
> (confident translate, clarification triggered, refuse-to-fabricate
> triggered) and paste the real waterfall trace for each.

## T1.15 · Decision log — `backend/waterfall/decision_log.py` + `frontend/decision_log_panel/panel.ts` · P0 · depends: T1.14

> **PROMPT**
> Goal: implement the Observe→Decide→Action decision log exactly per
> architecture v3 §9, using the `DecisionLogEntry` contract from §B.2.
> Files you may touch: `backend/waterfall/decision_log.py`,
> `frontend/decision_log_panel/panel.ts`.
> Requirements: if T1.14 was built as a LangGraph graph, surface the graph's
> own state transitions as `DecisionLogEntry` rows rather than building
> separate parallel logging. Show structured decision factors only — never
> raw model chain-of-thought. Include a visible manual override/pause
> control, always rendered, regardless of state.
> **VERIFY:** run the same three scenarios from T1.14's VERIFY step and
> confirm each produces a correct, readable log sequence in the panel — paste
> the real rendered log content for each.

## T1.16 · OBS virtual camera/mic bridge — `backend/meeting_bridge/virtualcam.py` · P0 · depends: none (parallel)

> **PROMPT**
> Goal: send rendered avatar frames (from the Three.js/Kalidokit output) and
> TTS audio (from T1.6) to OBS's virtual camera and a virtual microphone
> device respectively.
> Files you may touch: `backend/meeting_bridge/virtualcam.py`.
> Requirements: `pip install pyvirtualcam`. Verify the package's current API
> for frame sending in this session before relying on a specific method name
> from memory. OBS Studio's built-in Virtual Camera must be started manually
> before this connects — document that as a manual pre-step, don't try to
> automate OBS startup itself for Stage 1.
> **VERIFY:** send one real test frame and confirm it's visible in a
> real video-call app's camera preview (or another virtual-cam-consuming
> app) — paste a description of the real observed result.

---

# §D. Build order

Illustrative for a ~24–30 hour hackathon window (previously assumed in this
project's feasibility discussion) — adjust proportionally to the actual
timeline, keeping T1.2 first regardless of total duration.

| Window | Tasks | Expected outcome by end of window |
|---|---|---|
| Hour 0–2 | T1.0, T1.1, T1.2 | Contracts in place, Groq client verified working, curated vocabulary manifest reviewed and locked |
| Hour 2–6 | T1.3, T1.4, T1.11 (parallel) | Pose extraction working on real recordings; a first-pass recognition classifier; avatar loads and stands correctly in an isolated scene |
| Hour 6–10 | T1.5, T1.6, T1.16 | Sign→Speech direction producing real audible speech through the virtual mic |
| Hour 10–15 | T1.7, T1.8, T1.9, T1.10 | Speech→Sign direction producing real smoothed `.pose` output, with fingerspelling fallback confirmed working on an out-of-vocabulary term |
| Hour 15–19 | T1.12 | Avatar visibly animates from a real looked-up pose sequence |
| Hour 19–24 | T1.13, T1.14, T1.15 | Escalation waterfall, collision management, and decision log all wired and passing their real VERIFY scenarios |
| Hour 24+ (buffer) | Rehearsal against demo scenes 1–5 (architecture v3 §10) | End-to-end run against the curated vocabulary, in the actual demo room if possible, per v3 §11.4's warning about collision-management needing real rehearsal, not just dev-environment testing |

---

# §E. Final acceptance

1. ✅ `data/vocab/vocab_manifest.md` exists, was reviewed by the project owner, and every entry has a corresponding extracted `.pose` file.
2. ✅ A real recognized sign (T1.4) produces a real fluent English sentence (T1.5) audible through the virtual mic (T1.6, T1.16).
3. ✅ A real spoken sentence (T1.7) produces real ISL gloss (T1.8), a real looked-up and smoothed pose sequence (T1.9, T1.10) with a real `--coverage-info` report, and visible avatar motion (T1.11, T1.12).
4. ✅ A deliberately out-of-vocabulary term triggers the real fingerspelling fallback (T1.9) — confirmed via real `--coverage-info` output showing `FINGERSPELLING` or `UNMATCHED`, not a silent guess.
5. ✅ A deliberately low-confidence recognition triggers the real escalation ladder (T1.14) — both sides stall, a targeted or generic clarification is issued, and it resumes correctly on a clean follow-up segment.
6. ✅ The `SETU_TTS_ACTIVE` flag (T1.6, T1.13) is confirmed, via the Day-1 acceptance test (v3 §8.4), not to falsely trigger a collision hold on Setu's own TTS output.
7. ✅ A real simulated overlap (T1.13) produces a correct hold-and-resume, never mid-gesture, always at a clip boundary.
8. ✅ The decision log (T1.15) shows a correct, readable Observe→Decide→Action trace for all of the above scenarios, with the manual override control visibly present at all times.
9. ✅ No task touched a file outside its listed scope; no component from "What Stage 1 does NOT include" was built, even partially.
10. ✅ Demo scenes 1–5 from architecture v3 §10 can be run live, in sequence, against the curated vocabulary, without manual intervention beyond the scripted inputs.
