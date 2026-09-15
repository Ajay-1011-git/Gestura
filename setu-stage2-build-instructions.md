# Setu — Stage 2: Resilience & Adaptive Behaviors — Complete Build Instructions
### Drift-proof, hallucination-resistant prompts aligned to `setu-final-architecture-v3.md`, `setu-stage2-prd.md`, and `setu-stage2-trd-implementation-plan.md`

> **Purpose.** This builds the resilience and adaptive-behavior layer on top of
> Stage 1's already-verified core pipeline: session and domain glossaries,
> safe-mode degradation with a local fallback stack, duplicate-ack batching,
> and a real ISL fingerspelling path. This is everything tagged Tier 1 / P1 in
> architecture v3 §4.2 and §12, plus the fingerspelling gap discovered live
> during Stage 1's own acceptance run.
>
> **Alignment guarantee.** If anything in this document conflicts with
> `setu-final-architecture-v3.md`, `setu-stage2-prd.md`, or
> `setu-stage2-trd-implementation-plan.md`, those win, in that order. If this
> document conflicts with code Stage 1 already built and verified, **the
> as-built Stage 1 code is ground truth** — do not silently "fix" or refactor
> working Stage 1 code to match an assumption made here.
>
> **Solo build note.** Same as Stage 1 — single-person, sequential build, not
> split across parallel agents. Tasks are scoped to explicit file boundaries,
> most of them *extending* a named Stage 1 file at one specific point rather
> than replacing it.
>
> **Prerequisite, not optional:** Stage 1's §E Final Acceptance must be fully
> green, including the live-camera-run item, before starting any task below.
>
> **What Stage 2 does NOT include** (deferred to Stage 3, do not build here):
> agent-directed reprocessing, collaborative sign coinage, policy handshake,
> asymmetric transparency, data export, Tamil, optional lip-sync, the
> compositional pose modifier, shared bidirectional discourse state, the
> medical/legal boundary gate.
>
> **What Stage 2 also does NOT include, named explicitly because it's
> tempting to fold in while you're in these files:** any change to the
> curated recognition or rendering vocabulary itself — `data/vocab/`,
> `build_lexicon()`, the lexicon resolution pipeline, `backend/recognition/
> classifier.py`, or `backend/recognition/capture.py`. The 40-word/16-word
> vocabulary asymmetry and the 1920×1080/854×480 resolution mismatch found
> during Stage 1's live run are real (see PRD R-10) but are a content
> problem, not a resilience problem — out of scope here on purpose.

---

# §A. Operating Contract — paste into CLAUDE.md (or equivalent) first

**What this module is.** A resilience and context-awareness layer added on
top of Setu's already-working Stage 1 pipeline. It gives the system a second,
local path for STT and LLM reasoning when Groq falters; a way to remember
what's already been resolved in a call and to prefer domain-appropriate
vocabulary when told which domain it's in; a way to combine filler speech
without losing real content; and a real ISL fingerspelling fallback where
Stage 1 only had a defined-but-unreachable status for one. It does not touch
recognition, rendering, or the collision manager — those are stable Stage 1
surfaces this stage builds around, not on top of.

**GROUND TRUTH — do not silently change these:**

- **`Interpreter` (`backend/orchestrator.py`, built in Stage 1 as T1.17) is
  the single integration point.** Every task below that needs to hook into
  the live per-segment loop does so through `Interpreter`, not through a new
  parallel supervisor, poller, or process. If a task's instructions below
  seem to require touching `Interpreter`, that is expected and listed
  explicitly in that task's file scope — it is not scope creep.
- **LLM reasoning (Groq side, unchanged from Stage 1):** `openai/gpt-oss-20b`
  / `openai/gpt-oss-120b`, **both requiring an explicit `reasoning_effort`
  parameter** or they return empty content. Real observed limits are 1,000
  req/min and **8,000 tokens/min** — the token-per-minute limit, not the
  request-per-minute one, is what session/domain glossary lookups (T2.1,
  T2.2) exist to protect, since every glossary hit is one fewer Groq call
  spending that budget.
- **Stage 1's recognition classifier is a trained neural classifier**
  (`neural.py`, ~510k params, 76.1% on the 40-word vocabulary) — not DTW, and
  not AI4Bharat INCLUDE's checkpoints (independently tried and rejected at
  47.8% due to an absolute-pixel-coordinate mismatch). This is settled;
  Stage 2 does not touch it and does not try to improve on it.
- **Stage 1's avatar timeline/resampling logic lives in
  `build_avatar_sequence`** (T1.10) and already owns both concerns jointly,
  after a real drift bug was found and fixed there. Stage 2 does not touch
  `backend/speech_to_sign/pose_smoothing.py`.
- **A known, named, out-of-Stage-2-scope asymmetry exists**: Sign→Speech
  recognizes 40 words, Speech→Sign renders 16; the rendering lexicon
  (ISLRTC, 1920×1080) and the recognition training corpus (854×480) are
  different sources at different resolutions. Do not touch `data/vocab/`,
  `build_lexicon()`, or the lexicon resolution pipeline in any task below —
  see the note at the top of this document.
- **Fingerspelling has no usable off-the-shelf ISL dataset.** Checked in
  planning: `RealSign62/RealSign-Indian-Sign-Language-Dataset`, the IEEE
  DataPort ISL set, and `ayeshatasnim-h/Indian-Sign-Language-dataset` are
  static labeled images for classifier training, not `.pose` motion data; a
  2025 continuous ISL fingerspelling corpus (Kirandevraj et al., ACL WSLP) is
  unaligned news-broadcast video with no extracted poses or confirmed usable
  license. T2.6 records the handshapes personally, the same way T1.2's
  vocabulary was built — do not go looking for a shortcut dataset here, this
  was already checked.
- **New packages for this stage:** `QuentinFuxa/WhisperLiveKit` (STT
  fallback, MLX backend for the target Apple Silicon machine) and Ollama
  (LLM fallback, model not yet chosen). Both were partially verified in prior
  research but **their exact current CLI/API were not reconfirmed in this
  planning session** — see the anti-hallucination rules below, this applies
  with extra weight to these two.

**ANTI-HALLUCINATION RULES:**

1. `WhisperLiveKit`'s CLI/flags were observed evolving fast in prior
   research — `qwen3-streaming`, `causal mode`, and `voxtral-mlx` were all
   found as current options in one earlier session. **Re-check its current
   README/flags in this session before writing any code against it** — do
   not assume a previously-observed flag (including `voxtral-mlx` itself) is
   still current.
2. Ollama's current Python client/API shape was not verified in planning.
   Confirm the real, current shape before writing `llm_fallback.py`.
3. The real signal the currently-installed Groq Python SDK exposes for
   latency, errors, and rate-limit/queue state was not verified in planning.
   Before wiring T2.3's trigger conditions, confirm what's actually available
   (exception types, response headers, timing you measure yourself) rather
   than assuming a specific shape.
4. Never assume a package version, CLI flag, or function signature for any
   new dependency in this stage — confirm against the installed package's
   real `--help` output, docstrings, or current docs before use.
5. Import `GlossaryHit`, `DomainContext`, and `SafeModeEvent` from their one
   canonical definition (§B.2, appended to `backend/contracts.py`). Import
   `Segment`, `CoverageStatus`, and `DecisionLogEntry` from Stage 1's
   existing `backend/contracts.py` unchanged — never redefine any of them a
   second time.
6. Never fabricate VERIFY output. Run the real command, paste the real
   result — including real measured latency numbers for T2.4, not estimates.

**ANTI-DRIFT RULES:**

1. Only touch the files listed in a task's "Files you may touch." Most tasks
   below edit an *existing* Stage 1 file at one specific, named point (a
   hook, a routing check) — do not refactor anything else in that file while
   you're in it.
2. The following Stage 1 files are explicitly **not** in scope for any Stage
   2 task and must not be touched: `backend/recognition/classifier.py`,
   `backend/recognition/capture.py`, `backend/recognition/extract.py` (T2.6
   *calls* it, it does not modify it), `backend/speech_to_sign/
   pose_smoothing.py`, `data/vocab/`, and anything implementing
   `build_lexicon()`.
3. Don't start building anything listed under "What Stage 2 does NOT
   include" above, even partially, even as a stub with more than a
   `# TODO: Stage 3` comment.
4. Keep §B.2's schemas byte-aligned with this document, and keep Stage 1's
   existing `Segment`/`CoverageStatus`/`DecisionLogEntry` fields exactly as
   they already are.

**QUALITY GATES** (must hold at the end of every task):

- Type safety on all function signatures, matching whichever language the
  file already uses.
- Every external input (a fallback STT transcript, an Ollama completion, a
  glossary file's parsed content) validated at the boundary before being
  trusted downstream — same discipline Stage 1 already applied to Groq.
- No secrets committed in code — environment variables only (§B.1).
- Every safe-mode transition and every glossary hit/miss produces a real
  `DecisionLogEntry` — no console-only debug logging as a substitute.

**WORKING METHOD** (every task):

1. State a short plan before touching multi-file tasks; wait for
   confirmation if the plan touches more than 2 files.
2. Implement.
3. Run the real VERIFY command(s) for that task and paste the real output.
4. Commit with message format: `[T2.n] <short description>`.

**DEFINITION OF DONE** (every task):

- Runs cleanly with no errors; typechecks.
- VERIFY passes with real pasted output, not a description of expected
  behavior.
- Only the files listed in "Files you may touch" were touched, and only at
  the specific point named (a hook, a routing check) where the task says so.
- Any shared contract (§B.2, or Stage 1's existing contracts) used by the
  task is unchanged from its canonical definition.

---

# §B. Canonical Specifications

## B.1 Environment variables

```bash
# .env additions for Stage 2 — append to Stage 1's existing .env, don't replace it
OLLAMA_MODEL=                       # chosen after real benchmarking on the target machine — see T2.4
WHISPERLIVEKIT_BACKEND=             # confirm the current correct value before setting this — see T2.4's anti-hallucination note
SAFE_MODE_LATENCY_THRESHOLD_MS=     # tune against Stage 1's real observed latency (its live run measured ~1.8s end-to-end) — do not guess a number unrelated to that baseline
SAFE_MODE_QUEUE_AGE_THRESHOLD_S=    # same — tune against real observed behavior, not a round-number guess
DOMAIN_GLOSSARY_DEFAULT=general     # general | medical | technical
```

## B.2 Data contracts

Canonical — copied verbatim from `setu-stage2-trd-implementation-plan.md`
§6. This copy and that one must never diverge; if a change is needed, update
both. These **extend** `backend/contracts.py` (append only) — Stage 1's
existing `Segment`, `CoverageStatus`, and `DecisionLogEntry` classes in that
file are unchanged and must not be edited.

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
    resolution: str                  # resolved gloss text or pose reference
    source: str                      # "session" | "domain"
    coverage_status: CoverageStatus   # reuse Stage 1's enum — do not define a second one

@dataclass
class SafeModeEvent:
    timestamp: float
    triggered_by: str       # "latency" | "error" | "queue_age"
    entering: bool           # True = entering safe mode, False = exiting
```

## B.3 File/folder structure — target end state after Stage 2

Existing Stage 1 files are marked **EXISTING**; only the named point in that
file is edited, nothing else. New files are marked **NEW**.

```
setu/
  .env                                   # EXISTING, extended per §B.1
  data/
    vocab/                               # EXISTING — not touched in Stage 2
    lexicon/                             # EXISTING — not touched in Stage 2
    glossary/
      general.md                          # NEW — T2.2
      medical.md                           # NEW — T2.2
      technical.md                          # NEW — T2.2
    fingerspelling/
      manifest.md                           # NEW — T2.6
      *.pose                                  # NEW — T2.6, extracted via T1.3's existing pipeline
  backend/
    contracts.py                          # EXISTING — T2.0 appends new types only
    orchestrator.py                       # EXISTING (T1.17) — edited at named points only, in T2.1/T2.2/T2.3
    recognition/
      extract.py                          # EXISTING (T1.3) — called by T2.6, not modified
      classifier.py                        # EXISTING — NOT touched in Stage 2
      capture.py                            # EXISTING (T1.18) — NOT touched in Stage 2
    waterfall/
      escalation.py                       # EXISTING (T1.14) — two named hooks filled in, T2.1/T2.2
      decision_log.py                      # EXISTING (T1.15) — reused, not modified
      session_glossary.py                   # NEW — T2.1
      domain_glossary.py                     # NEW — T2.2
    resilience/                            # NEW package
      safe_mode.py                          # NEW — T2.3
      stt_fallback.py                        # NEW — T2.4
      llm_fallback.py                         # NEW — T2.4
      dedup_batch.py                           # NEW — T2.5
    speech_to_sign/
      stt_input.py                         # EXISTING (T1.7) — routing checks added, T2.4 & T2.5
      reasoning.py                          # EXISTING (T1.8) — routing check added, T2.4
      gloss_lookup.py                        # EXISTING (T1.9) — fingerspelling fallthrough added, T2.6
      pose_smoothing.py                       # EXISTING — NOT touched in Stage 2
      fingerspell.py                            # NEW — T2.6
    sign_to_speech/
      reasoning.py                          # EXISTING (T1.5) — routing check added, T2.4
      tts_output.py                          # EXISTING — NOT touched in Stage 2
    collision/
      state_machine.py                      # EXISTING — NOT touched in Stage 2
    meeting_bridge/
      virtualcam.py                          # EXISTING — NOT touched in Stage 2
```

---

# §C. Tasks

## T2.0 · Stage 2 data contracts — `backend/contracts.py` · P0 · depends: Stage 1 complete

> **PROMPT**
> Goal: append `GlossaryHit`, `DomainContext`, and `SafeModeEvent` (§B.2,
> verbatim) to the existing `backend/contracts.py`.
> Files you may touch: `backend/contracts.py` (append only).
> Requirements: do not modify, reorder, or reformat any existing class in
> this file — `Segment`, `Direction`, `CoverageStatus`, `DecisionLogEntry`
> must be byte-identical to their Stage 1 form afterward.
> **VERIFY:** `python -c "from backend.contracts import Segment, CoverageStatus, DecisionLogEntry, GlossaryHit, DomainContext, SafeModeEvent; print('ok')"` — paste real output.

## T2.1 · Session-scope glossary — `backend/waterfall/session_glossary.py` · P0 · depends: T2.0

> **PROMPT**
> Goal: implement a `SessionGlossary` class — an in-memory dict scoped to one
> `Interpreter` call — that the escalation waterfall checks before falling
> through to the domain glossary or LLM reasoning, and that records a term's
> resolution once the waterfall resolves it via LLM reasoning so it isn't
> re-asked in the same call.
> Files you may touch: `backend/waterfall/session_glossary.py` (new);
> `backend/waterfall/escalation.py` (**edit only** the existing
> `session_glossary` hook — do not touch any other logic in this file);
> `backend/orchestrator.py` (**edit only** to construct one
> `SessionGlossary` instance per call and pass it into `Interpreter`'s
> waterfall calls).
> Requirements: no persistence of any kind — must not write to disk or any
> external store, and must be discarded when the call session ends. Every
> resolution recorded or retrieved must produce a `GlossaryHit` (T2.0) with
> `source="session"`.
> **VERIFY:** run one real term through the waterfall twice within the same
> simulated call; confirm the second occurrence resolves with zero
> additional Groq reasoning calls — paste real call-count evidence (e.g. a
> logged Groq call counter before and after the second occurrence).

## T2.2 · Domain glossary with visible hit/miss — `backend/waterfall/domain_glossary.py` + `data/glossary/*.md` · P0 · depends: T2.0

> **PROMPT**
> Goal: build curated domain glossaries (general, medical, technical) as
> manifest files (same content pattern as T1.2's vocabulary manifest),
> selected at session start via a human-supplied domain flag, and wire the
> waterfall's existing `domain_glossary` hook to check the selected
> glossary before LLM reasoning, reporting hit/miss through the existing
> `CoverageStatus` vocabulary.
> Files you may touch: `backend/waterfall/domain_glossary.py` (new);
> `data/glossary/general.md`, `data/glossary/medical.md`,
> `data/glossary/technical.md` (new content files); `backend/waterfall/
> escalation.py` (**edit only** the existing `domain_glossary` hook);
> `backend/orchestrator.py` (**edit only** to accept and pass through a
> domain selection at session construction).
> Requirements: the glossary content files are a human-curation task, same
> as T1.2 — keep each one small and reviewed, not auto-generated at build
> time. A miss must fall through to the next waterfall step completely
> unchanged — never substitute a wrong-domain resolution in place of a real
> miss (PRD FR-20). Every hit or miss must produce a `GlossaryHit` (T2.0)
> with `source="domain"`, or a clean fallthrough with no `GlossaryHit`
> fabricated for a miss.
> **VERIFY:** run one real term present only in the medical glossary with
> `domain=medical` selected and confirm a real hit is reported; run the
> identical term with `domain=general` selected and confirm a real
> miss/fallthrough — paste both real traces.

## T2.3 · Safe-mode state machine — `backend/resilience/safe_mode.py` · P0 · depends: T2.0

> **PROMPT**
> Goal: implement the safe-mode state machine: enter on a Groq API error, a
> latency spike, or a queue-age threshold being crossed; while active, stop
> starting new avatar animations, show the latest transcript/gloss with a
> visibly "unrendered" label, and play a concise TTS hold message; exit
> automatically once a subsequent real Groq call succeeds.
> Files you may touch: `backend/resilience/safe_mode.py` (new);
> `backend/orchestrator.py` (**edit only** to instantiate the state machine
> and check `is_active` at the two existing points where new avatar
> animation work is started and where Groq calls are made).
> Requirements: **before wiring the trigger conditions, verify in this
> session what real signal the currently-installed Groq Python SDK actually
> exposes for latency/error/rate-limit state** — do not assume a specific
> exception type or response header without checking the installed SDK's
> real behavior. Log every entry and exit as a `SafeModeEvent` (T2.0)
> through the existing decision log (T1.15) — do not build a second,
> separate logging path for this.
> **VERIFY:** induce one real failure (a deliberately invalid API key for
> one call, or a real rate-limit hit from sending requests past the known
> per-minute limit) and confirm safe mode enters automatically, is visible
> in the decision log, and exits automatically once a subsequent real call
> succeeds — paste the real state transition log.

## T2.4 · Local fallback stack: WhisperLiveKit + Ollama — `backend/resilience/stt_fallback.py`, `backend/resilience/llm_fallback.py` · P0 · depends: T2.3

> **PROMPT**
> Goal: implement the STT and LLM-reasoning fallback wrappers that safe mode
> (T2.3) routes to instead of Groq, and wire the existing Stage 1 call sites
> to check `safe_mode.is_active` and route accordingly.
> Files you may touch: `backend/resilience/stt_fallback.py`,
> `backend/resilience/llm_fallback.py` (new); `backend/speech_to_sign/
> stt_input.py` (**edit only** to add the routing check, no other change);
> `backend/sign_to_speech/reasoning.py`, `backend/speech_to_sign/
> reasoning.py` (**edit only** to add the same routing check in each).
> Requirements: `pip install whisperlivekit` (or per its repo's current
> instructions). **This package's CLI/flags were observed evolving fast in
> prior research — re-check its current README/flags in this session before
> writing any code against it; do not assume a previously-observed flag,
> including `voxtral-mlx`, is still current.** Use its MLX backend for the
> target Apple Silicon machine once confirmed. For Ollama, **verify its
> current Python client/API shape in this session before writing code
> against it.** Model choice is not decided — pick something that fits
> comfortably in the target machine's memory and **benchmark its real local
> latency on the actual machine before finalizing** anything in
> `SAFE_MODE_LATENCY_THRESHOLD_MS`; do not commit to a demo timing threshold
> based on an assumed number. Both fallback functions must return the exact
> same shape the Stage 1 Groq-calling code already expects (same
> transcript/completion return type), so routing is a drop-in swap, not a
> second code path with different downstream handling.
> **VERIFY:** with safe mode forced active (a temporary test flag is fine),
> run one real audio chunk through the STT fallback and paste the real
> transcript; run one real prompt through the LLM fallback and paste the
> real completion. Separately, paste the real measured latency of each on
> the actual target machine.

## T2.5 · Duplicate-ack suppression and batching — `backend/resilience/dedup_batch.py` · P1 · depends: none (parallel)

> **PROMPT**
> Goal: combine adjacent short utterances and suppress repeated filler
> acknowledgments ("okay, okay") in the Speech→Sign STT chunk stream, before
> segments reach the waterfall.
> Files you may touch: `backend/resilience/dedup_batch.py` (new);
> `backend/speech_to_sign/stt_input.py` (**edit only** to pass chunks
> through this filter before yielding them downstream).
> Requirements: this is explicitly **not** summarization. It must never
> compress, merge, or drop substantive content — names, numbers, questions,
> or any chunk the filler-classifier isn't confidently certain is pure
> acknowledgment must pass through unmerged. When uncertain, prefer passing
> content through over merging it (PRD FR-26).
> **VERIFY:** run one real scripted test transcript containing both repeated
> filler ("okay, okay, okay") and substantive content (a name, a number, a
> direct question) interleaved; paste the real batched output and confirm
> every substantive token survived unchanged while the filler collapsed.

## T2.6 · ISL fingerspelling handshape library — `data/fingerspelling/manifest.md`, `backend/speech_to_sign/fingerspell.py` · P0 · depends: none (content task, parallel to T2.1–T2.5), integrates with T1.9

> **PROMPT**
> Goal: close the fingerspelling gap found live in Stage 1 —
> `spoken-to-signed-translation`'s bundled lexicons don't include one for
> ISL (`ins` is absent; `ise` is Italian), and ISL's two-handed manual
> alphabet has no drop-in substitute — by building a real, human-recorded
> ISL fingerspelling handshape library and wiring it as the actual
> fingerspelling fallback for out-of-vocabulary terms.
> Files you may touch: `data/fingerspelling/manifest.md` (content,
> human-curated, same pattern as T1.2 — list each letter/handshape with
> notes on any dynamic ones); `data/fingerspelling/` (`.pose` files,
> extracted via T1.3's existing `extract.py` — call it, do not modify it);
> `backend/speech_to_sign/fingerspell.py` (new — looks up a term
> letter-by-letter against the manifest and concatenates the resulting
> `.pose` sequences); `backend/speech_to_sign/gloss_lookup.py` (**edit
> only**: when T1.9's `spoken-to-signed-translation` call reports no
> lexicon match and no language-backup match, call this module instead of
> immediately returning `UNMATCHED`).
> Requirements: **already checked in planning — do not re-search for a
> shortcut dataset.** `RealSign62/RealSign-Indian-Sign-Language-Dataset`,
> the IEEE DataPort ISL fingerspelling set, and `ayeshatasnim-h/
> Indian-Sign-Language-dataset` are static labeled images for classifier
> training, not `.pose` motion data; a 2025 continuous ISL fingerspelling
> corpus (Kirandevraj et al., ACL WSLP) is unaligned news-broadcast video
> with no extracted poses or confirmed usable license. Record the ~26
> handshapes personally, the same way T1.2's vocabulary was built, and
> extract via T1.3's existing pipeline unmodified. If a letter has a dynamic
> (moving) form in ISL — a real, documented distinction for at least some
> letters in prior ISL fingerspelling research — record it as a short clip,
> not a static pose, consistent with how the rest of the vocabulary already
> handles motion. Output must be tagged `FINGERSPELLING` in `CoverageStatus`
> — never silently folded into `LEXICON_HIT`, and never left as `UNMATCHED`
> when a real fingerspelled result exists.
> **VERIFY:** run one real deliberately out-of-vocabulary, out-of-domain-
> glossary term through the full pipeline and confirm it now resolves via
> real letter-by-letter fingerspelling with a `FINGERSPELLING` coverage tag,
> where it previously returned `UNMATCHED` — paste the real before/after
> coverage report.

---

# §D. Build order

Illustrative — adjust proportionally to the actual time available, but keep
T2.0 first and treat T2.3/T2.4 (safe mode + its fallback stack) as the
highest-value pair if time runs short, since that's the single named risk
this whole stage exists to close.

| Window | Tasks | Expected outcome by end of window |
|---|---|---|
| Hour 0–1 | T2.0 | Contracts extended; import verified; Stage 1's existing contracts confirmed untouched |
| Hour 1–3 | T2.1, T2.2 | Session glossary resolving repeats with zero additional Groq calls; domain glossary reporting a real hit and a real miss |
| Hour 3–6 | T2.3 | Safe mode entering and exiting correctly on a real induced Groq failure, visible in the decision log |
| Hour 6–9 | T2.4 | Local fallback stack (WhisperLiveKit, Ollama) wired, producing real transcripts/completions, with real measured latency on the target machine |
| Hour 9–10 | T2.5 | Duplicate-ack batching passing its real scripted-transcript test, with no substantive content dropped |
| Hour 10–13 | T2.6 | Fingerspelling handshapes recorded, extracted, and integrated — the `FINGERSPELLING` tier reachable end to end |
| Hour 13+ (buffer) | Rehearsal: induce a real Groq outage mid-scene and confirm the scene continues in safe mode | Demo scene 6 (safe-mode degradation), named but undemonstrable in Stage 1's own TRD, becomes real |

---

# §E. Final acceptance

1. ✅ A real term resolves via the session glossary on its second occurrence within one call, with zero additional Groq reasoning calls logged (T2.1).
2. ✅ A real domain-specific term reports a real hit when its matching domain is selected, and a real miss/fallthrough when a different domain is selected — both through the existing `CoverageStatus` vocabulary (T2.2).
3. ✅ A real induced Groq failure triggers safe mode automatically — visible "unrendered" label, hold message played, entry logged in the decision log (T2.3).
4. ✅ Safe mode exits automatically on real recovery, with no manual restart, and the exit is logged (T2.3).
5. ✅ While safe mode is active, a real audio chunk and a real prompt are both served by the local fallback stack, with real measured latency numbers recorded on the actual target machine (T2.4).
6. ✅ A real scripted transcript with interleaved filler and substantive content batches correctly — filler collapses, every name/number/question survives unchanged (T2.5).
7. ✅ A real, deliberately out-of-vocabulary, out-of-domain-glossary term is fingerspelled and tagged `FINGERSPELLING`, where it previously returned `UNMATCHED` (T2.6).
8. ✅ No task touched `data/vocab/`, `build_lexicon()`, the lexicon resolution pipeline, `backend/recognition/classifier.py`, `backend/recognition/capture.py`, or `backend/speech_to_sign/pose_smoothing.py`.
9. ✅ No task touched any Stage 1 file outside the one specific hook or routing check it was scoped to.
10. ✅ Every safe-mode transition and every glossary hit/miss appears in the existing decision log — no second, parallel logging path was introduced anywhere in this stage.
11. ✅ A real induced Groq outage, triggered mid-scene during a rehearsal run, is survived without a hard failure — demo scene 6 (safe-mode degradation) can be run live.
