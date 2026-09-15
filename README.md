# Setu

A real-time, bidirectional Indian Sign Language ↔ spoken English interpreter
that joins ordinary video calls as a virtual participant through OBS, with no
platform SDK integration.

Two directions share one decision structure. **Sign → Speech** recognises signs
from a webcam and speaks reconstructed English. **Speech → Sign** transcribes
speech, converts it to ISL gloss, and renders it on a rigged 3D avatar. Both run
through the same escalation waterfall, which decides *translate, clarify or
refuse* — and only `TRANSLATE` produces output.

The governing rule, in both stages: **no stage may silently convert uncertainty
into confident-looking output**, and (Stage 2) **no stage may silently convert
unavailability into failure.** A sign the system cannot validate is refused, not
guessed; a cloud dependency that falters degrades visibly, not silently.

Setu is not a certified interpreter and makes no claim of general ISL fluency.

---

## What actually works today

| Capability | State |
|---|---|
| Sign → Speech recognition | 40-word vocabulary, learned classifier (~510k params), 76.1% held-out |
| Speech → Sign rendering | 16 signs with validated poses, ISLRTC-sourced |
| Fingerspelling | 26-letter ISL manual alphabet — out-of-vocabulary terms are spelled, not refused |
| Session / domain glossary | General, medical, technical; repeats resolve with zero extra cloud calls |
| Safe mode | Automatic degradation to a local STT/LLM stack when Groq errors, slows or rate-limits |
| Duplicate-ack batching | Filler collapses; names, numbers and questions survive verbatim |

Recognition covers 40 words and rendering covers 16 — a known, deliberate
asymmetry, documented as R-10 and out of scope for Stage 2.

---

## Setup

### Requirements

- **Python 3.12.** Not 3.13 — `pose-format` needs the legacy
  `mediapipe.python.solutions` API (`mediapipe<0.10.30`), and legacy MediaPipe
  ships no 3.13 wheels.
- **Node 18+** for the browser-side avatar.
- **macOS on Apple Silicon** for the local fallback stack (MLX). Everything else
  is platform-independent; without MLX, safe mode degrades to a disclosed pause
  rather than a local continuation.
- A free **Groq** API key — <https://console.groq.com>.
- **[Ollama](https://ollama.com)** installed natively, for the LLM fallback.
- **OBS** with Virtual Camera, only for the meeting-bridge path.

### Install

```bash
# Python
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install 'whisperlivekit[mlx-whisper]'   # local STT fallback (Apple Silicon)

# Local LLM fallback
ollama pull qwen3:4b

# Frontend
npm install
```

### Configure

```bash
cp .env.example .env
# then fill in GROQ_API_KEY — every other value has a working default
```

### Fetch the data a clean checkout does not carry

The extracted `.pose` files are committed; the source video they came from is
not, because it is large and re-downloadable.

```bash
# 40-word vocabulary (only needed to rebuild or extend recognition)
.venv/bin/python scripts/fetch_vocabulary.py --all

# 26-letter fingerspelling library (committed, but rebuildable)
.venv/bin/python scripts/build_fingerspelling.py
```

### Run

```bash
.venv/bin/python scripts/run_interpreter.py     # the live bidirectional loop
npm run dev                                      # the avatar + decision-log panel
```

---

## Verifying it

Every claim above has a command behind it. These run against real services and
real data — none of them mock the thing they are testing.

```bash
.venv/bin/python scripts/verify_stage1.py        # Stage 1 acceptance
.venv/bin/python scripts/verify_stage2.py        # Stage 2 acceptance
.venv/bin/python scripts/benchmark_fallback.py   # real local-fallback latency (TNFR-7)
npm run typecheck
```

`benchmark_fallback.py` writes `data/out/fallback_benchmark.json`. Safe mode's
timing thresholds are tuned against those measured numbers, not against
assumptions about how fast the local models "should" be.

---

## Layout

```
backend/
  contracts.py            the shared data contracts; imported everywhere, depends on nothing
  orchestrator.py         Interpreter — the one integration point both directions run through
  recognition/            webcam capture, pose extraction, the learned classifier
  speech_to_sign/         STT, gloss generation, lexicon lookup, fingerspelling, avatar sequencing
  sign_to_speech/         sentence reconstruction, TTS
  waterfall/              the escalation ladder, the decision log, session + domain glossaries
  resilience/             safe mode, the local fallback stack, duplicate-ack batching
  collision/              who-speaks-when between the two directions
  meeting_bridge/         OBS virtual camera and microphone
frontend/
  avatar/                 Three.js + Kalidokit retargeting, hand anatomy, handshape solving
  decision_log_panel/     the live Observe/Decide/Action panel
data/
  vocab/                  40-word recognition corpus, extracted poses
  lexicon/                16-sign render lexicon (ISLRTC)
  fingerspelling/         26-letter ISL manual alphabet
  glossary/               curated general / medical / technical vocabularies
scripts/                  build, verification and benchmark entry points
```

---

## Documents

`setu-stage1-*.md` and `setu-stage2-*.md` carry the PRD, technical plan and
build instructions for each stage. `GesturaProjectArchitecture.md` is the
architecture of record — where it and a stage document disagree, it wins.
`setu-stage3-plan.md` is a scoped brief, not a build doc.

---

## Credits and licensing

The project's own code carries no license grant yet. Third-party assets it
depends on do, and those terms apply to anything built on them:

- **Avatar** — "Pixar style girl (rigging)" by
  [Black vizual](https://sketchfab.com/Black_vizual),
  [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/). Credit required;
  commercial use permitted. Shipped as `assets/avatar.glb`.
- **Fingerspelling handshapes** — derived from
  [`kirandevraj/ISL-Fingerspelling`](https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling)
  (Kirandevraj et al., ACL WSLP 2025), **CC-BY-NC-4.0 — non-commercial**. This
  is the most restrictive term in the project: `data/fingerspelling/` may not be
  used commercially without separate permission.
- **Recognition vocabulary** — `vidit031/isl-isolated-40words`, aggregated from
  ISL500, INCLUDE, CISLR and the ISLRTC dictionary.
- **Render lexicon** — ISLRTC dictionary video.
