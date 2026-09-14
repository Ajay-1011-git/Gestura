# Setu — Stage 3: Novelty & Differentiation Layer — Planning Doc
### Lighter-weight than Stage 1 — intended to be improvised into full build instructions when Stage 2 is stable

> Same caveat as Stage 2: this is a scoped brief, not a drift-proof
> task-by-task build doc. Improvise it into `stage_build_instructions_template.md`
> form when Stage 2 is demo-stable and you're ready to build this stage.

**Do not start this stage until Stage 2 is stable.** This is the layer that
moves the project's honest novelty rating (discussed earlier in this
project's planning) from a defensible 6/10 toward the higher end — it's real
differentiation work, but it's genuinely optional relative to having a
working demo at all. If time runs out, cutting from the bottom of this list
costs nothing to the core pitch.

---

## What this stage covers, in priority order

Maps to architecture v3 §5 and §12's P2 block, highest-value first:

1. **Agent-directed reprocessing (scoped version)** — the reasoning step can
   request a second recognition pass over the already-buffered landmark
   window (relaxed threshold or longer window) before escalating to human
   clarification. This is the single most novel piece in the whole project —
   build it before anything else in this stage if time is limited. **Build
   only the scoped version** (re-query on already-buffered data) — the full
   version (re-cropping raw video and re-detecting) is 1–2 days of real CV
   work for a rarely-triggered path and is explicitly not worth it here.
2. **Collaborative sign coinage** — when the waterfall bottoms out with no
   validated pose, invite the Deaf user to construct one live via the same
   webcam→pose-extraction capture path already built in Stage 1, save it into
   the session glossary. **Store the result via `pose-format`'s `.pose`
   format** (same package as Stage 1's T1.3) — this is the same format the
   rest of the pipeline already reads/writes, don't invent a second one for
   this feature specifically.
3. **Policy handshake, asymmetric transparency, data export** — cheap,
   roughly half a day combined:
   - one question at session start calibrating the escalation ladder's
     thresholds for that session;
   - full decision-log detail to the Deaf participant, a short plain-language
     line to the hearing side;
   - confirmed corrections/coined signs exportable via an explicit opt-in
     toggle, retention off by default.
4. **One hard-coded compositional pose modifier** — a single demo beat, not
   a general system: exactly one modifier (e.g. movement amplitude for
   "big/small") on exactly one pose library entry. Do not attempt a general
   compositional system in this stage.

## Optional stretch items (only if the above is fully stable with time to spare)

5. **Tamil TTS path** — optional per the project's scope decision (Malayalam
   dropped, Tamil optional, English mandatory). **Use `ai4bharat/indic-parler-tts`**,
   self-hosted, confirmed to cover both Tamil and English in the same model,
   no reference-audio/voice-cloning complexity (unlike `AI4Bharat/IndicF5`,
   which was checked and explicitly ruled out for this — it doesn't cover
   English and needs per-utterance reference audio plus real consent
   handling). Self-host via Apple MPS on the target laptop. **Its real latency
   on the actual machine was never benchmarked in this project's planning —
   test that before deciding this is demo-ready, not after.**
6. **Optional lip-sync/facial layer** — `met4citizen/TalkingHead`. Real,
   actively maintained, built specifically for Mixamo-compatible rigs (a
   better match for this project's actual avatar than most Kalidokit demos,
   which lean VRM). **Hard gate before starting any work here: open the
   avatar mesh in Blender and confirm it has ARKit and Oculus viseme blend
   shapes** — this is TalkingHead's specific lip-sync requirement, and it was
   never confirmed against the actual asset during planning. If the blend
   shapes aren't there, skip this item entirely without regret — it was never
   load-bearing for the core pitch. If pursued, it's layered on top of
   Kalidokit for face/mouth only — Kalidokit still owns all hand/finger
   retargeting, this item doesn't change that.
   - **Unverified bonus, not vetted, check before relying on it**:
     `met4citizen/HeadTTS` — a free, local, Kokoro-based TTS with viseme
     timing output, from the same author. Could pair with the item above if
     it's pursued. Nothing about it beyond a surface description was
     independently confirmed during this project's research.
7. **Shared bidirectional discourse state** — one rolling context object both
   directions read/write so references resolve across sides. Genuinely not
   hard to build, but hard to demo cleanly in a short live demo — build only
   if everything else in this document is done and stable.
8. **Boundary gate for medical/legal advice** — "I can help with logistics,
   not medical/legal advice." Cheap, ties directly into the project's existing
   honesty-about-scope positioning (architecture v3 §1.2).

## What NOT to build here, even if time allows

Per architecture v3 §4.4, carried forward without exception: real speaker
diarization, CV-based signer emotion/feedback detection, broad Calendar/Docs
OAuth, automatic topic-shift detection, live automatic language
code-switching detection, cross-session personalization, any form of neural
photoreal avatar rendering, and any general (non-hard-coded) compositional
sign generation system.

## Demo payoff

Items 1–4 are what unlock demo scenes that go beyond Stage 1's baseline and
are the direct answer if a judge asks "what's actually novel here" — see the
project's novelty discussion for how each maps to a specific defensible
claim. Items 5–8 are polish and reach; cutting them costs the project
nothing structurally.
