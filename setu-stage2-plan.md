# Setu — Stage 2: Resilience & Adaptive Behaviors — Planning Doc
### Lighter-weight than Stage 1 — intended to be improvised into full build instructions when Stage 1 is stable

> This is **not** a drift-proof, task-by-task build doc like Stage 1. It's a
> scoped brief: what Stage 2 covers, what it builds on top of (Stage 1's
> contracts, unchanged), which repos apply, and what's still open. Turn this
> into full `stage_build_instructions_template.md`-style tasks (PROMPT/VERIFY
> per task) once Stage 1 is demo-stable and you're ready to build this stage —
> don't start Stage 2 before then, per architecture v3 §12.

**Do not start this stage until Stage 1's §E Final Acceptance checklist is
fully green.** Everything here assumes Stage 1's contracts (`Segment`,
`CoverageStatus`, `DecisionLogEntry`, the `.pose` format via `pose-format`)
exist and are unchanged — reuse them, don't redefine.

---

## What this stage covers

Maps to architecture v3 §4.2 (Tier 1) and §12's P1 block:

1. **Session-scope glossary** — an in-memory dict of terms resolved during
   the current call, checked before falling through to the domain glossary
   or LLM reasoning. Discarded at call end, no persistence.
2. **Safe-mode degradation** — on latency spike, Groq API error, or queue age
   crossing a threshold, stop starting new avatar animations, show the latest
   transcript/gloss with an "unrendered" label, play a concise TTS hold
   message, resume automatically once recovered.
3. **Local fallback stack** — the actual mechanism safe-mode degrades *into*.
4. **Duplicate-ack suppression and batching** — combine adjacent short
   utterances, suppress repeated filler ("okay, okay"). Explicitly does
   **not** mean summarizing or dropping substantive content, names, numbers,
   or questions — architecture v3 is emphatic on this point, don't relax it
   for convenience.
5. **Domain glossary with visible hit/miss** — a small, curated,
   room-context-selected glossary (medical/technical/general), logged the
   same way T1.9's coverage status already is.

## Repo references for this stage — use these, don't rebuild

- **`QuentinFuxa/WhisperLiveKit`** (`pip install whisperlivekit` or per its
  repo instructions) — the local STT fallback. Confirmed in prior research to
  have a native MLX backend for Apple Silicon, invoked via
  `pip install -e ".[voxtral-mlx]"` then `wlk --backend voxtral-mlx`. **This
  project's CLI/flags were observed to be evolving fast in this session's
  research (qwen3-streaming, causal mode, voxtral-mlx were all found as
  current options) — re-check its current README/flags before building
  against it, don't assume the flags above are still current by the time this
  stage is built.**
- **Ollama** — local LLM fallback for the reasoning step when Groq is
  rate-limited or erroring. Model choice not yet decided; pick something
  small enough to run acceptably on the target laptop and verify its real
  local latency before committing to it in the safe-mode design.
- **`sign-language-processing/spoken-to-signed-translation`'s `--coverage-info`
  schema** (already in use since Stage 1 via T1.9/`CoverageStatus`) — reuse
  the identical four-tier status for the domain glossary's hit/miss reporting
  in item 5 above, rather than inventing a second status vocabulary.
- **Hardware caveat carried over from architecture v3 §11.6**: no component
  in this stage's local-fallback design (WhisperLiveKit's MLX backend,
  Ollama's local latency, `indic-parler-tts` if Stage 3's Tamil path is also
  in progress) has been independently benchmarked on the actual target
  machine. Do this before the stage's safe-mode timing thresholds are
  finalized, not after.

## What's genuinely open, to resolve when this stage is actually built

- Exact latency/queue-age thresholds that trigger safe mode — these should be
  tuned against Stage 1's actual observed real-world latency once it's
  running, not guessed in advance.
- Which Ollama model to run locally — a real choice that depends on
  benchmarking the actual laptop, not something to pre-decide here.
- Whether domain glossary context is supplied via a simple pre-call text box
  or a room-context selector UI — either is consistent with architecture v3's
  explicit rejection of Calendar/Docs OAuth (v3 §4.4); pick whichever is
  faster to build when this stage starts.

## Demo payoff

This stage is what makes demo scene 6 (architecture v3 §10, safe-mode
degradation) honestly demonstrable — that scene should not be attempted
in the Stage 1 demo, and this stage is a prerequisite for including it.
