#!/usr/bin/env python3
"""Measure the local fallback stack's real latency on this machine (T2.4, TNFR-7).

TNFR-7 is explicit that these numbers must be *measured and disclosed*, not
assumed from the components' general reputation, before Stage 2 is called
demo-stable. Safe mode's whole justification is that degrading to local is
better than failing outright — and that claim is only true if the local path
actually returns in a usable time. This script is what makes it checkable.

It measures both halves against a known ground truth:

- **STT** — a sentence is synthesized with Groq TTS (so the reference text is
  exact, not a human transcription), then transcribed by each candidate MLX
  Whisper model. Word error rate is reported alongside latency, because a fast
  transcript that loses the name and the number is not a usable fallback.
- **LLM** — the same two prompts Stage 1 actually sends (gloss generation and
  sentence reconstruction) are run against the Ollama model, and the output is
  checked against the same validators the real call sites use.

    .venv/bin/python scripts/benchmark_fallback.py
    .venv/bin/python scripts/benchmark_fallback.py --stt-models mlx-community/whisper-small.en-mlx

Results are written to `data/out/fallback_benchmark.json` and printed as the
table that goes in the acceptance record.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ordered cheapest-first. large-v3-turbo is the local twin of Stage 1's Groq
# STT model (whisper-large-v3-turbo), so it is the accuracy reference point
# rather than an aspiration.
DEFAULT_STT_MODELS = (
    "mlx-community/whisper-base.en-mlx",
    "mlx-community/whisper-small.en-mlx",
    "mlx-community/whisper-large-v3-turbo",
)

# Two sentences, chosen to carry exactly the content FR-26 says must survive:
# a proper name, a number, and a direct question.
BENCH_SENTENCES = (
    "Where is the hospital? My name is Ajay and I need help at gate number seven.",
    "Please sit down. The teacher will see you today at four o'clock.",
)

TARGET_SR = 16_000


@dataclass
class SttResult:
    model: str
    reference: str
    hypothesis: str
    audio_s: float
    latency_s: float
    real_time_factor: float
    word_error_rate: float


@dataclass
class LlmResult:
    model: str
    task: str
    prompt: str
    output: str
    latency_s: float
    valid: bool
    note: str = ""


def _normalise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard Levenshtein WER over word tokens."""
    ref, hyp = _normalise(reference), _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Full DP table; these sentences are short enough that the memory-efficient
    # variant would only obscure the code.
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)]) / len(ref)


def wav_to_float32(wav_bytes: bytes, target_sr: int = TARGET_SR) -> tuple[np.ndarray, float]:
    """Decode a WAV container to mono float32 at ``target_sr``.

    MLX Whisper expects 16 kHz mono float32 in [-1, 1]; Groq TTS returns 24 kHz.
    Resampling here rather than inside the fallback keeps the fallback's own
    latency measurement free of conversion cost it would not pay in production,
    where `PauseChunker` already yields 16 kHz frames.
    """
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    duration = len(audio) / sample_rate
    if sample_rate != target_sr:
        target_len = int(round(duration * target_sr))
        audio = np.interp(
            np.linspace(0.0, len(audio) - 1, target_len, dtype=np.float64),
            np.arange(len(audio), dtype=np.float64),
            audio.astype(np.float64),
        ).astype(np.float32)
    return audio, duration


def synthesize_reference(sentence: str) -> bytes:
    from backend.sign_to_speech.tts_output import synthesize

    return synthesize(sentence).audio


def benchmark_stt(models: tuple[str, ...], sentences: tuple[str, ...]) -> list[SttResult]:
    import mlx_whisper

    clips = []
    for sentence in sentences:
        audio, duration = wav_to_float32(synthesize_reference(sentence))
        clips.append((sentence, audio, duration))
        print(f"  reference audio: {duration:5.2f}s  {sentence[:50]}...")

    results: list[SttResult] = []
    for model in models:
        print(f"\n  {model}")
        # Warm the model once; the first call pays a download/compile cost that
        # safe mode would not pay per utterance in a running session.
        mlx_whisper.transcribe(clips[0][1], path_or_hf_repo=model, verbose=False)
        for sentence, audio, duration in clips:
            started = time.monotonic()
            out = mlx_whisper.transcribe(audio, path_or_hf_repo=model, verbose=False)
            latency = time.monotonic() - started
            hypothesis = (out.get("text") or "").strip()
            wer = word_error_rate(sentence, hypothesis)
            results.append(
                SttResult(
                    model=model,
                    reference=sentence,
                    hypothesis=hypothesis,
                    audio_s=round(duration, 3),
                    latency_s=round(latency, 3),
                    real_time_factor=round(latency / duration, 3),
                    word_error_rate=round(wer, 4),
                )
            )
            print(f"    {latency:6.2f}s  RTF {latency/duration:5.2f}  WER {wer:5.1%}  {hypothesis[:60]}")
    return results


def benchmark_llm(model: str) -> list[LlmResult]:
    from backend.resilience.llm_fallback import complete, LlmFallbackError
    from backend.speech_to_sign.reasoning import _validate as validate_gloss, SYSTEM_PROMPT as GLOSS_PROMPT
    from backend.sign_to_speech.reasoning import _validate as validate_sentence, SYSTEM_PROMPT as SENTENCE_PROMPT

    cases = [
        ("gloss", GLOSS_PROMPT, "Where is the hospital?", validate_gloss),
        ("reconstruct", SENTENCE_PROMPT, "Gloss: HOSPITAL WHERE", validate_sentence),
    ]

    results: list[LlmResult] = []
    for task, system, user, validator in cases:
        started = time.monotonic()
        note, valid, text = "", False, ""
        try:
            text = complete(user, system=system, model=model)
            latency = time.monotonic() - started
            try:
                validator(text, user)
                valid = True
            except Exception as exc:  # the real call sites' own boundary check
                note = f"failed Stage 1 validation: {exc}"
        except LlmFallbackError as exc:
            latency = time.monotonic() - started
            note = str(exc)
        results.append(
            LlmResult(
                model=model, task=task, prompt=user, output=text,
                latency_s=round(latency, 3), valid=valid, note=note,
            )
        )
        flag = "ok " if valid else "BAD"
        print(f"  {flag} {task:12s} {latency:6.2f}s  {text[:60]!r} {note}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-models", nargs="*", default=list(DEFAULT_STT_MODELS))
    parser.add_argument("--llm-model", default=None, help="defaults to OLLAMA_MODEL")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "out" / "fallback_benchmark.json")
    args = parser.parse_args()

    payload: dict = {"measured_at": time.time(), "machine": _machine()}

    if not args.skip_stt:
        print("STT — MLX Whisper (WhisperLiveKit's Apple Silicon engine)")
        payload["stt"] = [asdict(r) for r in benchmark_stt(tuple(args.stt_models), BENCH_SENTENCES)]

    if not args.skip_llm:
        # Read straight from the env, not through Stage 1's `get_settings()`:
        # `config.py` is not in any Stage 2 task's file scope, and the fallback
        # modules already resolve OLLAMA_MODEL themselves.
        from backend.resilience.llm_fallback import _model as ollama_model

        model = args.llm_model or ollama_model()
        print(f"\nLLM — Ollama {model}")
        payload["llm"] = [asdict(r) for r in benchmark_llm(model)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


def _machine() -> str:
    import platform

    return f"{platform.machine()} / {platform.platform()} / python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
