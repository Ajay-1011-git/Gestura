"""Local STT fallback, routed to while safe mode is active (T2.4).

**Which engine, and why not the whole of WhisperLiveKit.** The plan named
`QuentinFuxa/WhisperLiveKit` for its Apple Silicon backend. Checked against the
installed package (whisperlivekit 0.2.26, 2026-09-15): its current `--backend`
options are `auto`, `mlx-whisper`, `faster-whisper`, `whisper`, `openai-api`,
`funasr`, `voxtral-mlx`, `voxtral`, `qwen3-vllm`, `qwen3-vllm-metal`,
`qwen3-streaming` and `canary` — so `voxtral-mlx` is still real, but
`mlx-whisper` is the Apple Silicon Whisper path and the closer match to Stage
1's `whisper-large-v3-turbo`.

The package's own Python entry point, `transcribe_audio()`, is a **websocket
client**: it feeds audio to a running WhisperLiveKit server at
`ws://localhost:8000/asr`. That is the wrong shape here for two reasons. Stage
1 already owns voice-activity chunking — `PauseChunker` yields complete
utterances, so WhisperLiveKit's streaming and LocalAgreement machinery would be
solving a problem that is already solved. And requiring a separate server
process and an open port puts a new failure mode *inside* the path whose entire
job is to keep working when things fail.

So this calls `mlx_whisper` directly — the same MLX engine WhisperLiveKit's
`mlx-whisper` backend drives, installed as part of `whisperlivekit[mlx-whisper]`
— in-process, no server, no port. The dependency the plan named is installed
and its engine is what runs; only the redundant transport layer is skipped.

**Drop-in shape is a hard requirement.** This returns T1.7's `Transcript`
exactly, so `stt_input.py` routes without branching on which path served it and
nothing downstream has to know. A fallback with a different return type would
be a second code path, not a fallback.
"""

from __future__ import annotations

import io
import os
import time
import wave
from functools import lru_cache
from typing import Any

import numpy as np

from backend.speech_to_sign.stt_input import SAMPLE_RATE, Transcript, TranscriptionError

# Chosen by measurement on this machine, not by size. Real numbers from
# `scripts/benchmark_fallback.py` over 6.6s and 4.6s of speech carrying a name,
# a number and a question (recorded in `data/out/fallback_benchmark.json`):
#
#   base.en-mlx        0.05-0.06s   RTF 0.01   WER 6.2-8.3%
#   small.en-mlx       0.14-0.17s   RTF 0.03   WER 0.0%     <- default
#   large-v3-turbo     0.69s        RTF 0.10   WER 6.2-8.3%
#
# small.en wins outright: perfect transcription, and four times faster than the
# large model. The larger model is not more accurate here — its errors and
# base.en's are the same kind, writing "7" and "4 o'clock" where the reference
# says "seven" and "four o'clock". That is a formatting difference rather than a
# misheard word, and it costs nothing downstream, but it is what the WER column
# is counting. Every model kept the proper noun.
DEFAULT_MODEL = "mlx-community/whisper-small.en-mlx"

MAX_TRANSCRIPT_CHARS = 2_000


class SttFallbackError(TranscriptionError):
    """The local STT fallback could not produce a transcript."""


def _model() -> str:
    return os.environ.get("WHISPERLIVEKIT_BACKEND", "").strip() or DEFAULT_MODEL


@lru_cache(maxsize=1)
def _engine() -> Any:
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SttFallbackError(
            "mlx_whisper is not installed — run "
            "`uv pip install 'whisperlivekit[mlx-whisper]'`. Safe mode falls "
            "back to a disclosed pause without it (TRD §8)."
        ) from exc
    return mlx_whisper


def wav_bytes_to_float32(wav: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV container to mono float32, as MLX Whisper expects."""
    with wave.open(io.BytesIO(wav)) as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _as_float32(audio: np.ndarray | bytes) -> tuple[np.ndarray, float]:
    """Normalise either accepted input shape to (float32 mono, duration_s)."""
    if isinstance(audio, bytes):
        samples, sample_rate = wav_bytes_to_float32(audio)
    else:
        samples = audio.astype(np.float32)
        # PauseChunker yields int16 PCM; MLX Whisper wants float32 in [-1, 1].
        # Scale by magnitude rather than dtype, so an already-normalised float
        # array passed straight in is not divided a second time.
        if np.abs(samples).max(initial=0.0) > 1.5:
            samples = samples / 32768.0
        sample_rate = SAMPLE_RATE
    if samples.size == 0:
        raise SttFallbackError("empty audio supplied to the local STT fallback")
    return samples, len(samples) / float(sample_rate)


def transcribe(audio: np.ndarray | bytes, *, model: str | None = None) -> Transcript:
    """Transcribe one utterance locally. Same return shape as T1.7's Groq call.

    Validated at the boundary before being trusted downstream, exactly as the
    Groq transcript already is: a local model is no more inherently trustworthy
    than a remote one, and a silent empty string would reach the waterfall as a
    real utterance.
    """
    name = model or _model()
    samples, duration = _as_float32(audio)

    started = time.monotonic()
    try:
        result = _engine().transcribe(samples, path_or_hf_repo=name, verbose=False)
    except SttFallbackError:
        raise
    except Exception as exc:
        raise SttFallbackError(f"local STT failed on {duration:.1f}s of audio: {exc}") from exc

    text = (result.get("text") if isinstance(result, dict) else None) or ""
    text = text.strip()
    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise SttFallbackError(
            f"local STT returned {len(text)} chars for {duration:.1f}s of audio — "
            f"refusing a runaway transcript"
        )

    return Transcript(
        text=text,
        duration_s=duration,
        latency_s=time.monotonic() - started,
        model=f"mlx-whisper/{name}",
    )


def warm_up(model: str | None = None) -> float:
    """Load the model once so the first real utterance is not the slow one.

    Worth doing at session start rather than on entry to safe mode: paying a
    cold-start cost at the exact moment Groq is already failing is the worst
    possible time for it.
    """
    started = time.monotonic()
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), model=model)
    return time.monotonic() - started


def is_available() -> bool:
    """Whether the local STT path can run at all — checked before promising it."""
    try:
        _engine()
        return True
    except SttFallbackError:
        return False
