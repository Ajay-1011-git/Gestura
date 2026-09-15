"""Speech->Sign: pause-chunked microphone capture and Groq Whisper STT (T1.7).

Chunking is driven by voice activity, never by a fixed window and never
per-word (FR-5). That is a rate-limit requirement, not a stylistic one: Groq's
free STT tier was measured at 2,000 requests on this key, so a per-word or
per-second caller would exhaust the budget during a single conversation
(NFR-2).

Voice activity uses a calibrated RMS energy gate with hysteresis rather than a
neural VAD. An energy gate has no extra dependency, runs at negligible cost, and
is adequate for deciding *where a pause is* — which is all the chunker needs.
It is not trying to decide whether a sound is speech.
"""

from __future__ import annotations

import io
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from groq import APIConnectionError, APIStatusError, GroqError, RateLimitError

from backend.config import STT_MODEL, get_client

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# An utterance opens after this much continuous voice, and closes after this
# much continuous silence. The silence window is the "natural pause" of FR-5 —
# long enough not to split mid-sentence, short enough to stay conversational.
SPEECH_ONSET_FRAMES = 3       # ~90ms
SILENCE_CLOSE_FRAMES = 20     # ~600ms
PREROLL_FRAMES = 5            # ~150ms kept before onset so no word is clipped

MIN_UTTERANCE_S = 0.4         # shorter than this is a cough or a click
MAX_UTTERANCE_S = 25.0        # hard stop; also keeps every request well under
                              # Whisper's 25MB file cap
NOISE_CALIBRATION_S = 0.6
ENERGY_MARGIN = 3.0           # speech must exceed noise floor by this factor


class TranscriptionError(RuntimeError):
    """Audio could not be transcribed."""


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_s: float
    latency_s: float
    model: str


def pcm_to_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap int16 mono PCM as a WAV container for upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return buffer.getvalue()


def frame_energy(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


class PauseChunker:
    """Splits a stream of audio frames into utterances bounded by pauses.

    Feed 30ms int16 frames to :meth:`push`; it returns a completed utterance
    whenever a pause closes one, otherwise ``None``.
    """

    def __init__(self, *, noise_floor: float = 0.0) -> None:
        self.noise_floor = noise_floor
        self._preroll: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
        self._voiced: list[np.ndarray] = []
        self._run_voice = 0
        self._run_silence = 0
        self._open = False

    @property
    def threshold(self) -> float:
        return max(self.noise_floor * ENERGY_MARGIN, 120.0)

    def calibrate(self, frames: list[np.ndarray]) -> None:
        """Seed the noise floor from an initial window.

        Uses a low percentile, not the median: if the speaker is already talking
        when calibration runs, a median lands on speech energy and the gate ends
        up set above the very speech it is meant to detect.
        """
        if frames:
            self.noise_floor = float(
                np.percentile([frame_energy(f) for f in frames], 20)
            )

    def _track_noise(self, energy: float) -> None:
        """Adapt the noise floor continuously.

        Falls quickly toward quieter frames and rises only slowly, so a bad
        initial calibration self-corrects within a second instead of deafening
        the chunker for the whole session.
        """
        if self.noise_floor <= 0.0:
            self.noise_floor = energy
        elif energy < self.noise_floor:
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * energy
        else:
            self.noise_floor = 0.995 * self.noise_floor + 0.005 * energy

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        energy = frame_energy(frame)
        is_voice = energy > self.threshold
        if not is_voice or not self._open:
            self._track_noise(energy)

        if not self._open:
            self._preroll.append(frame)
            self._run_voice = self._run_voice + 1 if is_voice else 0
            if self._run_voice >= SPEECH_ONSET_FRAMES:
                self._open = True
                self._voiced = list(self._preroll)
                self._preroll.clear()
                self._run_silence = 0
            return None

        self._voiced.append(frame)
        self._run_silence = 0 if is_voice else self._run_silence + 1

        too_long = len(self._voiced) * FRAME_MS / 1000 >= MAX_UTTERANCE_S
        if self._run_silence >= SILENCE_CLOSE_FRAMES or too_long:
            return self._close()
        return None

    def flush(self) -> np.ndarray | None:
        return self._close() if self._open else None

    def _close(self) -> np.ndarray | None:
        audio = np.concatenate(self._voiced) if self._voiced else None
        self._open = False
        self._voiced = []
        self._run_voice = self._run_silence = 0
        self._preroll.clear()
        if audio is None or len(audio) / SAMPLE_RATE < MIN_UTTERANCE_S:
            return None
        return audio


def transcribe(audio: np.ndarray | bytes, *, model: str = STT_MODEL) -> Transcript:
    """Transcribe one utterance via Groq Whisper, or locally while degraded.

    Stage 2 (T2.4) routing check. The fallback returns this same `Transcript`,
    so nothing downstream branches on which path served the audio — only the
    `model` field differs, and that is there so the decision log can say so.
    """
    from backend.resilience import safe_mode

    if safe_mode.is_active():
        from backend.resilience import stt_fallback

        return stt_fallback.transcribe(audio)

    wav = pcm_to_wav(audio) if isinstance(audio, np.ndarray) else audio
    duration = (len(wav) - 44) / (SAMPLE_RATE * 2)
    started = time.monotonic()
    try:
        response = get_client().audio.transcriptions.create(
            file=("utterance.wav", wav), model=model
        )
    except RateLimitError as exc:
        raise TranscriptionError(
            f"Groq STT rate limit reached — Stage 1 has no local fallback by "
            f"design (TRD §8); WhisperLiveKit safe-mode is Stage 2: {exc}"
        ) from exc
    except (APIConnectionError, APIStatusError, GroqError) as exc:
        raise TranscriptionError(f"Groq transcription failed: {exc}") from exc

    text = (getattr(response, "text", "") or "").strip()
    return Transcript(
        text=text,
        duration_s=duration,
        latency_s=time.monotonic() - started,
        model=model,
    )


def stream_transcripts(
    *,
    device: int | None = None,
    should_listen: Callable[[], bool] = lambda: True,
) -> Iterator[Transcript]:
    """Capture the microphone and yield one `Transcript` per detected utterance.

    ``should_listen`` is the gate T1.13 wires to ``SETU_TTS_ACTIVE`` so Gestura's
    own synthesized speech is never captured and transcribed as if it were the
    hearing participant (FR-13).
    """
    import sounddevice as sd  # imported lazily so the module works without audio hardware

    from backend.resilience.dedup_batch import Chunk, DedupBatcher

    chunker = PauseChunker()
    # Stage 2 (T2.5). Filler is collapsed here, before a segment reaches the
    # waterfall or costs a reasoning token — not generated and discarded later.
    batcher = DedupBatcher()
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMPLES, device=device,
    ) as stream:
        calibration: list[np.ndarray] = []
        needed = int(NOISE_CALIBRATION_S * 1000 / FRAME_MS)
        while len(calibration) < needed:
            block, _ = stream.read(FRAME_SAMPLES)
            calibration.append(block[:, 0])
        chunker.calibrate(calibration)

        while True:
            block, _ = stream.read(FRAME_SAMPLES)
            if not should_listen():
                continue
            utterance = chunker.push(block[:, 0])
            if utterance is None:
                continue
            transcript = transcribe(utterance)
            if not transcript.text:
                continue
            for kept in batcher.push(Chunk(text=transcript.text, timestamp=time.time())):
                # Ordering is preserved and only pure filler is ever dropped, so
                # the transcript carried forward is the batched text with the
                # original's timing and model attribution intact.
                yield Transcript(
                    text=kept.text, duration_s=transcript.duration_s,
                    latency_s=transcript.latency_s, model=transcript.model,
                )
