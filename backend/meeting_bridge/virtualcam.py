"""Meeting bridge: virtual camera and virtual microphone sinks (T1.16).

This is the layer that makes Gestura platform-agnostic (NFR-6). Nothing here
talks to Meet, Zoom or Teams — it writes to OS-level virtual devices that every
conferencing app already consumes as an ordinary webcam and microphone.

**Two separate devices, and only one of them comes from OBS.** OBS Virtual
Camera carries video only, and ``pyvirtualcam`` has no audio path at all. macOS
ships no virtual microphone, so the audio half needs a separate HAL driver —
BlackHole (free, MIT) is the standard choice. The build instructions describe
both sinks as if OBS provided them; verified on 2026-09-15, it does not.

Manual pre-steps, deliberately not automated for Stage 1:

1. Open OBS and click **Start Virtual Camera** once, approving the system
   extension in System Settings and restarting if prompted. That installs the
   camera extension; OBS need not be running afterwards.
2. ``brew install blackhole-2ch``, then set ``OBS_VIRTUALCAM_DEVICE`` or pass
   the device name explicitly.

**Checking it works means opening a consumer, not OBS.** Photo Booth, QuickTime's
New Movie Recording, or a browser camera test, with "OBS Virtual Camera"
selected there. OBS's own preview window shows OBS's *scene* — it is the other
end of the pipe — and will never show frames written by this module, whatever
this module is doing. That is worth stating plainly because the failure is
silent in both directions: `pyvirtualcam` reports a healthy 30fps whether or not
anything is consuming the device, so "the code says it sent 2700 frames" and
"nothing is visible" are entirely compatible and neither one diagnoses the other.

Confirmed on 2026-09-15: frames written here do reach Photo Booth with "OBS
Virtual Camera" selected. Whether OBS's own Virtual Camera can be running at the
same time was not established — if frames ever stop arriving, stopping it is the
first thing to rule out, but there is no evidence it has to be stopped.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Iterator

import numpy as np

VIRTUAL_AUDIO_HINTS = ("blackhole", "loopback", "soundflower", "vb-audio", "virtual")

OBS_SETUP_HINT = (
    "OBS Virtual Camera is not available. Open OBS, click 'Start Virtual "
    "Camera' to trigger installation, approve the system extension in System "
    "Settings > Privacy & Security, and restart if prompted."
)
AUDIO_SETUP_HINT = (
    "No virtual audio device found. OBS Virtual Camera is video-only and "
    "macOS has no built-in virtual microphone, so Gestura's speech cannot "
    "reach a call without one. Install one with: brew install blackhole-2ch"
)


class BridgeError(RuntimeError):
    """A virtual device is unavailable or could not be written to."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int


def list_audio_devices(*, output_only: bool = True) -> list[AudioDevice]:
    import sounddevice as sd

    devices: list[AudioDevice] = []
    for index, device in enumerate(sd.query_devices()):
        channels = device["max_output_channels" if output_only else "max_input_channels"]
        if channels > 0:
            devices.append(
                AudioDevice(
                    index=index,
                    name=device["name"],
                    channels=channels,
                    sample_rate=int(device["default_samplerate"]),
                )
            )
    return devices


def find_virtual_audio_device(preferred: str | None = None) -> AudioDevice:
    """Locate the virtual audio device Gestura's speech should be written to."""
    devices = list_audio_devices(output_only=True)
    if preferred:
        for device in devices:
            if preferred.lower() in device.name.lower():
                return device
        raise BridgeError(
            f"requested audio device {preferred!r} not found. Available: "
            f"{[d.name for d in devices]}"
        )
    for device in devices:
        if any(hint in device.name.lower() for hint in VIRTUAL_AUDIO_HINTS):
            return device
    raise BridgeError(f"{AUDIO_SETUP_HINT}\nAvailable outputs: {[d.name for d in devices]}")


class VirtualCamera:
    """Sends rendered avatar frames to the OBS virtual camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.width, self.height, self.fps = width, height, fps
        self._camera = None

    def __enter__(self) -> "VirtualCamera":
        import pyvirtualcam

        try:
            self._camera = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=self.fps
            )
        except RuntimeError as exc:
            raise BridgeError(f"{OBS_SETUP_HINT}\nUnderlying error: {exc}") from exc
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    @property
    def device(self) -> str:
        if self._camera is None:
            raise BridgeError("camera not open — use VirtualCamera as a context manager")
        return str(self._camera.device)

    def send(self, frame: np.ndarray) -> None:
        """Send one RGB uint8 frame of shape ``(height, width, 3)``."""
        if self._camera is None:
            raise BridgeError("camera not open — use VirtualCamera as a context manager")
        expected = (self.height, self.width, 3)
        if frame.shape != expected:
            raise BridgeError(f"frame shape {frame.shape} does not match {expected}")
        self._camera.send(np.ascontiguousarray(frame, dtype=np.uint8))
        self._camera.sleep_until_next_frame()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to ``(int16 samples, sample_rate)``.

    Sample count is derived from the data chunk's real length, not the header's
    frame count: Groq's TTS returns a streaming WAV whose header declares a
    placeholder length (observed as 89,478 seconds for ~2.6s of audio), so
    trusting the header yields nonsense durations.
    """
    with wave.open(io.BytesIO(data), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


class VirtualMicrophone:
    """Plays Gestura's synthesized speech into a virtual audio device.

    A conferencing app selecting that device as its microphone receives the
    audio as though it were spoken into a real one (FR-4).
    """

    def __init__(self, device: AudioDevice | None = None, preferred: str | None = None) -> None:
        self.device = device or find_virtual_audio_device(preferred)

    def play_wav(self, data: bytes, *, blocking: bool = True) -> float:
        """Play WAV bytes to the virtual device. Returns duration in seconds."""
        import sounddevice as sd

        samples, sample_rate = decode_wav(data)
        duration = len(samples) / sample_rate
        try:
            sd.play(samples, samplerate=sample_rate, device=self.device.index, blocking=blocking)
        except Exception as exc:
            raise BridgeError(f"playback to {self.device.name!r} failed: {exc}") from exc
        return duration


def describe_bridge() -> dict[str, object]:
    """Report which halves of the bridge are actually available right now."""
    status: dict[str, object] = {}
    try:
        import pyvirtualcam

        with pyvirtualcam.Camera(width=320, height=240, fps=30) as cam:
            status["video"] = {"available": True, "device": str(cam.device), "backend": cam.backend}
    except Exception as exc:
        status["video"] = {"available": False, "reason": str(exc)[:200]}

    try:
        device = find_virtual_audio_device()
        status["audio"] = {"available": True, "device": device.name, "index": device.index}
    except Exception as exc:
        status["audio"] = {"available": False, "reason": str(exc)[:200]}

    return status
