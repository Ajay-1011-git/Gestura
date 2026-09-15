#!/usr/bin/env python3
"""Stream rendered avatar frames to the OBS virtual camera, with optional speech.

    node scripts/render_frames.mjs /tmp/frames        # render once
    .venv/bin/python scripts/stream_avatar.py /tmp/frames --loops 3
    .venv/bin/python scripts/stream_avatar.py /tmp/frames --say "Hello. Please sit."

Any conferencing app that selects "OBS Virtual Camera" as its webcam sees the
avatar; one that selects the virtual audio device (BlackHole) as its microphone
hears the speech. That split is not a simplification — OBS Virtual Camera
carries no audio at all, so the two halves genuinely come from different devices.

Loading every frame into memory first is deliberate: decoding a PNG mid-stream
costs more than a frame interval, and a camera that misses its deadline shows a
visible hitch rather than a dropped frame.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.meeting_bridge.virtualcam import (  # noqa: E402
    BridgeError, VirtualCamera, VirtualMicrophone,
)


def load_frames(directory: Path) -> list[np.ndarray]:
    from PIL import Image

    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNG frames in {directory} — run scripts/render_frames.mjs first")
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]


def test_pattern(width: int = 1280, height: int = 720, frames: int = 90) -> list[np.ndarray]:
    """A pattern nothing else could be mistaken for.

    Worth having as a first step: it separates "the camera path does not work"
    from "the avatar did not render", which otherwise look identical from the
    far end of a virtual device.
    """
    base = np.zeros((height, width, 3), np.uint8)
    base[:, :, 0] = 255           # magenta
    base[:, :, 2] = 255
    out = []
    for i in range(frames):
        frame = base.copy()
        x = int((i / frames) * (width - 220))
        frame[height // 3: height // 3 + 140, x:x + 220] = 255
        out.append(frame)
    return out


def speak_async(text: str) -> threading.Thread | None:
    """Synthesize and play in the background so speech overlaps the animation."""
    from backend.sign_to_speech.tts_output import TTSError, synthesize

    try:
        utterance = synthesize(text)
    except TTSError as exc:
        print(f"  speech skipped: {exc}")
        return None

    microphone = VirtualMicrophone()
    print(f"  speaking {text!r} -> {microphone.device.name!r} ({utterance.duration_s:.2f}s)")
    thread = threading.Thread(
        target=microphone.play_wav, args=(utterance.audio,), kwargs={"blocking": True},
        daemon=True,
    )
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path, nargs="?", default=None,
                        help="directory of rendered PNG frames")
    parser.add_argument("--pattern", action="store_true",
                        help="stream an unmistakable test pattern instead of frames, "
                             "to prove the camera path before involving the avatar")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loops", type=int, default=1, help="0 means loop until interrupted")
    parser.add_argument("--say", default=None, help="speak this into the virtual microphone")
    args = parser.parse_args()

    if args.pattern or args.frames is None:
        frames = test_pattern()
        print("streaming a test pattern — magenta with a moving white bar.")
        print("Look in a CONSUMER app (Photo Booth, QuickTime > New Movie Recording)")
        print("with 'OBS Virtual Camera' selected there — NOT in OBS itself, whose")
        print("preview shows OBS's own scene and never these frames.")
        print("Keep looking while this runs: it exits when the loops finish, and an")
        print("idle device shows nothing.")
    else:
        frames = load_frames(args.frames)
    height, width = frames[0].shape[:2]
    print(f"{len(frames)} frames at {width}x{height}, {args.fps:g}fps "
          f"({len(frames) / args.fps:.2f}s per loop)")

    speech = speak_async(args.say) if args.say else None

    sent = 0
    started = time.monotonic()
    try:
        with VirtualCamera(width=width, height=height, fps=args.fps) as camera:
            print(f"streaming to {camera.device!r} — Ctrl-C to stop")
            loop = 0
            while args.loops == 0 or loop < args.loops:
                for frame in frames:
                    camera.send(frame)
                    sent += 1
                loop += 1
    except BridgeError as exc:
        print(f"bridge error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped")

    elapsed = time.monotonic() - started
    print(f"sent {sent} frames in {elapsed:.2f}s ({sent / elapsed:.1f}fps actual)")
    if speech is not None:
        speech.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
