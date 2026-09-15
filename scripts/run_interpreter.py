#!/usr/bin/env python3
"""Run Gestura live — the loop the components were built for.

    .venv/bin/python scripts/run_interpreter.py --sign-to-speech
    .venv/bin/python scripts/run_interpreter.py --speech-to-sign "Hello, please sit down."
    .venv/bin/python scripts/run_interpreter.py --speech-to-sign - < lines.txt
    .venv/bin/python scripts/run_interpreter.py --dry-run --clip data/vocab/HELLO/*.pose

`--sign-to-speech` opens the webcam, segments signing by motion, recognises each
segment, runs it through the escalation waterfall and speaks the result into the
virtual microphone. macOS will ask for camera permission the first time, and the
terminal must have it: System Settings > Privacy & Security > Camera.

`--dry-run` feeds a `.pose` file instead of a camera, so the whole path can be
exercised without hardware or a microphone — useful for confirming a change did
not break the loop before standing in front of it.

Every decision goes to the same log the panel renders, printed after each turn.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.orchestrator import Interpreter  # noqa: E402
from backend.waterfall.escalation import Action  # noqa: E402

ARROW = {Action.TRANSLATE: "->", Action.CLARIFY: "??", Action.REFUSE: "!!"}


def build(speaking: bool, rendering: bool) -> Interpreter:
    speak = render = None
    if speaking:
        from backend.meeting_bridge.virtualcam import BridgeError, VirtualMicrophone
        from backend.sign_to_speech.tts_output import TTSError, speak as tts_speak

        try:
            microphone = VirtualMicrophone()
            print(f"speech -> {microphone.device.name!r}")

            def speak(text: str) -> None:
                try:
                    tts_speak(text, microphone=microphone)
                except TTSError as exc:
                    print(f"  [speech failed: {exc}]")
        except BridgeError as exc:
            print(f"no virtual microphone ({exc}); printing instead of speaking")

    if rendering:
        out = ROOT / "data" / "out"
        out.mkdir(parents=True, exist_ok=True)

        def render(pose, timeline) -> None:
            import json
            with (out / "live_sequence.pose").open("wb") as handle:
                pose.write(handle)
            (out / "live_timeline.json").write_text(json.dumps(timeline, indent=2))
            print(f"  avatar sequence -> data/out/live_sequence.pose "
                  f"({len(timeline)} signs)")

    return Interpreter(speak=speak, render=render)


def report(outcome, interpreter: Interpreter) -> None:
    mark = ARROW.get(outcome.action, "  ")
    print(f"  {mark} {outcome.action.value.upper()}", end="")
    if outcome.spoken:
        print(f"  {outcome.spoken!r}" + ("   [UNCERTAIN]" if outcome.uncertain else ""))
    elif outcome.question:
        print(f"  asks: {outcome.question!r}")
    elif outcome.gloss:
        print(f"  gloss {outcome.gloss!r}")
    else:
        print(f"  {outcome.detail}")
    if outcome.coverage:
        unmatched = [c.gloss for c in outcome.coverage if c.status.value != "lexicon_hit"]
        if unmatched:
            print(f"     no validated sign for: {', '.join(unmatched)}")
    print("\n" + interpreter.trace(limit=6) + "\n")


def run_sign_to_speech(interpreter: Interpreter, args) -> int:
    from backend.recognition.capture import (
        CaptureError, SegmenterConfig, pose_from_segment, segments_from_camera,
    )

    print("watching for signing — Ctrl-C to stop\n")
    config = SegmenterConfig(motion_threshold=args.motion)
    shown = [0.0]

    def on_state(state, motion):
        now = time.time()
        if now - shown[0] > 0.5:
            shown[0] = now
            bar = "#" * min(30, int(motion * 600))
            print(f"\r  {state.value:<8} {bar:<30}", end="", flush=True)

    try:
        for frames, fps, width, height in segments_from_camera(
            camera_index=args.camera, config=config, on_state=on_state,
            max_segments=args.limit,
        ):
            print(f"\r  segment: {len(frames)} frames" + " " * 30)
            try:
                pose = pose_from_segment(frames, fps, width, height)
            except Exception as exc:
                print(f"  [pose extraction failed: {exc}]")
                continue
            report(interpreter.sign_to_speech(pose), interpreter)
    except CaptureError as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sign-to-speech", action="store_true")
    mode.add_argument("--speech-to-sign", metavar="TEXT")
    mode.add_argument("--dry-run", action="store_true",
                      help="feed a .pose file instead of the camera")
    parser.add_argument("--clip", type=Path, help="pose file for --dry-run")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--motion", type=float, default=0.012)
    parser.add_argument("--limit", type=int, default=None, help="stop after N signs")
    parser.add_argument("--silent", action="store_true", help="do not use the virtual mic")
    args = parser.parse_args()

    speaking = bool(args.sign_to_speech or args.dry_run) and not args.silent
    interpreter = build(speaking=speaking, rendering=args.speech_to_sign is not None)
    print(f"recogniser: {interpreter.recognizer_kind}, "
          f"{len(interpreter.recognizer.vocabulary)} signs | "
          f"lexicon: {len(interpreter.lookup.glosses)} signs\n")

    if args.sign_to_speech:
        return run_sign_to_speech(interpreter, args)

    if args.dry_run:
        from backend.recognition.extract import load_pose
        if not args.clip or not args.clip.is_file():
            raise SystemExit("--dry-run needs --clip <file.pose>")
        print(f"feeding {args.clip.name}")
        report(interpreter.sign_to_speech(load_pose(args.clip)), interpreter)
        return 0

    lines = ([line.strip() for line in sys.stdin if line.strip()]
             if args.speech_to_sign == "-" else [args.speech_to_sign])
    for line in lines:
        print(f"heard: {line!r}")
        report(interpreter.speech_to_sign(line), interpreter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
