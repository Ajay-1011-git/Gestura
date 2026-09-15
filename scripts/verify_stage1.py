#!/usr/bin/env python3
"""Run Stage 1's §E Final Acceptance checklist and report what is actually green.

The checklist in `setu-stage1-build-instructions.md` §E is written as ten
statements about real behaviour, each of which was meant to be confirmed by a
real run. Nothing in the repository ran them, so "done" meant "the task has a
commit". This does the runs it can and says plainly which items it cannot reach
from here.

    .venv/bin/python scripts/verify_stage1.py            # local only, no API calls
    .venv/bin/python scripts/verify_stage1.py --live     # also exercises Groq

Items 2, 3 and 10 need Groq (`--live`); the audible-output half of item 2 and
all of item 10 additionally need OBS's virtual camera and microphone running,
which no script can stand in for. Those are reported as BLOCKED with the reason,
never as passing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.contracts import CoverageStatus, Direction, Segment  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []   # (status, item, evidence)


def record(status: str, item: str, evidence: str) -> None:
    RESULTS.append((status, item, evidence))
    print(f"  {status:<7} {item}\n          {evidence}")


def check(ok: bool, item: str, evidence: str) -> bool:
    record("PASS" if ok else "FAIL", item, evidence)
    return ok


def blocked(item: str, reason: str) -> None:
    record("BLOCKED", item, reason)


# ---------------------------------------------------------------- item 1 ----
def item_1_vocabulary() -> None:
    manifest = ROOT / "data" / "vocab" / "vocab_manifest.md"
    text = manifest.read_text()
    approved = "approved" in text.lower()

    # The active vocabulary is the first table; the second lists deferred entries.
    tables = re.findall(r"\| gloss_id \|.*?\n(?:\|.*\n)+", text)
    active = re.findall(r"^\| ([A-Z][A-Z\-]+) \|", tables[0], re.M) if tables else []

    vocab_dir = ROOT / "data" / "vocab"
    lexicon_index = ROOT / "data" / "lexicon" / "index.csv"
    lexicon = {}
    if lexicon_index.exists():
        with lexicon_index.open() as handle:
            lexicon = {r["glosses"].upper(): r["path"] for r in csv.DictReader(handle)}

    missing_training, missing_lexicon = [], []
    for gloss in active:
        poses = list((vocab_dir / gloss).glob("*.pose")) if (vocab_dir / gloss).is_dir() else []
        if not poses:
            missing_training.append(gloss)
        path = lexicon.get(gloss)
        if not path or not (ROOT / "data" / "lexicon" / path).exists():
            missing_lexicon.append(gloss)

    check(
        bool(active) and approved and not missing_training and not missing_lexicon,
        "1. curated vocabulary exists, is approved, and every entry has a .pose",
        f"{len(active)} active glosses, manifest approved={approved}, "
        f"training poses missing for {missing_training or 'none'}, "
        f"lexicon entries missing for {missing_lexicon or 'none'}",
    )


# ---------------------------------------------------------------- item 4 ----
def item_4_fingerspelling() -> None:
    from backend.speech_to_sign.gloss_lookup import GlossLookup

    lookup = GlossLookup(ROOT / "data" / "lexicon")
    tokens = ["HELLO", "YOU", "ZYGOMORPHIC", "SIT"]
    coverage = lookup.coverage_for(tokens)
    by_gloss = {c.gloss.upper(): c.status for c in coverage}

    oov = by_gloss.get("ZYGOMORPHIC")
    known_ok = all(
        by_gloss.get(g) is CoverageStatus.LEXICON_HIT for g in ("HELLO", "YOU", "SIT")
    )
    flagged = oov in (CoverageStatus.FINGERSPELLING, CoverageStatus.UNMATCHED)
    check(
        flagged and known_ok,
        "4. an out-of-vocabulary term is reported, not silently guessed",
        "coverage: " + ", ".join(f"{c.gloss}={c.status.value}" for c in coverage),
    )


# ---------------------------------------------------------------- item 5 ----
def item_5_escalation() -> None:
    from backend.waterfall.escalation import EscalationWaterfall

    entries: list = []
    waterfall = EscalationWaterfall(on_log=entries.append)

    def segment(sid: str, confidence: float, raw: str = "HELP") -> Segment:
        return Segment(
            id=sid, direction=Direction.SIGN_TO_SPEECH, raw_input=raw,
            confidence=confidence, coverage_status=None, timestamp=time.time(),
        )

    low = [waterfall.process(segment(f"low-{i}", 0.21)) for i in range(3)]
    stalled = [d for d in low if d.stalls_both_sides]
    clarified = [d for d in low if "clarif" in d.action.value.lower()]
    clean = waterfall.process(segment("clean-1", 0.97, "HELLO"))

    check(
        bool(stalled) and bool(clarified) and not clean.stalls_both_sides,
        "5. low confidence stalls both sides and asks, then a clean segment resumes",
        f"3 low-confidence segments -> {[d.action.value for d in low]}; "
        f"stalls={len(stalled)}, clarifications={len(clarified)}; "
        f"clean follow-up -> {clean.action.value} (stalls={clean.stalls_both_sides})",
    )
    return entries


# ------------------------------------------------------------- items 6,7 ----
class Clock:
    """Controllable clock, so the timing rules are tested rather than raced."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def item_6_self_tts() -> None:
    from backend.collision.state_machine import CollisionManager, HoldDecision

    clock = Clock()
    manager = CollisionManager(clock=clock, self_speaking=lambda: True)
    manager.submit("please sit", duration_s=2.0)

    decisions = []
    for _ in range(40):                      # ~1.3s of sustained "voice"
        decisions.append(manager.observe_audio(True))
        clock.advance(1 / 30)

    check(
        manager.suppressed_self_frames == len(decisions)
        and all(d is not HoldDecision.HOLD for d in decisions),
        "6. Gestura's own TTS does not trigger a collision hold",
        f"{manager.suppressed_self_frames}/{len(decisions)} voice frames attributed to "
        f"SETU_TTS_ACTIVE; holding={manager.holding}",
    )


def item_7_overlap() -> None:
    from backend.collision.state_machine import CollisionManager, HoldDecision

    clock = Clock()
    manager = CollisionManager(clock=clock, self_speaking=lambda: False)
    manager.submit("please sit", duration_s=2.0)

    held_at = None
    for i in range(60):                      # real external speech, 2s
        decision = manager.observe_audio(True)
        if decision is HoldDecision.HOLD and held_at is None:
            held_at = i / 30
        clock.advance(1 / 30)

    released_at = None
    for i in range(120):                     # speaker stops
        decision = manager.observe_audio(False)
        if decision is HoldDecision.RELEASE and released_at is None:
            released_at = i / 30
            break
        clock.advance(1 / 30)

    mid_gesture = [
        t for t in manager.transitions
        if t.decision.name == "HOLD" and "mid" in t.detail.lower()
    ]
    check(
        held_at is not None and released_at is not None and not mid_gesture,
        "7. real overlap holds, and silence resumes it",
        f"held after {held_at:.2f}s of speech, released {released_at:.2f}s after silence; "
        f"{len(manager.transitions)} transitions, none recorded mid-gesture",
    )


# ---------------------------------------------------------------- item 8 ----
def item_8_decision_log(entries) -> None:
    from backend.waterfall.decision_log import DecisionLog

    log = DecisionLog()
    for entry in entries:
        log.add(entry)

    stages = [e.stage for e in entries]
    rendered = log.render()
    has_trace = all(s in stages for s in ("OBSERVE", "DECIDE", "ACTION"))
    parsed = json.loads(log.to_json())
    # The override control must be visible at all times (FR-16), so it is part
    # of the payload rather than something the panel invents.
    override_shown = "override" in parsed and "OVERRIDE" in rendered.upper()

    check(
        has_trace and override_shown and len(parsed["entries"]) == len(entries),
        "8. the decision log shows a readable Observe->Decide->Action trace",
        f"{len(entries)} entries covering {sorted(set(stages))}; "
        f"render() {len(rendered.splitlines())} lines; "
        f"to_json() {len(parsed['entries'])} entries + override={parsed['override']!r}",
    )

    # The panel reads a file of exactly this shape; if the two drift, the demo
    # shows an empty panel and nothing else reports it.
    shipped = json.loads((ROOT / "assets" / "decision_log.json").read_text())
    same_shape = (
        set(shipped) == set(parsed)
        and shipped["entries"]
        and set(shipped["entries"][0]) == set(parsed["entries"][0])
    )
    check(
        same_shape,
        "8b. the shipped panel payload matches what DecisionLog.to_json produces",
        f"assets/decision_log.json: {len(shipped['entries'])} entries, "
        f"keys {sorted(shipped['entries'][0])}; live keys {sorted(parsed['entries'][0])}",
    )


# ---------------------------------------------------------------- item 9 ----
def item_9_scope() -> None:
    import subprocess

    out = subprocess.run(
        ["git", "log", "--name-only", "--pretty=format:", "--", "backend", "frontend"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    touched = {line for line in out.splitlines() if line.strip()}
    forbidden = [f for f in touched if "/tier2" in f or "avatar_customis" in f]
    check(
        not forbidden,
        "9. no out-of-scope component was built",
        f"{len(touched)} source files across all commits; "
        f"no Stage 2/3 components present ({forbidden or 'none found'})",
    )


# ------------------------------------------------------------- items 2,3 ----
def item_3_speech_to_sign_live() -> None:
    """A real spoken sentence -> real gloss -> real pose sequence, with coverage."""
    from backend.speech_to_sign.gloss_lookup import GlossLookup
    from backend.speech_to_sign.pose_smoothing import build_avatar_sequence, smooth
    from backend.speech_to_sign.reasoning import to_gloss

    transcript = "Hello, please sit down."
    result = to_gloss(transcript)
    lookup = GlossLookup(ROOT / "data" / "lexicon")
    coverage = lookup.coverage_for(list(result.tokens))
    renderable = [c.gloss for c in coverage if c.status is CoverageStatus.LEXICON_HIT]

    pose, timeline = build_avatar_sequence(
        renderable, ROOT / "data" / "lexicon", target_fps=30.0,
    ) if renderable else (None, [])
    if pose is not None:
        pose, report = smooth(pose)

    # What is asserted is the pipeline's contract, not the model's wording. The
    # gloss varies between runs — the same transcript returned 'HELLO ME SIT
    # DOWN' and 'SIT DOWN' on two consecutive calls — so pinning the output
    # would make this a test of Groq's determinism. Every token must carry a
    # coverage status, and whatever is renderable must render into a pose whose
    # timeline fits it; producing nothing renderable is a legitimate outcome
    # (FR-12 refuses to fabricate) rather than a failure.
    every_token_classified = len(coverage) == len(result.tokens)
    pose_consistent = pose is None or timeline[-1]["end_frame"] <= pose.body.data.shape[0]
    check(
        bool(result.tokens) and every_token_classified and pose_consistent,
        "3. a real sentence becomes real gloss, a real pose sequence, and real coverage",
        f"{transcript!r} -> gloss {result.gloss!r} ({result.latency_s:.2f}s, {result.model}); "
        f"coverage " + ", ".join(f"{c.gloss}={c.status.value}" for c in coverage) +
        (f"; rendered {pose.body.data.shape[0]} frames over {len(timeline)} signs, "
         f"{report.gaps_filled} gaps filled" if pose is not None else "; nothing renderable"),
    )


def item_2_sign_to_speech_live() -> None:
    """A recognized sign -> a real English sentence -> real synthesized audio.

    Two separate claims, kept apart deliberately. Whether the *pipeline* carries a
    recognition through to audio is a yes/no. How often recognition is right is a
    measurement, and asserting it on one hand-picked clip would say nothing: the
    same class's two held-out clips score 0.818 and 0.055 here. The escalation
    ladder (item 5) exists precisely because recognition is not reliable, so an
    acceptance check that demanded perfect recognition would be testing for
    something Stage 1 never claimed.
    """
    import random

    from backend.recognition.classifier import (
        ClassifierError, SignClassifier, pose_features,
    )
    from backend.recognition.extract import load_pose
    from backend.sign_to_speech.reasoning import reconstruct_sentence
    from backend.sign_to_speech.tts_output import TTSError, synthesize

    vocab = ROOT / "data" / "vocab"
    classes = sorted(p.name for p in vocab.iterdir() if p.is_dir() and p.name != "raw_video")
    random.seed(7)
    held = {c: random.sample(sorted((vocab / c).glob("*.pose")), 1) for c in classes}
    excluded = [p for v in held.values() for p in v]

    classifier = SignClassifier()
    count, failures = classifier.fit_directory(vocab, exclude=excluded)

    # ---- 2a: does a recognition reach audio at all? -------------------------
    probe = held["HELLO"][0]
    segment = classifier.classify(load_pose(probe))
    sentence = reconstruct_sentence(segment.raw_input, segment.confidence)
    try:
        utterance = synthesize(sentence.sentence)
        audio_ok = len(utterance.audio) > 0 and utterance.duration_s > 0
        audio_note = (f"{len(utterance.audio)} bytes, {utterance.duration_s:.2f}s "
                      f"at {utterance.sample_rate}Hz")
    except TTSError as exc:
        audio_ok, audio_note = False, f"TTS failed: {exc}"

    check(
        bool(segment.raw_input) and bool(sentence.sentence) and audio_ok,
        "2a. a recognition carries through to a real sentence and real audio",
        f"{count} templates ({len(failures)} unreadable); {probe.name[:38]} -> "
        f"{segment.raw_input!r} @ {segment.confidence:.2f} "
        f"({segment.coverage_status.value if segment.coverage_status else 'none'}) -> "
        f"{sentence.sentence!r} -> {audio_note}",
    )

    # ---- 2b: how often is it right? ----------------------------------------
    correct = total = confident_right = confident_wrong = 0
    unreadable = []
    for gloss, paths in held.items():
        for path in paths:
            try:
                prediction = classifier.classify_features(pose_features(load_pose(path)))
            except (ClassifierError, ValueError) as exc:
                unreadable.append((path.name, str(exc)[:60]))
                continue
            total += 1
            hit = prediction.gloss_id == gloss
            correct += hit
            if prediction.confidence >= 0.22:
                confident_right += hit
                confident_wrong += not hit

    accuracy = correct / total if total else 0.0
    check(
        accuracy >= 0.50 and confident_wrong <= confident_right,
        "2b. held-out recognition accuracy has not regressed",
        f"top-1 {correct}/{total} = {accuracy:.0%} over {total} unseen clips; "
        f"above the 0.22 confidence threshold {confident_right} right / "
        f"{confident_wrong} wrong; {len(unreadable)} clips unreadable"
        + (f" ({unreadable[0][0]}: {unreadable[0][1]})" if unreadable else ""),
    )


# --------------------------------------------------------------- bridge ----
def item_2_audible(frames_dir: Path | None) -> None:
    """The other half of item 2: the sentence has to leave the machine.

    OBS supplies only the video half. `virtualcam.py` documents why — OBS
    Virtual Camera carries no audio and macOS ships no virtual microphone — so
    the speech path needs a separate HAL driver, and this check reports which
    one it actually found rather than assuming.
    """
    from backend.meeting_bridge.virtualcam import (
        BridgeError, VirtualCamera, VirtualMicrophone, describe_bridge,
    )
    from backend.sign_to_speech.reasoning import reconstruct_sentence
    from backend.sign_to_speech.tts_output import synthesize

    status = describe_bridge()
    video, audio = status.get("video", {}), status.get("audio", {})
    check(
        bool(video.get("available")) and bool(audio.get("available")),
        "T1.16. both virtual devices are present",
        f"video: {video.get('device', video.get('reason'))} "
        f"(backend {video.get('backend', '-')}); audio: "
        f"{audio.get('device', audio.get('reason'))}",
    )
    if not (video.get("available") and audio.get("available")):
        return

    # ---- audio: a real sentence, spoken into the virtual microphone --------
    sentence = reconstruct_sentence("HELLO", 0.87)
    utterance = synthesize(sentence.sentence)
    microphone = VirtualMicrophone()
    started = time.monotonic()
    try:
        played = microphone.play_wav(utterance.audio, blocking=True)
        elapsed = time.monotonic() - started
        audio_ok = played > 0 and elapsed >= played * 0.5
        note = (f"{sentence.sentence!r} -> {played:.2f}s of audio into "
                f"{microphone.device.name!r}, playback took {elapsed:.2f}s")
    except BridgeError as exc:
        audio_ok, note = False, str(exc)
    check(audio_ok, "2. a real sentence is audible through the virtual microphone", note)

    # ---- video: real rendered avatar frames into the virtual camera --------
    frames: list = []
    source = "generated test pattern"
    if frames_dir and frames_dir.is_dir():
        from PIL import Image

        paths = sorted(frames_dir.glob("*.png"))
        for path in paths[:120]:
            frames.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
        if frames:
            source = f"{len(frames)} rendered avatar frames from {frames_dir.name}/"
    if not frames:
        # Still a real send, just not of the avatar — reported as such.
        gradient = np.linspace(0, 255, 1280, dtype=np.uint8)
        frames = [np.dstack([
            np.tile(gradient, (720, 1)),
            np.full((720, 1280), 80, np.uint8),
            np.full((720, 1280), 160, np.uint8),
        ])]

    height, width = frames[0].shape[:2]
    try:
        with VirtualCamera(width=width, height=height, fps=30.0) as camera:
            device = camera.device
            for frame in frames:
                camera.send(frame)
        video_ok, note = True, f"sent {len(frames)} frame(s) of {source} to {device!r} at {width}x{height}"
    except BridgeError as exc:
        video_ok, note = False, str(exc)
    check(video_ok, "T1.16. real frames reach the OBS virtual camera", note)


def item_10_demo_scenes() -> None:
    """Every demo scene runs end to end and reaches the decision it claims to."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_demo_scenes.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = result.stdout
    expectations = {
        "SCENE 1": "action:  translate",
        "SCENE 2": "[STALLS BOTH SIDES]",
        "SCENE 3": "resolved at stage",
        "SCENE 4": "-> refuse",
        "SCENE 5": "released",
    }
    reached = {name: marker in out for name, marker in expectations.items()}
    overrides = out.count("override: RUNNING")

    check(
        result.returncode == 0 and all(reached.values()) and overrides >= 5,
        "10. all five demo scenes run and reach their stated outcome",
        ", ".join(f"{name.lower()} {'ok' if ok else 'MISSED'}" for name, ok in reached.items())
        + f"; override control shown in {overrides} traces",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="also run the items that call Groq (costs API usage)")
    parser.add_argument("--bridge", action="store_true",
                        help="also write to the OBS virtual camera and virtual mic")
    parser.add_argument("--frames", type=Path, default=None,
                        help="directory of rendered avatar PNGs to send to the camera")
    args = parser.parse_args()

    print("\nStage 1 §E Final Acceptance\n")
    item_1_vocabulary()
    item_4_fingerspelling()
    entries = item_5_escalation()
    item_6_self_tts()
    item_7_overlap()
    item_8_decision_log(entries)
    item_9_scope()

    if args.live:
        print()
        item_3_speech_to_sign_live()
        item_2_sign_to_speech_live()
    else:
        blocked("2/3. speech<->sign round trips through real LLM output",
                "needs Groq; re-run with --live")
    if args.bridge:
        print()
        item_2_audible(args.frames)
    else:
        blocked("2. audible through the virtual microphone",
                "needs OBS Virtual Camera + a virtual mic device; re-run with --bridge")
    if args.live:
        item_10_demo_scenes()
    else:
        blocked("10. demo scenes 1-5", "scene 1 calls Groq; re-run with --live")
    blocked("10. ...performed live in front of an audience",
            "the scenes run and are asserted above; watching a live run is a human step")

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    held = sum(1 for s, _, _ in RESULTS if s == "BLOCKED")
    print(f"\n{passed} passed, {failed} failed, {held} blocked\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
