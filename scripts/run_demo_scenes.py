#!/usr/bin/env python3
"""Run the demo script's scenes (architecture v3 §10) through the real pipeline.

    .venv/bin/python scripts/run_demo_scenes.py            # all Stage 1 scenes
    .venv/bin/python scripts/run_demo_scenes.py --scene 4
    .venv/bin/python scripts/run_demo_scenes.py --speak     # also into the virtual mic

Each scene runs the real components and prints the decision log it produced, so
what the audience is told is happening is what actually happened. Nothing here
scripts an outcome: scene 2's ambiguity is a real confusion pair from the corpus
and scene 4's refusal is a real lookup miss, not a flag set to make a point.

Scene 3 (glossary hit) is Stage 2 — the session and domain glossary steps are
Tier 1 in architecture v3 §4.2. It is run anyway, because the trace showing the
waterfall trying those steps and falling through them is honest and is what
Stage 1 has; it is labelled as such rather than presented as a hit.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.contracts import CoverageStatus, Direction, Segment  # noqa: E402
from backend.waterfall.decision_log import DecisionLog  # noqa: E402
from backend.waterfall.escalation import EscalationWaterfall  # noqa: E402

SPEAK = False
DEMO_PATHS: dict = {}


def banner(number: str, title: str, expected: str) -> None:
    print(f"\n{'=' * 78}\nSCENE {number} — {title}\n  expected: {expected}\n{'-' * 78}")


def show(log: DecisionLog) -> None:
    print(log.render())


def say(text: str) -> None:
    """Speak a line into the virtual microphone, if the bridge is available."""
    if not SPEAK:
        return
    from backend.meeting_bridge.virtualcam import BridgeError, VirtualMicrophone
    from backend.sign_to_speech.tts_output import TTSError, synthesize

    try:
        VirtualMicrophone().play_wav(synthesize(text).audio, blocking=True)
    except (BridgeError, TTSError) as exc:
        print(f"  [speech unavailable: {exc}]")


# The clips the scenes recognise. They are held out of the classifier's
# templates, without exception.
#
# Fitting on everything and then demoing a clip that is in the training set
# returns confidence 1.00 every time — the clip matches itself at distance zero.
# That is not a high-confidence recognition, it is a lookup, and a demo built on
# it would show the audience a number that means nothing. Architecture v3 is
# explicit that scene 2's ambiguity has to be genuine rather than staged; it can
# only be genuine if the classifier has never seen the clip.
DEMO_CLIPS = {
    "scene1": ("HELLO", "*INCLUDE*00570*.pose"),
    "scene2": ("SHE", "*ISL500*00095*.pose"),
}


def demo_clip_paths(vocab: Path) -> dict[str, Path]:
    chosen: dict[str, Path] = {}
    for key, (gloss, pattern) in DEMO_CLIPS.items():
        matches = sorted((vocab / gloss).glob(pattern)) or sorted((vocab / gloss).glob("*.pose"))
        chosen[key] = matches[0]
    return chosen


def classifier_for(vocab: Path, held_out: list[Path]):
    from backend.recognition.classifier import SignClassifier

    classifier = SignClassifier()
    count, failures = classifier.fit_directory(vocab, exclude=held_out)
    print(f"  {count} templates, holding out {len(held_out)} clip(s) the scenes will "
          f"recognise ({len(failures)} unreadable)")
    return classifier


def recognise(classifier, path: Path) -> Segment:
    from backend.recognition.extract import load_pose

    return classifier.classify(load_pose(path))


# ------------------------------------------------------------------ scenes --
def scene_1(classifier, vocab: Path) -> None:
    from backend.sign_to_speech.reasoning import reconstruct_sentence

    banner("1", "Normal exchange", "high confidence -> translate, no clarification")
    log, waterfall = DecisionLog(), None
    waterfall = EscalationWaterfall(on_log=log.add)

    clip = DEMO_PATHS["scene1"]
    segment = recognise(classifier, clip)
    decision = waterfall.process(segment)
    sentence = reconstruct_sentence(segment.raw_input, segment.confidence)

    print(f"  signed:  {clip.name[:52]}")
    print(f"  read as: {segment.raw_input!r} @ {segment.confidence:.2f} "
          f"({segment.coverage_status.value})")
    print(f"  action:  {decision.action.value}")
    print(f"  spoken:  {sentence.sentence!r}\n")
    say(sentence.sentence)
    show(log)


def scene_2(classifier, vocab: Path) -> None:
    banner("2", "Ambiguity -> clarification",
           "a real confusion pair stalls both sides and asks")
    log = DecisionLog()
    waterfall = EscalationWaterfall(on_log=log.add)

    # HE and SHE differ by referent, not handshape — the manifest names this as
    # the strongest genuine confusion pair in the vocabulary, and the held-out
    # evaluation confirms it: this clip reads HE, with SHE as runner-up.
    clip = DEMO_PATHS["scene2"]
    stalled = False
    # The ladder needs two consecutive low-confidence segments to trigger (§4.1
    # hysteresis), so the same genuine misread is presented twice — which is what
    # a signer repeating themselves after no response actually produces.
    for attempt in range(2):
        segment = recognise(classifier, clip)
        from backend.recognition.classifier import pose_features
        from backend.recognition.extract import load_pose

        prediction = classifier.classify_features(pose_features(load_pose(clip)))
        decision = waterfall.process(
            segment, candidates=[prediction.gloss_id, prediction.runner_up]
        )
        print(f"  attempt {attempt + 1}: {clip.name[:40]:<42} -> "
              f"{segment.raw_input:<5} @ {segment.confidence:.2f} "
              f"(runner-up {prediction.runner_up}, margin "
              f"{prediction.runner_up_distance - prediction.best_distance:.1f})  "
              f"{decision.action.value}"
              f"{'  [STALLS BOTH SIDES]' if decision.stalls_both_sides else ''}")
        if decision.stalls_both_sides:
            question = f"Did you sign {decision.candidates[0]} or {decision.candidates[1]}?"
            print(f"     asks: {question!r}")
            say(question)
            stalled = True
            break
    if not stalled:
        print("  NOTE: this clip did not read ambiguously on this run — the scene "
              "depends on real classifier output, so it is reported, not forced.")
    print()
    show(log)


def scene_3(classifier, vocab: Path) -> None:
    banner("3", "Glossary step (Stage 2 — shown falling through)",
           "the waterfall tries the glossary steps and reports which resolved it")
    log = DecisionLog()
    waterfall = EscalationWaterfall(on_log=log.add)
    segment = Segment(
        id="scene-3", direction=Direction.SIGN_TO_SPEECH, raw_input="HOSPITAL",
        confidence=0.34, coverage_status=CoverageStatus.LANGUAGE_BACKUP,
        timestamp=time.time(),
    )
    decision = waterfall.process(segment)
    print(f"  term 'HOSPITAL' @ {segment.confidence:.2f} -> {decision.action.value}, "
          f"resolved at stage {decision.stage_reached.value}")
    print("  (session and domain glossary are Stage 2; the trace shows them tried)\n")
    show(log)


def scene_4(classifier, vocab: Path) -> None:
    from backend.speech_to_sign.gloss_lookup import GlossLookup

    banner("4", "Unsupported sign -> refuse to fabricate",
           "no validated pose exists, so nothing is rendered and the log says why")
    log = DecisionLog()
    waterfall = EscalationWaterfall(on_log=log.add)

    lookup = GlossLookup(ROOT / "data" / "lexicon")
    tokens = ["HELLO", "AMBULANCE", "PLEASE"]
    coverage = lookup.coverage_for(tokens)
    unmatched = [c.gloss for c in coverage if c.status is CoverageStatus.UNMATCHED]

    segment = Segment(
        id="scene-4", direction=Direction.SPEECH_TO_SIGN,
        raw_input=" ".join(tokens), confidence=0.91,
        coverage_status=CoverageStatus.UNMATCHED, timestamp=time.time(),
    )
    decision = waterfall.process(segment, unmatched=unmatched)

    for c in coverage:
        print(f"  {c.gloss:<12} {c.status.value}")
    print(f"  -> {decision.action.value}: {decision.reason}")
    print(f"  uncertain terms surfaced: {decision.uncertain_terms}\n")
    if decision.uncertain_terms:
        say(f"I do not have a validated sign for {decision.uncertain_terms[0]}.")
    show(log)


def scene_5(classifier, vocab: Path) -> None:
    from backend.collision.state_machine import CollisionManager, HoldDecision

    banner("5", "Rendering collision",
           "someone speaks mid-sign: hold, queue, resume at a clip boundary")
    log = DecisionLog()

    clock = [1000.0]
    manager = CollisionManager(clock=lambda: clock[0], self_speaking=lambda: False)
    manager.submit("please sit down", duration_s=3.0)
    print("  avatar queued: 'please sit down' (3.0s)")

    held_at = released_at = None
    for i in range(60):                      # a real speaker interrupts for 2s
        if manager.observe_audio(True) is HoldDecision.HOLD and held_at is None:
            held_at = i / 30
        clock[0] += 1 / 30
    for i in range(120):                     # they stop
        if manager.observe_audio(False) is HoldDecision.RELEASE and released_at is None:
            released_at = i / 30
            break
        clock[0] += 1 / 30

    print(f"  held {held_at:.2f}s into the overlap; released {released_at:.2f}s after silence")
    for transition in manager.transitions:
        log.add_collision(transition)
    print()
    show(log)


SCENES = {1: scene_1, 2: scene_2, 3: scene_3, 4: scene_4, 5: scene_5}


def main() -> int:
    global SPEAK
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=int, choices=sorted(SCENES), default=None)
    parser.add_argument("--speak", action="store_true",
                        help="also speak each line into the virtual microphone")
    args = parser.parse_args()
    SPEAK = args.speak

    random.seed(7)
    vocab = ROOT / "data" / "vocab"
    global DEMO_PATHS
    DEMO_PATHS = demo_clip_paths(vocab)
    print("fitting the recognition classifier on the curated vocabulary...")
    classifier = classifier_for(vocab, list(DEMO_PATHS.values()))

    for number in ([args.scene] if args.scene else sorted(SCENES)):
        SCENES[number](classifier, vocab)
    print(f"\n{'=' * 78}\nmanual override control is present in every trace above "
          f"(the final line of each log).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
