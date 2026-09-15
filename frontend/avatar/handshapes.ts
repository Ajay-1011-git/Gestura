/**
 * Canonical ISL handshapes, and matching frames to them (T1.16).
 *
 * Constraining the fingers anatomically (see `hand_anatomy.ts`) stops them
 * bending in ways a hand cannot. It does not, on its own, make them read as ISL:
 * 21 landmarks with no finger roll and weak depth do not resolve the contact and
 * spacing that tell one handshape from another, so a purely measured pose lands
 * near a handshape without ever being one.
 *
 * So each frame is matched against a small vocabulary of authored handshapes and
 * pulled toward the match in proportion to how well it fits. A confident frame
 * renders a clean, unambiguous shape; an unconfident one keeps the measurement
 * and simply stays anatomical. This is how production sign avatars work, and it
 * is the difference between "approximately a V" and a V.
 *
 * The vocabulary covers the Stage 1 sign list — HELLO, THANK-YOU, PLEASE, YES,
 * NO, YOU, HE, SHE, HOSPITAL, HELP, GO, SIT, OKAY, WHAT, WHERE, TODAY — rather
 * than all of ISL. Adding a sign that needs a new shape means adding a row here.
 */

import * as THREE from "three";
import {
  FINGERS, HandAngleFilter, poseExtension,
  type DigitAngles, type FingerName, type HandAngles, type HandMeasurement,
  cloneAngles, mixAngles, zeroAngles,
} from "./hand_anatomy";

/** `[mcp, spread, pip, dip]` in degrees, the order joints run proximal to distal. */
type DigitSpec = readonly [number, number, number, number];

export interface Handshape {
  /** Stable id used in the HUD and in verification output. */
  name: string;
  /** Nearest equivalent in the widely used handshape naming, for orientation. */
  alias: string;
  description: string;
  angles: HandAngles;
  /** Per-digit extension implied by `angles`, solved not authored. */
  extension: Record<FingerName, number>;
}

function shape(
  name: string, alias: string, description: string,
  spec: Record<FingerName, DigitSpec>,
): Handshape {
  const angles = zeroAngles();
  for (const f of FINGERS) {
    const [mcp, spread, pip, dip] = spec[f];
    angles[f] = { mcp, spread, pip, dip };
  }
  return {
    name, alias, description, angles,
    extension: Object.fromEntries(
      FINGERS.map((f) => [f, poseExtension(angles[f])]),
    ) as Record<FingerName, number>,
  };
}

/**
 * The authored reference poses.
 *
 * Hand-written, and deliberately so: these are the clean targets the noisy
 * measurement is snapped toward, so they have to be cleaner than anything the
 * landmarks can produce. Spread values decrease across the hand because the
 * little finger abducts away from the midline while the middle barely moves.
 */
export const HANDSHAPES: readonly Handshape[] = [
  shape("FLAT", "B", "flat hand, fingers together — HELLO, THANK-YOU, PLEASE, TODAY", {
    thumb: [15, 10, 5, 5], index: [3, 4, 4, 3], middle: [2, 0, 4, 3],
    ring: [2, -4, 5, 3], little: [3, -10, 6, 4],
  }),
  shape("FLAT_SPREAD", "5", "open hand, fingers spread — WHAT, WHERE", {
    thumb: [5, 45, 0, 0], index: [0, 20, 2, 2], middle: [0, 5, 2, 2],
    ring: [0, -12, 3, 2], little: [0, -28, 4, 3],
  }),
  shape("FIST", "A/S", "closed fist, thumb across the front — YES, HELP", {
    thumb: [30, 5, 25, 15], index: [85, 2, 100, 55], middle: [85, 0, 102, 55],
    ring: [85, -3, 102, 55], little: [85, -6, 100, 55],
  }),
  shape("POINT", "1/G", "index extended, rest closed — YOU, HE, SHE, GO, WHERE", {
    thumb: [35, 0, 25, 10], index: [2, 2, 3, 2], middle: [88, 0, 105, 58],
    ring: [88, -3, 105, 58], little: [88, -6, 103, 58],
  }),
  shape("TWO", "V", "index and middle extended, spread — NO", {
    thumb: [35, 0, 28, 12], index: [3, 14, 4, 3], middle: [3, -6, 4, 3],
    ring: [88, -5, 105, 58], little: [88, -8, 103, 58],
  }),
  shape("TWO_TOGETHER", "U/H", "index and middle extended, together — SIT, HOSPITAL", {
    thumb: [35, 0, 28, 12], index: [3, 1, 4, 3], middle: [3, -1, 4, 3],
    ring: [88, -5, 105, 58], little: [88, -8, 103, 58],
  }),
  shape("THUMB_UP", "A+", "fist with the thumb extended — OKAY", {
    thumb: [0, 45, 0, 0], index: [88, 2, 105, 58], middle: [88, 0, 105, 58],
    ring: [88, -3, 105, 58], little: [88, -6, 103, 58],
  }),
  shape("RING", "F", "thumb and index tips meet, three extended — OKAY", {
    thumb: [40, 12, 38, 25], index: [48, 8, 58, 20], middle: [5, 2, 6, 4],
    ring: [5, -6, 7, 4], little: [6, -14, 8, 5],
  }),
  shape("OPEN_O", "O", "fingertips curved in toward the thumb", {
    thumb: [35, 20, 28, 18], index: [45, 6, 48, 22], middle: [45, 1, 50, 22],
    ring: [45, -5, 50, 22], little: [46, -12, 50, 22],
  }),
  shape("CLAW", "C", "curved and spread — HOSPITAL, WHAT", {
    thumb: [25, 35, 15, 10], index: [35, 14, 42, 25], middle: [33, 4, 44, 25],
    ring: [33, -8, 44, 25], little: [34, -18, 44, 25],
  }),
  shape("BENT_FLAT", "bent-B", "knuckles folded, phalanges straight — HELP", {
    thumb: [20, 12, 8, 5], index: [80, 3, 5, 3], middle: [80, 0, 5, 3],
    ring: [80, -4, 6, 3], little: [80, -8, 8, 4],
  }),
  shape("RELAXED", "—", "neutral hand between signs", {
    thumb: [18, 22, 12, 10], index: [22, 6, 30, 18], middle: [24, 1, 34, 20],
    ring: [26, -5, 38, 22], little: [28, -12, 40, 24],
  }),
];

export const RELAXED: Handshape = HANDSHAPES.find((s) => s.name === "RELAXED")!;

/**
 * What a frame is actually matched against.
 *
 * RELAXED is excluded deliberately. It is the neutral pose a hand settles to
 * when it leaves frame, not a handshape anyone signs — and because it sits near
 * the middle of every axis it is the nearest neighbour to *any* ambiguous pose.
 * With it in the vocabulary it won 210 of 281 frames on one hand, swallowing
 * shapes that were genuinely being made. A pose that resembles nothing is
 * already handled: its fitness is low, so little of the canonical shape is
 * applied and the measurement shows through. That is the correct way to say
 * "unknown" — a label for it is not needed.
 */
export const VOCABULARY: readonly Handshape[] = HANDSHAPES.filter((s) => s !== RELAXED);

/**
 * Per-joint weights for matching.
 *
 * MCP and PIP carry the shape — whether a finger is up or down — so they
 * dominate. DIP is downweighted because it is the noisiest measured channel and
 * is largely determined by PIP anyway. Spread matters (it separates V from U)
 * but far less than whether the finger is extended at all.
 */
const WEIGHTS = { mcp: 1.0, pip: 1.0, dip: 0.4, spread: 0.35 } as const;
/** Normalising spans, roughly the usable range of each joint. */
const SPANS = { mcp: 90, pip: 110, dip: 80, spread: 45 } as const;
/** Extension is the most reliable cue, so it is weighted above any one joint. */
const EXTENSION_WEIGHT = 1.4;
/** Straight (1.0) to fully folded (~0.4). */
const EXTENSION_SPAN = 0.6;

function digitDistanceSq(a: DigitAngles, b: DigitAngles): { sum: number; weight: number } {
  let sum = 0, weight = 0;
  for (const key of ["mcp", "pip", "dip", "spread"] as const) {
    const d = (a[key] - b[key]) / SPANS[key];
    sum += WEIGHTS[key] * d * d;
    weight += WEIGHTS[key];
  }
  return { sum, weight };
}

/**
 * Normalised distance from a measurement to a handshape, 0 = identical.
 *
 * `extension` is optional so a pose can be compared to a shape on angles alone
 * (which is what the rig's rest pose and the handshape table have); when a real
 * measurement supplies it, it carries more weight than any single joint.
 */
export function handDistance(
  a: HandAngles,
  b: Handshape | { angles: HandAngles; extension?: Record<FingerName, number> },
  extension?: Record<FingerName, number>,
): number {
  const target = b.angles;
  let sum = 0, weight = 0;
  for (const f of FINGERS) {
    // The thumb is the least reliably tracked digit, so it counts for less.
    const scale = f === "thumb" ? 0.6 : 1;
    const d = digitDistanceSq(a[f], target[f]);
    sum += scale * d.sum;
    weight += scale * d.weight;
    if (extension && b.extension) {
      const e = (extension[f] - b.extension[f]) / EXTENSION_SPAN;
      sum += scale * EXTENSION_WEIGHT * e * e;
      weight += scale * EXTENSION_WEIGHT;
    }
  }
  return Math.sqrt(sum / weight);
}

export interface Match {
  shape: Handshape;
  distance: number;
  /** Distance to the next-best shape; the gap is what makes a match meaningful. */
  runnerUp: number;
  /** 0–1. High only when the fit is close *and* clearly better than the rest. */
  confidence: number;
}

/**
 * Distance at which a pose counts as a clean instance of a shape, and the
 * distance beyond which it is not really any of them.
 *
 * Calibrated against `demo_sequence.pose` rather than picked: a correct,
 * unambiguous match there measures 0.12–0.23. The upper figures are not noise —
 * MediaPipe systematically under-reports a folded finger, because the folded
 * fingertip is occluded and the landmark regresses toward the palm, so the
 * measured middle finger of a clear POINT reads 52° of PIP flexion where the
 * finger is at ~105°. Setting the clean threshold below that would refuse to
 * snap on exactly the frames snapping exists for.
 */
const CLEAN_FIT = 0.15;
const MAX_FIT = 0.50;
/**
 * Gap at which a match counts as unambiguous. Small because twelve shapes in one
 * 20-dimensional angle space sit close together by construction; observed gaps
 * for correct matches run 0.037–0.095.
 */
const DECISIVE_GAP = 0.06;

export function classify(
  angles: HandAngles,
  extension?: Record<FingerName, number>,
  vocabulary: readonly Handshape[] = VOCABULARY,
): Match {
  let best = vocabulary[0], bestD = Infinity, secondD = Infinity;
  for (const candidate of vocabulary) {
    const d = handDistance(angles, candidate, extension);
    if (d < bestD) { secondD = bestD; bestD = d; best = candidate; }
    else if (d < secondD) secondD = d;
  }
  return {
    shape: best, distance: bestD, runnerUp: secondD,
    confidence: fitness(bestD) * THREE.MathUtils.clamp((secondD - bestD) / DECISIVE_GAP, 0, 1),
  };
}

/** 1 at or below a clean match, falling to 0 at the point of no resemblance. */
function fitness(distance: number): number {
  return THREE.MathUtils.clamp((MAX_FIT - distance) / (MAX_FIT - CLEAN_FIT), 0, 1);
}

export interface TrackerOptions {
  /** Upper bound on how far a frame is pulled toward its matched shape. */
  maxSnap?: number;
  /** Consecutive frames a challenger must win before the held shape changes. */
  holdFrames?: number;
  /**
   * Time constant, seconds, over which per-shape evidence is accumulated.
   *
   * Classifying each frame independently and then debouncing the label is a weak
   * estimator when the thing being estimated is constant: the handshape does not
   * change during a sign, but the measurement of it carries the landmark noise
   * measured at 3–5°/frame, so the per-frame winner wanders between neighbouring
   * shapes — 15 label changes across 281 frames, four different shapes inside a
   * single 45-frame sign. Averaging each shape's distance over a window instead
   * averages that noise down before the decision is made, which is where it
   * belongs. 0.25s is short against a sign (1.7–3.0s here) and long against the
   * noise.
   */
  evidenceSeconds?: number;
  /**
   * Frames a shape must be held before it is trusted as fully as an unambiguous
   * single-frame match.
   *
   * A sign holds one handshape for its whole duration, so a shape that has won
   * every frame for a third of a second is better evidence than any one frame's
   * margin — and the hold is what a reader actually reads. Transitions, where
   * the shape is genuinely changing, never accumulate dwell and so stay close to
   * the measurement, which is what keeps movement between signs looking like
   * movement rather than a cut between poses.
   */
  dwellFrames?: number;
  /** Seconds for an untracked hand to settle to neutral. */
  releaseSeconds?: number;
  /**
   * Frames to cross-fade over when the held handshape changes.
   *
   * Without it the canonical target jumps the instant hysteresis flips the
   * label, and at a snap of ~0.7 most of that jump reaches the rig as a visible
   * pop — the hand arrives at the next shape before the arm has finished moving
   * there. Fading the target is not smoothing the measurement: both endpoints
   * stay canonical, so the hand passes between two clean shapes rather than
   * through a blur.
   */
  transitionFrames?: number;
}

export interface TrackedHand {
  angles: HandAngles;
  shape: Handshape | null;
  confidence: number;
  distance: number;
  /** How far this frame was pulled toward the canonical shape, 0–1. */
  snap: number;
  /** True when the hand was not tracked and the pose is a held/decaying one. */
  coasting: boolean;
  /** Consecutive frames the current shape has been held. */
  held: number;
}

/**
 * Per-hand state: jitter filtering, shape hysteresis, and graceful handling of
 * frames where the hand is not tracked at all.
 *
 * Hysteresis matters more than it might seem. Classification on raw frames
 * flickers between neighbouring shapes mid-hold — V to U and back — and a
 * flickering handshape is less readable than a slightly wrong stable one. A sign
 * holds its handshape; so does this.
 *
 * Coasting matters for the same reason. The left hand is tracked in 59 of 281
 * frames of `demo_sequence.pose`; snapping it to the rig's rest pose on the other
 * 222 makes it twitch open and shut. Holding the last shape and settling toward
 * neutral looks like a hand that simply stopped being filmed.
 */
export class HandshapeTracker {
  private readonly filter = new HandAngleFilter();
  private readonly maxSnap: number;
  private readonly holdFrames: number;
  private readonly evidenceSeconds: number;
  private readonly dwellFrames: number;
  private readonly transitionFrames: number;
  private readonly releaseSeconds: number;

  /** Running mean distance to each shape — the accumulated evidence. */
  private readonly scores = new Map<string, number>();
  private current: Handshape | null = null;
  private previous: Handshape | null = null;
  private heldCount = 0;
  private challenger: Handshape | null = null;
  private challengerCount = 0;
  private last: HandAngles = cloneAngles(RELAXED.angles);
  private lostSeconds = 0;

  constructor(options: TrackerOptions = {}) {
    this.maxSnap = options.maxSnap ?? 0.85;
    this.holdFrames = options.holdFrames ?? 3;
    this.evidenceSeconds = options.evidenceSeconds ?? 0.25;
    this.dwellFrames = options.dwellFrames ?? 10;
    this.transitionFrames = options.transitionFrames ?? 5;
    this.releaseSeconds = options.releaseSeconds ?? 0.25;
  }

  reset(): void {
    this.filter.reset();
    this.scores.clear();
    this.current = null;
    this.previous = null;
    this.heldCount = 0;
    this.challenger = null;
    this.challengerCount = 0;
    this.last = cloneAngles(RELAXED.angles);
    this.lostSeconds = 0;
  }

  update(measured: HandMeasurement | null, dt: number): TrackedHand {
    if (!measured) {
      this.lostSeconds += Math.max(dt, 0);
      const t = THREE.MathUtils.clamp(this.lostSeconds / this.releaseSeconds, 0, 1);
      // Ease out, so a hand leaving frame settles rather than slides linearly.
      const eased = t * t * (3 - 2 * t);
      // Once fully released the hand is showing a neutral pose, so say so rather
      // than keep reporting the shape it held before it left frame — and drop
      // the accumulated evidence, which describes a hand that is no longer there.
      if (t >= 1 && this.current !== RELAXED) {
        this.current = RELAXED;
        this.previous = null;
        this.heldCount = 0;
        this.scores.clear();
      }
      return {
        angles: mixAngles(this.last, RELAXED.angles, eased),
        shape: this.current,
        confidence: 0,
        distance: Infinity,
        snap: 0,
        coasting: true,
        held: this.heldCount,
      };
    }

    if (this.lostSeconds > 0) { this.lostSeconds = 0; this.filter.reset(); }

    const smoothed = this.filter.apply(measured.angles, dt);
    const extension = measured.extension;

    // Accumulate evidence: a running mean of each shape's distance, rather than
    // a fresh verdict per frame.
    const alpha = dt > 0 ? 1 - Math.exp(-dt / this.evidenceSeconds) : 1;
    let leader = VOCABULARY[0], leaderScore = Infinity, runnerUpScore = Infinity;
    for (const candidate of VOCABULARY) {
      const d = handDistance(smoothed, candidate, extension);
      const previousScore = this.scores.get(candidate.name);
      const score = previousScore === undefined ? d : previousScore + alpha * (d - previousScore);
      this.scores.set(candidate.name, score);
      if (score < leaderScore) { runnerUpScore = leaderScore; leaderScore = score; leader = candidate; }
      else if (score < runnerUpScore) runnerUpScore = score;
    }

    // Hysteresis on the accumulated scores: a challenger has to lead for several
    // frames *and* by more than a hair before the handshape is allowed to change.
    const previous = this.current;
    if (!this.current) {
      this.current = leader;
    } else if (leader !== this.current) {
      const heldScore = this.scores.get(this.current.name) ?? Infinity;
      this.challengerCount = this.challenger === leader ? this.challengerCount + 1 : 1;
      this.challenger = leader;
      if (leaderScore < heldScore - 0.01 && this.challengerCount >= this.holdFrames) {
        this.previous = this.current;
        this.current = leader;
        this.challengerCount = 0;
      }
    } else {
      this.challenger = null;
      this.challengerCount = 0;
    }
    this.heldCount = this.current === previous ? this.heldCount + 1 : 0;

    const held = this.current;
    // Cross-fade out of the shape just left, so the canonical target moves
    // continuously even though the label changed in one frame.
    const fade = THREE.MathUtils.clamp(this.heldCount / this.transitionFrames, 0, 1);
    const target = this.previous && fade < 1
      ? mixAngles(this.previous.angles, held.angles, fade * fade * (3 - 2 * fade))
      : held.angles;
    if (fade >= 1) this.previous = null;

    const heldDistance = this.scores.get(held.name) ?? handDistance(smoothed, held, extension);
    const fit = fitness(heldDistance);
    const margin = THREE.MathUtils.clamp((runnerUpScore - leaderScore) / DECISIVE_GAP, 0, 1);
    const dwell = THREE.MathUtils.clamp(this.heldCount / this.dwellFrames, 0, 1);
    // Either a clean separation this frame or a sustained hold counts as
    // evidence; the floor keeps a transition from dissolving into raw landmarks.
    const evidence = Math.max(margin, dwell);
    const snap = this.maxSnap * fit * (0.45 + 0.55 * evidence);

    const angles = mixAngles(smoothed, target, snap);
    this.last = cloneAngles(angles);
    return { angles, shape: held, confidence: fit * evidence, distance: heldDistance,
             snap, coasting: false, held: this.heldCount };
  }
}
