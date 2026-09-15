/**
 * Anatomical hand model — measurement, constraint, and application (T1.16).
 *
 * **Why the previous finger solving looked wrong, measured rather than guessed.**
 * Aiming each finger bone at the direction between its two landmarks gives every
 * joint three free rotational degrees of freedom. A real finger has one at PIP
 * and DIP (pure hinges) and two at MCP (flexion + abduction). The extra freedom
 * is not harmless: it is filled with whatever sideways component the landmarks
 * happen to carry, and in this data that component is large.
 *
 * Deviation of the landmark bend axis from the true knuckle hinge, over frames
 * where the joint is genuinely bent (>25°), right hand of `demo_sequence.pose`:
 *
 *     MCP   18–21° median, 40–58° p90
 *     PIP   31–36° median, 57–78° p90
 *     DIP   31–49° median, 60–76° p90
 *
 * PIP and DIP cannot deviate at all. Rendering 30° of it — 75° on bad frames —
 * is what "fingers bending unrealistically" looks like. Signed flexion is also
 * negative on 15–23% of right-hand frames, i.e. joints folding backwards.
 *
 * So this module does not aim bones. It **measures scalar joint angles** by
 * projecting each segment onto the plane of its own hinge, discards the
 * out-of-plane component instead of rendering it, clamps to clinical range of
 * motion, and rebuilds the pose as rotations about the rig's own hinge axes.
 * A finger can then only do what a finger does — by construction, not tuning.
 */

import * as THREE from "three";
import type { HumanoidBone, LoadedAvatar } from "./loader";
import { OneEuro } from "./filters";

export type FingerName = "thumb" | "index" | "middle" | "ring" | "little";
export const FINGERS: readonly FingerName[] = ["thumb", "index", "middle", "ring", "little"];

/** The three drivable joints of a digit, proximal first. */
export type JointName = "mcp" | "pip" | "dip";
export const JOINTS: readonly JointName[] = ["mcp", "pip", "dip"];

/**
 * One digit's pose, in degrees.
 *
 * `mcp`/`pip`/`dip` are flexion — positive curls toward the palm. `spread` is
 * abduction at the base joint only, positive toward the thumb. PIP and DIP carry
 * no spread term because they are hinges; that is the whole point.
 *
 * For the thumb the same four numbers mean CMC flexion, CMC abduction, MCP
 * flexion and IP flexion, which is the same proximal-to-distal ordering.
 */
export interface DigitAngles { mcp: number; spread: number; pip: number; dip: number }
export type HandAngles = Record<FingerName, DigitAngles>;

export type Range = readonly [number, number];
export interface DigitLimits { mcp: Range; spread: Range; pip: Range; dip: Range }

/**
 * Clinical range of motion, degrees.
 *
 * Asymmetric spread limits are anatomy, not tuning: index and little abduct
 * freely away from the midline while middle and ring barely move, and the little
 * finger's abduction is mostly *away* from the thumb, hence its lopsided range.
 * A few degrees of MCP hyperextension is real; PIP and DIP get none.
 */
export const ROM: Record<FingerName, DigitLimits> = {
  thumb:  { mcp: [-20, 65], spread: [-20, 80], pip: [-10, 55], dip: [-15, 80] },
  index:  { mcp: [-20, 90], spread: [-12, 30], pip: [0, 110], dip: [0, 80] },
  middle: { mcp: [-20, 90], spread: [-15, 15], pip: [0, 110], dip: [0, 80] },
  ring:   { mcp: [-20, 90], spread: [-22, 10], pip: [0, 110], dip: [0, 80] },
  little: { mcp: [-20, 95], spread: [-35, 10], pip: [0, 110], dip: [0, 80] },
};

/** MediaPipe hand landmark indices, `[base, …, tip]` per digit. */
export const DIGIT_LANDMARKS: Record<FingerName, readonly [number, number, number, number]> = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  little: [17, 18, 19, 20],
};
export const WRIST = 0;
/** Base knuckles, used for the palm frame. */
export const MIDDLE_MCP = 9, INDEX_MCP = 5, LITTLE_MCP = 17;

const BONE_SUFFIX: Record<JointName, string> = {
  mcp: "Proximal", pip: "Intermediate", dip: "Distal",
};
const DIGIT_PREFIX: Record<FingerName, string> = {
  thumb: "Thumb", index: "Index", middle: "Middle", ring: "Ring", little: "Little",
};

export function digitBone(side: "left" | "right", finger: FingerName, joint: JointName): HumanoidBone {
  return `${side}${DIGIT_PREFIX[finger]}${BONE_SUFFIX[joint]}` as HumanoidBone;
}

export function zeroAngles(): HandAngles {
  return Object.fromEntries(
    FINGERS.map((f) => [f, { mcp: 0, spread: 0, pip: 0, dip: 0 }]),
  ) as HandAngles;
}

export function cloneAngles(a: HandAngles): HandAngles {
  return Object.fromEntries(FINGERS.map((f) => [f, { ...a[f] }])) as HandAngles;
}

export function mixAngles(a: HandAngles, b: HandAngles, t: number): HandAngles {
  const out = zeroAngles();
  for (const f of FINGERS) {
    out[f].mcp = THREE.MathUtils.lerp(a[f].mcp, b[f].mcp, t);
    out[f].spread = THREE.MathUtils.lerp(a[f].spread, b[f].spread, t);
    out[f].pip = THREE.MathUtils.lerp(a[f].pip, b[f].pip, t);
    out[f].dip = THREE.MathUtils.lerp(a[f].dip, b[f].dip, t);
  }
  return out;
}

/** Clamp every joint into its clinical range. Returns how many were out of it. */
export function clampToROM(angles: HandAngles): number {
  let violations = 0;
  for (const f of FINGERS) {
    for (const key of ["mcp", "spread", "pip", "dip"] as const) {
      const [lo, hi] = ROM[f][key];
      const v = angles[f][key];
      const c = THREE.MathUtils.clamp(v, lo, hi);
      if (Math.abs(c - v) > 1e-6) violations += 1;
      angles[f][key] = c;
    }
  }
  return violations;
}

/**
 * The palm's orthonormal frame.
 *
 * `flex` is the axis a positive flexion rotates about; `abduct` is the axis a
 * positive spread rotates about (toward the thumb on either hand). Handedness
 * only enters through `chirality`, so left and right share one code path and one
 * sign convention — which is what lets the rig's rest pose and the landmark data
 * be measured by the same function.
 */
export interface PalmFrame {
  origin: THREE.Vector3;
  /** Wrist toward the middle knuckle. */
  forward: THREE.Vector3;
  /** Across the knuckles, toward the thumb. */
  radial: THREE.Vector3;
  /** Out of the palm, on the side the fingers curl toward. */
  palmar: THREE.Vector3;
  flex: THREE.Vector3;
  abduct: THREE.Vector3;
  /** Wrist-to-middle-knuckle distance, the natural unit for this hand. */
  scale: number;
}

/**
 * Build the palm frame from four points.
 *
 * Chirality is taken from the side label rather than inferred per frame, but the
 * two agree: `sign(dot(thumbMCP − indexMCP, forward × radial))` is positive for
 * left and negative for right in both `avatar.glb`'s rest pose and the landmark
 * data, so the LEFT/RIGHT component mapping is confirmed rather than assumed.
 */
export function palmFrame(
  wrist: THREE.Vector3,
  middleMcp: THREE.Vector3,
  indexMcp: THREE.Vector3,
  littleMcp: THREE.Vector3,
  side: "left" | "right",
): PalmFrame {
  const forward = middleMcp.clone().sub(wrist);
  const scale = forward.length() || 1;
  forward.normalize();

  const radial = indexMcp.clone().sub(littleMcp);
  radial.addScaledVector(forward, -radial.dot(forward));   // Gram-Schmidt
  if (radial.lengthSq() < 1e-12) radial.set(1, 0, 0).addScaledVector(forward, -forward.x);
  radial.normalize();

  const normal = new THREE.Vector3().crossVectors(forward, radial).normalize();
  const chirality = side === "left" ? 1 : -1;

  return {
    origin: wrist.clone(),
    forward,
    radial,
    // Rotating `forward` about `normal` by +φ yields `radial`, so `normal` is the
    // abduction axis with "toward the thumb" positive on both hands.
    abduct: normal.clone(),
    palmar: normal.clone().multiplyScalar(chirality),
    // Rotating `forward` about this by +θ yields `palmar`: flexion curls inward.
    flex: radial.clone().multiplyScalar(-chirality),
    scale,
  };
}

/** The palm frame as a right-handed rotation matrix, columns `[forward, radial, abduct]`. */
export function palmBasis(frame: PalmFrame): THREE.Matrix4 {
  return new THREE.Matrix4().makeBasis(frame.forward, frame.radial, frame.abduct);
}

/**
 * Signed rotation from `ref` to `seg` **about** `axis`, in degrees.
 *
 * Both vectors are projected onto the plane perpendicular to the axis first. The
 * projection is the constraint: whatever the landmarks say about sideways motion
 * at a hinge is dropped here rather than rendered as a twisted finger.
 */
export function hingeAngle(ref: THREE.Vector3, seg: THREE.Vector3, axis: THREE.Vector3): number {
  const a = ref.clone().addScaledVector(axis, -ref.dot(axis));
  const b = seg.clone().addScaledVector(axis, -seg.dot(axis));
  if (a.lengthSq() < 1e-12 || b.lengthSq() < 1e-12) return 0;
  a.normalize(); b.normalize();
  const sin = new THREE.Vector3().crossVectors(a, b).dot(axis);
  return THREE.MathUtils.radToDeg(Math.atan2(sin, a.dot(b)));
}

export interface MeasureOptions {
  /**
   * How far DIP is pulled toward its anatomical coupling with PIP, 0–1.
   *
   * The distal phalanx cannot be flexed independently — the tendon linkage holds
   * DIP at roughly ⅔ of PIP — and its landmarks are the worst in the hand: the
   * shortest segments (~15px against an 88px palm here) carry the most
   * quantisation noise, which shows up as a measured DIP range above 100° where
   * human ROM is ~80°. Deriving most of DIP from PIP removes the noisiest
   * channel and costs nothing a real hand could have expressed.
   */
  dipCoupling?: number;
}

const DIP_FROM_PIP = 2 / 3;

/**
 * The four directions a digit's joints are measured against: a reference for the
 * base joint, then the three phalanges.
 *
 * For the fingers the reference is that finger's own metacarpal. **For the thumb
 * it is the palm's long axis instead**, because `WRIST`→`THUMB_CMC` is a short
 * wrist-to-thumb-base offset pointing sideways across the palm, not along the
 * thumb: measured against it, the thumb's base joint sat clamped at its
 * range-of-motion floor (`spread −15`) on nearly every frame. Against the palm
 * axis the same angle means what the handshape table means by it — how far the
 * thumb is carried away from the fingers.
 */
function digitSegments(
  points: readonly THREE.Vector3[],
  finger: FingerName,
  frame: PalmFrame,
): THREE.Vector3[] {
  const [base, a, b, tip] = DIGIT_LANDMARKS[finger];
  const reference = finger === "thumb"
    ? frame.forward.clone().multiplyScalar(frame.scale)
    : points[base].clone().sub(points[WRIST]);
  return [
    reference,
    points[a].clone().sub(points[base]),
    points[b].clone().sub(points[a]),
    points[tip].clone().sub(points[b]),
  ];
}

/**
 * The flexion axis for each joint of a digit, in the same space as `points`.
 *
 * The four fingers share the palm's knuckle hinge. **The thumb does not, and
 * measuring it as though it did is wrong in a way that shows up immediately**:
 * against the finger hinge every thumb joint sat pinned at a range-of-motion
 * limit on every frame of `demo_sequence.pose` — `mcp −10, spread −15, pip −10`
 * — because the thumb's real motion is almost perpendicular to that axis. A
 * digit stuck at its limits carries no information and, worse, contributes a
 * large constant error to every handshape distance, which suppressed matching
 * across the board.
 *
 * The thumb's carpometacarpal joint is a saddle, not a hinge; its useful
 * decomposition is rotation out of the palm plane (opposition, toward the palm)
 * and rotation within it (abduction, away from the index). Taking each joint's
 * axis as `segment × palmar` gives the first, and the shared `abduct` axis gives
 * the second.
 */
function flexAxes(segments: readonly THREE.Vector3[], frame: PalmFrame, finger: FingerName): THREE.Vector3[] {
  if (finger !== "thumb") return [frame.flex, frame.flex, frame.flex];
  return segments.slice(0, 3).map((segment) => {
    const axis = new THREE.Vector3().crossVectors(segment, frame.palmar);
    // Degenerate only if the thumb points straight out of the palm, where any
    // perpendicular is as good as another.
    return axis.lengthSq() < 1e-12 ? frame.flex.clone() : axis.normalize();
  });
}

/**
 * Fraction of its own length a digit spans from base knuckle to tip: 1.0 dead
 * straight, ~0.4 fully folded.
 *
 * This is measured from **positions**, not from the chain of joint angles, and
 * that is the point. Chaining three noisy segment directions accumulates their
 * error, and it accumulates worst exactly where it matters — a folded finger
 * points toward the camera, its segments foreshorten, and the summed flexion
 * comes out far short of the truth. On `demo_sequence.pose` a plainly closed
 * fist measures 52° of PIP flexion this way where the finger is near 105°. The
 * ratio of two lengths degrades far more gracefully: across the same sequence it
 * reads 1.00 for every extended finger of a flat hand and 0.46–0.66 for a fist,
 * with no overlap. It is the strongest single cue for what a handshape is.
 */
export function digitExtension(points: readonly THREE.Vector3[], finger: FingerName): number {
  const [base, a, b, tip] = DIGIT_LANDMARKS[finger];
  const chain = points[base].distanceTo(points[a])
    + points[a].distanceTo(points[b])
    + points[b].distanceTo(points[tip]);
  return chain < 1e-9 ? 1 : points[base].distanceTo(points[tip]) / chain;
}

/**
 * The same quantity for an *authored* pose, by planar forward kinematics.
 *
 * Phalanx proportions are the standard ones and only their ratios matter, since
 * the result is normalised. Solving it rather than authoring it keeps the
 * handshape table to the joint angles that actually drive the rig.
 */
const PHALANX = [0.45, 0.28, 0.27];
export function poseExtension(digit: DigitAngles): number {
  let heading = 0, x = 0, y = 0;
  [digit.mcp, digit.pip, digit.dip].forEach((flex, i) => {
    heading += THREE.MathUtils.degToRad(flex);
    x += PHALANX[i] * Math.cos(heading);
    y += PHALANX[i] * Math.sin(heading);
  });
  return Math.hypot(x, y);
}

export interface HandMeasurement {
  angles: HandAngles;
  /** Per-digit `digitExtension`, the position-derived open/closed cue. */
  extension: Record<FingerName, number>;
  frame: PalmFrame;
}

/** Measure joint angles and extension from 21 hand landmarks in any 3D space. */
export function measureHand(
  points: readonly THREE.Vector3[],
  side: "left" | "right",
  options: MeasureOptions = {},
): HandMeasurement {
  const coupling = options.dipCoupling ?? 0.55;
  const frame = palmFrame(
    points[WRIST], points[MIDDLE_MCP], points[INDEX_MCP], points[LITTLE_MCP], side,
  );
  const angles = zeroAngles();

  for (const finger of FINGERS) {
    const [reference, proximal, middle, distal] = digitSegments(points, finger, frame);
    const axis = flexAxes([reference, proximal, middle], frame, finger);

    const d = angles[finger];
    d.mcp = hingeAngle(reference, proximal, axis[0]);
    d.spread = hingeAngle(reference, proximal, frame.abduct);
    d.pip = hingeAngle(proximal, middle, axis[1]);
    const measuredDip = hingeAngle(middle, distal, axis[2]);
    // The thumb's IP genuinely does act on its own, so it keeps more of its
    // measurement than the fingers do.
    const k = finger === "thumb" ? coupling * 0.45 : coupling;
    d.dip = THREE.MathUtils.lerp(measuredDip, d.pip * DIP_FROM_PIP, k);
  }

  clampToROM(angles);
  const extension = Object.fromEntries(
    FINGERS.map((f) => [f, digitExtension(points, f)]),
  ) as Record<FingerName, number>;
  return { angles, extension, frame };
}

/**
 * A one-euro filter per joint angle — 20 channels for one hand.
 *
 * `beta` is unit-dependent and these channels are **degrees**, so a derivative
 * during a fast transition runs to several hundred deg/s. The textbook beta of
 * ~0.3 assumes normalised inputs; at this scale it drives the cutoff to ~100Hz
 * and the filter stops filtering. 0.012 keeps the cutoff near 4Hz through a fast
 * transition and at the 1.5Hz floor on a hold, which is where signing needs it.
 */
export class HandAngleFilter {
  private readonly channels = new Map<string, OneEuro>();
  constructor(
    private readonly minCutoff = 1.5,
    private readonly beta = 0.012,
  ) {}

  apply(angles: HandAngles, dt: number): HandAngles {
    const out = zeroAngles();
    for (const f of FINGERS) {
      for (const key of ["mcp", "spread", "pip", "dip"] as const) {
        const id = `${f}.${key}`;
        let ch = this.channels.get(id);
        if (!ch) { ch = new OneEuro(this.minCutoff, this.beta); this.channels.set(id, ch); }
        out[f][key] = ch.filter(angles[f][key], dt);
      }
    }
    return out;
  }

  reset(): void { for (const ch of this.channels.values()) ch.reset(); }
}

/**
 * Per-bone hinge axes and rest offsets, measured once from the rig.
 *
 * Axes live in each bone's **parent** local space, which is what makes the chain
 * behave: the PIP hinge is fixed to the proximal phalanx, so it tilts with MCP
 * flexion exactly as a real finger's does, with no extra bookkeeping.
 */
export interface HandCalibration {
  side: "left" | "right";
  axes: Map<HumanoidBone, { flex: THREE.Vector3; abduct: THREE.Vector3 }>;
  /** The rig's own residual curl, subtracted so 0° means a straight digit. */
  rest: HandAngles;
  /**
   * Palm basis in the hand bone's local space, columns `[forward, radial, abduct]`.
   *
   * That ordering is right-handed — `forward × radial = abduct` by construction —
   * so the matrix is a true rotation and converts to a quaternion cleanly.
   * Ordering it `[radial, forward, abduct]` instead would be a reflection; the
   * reflections cancel between calibration and use, so the rig still poses
   * correctly, but any quaternion taken from one on its own is meaningless.
   */
  palmLocal: THREE.Matrix4;
  /** Wrist-to-middle-knuckle length in avatar units. */
  scale: number;
  /**
   * The wrist's own axes, in the **forearm's** local space.
   *
   * A wrist is not a ball joint with one cone of travel. It flexes and extends
   * freely about the knuckle-line axis (~80°/70°) but deviates very little side
   * to side (~20° radial, ~35° ulnar), and those two motions have to be limited
   * separately or the hand is allowed into positions no wrist reaches while
   * being stopped short of ones every wrist does.
   */
  wrist: { axis: THREE.Vector3; flex: THREE.Vector3; deviate: THREE.Vector3 };
}

function boneOf(avatar: LoadedAvatar, name: HumanoidBone): THREE.Bone | undefined {
  return avatar.bones.get(name);
}

/**
 * The rig's own hand as 21 landmark-shaped points, from wherever it is posed now.
 *
 * Feeding these through the same `measureHand` used on the source data is
 * what lets calibration and verification share one code path with the solver:
 * a convention error cancels between the two rather than becoming an offset to
 * tune, and the rendered pose can be read back in the same units it was
 * commanded in. Distal bones are leaves in this rig, so a fingertip is
 * extrapolated along the distal bone's own axis — enough to carry its
 * orientation, which is all any of these measurements need.
 */
export function sampleHandPoints(
  avatar: LoadedAvatar,
  side: "left" | "right",
): THREE.Vector3[] | null {
  const hand = boneOf(avatar, `${side}Hand` as HumanoidBone);
  if (!hand) return null;

  const world = (b: THREE.Object3D) => b.getWorldPosition(new THREE.Vector3());
  const points: THREE.Vector3[] = new Array(21);
  points[WRIST] = world(hand);

  for (const finger of FINGERS) {
    const chain = JOINTS.map((j) => boneOf(avatar, digitBone(side, finger, j)));
    if (chain.some((b) => !b)) return null;
    const [p, i, d] = chain as THREE.Bone[];
    const [li, la, lb, lt] = DIGIT_LANDMARKS[finger];
    points[li] = world(p);
    points[la] = world(i);
    points[lb] = world(d);

    const axis = avatar.restPose.get(digitBone(side, finger, "dip"))?.axis ?? new THREE.Vector3(0, 1, 0);
    const dir = axis.clone()
      .applyQuaternion(d.getWorldQuaternion(new THREE.Quaternion()))
      .normalize();
    points[lt] = points[lb].clone().addScaledVector(dir, points[la].distanceTo(points[lb]) || 0.02);
  }
  return points;
}

/** Measure the rig's hand at rest, and the hinge axes to drive it by. */
export function calibrateHand(avatar: LoadedAvatar, side: "left" | "right"): HandCalibration | null {
  const hand = boneOf(avatar, `${side}Hand` as HumanoidBone);
  const points = sampleHandPoints(avatar, side);
  if (!hand || !points) return null;

  // dipCoupling 0: the rig's rest angles must be measured, not modelled.
  const { angles: rest, frame } = measureHand(points, side, { dipCoupling: 0 });

  const axes = new Map<HumanoidBone, { flex: THREE.Vector3; abduct: THREE.Vector3 }>();
  for (const finger of FINGERS) {
    // Same axis rule the measurement uses, evaluated on the rig's rest geometry,
    // so a joint driven to angle θ is driven about the axis θ was read off.
    const worldFlex = flexAxes(digitSegments(points, finger, frame), frame, finger);
    JOINTS.forEach((joint, i) => {
      const name = digitBone(side, finger, joint);
      const bone = boneOf(avatar, name);
      if (!bone?.parent) return;
      const toParent = bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
      axes.set(name, {
        flex: worldFlex[i].clone().applyQuaternion(toParent).normalize(),
        abduct: frame.abduct.clone().applyQuaternion(toParent).normalize(),
      });
    });
  }

  const palmLocal = new THREE.Matrix4()
    .makeRotationFromQuaternion(hand.getWorldQuaternion(new THREE.Quaternion()).invert())
    .multiply(palmBasis(frame));

  // The forearm's long axis is simply where it holds the hand; the wrist's two
  // swing axes are the palm's own, carried into the same space.
  const toForearm = (hand.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion()).invert();
  const axis = hand.position.lengthSq() > 1e-12
    ? hand.position.clone().normalize()
    : new THREE.Vector3(0, 1, 0);
  const flex = frame.flex.clone().applyQuaternion(toForearm);
  flex.addScaledVector(axis, -flex.dot(axis));
  if (flex.lengthSq() < 1e-12) flex.set(1, 0, 0).addScaledVector(axis, -axis.x);
  flex.normalize();
  const deviate = new THREE.Vector3().crossVectors(axis, flex).normalize();
  // Orient it so a positive rotation is radial deviation, matching `abduct`.
  if (deviate.dot(frame.abduct.clone().applyQuaternion(toForearm)) < 0) deviate.negate();

  return { side, axes, rest, palmLocal, scale: frame.scale, wrist: { axis, flex, deviate } };
}

/**
 * Write joint angles onto the rig.
 *
 * Each joint becomes `R(abduct, spread) · R(flex, θ) · rest` in its parent's
 * space. PIP and DIP are given no abduction term at all, so a hinge cannot
 * splay sideways however noisy the source was — the constraint is structural,
 * not a clamp applied afterwards.
 */
export function applyHandAngles(
  avatar: LoadedAvatar,
  calibration: HandCalibration,
  angles: HandAngles,
): number {
  const { side, axes, rest } = calibration;
  let written = 0;

  for (const finger of FINGERS) {
    for (const joint of JOINTS) {
      const name = digitBone(side, finger, joint);
      const bone = boneOf(avatar, name);
      const restPose = avatar.restPose.get(name);
      const axis = axes.get(name);
      if (!bone || !restPose || !axis) continue;

      const flex = angles[finger][joint] - rest[finger][joint];
      const q = new THREE.Quaternion().setFromAxisAngle(
        axis.flex, THREE.MathUtils.degToRad(flex),
      );
      if (joint === "mcp") {
        const spread = angles[finger].spread - rest[finger].spread;
        q.premultiply(new THREE.Quaternion().setFromAxisAngle(
          axis.abduct, THREE.MathUtils.degToRad(spread),
        ));
      }
      bone.quaternion.copy(q.multiply(restPose.quaternion));
      written += 1;
    }
  }

  avatar.bones.get(`${side}Hand` as HumanoidBone)?.updateMatrixWorld(true);
  return written;
}
