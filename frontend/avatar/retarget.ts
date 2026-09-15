/**
 * Pose retargeting — playback of looked-up pose sequences (T1.12, T1.16).
 *
 * Speech->Sign direction only. No live camera feed is wired here.
 *
 * **Why this does direct aim retargeting instead of using Kalidokit's solvers.**
 * Both were tried against the real data and both failed visibly:
 *
 *  - Kalidokit's output is expressed in VRM's humanoid axis convention against a
 *    T-pose rest. This rig is a Mixamo-named glTF resting in an A-pose (~39.7°
 *    droop, measured in T1.11). Applying those Eulers — replaced or composed
 *    onto rest — snapped the arms to a T-pose spread instead of signing.
 *  - The landmark scales are inconsistent: `.pose` stores x and y in pixels but
 *    leaves z in MediaPipe's normalised units, so z is ~1/500th the magnitude of
 *    x and y. Depth is real, but only after `z * width`.
 *
 * Arms are solved by two-bone IK onto a wrist target. **Fingers are not aimed at
 * all** — see `hand_anatomy.ts`. Aiming a finger bone at a landmark direction
 * gives each joint three rotational degrees of freedom where a real joint has
 * one or two, and the surplus gets filled with landmark noise: measured 31–36°
 * median deviation from the hinge at PIP, 57–78° at p90. That is what fingers
 * bending unrealistically actually was. Joint angles are now measured as scalars
 * on their own hinge planes, matched to a canonical ISL handshape, and rebuilt
 * on the rig's own axes.
 */

import * as THREE from "three";
import type { HumanoidBone, LoadedAvatar } from "./loader";
import {
  INDEX_MCP, LITTLE_MCP, MIDDLE_MCP, WRIST,
  applyHandAngles, calibrateHand, measureHand, palmBasis, palmFrame,
  type HandCalibration,
} from "./hand_anatomy";
import { HandshapeTracker, type TrackedHand, type TrackerOptions } from "./handshapes";
import { GlitchGate, SmoothedRotation, SmoothedVector3, softLimit, softRange } from "./filters";

interface PosePoint { X: number; Y: number; Z?: number; C?: number }

export interface SignSpan {
  gloss: string;
  start_frame: number;
  end_frame: number;
  start_s: number;
  duration_s: number;
}

export interface PoseSequence {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  componentPoints: Record<string, string[]>;
  frames: Record<string, PosePoint[]>[];
}

export async function loadPoseSequence(url: string): Promise<PoseSequence> {
  const { Pose } = await import("pose-format");
  const parsed: any = await Pose.fromRemote(url);
  const header = parsed.header;

  const componentPoints: Record<string, string[]> = {};
  for (const component of header.components) {
    componentPoints[component.name] = component.points;
  }

  // `body.frames` is a Proxy over the binary buffer, not the array its .d.ts
  // declares: Object.keys() is empty and .map() does not exist, but integer
  // indexing works and `_frames` holds the count (pose-format@1.6.2).
  const frameCount: number = parsed.body._frames ?? 0;
  const frames: Record<string, PosePoint[]>[] = [];
  for (let i = 0; i < frameCount; i++) {
    const frame = parsed.body.frames[i];
    if (frame?.people?.[0]) frames.push(frame.people[0]);
  }

  return {
    fps: parsed.body.fps,
    frameCount: frames.length,
    width: header.width,
    height: header.height,
    componentPoints,
    frames,
  };
}

/** Orthonormal basis, columns `[side, forward, normal]`. */
function basis(forward: THREE.Vector3, side: THREE.Vector3): THREE.Matrix4 {
  const f = forward.clone().normalize();
  const n = new THREE.Vector3().crossVectors(f, side).normalize();
  const s = new THREE.Vector3().crossVectors(n, f).normalize();
  return new THREE.Matrix4().makeBasis(s, f, n);
}

/**
 * Split `q` into `twist · swing`, twist being the part about `axis`.
 *
 * Used to move forearm roll off the wrist. A palm orientation often needs more
 * roll than a wrist has; a human supplies it by pronating the forearm, and a rig
 * that does not will instead show a wrist twisted past its stop.
 */
function swingTwist(q: THREE.Quaternion, axis: THREE.Vector3) {
  const v = new THREE.Vector3(q.x, q.y, q.z);
  const projected = axis.clone().multiplyScalar(v.dot(axis));
  const twist = new THREE.Quaternion(projected.x, projected.y, projected.z, q.w);
  if (twist.lengthSq() < 1e-12) twist.identity(); else twist.normalize();
  if (twist.w < 0) { twist.x *= -1; twist.y *= -1; twist.z *= -1; twist.w *= -1; }
  return { twist, swing: twist.clone().invert().multiply(q) };
}

/** Signed rotation of a twist quaternion about its axis, radians. */
function twistAngle(twist: THREE.Quaternion, axis: THREE.Vector3): number {
  const v = new THREE.Vector3(twist.x, twist.y, twist.z);
  const sign = v.dot(axis) < 0 ? -1 : 1;
  return 2 * Math.atan2(v.length() * sign, twist.w);
}

export interface RetargetOptions {
  /** How much the (small-magnitude) z channel is trusted for body landmarks. */
  depthScale?: number;
  /**
   * Minimum clearance in front of the torso for any wrist target, as a multiple
   * of the measured torso half-depth.
   *
   * A hard clamp rather than an additive bias. Depth in the source is weak and
   * noisy, so a bias large enough to clear the chest on bad frames overshoots on
   * good ones; a clamp only acts when the target would actually intersect.
   */
  torsoClearance?: number;
  /**
   * Depth trust for *hand* landmarks, separately from the body.
   *
   * These must not share a scale. The body's 0.06 was calibrated so wrist IK
   * targets stay inside the arm's reach; applying it to the 21 hand points
   * compresses the hand's depth 16-fold and flattens every handshape into the
   * image plane — after which no amount of finger solving can read correctly.
   * Measured across `demo_sequence.pose`, `z * width` puts the hand's per-frame
   * depth at 0.72 palm lengths (median), which is what a real hand measures, so
   * hands take the depth channel at full strength.
   */
  handDepthScale?: number;
  /**
   * Largest believable hand depth, in palm lengths, before the frame's z channel
   * is compressed back into range. The median is 0.72 but p90 reaches 4.2, and
   * those outliers would otherwise throw the palm frame right over.
   */
  maxHandDepth?: number;
  /**
   * Wrist range of motion, degrees, as separate flexion/extension and
   * radial/ulnar deviation limits rather than one cone.
   */
  wristRange?: { flex?: number; extend?: number; radial?: number; ulnar?: number };
  /** Forearm pronation/supination limit, degrees. */
  forearmTwistLimit?: number;
  /** Wrist roll left over after the forearm takes its share, degrees. */
  wristTwistLimit?: number;
  /**
   * Shoulder internal/external rotation, degrees — the third contributor to palm
   * roll.
   *
   * Turning a palm over is not a forearm movement alone; the whole arm rotates,
   * and the shoulder has the largest range of the three. Leaving it out left the
   * chain about 80° short of what the source asks for at its extremes, and the
   * shortfall varies frame to frame, which is visible as the wrist juddering
   * rather than as a pose simply being unreachable.
   */
  shoulderTwistLimit?: number;
  /**
   * How far the elbow may be swung around its own circle, degrees, to keep the
   * wrist inside its range of motion.
   *
   * Two-bone IK leaves exactly one degree of freedom once the wrist target is
   * fixed: the elbow slides around a circle about the shoulder-to-wrist axis,
   * and every point on it puts the wrist in the same place. Picking that angle
   * from the landmark elbow alone leaves the forearm pointing wherever a noisy,
   * depth-flattened landmark says — and the wrist then has to make up the whole
   * difference. Measured on `demo_sequence.pose`, that difference asks for 30°
   * of radial deviation at the median and 63° at p90, against the 20° a real
   * wrist has; the rest gets clipped, which is the hand visibly failing to
   * follow. Choosing the elbow angle so the wrist stays inside its range costs
   * nothing in wrist position and is what a person does — you move your elbow.
   */
  elbowSearchRange?: number;
  /**
   * Wider arc searched only when the normal one leaves the wrist strained.
   *
   * Searching this wide all the time is worse, not better: with a solution
   * usually available close to the signer's own elbow, a wide search just gives
   * the elbow room to wander and puts that wandering into the forearm as extra
   * motion (measured, it made the SIT span's shake worse). But the normal arc is
   * genuinely too small sometimes — in the tail of `demo_sequence.pose` a
   * reachable pose exists at +60° and at +100°, and stopping at 45° left up to
   * 53° of wrist rotation discarded, which is the hand locking against its stop.
   * So: search close first, and only reach further when close does not work.
   */
  elbowWideRange?: number;
  /**
   * Cost per degree of departing from the signer's own elbow, against a strain
   * measured in degrees outside the wrist's range.
   *
   * Higher keeps the elbow where the landmarks put it and lets the wrist take
   * the strain; lower moves the elbow freely to spare the wrist. Both extremes
   * are visible — a wandering elbow adds its own motion to the forearm, and a
   * strained wrist stops following the palm.
   */
  elbowBias?: number;
  /**
   * Smoothing for the palm orientation and the arm's IK targets.
   *
   * Without this the wrist reproduces the landmark palm's jitter essentially
   * 1:1 — measured 1.14°/frame² of angular shake against the source's 1.23°,
   * while the fingers (which are filtered and snapped to canonical shapes) sit
   * at 0.74°. A hand whose fingers are steady on a wrist that is not looks worse
   * than one where both shake.
   */
  smoothing?: {
    palmCutoff?: number; palmMaxRate?: number;
    targetCutoff?: number;
    /** Degrees per second above which a palm measurement is not believed. */
    palmGateRate?: number;
    /** Frames an unbelievable palm orientation must persist before it is taken as real. */
    palmGatePatience?: number;
  };
  /** Passed to each hand's `HandshapeTracker`. */
  handshape?: TrackerOptions;
}

export class PoseRetargeter {
  private readonly avatar: LoadedAvatar;
  private readonly depthScale: number;
  private readonly handDepthScale: number;
  private readonly maxHandDepth: number;
  private readonly torsoClearance: number;
  private readonly wristRange: { flex: number; extend: number; radial: number; ulnar: number };
  private readonly forearmTwistLimit: number;
  private readonly wristTwistLimit: number;
  private readonly shoulderTwistLimit: number;
  private readonly elbowSearchRange: number;
  private readonly elbowWideRange: number;
  private readonly elbowBias: number;
  private readonly palmFilter: Record<"left" | "right", SmoothedRotation>;
  private readonly palmGate: Record<"left" | "right", GlitchGate>;
  /** Frames the palm measurement was rejected as impossible, for verification. */
  readonly lastPalmRejected: Record<"left" | "right", boolean> = { left: false, right: false };
  private readonly targetCutoff: number;
  private readonly targetFilter: Record<"left" | "right", { wrist: SmoothedVector3; elbow: SmoothedVector3 }>;
  readonly driven = new Set<HumanoidBone>();
  lastHandsSolved = 0;
  /** Wrist target error in avatar units, for verification. */
  lastReachError = 0;
  /** Matched handshape per hand on the last frame, for the HUD and verification. */
  readonly lastHand: Record<"left" | "right", TrackedHand | null> = { left: null, right: null };
  /** Wrist solve diagnostics, degrees, for verification. */
  readonly lastWrist: Record<"left" | "right", {
    twistWanted: number; twistApplied: number;
    swingWanted: number; swingApplied: number; wristMoved: number;
    bendWanted: number; bendApplied: number;
    deviateWanted: number; deviateApplied: number;
  } | null> = { left: null, right: null };
  /** Filtered palm orientation target, before any joint limit, for verification. */
  readonly lastPalmTarget: Record<"left" | "right", THREE.Quaternion | null> = { left: null, right: null };
  /** Arm extension as a fraction of full reach, for verification. */
  readonly lastExtension: Record<"left" | "right", number> = { left: 0, right: 0 };
  /** Wrist target this frame, so the shoulder roll can re-aim the arm onto it. */
  private readonly armTarget: Record<"left" | "right", THREE.Vector3 | null> = { left: null, right: null };
  /** Elbow the landmark pole hint produced, the centre of the search below. */
  private readonly armElbow: Record<"left" | "right", THREE.Vector3 | null> = { left: null, right: null };
  /** Elbow swing actually used, degrees, for verification. */
  readonly lastElbowSwing: Record<"left" | "right", number> = { left: 0, right: 0 };

  private readonly tracker: Record<"left" | "right", HandshapeTracker>;

  /** Avatar rest measurements, captured once at rest. */
  private rig: {
    shoulderSpan: number;
    /** Z in front of which a wrist must stay to avoid entering the torso. */
    torsoFrontZ: number;
    arm: Record<"left" | "right", { upper: number; lower: number }>;
    hand: Record<"left" | "right", HandCalibration | null>;
  } | null = null;

  constructor(avatar: LoadedAvatar, options: RetargetOptions = {}) {
    this.avatar = avatar;
    // Calibrated, not guessed: at 1.0 the z axis spans 3937px against a 384px
    // shoulder span - 10.25x - so depth swamped x/y and pushed every IK target
    // outside the arm's reach. A sweep found wrist IK exact (error 0) at 0.25
    // and below; 0.06 keeps depth span near 0.6x shoulder width, which is a
    // plausible amount of forward motion for signing.
    this.depthScale = options.depthScale ?? 0.06;
    this.handDepthScale = options.handDepthScale ?? 1.0;
    this.maxHandDepth = options.maxHandDepth ?? 2.5;
    this.torsoClearance = options.torsoClearance ?? 1.15;
    // Clinical wrist range: flexion and extension are generous, deviation is not.
    this.wristRange = {
      flex: options.wristRange?.flex ?? 80,
      extend: options.wristRange?.extend ?? 70,
      radial: options.wristRange?.radial ?? 20,
      ulnar: options.wristRange?.ulnar ?? 35,
    };
    this.forearmTwistLimit = options.forearmTwistLimit ?? 85;
    this.wristTwistLimit = options.wristTwistLimit ?? 25;
    this.shoulderTwistLimit = options.shoulderTwistLimit ?? 60;
    this.elbowSearchRange = options.elbowSearchRange ?? 45;
    this.elbowWideRange = options.elbowWideRange ?? 110;
    this.elbowBias = options.elbowBias ?? 0.08;
    this.tracker = {
      left: new HandshapeTracker(options.handshape),
      right: new HandshapeTracker(options.handshape),
    };

    const s = options.smoothing ?? {};
    const palm = () => new SmoothedRotation(s.palmCutoff ?? 0.6, s.palmMaxRate ?? 750);
    this.palmFilter = { left: palm(), right: palm() };
    const gate = () => new GlitchGate(s.palmGateRate ?? 1200, s.palmGatePatience ?? 4);
    this.palmGate = { left: gate(), right: gate() };
    this.targetCutoff = s.targetCutoff ?? 2.5;
    const make = () => ({
      wrist: new SmoothedVector3(this.targetCutoff),
      elbow: new SmoothedVector3(this.targetCutoff),
    });
    this.targetFilter = { left: make(), right: make() };
  }

  private worldPos(name: HumanoidBone): THREE.Vector3 {
    return this.avatar.bones.get(name)!.getWorldPosition(new THREE.Vector3());
  }

  /** Measure bone lengths and hand calibration from the untouched rest pose. */
  private measureRig(): NonNullable<PoseRetargeter["rig"]> {
    if (this.rig) return this.rig;
    this.restoreRest();
    this.avatar.gltfScene.updateMatrixWorld(true);

    const arm = {} as Record<"left" | "right", { upper: number; lower: number }>;
    const hand = {} as Record<"left" | "right", HandCalibration | null>;

    for (const side of ["left", "right"] as const) {
      const shoulder = this.worldPos(`${side}UpperArm` as HumanoidBone);
      const elbow = this.worldPos(`${side}LowerArm` as HumanoidBone);
      const wrist = this.worldPos(`${side}Hand` as HumanoidBone);
      arm[side] = { upper: shoulder.distanceTo(elbow), lower: elbow.distanceTo(wrist) };
      hand[side] = calibrateHand(this.avatar, side);
    }

    // Torso depth measured from the mesh itself, so the clamp adapts to whatever
    // avatar is loaded instead of assuming this rig's proportions.
    const box = new THREE.Box3().setFromObject(this.avatar.gltfScene);
    const chestZ = this.worldPos("chest").z;
    const halfDepth = (box.max.z - box.min.z) / 2;

    this.rig = {
      shoulderSpan: this.worldPos("leftUpperArm").distanceTo(this.worldPos("rightUpperArm")),
      torsoFrontZ: chestZ + halfDepth * this.torsoClearance,
      arm,
      hand,
    };
    return this.rig;
  }

  /**
   * Two-bone IK: rotate the arm so the wrist lands **on** the target.
   *
   * Directional aiming only matched bone directions, so the wrist ended up
   * wherever the avatar's own bone lengths put it — measured error up to 0.4994
   * against a total arm length of 0.3957, and hands that should be 0.887 apart
   * rendered 0.519 apart. Two hands can never meet under direction matching;
   * they can only meet if the wrist position itself is solved for.
   *
   * The elbow is placed by the law of cosines, with the real landmark elbow used
   * as the pole hint so the arm bends the way the signer's did.
   */
  private solveArm(
    side: "left" | "right",
    target: THREE.Vector3,
    poleHint: THREE.Vector3,
  ): number {
    const rig = this.measureRig();
    const { upper, lower } = rig.arm[side];
    const upperBone = `${side}UpperArm` as HumanoidBone;
    const lowerBone = `${side}LowerArm` as HumanoidBone;

    const shoulder = this.worldPos(upperBone);
    const toTarget = target.clone().sub(shoulder);
    const reach = upper + lower;
    // Clamp just inside full extension: an exactly-straight arm has no defined
    // elbow circle and the joint snaps.
    const distance = THREE.MathUtils.clamp(
      toTarget.length(),
      Math.abs(upper - lower) + 1e-4,
      reach - 1e-4,
    );
    const axis = toTarget.clone().normalize();
    this.lastExtension[side] = toTarget.length() / reach;
    this.armTarget[side] = target.clone();

    const along = (distance * distance + upper * upper - lower * lower) / (2 * distance);
    const radius = Math.sqrt(Math.max(0, upper * upper - along * along));
    const centre = shoulder.clone().addScaledVector(axis, along);

    // Pole: the landmark elbow, projected perpendicular to the shoulder->target
    // axis. Falls back to "forward and down" if the hint is degenerate.
    let pole = poleHint.clone().sub(shoulder);
    pole.addScaledVector(axis, -pole.dot(axis));
    if (pole.lengthSq() < 1e-8) {
      pole = new THREE.Vector3(0, -1, 0.5);
      pole.addScaledVector(axis, -pole.dot(axis));
    }
    pole.normalize();

    const elbow = centre.clone().addScaledVector(pole, radius);
    this.armElbow[side] = elbow.clone();

    this.aim(upperBone, elbow.clone().sub(shoulder));
    this.aim(lowerBone, target.clone().sub(this.worldPos(lowerBone)));

    return this.worldPos(`${side}Hand` as HumanoidBone).distanceTo(target);
  }

  /**
   * Convert a landmark to the avatar's world axes.
   *
   * `.pose` x/y are pixels while z stays in MediaPipe's normalised units, so z
   * is multiplied by the frame width to bring it onto the same scale. Image y
   * runs downward and the avatar's +Y is up, so y is negated. The avatar faces
   * +Z with its left at +X (measured in T1.11), matching how a front-facing
   * signer's left appears at larger image x — so x needs no mirroring.
   *
   * Negating exactly two axes keeps the determinant positive, so handedness
   * survives the mapping: a left hand in the source stays a left hand here,
   * which is what lets `hand_anatomy` take chirality from the side label.
   */
  private toWorld(point: PosePoint, sequence: PoseSequence, depthScale = this.depthScale): THREE.Vector3 {
    return new THREE.Vector3(
      point.X,
      -point.Y,
      -(point.Z ?? 0) * sequence.width * depthScale,
    );
  }

  /**
   * The 21 hand landmarks in avatar axes, at full depth, with blown-up frames
   * compressed rather than discarded.
   *
   * Only directions are read from these, so no translation or scaling to avatar
   * units is needed — but the *relative* scale of z matters completely, which is
   * why it does not share the body's depth trust.
   */
  private handPoints(points: PosePoint[], sequence: PoseSequence): THREE.Vector3[] {
    const pts = points.slice(0, 21).map((p) => this.toWorld(p, sequence, this.handDepthScale));
    const palm = Math.hypot(
      pts[MIDDLE_MCP].x - pts[WRIST].x,
      pts[MIDDLE_MCP].y - pts[WRIST].y,
    ) || 1;

    let lo = Infinity, hi = -Infinity;
    for (const p of pts) { lo = Math.min(lo, p.z); hi = Math.max(hi, p.z); }
    const extent = (hi - lo) / palm;
    if (extent > this.maxHandDepth) {
      const mid = (lo + hi) / 2;
      const k = this.maxHandDepth / extent;
      for (const p of pts) p.z = mid + (p.z - mid) * k;
    }
    return pts;
  }

  /**
   * Rotate a bone so its own pointing axis aligns with `target`.
   *
   * The axis comes from the rest pose in the bone's **local** space and is
   * transformed by the bone's current world rotation, so it moves as the bone
   * moves and the solve converges. Deriving the direction from world joint
   * positions instead breaks leaf bones: a fingertip's own rotation does not
   * change its world position, so the delta never shrinks and the bone spins
   * indefinitely.
   *
   * Arms only. Fingers are posed from scalar joint angles instead — a rotation
   * that aligns one direction with another leaves the perpendicular roll
   * unconstrained, which is exactly the freedom a hinge does not have.
   */
  private aim(name: HumanoidBone, target: THREE.Vector3): boolean {
    const bone = this.avatar.bones.get(name);
    const rest = this.avatar.restPose.get(name);
    if (!bone || !bone.parent || !rest || target.lengthSq() < 1e-9) return false;

    const direction = target.clone().normalize();
    const parentWorld = bone.parent.getWorldQuaternion(new THREE.Quaternion());

    // Where the bone's axis points if the bone sits at its rest rotation.
    const restWorldAxis = rest.axis
      .clone()
      .applyQuaternion(rest.quaternion)
      .applyQuaternion(parentWorld)
      .normalize();

    const delta = new THREE.Quaternion().setFromUnitVectors(restWorldAxis, direction);
    const world = delta.multiply(parentWorld.clone().multiply(rest.quaternion));
    const local = parentWorld.clone().invert().multiply(world);

    bone.quaternion.copy(local);
    bone.updateMatrixWorld(true);
    this.driven.add(name);
    return true;
  }

  /**
   * Orient the wrist from the full palm frame, spilling excess roll into the
   * forearm.
   *
   * Aiming the hand bone at `middleMCP − wrist` fixed only where the palm points
   * and left its roll to whatever minimal rotation happened to fall out — but
   * palm orientation is a contrastive parameter in ISL, not a detail, and a
   * handshape rendered at an arbitrary roll is a different sign or no sign.
   * Matching the whole basis fixes it. The forearm then takes the pronation,
   * because it is where a human takes it and because the wrist cannot.
   *
   * Rolling the forearm about its own long axis does not move the wrist, so the
   * arm IK solved just before this stays exact.
   */
  private orientHand(side: "left" | "right", points: THREE.Vector3[], dt: number): void {
    const rig = this.measureRig();
    const calibration = rig.hand[side];
    const handName = `${side}Hand` as HumanoidBone;
    const lowerName = `${side}LowerArm` as HumanoidBone;
    const hand = this.avatar.bones.get(handName);
    const lower = this.avatar.bones.get(lowerName);
    const rest = this.avatar.restPose.get(handName);
    if (!calibration || !hand?.parent || !rest) return;

    const frame = palmFrame(
      points[WRIST], points[MIDDLE_MCP], points[INDEX_MCP], points[LITTLE_MCP], side,
    );
    // Smoothed in world space, before anything is solved. The palm's measured
    // orientation is the noisiest input in the whole solve: it comes from cross
    // products of short in-hand vectors whose z is the weakest channel there is.
    const measured = new THREE.Quaternion().setFromRotationMatrix(
      palmBasis(frame).multiply(calibration.palmLocal.clone().invert()),
    );
    // Reject first, then smooth. A measurement that says the hand turned over in
    // one frame is not a fast measurement, it is a wrong one, and averaging it
    // in produces an orientation between palm-up and palm-down that the wrist
    // cannot reach at all.
    const believed = this.palmGate[side].accept(measured, dt);
    this.lastPalmRejected[side] = believed.angleTo(measured) > 1e-6;
    const desiredWorld = this.palmFilter[side].filter(believed, dt);
    this.lastPalmTarget[side] = desiredWorld.clone();

    const parentWorld = hand.parent.getWorldQuaternion(new THREE.Quaternion());
    const delta = parentWorld.clone().invert()
      .multiply(desiredWorld)
      .multiply(rest.quaternion.clone().invert());

    // Captured before the elbow search, which must also leave the wrist put.
    const before = hand.getWorldPosition(new THREE.Vector3());

    // Place the elbow first: it costs the wrist nothing in position and decides
    // how much the wrist is asked for in the first place.
    this.placeElbow(side, calibration, desiredWorld);
    delta.copy(hand.parent.getWorldQuaternion(new THREE.Quaternion()).invert()
      .multiply(desiredWorld)
      .multiply(rest.quaternion.clone().invert()));

    const axis = calibration.wrist.axis;
    const wanted = swingTwist(delta, axis);
    let total = twistAngle(wanted.twist, axis);
    const twistWanted = total;

    // Spend the roll down the arm the way a human does: shoulder first for
    // whatever exceeds the forearm's range, then the forearm, then the wrist.
    // Rolling the upper arm about its own bone axis leaves the elbow where it
    // is, so re-aiming the forearm afterwards puts the wrist back exactly on its
    // IK target and the reach stays solved.
    const beyondForearm = Math.max(
      0, Math.abs(THREE.MathUtils.radToDeg(total)) - this.forearmTwistLimit,
    );
    let shoulderShare = 0;
    if (beyondForearm > 0.5) {
      shoulderShare = THREE.MathUtils.degToRad(
        Math.sign(total) * softLimit(beyondForearm, this.shoulderTwistLimit),
      );
      const rolled = this.rollUpperArm(side, shoulderShare);
      if (rolled) {
        const parentNow = hand.parent.getWorldQuaternion(new THREE.Quaternion());
        delta.copy(parentNow.invert()
          .multiply(desiredWorld)
          .multiply(rest.quaternion.clone().invert()));
        total = twistAngle(swingTwist(delta, axis).twist, axis);
      } else {
        shoulderShare = 0;
      }
    }

    // The forearm takes what is left, softly: a hard stop here is the wrist
    // visibly breaking, and it was being hit on 11% of frames.
    const forearmShare = THREE.MathUtils.degToRad(
      softLimit(THREE.MathUtils.radToDeg(total), this.forearmTwistLimit),
    );

    if (lower && Math.abs(forearmShare) > 1e-6) {
      const applied = new THREE.Quaternion().setFromAxisAngle(axis, forearmShare);
      lower.quaternion.multiply(applied);
      lower.updateMatrixWorld(true);
      this.driven.add(lowerName);
      // What the forearm absorbed is no longer the wrist's to supply.
      delta.premultiply(applied.invert());
    }

    const split = swingTwist(delta, axis);
    const residual = THREE.MathUtils.degToRad(
      softLimit(THREE.MathUtils.radToDeg(twistAngle(split.twist, axis)), this.wristTwistLimit),
    );
    const report = { bendWanted: 0, bendApplied: 0, deviateWanted: 0, deviateApplied: 0 };
    const constrained = new THREE.Quaternion()
      .setFromAxisAngle(axis, residual)
      .multiply(this.limitSwing(split.swing, calibration, report));

    hand.quaternion.copy(constrained.multiply(rest.quaternion));
    hand.updateMatrixWorld(true);
    this.driven.add(handName);

    const swingWanted = 2 * Math.acos(THREE.MathUtils.clamp(Math.abs(wanted.swing.w), -1, 1));
    const swingApplied = 2 * Math.acos(THREE.MathUtils.clamp(
      Math.abs(this.limitSwing(split.swing, calibration).w), -1, 1));
    this.lastWrist[side] = {
      twistWanted: THREE.MathUtils.radToDeg(twistWanted),
      twistApplied: THREE.MathUtils.radToDeg(shoulderShare + forearmShare + residual),
      swingWanted: THREE.MathUtils.radToDeg(swingWanted),
      swingApplied: THREE.MathUtils.radToDeg(swingApplied),
      wristMoved: before.distanceTo(hand.getWorldPosition(new THREE.Vector3())),
      ...report,
    };
  }

  /**
   * Swing the elbow around its circle, keeping the wrist exactly where the IK
   * put it.
   *
   * The circle is centred on the shoulder-to-wrist axis, so both bone lengths
   * and the wrist position are preserved by construction — this only changes
   * which way the forearm rolls and therefore what the wrist is asked to do.
   */
  private swingElbow(side: "left" | "right", radians: number): boolean {
    const target = this.armTarget[side];
    const base = this.armElbow[side];
    if (!target || !base) return false;
    const upperName = `${side}UpperArm` as HumanoidBone;
    const lowerName = `${side}LowerArm` as HumanoidBone;

    const shoulder = this.worldPos(upperName);
    const axis = target.clone().sub(shoulder);
    if (axis.lengthSq() < 1e-12) return false;
    axis.normalize();

    const elbow = base.clone().sub(shoulder).applyAxisAngle(axis, radians).add(shoulder);
    this.aim(upperName, elbow.clone().sub(shoulder));
    this.aim(lowerName, target.clone().sub(this.worldPos(lowerName)));
    return true;
  }

  /**
   * How far outside its range of motion the wrist would be, for a given arm
   * pose. Zero means the palm orientation is reachable as posed.
   */
  private wristStrain(
    side: "left" | "right",
    calibration: HandCalibration,
    desiredWorld: THREE.Quaternion,
  ): number {
    const hand = this.avatar.bones.get(`${side}Hand` as HumanoidBone);
    const rest = this.avatar.restPose.get(`${side}Hand` as HumanoidBone);
    if (!hand?.parent || !rest) return 0;

    const delta = hand.parent.getWorldQuaternion(new THREE.Quaternion()).invert()
      .multiply(desiredWorld)
      .multiply(rest.quaternion.clone().invert());
    const { axis, flex, deviate } = calibration.wrist;
    const split = swingTwist(delta, axis);

    const q = split.swing.clone().normalize();
    if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    const rotation = new THREE.Vector3(q.x, q.y, q.z);
    const angle = 2 * Math.acos(THREE.MathUtils.clamp(q.w, -1, 1));
    let bend = 0, sideways = 0;
    if (rotation.lengthSq() > 1e-12 && angle > 1e-9) {
      rotation.normalize().multiplyScalar(THREE.MathUtils.radToDeg(angle));
      bend = rotation.dot(flex);
      sideways = rotation.dot(deviate);
    }
    const twist = Math.abs(THREE.MathUtils.radToDeg(twistAngle(split.twist, axis)));

    const over = (value: number, low: number, high: number) =>
      Math.max(0, value - high, low - value);
    return over(bend, -this.wristRange.extend, this.wristRange.flex)
      + over(sideways, -this.wristRange.ulnar, this.wristRange.radial)
      // Twist has the shoulder and forearm behind it, so it strains far later.
      + 0.3 * Math.max(0, twist - this.forearmTwistLimit - this.shoulderTwistLimit);
  }

  /**
   * Pick the elbow angle that keeps the wrist reachable, staying as close to the
   * signer's own elbow as that allows.
   *
   * A sampled search rather than a solve: measured across this sequence the cost
   * is flat zero over a wide arc — typically 60–90° of the circle — and rises
   * linearly outside it, so a gradient has nothing to follow through the middle.
   * Sampling finds the plateau and the distance term then picks the point in it
   * nearest the signer's own elbow, which is why the chosen angle is 0° whenever
   * the landmark elbow already works.
   */
  private placeElbow(
    side: "left" | "right",
    calibration: HandCalibration,
    desiredWorld: THREE.Quaternion,
  ): void {
    if (!this.armTarget[side] || !this.armElbow[side]) return;

    /** Best angle over one arc, and the strain it leaves. */
    const search = (range: number, samples: number) => {
      let best = 0, bestCost = Infinity, bestStrain = Infinity;
      for (let i = 0; i < samples; i++) {
        const angle = -range + (2 * range * i) / (samples - 1);
        if (!this.swingElbow(side, angle)) return null;
        const strain = this.wristStrain(side, calibration, desiredWorld);
        // Prefer the signer's own elbow: only depart from it to buy reachability.
        const cost = strain + this.elbowBias * Math.abs(THREE.MathUtils.radToDeg(angle));
        if (cost < bestCost) { bestCost = cost; best = angle; bestStrain = strain; }
      }
      return { best, strain: bestStrain };
    };

    const near = search(THREE.MathUtils.degToRad(this.elbowSearchRange), 21);
    if (!near) return;
    // Only reach wider when reaching close by has left the wrist genuinely short.
    const chosen = near.strain <= 3
      ? near
      : search(THREE.MathUtils.degToRad(this.elbowWideRange), 45) ?? near;

    this.swingElbow(side, chosen.best);
    this.lastElbowSwing[side] = THREE.MathUtils.radToDeg(chosen.best);
  }

  /**
   * Roll the upper arm about its own bone axis and re-solve the forearm onto the
   * same wrist target.
   *
   * The elbow lies along the upper arm's axis, so this rotation does not move
   * it; re-aiming the forearm then restores the wrist position exactly. What it
   * does change is the forearm's roll, which is what the palm needed.
   */
  private rollUpperArm(side: "left" | "right", radians: number): boolean {
    const upperName = `${side}UpperArm` as HumanoidBone;
    const lowerName = `${side}LowerArm` as HumanoidBone;
    const upper = this.avatar.bones.get(upperName);
    const lower = this.avatar.bones.get(lowerName);
    const target = this.armTarget[side];
    if (!upper || !lower || !target) return false;

    const axis = lower.position.lengthSq() > 1e-12
      ? lower.position.clone().normalize()
      : new THREE.Vector3(0, 1, 0);
    upper.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(axis, radians));
    upper.updateMatrixWorld(true);
    this.driven.add(upperName);

    this.aim(lowerName, target.clone().sub(this.worldPos(lowerName)));
    return true;
  }

  /**
   * Hold a wrist swing inside a real wrist's travel.
   *
   * The swing is resolved onto the wrist's own two axes and each is limited on
   * its own, because their ranges differ by a factor of three: the hand flexes
   * and extends 70–80° but deviates only 20° radially and 35° toward the little
   * finger. A single cone wide enough for flexion lets the hand bend sideways
   * into a pose no wrist reaches, and one narrow enough for deviation stops it
   * short of ordinary flexion. Both limits are soft, so the wrist eases into its
   * travel rather than stopping dead against it.
   */
  private limitSwing(
    swing: THREE.Quaternion,
    calibration: HandCalibration,
    report?: { bendWanted: number; bendApplied: number; deviateWanted: number; deviateApplied: number },
  ): THREE.Quaternion {
    const q = swing.clone().normalize();
    if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    const angle = 2 * Math.acos(THREE.MathUtils.clamp(q.w, -1, 1));
    const rotation = new THREE.Vector3(q.x, q.y, q.z);
    if (rotation.lengthSq() < 1e-12 || angle < 1e-9) {
      if (report) { report.bendWanted = report.bendApplied = 0; report.deviateWanted = report.deviateApplied = 0; }
      return new THREE.Quaternion();
    }
    rotation.normalize().multiplyScalar(THREE.MathUtils.radToDeg(angle));

    const { flex, deviate } = calibration.wrist;
    const bend = softRange(rotation.dot(flex), -this.wristRange.extend, this.wristRange.flex);
    const sideways = softRange(rotation.dot(deviate), -this.wristRange.ulnar, this.wristRange.radial);

    if (report) {
      report.bendWanted = rotation.dot(flex); report.bendApplied = bend;
      report.deviateWanted = rotation.dot(deviate); report.deviateApplied = sideways;
    }
    const limited = flex.clone().multiplyScalar(THREE.MathUtils.degToRad(bend))
      .addScaledVector(deviate, THREE.MathUtils.degToRad(sideways));
    const magnitude = limited.length();
    return magnitude < 1e-9
      ? new THREE.Quaternion()
      : new THREE.Quaternion().setFromAxisAngle(limited.normalize(), magnitude);
  }

  applyFrame(sequence: PoseSequence, index: number, dt = 1 / sequence.fps): number {
    const frame = sequence.frames[index];
    if (!frame) return 0;
    const rig = this.measureRig();
    this.restoreRest();
    this.avatar.gltfScene.updateMatrixWorld(true);

    let written = 0;
    const poseNames = sequence.componentPoints["POSE_LANDMARKS"] ?? [];
    const posePoints = frame["POSE_LANDMARKS"] ?? [];
    const body: Record<string, PosePoint> = {};
    poseNames.forEach((n, i) => { if (posePoints[i]) body[n] = posePoints[i]; });

    const tracked = (p?: PosePoint) => !!p && (p.C ?? 0) > 0;
    if (!tracked(body["LEFT_SHOULDER"]) || !tracked(body["RIGHT_SHOULDER"])) return 0;

    // Map landmark space into avatar space by matching shoulder spans, so the
    // signer's proportions transfer regardless of frame size or camera distance.
    const lmLeft = this.toWorld(body["LEFT_SHOULDER"], sequence);
    const lmRight = this.toWorld(body["RIGHT_SHOULDER"], sequence);
    const lmSpan = lmLeft.distanceTo(lmRight);
    if (lmSpan < 1e-6) return 0;
    const scale = rig.shoulderSpan / lmSpan;
    const lmOrigin = lmLeft.clone().add(lmRight).multiplyScalar(0.5);
    const avOrigin = this.worldPos("leftUpperArm")
      .add(this.worldPos("rightUpperArm"))
      .multiplyScalar(0.5);
    // Map into avatar space, then push forward only if the target would land
    // inside the torso. Hands that already clear the chest are left alone.
    const toAvatar = (p: PosePoint, clampToFront = false) => {
      const v = this.toWorld(p, sequence)
        .sub(lmOrigin)
        .multiplyScalar(scale)
        .add(avOrigin);
      if (clampToFront) v.z = Math.max(v.z, rig.torsoFrontZ);
      return v;
    };

    this.lastReachError = 0;
    for (const side of ["left", "right"] as const) {
      const S = side.toUpperCase();
      const elbow = body[`${S}_ELBOW`];
      const wrist = body[`${S}_WRIST`];
      if (!tracked(elbow) || !tracked(wrist)) continue;
      // Filtered before the solve, so shoulder, elbow and wrist all inherit one
      // steady target rather than each chasing its own noise.
      const filter = this.targetFilter[side];
      this.lastReachError = Math.max(
        this.lastReachError,
        this.solveArm(
          side,
          filter.wrist.filter(toAvatar(wrist, true), dt),
          filter.elbow.filter(toAvatar(elbow, true), dt),
        ),
      );
      written += 2;
    }

    // Hands: palm orientation from the landmark frame, fingers from measured
    // joint angles snapped toward a canonical handshape. Both hands are posed
    // every frame — an untracked hand coasts rather than snapping back to rest.
    this.lastHandsSolved = 0;
    for (const [component, side] of [
      ["LEFT_HAND_LANDMARKS", "left"],
      ["RIGHT_HAND_LANDMARKS", "right"],
    ] as const) {
      const calibration = rig.hand[side];
      if (!calibration) continue;

      const raw = frame[component];
      const usable = !!raw && raw.length >= 21 && raw.slice(0, 21).some((p) => (p.C ?? 0) > 0);

      let measured = null;
      if (usable) {
        const points = this.handPoints(raw!, sequence);
        this.avatar.gltfScene.updateMatrixWorld(true);
        this.orientHand(side, points, dt);
        measured = measureHand(points, side);
        this.lastHandsSolved += 1;
        written += 1;
      }

      const result = this.tracker[side].update(measured, dt);
      this.lastHand[side] = result;
      written += applyHandAngles(this.avatar, calibration, result.angles);
      for (const [name] of calibration.axes) this.driven.add(name);
    }

    return written;
  }

  /** Return every bone to its captured rest rotation, leaving solver state alone. */
  private restoreRest(): void {
    for (const [name, rest] of this.avatar.restPose) {
      this.avatar.bones.get(name)?.quaternion.copy(rest.quaternion);
    }
    this.driven.clear();
    this.lastHandsSolved = 0;
  }

  /**
   * Drop sign-level state but keep motion continuity.
   *
   * For the playback loop, where the last frame of the sequence is followed by
   * the first. That is a cut in the data — 149° of palm rotation and a third of
   * a forearm length of wrist travel between two adjacent frames — but it is a
   * cut in *playback*, not in the signing, and a full reset renders it as a
   * teleport. Keeping the orientation and position filters lets the rate cap
   * carry the avatar across in a few frames instead, while the handshape
   * trackers still start clean so no stale shape is dragged into the new pass.
   */
  rewind(): void {
    this.restoreRest();
    this.tracker.left.reset();
    this.tracker.right.reset();
    this.lastHand.left = null;
    this.lastHand.right = null;
  }

  /**
   * Rest the rig **and** drop temporal state.
   *
   * Filtering and handshape hysteresis carry across frames, which is what keeps a
   * held sign steady — but after a scrub or a loop the previous frame is not the
   * previous moment, and carrying that state in would drag a stale handshape
   * into the new one.
   */
  reset(): void {
    this.restoreRest();
    this.tracker.left.reset();
    this.tracker.right.reset();
    this.palmFilter.left.reset();
    this.palmFilter.right.reset();
    this.palmGate.left.reset();
    this.palmGate.right.reset();
    for (const side of ["left", "right"] as const) {
      this.targetFilter[side].wrist.reset();
      this.targetFilter[side].elbow.reset();
    }
    this.lastHand.left = null;
    this.lastHand.right = null;
    this.lastWrist.left = null;
    this.lastWrist.right = null;
  }
}
