/**
 * Pose retargeting — playback of looked-up pose sequences (T1.12).
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
 *    x and y. Anything reading them as one 3D space sees a flat plane. Depth is
 *    real, but only after `z * width`.
 *
 * Aiming each bone along the direction between its two landmarks sidesteps both.
 * It works from the rig's own rest geometry, so it needs no axis convention, and
 * it drives fingers the same way it drives arms — which is what makes handshapes
 * actually readable rather than folded.
 */

import * as THREE from "three";
import type { HumanoidBone, LoadedAvatar } from "./loader";

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

/** MediaPipe hand landmark indices. */
const H = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
} as const;

/** Finger bone chains, parent first. Driving all three segments is what makes a
 *  handshape legible; driving only the proximal joint leaves fingers looking
 *  folded regardless of the source data. */
function fingerChain(side: "left" | "right"): Array<[HumanoidBone, number, number]> {
  const s = side;
  return [
    [`${s}ThumbProximal` as HumanoidBone, H.THUMB_CMC, H.THUMB_MCP],
    [`${s}ThumbIntermediate` as HumanoidBone, H.THUMB_MCP, H.THUMB_IP],
    [`${s}ThumbDistal` as HumanoidBone, H.THUMB_IP, H.THUMB_TIP],
    [`${s}IndexProximal` as HumanoidBone, H.INDEX_MCP, H.INDEX_PIP],
    [`${s}IndexIntermediate` as HumanoidBone, H.INDEX_PIP, H.INDEX_DIP],
    [`${s}IndexDistal` as HumanoidBone, H.INDEX_DIP, H.INDEX_TIP],
    [`${s}MiddleProximal` as HumanoidBone, H.MIDDLE_MCP, H.MIDDLE_PIP],
    [`${s}MiddleIntermediate` as HumanoidBone, H.MIDDLE_PIP, H.MIDDLE_DIP],
    [`${s}MiddleDistal` as HumanoidBone, H.MIDDLE_DIP, H.MIDDLE_TIP],
    [`${s}RingProximal` as HumanoidBone, H.RING_MCP, H.RING_PIP],
    [`${s}RingIntermediate` as HumanoidBone, H.RING_PIP, H.RING_DIP],
    [`${s}RingDistal` as HumanoidBone, H.RING_DIP, H.RING_TIP],
    [`${s}LittleProximal` as HumanoidBone, H.PINKY_MCP, H.PINKY_PIP],
    [`${s}LittleIntermediate` as HumanoidBone, H.PINKY_PIP, H.PINKY_DIP],
    [`${s}LittleDistal` as HumanoidBone, H.PINKY_DIP, H.PINKY_TIP],
  ];
}

/**
 * Orthonormal basis, columns `[side, forward, normal]`.
 *
 * Fingers are solved in the palm's frame rather than world space. Aiming a
 * finger segment at a world direction ignores which way the palm faces, so the
 * same handshape renders differently depending on wrist orientation — the
 * "fingers bending weirdly" symptom. Expressing each segment relative to the
 * signer's palm and rebuilding it relative to the avatar's palm makes a
 * handshape mean the same thing in both.
 */
function basis(forward: THREE.Vector3, side: THREE.Vector3): THREE.Matrix4 {
  const f = forward.clone().normalize();
  const n = new THREE.Vector3().crossVectors(f, side).normalize();
  const s = new THREE.Vector3().crossVectors(n, f).normalize();
  return new THREE.Matrix4().makeBasis(s, f, n);
}

export interface RetargetOptions {
  /** How much the (small-magnitude) z channel is trusted, relative to x/y. */
  depthScale?: number;
  /**
   * Minimum clearance in front of the torso for any wrist target, as a multiple
   * of the measured torso half-depth.
   *
   * A hard clamp rather than an additive bias. Depth in the source is weak and
   * noisy, so a bias large enough to clear the chest on bad frames overshoots on
   * good ones; a clamp only acts when the target would actually intersect.
   * Signing happens in front of the body, always, so this is a real constraint
   * rather than a fudge factor.
   */
  torsoClearance?: number;
}

export class PoseRetargeter {
  private readonly avatar: LoadedAvatar;
  private readonly depthScale: number;
  private readonly torsoClearance: number;
  readonly driven = new Set<HumanoidBone>();
  lastHandsSolved = 0;
  /** Wrist target error in avatar units, for verification. */
  lastReachError = 0;

  /** Avatar rest measurements, captured once at rest. */
  private rig: {
    shoulderSpan: number;
    /** Z in front of which a wrist must stay to avoid entering the torso. */
    torsoFrontZ: number;
    arm: Record<"left" | "right", { upper: number; lower: number }>;
    palmLocal: Record<"left" | "right", THREE.Matrix4>;
  } | null = null;

  constructor(avatar: LoadedAvatar, options: RetargetOptions = {}) {
    this.avatar = avatar;
    // Calibrated, not guessed: at 1.0 the z axis spans 3937px against a 384px
    // shoulder span - 10.25x - so depth swamped x/y and pushed every IK target
    // outside the arm's reach. A sweep found wrist IK exact (error 0) at 0.25
    // and below; 0.06 keeps depth span near 0.6x shoulder width, which is a
    // plausible amount of forward motion for signing.
    this.depthScale = options.depthScale ?? 0.06;
    this.torsoClearance = options.torsoClearance ?? 1.15;
  }

  private worldPos(name: HumanoidBone): THREE.Vector3 {
    return this.avatar.bones.get(name)!.getWorldPosition(new THREE.Vector3());
  }

  /** Measure bone lengths and palm bases from the untouched rest pose. */
  private measureRig(): NonNullable<PoseRetargeter["rig"]> {
    if (this.rig) return this.rig;
    this.reset();
    this.avatar.gltfScene.updateMatrixWorld(true);

    const arm = {} as Record<"left" | "right", { upper: number; lower: number }>;
    const palmLocal = {} as Record<"left" | "right", THREE.Matrix4>;

    for (const side of ["left", "right"] as const) {
      const shoulder = this.worldPos(`${side}UpperArm` as HumanoidBone);
      const elbow = this.worldPos(`${side}LowerArm` as HumanoidBone);
      const wrist = this.worldPos(`${side}Hand` as HumanoidBone);
      arm[side] = { upper: shoulder.distanceTo(elbow), lower: elbow.distanceTo(wrist) };

      const middle = this.worldPos(`${side}MiddleProximal` as HumanoidBone);
      const index = this.worldPos(`${side}IndexProximal` as HumanoidBone);
      const little = this.worldPos(`${side}LittleProximal` as HumanoidBone);
      const world = basis(middle.clone().sub(wrist), index.clone().sub(little));
      const handQuat = this.avatar.bones
        .get(`${side}Hand` as HumanoidBone)!
        .getWorldQuaternion(new THREE.Quaternion());
      palmLocal[side] = new THREE.Matrix4()
        .makeRotationFromQuaternion(handQuat.invert())
        .multiply(world);
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
      palmLocal,
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
   */
  private toWorld(point: PosePoint, sequence: PoseSequence): THREE.Vector3 {
    return new THREE.Vector3(
      point.X,
      -point.Y,
      -(point.Z ?? 0) * sequence.width * this.depthScale,
    );
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
   * Solved directly rather than slerped toward, because a partial step leaves a
   * finger between two handshapes — which reads as neither.
   */
  private aim(name: HumanoidBone, target: THREE.Vector3, bias = 0): boolean {
    const bone = this.avatar.bones.get(name);
    const rest = this.avatar.restPose.get(name);
    if (!bone || !bone.parent || !rest || target.lengthSq() < 1e-9) return false;

    const direction = target.clone().normalize();
    if (bias) {
      direction.z += bias;
      direction.normalize();
    }

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

  applyFrame(sequence: PoseSequence, index: number): number {
    const frame = sequence.frames[index];
    if (!frame) return 0;
    const rig = this.measureRig();
    this.reset();
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
      this.lastReachError = Math.max(
        this.lastReachError,
        this.solveArm(side, toAvatar(wrist, true), toAvatar(elbow, true)),
      );
      written += 2;
    }

    // Hands: wrist orientation, then every finger segment in the palm's frame.
    this.lastHandsSolved = 0;
    for (const [component, side] of [
      ["LEFT_HAND_LANDMARKS", "left"],
      ["RIGHT_HAND_LANDMARKS", "right"],
    ] as const) {
      const points = frame[component];
      if (!points || points.length < 21) continue;
      if (points.every((p) => (p.C ?? 0) === 0)) continue;
      this.lastHandsSolved += 1;

      const at = (i: number) => this.toWorld(points[i], sequence);
      const handBone = `${side}Hand` as HumanoidBone;

      this.avatar.gltfScene.updateMatrixWorld(true);
      if (this.aim(handBone, at(H.MIDDLE_MCP).clone().sub(at(H.WRIST)))) written += 1;
      this.avatar.gltfScene.updateMatrixWorld(true);

      // Signer's palm frame this frame, and the avatar's palm frame as posed.
      const source = basis(
        at(H.MIDDLE_MCP).clone().sub(at(H.WRIST)),
        at(H.INDEX_MCP).clone().sub(at(H.PINKY_MCP)),
      );
      const target = new THREE.Matrix4()
        .makeRotationFromQuaternion(
          this.avatar.bones.get(handBone)!.getWorldQuaternion(new THREE.Quaternion()),
        )
        .multiply(rig.palmLocal[side]);
      const sourceInverse = source.clone().transpose();   // orthonormal: transpose == inverse

      for (const [bone, from, to] of fingerChain(side)) {
        if ((points[from]?.C ?? 0) === 0 || (points[to]?.C ?? 0) === 0) continue;
        const inPalm = at(to).clone().sub(at(from)).applyMatrix4(sourceInverse);
        this.avatar.gltfScene.updateMatrixWorld(true);
        if (this.aim(bone, inPalm.applyMatrix4(target))) written += 1;
      }
    }

    return written;
  }

  /** Return every driven bone to its captured rest rotation. */
  reset(): void {
    for (const [name, rest] of this.avatar.restPose) {
      this.avatar.bones.get(name)?.quaternion.copy(rest.quaternion);
    }
    this.driven.clear();
    this.lastHandsSolved = 0;
  }
}
