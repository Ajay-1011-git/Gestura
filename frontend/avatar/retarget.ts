/**
 * Kalidokit retargeting — playback of looked-up pose sequences (T1.12).
 *
 * This is the Speech->Sign direction only. It replays `.pose` sequences produced
 * by T1.9/T1.10 onto the avatar loaded in T1.11. **No live camera feed is wired
 * here**: the Sign->Speech direction extracts landmarks for recognition and does
 * not render an avatar at all.
 *
 * Four things measured from the real data rather than assumed, each of which
 * changed the implementation:
 *
 *  1. `spoken-to-signed-translation` **reduces** POSE_LANDMARKS from 33 points
 *     to 8 — shoulders, elbows, wrists, hips — and legs are not among them.
 *  2. It also **flattens depth**: every retained Z is ~256, a constant. That is
 *     what forced arms off Kalidokit's pose solver and onto 2D aim; see
 *     `ARM_CHAINS`. Hands keep Kalidokit, whose solver has real intra-hand
 *     geometry to work with.
 *  3. The rig is an **A-pose** (~39.7° droop, measured in T1.11) while Kalidokit
 *     solves against a T-pose reference, so hand rotations are composed onto the
 *     captured rest pose rather than replacing it. See `applyEuler`.
 *  4. `pose-format`'s JS `body.frames` is a Proxy over the binary buffer, not
 *     the array its own type declaration promises. See `loadPoseSequence`.
 */

import * as THREE from "three";
import { Hand as KalidoHand } from "kalidokit";
import type { HumanoidBone, LoadedAvatar } from "./loader";

/** MediaPipe pose indices for the 8 landmarks the lexicon retains. */
const POSE_INDEX: Record<string, number> = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
};
const MEDIAPIPE_POSE_LANDMARKS = 33;

interface PosePoint { X: number; Y: number; Z?: number; C?: number }
interface Landmark { x: number; y: number; z: number; visibility: number }

export interface PoseSequence {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  componentPoints: Record<string, string[]>;
  frames: Record<string, PosePoint[]>[];
}

/** Load and parse a `.pose` file using the format's own JS reader. */
export async function loadPoseSequence(url: string): Promise<PoseSequence> {
  const { Pose } = await import("pose-format");
  const parsed: any = await Pose.fromRemote(url);
  const header = parsed.header;

  const componentPoints: Record<string, string[]> = {};
  for (const component of header.components) {
    componentPoints[component.name] = component.points;
  }

  // `body.frames` is a Proxy over the binary buffer, not the array its own
  // .d.ts declares: Object.keys() is empty and .map() does not exist, but
  // integer indexing works and `_frames` carries the count. Verified against
  // pose-format@1.6.2 on 2026-09-15.
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

function normalise(point: PosePoint, width: number, height: number): Landmark {
  return {
    x: point.X / (width || 1),
    y: point.Y / (height || 1),
    z: (point.Z ?? 0) / (width || 1),
    visibility: point.C ?? 0,
  };
}

/**
 * Rebuild a 33-slot MediaPipe pose array from the lexicon's 8 retained points.
 *
 * Kalidokit reads landmarks by index, so scattering matters: a dense 8-element
 * array would put the left shoulder where the nose belongs and still "work",
 * producing confident nonsense.
 */
export function toMediaPipePose(
  frame: Record<string, PosePoint[]>,
  componentPoints: Record<string, string[]>,
  width: number,
  height: number,
): Landmark[] {
  const empty: Landmark = { x: 0, y: 0, z: 0, visibility: 0 };
  const out: Landmark[] = Array.from({ length: MEDIAPIPE_POSE_LANDMARKS }, () => ({ ...empty }));

  const points = frame["POSE_LANDMARKS"] ?? [];
  const names = componentPoints["POSE_LANDMARKS"] ?? [];
  names.forEach((name, i) => {
    const index = POSE_INDEX[name];
    if (index !== undefined && points[i]) out[index] = normalise(points[i], width, height);
  });
  return out;
}

export function toHandLandmarks(
  frame: Record<string, PosePoint[]>,
  component: "LEFT_HAND_LANDMARKS" | "RIGHT_HAND_LANDMARKS",
  width: number,
  height: number,
): Landmark[] | null {
  const points = frame[component];
  if (!points || points.length < 21) return null;
  if (points.every((p) => (p.C ?? 0) === 0)) return null;   // hand not tracked this frame
  return points.map((p) => normalise(p, width, height));
}

/**
 * Arm chains driven by 2D aim rather than Kalidokit's pose solver.
 *
 * **Why Kalidokit.Pose.solve is not used for arms.** Measured on the real
 * lexicon output on 2026-09-15: every POSE_LANDMARKS Z value is ~256, a
 * constant — `spoken-to-signed-translation` flattens depth when it reduces the
 * pose. Kalidokit's arm solver works in 3D, so with a degenerate depth plane it
 * clamps to a fixed ±1.25 rad and returns the *same* rotation on every frame.
 * That is not a Kalidokit defect; it is being handed data it cannot solve.
 *
 * Aiming each bone along the 2D shoulder→elbow→wrist direction uses the signal
 * that is actually present. For a front-facing signing avatar the image plane
 * carries nearly all of it. Kalidokit is still used for the hands (below),
 * where intra-hand geometry is real and its solver works.
 */
const ARM_CHAINS: ReadonlyArray<readonly [HumanoidBone, HumanoidBone, string, string]> = [
  ["leftUpperArm", "leftLowerArm", "LEFT_SHOULDER", "LEFT_ELBOW"],
  ["leftLowerArm", "leftHand", "LEFT_ELBOW", "LEFT_WRIST"],
  ["rightUpperArm", "rightLowerArm", "RIGHT_SHOULDER", "RIGHT_ELBOW"],
  ["rightLowerArm", "rightHand", "RIGHT_ELBOW", "RIGHT_WRIST"],
];

function handBone(key: string): HumanoidBone | undefined {
  // "LeftIndexProximal" -> "leftIndexProximal"; "LeftWrist" maps to the hand bone.
  if (/^(Left|Right)Wrist$/.test(key)) {
    return (key.startsWith("Left") ? "leftHand" : "rightHand") as HumanoidBone;
  }
  const camel = key.charAt(0).toLowerCase() + key.slice(1);
  return camel as HumanoidBone;
}

export interface RetargetOptions {
  /** Compose solved rotations onto the rig's rest pose instead of replacing it.
   *  Required for this A-pose rig; a T-pose rig could set it false. */
  composeWithRest?: boolean;
  /** 0..1 slerp factor per frame; damps jitter without lagging the motion. */
  smoothing?: number;
}

export class PoseRetargeter {
  private readonly avatar: LoadedAvatar;
  private readonly composeWithRest: boolean;
  private readonly smoothing: number;
  /** Bones this retargeter has actually written to — used by VERIFY. */
  readonly driven = new Set<HumanoidBone>();

  constructor(avatar: LoadedAvatar, options: RetargetOptions = {}) {
    this.avatar = avatar;
    this.composeWithRest = options.composeWithRest ?? true;
    this.smoothing = options.smoothing ?? 0.45;
  }

  private applyEuler(name: HumanoidBone, rotation: { x: number; y: number; z: number }): void {
    const bone = this.avatar.bones.get(name);
    if (!bone) return;

    const target = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(rotation.x, rotation.y, rotation.z, "XYZ"),
    );

    // Kalidokit's output is expressed against a T-pose reference. This rig rests
    // in an A-pose, so composing onto the rest rotation keeps the rig's own
    // shoulder geometry instead of snapping limbs to a T-pose they never had.
    if (this.composeWithRest) {
      const rest = this.avatar.restPose.get(name);
      if (rest) target.premultiply(rest.quaternion);
    }

    bone.quaternion.slerp(target, this.smoothing);
    this.driven.add(name);
  }

  /**
   * Aim a bone along a direction taken from the 2D landmarks.
   *
   * Image space is x-right / y-down; world space here is x-right / y-up, so y is
   * flipped. The avatar faces +Z and its left side is at +X (measured in T1.11),
   * which matches how a front-facing signer's left appears at larger image x —
   * so x needs no mirroring.
   */
  private aimBone(
    name: HumanoidBone,
    from: { x: number; y: number },
    to: { x: number; y: number },
  ): boolean {
    const bone = this.avatar.bones.get(name);
    const rest = this.avatar.restPose.get(name);
    if (!bone || !rest || !bone.parent) return false;

    const target = new THREE.Vector3(to.x - from.x, -(to.y - from.y), 0);
    if (target.lengthSq() < 1e-8) return false;
    target.normalize();

    // Rest direction of this bone in world space: toward its first child.
    const child = bone.children.find((c) => (c as THREE.Bone).isBone) as THREE.Bone | undefined;
    if (!child) return false;
    const restDirection = child
      .getWorldPosition(new THREE.Vector3())
      .sub(bone.getWorldPosition(new THREE.Vector3()));
    if (restDirection.lengthSq() < 1e-8) return false;
    restDirection.normalize();

    const delta = new THREE.Quaternion().setFromUnitVectors(restDirection, target);
    const world = bone.getWorldQuaternion(new THREE.Quaternion()).premultiply(delta);
    const parentWorld = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    const local = parentWorld.invert().multiply(world);

    bone.quaternion.slerp(local, this.smoothing);
    this.driven.add(name);
    return true;
  }

  /** Solve and apply one frame. Returns how many bones were written. */
  applyFrame(sequence: PoseSequence, index: number): number {
    const frame = sequence.frames[index];
    if (!frame) return 0;
    let written = 0;

    const names = sequence.componentPoints["POSE_LANDMARKS"] ?? [];
    const points = frame["POSE_LANDMARKS"] ?? [];
    const byName: Record<string, PosePoint> = {};
    names.forEach((n, i) => { if (points[i]) byName[n] = points[i]; });

    for (const [bone, , fromName, toName] of ARM_CHAINS) {
      const from = byName[fromName];
      const to = byName[toName];
      if (!from || !to || (from.C ?? 0) === 0 || (to.C ?? 0) === 0) continue;
      this.avatar.gltfScene.updateMatrixWorld(true);
      if (this.aimBone(bone, { x: from.X, y: from.Y }, { x: to.X, y: to.Y })) written += 1;
    }

    for (const [component, side] of [
      ["LEFT_HAND_LANDMARKS", "Left"],
      ["RIGHT_HAND_LANDMARKS", "Right"],
    ] as const) {
      const landmarks = toHandLandmarks(frame, component, sequence.width, sequence.height);
      if (!landmarks) continue;
      const hand = KalidoHand.solve(landmarks as any, side);
      if (!hand) continue;
      for (const [key, rotation] of Object.entries(hand as Record<string, any>)) {
        const bone = handBone(key);
        if (bone && rotation && typeof rotation.x === "number") {
          this.applyEuler(bone, rotation);
          written += 1;
        }
      }
    }

    return written;
  }

  /** Reset every driven bone to its captured rest rotation. */
  reset(): void {
    for (const [name, rest] of this.avatar.restPose) {
      this.avatar.bones.get(name)?.quaternion.copy(rest.quaternion);
    }
    this.driven.clear();
  }
}
