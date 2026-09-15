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

export interface RetargetOptions {
  /** Slerp factor per frame. Lower damps jitter but lags fast motion. */
  smoothing?: number;
  /** How much the (small-magnitude) z channel is trusted, relative to x/y. */
  depthScale?: number;
  /** Metres of forward offset applied to the whole upper body, so signing
   *  happens in front of the chest rather than intersecting it. */
  forwardBias?: number;
}

export class PoseRetargeter {
  private readonly avatar: LoadedAvatar;
  private readonly smoothing: number;
  private readonly depthScale: number;
  private readonly forwardBias: number;
  readonly driven = new Set<HumanoidBone>();
  lastHandsSolved = 0;

  constructor(avatar: LoadedAvatar, options: RetargetOptions = {}) {
    this.avatar = avatar;
    this.smoothing = options.smoothing ?? 0.4;
    this.depthScale = options.depthScale ?? 1.0;
    this.forwardBias = options.forwardBias ?? 0.35;
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

  /** Rotate a bone so its rest direction points along `target`. */
  private aim(name: HumanoidBone, target: THREE.Vector3, bias = 0): boolean {
    const bone = this.avatar.bones.get(name);
    if (!bone || !bone.parent || target.lengthSq() < 1e-9) return false;

    const direction = target.clone().normalize();
    if (bias) direction.z += bias, direction.normalize();

    // Rest direction is normally bone -> first child bone. Distal fingertip
    // bones are leaves with no child, which would skip the last joint of every
    // finger and leave fingertips permanently uncurled — 15 of 20 hand bones
    // driven instead of all 20. For a leaf, the parent -> bone direction
    // continues the chain and is the right reference.
    const origin = bone.getWorldPosition(new THREE.Vector3());
    const child = bone.children.find((c) => (c as THREE.Bone).isBone) as THREE.Bone | undefined;
    const restDirection = child
      ? child.getWorldPosition(new THREE.Vector3()).sub(origin)
      : origin.clone().sub(bone.parent.getWorldPosition(new THREE.Vector3()));
    if (restDirection.lengthSq() < 1e-9) return false;
    restDirection.normalize();

    const delta = new THREE.Quaternion().setFromUnitVectors(restDirection, direction);
    const world = bone.getWorldQuaternion(new THREE.Quaternion()).premultiply(delta);
    const local = bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(world);

    bone.quaternion.slerp(local, this.smoothing);
    this.driven.add(name);
    return true;
  }

  applyFrame(sequence: PoseSequence, index: number): number {
    const frame = sequence.frames[index];
    if (!frame) return 0;
    let written = 0;

    const poseNames = sequence.componentPoints["POSE_LANDMARKS"] ?? [];
    const posePoints = frame["POSE_LANDMARKS"] ?? [];
    const body: Record<string, PosePoint> = {};
    poseNames.forEach((n, i) => { if (posePoints[i]) body[n] = posePoints[i]; });

    const tracked = (p?: PosePoint) => p && (p.C ?? 0) > 0;
    const vec = (a: string, b: string) =>
      this.toWorld(body[b], sequence).sub(this.toWorld(body[a], sequence));

    // Arms, parent before child so each aim reads an up-to-date world matrix.
    const arms: Array<[HumanoidBone, string, string, number]> = [
      ["leftUpperArm", "LEFT_SHOULDER", "LEFT_ELBOW", this.forwardBias],
      ["leftLowerArm", "LEFT_ELBOW", "LEFT_WRIST", 0],
      ["rightUpperArm", "RIGHT_SHOULDER", "RIGHT_ELBOW", this.forwardBias],
      ["rightLowerArm", "RIGHT_ELBOW", "RIGHT_WRIST", 0],
    ];
    for (const [bone, from, to, bias] of arms) {
      if (!tracked(body[from]) || !tracked(body[to])) continue;
      this.avatar.gltfScene.updateMatrixWorld(true);
      if (this.aim(bone, vec(from, to), bias)) written += 1;
    }

    // Hands: wrist orientation first, then every finger segment.
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

      this.avatar.gltfScene.updateMatrixWorld(true);
      if (this.aim(`${side}Hand` as HumanoidBone, at(H.MIDDLE_MCP).sub(at(H.WRIST)))) written += 1;

      for (const [bone, from, to] of fingerChain(side)) {
        if ((points[from]?.C ?? 0) === 0 || (points[to]?.C ?? 0) === 0) continue;
        this.avatar.gltfScene.updateMatrixWorld(true);
        if (this.aim(bone, at(to).sub(at(from)))) written += 1;
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
