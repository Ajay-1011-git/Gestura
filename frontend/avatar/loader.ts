/**
 * Avatar loading and manual bone mapping (T1.11).
 *
 * Architecture v3 §7 makes the isolated load a hard ordering requirement: load
 * the .glb with loader and camera only, confirm it stands upright, and only then
 * connect retargeting. Export bugs and retargeting bugs look identical once both
 * are in play, and debugging them together burns hours.
 *
 * Three properties of this specific rig were measured from the file rather than
 * assumed, and each breaks a default assumption:
 *
 *  1. Bone names carry Sketchfab numeric suffixes — `LeftHand_18`, `Hips_54`.
 *     Exact-name matching against Mixamo names finds nothing and fails silently,
 *     leaving a T-posed avatar that never moves.
 *  2. The rig is in an **A-pose** (~39.7° arm droop), not the T-pose §7 step 3
 *     guessed. Kalidokit solves against a T-pose reference, so the rest pose is
 *     captured here for retargeting to compensate against.
 *  3. A baked `Pose_01` animation shipped in the source glTF and was stripped
 *     during the .glb build; if it ever reappears it will fight the solver.
 *
 * This is a plain glTF, not a `.vrm`. There is no VRMHumanoid metadata and no
 * automatic humanoid mapping — the table below is the mapping.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** VRM-humanoid bone names, which is the vocabulary Kalidokit's solvers use. */
export type HumanoidBone =
  | "hips" | "spine" | "chest" | "upperChest" | "neck" | "head"
  | "leftShoulder" | "leftUpperArm" | "leftLowerArm" | "leftHand"
  | "rightShoulder" | "rightUpperArm" | "rightLowerArm" | "rightHand"
  | "leftUpperLeg" | "leftLowerLeg" | "leftFoot" | "leftToes"
  | "rightUpperLeg" | "rightLowerLeg" | "rightFoot" | "rightToes"
  | `${"left" | "right"}${"Thumb" | "Index" | "Middle" | "Ring" | "Little"}${"Proximal" | "Intermediate" | "Distal"}`;

/** Mixamo/Unity bone name (suffix already stripped) -> VRM humanoid name. */
const BONE_MAP: Record<string, HumanoidBone> = {
  Hips: "hips", Spine: "spine", Spine1: "chest", Spine2: "upperChest",
  Neck: "neck", Head: "head",

  LeftShoulder: "leftShoulder", LeftArm: "leftUpperArm",
  LeftForeArm: "leftLowerArm", LeftHand: "leftHand",
  RightShoulder: "rightShoulder", RightArm: "rightUpperArm",
  RightForeArm: "rightLowerArm", RightHand: "rightHand",

  LeftUpLeg: "leftUpperLeg", LeftLeg: "leftLowerLeg",
  LeftFoot: "leftFoot", LeftToeBase: "leftToes",
  RightUpLeg: "rightUpperLeg", RightLeg: "rightLowerLeg",
  RightFoot: "rightFoot", RightToeBase: "rightToes",

  LeftHandThumb1: "leftThumbProximal", LeftHandThumb2: "leftThumbIntermediate", LeftHandThumb3: "leftThumbDistal",
  LeftHandIndex1: "leftIndexProximal", LeftHandIndex2: "leftIndexIntermediate", LeftHandIndex3: "leftIndexDistal",
  LeftHandMiddle1: "leftMiddleProximal", LeftHandMiddle2: "leftMiddleIntermediate", LeftHandMiddle3: "leftMiddleDistal",
  LeftHandRing1: "leftRingProximal", LeftHandRing2: "leftRingIntermediate", LeftHandRing3: "leftRingDistal",
  LeftHandPinky1: "leftLittleProximal", LeftHandPinky2: "leftLittleIntermediate", LeftHandPinky3: "leftLittleDistal",

  RightHandThumb1: "rightThumbProximal", RightHandThumb2: "rightThumbIntermediate", RightHandThumb3: "rightThumbDistal",
  RightHandIndex1: "rightIndexProximal", RightHandIndex2: "rightIndexIntermediate", RightHandIndex3: "rightIndexDistal",
  RightHandMiddle1: "rightMiddleProximal", RightHandMiddle2: "rightMiddleIntermediate", RightHandMiddle3: "rightMiddleDistal",
  RightHandRing1: "rightRingProximal", RightHandRing2: "rightRingIntermediate", RightHandRing3: "rightRingDistal",
  RightHandPinky1: "rightLittleProximal", RightHandPinky2: "rightLittleIntermediate", RightHandPinky3: "rightLittleDistal",
};

/** Bones without which retargeting cannot produce recognisable signing. */
const REQUIRED: HumanoidBone[] = [
  "hips", "spine", "neck", "head",
  "leftUpperArm", "leftLowerArm", "leftHand",
  "rightUpperArm", "rightLowerArm", "rightHand",
];

/** `LeftHand_18` -> `LeftHand`. Sketchfab appends `_<n>` to every bone. */
export function stripSuffix(name: string): string {
  return name.replace(/_\d+$/, "");
}

export interface RestPose {
  /** Bone-local rest quaternion, the reference Kalidokit's output is applied against. */
  quaternion: THREE.Quaternion;
  position: THREE.Vector3;
}

export interface AvatarReport {
  boneCount: number;
  mappedCount: number;
  missingRequired: HumanoidBone[];
  unmappedBones: string[];
  /** Arm droop from horizontal: ~0° is a T-pose, ~45° an A-pose. */
  armDroopDegrees: number;
  poseType: "T-pose" | "A-pose" | "unknown";
  height: number;
  facing: "+Z" | "-Z";
  upright: boolean;
  animationCount: number;
}

export interface LoadedAvatar {
  gltfScene: THREE.Group;
  bones: Map<HumanoidBone, THREE.Bone>;
  restPose: Map<HumanoidBone, RestPose>;
  report: AvatarReport;
}

function measureArmDroop(bones: Map<HumanoidBone, THREE.Bone>): number {
  const upper = bones.get("leftUpperArm");
  const hand = bones.get("leftHand");
  if (!upper || !hand) return Number.NaN;

  const a = upper.getWorldPosition(new THREE.Vector3());
  const b = hand.getWorldPosition(new THREE.Vector3());
  const horizontal = Math.hypot(b.x - a.x, b.z - a.z);
  return THREE.MathUtils.radToDeg(Math.atan2(a.y - b.y, horizontal));
}

/**
 * Load the avatar and build the humanoid bone map.
 *
 * Deliberately does no retargeting, adds no lights and starts no animation loop:
 * §7 step 5 wants this path provably correct in isolation first.
 */
export async function loadAvatar(url: string): Promise<LoadedAvatar> {
  const gltf = await new GLTFLoader().loadAsync(url);
  const gltfScene = gltf.scene;
  gltfScene.updateMatrixWorld(true);

  const bones = new Map<HumanoidBone, THREE.Bone>();
  const unmappedBones: string[] = [];
  let boneCount = 0;

  gltfScene.traverse((object) => {
    if (!(object as THREE.Bone).isBone) return;
    boneCount += 1;
    const humanoid = BONE_MAP[stripSuffix(object.name)];
    if (humanoid) bones.set(humanoid, object as THREE.Bone);
    else unmappedBones.push(object.name);
  });

  // Captured before anything drives the skeleton. Kalidokit solves against a
  // T-pose reference; this rig is an A-pose, so retargeting (T1.12) composes its
  // output onto these rest rotations rather than replacing them outright.
  const restPose = new Map<HumanoidBone, RestPose>();
  for (const [name, bone] of bones) {
    restPose.set(name, {
      quaternion: bone.quaternion.clone(),
      position: bone.position.clone(),
    });
  }

  const box = new THREE.Box3().setFromObject(gltfScene);
  const size = box.getSize(new THREE.Vector3());
  const armDroopDegrees = measureArmDroop(bones);

  const head = bones.get("head");
  const hips = bones.get("hips");
  const upright =
    !!head && !!hips &&
    head.getWorldPosition(new THREE.Vector3()).y >
      hips.getWorldPosition(new THREE.Vector3()).y;

  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(
    gltfScene.getWorldQuaternion(new THREE.Quaternion()),
  );

  return {
    gltfScene,
    bones,
    restPose,
    report: {
      boneCount,
      mappedCount: bones.size,
      missingRequired: REQUIRED.filter((b) => !bones.has(b)),
      unmappedBones,
      armDroopDegrees,
      poseType: Number.isNaN(armDroopDegrees)
        ? "unknown"
        : armDroopDegrees < 15 ? "T-pose" : "A-pose",
      height: size.y,
      facing: forward.z >= 0 ? "+Z" : "-Z",
      upright,
      animationCount: gltf.animations.length,
    },
  };
}

/** Minimal scene for the §7 step 5 isolated load test: loader and camera only. */
export function createIsolatedScene(
  avatar: LoadedAvatar,
  width: number,
  height: number,
): { scene: THREE.Scene; camera: THREE.PerspectiveCamera } {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202028);
  scene.add(avatar.gltfScene);
  scene.add(new THREE.AmbientLight(0xffffff, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(1, 3, 4);
  scene.add(key);

  const box = new THREE.Box3().setFromObject(avatar.gltfScene);
  const centre = box.getCenter(new THREE.Vector3());
  const span = box.getSize(new THREE.Vector3()).y || 1;

  const camera = new THREE.PerspectiveCamera(35, width / height, 0.01, 100);
  camera.position.set(centre.x, centre.y, centre.z + span * 2.2);
  camera.lookAt(centre);
  return { scene, camera };
}
