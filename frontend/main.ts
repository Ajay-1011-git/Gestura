import { Buffer } from "buffer";
(globalThis as any).Buffer = (globalThis as any).Buffer ?? Buffer;

import * as THREE from "three";
import { loadAvatar, createIsolatedScene } from "./avatar/loader";
import { loadPoseSequence, PoseRetargeter, toMediaPipePose } from "./avatar/retarget";
import * as Kalido from "kalidokit";
(globalThis as any).__K = Kalido;

const W = 900, H = 900;

async function main() {
  const avatar = await loadAvatar("/avatar.glb");
  const { scene, camera } = createIsolatedScene(avatar, W, H);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H);
  document.body.appendChild(renderer.domElement);

  const seq = await loadPoseSequence("/demo_sequence.pose");
  const retargeter = new PoseRetargeter(avatar);

  // Frame this on the upper body — signing happens between chest and head.
  const chest = avatar.bones.get("chest") ?? avatar.bones.get("spine")!;
  const centre = chest.getWorldPosition(new THREE.Vector3());
  camera.position.set(centre.x, centre.y + 0.15, centre.z + 1.25);
  camera.lookAt(centre.x, centre.y + 0.1, centre.z);

  const track = (name: string) => {
    const b = avatar.bones.get(name as any)!;
    return b.quaternion.clone();
  };

  const state: any = {
    fps: seq.fps, frameCount: seq.frameCount,
    components: Object.fromEntries(Object.entries(seq.componentPoints).map(([k, v]) => [k, v.length])),
    samples: [],
  };

  const rest = { l: track("leftUpperArm"), r: track("rightUpperArm"), lh: track("leftHand") };
  const capture: number[] = [];

  (window as any).__render = (frame: number) => {
    retargeter.reset();
    const written = retargeter.applyFrame(seq, frame);
    for (let i = 0; i < 6; i++) retargeter.applyFrame(seq, frame); // let slerp converge
    avatar.gltfScene.updateMatrixWorld(true);
    renderer.render(scene, camera);
    const l = track("leftUpperArm"), r = track("rightUpperArm");
    return {
      frame, written,
      leftUpperArmDeltaDeg: THREE.MathUtils.radToDeg(rest.l.angleTo(l)),
      rightUpperArmDeltaDeg: THREE.MathUtils.radToDeg(rest.r.angleTo(r)),
      leftHandDeltaDeg: THREE.MathUtils.radToDeg(rest.lh.angleTo(track("leftHand"))),
      drivenBones: retargeter.driven.size,
    };
  };

  // sanity: confirm landmarks land at real MediaPipe indices, not densely packed
  const lm = toMediaPipePose(seq.frames[Math.floor(seq.frameCount / 2)], seq.componentPoints, seq.width, seq.height);
  state.landmarkCheck = {
    leftShoulder11: lm[11].visibility > 0,
    rightShoulder12: lm[12].visibility > 0,
    leftWrist15: lm[15].visibility > 0,
    nose0_shouldBeEmpty: lm[0].visibility === 0,
    populated: lm.filter((p) => p.visibility > 0).length,
  };

  (window as any).__seq = state;
  (window as any).__seqObj = seq;
  (window as any).__ready = true;
  void capture;
}
main().catch((e) => {
  (window as any).__seq = { error: String(e), stack: String(e?.stack).slice(0, 400) };
  (window as any).__ready = true;
});
