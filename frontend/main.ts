import * as THREE from "three";
import { loadAvatar, createIsolatedScene } from "./avatar/loader";

const W = 900, H = 900;

async function main() {
  const avatar = await loadAvatar("/avatar.glb");
  const { scene, camera } = createIsolatedScene(avatar, W, H);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H);
  document.body.appendChild(renderer.domElement);
  renderer.render(scene, camera);

  const r = avatar.report;
  (document.getElementById("report") as HTMLElement).textContent = [
    `bones found        ${r.boneCount}`,
    `mapped to humanoid ${r.mappedCount}`,
    `missing required   ${r.missingRequired.length ? r.missingRequired.join(", ") : "none"}`,
    `unmapped           ${r.unmappedBones.join(", ") || "none"}`,
    `arm droop          ${r.armDroopDegrees.toFixed(1)}deg -> ${r.poseType}`,
    `height             ${r.height.toFixed(3)}`,
    `upright            ${r.upright}`,
    `facing             ${r.facing}`,
    `animations         ${r.animationCount}`,
  ].join("\n");

  (window as any).__t111 = r;
  (window as any).__ready = true;
}
main().catch((e) => {
  (document.getElementById("report") as HTMLElement).textContent = "ERROR: " + e;
  (window as any).__ready = true;
  (window as any).__t111 = { error: String(e) };
});
