import { Buffer } from "buffer";
(globalThis as any).Buffer = (globalThis as any).Buffer ?? Buffer;

import * as THREE from "three";
import { loadAvatar, createIsolatedScene } from "./avatar/loader";
import { loadPoseSequence, PoseRetargeter } from "./avatar/retarget";
import { DecisionLogPanel, loadLog } from "./decision_log_panel/panel";

async function main() {
  const wrap = document.getElementById("canvas-wrap") as HTMLElement;
  const hud = document.getElementById("hud") as HTMLElement;
  const panel = new DecisionLogPanel(document.getElementById("panel") as HTMLElement);

  const avatar = await loadAvatar("/avatar.glb");
  const width = wrap.clientWidth, height = wrap.clientHeight;
  const { scene, camera } = createIsolatedScene(avatar, width, height);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(width, height);
  wrap.appendChild(renderer.domElement);

  // Frame the signing space: chest to just above the head.
  const chest = avatar.bones.get("chest") ?? avatar.bones.get("spine")!;
  const centre = chest.getWorldPosition(new THREE.Vector3());
  camera.position.set(centre.x, centre.y + 0.18, centre.z + 1.3);
  camera.lookAt(centre.x, centre.y + 0.12, centre.z);

  const seq = await loadPoseSequence("/demo_sequence.pose");
  const retargeter = new PoseRetargeter(avatar);

  const log = await loadLog("/decision_log.json");
  panel.setEntries(log.entries);

  hud.innerHTML =
    `<b>Gestura</b> — Stage 1<br>` +
    `sequence &nbsp;HELLO YOU SIT PLEASE<br>` +
    `${seq.frameCount} frames @ ${seq.fps}fps (${(seq.frameCount / seq.fps).toFixed(2)}s)<br>` +
    `bones mapped &nbsp;${avatar.report.mappedCount}/${avatar.report.boneCount} &nbsp;` +
    `rig ${avatar.report.poseType}`;

  const scrub = document.getElementById("scrub") as HTMLInputElement;
  const playButton = document.getElementById("play") as HTMLButtonElement;
  const frameLabel = document.getElementById("frameLabel") as HTMLElement;
  scrub.max = String(seq.frameCount - 1);

  let frame = 0, playing = true, last = performance.now(), accumulator = 0;

  const show = (index: number) => {
    retargeter.applyFrame(seq, index);
    avatar.gltfScene.updateMatrixWorld(true);
    renderer.render(scene, camera);
    frameLabel.textContent =
      `frame ${String(index).padStart(3)} / ${seq.frameCount - 1}  ·  ${retargeter.driven.size} bones driven`;
  };

  playButton.addEventListener("click", () => {
    playing = !playing;
    playButton.textContent = playing ? "▶ PLAY" : "❚❚ PAUSED";
  });
  scrub.addEventListener("input", () => {
    playing = false; playButton.textContent = "❚❚ PAUSED";
    frame = Number(scrub.value); retargeter.reset(); show(frame);
  });
  // The supervisor override halts avatar output, exactly as it would live.
  panel.onOverrideChange = (paused) => {
    if (paused) { playing = false; playButton.textContent = "❚❚ PAUSED"; }
  };

  const loop = (now: number) => {
    const dt = (now - last) / 1000; last = now;
    if (playing && !panel.isPaused) {
      accumulator += dt;
      const step = 1 / seq.fps;
      while (accumulator >= step) {
        accumulator -= step;
        frame = (frame + 1) % seq.frameCount;
        if (frame === 0) retargeter.reset();
      }
      scrub.value = String(frame);
      show(frame);
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  addEventListener("resize", () => {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  });

  (window as any).__ready = true;
  (window as any).__info = { frames: seq.frameCount, logEntries: panel.count, bones: avatar.report.mappedCount };
}
main().catch((e) => {
  (document.getElementById("hud") as HTMLElement).textContent = "ERROR: " + e;
  (window as any).__ready = true;
  (window as any).__info = { error: String(e) };
});
