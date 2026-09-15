import { Buffer } from "buffer";
(globalThis as any).Buffer = (globalThis as any).Buffer ?? Buffer;

import * as THREE from "three";
import { loadAvatar, createIsolatedScene } from "./avatar/loader";
import { loadPoseSequence, PoseRetargeter } from "./avatar/retarget";
import { VOCABULARY } from "./avatar/handshapes";
import * as anatomy from "./avatar/hand_anatomy";
import * as shapes from "./avatar/handshapes";
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
  camera.position.set(centre.x, centre.y + 0.22, centre.z + 1.75);
  camera.lookAt(centre.x, centre.y + 0.10, centre.z);

  const seq = await loadPoseSequence("/demo_sequence.pose");
  const retargeter = new PoseRetargeter(avatar);

  const timeline: any[] = await (await fetch("/demo_timeline.json")).json();

  const log = await loadLog("/decision_log.json");
  panel.setEntries(log.entries);

  hud.innerHTML =
    `<b>Gestura</b> — Stage 1<br>` +
    `sequence &nbsp;${timeline.map((t:any)=>t.gloss).join(" ")}<br>` +
    `${seq.frameCount} frames @ ${seq.fps}fps (${(seq.frameCount / seq.fps).toFixed(2)}s)<br>` +
    `bones mapped &nbsp;${avatar.report.mappedCount}/${avatar.report.boneCount} &nbsp;` +
    `rig ${avatar.report.poseType}<br>` +
    `handshapes &nbsp;${VOCABULARY.length} canonical &nbsp;·&nbsp; fingers solved as constrained joints`;

  const scrub = document.getElementById("scrub") as HTMLInputElement;
  const playButton = document.getElementById("play") as HTMLButtonElement;
  const frameLabel = document.getElementById("frameLabel") as HTMLElement;
  scrub.max = String(seq.frameCount - 1);

  let frame = 0, playing = true, last = performance.now(), accumulator = 0;
  let speed = 1.0;

  // Arms solve directly from rest in one exact pass. Hands additionally carry
  // temporal state (jitter filtering, handshape hysteresis), so frames must be
  // fed in order and at a truthful dt — reset() is what breaks that continuity
  // when the timeline jumps.
  const show = (index: number, dt = 1 / seq.fps) => {
    retargeter.applyFrame(seq, index, dt);
    avatar.gltfScene.updateMatrixWorld(true);
    renderer.render(scene, camera);
    const span = timeline.find((t:any)=> index>=t.start_frame && index<t.end_frame);
    const hand = (side: "left" | "right") => {
      const h = retargeter.lastHand[side];
      if (!h?.shape) return "—";
      // "~" marks a coasting hand: not tracked this frame, holding its last
      // shape and settling toward neutral rather than snapping back to rest.
      const mark = h.coasting ? "~" : "";
      return `${mark}${h.shape.name} ${(h.snap * 100).toFixed(0)}%`;
    };
    frameLabel.textContent =
      `${span ? span.gloss : "\u2014"}  ·  ${String(index).padStart(3)}/${seq.frameCount - 1}  ·  ` +
      `L ${hand("left")}  ·  R ${hand("right")}`;
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
  const speedInput = document.getElementById("speed") as HTMLInputElement;
  const speedLabel = document.getElementById("speedLabel") as HTMLElement;
  const applySpeed = () => {
    speed = Number(speedInput.value) / 100;
    speedLabel.textContent = `${speed.toFixed(2)}x`;
  };
  speedInput.addEventListener("input", applySpeed);
  applySpeed();

  panel.onOverrideChange = (paused) => {
    if (paused) { playing = false; playButton.textContent = "❚❚ PAUSED"; }
  };

  const loop = (now: number) => {
    const dt = (now - last) / 1000; last = now;
    if (playing && !panel.isPaused) {
      accumulator += dt;
      const step = 1 / (seq.fps * speed);
      while (accumulator >= step) {
        accumulator -= step;
        frame = (frame + 1) % seq.frameCount;
        // rewind(), not reset(): the wrap is a cut in playback, not in the
        // signing, so the avatar sweeps across it rather than teleporting.
        if (frame === 0) retargeter.rewind();
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

  // Verification reads the *rendered* skeleton back through the same anatomy
  // module that drove it, so scripts/verify_hands.mjs measures the rig rather
  // than re-deriving what the solver believes it did.
  (window as any).__dbg = { avatar, retargeter, seq, timeline, THREE, show, HANDSHAPES: VOCABULARY, anatomy, shapes, PoseRetargeter, scene, camera, renderer };
  // Lets scripts/verify_hands.mjs re-render this scene from a second camera, so
  // hand close-ups come out of the real pipeline rather than a parallel one.
  // Pause playback first, or the render loop repaints over it.
  (window as any).__render2 = (c: THREE.Camera) => renderer.render(scene, c);
  (window as any).__ready = true;
  (window as any).__info = { frames: seq.frameCount, logEntries: panel.count, bones: avatar.report.mappedCount };
}
main().catch((e) => {
  (document.getElementById("hud") as HTMLElement).textContent = "ERROR: " + e;
  (window as any).__ready = true;
  (window as any).__info = { error: String(e) };
});
