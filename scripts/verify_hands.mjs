/**
 * Verify the hand solver against the real render (T1.16).
 *
 * The point of this script is that it measures the **rendered skeleton**, not
 * the solver's own opinion of it. Bone world positions are read back out of the
 * posed rig and pushed through the same `hand_anatomy` measurement used on the
 * source landmarks, so an axis or sign error shows up as a number rather than
 * as something that has to be noticed by eye.
 *
 *   node scripts/verify_hands.mjs [--keep]
 *
 * Checks, each with a stated pass condition:
 *   1. hinge violation   — PIP/DIP bend axes must stay on the knuckle hinge
 *   2. ROM violation     — no joint outside its clinical range
 *   3. hyperextension    — no PIP/DIP folding backwards
 *   4. jitter            — rendered angles must be steadier than raw landmarks
 *   5. curl fidelity     — a commanded fist must actually close toward the palm
 *   6. coverage          — a handshape identified for every frame of every sign
 */

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";

const KEEP = process.argv.includes("--keep");
const OUT = "/private/tmp/claude-501/-Users-ajay-macm5-Desktop-Gestura/f016bb59-006d-4c79-a746-5ae963d38ef9/scratchpad/verify";
mkdirSync(OUT, { recursive: true });

// Detached so the whole process group can be killed: `npx` forks vite, and
// signalling only the direct child leaves the port held.
const server = spawn("npx", ["vite", "--port", "5179", "--strictPort"], {
  cwd: process.cwd(), stdio: ["ignore", "pipe", "pipe"], detached: true,
});
const ready = new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("vite did not start")), 60000);
  server.stdout.on("data", (b) => {
    if (b.toString().includes("localhost:5179")) { clearTimeout(timer); resolve(); }
  });
  server.stderr.on("data", (b) => process.stderr.write(b));
});

const die = (code) => {
  try { process.kill(-server.pid, "SIGKILL"); } catch { server.kill("SIGKILL"); }
  process.exit(code);
};

try {
  await ready;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
  await page.goto("http://localhost:5179/", { waitUntil: "load" });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 });
  const info = await page.evaluate(() => window.__info);
  // Stop playback: the render loop repaints every frame, so without this the
  // stills below capture whatever frame it happened to be on.
  await page.click("#play");
  if (info?.error) throw new Error("page failed: " + info.error);

  const report = await page.evaluate(() => {
    const { avatar, retargeter, seq, timeline, THREE, show, anatomy, shapes, HANDSHAPES } = window.__dbg;
    const deg = (r) => THREE.MathUtils.radToDeg(r);
    const clamp = (v, a, b) => THREE.MathUtils.clamp(v, a, b);
    const SIDES = ["left", "right"];
    const CURLED = ["index", "middle", "ring", "little"];

    /**
     * Deviation of each rendered PIP/DIP bend from the hinge it should be on.
     *
     * Not the palm's knuckle line: abduction at the MCP tilts the rest of the
     * finger's hinge with it, by exactly the spread angle, which is anatomy
     * rather than error. The expected axis is therefore the knuckle line rotated
     * about the palm normal by that finger's rendered spread.
     */
    function hingeDeviation(points, frame, commanded, restAngles) {
      const out = [];
      for (const f of CURLED) {
        // Re-measuring spread off a strongly flexed finger is unreliable — the
        // proximal phalanx is then near-perpendicular to the palm and its
        // projection into the palm plane is mostly noise. The tilt applied was
        // the commanded spread minus the rig's own rest spread, so use that.
        const applied = commanded ? commanded[f].spread - restAngles[f].spread : 0;
        const expected = frame.flex.clone().applyAxisAngle(
          frame.abduct, THREE.MathUtils.degToRad(applied));
        const [b, a, c, t] = anatomy.DIGIT_LANDMARKS[f];
        const segs = [
          points[a].clone().sub(points[b]),
          points[c].clone().sub(points[a]),
          points[t].clone().sub(points[c]),
        ].map((v) => v.normalize());
        const prevs = [points[b].clone().sub(points[0]).normalize(), segs[0], segs[1]];
        // MCP is skipped: it legitimately abducts, so its bend axis is allowed
        // off the hinge. PIP and DIP are pure hinges and are not.
        for (let j = 1; j < 3; j++) {
          const p = prevs[j], s = segs[j];
          const bend = deg(Math.acos(clamp(p.dot(s), -1, 1)));
          if (bend < 25) continue;   // below this the bend axis is undefined
          const hinge = new THREE.Vector3().crossVectors(p, s).normalize();
          out.push(deg(Math.acos(clamp(Math.abs(hinge.dot(expected)), -1, 1))));
        }
      }
      return out;
    }

    /** The raw landmark points in the same axes the solver reads them in. */
    function rawPoints(i, side) {
      const comp = side === "left" ? "LEFT_HAND_LANDMARKS" : "RIGHT_HAND_LANDMARKS";
      const pts = seq.frames[i]?.[comp];
      if (!pts || pts.length < 21 || !pts.slice(0, 21).some((p) => (p.C ?? 0) > 0)) return null;
      return pts.slice(0, 21).map((p) =>
        new THREE.Vector3(p.X, -p.Y, -(p.Z ?? 0) * seq.width));
    }

    /** The raw landmark angles, i.e. what the old solver rendered verbatim. */
    function rawAngles(i, side) {
      const comp = side === "left" ? "LEFT_HAND_LANDMARKS" : "RIGHT_HAND_LANDMARKS";
      const v = rawPoints(i, side);
      return v ? anatomy.measureHand(v, side, { dipCoupling: 0 }).angles : null;
    }

    const KEYS = ["mcp", "spread", "pip", "dip"];
    // The rig's own rest angles, read before anything poses it.
    retargeter.reset();
    avatar.gltfScene.updateMatrixWorld(true);
    const restAngles = {};
    for (const side of SIDES) restAngles[side] = anatomy.calibrateHand(avatar, side).rest;

    const frames = [];
    retargeter.reset();
    for (let i = 0; i < seq.frameCount; i++) {
      show(i);
      const record = { i, hands: {} };
      for (const side of SIDES) {
        const points = anatomy.sampleHandPoints(avatar, side);
        const { angles, frame } = anatomy.measureHand(points, side, { dipCoupling: 0 });
        const tracked = retargeter.lastHand[side];
        // Palm orientation error: the basis the landmarks ask for against the one
        // the rig ended up with after wrist and forearm limits were applied.
        let palmError = null;
        // The palm frame as rendered — the same quantity the landmarks give, so
        // the two are directly comparable. (The hand *bone* quaternion is not:
        // it differs from the palm basis by the fixed palmLocal rotation.)
        const renderedPalm = new THREE.Quaternion().setFromRotationMatrix(
          anatomy.palmBasis(anatomy.palmFrame(points[0], points[9], points[5], points[17], side))
        ).toArray();
        const rawP = rawPoints(i, side);
        if (rawP) {
          const want = anatomy.palmFrame(rawP[0], rawP[9], rawP[5], rawP[17], side);
          const got = anatomy.palmFrame(points[0], points[9], points[5], points[17], side);
          palmError = deg(new THREE.Quaternion()
            .setFromRotationMatrix(anatomy.palmBasis(got))
            .angleTo(new THREE.Quaternion().setFromRotationMatrix(anatomy.palmBasis(want))));
        }
        record.hands[side] = {
          shape: tracked?.shape?.name ?? null,
          confidence: tracked?.confidence ?? 0,
          coasting: !!tracked?.coasting,
          snap: tracked?.snap ?? 0,
          rendered: angles,
          raw: rawAngles(i, side),
          deviation: hingeDeviation(points, frame, tracked?.angles, restAngles[side]),
          rawDeviation: (() => {
            const v = rawPoints(i, side);
            if (!v) return [];
            const fr = anatomy.palmFrame(v[0], v[9], v[5], v[17], side);
            return hingeDeviation(v, fr, null, null);
          })(),
          commanded: tracked?.angles ?? null,
          held: tracked?.held ?? 0,
          palmError,
          renderedPalm,
          rawPalm: rawP
            ? new THREE.Quaternion().setFromRotationMatrix(
                anatomy.palmBasis(anatomy.palmFrame(rawP[0], rawP[9], rawP[5], rawP[17], side))
              ).toArray()
            : null,
          reach: retargeter.lastReachError,
          wrist: retargeter.lastWrist[side],
          palmRejected: retargeter.lastPalmRejected[side],
          elbowSwing: retargeter.lastElbowSwing[side],
          palmTarget: retargeter.lastPalmTarget[side]?.toArray() ?? null,

          match: (() => { const m = shapes.classify(angles, anatomy.measureHand(points, side).extension); 
            return { best: m.shape.name, d: m.distance, runnerUp: m.runnerUp }; })(),
        };
      }
      frames.push(record);
    }

    // --- 12. the playback loop, and the stretches with no sign at all -------
    // Both were blind spots: every per-sign report above skips the 47 frames
    // after the last sign, and nothing looked at the wrap from the last frame
    // back to the first, which is where the avatar was seen glitching.
    const wrap = (() => {
      const probe = new window.__dbg.PoseRetargeter(avatar, {});
      const snap = () => ({
        q: avatar.bones.get("rightHand").getWorldQuaternion(new THREE.Quaternion()),
        p: avatar.bones.get("rightHand").getWorldPosition(new THREE.Vector3()),
      });
      probe.reset();
      for (let i = 0; i < seq.frameCount; i++) probe.applyFrame(seq, i);
      avatar.gltfScene.updateMatrixWorld(true);
      const last = snap();
      probe.rewind();                       // exactly what the render loop does
      probe.applyFrame(seq, 0);
      avatar.gltfScene.updateMatrixWorld(true);
      const first = snap();
      return { turn: deg(last.q.angleTo(first.q)), move: last.p.distanceTo(first.p) };
    })();

    const covered = new Set();
    for (const span of timeline) {
      for (let i = span.start_frame; i < span.end_frame; i++) covered.add(i);
    }
    const unglossed = [];
    for (let i = 0; i < seq.frameCount; i++) if (!covered.has(i)) unglossed.push(i);

    // --- baseline: the same solve with the elbow search disabled, so check 11
    // can say what the search is worth rather than just asserting a number.
    const baseline = (() => {
      // Both stages of the search must be off, or the baseline is not a baseline.
      const plain = new window.__dbg.PoseRetargeter(avatar,
        { elbowSearchRange: 0.001, elbowWideRange: 0.001 });
      const strain = { left: [], right: [] };
      plain.reset();
      for (let i = 0; i < seq.frameCount; i++) {
        plain.applyFrame(seq, i);
        for (const side of SIDES) {
          const w = plain.lastWrist[side];
          if (w) strain[side].push(
            Math.abs(w.bendWanted - w.bendApplied) + Math.abs(w.deviateWanted - w.deviateApplied));
        }
      }
      return strain;
    })();

    // --- rig floor ---------------------------------------------------------
    // The rig's fingers do not rest perfectly planar (measured |flex| up to 9.4°,
    // |lateral| up to 9.4°), so a bone rotated purely about its hinge still
    // traces a slightly non-planar path. Driving a pure-flexion sweep with zero
    // spread measures that floor, which is the only honest bar for check 1: the
    // solver must add nothing on top of it.
    retargeter.reset();
    avatar.gltfScene.updateMatrixWorld(true);
    const floor = {};
    for (const side of SIDES) {
      const calibration = anatomy.calibrateHand(avatar, side);
      const devs = [];
      for (const bend of [30, 50, 70, 90]) {
        const a = anatomy.zeroAngles();
        for (const f of ["thumb", ...CURLED]) a[f] = { mcp: bend, spread: 0, pip: bend, dip: bend * 0.66 };
        anatomy.applyHandAngles(avatar, calibration, a);
        avatar.gltfScene.updateMatrixWorld(true);
        const p = anatomy.sampleHandPoints(avatar, side);
        const fr = anatomy.palmFrame(p[0], p[9], p[5], p[17], side);
        devs.push(...hingeDeviation(p, fr, null, null));
        retargeter.reset();
        avatar.gltfScene.updateMatrixWorld(true);
      }
      devs.sort((x, y) => x - y);
      floor[side] = { med: devs[devs.length >> 1], max: devs[devs.length - 1] };
    }

    // --- 5. curl fidelity ------------------------------------------------
    // Two commanded poses with unambiguous geometry. BENT_FLAT is pure MCP
    // flexion, so the proximal phalanx must swing to point out of the palm on
    // the palmar side — that alone pins the sign of the flexion axis. FIST then
    // checks the whole chain closes, by bringing the tips in toward the wrist.
    const fist = HANDSHAPES.find((s) => s.name === "FIST");
    const bent = HANDSHAPES.find((s) => s.name === "BENT_FLAT");
    const curl = {};
    // Calibration reads the rig's *rest* geometry, so the playback pose left by
    // the loop above has to be cleared first or every axis it measures is wrong.
    retargeter.reset();
    avatar.gltfScene.updateMatrixWorld(true);
    for (const side of SIDES) {
      const calibration = anatomy.calibrateHand(avatar, side);
      const before = anatomy.sampleHandPoints(avatar, side);
      const rest = anatomy.palmFrame(before[0], before[9], before[5], before[17], side);

      anatomy.applyHandAngles(avatar, calibration, bent.angles);
      avatar.gltfScene.updateMatrixWorld(true);
      const bentPoints = anatomy.sampleHandPoints(avatar, side);
      const palmarDot = CURLED.map((f) => {
        const [b, a] = anatomy.DIGIT_LANDMARKS[f];
        return bentPoints[a].clone().sub(bentPoints[b]).normalize().dot(rest.palmar);
      });
      retargeter.reset();
      avatar.gltfScene.updateMatrixWorld(true);

      anatomy.applyHandAngles(avatar, calibration, fist.angles);
      avatar.gltfScene.updateMatrixWorld(true);
      const after = anatomy.sampleHandPoints(avatar, side);
      // Fingertip displacement projected on the palmar axis: positive means the
      // tips moved to the side the fingers are meant to curl toward.
      const palmar = CURLED.map((f) => {
        const tip = anatomy.DIGIT_LANDMARKS[f][3];
        return after[tip].clone().sub(before[tip]).dot(rest.palmar) / rest.scale;
      });
      const reach = CURLED.map((f) => {
        const tip = anatomy.DIGIT_LANDMARKS[f][3];
        return (after[tip].distanceTo(after[0]) - before[tip].distanceTo(before[0])) / rest.scale;
      });
      curl[side] = { palmar, reach, palmarDot };
      retargeter.reset();
      avatar.gltfScene.updateMatrixWorld(true);
    }

    return { frames, timeline, keys: KEYS, fingers: CURLED, curl, floor, baseline, wrap, unglossed,
             shapes: HANDSHAPES.map((s) => s.name) };
  });

  // ---- analysis (node side) -------------------------------------------------
  const { frames, timeline, keys, curl, floor, baseline, wrap, unglossed } = report;
  const ROM = {
    thumb:  { mcp: [-10, 60], spread: [-15, 60], pip: [-10, 55], dip: [-15, 80] },
    index:  { mcp: [-20, 90], spread: [-12, 30], pip: [0, 110], dip: [0, 80] },
    middle: { mcp: [-20, 90], spread: [-15, 15], pip: [0, 110], dip: [0, 80] },
    ring:   { mcp: [-20, 90], spread: [-22, 10], pip: [0, 110], dip: [0, 80] },
    little: { mcp: [-20, 95], spread: [-35, 10], pip: [0, 110], dip: [0, 80] },
  };
  const FINGERS = Object.keys(ROM);
  /** Angle between two quaternions given as [x,y,z,w], degrees. */
  const quaternionAngle = (a, b) => {
    const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    return 2 * Math.acos(Math.min(1, dot)) * 180 / Math.PI;
  };
  const stat = (a) => {
    if (!a.length) return { n: 0, med: NaN, p90: NaN, max: NaN };
    const s = [...a].sort((x, y) => x - y);
    const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    return { n: s.length, med: q(0.5), p90: q(0.9), max: s[s.length - 1] };
  };

  const lines = [];
  const say = (s = "") => { lines.push(s); console.log(s); };
  let failures = 0;
  const check = (ok, label, detail) => {
    if (!ok) failures += 1;
    say(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? "  — " + detail : ""}`);
  };

  say(`\nRendered ${frames.length} frames\n`);

  for (const side of ["left", "right"]) {
    const rows = frames.map((f) => f.hands[side]);
    const live = rows.filter((r) => !r.coasting);
    say(`=== ${side} hand  (${live.length}/${rows.length} frames from tracked landmarks) ===`);

    // 1. hinge violation, against the rig's own floor rather than against zero
    const dev = stat(rows.flatMap((r) => r.deviation));
    const rawDev = stat(rows.flatMap((r) => r.rawDeviation ?? []));
    const bar = floor[side];
    check(dev.med <= bar.med + 1 && dev.max <= bar.max + 2,
      "1. PIP/DIP bend axes add no deviation beyond the rig's own",
      `rendered median ${dev.med?.toFixed(1)}° max ${dev.max?.toFixed(1)}°  ` +
      `| rig floor ${bar.med.toFixed(1)}°/${bar.max.toFixed(1)}°  ` +
      `| raw landmarks ${rawDev.med?.toFixed(1)}°/${rawDev.p90?.toFixed(1)}° (p90)`);

    // 2. ROM
    let romBad = 0, worst = "";
    for (const r of rows) for (const f of FINGERS) for (const k of keys) {
      const v = r.rendered[f][k], [lo, hi] = ROM[f][k];
      if (v < lo - 0.75 || v > hi + 0.75) { romBad += 1; worst = `${f}.${k}=${v.toFixed(1)}`; }
    }
    check(romBad === 0, "2. every joint inside clinical range",
      romBad ? `${romBad} violations, e.g. ${worst}` : `${rows.length * 20} joint-frames clean`);

    // 3. hyperextension
    const hyper = (rowsIn, pick) => {
      let bad = 0, total = 0;
      for (const r of rowsIn) {
        const a = pick(r); if (!a) continue;
        for (const f of FINGERS) { if (f === "thumb") continue;
          for (const k of ["pip", "dip"]) { total += 1; if (a[f][k] < -5) bad += 1; } }
      }
      return total ? (100 * bad) / total : 0;
    };
    const hyperRaw = hyper(live, (r) => r.raw);
    const hyperNow = hyper(rows, (r) => r.rendered);
    check(hyperNow < 0.5, "3. no PIP/DIP folding backwards",
      `rendered ${hyperNow.toFixed(2)}% of joint-frames (raw landmarks: ${hyperRaw.toFixed(1)}%)`);

    // 4. jitter, measured as the second difference: real signing motion is
    //    smooth and nearly cancels, so what is left is landmark noise.
    const shake = (pick, gate = () => true) => {
      const d = [];
      for (let i = 2; i < frames.length; i++) {
        if (!gate(i) || !gate(i - 1) || !gate(i - 2)) continue;
        const a = pick(frames[i - 2].hands[side]);
        const b = pick(frames[i - 1].hands[side]);
        const c2 = pick(frames[i].hands[side]);
        if (!a || !b || !c2) continue;
        for (const f of FINGERS) for (const k of ["mcp", "pip"]) {
          d.push(Math.abs(c2[f][k] - 2 * b[f][k] + a[f][k]));
        }
      }
      return d.length ? d.reduce((x, y) => x + y, 0) / d.length : NaN;
    };
    // Restricted to holds: during a genuine transition a high second difference
    // is the sign moving, not noise, and suppressing it would be the wrong goal.
    const onHold = (i) => frames[i].hands[side].held >= 8;
    const jRaw = shake((r) => r.raw, onHold), jNow = shake((r) => r.rendered, onHold);
    const jRawAll = shake((r) => r.raw), jNowAll = shake((r) => r.rendered);
    check(jNow < jRaw * 0.5, "4. landmark shake suppressed during holds (2nd difference)",
      `${jNow.toFixed(2)}°/frame² vs raw ${jRaw.toFixed(2)}°/frame²` +
      ` (${(100 - (100 * jNow) / jRaw).toFixed(0)}% less)` +
      `  | whole sequence ${jNowAll.toFixed(2)} vs ${jRawAll.toFixed(2)}`);
    const switches = frames.filter((f, i) => i > 0 &&
      f.hands[side].shape !== frames[i - 1].hands[side].shape).length;
    say(`  handshape changes: ${switches} over ${frames.length} frames`);

    // 5. curl direction
    const c = curl[side];
    check(c.palmarDot.every((v) => v > 0.7),
      "5a. flexion bends toward the palm, not the back of the hand",
      `proximal phalanx palmar component ${c.palmarDot.map((v) => v.toFixed(2)).join("/")} (need >0.70)`);
    check(c.reach.every((v) => v < -0.5),
      "5b. a commanded fist closes",
      `tips draw ${c.reach.map((v) => (-v).toFixed(2)).join("/")} palm-lengths toward the wrist`);

    // 10. wrist steadiness: the rendered palm must be markedly calmer than the
    //     landmarks it is driven by, measured as angular second difference.
    const angleShake = (pick) => {
      const step = [];
      for (let i = 1; i < frames.length; i++) {
        const a = pick(frames[i - 1].hands[side]), b = pick(frames[i].hands[side]);
        if (!a || !b) { step.push(null); continue; }
        step.push(quaternionAngle(a, b));
      }
      const d2 = [];
      for (let i = 1; i < step.length; i++) {
        if (step[i] !== null && step[i - 1] !== null) d2.push(Math.abs(step[i] - step[i - 1]));
      }
      return stat(d2);
    };
    const shakeRendered = angleShake((r) => r.renderedPalm);
    const shakeSource = angleShake((r) => r.rawPalm);
    check(shakeRendered.med < shakeSource.med * 0.6,
      "10. wrist steadier than the landmark palm it follows",
      `rendered ${shakeRendered.med?.toFixed(2)}°/frame² p90 ${shakeRendered.p90?.toFixed(2)}  ` +
      `| landmarks ${shakeSource.med?.toFixed(2)} p90 ${shakeSource.p90?.toFixed(2)}`);

    // 11. what the wrist was asked for, versus what a wrist can do
    const wr = rows.map((r) => r.wrist).filter(Boolean);
    const strain = stat(wr.map((x) =>
      Math.abs(x.bendWanted - x.bendApplied) + Math.abs(x.deviateWanted - x.deviateApplied)));
    const twistLost = stat(wr.map((x) => Math.abs(x.twistWanted) - Math.abs(x.twistApplied)));
    const swung = stat(wr.map((_, i) => Math.abs(rows.filter((r) => r.wrist)[i].elbowSwing)));
    const plain = stat(baseline[side]);
    check(strain.med < plain.med * 0.7 && twistLost.med < 1,
      "11. placing the elbow spares the wrist, and the arm supplies the roll",
      `wrist swing discarded median ${strain.med?.toFixed(1)}° (elbow fixed: ${plain.med?.toFixed(1)}°)  ` +
      `| roll unmet median ${twistLost.med?.toFixed(1)}°  | elbow swung median ${swung.med?.toFixed(0)}°`);

    // 8. palm orientation, which the old solver left unconstrained in roll.
    //    Measured against a zero-phase smoothing of the landmark palm — the same
    //    filter run forwards then backwards, so it has no delay. Comparing to the
    //    raw signal instead would count the smoothing's deliberate refusal to
    //    follow jitter as error: by that measure even a solve with no filter at
    //    all sits 9.5° away.
    const truth = (() => {
      const pass = (list) => {
        let acc = null;
        return list.map((q) => {
          if (!q) return null;
          if (!acc) { acc = [...q]; return [...acc]; }
          const dot = acc[0]*q[0] + acc[1]*q[1] + acc[2]*q[2] + acc[3]*q[3];
          const aim = dot < 0 ? q.map((v) => -v) : q;
          acc = acc.map((v, i) => v + 0.25 * (aim[i] - v));
          const n = Math.hypot(...acc); acc = acc.map((v) => v / n);
          return [...acc];
        });
      };
      const forward = pass(rows.map((r) => r.rawPalm));
      return pass(forward.slice().reverse()).reverse();
    })();
    // Frames where the palm measurement was rejected as physically impossible are
    // excluded: on those the solver is deliberately *not* following the input, so
    // counting the distance as tracking error would penalise the guard for working.
    const lag = [], lagAll = [];
    rows.forEach((r, i) => {
      if (!r.renderedPalm || !truth[i]) return;
      const d = quaternionAngle(r.renderedPalm, truth[i]);
      lagAll.push(d);
      if (!r.palmRejected) lag.push(d);
    });
    const palm = stat(rows.map((r) => r.palmError).filter((v) => v !== null));
    const trueLag = stat(lag), withRejects = stat(lagAll);
    const rejected = rows.filter((r) => r.palmRejected).length;
    check(trueLag.med < 25 && trueLag.p90 < 60,
      "8. palm orientation follows the landmarks, roll included",
      `lag behind the zero-phase palm: median ${trueLag.med?.toFixed(1)}° p90 ${trueLag.p90?.toFixed(1)}°  ` +
      `| ${rejected} impossible frames held (incl. them: ${withRejects.p90?.toFixed(1)}° p90)  ` +
      `| vs the raw, jittering palm: ${palm.med?.toFixed(1)}°`);

    // 9. rolling the forearm about its own axis must not move the wrist
    const w = rows.map((r) => r.wrist).filter(Boolean);
    const moved = stat(w.map((x) => x.wristMoved));
    check(moved.max < 1e-4, "9. forearm roll leaves the wrist position untouched",
      `max shift ${moved.max?.toExponential(1)} avatar units`);
    const tw = stat(w.map((x) => Math.abs(x.twistWanted)));
    const ta = stat(w.map((x) => Math.abs(x.twistApplied)));
    const sw = stat(w.map((x) => x.swingWanted));
    const sa = stat(w.map((x) => x.swingApplied));
    say(`  wrist solve: twist wanted med ${tw.med?.toFixed(0)}° p90 ${tw.p90?.toFixed(0)}° -> applied med ${ta.med?.toFixed(0)}°  |  ` +
        `swing wanted med ${sw.med?.toFixed(0)}° p90 ${sw.p90?.toFixed(0)}° -> applied med ${sa.med?.toFixed(0)}°`);

    // 7. round trip: what was commanded is what the rig actually shows
    const rt = { fingers: [], thumb: [] };
    for (const r of rows) {
      if (!r.commanded) continue;
      for (const f of FINGERS) for (const k of ["mcp", "pip"]) {
        (f === "thumb" ? rt.thumb : rt.fingers).push(
          Math.abs(r.rendered[f][k] - r.commanded[f][k]));
      }
    }
    const rtF = stat(rt.fingers), rtT = stat(rt.thumb);
    check(rtF.med < 3 && rtF.p90 < 8,
      "7. commanded angles are what the rig renders",
      `fingers median ${rtF.med?.toFixed(1)}° p90 ${rtF.p90?.toFixed(1)}°  ` +
      `| thumb median ${rtT.med?.toFixed(1)}° p90 ${rtT.p90?.toFixed(1)}°`);

    // 6. coverage, from the first frame the hand is actually seen onward
    const first = rows.findIndex((r) => !r.coasting);
    const since = first < 0 ? [] : rows.slice(first);
    const named = since.filter((r) => r.shape).length;
    check(first >= 0 && named === since.length,
      "6. a handshape is held on every frame once the hand is seen",
      `${named}/${since.length} (first sighting frame ${first})`);

    const held = {};
    for (const r of rows) held[r.shape ?? "—"] = (held[r.shape ?? "—"] ?? 0) + 1;
    say(`  shapes held: ${Object.entries(held).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k} ${v}`).join(", ")}`);
    const conf = stat(live.map((r) => r.confidence));
    const snap = stat(live.map((r) => r.snap));
    const d0 = stat(live.map((r) => r.match.d));
    const gap = stat(live.map((r) => r.match.runnerUp - r.match.d));
    say(`  match confidence median ${conf.med?.toFixed(2)}   snap median ${snap.med?.toFixed(2)}`);
    say(`  best distance med ${d0.med?.toFixed(3)} p90 ${d0.p90?.toFixed(3)}   ` +
        `runner-up gap med ${gap.med?.toFixed(3)} p90 ${gap.p90?.toFixed(3)}\n`);
  }

  // Source-data quality. The solver can only render what the landmarks contain,
  // so a sign that reads wrong needs this section before anything is tuned.
  say("=== source data per sign span ===");
  for (const span of timeline) {
    const inSpan = frames.filter((f) => f.i >= span.start_frame && f.i < span.end_frame);
    const live = inSpan.filter((f) => !f.hands.right.coasting);
    // A hard cut: extension jumping further in one frame than a hand can move.
    let cuts = 0, longestGap = 0, gap = 0;
    for (let i = 1; i < inSpan.length; i++) {
      const a = inSpan[i - 1].hands.right, b = inSpan[i].hands.right;
      gap = b.coasting ? gap + 1 : 0;
      longestGap = Math.max(longestGap, gap);
      if (a.coasting || b.coasting || !a.raw || !b.raw) continue;
      const jump = Math.max(...FINGERS.map((f) => Math.abs(b.rendered[f].pip - a.rendered[f].pip)));
      if (jump > 35) cuts += 1;
    }
    let run = 0, best = 0, name = "—";
    for (let i = 0; i < inSpan.length; i++) {
      const sh = inSpan[i].hands.right.shape;
      run = i > 0 && sh === inSpan[i - 1].hands.right.shape ? run + 1 : 1;
      if (run > best) { best = run; name = sh ?? "—"; }
    }
    say(`  ${span.gloss.padEnd(10)} tracked ${String(Math.round((100 * live.length) / inSpan.length)).padStart(3)}%  ` +
        `longest dropout ${String(longestGap).padStart(2)}f  hard cuts ${cuts}  ` +
        `longest steady shape ${name} × ${best}f of ${inSpan.length}`);
  }

  say("\n=== handshape per sign (over the hold: middle half of each span) ===");
  for (const span of timeline) {
    const q = (span.end_frame - span.start_frame) / 4;
    const lo = Math.round(span.start_frame + q), hi = Math.round(span.end_frame - q);
    const inSpan = frames.filter((f) => f.i >= lo && f.i < hi);
    const dominant = (side) => {
      const tally = {};
      for (const f of inSpan) { const s = f.hands[side].shape ?? "—"; tally[s] = (tally[s] ?? 0) + 1; }
      const ranked = Object.entries(tally).sort((a, b) => b[1] - a[1]);
      return ranked.slice(0, 2)
        .map(([n, c]) => `${n} ${Math.round((100 * c) / inSpan.length)}%`).join(", ");
    };
    say(`  ${span.gloss.padEnd(10)} L ${dominant("left").padEnd(26)} R ${dominant("right")}`);
  }

  say("\n=== playback loop and unglossed stretches ===");
  let failedWrap = 0;
  const wrapOk = wrap.turn < 30 && wrap.move < 0.06;
  if (!wrapOk) failedWrap = 1;
  say(`  ${wrapOk ? "PASS" : "FAIL"}  12. looping the sequence does not teleport the avatar` +
      `  — wrap turns the wrist ${wrap.turn.toFixed(1)}° and moves it ${wrap.move.toFixed(3)} units in one frame`);
  const runs = [];
  for (const i of unglossed) {
    if (runs.length && runs[runs.length - 1][1] === i - 1) runs[runs.length - 1][1] = i;
    else runs.push([i, i]);
  }
  say(`  frames with no gloss (shown as "—" in the HUD): ${runs.map((r) => `${r[0]}–${r[1]}`).join(", ")}`);
  say(`  the last sign ends at frame ${timeline[timeline.length - 1].end_frame} of ${frames.length};` +
      ` the tail is source data with the clip run out (its final frames are duplicates)`);
  for (const side of ["left", "right"]) {
    const tail = frames.filter((f) => f.i >= timeline[timeline.length - 1].end_frame);
    const st = stat(tail.map((f) => f.hands[side].wrist)
      .filter(Boolean)
      .map((w) => Math.abs(w.bendWanted - w.bendApplied) + Math.abs(w.deviateWanted - w.deviateApplied)));
    const rej = tail.filter((f) => f.hands[side].palmRejected).length;
    say(`  ${side} tail: wrist swing discarded median ${st.med?.toFixed(1)}° p90 ${st.p90?.toFixed(1)}°` +
        `  | impossible palm measurements rejected on ${rej}/${tail.length} frames`);
  }
  failures += failedWrap;

  say("\n=== right-hand shape trace ===");
  for (const span of timeline) {
    const trace = [];
    for (const f of frames) {
      if (f.i < span.start_frame || f.i >= span.end_frame) continue;
      const s2 = f.hands.right.shape ?? "-";
      if (!trace.length || trace[trace.length - 1][0] !== s2) trace.push([s2, 0]);
      trace[trace.length - 1][1] += 1;
    }
    say(`  ${span.gloss.padEnd(10)} ${trace.map(([n, c]) => `${n}×${c}`).join(" -> ")}`);
  }

  // screenshots at each sign's midpoint
  say("\n=== stills ===");
  for (const span of timeline) {
    const mid = Math.floor((span.start_frame + span.end_frame) / 2);
    await page.evaluate((m) => {
      const { retargeter, show } = window.__dbg;
      retargeter.reset();
      for (let i = Math.max(0, m - 12); i <= m; i++) show(i);   // warm the filters
    }, mid);
    const file = `${OUT}/${span.gloss}_${mid}.png`;
    await page.locator("#canvas-wrap").screenshot({ path: file });
    say(`  ${span.gloss} frame ${mid} -> ${file}`);
  }

  writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 1));
  say(`\n${failures === 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED"}`);
  writeFileSync(`${OUT}/report.txt`, lines.join("\n"));

  if (!KEEP) await browser.close();
  die(failures === 0 ? 0 : 1);
} catch (error) {
  console.error(error);
  die(2);
}
