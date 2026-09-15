/**
 * Render the demo sequence to PNG frames for the virtual camera.
 *
 *   node scripts/render_frames.mjs [out_dir] [--width 1280] [--height 720]
 *
 * The avatar is driven by Three.js in the page, and `pyvirtualcam` is Python, so
 * the frames cross that boundary as files. Pre-rendering rather than streaming
 * live is deliberate for a demo: a dropped frame here is a slow build, not a
 * stutter in front of an audience, and the same frames can be replayed exactly.
 *
 * Playback is paused before capture — the render loop repaints continuously, so
 * without that the screenshot catches whichever frame it happened to be on
 * rather than the one asked for.
 */

import { spawn } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const OUT = args.find((a) => !a.startsWith("--") && !/^\d+$/.test(a)) ?? "frames";
const WIDTH = flag("width", 1280);
const HEIGHT = flag("height", 720);
const PORT = 5211;

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const server = spawn("npx", ["vite", "--port", String(PORT), "--strictPort"], {
  cwd: process.cwd(), stdio: ["ignore", "pipe", "pipe"], detached: true,
});
const stop = () => { try { process.kill(-server.pid, "SIGKILL"); } catch {} };

try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("vite did not start")), 60000);
    server.stdout.on("data", (b) => {
      if (b.toString().includes(String(PORT))) { clearTimeout(timer); resolve(); }
    });
  });

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 });

  await page.click("#play");                       // stop the render loop repainting
  await page.evaluate(() => {
    for (const id of ["hud", "panel", "controls"]) {
      const el = document.getElementById(id);
      if (el) el.style.display = "none";
    }
    window.dispatchEvent(new Event("resize"));     // let the canvas take the space
  });
  await page.waitForTimeout(250);

  const total = await page.evaluate(() => {
    const { retargeter, seq } = window.__dbg;
    retargeter.reset();
    return seq.frameCount;
  });

  const target = page.locator("#canvas-wrap");
  for (let i = 0; i < total; i++) {
    await page.evaluate((f) => window.__dbg.show(f), i);
    await target.screenshot({ path: `${OUT}/${String(i).padStart(4, "0")}.png` });
    if (i % 40 === 0) process.stdout.write(`\r  rendered ${i}/${total}`);
  }
  console.log(`\r  rendered ${total}/${total} frames into ${OUT}/        `);

  const fps = await page.evaluate(() => window.__dbg.seq.fps);
  console.log(`  sequence plays at ${fps}fps (${(total / fps).toFixed(2)}s)`);
  await browser.close();
  stop();
  process.exit(0);
} catch (error) {
  console.error(error);
  stop();
  process.exit(1);
}
