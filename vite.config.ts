import { defineConfig } from "vite";

export default defineConfig({
  root: "frontend",
  publicDir: "../assets",
  server: { port: 5173 },
  // pose-format's JS reader is written against Node's Buffer; the browser has no
  // global for it, so it is aliased to the npm polyfill and shimmed onto window.
  resolve: { alias: { buffer: "buffer" } },
  define: { "global": "globalThis" },
  optimizeDeps: { include: ["buffer", "pose-format"] },
});
