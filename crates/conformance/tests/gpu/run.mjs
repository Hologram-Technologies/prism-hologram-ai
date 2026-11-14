// GP-01's browser half: serve the repository on a local port, open it in
// headless Chromium with WebGPU on its software renderer, and run the engine
// and the CPU reference over the same prompt, step by step. Prints one JSON
// line. Uses the Playwright the SDK image carries, so nothing is installed.
//
//   node crates/conformance/tests/gpu/run.mjs <repository root>

import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = process.argv[2];
const playwright = process.env.PRISM_PLAYWRIGHT ?? "/opt/prismpm/oracles/node_modules/playwright/index.mjs";
const { chromium } = await import(playwright);

const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".html": "text/html" };
const server = http.createServer(async (request, response) => {
  const path = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname));
  if (path === "/") {
    response.setHeader("content-type", "text/html");
    response.end("<!doctype html><title>gpu</title>");
    return;
  }
  try {
    const body = await readFile(join(root, path));
    response.setHeader("content-type", types[extname(path)] ?? "application/octet-stream");
    response.end(body);
  } catch {
    response.statusCode = 404;
    response.end();
  }
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  await page.goto(`${origin}/`);
  const result = await page.evaluate(async (origin) => {
    const { createEngine } = await import(`${origin}/site/app/engine/gpu.mjs`);
    const { tinyModel, referenceModel } = await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`);
    if (!navigator.gpu) return { error: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: "no WebGPU adapter" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));

    const model = tinyModel(1);
    const engine = await createEngine({ device, dims: model.dims, tensors: model.tensors, capacity: 64 });
    const reference = referenceModel(model);

    const steps = [];
    let gpuNext = null;
    let refNext = null;
    const generate = 10;
    for (let i = 0; i < model.prompt.length + generate; i++) {
      const input = i < model.prompt.length ? model.prompt[i] : refNext;
      const g = await engine.step(input, { logits: true });
      const r = reference.step(input);
      let worst = 0;
      for (let j = 0; j < r.logits.length; j++) worst = Math.max(worst, Math.abs(g.logits[j] - r.logits[j]));
      const sorted = Array.from(r.logits).sort((a, b) => b - a);
      steps.push({ input, gpu: g.token, reference: r.token, worst, margin: sorted[0] - sorted[1], spread: sorted[0] - sorted[sorted.length - 1] });
      gpuNext = g.token;
      refNext = r.token;
    }
    return { adapter: adapter.info ? adapter.info.description || adapter.info.vendor : "?", errors, steps, gpuNext };
  }, origin);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
