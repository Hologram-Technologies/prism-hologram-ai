// GP-16's browser half: a prompt's computed prefix kept and restored, and the
// positions after it compared bit for bit with computing the whole prompt.
// Prints one JSON line. Uses the Playwright the SDK image carries.
//
//   node crates/conformance/tests/gpu/prefix-run.mjs <repository root>

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
    const { tinyQwen2 } = await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`);
    if (!navigator.gpu) return { error: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: "no WebGPU adapter" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));
    const llama3 = { factor: 4, low: 1, high: 4, orig: 64 };
    const results = {};
    for (const [key, model] of [
      ["qwen2", tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16" })],
      ["llama", tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16", llama3 })],
    ]) {
      const make = () => createEngine({ device, dims: model.dims, tensors: model.tensors, capacity: 128 });
      const vocab = model.dims.vocab;
      // A 40-token prefix, more than one 32-position submission, and an
      // 8-token suffix; then 8 greedy picks.
      const prefix = Array.from({ length: 40 }, (_, i) => (i * 37 + 11) % vocab);
      const suffix = Array.from({ length: 8 }, (_, i) => (i * 53 + 7) % vocab);
      const after = async (engine, tokens) => {
        const logits = [];
        let next = null;
        for (let i = 0; i < tokens.length + 8; i++) {
          const g = await engine.step(i < tokens.length ? tokens[i] : next, { logits: true });
          logits.push(Array.from(g.logits));
          next = g.token;
        }
        return logits;
      };
      const t0 = performance.now();
      const whole = await make();
      for (const t of prefix) await whole.step(t);
      const wanted = await after(whole, suffix);
      const wholeMs = performance.now() - t0;

      // B computes the prefix, keeps it, then starts again from position 0 with
      // 50 other tokens, overwriting every row the prefix held, then restores
      // the prefix and continues.
      const other = Array.from({ length: 50 }, (_, i) => (i * 91 + 5) % vocab);
      const b = await make();
      for (const t of prefix) await b.step(t);
      const kept = b.saveState();
      b.reset();
      for (const t of other) await b.step(t);
      const t1 = performance.now();
      b.restoreState(kept);
      const restoredAt = b.position;
      const fromB = await after(b, suffix);
      const restoredMs = performance.now() - t1;
      // C is a fresh engine of the same model, given B's prefix.
      const c = await make();
      c.restoreState(kept);
      const fromC = await after(c, suffix);
      const differing = (got) => got.reduce((n, row, i) => n + row.reduce((m, v, j) => m + (Object.is(v, wanted[i][j]) ? 0 : 1), 0), 0);

      // The batched path: prefix in blocks, kept, restored, suffix run.
      const d = await make();
      await d.run(prefix, 1);
      const keptRun = d.saveState(prefix.length);
      d.reset();
      await d.run(other, 1);
      d.restoreState(keptRun);
      const resumed = await d.run(suffix, 12);
      const e = await make();
      const straight = await e.run([...prefix, ...suffix], 12);

      const refusal = (f) => { try { f(); return null; } catch (x) { return x.message; } };
      results[key] = {
        restoredAt, positions: wanted.length, differingB: differing(fromB), differingC: differing(fromC),
        resumed, straight, bytes: kept.bytes, wholeMs, restoredMs,
        tooLong: refusal(() => b.saveState(b.position + 1)),
        otherShape: refusal(() => c.restoreState({ ...kept, shape: { ...kept.shape, layers: kept.shape.layers + 1 } })),
      };
    }
    return { adapter: adapter.info ? adapter.info.description || adapter.info.vendor : "?", errors, results };
  }, origin);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
