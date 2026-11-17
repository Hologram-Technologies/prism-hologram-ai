// GP-18's browser half: the CPU engine in headless Chromium with WebGPU off,
// beside the CPU reference. Chromium is launched with its GPU disabled and
// without the flags that turn WebGPU on, and the page checks that no adapter
// is offered before anything runs. Prints one JSON line. Uses the Playwright
// the SDK image carries.
//
//   node crates/conformance/tests/gpu/cpu-run.mjs <repository root>

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

const browser = await chromium.launch({ headless: true, args: ["--disable-gpu", "--disable-features=WebGPU,Vulkan"] });
try {
  const page = await browser.newPage();
  await page.goto(`${origin}/`);
  const result = await page.evaluate(async (origin) => {
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
    if (adapter) return { error: "a WebGPU adapter is offered; this test needs WebGPU off" };
    const { createCpuEngine } = await import(`${origin}/site/app/engine/cpu.mjs`);
    const { tinyQwen2, referenceModel } = await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`);
    const llama3 = { factor: 4, low: 1, high: 4, orig: 64 };
    const models = {
      q4: tinyQwen2(3),
      qwen25: tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16" }),
      llama: tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16", llama3 }),
      qwen3: tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16", qkNorm: true }),
    };
    const out = { webgpu: navigator.gpu ? "no adapter" : "no navigator.gpu" };
    for (const [key, model] of Object.entries(models)) {
      const make = () => createCpuEngine({ dims: model.dims, tensors: model.tensors, capacity: 192 });
      // Sixteen steps against the reference, greedy, every logit.
      const engine = make();
      const reference = referenceModel(model);
      const steps = [];
      let next = null;
      for (let i = 0; i < model.prompt.length + 10; i++) {
        const input = i < model.prompt.length ? model.prompt[i] : next;
        const g = await engine.step(input, { logits: true });
        const r = reference.step(input);
        let worst = 0;
        for (let j = 0; j < r.logits.length; j++) worst = Math.max(worst, Math.abs(g.logits[j] - r.logits[j]));
        const sorted = Array.from(r.logits).sort((a, b) => b - a);
        steps.push({ gpu: g.token, reference: r.token, worst, margin: sorted[0] - sorted[1] });
        next = r.token;
      }
      // 140 positions, past where a cache would wrap or a chunk would end.
      const long = make();
      const longReference = referenceModel(model);
      let longWorst = 0;
      for (let i = 0; i < 140; i++) {
        const input = (i * 53 + 7) % model.dims.vocab;
        const g = await long.step(input, { logits: true });
        const r = longReference.step(input);
        for (let j = 0; j < r.logits.length; j++) longWorst = Math.max(longWorst, Math.abs(g.logits[j] - r.logits[j]));
      }
      // The run path, as the page calls it, against stepping one at a time.
      const prompt = Array.from({ length: 40 }, (_, i) => (i * 37 + 11) % model.dims.vocab);
      const one = make();
      const stepwise = [];
      let t = null;
      for (let i = 0; i < prompt.length + 11; i++) {
        t = (await one.step(i < prompt.length ? prompt[i] : t)).token;
        if (i >= prompt.length - 1) stepwise.push(t);
      }
      const together = await make().run(prompt, 12);
      // A kept prefix restored after its rows were overwritten.
      const b = make();
      for (const p of prompt) await b.step(p);
      const kept = b.saveState();
      b.reset();
      for (let i = 0; i < 50; i++) await b.step((i * 91 + 5) % model.dims.vocab);
      b.restoreState(kept);
      const resumed = await b.run([3, 4, 5], 6);
      const fresh = make();
      const straight = await fresh.run([...prompt, 3, 4, 5], 6);
      const t0 = performance.now();
      const timed = make();
      for (let i = 0; i < 32; i++) await timed.step(i % model.dims.vocab);
      out[key] = { steps, longWorst, longPositions: 140, stepwise, together, resumed, straight, msPerStep: (performance.now() - t0) / 32 };
    }
    return out;
  }, origin);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
