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
    const { tinyModel, tinyQwen2, referenceModel } = await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`);
    if (!navigator.gpu) return { error: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: "no WebGPU adapter" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));

    // The same three checks for each model: step by step against the
    // reference, 140 positions against it, and blocks against one at a time.
    const exercise = async (model) => {
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
    // GP-01 past one attention chunk: 140 positions span three of attention's
    // 64-position chunks, so the combine of partials is compared with the
    // reference, not only one chunk alone.
    const longEngine = await createEngine({ device, dims: model.dims, tensors: model.tensors, capacity: 192 });
    const longReference = referenceModel(model);
    let longWorst = 0;
    let longPositions = 0;
    for (let i = 0; i < 140; i++) {
      const input = (i * 53 + 7) % model.dims.vocab;
      const g = await longEngine.step(input, { logits: true });
      const r = longReference.step(input);
      for (let j = 0; j < r.logits.length; j++) longWorst = Math.max(longWorst, Math.abs(g.logits[j] - r.logits[j]));
      longPositions++;
    }
    // GP-04: many positions per submission against one per submission, on the
    // same model, for a short prompt and one longer than a submission's slots.
    const batched = [];
    for (const prompt of [model.prompt, Array.from({ length: 100 }, (_, i) => (i * 37 + 11) % model.dims.vocab)]) {
      const one = await createEngine({ device, dims: model.dims, tensors: model.tensors, capacity: 128 });
      const many = await createEngine({ device, dims: model.dims, tensors: model.tensors, capacity: 128 });
      const stepwise = await one.generate(prompt, 12);
      const together = await many.run(prompt, 12);
      batched.push({ promptLength: prompt.length, stepwise, together, position: many.position });
    }
    return { steps, gpuNext, batched, longWorst, longPositions, model: model.tensors };
    };
    // GP-01 and GP-04 on the ternary Qwen3 model; GP-05 on the q4 Qwen2 one.
    const ternary = await exercise(tinyModel(1));
    const qwen2 = await exercise(tinyQwen2(3));
    // GP-08: the same model with every weight given as ONNX's MatMulNBits
    // tensors, f32 scales (a q4 export) and f16 ones (q4f16). sameBytes records
    // whether each q4 export tensor is byte for byte the engine's q4 block.
    const q4 = tinyQwen2(3, "onnx", "f32");
    const formats = (model) => [...new Set([...model.tensors.values()].filter((t) => t.parts).map((t) => t.fmt))];
    const sameBytes = [...q4.tensors].filter(([, t]) => t.parts).every(([name, t]) => {
      const block = qwen2.model.get(name).bytes;
      const [codes, scales] = t.parts;
      return codes.every((b, i) => b === block[i]) && scales.every((b, i) => b === block[codes.length + i]);
    });
    const q4f16 = tinyQwen2(3, "onnx", "f16");
    const onnx = {
      sameBytes,
      q4: { formats: formats(q4), ...(await exercise(q4)) },
      q4f16: { formats: formats(q4f16), ...(await exercise(q4f16)) },
    };
    delete qwen2.model;
    delete ternary.model;
    return { adapter: adapter.info ? adapter.info.description || adapter.info.vendor : "?", errors, ...ternary, qwen2, onnx };
  }, origin);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
