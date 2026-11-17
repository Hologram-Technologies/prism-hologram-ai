// GP-13 and GP-14's browser half: serve the repository on a local port, open it
// in headless Chromium with WebGPU on its software renderer, and run the
// encoder and its CPU reference over the same sentences. Prints one JSON line.
// Uses the Playwright the SDK image carries, so nothing is installed.
//
//   node crates/conformance/tests/gpu/encode.mjs <repository root>

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
    const { createEncoder } = await import(`${origin}/site/app/engine/encoder.mjs`);
    const { tinyEncoder, referenceEncoder } = await import(`${origin}/crates/conformance/tests/gpu/encoder-reference.mjs`);
    if (!navigator.gpu) return { error: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: "no WebGPU adapter" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));
    const worst = (a, b) => a.reduce((w, v, i) => Math.max(w, Math.abs(v - b[i])), 0);

    // GP-13: all-MiniLM-L6-v2's shape. Every hidden state, and the pooled,
    // unit-length embedding, against the reference, for a short sentence and
    // one of 60 tokens.
    const bert = tinyEncoder(5, "bert");
    const bertEngine = await createEncoder({ device, dims: bert.dims, tensors: bert.tensors, capacity: 64 });
    const bertReference = referenceEncoder(bert);
    const embeddings = [];
    for (const tokens of bert.sentences) {
      const { states } = await bertEngine.hidden(tokens);
      const want = bertReference.hidden(tokens);
      const e = await bertEngine.embed(tokens);
      const r = bertReference.embed(tokens);
      let dot = 0;
      for (let k = 0; k < r.length; k++) dot += e[k] * r[k];
      embeddings.push({ tokens: tokens.length, statesWorst: worst(states, want.flatMap((row) => Array.from(row))), embeddingWorst: worst(e, r), cosine: dot });
    }

    // GP-14: DistilBERT SST-2's shape: no token types, a classifier head. Seed
    // 5 gives the two sentences different labels, 0.38 apart or more.
    const distil = tinyEncoder(5, "distil");
    const distilEngine = await createEncoder({ device, dims: distil.dims, tensors: distil.tensors, capacity: 64 });
    const distilReference = referenceEncoder(distil);
    const classes = [];
    for (const tokens of distil.sentences) {
      const g = await distilEngine.classify(tokens);
      const r = distilReference.classify(tokens);
      const pick = (p) => p.indexOf(Math.max(...p));
      classes.push({ tokens: tokens.length, logitsWorst: worst(Array.from(g.logits), r.logits), probabilityWorst: worst(g.probabilities, r.probabilities), gpu: pick(g.probabilities), reference: pick(r.probabilities), margin: Math.abs(r.logits[0] - r.logits[1]) });
    }
    // GP-17: both encoders written as exports, placed by the role map from
    // their graphs alone, and run from the tensors it names.
    const { asEncoderExport } = await import(`${origin}/crates/conformance/tests/gpu/export.mjs`);
    const { encoderRoles, engineTensors } = await import(`${origin}/site/app/engine/onnx-roles.mjs`);
    const placed = {};
    for (const [key, model, family] of [["bert", bert, "bert"], ["distilbert", distil, "distilbert"]]) {
      const exp = asEncoderExport(model, family);
      const { dims, weights, labels } = encoderRoles(exp, exp.config);
      const engine = await createEncoder({ device, dims, tensors: engineTensors(weights, (n) => exp.bytes.get(n)), capacity: 64 });
      const reference = referenceEncoder(model);
      const sameAsModel = [...weights.keys()].every((n) => model.tensors.has(n)) && [...model.tensors.keys()].every((n) => weights.has(n));
      const runs = [];
      for (const tokens of model.sentences) {
        if (family === "bert") runs.push({ tokens: tokens.length, worst: worst(await engine.embed(tokens), reference.embed(tokens)) });
        else {
          const g = await engine.classify(tokens);
          const r = reference.classify(tokens);
          runs.push({ tokens: tokens.length, worst: worst(Array.from(g.logits), r.logits), gpu: g.probabilities.indexOf(Math.max(...g.probabilities)), reference: r.probabilities.indexOf(Math.max(...r.probabilities)) });
        }
      }
      placed[key] = { placed: weights.size, sameAsModel, labels, runs };
    }
    return { adapter: adapter.info ? adapter.info.description || adapter.info.vendor : "?", errors, embeddings, classes, placed };
  }, origin);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
