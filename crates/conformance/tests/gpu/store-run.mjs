// GP-15's browser half: two visits to the same origin in one browser profile.
// The first stores two models' tensors by address; the second, with the
// network off, reads one model back, admits each tensor again, runs it on
// WebGPU beside the CPU reference, and meets a tampered file. Prints one JSON
// line. Uses the Playwright the SDK image carries, so nothing is installed.
//
//   node crates/conformance/tests/gpu/store-run.mjs <repository root>

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
  const context = await browser.newContext();
  const first = await context.newPage();
  await first.goto(`${origin}/`);
  const visit = await first.evaluate(async (origin) => {
    const { openStore } = await import(`${origin}/site/app/engine/store.mjs`);
    const { blake3Hex } = await import(`${origin}/try/vendor/prism-code/src/blake3.mjs`);
    const { tinyQwen2 } = await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`);
    const address = (bytes) => `blake3:${blake3Hex(bytes)}`;
    // The test's admission stands in for the kernel's: the label of the bytes
    // equals the address asked for. The page asks the kernel the same question.
    const store = await openStore({ admit: async (a, bytes) => a === address(bytes) });
    const blocks = (model) => [...model.tensors].flatMap(([name, t]) => (t.parts ?? [t.bytes]).map((bytes, part) => ({ name, part, bytes, address: address(bytes) })));
    const a = tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16" });
    // B is A with layer 1's feed-forward from another seed: three tensors,
    // five blocks, differ; every other block is A's.
    const b = tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16" });
    const other = tinyQwen2(4, "onnx", "f16", "dense", { vectors: "f16" });
    for (const n of ["l1.w_up", "l1.w_down", "l1.ffn_norm"]) b.tensors.set(n, other.tensors.get(n));
    const putAll = async (list) => {
      for (const x of list) await store.put(x.address, x.bytes);
      return { ...store.counts };
    };
    const aBlocks = blocks(a);
    const bBlocks = blocks(b);
    const afterA = await putAll(aBlocks);
    const afterB = await putAll(bBlocks);
    const afterAgain = await putAll(aBlocks);
    const aSet = new Set(aBlocks.map((x) => x.address));
    const newInB = bBlocks.filter((x) => !aSet.has(x.address));
    const sharedBytes = bBlocks.filter((x) => aSet.has(x.address)).reduce((s, x) => s + x.bytes.byteLength, 0);
    // What the second visit needs to rebuild A: each tensor's format and
    // shape, and the addresses of its blocks. In the page this is the
    // manifest the kernel admits (decision 28).
    const manifest = [...a.tensors].map(([name, t]) => ({ name, fmt: t.fmt, N: t.N, K: t.K, parts: !!t.parts, blocks: aBlocks.filter((x) => x.name === name).map((x) => ({ address: x.address, length: x.bytes.byteLength })) }));
    return {
      a: { blocks: aBlocks.length, bytes: aBlocks.reduce((s, x) => s + x.bytes.byteLength, 0), distinct: aSet.size },
      b: { blocks: bBlocks.length, newBlocks: newInB.length, newBytes: newInB.reduce((s, x) => s + x.bytes.byteLength, 0), sharedBytes },
      afterA, afterB, afterAgain, held: (await store.list()).length, manifest,
    };
  }, origin);

  const second = await context.newPage();
  await second.goto(`${origin}/`);
  // Every module the second visit runs is loaded before the network goes.
  await second.evaluate(async (origin) => {
    globalThis.parts = {
      ...(await import(`${origin}/site/app/engine/store.mjs`)),
      ...(await import(`${origin}/site/app/engine/gpu.mjs`)),
      ...(await import(`${origin}/try/vendor/prism-code/src/blake3.mjs`)),
      ...(await import(`${origin}/crates/conformance/tests/gpu/reference.mjs`)),
    };
  }, origin);
  await context.setOffline(true);
  const back = await second.evaluate(async ({ origin, manifest }) => {
    const { openStore, createEngine, blake3Hex, tinyQwen2, referenceModel } = globalThis.parts;
    let offline;
    try {
      await fetch(`${origin}/README.md`, { cache: "no-store" });
      offline = false;
    } catch {
      offline = true;
    }
    let admitted = 0;
    const store = await openStore({ admit: async (a, bytes) => (a === `blake3:${blake3Hex(bytes)}` ? (admitted++, true) : false) });
    const tensors = new Map();
    let missing = 0;
    for (const t of manifest) {
      const got = [];
      for (const blk of t.blocks) {
        const bytes = await store.get(blk.address, blk.length);
        if (!bytes) missing++;
        got.push(bytes);
      }
      tensors.set(t.name, t.parts ? { fmt: t.fmt, N: t.N, K: t.K, parts: got } : { fmt: t.fmt, N: t.N, K: t.K, bytes: got[0] });
    }
    // The model rebuilt from the store against the reference of the model
    // generated afresh: sixteen steps, greedy, every logit.
    const model = tinyQwen2(3, "onnx", "f16", "dense", { vectors: "f16" });
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const engine = await createEngine({ device, dims: model.dims, tensors, capacity: 64 });
    const reference = referenceModel(model);
    const steps = [];
    let next = null;
    for (let i = 0; i < 16; i++) {
      const input = i < model.prompt.length ? model.prompt[i] : next;
      const g = await engine.step(input, { logits: true });
      const r = reference.step(input);
      let worst = 0;
      for (let j = 0; j < r.logits.length; j++) worst = Math.max(worst, Math.abs(g.logits[j] - r.logits[j]));
      const sorted = Array.from(r.logits).sort((x, y) => y - x);
      steps.push({ gpu: g.token, reference: r.token, worst, margin: sorted[0] - sorted[1] });
      next = r.token;
    }
    // One stored file changed by a byte: not admitted, removed, and the
    // tensor reads as missing.
    const victim = manifest.find((t) => t.name === "l0.wq").blocks[0];
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-tensors");
    const file = await (await dir.getFileHandle(victim.address.slice(7))).getFile();
    const bytes = new Uint8Array(await file.arrayBuffer());
    bytes[100] ^= 1;
    const w = await (await dir.getFileHandle(victim.address.slice(7))).createWritable();
    await w.write(bytes);
    await w.close();
    const tampered = await store.get(victim.address, victim.length);
    const afterTamper = await store.has(victim.address);
    let wrongPrefix = null;
    try {
      await store.get(`sha256:${victim.address.slice(7)}`);
    } catch (e) {
      wrongPrefix = e.message;
    }
    return { offline, missing, admitted, counts: { ...store.counts }, steps, tampered: tampered === null, afterTamper, wrongPrefix };
  }, { origin, manifest: visit.manifest });
  delete visit.manifest;
  console.log(JSON.stringify({ visit, back }));
} finally {
  await browser.close();
  server.close();
}
