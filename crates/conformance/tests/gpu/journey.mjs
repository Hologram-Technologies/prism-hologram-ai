// GP-03's journey: the engine page, served with the holo and the wasm module
// the build made, loads a weight container through the verified kernel, and
// refuses a tampered one where it should. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/journey.mjs <assembled site directory>
//
// The site directory is what Pages publishes: site/ with app.holo, vendor/ and
// app/engine/engine.wasm in place.

import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { buildFixture, buildOnnxFixture, hubFiles, onnxHubFiles, onnxInitializers, uorIndexFiles } from "./fixture.mjs";
import { referenceModel } from "./reference.mjs";
import { blake3Hex } from "../../../../try/vendor/prism-code/src/blake3.mjs";
import { mintTrust, signRoot } from "./hub-sign.mjs";
import { buildEncoderFixture, ENCODER_TEXTS } from "./encoder-export.mjs";

const site = process.argv[2];
const playwright = process.env.PRISM_PLAYWRIGHT ?? "/opt/prismpm/oracles/node_modules/playwright/index.mjs";
// No browser is a failure, never a skip: a journey that did not run proves nothing.
let chromium;
try {
  ({ chromium } = await import(playwright));
} catch (error) {
  console.error(`journey: no Playwright at ${playwright} (set PRISM_PLAYWRIGHT): ${error.message}`);
  process.exit(1);
}

// GP-27: the page proves each index root by its signature against the policy
// it pins. The journey serves it a policy naming a chain minted here, and
// signs each fixture hub's root through that chain; the foreign hub's root is
// signed through another chain, which the policy does not name.
const trust = await mintTrust();
const foreignTrust = await mintTrust();
const bundles = new Map();
const bundleFor = async (root, through = trust) => {
  const key = `${through === trust ? "t" : "f"}:${blake3Hex(root)}`;
  if (!bundles.has(key)) bundles.set(key, await signRoot(through, root));
  return bundles.get(key);
};

const fixture = buildFixture();
// One byte inside a block the loader takes after many others: its label no
// longer matches, so the kernel must refuse that block and nothing after it.
const victim = fixture.dir.find((e) => e.name === "l1.wq");
const flipped = Buffer.from(fixture.container);
flipped[victim.off + Math.floor(victim.len / 2)] ^= 0x01;

const variants = {
  good: { container: fixture.container, sha256: fixture.sha256 },
  flip: { container: flipped, sha256: fixture.sha256 },
  root: { container: fixture.container, sha256: "f".repeat(64) },
  // The deploy lists a kernel other than the one the browser holds, as when the
  // service worker serves app.holo from an earlier deploy's cache.
  stale: { container: fixture.container, sha256: fixture.sha256, staleKernel: true },
  // GP-27: the index root signed by a key the page's policy does not name.
  foreign: { container: fixture.container, sha256: fixture.sha256, foreignKey: true },
};

// GP-19: an ONNX export read from its own files. A byte of the graph flipped
// inside layer 1's query weight, so the file is not the one the record names;
// and a record naming another SHA-256 for tokenizer.json.
// GP-23: a second turn, with other text, after the first.
async function secondTurn(page) {
  await page.evaluate(() => { window.__qengine.stats = null; });
  // Its first letter is not the first turn's, so no id past the head is shared.
  await page.fill("#prompt", "yes");
  await page.press("#prompt", "Enter");
  await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
  return page.evaluate(() => window.__qengine.output);
}

// GP-30 to GP-32: one visit to the engine page in `context`, loading `repo`
// from `variant`'s hub, then a turn per text; answers the page's state, its
// turns, the kept entries and the byte ranges the server sent meanwhile.
async function onnxVisit(context, { variant, repo = onnx.repo, precision = null, texts = ["hi"], before = async () => {} }) {
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e.message)));
  await before(page);
  const from = sent.length;
  const url = `${origin}/app/?onnx=${repo}&index=/onnx/${variant}/index${precision ? `&precision=${precision}` : ""}`;
  const opened = await page.goto(url).then(() => true, (e) => (errors.push(`goto: ${e.message}`), false));
  const turns = [];
  if (opened) {
    await page.click("#load");
    await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 }).catch(() => {});
    for (const text of (await page.evaluate(() => window.__qengine.phase)) === "ready" ? texts : []) {
      await page.evaluate(() => { window.__qengine.stats = null; });
      await page.fill("#prompt", text);
      await page.press("#prompt", "Enter");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      turns.push(await page.evaluate(() => ({ ...window.__qengine.output, stats: window.__qengine.stats })));
    }
  }
  const state = opened ? await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, device: window.__qengine.device ?? null, checks: window.__qengine.checks })) : { phase: "unopened", error: errors[0], checks: [] };
  const held = opened ? await page.evaluate(async () => {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-files", { create: true });
    const out = [];
    for await (const [n, h] of dir.entries()) if (h.kind === "file") out.push([n, (await h.getFile()).size]);
    return out;
  }) : [];
  const worker = opened ? await page.evaluate(() => navigator.serviceWorker?.ready.then((r) => Boolean(r.active))) : false;
  await page.close();
  return { ...state, turns, output: turns[0] ?? null, errors, held, worker, sent: sent.slice(from) };
}

const onnx = buildOnnxFixture();
// GP-35 and GP-36: an embedding model and a classifier, each its own
// repository on its own hub, served as the ONNX export is.
const encoders = { bert: buildEncoderFixture("bert"), distil: buildEncoderFixture("distil") };
const encoderExpected = JSON.parse(await readFile(new URL("./encoder-expected.json", import.meta.url), "utf8"));
// GP-24: another model's kept file, named by its BLAKE3.
const foreign = { size: 300000 };
foreign.name = blake3Hex(new Uint8Array(foreign.size).fill(7));
const flippedGraph = Buffer.from(onnx.files.get(onnx.path));
flippedGraph[Math.floor(flippedGraph.length * 0.6)] ^= 0x01;
const onnxVariants = {
  good: { files: onnx.files },
  cpu: { files: onnx.files, noGpu: true },
  flip: { files: new Map([...onnx.files, [onnx.path, flippedGraph]]), blake3: { [onnx.path]: onnx.b3(onnx.files.get(onnx.path)) } },
  tokenizer: { files: onnx.files, blake3: { "tokenizer.json": "e".repeat(64) } },
  // GP-22: a precision the record does not offer.
  q4: { files: onnx.files, precision: "q4" },
  // As the live hub's records are: the files Git holds itself named by their
  // Git blob SHA-1 alone; and one of those naming another blob.
  // GP-21: visits in one browser profile; with offline set, the origin
  // serves the record but none of the files.
  kept: { files: onnx.files, offline: false },
  // GP-26: a record that names config.json no BLAKE3.
  noblake3: { files: onnx.files, noBlake3: ["config.json"] },
  // GP-30: visits in one profile, the last with the network off.
  home: { files: onnx.files },
  // GP-31: the export with its q4 graph beside the q4f16 one, and a second
  // repository holding the same files, whose origin answers nothing and whose
  // one listed source does.
  store: {
    files: new Map([...onnx.files, [onnx.q4.path, onnx.q4.bytes]]),
    twins: [{ repo: `${onnx.repo}-twin`, resolve: "/onnx/store/gone/", sources: [{ kind: "mirror", resolve: "/onnx/store/files/", missing: [] }] }],
  },
  // GP-31: a UOR index that does not list the model, and one that gives one
  // tensor an address that is not its bytes'.
  unlisted: { files: onnx.files, uorRepos: [] },
  lie: { files: onnx.files, lie: { repo: onnx.repo, name: "model.layers.0.attn.q_proj.MatMul.weight_Q4" } },
  // GP-32: two turns of the same text.
  again: { files: onnx.files },
};

// GP-30: with `down` set the server answers nothing at all, as with the
// network off; GP-31: every file byte range the server sent, by variant.
let down = false;
const sent = [];
// The UOR index each variant's hub publishes, computed once from the true
// files (a variant serving altered bytes still has the hub's true addresses).
const uorFor = new Map();
const uorOf = (name) => {
  if (!uorFor.has(name)) {
    const v = onnxVariants[name];
    const graphs = [...(v.uorFiles ?? (v.twins ? v.files : onnx.files))].filter(([p]) => p.endsWith(".onnx"));
    const repos = v.uorRepos ?? [onnx.repo, ...(v.twins ?? []).map((t) => t.repo)];
    const rows = repos.flatMap((repo) => graphs.map(([path, bytes]) => ({ repo, revision: onnx.revision, path, bytes, resolve: `/onnx/${name}/files/` })));
    // An index that does not list the model still lists another.
    if (!rows.length) rows.push({ repo: "fixture/another", revision: onnx.revision, path: onnx.path, bytes: onnx.files.get(onnx.path), resolve: "/" });
    uorFor.set(name, uorIndexFiles(rows, { lie: v.lie }));
  }
  return uorFor.get(name);
};
// The tensors kept apart from the rest of a graph (APART in files.mjs).
const apart = (bytes) => onnxInitializers(bytes).filter((t) => t.length >= 4096).map((t) => ({ ...t, address: blake3Hex(bytes.subarray(t.offset, t.offset + t.length)) }));

const types = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm" };
const server = http.createServer(async (request, response) => {
  if (down) return request.socket.destroy();
  const path = decodeURIComponent(new URL(request.url, "http://x").pathname);
  const fixtureRoute = path.match(/^\/fixture\/(\w+)\/(.*)$/);
  if (fixtureRoute && variants[fixtureRoute[1]]) {
    const v = variants[fixtureRoute[1]];
    const hub = hubFiles(fixture, { resolve: `/fixture/${fixtureRoute[1]}/`, sha256: v.sha256 });
    const rest = fixtureRoute[2];
    if (rest === "index/root.json") return response.end(hub.root);
    if (rest === `index/models/${fixture.repo}.json`) return response.end(hub.record);
    if (rest === "index/index.sigstore.json") return response.end(await bundleFor(hub.root, v.foreignKey ? foreignTrust : trust));
    if (rest === fixture.path) {
      const range = /bytes=(\d+)-(\d+)?/.exec(request.headers.range ?? "");
      if (!range) return response.end(v.container);
      const start = Number(range[1]);
      const end = range[2] === undefined ? v.container.length - 1 : Number(range[2]);
      response.statusCode = 206;
      response.setHeader("content-range", `bytes ${start}-${end}/${v.container.length}`);
      return response.end(v.container.subarray(start, end + 1));
    }
    response.statusCode = 404;
    return response.end();
  }
  const encoderRoute = path.match(/^\/encoder\/(bert|distil)\/(.*)$/);
  if (encoderRoute) {
    const f = encoders[encoderRoute[1]];
    const resolve = `/encoder/${encoderRoute[1]}/files/`;
    const hub = onnxHubFiles(f, { resolve });
    const uor = uorIndexFiles([{ repo: f.repo, revision: f.revision, path: f.path, bytes: f.files.get(f.path), resolve }]);
    const rest = encoderRoute[2];
    if (rest === "index/root.json") return response.end(hub.root);
    if (rest === `index/models/${f.repo}.json`) return response.end(hub.record);
    if (rest === "index/index.sigstore.json") return response.end(await bundleFor(hub.root));
    if (rest === "uor/root.json") return response.end(uor.root);
    const shard = rest.startsWith("uor/") ? uor.shards.get(rest.slice(4)) : null;
    if (shard) return response.end(shard);
    const served = rest.startsWith("files/") ? f.files.get(rest.slice(6)) : null;
    if (served) return response.end(served);
    response.statusCode = 404;
    return response.end();
  }
  const onnxRoute = path.match(/^\/onnx\/(\w+)\/(.*)$/);
  if (onnxRoute && onnxVariants[onnxRoute[1]]) {
    const v = onnxVariants[onnxRoute[1]];
    const hub = onnxHubFiles(onnx, { resolve: `/onnx/${onnxRoute[1]}/files/`, blake3: v.blake3, noBlake3: v.noBlake3, files: v.files, twins: v.twins });
    const rest = onnxRoute[2];
    if (rest === "index/root.json") return response.end(hub.root);
    const model = /^index\/models\/(.+)\.json$/.exec(rest);
    if (model && hub.records.has(model[1])) return response.end(hub.records.get(model[1]));
    if (rest === "index/index.sigstore.json") return response.end(await bundleFor(hub.root));
    if (rest === "uor/root.json") return response.end(uorOf(onnxRoute[1]).root);
    const shard = rest.startsWith("uor/") ? uorOf(onnxRoute[1]).shards.get(rest.slice(4)) : null;
    if (shard) return response.end(shard);
    const served = rest.startsWith("files/") && !v.offline ? v.files.get(rest.slice(6)) : null;
    if (served) {
      // A Range read answers that span, as Hugging Face does.
      const range = /bytes=(\d+)-(\d+)/.exec(request.headers.range ?? "");
      const [from, to] = range ? [Number(range[1]), Number(range[2]) + 1] : [0, served.length];
      sent.push({ variant: onnxRoute[1], path: rest.slice(6), from, to });
      if (!range) return response.end(served);
      response.statusCode = 206;
      response.setHeader("content-range", `bytes ${from}-${to - 1}/${served.length}`);
      response.setHeader("access-control-expose-headers", "content-range");
      return response.end(served.subarray(from, to));
    }
    response.statusCode = 404;
    return response.end();
  }
  // The page's pinned policy, as a deploy pinning this test's chain would carry it.
  if (path === "/app/engine/hub/policy.json") {
    response.setHeader("content-type", "application/json");
    return response.end(JSON.stringify(trust.policy));
  }
  const file = path.endsWith("/") ? `${path}index.html` : path;
  try {
    const body = await readFile(join(site, normalize(file)));
    response.setHeader("content-type", types[extname(file)] ?? "application/octet-stream");
    response.end(body);
  } catch {
    response.statusCode = 404;
    response.end();
  }
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] });
const results = {};
try {
  for (const name of Object.keys(variants)) {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    if (variants[name].staleKernel) {
      await page.route("**/precache.json", async (route) => {
        const listing = await (await route.fetch()).json();
        for (const f of listing.files) if (f.path === "./app.holo") f.blake3 = "0".repeat(64);
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(listing) });
      });
    }
    await page.goto(`${origin}/app/?repo=${fixture.repo}&path=${fixture.path}&index=/fixture/${name}/index`);
    await page.click("#load");
    page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
    try {
      await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 });
    } catch {
      // Say where it stuck, not only that it did.
      const stuck = await page.evaluate(() => ({ phase: window.__qengine?.phase, status: document.getElementById("status")?.textContent, checks: window.__qengine?.checks ?? [] }));
      results[name] = { ...stuck, timeout: true, errors, victim: victim.name };
      await page.close();
      continue;
    }
    let output = null;
    let promptStart = null;
    if (name === "good") {
      promptStart = await page.inputValue("#prompt");
      await page.fill("#prompt", "hi");
      await page.press("#prompt", "Enter");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      output = await page.evaluate(() => window.__qengine.output);
    }
    const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, checks: window.__qengine.checks }));
    results[name] = { ...state, output, promptStart, errors, victim: victim.name };
    await page.close();
  }

  // GP-19: the ONNX export, on WebGPU and, with navigator.gpu taken away, on
  // the CPU; and refused where its files are not the record's.
  results.onnx = {};
  for (const name of ["good", "cpu", "flip", "tokenizer", "q4", "noblake3"]) {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    if (onnxVariants[name].noGpu) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined }));
    await page.goto(`${origin}/app/?onnx=${onnx.repo}&index=/onnx/${name}/index${onnxVariants[name].precision ? `&precision=${onnxVariants[name].precision}` : ""}`);
    await page.click("#load");
    await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 }).catch(() => {});
    let output = null;
    if ((await page.evaluate(() => window.__qengine.phase)) === "ready") {
      await page.fill("#prompt", "hi");
      await page.press("#prompt", "Enter");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      output = await page.evaluate(() => window.__qengine.output);
      output.second = await secondTurn(page);
    }
    const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, device: window.__qengine.device ?? null, integer: window.__qengine.integer ?? false, checks: window.__qengine.checks }));
    results.onnx[name] = { ...state, output, errors };
    await page.close();
  }

  // GP-25: the one call, imported and called on a page with no page code in
  // between, and refused by the kernel for a task it does not run.
  for (const [name, task] of [["pipeline", "text-generation"], ["pipeline task", "summarization"]]) {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    await page.goto(`${origin}/app/`);
    const run = await page.evaluate(async ({ repo, task }) => {
      const { pipeline } = await import("./engine/pipeline.mjs");
      const { checklist } = await import("./engine/loader.mjs");
      const checks = checklist();
      try {
        const generate = await pipeline(task, repo, { index: "/onnx/good/index", checks });
        const out = await generate("hi", { max_new_tokens: 16 });
        return { phase: "ready", device: generate.device, dtype: generate.dtype, checks: checks.items, output: { promptIds: out.promptIds, tokens: out.tokens, text: out.text } };
      } catch (e) {
        return { phase: "failed", error: String(e.message ?? e), checks: checks.items, output: null };
      }
    }, { repo: onnx.repo, task });
    results.onnx[name] = { ...run, errors };
    await page.close();
  }

  // GP-35 and GP-36: the one call for an embedding and for a label, on the
  // module's integer encoder, with no page code in between.
  for (const [kind, task] of [["bert", "feature-extraction"], ["distil", "text-classification"]]) {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    await page.goto(`${origin}/app/`);
    const run = await page.evaluate(async ({ repo, task, kind, texts }) => {
      const { pipeline } = await import("./engine/pipeline.mjs");
      const { checklist } = await import("./engine/loader.mjs");
      const checks = checklist();
      const hex = (bits) => Array.from(new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength), (b) => b.toString(16).padStart(2, "0")).join("");
      try {
        const run = await pipeline(task, repo, { index: `/encoder/${kind}/index`, checks });
        const outputs = [];
        for (const text of texts) {
          const out = await run(text);
          outputs.push({ text, ids: Array.from(out.ids), label: out.index ?? null, name: out.label ?? null, bits: hex(out.bits) });
        }
        return { phase: "ready", device: run.device, dtype: run.dtype, checks: checks.items, outputs };
      } catch (e) {
        return { phase: "failed", error: String(e.message ?? e), checks: checks.items, outputs: [] };
      }
    }, { repo: encoders[kind].repo, task, kind, texts: ENCODER_TEXTS[kind] });
    results.encoder ??= {};
    results.encoder[kind] = { ...run, errors };
    await page.close();
  }

  // GP-21: four visits in one profile. The first downloads and keeps the
  // files; the second, with the origin serving none, reads them all from the
  // browser; the third meets a kept graph with one byte flipped, which the
  // kernel refuses, and the page forgets the files it read; the fourth, the
  // origin back, downloads them again and answers.
  {
    const context = await browser.newContext();
    const visit = async (offline) => {
      onnxVariants.kept.offline = offline;
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e.message)));
      await page.goto(`${origin}/app/?onnx=${onnx.repo}&index=/onnx/kept/index`);
      await page.click("#load");
      await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 }).catch(() => {});
      let output = null;
      if ((await page.evaluate(() => window.__qengine.phase)) === "ready") {
        await page.fill("#prompt", "hi");
        await page.press("#prompt", "Enter");
        await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
        output = await page.evaluate(() => window.__qengine.output);
        output.second = await secondTurn(page);
      }
      const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, device: window.__qengine.device ?? null, checks: window.__qengine.checks }));
      const held = await page.evaluate(async () => {
        const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-files", { create: true });
        const out = [];
        for await (const [n, h] of dir.entries()) if (h.kind === "file") out.push([n, (await h.getFile()).size]);
        return out;
      });
      await page.close();
      return { ...state, output, errors, held };
    };
    results.onnx["kept first"] = await visit(false);
    results.onnx["kept offline"] = await visit(true);
    const tamper = await context.newPage();
    await tamper.goto(`${origin}/app/`);
    await tamper.evaluate(async ({ name, bytes }) => {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle("hologram-files");
      const w = await (await dir.getFileHandle(name)).createWritable();
      await w.write(new Uint8Array(bytes));
      await w.close();
    }, (() => {
      // The graph is kept as its tensors (GP-31): one byte of its largest
      // kept tensor flipped, so the graph read back is not its BLAKE3.
      const graph = onnx.files.get(onnx.path);
      const t = apart(graph).sort((a, b) => b.length - a.length)[0];
      const bytes = Buffer.from(graph.subarray(t.offset, t.offset + t.length));
      bytes[Math.floor(bytes.length / 2)] ^= 0x01;
      return { name: t.address, bytes: [...bytes] };
    })());
    await tamper.close();
    results.onnx["kept tampered"] = await visit(true);
    results.onnx["kept again"] = await visit(false);
    await context.close();
  }

  // GP-24: a profile holding another model's file, in a browser reporting no
  // free storage. The kernel decides what is dropped and what is not kept.
  {
    const context = await browser.newContext();
    const plant = await context.newPage();
    await plant.goto(`${origin}/app/`);
    await plant.evaluate(async ({ name, size }) => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-files", { create: true });
      const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await w.write(new Uint8Array(size).fill(7));
      await w.close();
    }, { name: foreign.name, size: foreign.size });
    await plant.close();
    onnxVariants.kept.offline = false;
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    await page.addInitScript(() => { StorageManager.prototype.estimate = async () => ({ quota: 1000000, usage: 1000000 }); });
    await page.goto(`${origin}/app/?onnx=${onnx.repo}&index=/onnx/kept/index`);
    await page.click("#load");
    await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 }).catch(() => {});
    let output = null;
    if ((await page.evaluate(() => window.__qengine.phase)) === "ready") {
      await page.fill("#prompt", "hi");
      await page.press("#prompt", "Enter");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      output = await page.evaluate(() => window.__qengine.output);
      output.second = await secondTurn(page);
    }
    const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, device: window.__qengine.device ?? null, checks: window.__qengine.checks }));
    const held = await page.evaluate(async () => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-files", { create: true });
      const out = [];
      for await (const [n, h] of dir.entries()) if (h.kind === "file") out.push([n, (await h.getFile()).size]);
      return out;
    });
    await page.close();
    await context.close();
    results.onnx["kept short"] = { ...state, output, errors, held };
  }

  // Something done on the engine page of `context` (served by its worker when
  // the network is off): `run(arg)` evaluated there.
  // A page that does not open leaves the visits after it to say so.
  const onPage = async (context, run, arg) => {
    const page = await context.newPage();
    const out = await page.goto(`${origin}/app/`).then(() => page.evaluate(run, arg), (e) => ({ unopened: e.message }));
    await page.close();
    return out;
  };

  // GP-30: a first visit with the network on, then visits with it off: the
  // server answers nothing at all, so the page, its engine and kernel come
  // from its worker, the hub's index from the browser and every tensor and
  // file from the browser. Then the kept record changed by one byte; then no
  // hub index kept at all.
  {
    const context = await browser.newContext();
    results.home = {};
    results.home.online = await onnxVisit(context, { variant: "home" });
    down = true;
    await context.setOffline(true);
    results.home.offline = await onnxVisit(context, { variant: "home" });
    await onPage(context, async ({ url }) => {
      const hub = await caches.open("hologram-hub");
      const kept = await (await hub.match(url)).text();
      await hub.put(url, new Response(`${kept} `));
    }, { url: `${origin}/onnx/home/index/models/${onnx.repo}.json` });
    results.home.record = await onnxVisit(context, { variant: "home" });
    await onPage(context, () => caches.delete("hologram-hub"));
    results.home.none = await onnxVisit(context, { variant: "home" });
    await context.setOffline(false);
    down = false;
    await context.close();
  }

  // GP-31: one profile. The export's q4f16 graph; the same files under a
  // second repository whose origin answers nothing; the export's q4 graph;
  // then a kept tensor with one byte changed, and the export once more.
  {
    const context = await browser.newContext();
    results.store = {};
    results.store.first = await onnxVisit(context, { variant: "store" });
    results.store.twin = await onnxVisit(context, { variant: "store", repo: `${onnx.repo}-twin` });
    // The q4 graph under the second repository: what it lacks comes from its
    // listed mirror, its own origin answering nothing.
    results.store.q4 = await onnxVisit(context, { variant: "store", repo: `${onnx.repo}-twin`, precision: "q4" });
    const graph = onnx.files.get(onnx.path);
    const victim = apart(graph).sort((a, b) => b.length - a.length)[0];
    await onPage(context, async ({ name, at }) => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("hologram-files");
      const handle = await dir.getFileHandle(name);
      const bytes = new Uint8Array(await (await handle.getFile()).arrayBuffer());
      bytes[at] ^= 0x01;
      const w = await handle.createWritable();
      await w.write(bytes);
      await w.close();
    }, { name: victim.address, at: Math.floor(victim.length / 2) });
    results.store.tampered = await onnxVisit(context, { variant: "store" });
    results.store.again = await onnxVisit(context, { variant: "store" });
    results.store.victim = victim.address;
    await context.close();
    // With no free storage: the q4 graph after the q4f16 one. Decision 26 may
    // drop kept tensors this load does not read, never one it reads.
    {
      const short = await browser.newContext();
      await onnxVisit(short, { variant: "store" });
      const noRoom = async (page) => page.addInitScript(() => { StorageManager.prototype.estimate = async () => ({ quota: 1000000, usage: 1000000 }); });
      results.store.short = await onnxVisit(short, { variant: "store", precision: "q4", before: noRoom });
      await short.close();
    }
    for (const variant of ["unlisted", "lie"]) {
      const alone = await browser.newContext();
      results.store[variant] = await onnxVisit(alone, { variant });
      await alone.close();
    }
  }

  // GP-32: two turns of the same text, on WebGPU and on the CPU.
  results.reuse = {};
  for (const [name, noGpu] of [["webgpu", false], ["cpu", true]]) {
    const context = await browser.newContext();
    const before = async (page) => { if (noGpu) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined })); };
    results.reuse[name] = await onnxVisit(context, { variant: "again", texts: ["hi", "hi"], before });
    await context.close();
  }

  // A cache left by an earlier deploy (GP-07): one file that is not this
  // deploy's, and one that is. The page must remove the first, keep the
  // second, reload once, and then import the engine.
  const earlier = async (context) => {
    const seed = await context.newPage();
    await seed.goto(`${origin}/precache.json`);
    await seed.evaluate(async (origin) => {
      const cache = await caches.open("prism-earlier");
      await cache.put(`${origin}/app/engine/gpu.mjs`, new Response("// an earlier deploy's engine"));
      await cache.put(`${origin}/app/app.css`, await fetch(`${origin}/app/app.css`));
    }, origin);
    await seed.close();
  };
  const held = (page) => page.evaluate(async (origin) => {
    const cache = await caches.open("prism-earlier");
    return { engine: Boolean(await cache.match(`${origin}/app/engine/gpu.mjs`)), css: Boolean(await cache.match(`${origin}/app/app.css`)) };
  }, origin);
  {
    const context = await browser.newContext();
    await earlier(context);
    const page = await context.newPage();
    let loads = 0;
    page.on("framenavigated", (frame) => { if (frame === page.mainFrame()) loads += 1; });
    await page.goto(`${origin}/app/`);
    const imported = await page.waitForFunction(() => window.__qengine && window.__qpage, null, { timeout: 30000 }).then(() => true, () => false);
    results.earlier = { imported, loads, page: await page.evaluate(() => window.__qpage ?? null), status: await page.textContent("#status"), held: await held(page) };
    await context.close();
  }
  // The same cache when one reload has already been spent: the page must stop
  // and not import the engine, or it would reload for ever.
  {
    const context = await browser.newContext();
    await earlier(context);
    const page = await context.newPage();
    await page.addInitScript(() => sessionStorage.setItem("prism-stale-reload", "1"));
    let loads = 0;
    page.on("framenavigated", (frame) => { if (frame === page.mainFrame()) loads += 1; });
    await page.goto(`${origin}/app/`);
    await page.waitForFunction(() => window.__qpage, null, { timeout: 30000 }).catch(() => {});
    // Long enough for an engine imported by mistake to show itself.
    await page.waitForFunction(() => window.__qengine, null, { timeout: 3000 }).catch(() => {});
    results.again = { loads, imported: await page.evaluate(() => Boolean(window.__qengine)), page: await page.evaluate(() => window.__qpage ?? null), status: await page.textContent("#status"), disabled: await page.isDisabled("#load") };
    await context.close();
  }
} finally {
  await browser.close();
  server.close();
}

// The answer the page gave, against the reference run over the same prompt ids.
const good = results.good;
if (good.output) {
  const reference = referenceModel(fixture.model);
  let last;
  for (const id of good.output.promptIds) last = reference.step(id);
  const expected = [];
  while (expected.length < 16) {
    expected.push(last.token);
    last = reference.step(last.token);
  }
  good.expected = expected;
}
// The ONNX answers against the reference over the same prompt ids.
for (const run of Object.values(results.onnx)) {
  if (!run.output) continue;
  const reference = referenceModel(onnx.model);
  let last;
  for (const id of run.output.promptIds) last = reference.step(id);
  run.expected = [];
  while (run.expected.length < 16) {
    run.expected.push(last.token);
    last = reference.step(last.token);
  }
  // The second turn against a reference given its whole prompt.
  if (!run.output.second) continue;
  const again = referenceModel(onnx.model);
  for (const id of run.output.second.promptIds) last = again.step(id);
  run.expectedSecond = [];
  while (run.expectedSecond.length < 16) {
    run.expectedSecond.push(last.token);
    last = again.step(last.token);
  }
}
console.log(JSON.stringify(results));

// The verdicts. Each says where the page must have stopped, not only that it
// did: a page that refuses for the wrong reason fails too.
const failures = [];
const fail = (why) => failures.push(why);
const stoppedAt = (r) => r.checks.filter((c) => c.verdict === "fail")[0];

if (good.phase !== "ready") fail(`good: the page did not reach ready (${good.phase}: ${good.error ?? (good.timeout ? "timed out" : "")})`);
if (good.checks.some((c) => c.verdict === "fail")) fail(`good: a check failed: ${stoppedAt(good)?.subject}`);
const alias = good.checks.find((c) => c.step === "alias");
if (!alias || alias.verdict !== "pass") fail("good: the tied output head was not confirmed as the embedding's bytes");
if (!good.checks.some((c) => c.step === "root" && c.verdict === "pass")) fail("good: the whole-file root was not checked");
if (!good.output) fail("good: no answer");
else {
  const got = good.output.tokens.slice(0, 16);
  if (got.length < 16 || got.some((t, i) => t !== good.expected[i])) fail(`good: the page's tokens ${got} are not the reference's ${good.expected}`);
}
const listed = good.checks.find((c) => c.subject === "kernel is the one this deploy lists");
if (listed?.verdict !== "pass") fail(`good: the kernel was not compared with the deploy's listing (${listed?.verdict ?? "no check"})`);
if (good.promptStart !== "") fail(`good: the prompt box started with text, so typing would join it: ${JSON.stringify(good.promptStart)}`);
if (good.errors.length) fail(`good: page errors: ${good.errors.join(" | ")}`);

const flip = results.flip;
const flipStop = stoppedAt(flip);
if (flip.phase !== "failed" || flipStop?.step !== "admit" || flipStop?.subject !== `${victim.name} label and length`)
  fail(`flip: expected to stop at the admission of ${victim.name}, stopped at ${flipStop?.step}: ${flipStop?.subject} (${flip.phase})`);
if (flip.checks.some((c) => c.subject === "download → compile")) fail("flip: the load went on to compile");

const root = results.root;
const rootStop = stoppedAt(root);
if (root.phase !== "failed" || rootStop?.step !== "root") fail(`root: expected to stop at the whole-file root, stopped at ${rootStop?.step}: ${rootStop?.subject} (${root.phase})`);

const signature = (r) => r.checks.find((c) => c.subject === "index root signature");
if (signature(good)?.verdict !== "pass") fail("good: the index root's signature was not checked");
const foreignHub = results.foreign;
const foreignStop = stoppedAt(foreignHub);
if (foreignHub.phase !== "failed" || foreignStop?.subject !== "index root signature" || !/^RootUnverified: .*chain/.test(foreignStop?.detail ?? ""))
  fail(`foreign: expected to stop at the index root signature as RootUnverified, chain failed; stopped at ${foreignStop?.step}: ${foreignStop?.subject} (${foreignStop?.detail})`);
if (foreignHub.checks.some((c) => c.subject === "hub record matches the index root")) fail("foreign: the page read a record under a root it could not prove");

const stale = results.stale;
const staleStop = stoppedAt(stale);
if (stale.phase !== "failed" || staleStop?.subject !== "kernel is the one this deploy lists" || !/^stale kernel/.test(stale.error ?? ""))
  fail(`stale: expected to stop with "stale kernel" at the deploy comparison, stopped at ${staleStop?.step}: ${staleStop?.subject} (${stale.phase}: ${stale.error})`);
if (stale.checks.some((c) => !["kernel", "stop"].includes(c.step))) fail(`stale: the page went on past the kernel: ${stale.checks.map((c) => c.subject).join(", ")}`);

// GP-07: an earlier deploy's file is removed, this deploy's kept, one reload.
const earlier = results.earlier;
if (!earlier.imported) fail(`earlier: the engine was not imported after the stale file was removed (${earlier.status})`);
if (earlier.loads !== 2) fail(`earlier: the page loaded ${earlier.loads} times, expected one reload`);
if (earlier.page?.removed !== 1 || earlier.page?.stale.length !== 0) fail(`earlier: expected one file removed and none stale after the reload, saw ${JSON.stringify(earlier.page)}`);
if (earlier.held.engine) fail("earlier: the earlier deploy's engine file is still in the cache");
if (!earlier.held.css) fail("earlier: a cached file that is this deploy's was removed");
if (!/^Updated: files from an earlier deploy were replaced \(1\)\.$/.test(earlier.status ?? "")) fail(`earlier: the page did not say what it replaced: ${earlier.status}`);
const again = results.again;
if (again.imported) fail("again: the engine was imported with a stale file still served after a reload");
if (again.loads !== 1) fail(`again: the page loaded ${again.loads} times, expected no further reload`);
if (!/^Stopped: stale page/.test(again.status ?? "") || !again.disabled) fail(`again: expected "Stopped: stale page" and no Load button, saw ${again.status} (disabled: ${again.disabled})`);

// GP-19: the ONNX export.
for (const [name, device] of [["good", "webgpu"], ["cpu", "cpu"], ["kept first", "webgpu"], ["kept offline", "webgpu"], ["kept again", "webgpu"], ["kept short", "webgpu"]]) {
  const run = results.onnx[name];
  if (run.phase !== "ready") fail(`onnx ${name}: the page did not reach ready (${run.phase}: ${run.error})`);
  if (run.checks.some((c) => c.verdict === "fail")) fail(`onnx ${name}: a check failed: ${stoppedAt(run)?.subject}`);
  if (run.device !== device) fail(`onnx ${name}: ran on ${run.device}, expected ${device}`);
  // GP-28: the CPU device runs the integer engine.
  if (name === "cpu" && run.integer !== true) fail(`onnx ${name}: the CPU device did not run the integer engine`);
  for (const step of ["record", "file", "template", "import", "roles", "stop"]) {
    if (!run.checks.some((c) => c.step === step && c.verdict === "pass")) fail(`onnx ${name}: no passing ${step} check`);
  }
  // GP-22: the precision, the file and the device are the kernel's answers.
  const said = (step) => /^kernel answered "(.*)"$/.exec(run.checks.find((c) => c.step === step && c.verdict === "pass")?.detail ?? "")?.[1];
  if (said("precision") !== "q4f16") fail(`onnx ${name}: the precision was not the kernel's q4f16: ${said("precision")}`);
  if (said("source") !== onnx.path) fail(`onnx ${name}: the graph's file was not the kernel's ${onnx.path}: ${said("source")}`);
  if (said("device") !== device) fail(`onnx ${name}: the device was not the kernel's ${device}: ${said("device")}`);
  const imported = run.checks.find((c) => c.step === "import");
  if (!/^\d+ tensors named by their BLAKE3, [1-9]\d* questions to the kernel/.test(imported?.detail ?? "")) fail(`onnx ${name}: the import did not ask the kernel: ${imported?.detail}`);
  if (!run.output) fail(`onnx ${name}: no answer`);
  else {
    const got = run.output.tokens.slice(0, 16);
    if (got.length < 16 || got.some((t, i) => t !== run.expected[i])) fail(`onnx ${name}: the page's tokens ${got} are not the reference's ${run.expected}`);
  }
  if (run.errors.length) fail(`onnx ${name}: page errors: ${run.errors.join(" | ")}`);
  // GP-23: the second turn reuses the head the kernel admits, and only that.
  const first = run.output, second = run.output?.second;
  if (first?.reuse) fail(`onnx ${name}: the first turn asked to reuse a prefix, with nothing computed before it`);
  if (!second?.reuse) fail(`onnx ${name}: the second turn did not ask the kernel to reuse the head`);
  else {
    const { length, asked, held, verdict } = second.reuse;
    const ids = Buffer.alloc(4 * length);
    second.promptIds.slice(0, length).forEach((id, i) => ids.writeUInt32LE(id, 4 * i));
    if (asked !== blake3Hex(ids)) fail(`onnx ${name}: the label asked is not the BLAKE3 of the prompt's first ${length} ids`);
    if (verdict !== "reuse" || held !== asked || !(length > 0 && length < second.promptIds.length)) fail(`onnx ${name}: the second turn was not answered reuse of a head shorter than its prompt: ${verdict}, ${length} of ${second.promptIds.length}`);
    if (second.computed !== second.promptIds.length - length) fail(`onnx ${name}: the second turn computed ${second.computed} prompt positions, not the ${second.promptIds.length - length} after the head`);
    const reused = run.checks.filter((c) => c.step === "reuse");
    if (reused.length !== 1 || reused[0].verdict !== "pass" || reused[0].detail !== 'kernel answered "reuse"') fail(`onnx ${name}: expected one passing reuse check, saw ${JSON.stringify(reused)}`);
    const got = second.tokens.slice(0, 16);
    if (got.length < 16 || got.some((t, i) => t !== run.expectedSecond[i])) fail(`onnx ${name}: the second turn's tokens ${got} are not the reference's ${run.expectedSecond}`);
  }
}
{
  const run = results.onnx.flip;
  if (run.phase !== "failed" || !/FileRefused/.test(run.error ?? "")) fail(`onnx flip: expected the kernel to refuse the graph file, saw ${run.phase}: ${run.error}`);
  if (run.checks.some((c) => c.step === "import" || c.step === "roles")) fail("onnx flip: the load went on past the import");
  const run2 = results.onnx.tokenizer;
  const stop = stoppedAt(run2);
  if (run2.phase !== "failed" || stop?.subject !== "tokenizer.json is the record's") fail(`onnx tokenizer: expected to stop at tokenizer.json, stopped at ${stop?.subject} (${run2.phase}: ${run2.error})`);
  // GP-26: every file checked by its BLAKE3, and a record naming a file no
  // BLAKE3 refused by name, nothing checked by SHA-256 or SHA-1.
  const config = results.onnx.good.checks.find((c) => c.subject === "config.json is the record's");
  if (config?.detail !== `${onnx.files.get("config.json").length} bytes, blake3 ${onnx.b3(onnx.files.get("config.json"))}`) fail(`onnx good: config.json was not checked by its BLAKE3: ${config?.detail}`);
  if (results.onnx.good.checks.some((c) => /sha256|sha1/i.test(c.detail ?? "") && c.step === "file")) fail("onnx good: a file was checked by SHA-256 or SHA-1");
  const run3 = results.onnx.noblake3;
  if (run3.phase !== "failed" || !/the record names no BLAKE3 for config\.json/.test(run3.error ?? "")) fail(`onnx noblake3: expected the record refused for naming config.json no BLAKE3, saw ${run3.phase}: ${run3.error}`);
  if (run3.checks.some((c) => c.step === "import")) fail("onnx noblake3: the load went on past the record");
  const run4 = results.onnx.q4;
  const stop4 = stoppedAt(run4);
  if (run4.phase !== "failed" || stop4?.step !== "precision" || stop4?.detail !== 'kernel answered "none fits"') fail(`onnx q4: expected the kernel to refuse a precision the record does not offer, stopped at ${stop4?.step} (${run4.phase}: ${run4.error})`);
  if (run4.checks.some((c) => ["file", "kept", "import"].includes(c.step))) fail("onnx q4: the load went on past the precision");
}

// GP-25: the one call, and a task the kernel does not run.
{
  const run = results.onnx.pipeline;
  if (run.phase !== "ready") fail(`pipeline: the call did not load (${run.error})`);
  else {
    for (const step of ["kernel", "record", "source", "precision", "import", "roles", "device", "stop"]) {
      if (!run.checks.some((c) => c.step === step && c.verdict === "pass")) fail(`pipeline: no passing ${step} check`);
    }
    if (run.checks.some((c) => c.verdict === "fail")) fail(`pipeline: a check failed: ${stoppedAt(run)?.subject}`);
    const got = run.output.tokens.slice(0, 16);
    if (got.length < 16 || got.some((t, i) => t !== run.expected[i])) fail(`pipeline: the call's tokens ${got} are not the reference's ${run.expected}`);
    if (run.dtype !== "q4f16" || !["webgpu", "cpu"].includes(run.device)) fail(`pipeline: chose ${run.dtype} on ${run.device}`);
  }
  if (run.errors.length) fail(`pipeline: page errors: ${run.errors.join(" | ")}`);
  const task = results.onnx["pipeline task"];
  const stop = stoppedAt(task);
  if (task.phase !== "failed" || stop?.step !== "source" || stop?.detail !== 'kernel answered "unsupported task"') fail(`pipeline task: expected the kernel to refuse summarization, saw ${stop?.step} (${task.phase}: ${task.error})`);
  if (task.checks.some((c) => ["file", "kept", "import"].includes(c.step))) fail("pipeline task: the call went on past the task");
}

// GP-35 and GP-36: the page's encoder answers equal hologram-ai-cpu-encode's
// bit for bit (encoder-expected.json, which EN-64 recomputes), on the CPU
// device decision 22 chose, with every operator asked of the encoder style.
for (const kind of ["bert", "distil"]) {
  const run = results.encoder[kind];
  const what = kind === "bert" ? "encoder embedding" : "encoder label";
  if (run.phase !== "ready") fail(`${what}: ${run.error}`);
  if (run.device !== "cpu" || run.dtype !== "q4f16") fail(`${what}: chose ${run.dtype} on ${run.device}`);
  if (!run.checks.some((c) => c.step === "engine" && c.verdict === "pass")) fail(`${what}: no check names the module's integer encoder`);
  const want = encoderExpected[kind];
  if (run.outputs.length !== want.length) fail(`${what}: ${run.outputs.length} answers for ${want.length} sentences`);
  run.outputs.forEach((got, i) => {
    const w = want[i];
    if (JSON.stringify(got.ids) !== JSON.stringify(w.ids)) fail(`${what}: "${got.text}" is ids ${got.ids}, not ${w.ids}`);
    if (got.bits !== w.bits) fail(`${what}: "${got.text}": the page's bits are not hologram-ai-cpu-encode's`);
    if (kind === "distil" && (got.label !== w.label || got.name !== ["NEGATIVE", "POSITIVE"][w.label])) fail(`${what}: "${got.text}" is ${got.name} (${got.label}), not label ${w.label}`);
  });
  if (run.errors.length) fail(`${what}: page errors: ${run.errors.join(" | ")}`);
}

// GP-21: the files kept in the browser, read back checked as a download is.
{
  const count = onnx.files.size;
  const kept = (name) => results.onnx[name].checks.find((c) => c.step === "kept");
  const said = (name) => /^(\d+) read here \([\d.]+ MB\), (\d+) downloaded and kept/.exec(kept(name)?.detail ?? "");
  const first = said("kept first");
  if (!first || first[1] !== "0" || first[2] !== String(count)) fail(`kept first: expected 0 read here and ${count} downloaded, saw ${kept("kept first")?.detail}`);
  // Each small file under its BLAKE3; the graph as its tensors, each under
  // its address, and its rest under its own name (GP-31).
  const keptFirst = results.onnx["kept first"].held.filter(([, size]) => size > 0).length;
  const keptWant = count - 1 + apart(onnx.files.get(onnx.path)).length + 1;
  if (keptFirst !== keptWant) fail(`kept first: expected ${count - 1} files, the graph's tensors and its rest kept (${keptWant}), saw ${keptFirst}`);
  const named = /each named by its BLAKE3: (.*)$/.exec(kept("kept first")?.detail ?? "")?.[1]?.split(", ") ?? [];
  if (named.length !== count || named.some((n) => !/ blake3:[0-9a-f]{64}$/.test(n))) fail(`kept first: expected every kept file named by its BLAKE3, saw ${named.join(" | ")}`);
  const offline = said("kept offline");
  if (!offline || offline[1] !== String(count) || offline[2] !== "0") fail(`kept offline: expected ${count} read here and none downloaded, saw ${kept("kept offline")?.detail}`);
  const run = results.onnx["kept tampered"];
  if (run.phase !== "failed" || !/FileRefused/.test(run.error ?? "")) fail(`kept tampered: expected the kept graph refused as not its BLAKE3, saw ${run.phase}: ${run.error}`);
  if (run.held.some(([, size]) => size > 0)) fail(`kept tampered: the files this load read were not forgotten: ${JSON.stringify(run.held)}`);
  const again = said("kept again");
  if (!again || again[1] !== "0" || again[2] !== String(count)) fail(`kept again: expected every file downloaded again, saw ${kept("kept again")?.detail}`);

  // GP-24: the other model's file dropped on evict, one file kept, the rest not.
  const short = results.onnx["kept short"];
  const room = /, (\d+) not kept for want of room, (\d+) dropped for room \((\d+) bytes\)/.exec(kept("kept short")?.detail ?? "");
  if (!room || room[1] !== String(count - 1) || room[2] !== "1" || room[3] !== String(foreign.size)) fail(`kept short: expected the other model's file dropped and ${count - 1} files not kept, saw ${kept("kept short")?.detail}`);
  if (short.held.some(([n]) => n === foreign.name)) fail("kept short: the other model's file was not dropped");
  const keptShort = short.held.filter(([n, size]) => /^[0-9a-f]{64}$/.test(n) && size > 0).length;
  if (keptShort !== 1) fail(`kept short: expected one file kept, saw ${JSON.stringify(short.held)}`);
}

// The reference's first sixteen picks after a prompt.
const picks = (model, promptIds) => {
  const reference = referenceModel(model);
  let last;
  for (const id of promptIds) last = reference.step(id);
  const out = [];
  while (out.length < 16) {
    out.push(last.token);
    last = reference.step(last.token);
  }
  return out;
};
const answered = (name, run, model = onnx.model) => {
  if (run.phase !== "ready") return fail(`${name}: the page did not reach ready (${run.phase}: ${run.error})`);
  if (run.checks.some((c) => c.verdict === "fail")) fail(`${name}: a check failed: ${stoppedAt(run)?.subject}: ${stoppedAt(run)?.detail}`);
  if (run.errors.length) fail(`${name}: page errors: ${run.errors.join(" | ")}`);
  for (const [i, turn] of run.turns.entries()) {
    const want = picks(model, turn.promptIds);
    const got = turn.tokens.slice(0, 16);
    if (got.length < 16 || got.some((t, k) => t !== want[k])) fail(`${name}: turn ${i + 1}'s tokens ${got} are not the reference's ${want}`);
  }
};
const detail = (run, step) => run.checks.find((c) => c.step === step && c.verdict === "pass")?.detail ?? "";
const storeSaid = (run) => /^(\d+) tensors read here \((\d+) bytes\), (\d+) downloaded \((\d+) bytes\), (\d+) newly kept \((\d+) bytes\), (\d+) bytes of the rest downloaded, in (\d+) reads$/.exec(detail(run, "store"))?.slice(1).map(Number);

// GP-30: a return visit with the network off.
{
  const { online, offline, record, none } = results.home;
  answered("home online", online);
  if (!online.worker) fail("home online: the engine page's worker was not active after the first visit");
  answered("home offline", offline);
  if (offline.device !== "webgpu") fail(`home offline: ran on ${offline.device}, expected webgpu`);
  for (const subject of ["index root signature", "hub record matches the index root", "hub index read from this browser, checked again"]) {
    if (!offline.checks.some((c) => c.subject === subject && c.verdict === "pass")) fail(`home offline: no passing check "${subject}"`);
  }
  const kernel = offline.checks.find((c) => c.subject === "kernel is the one this deploy lists");
  if (kernel?.verdict !== "not checked" || !/^the network is off/.test(kernel.detail)) fail(`home offline: the deploy comparison should say the network is off, saw ${kernel?.verdict}: ${kernel?.detail}`);
  const kept = /^(\d+) read here \([\d.]+ MB\), (\d+) downloaded and kept/.exec(detail(offline, "kept"));
  if (!kept || kept[1] !== String(onnx.files.size) || kept[2] !== "0") fail(`home offline: expected every file read here and none downloaded, saw ${detail(offline, "kept")}`);
  const store = storeSaid(offline);
  if (!store || store[0] !== apart(onnx.files.get(onnx.path)).length || store[2] !== 0 || store[6] !== 0 || store[7] !== 0) fail(`home offline: expected every tensor read here and no read of the network, saw ${detail(offline, "store")}`);
  const ready = offline.checks.find((c) => /decision 28/.test(c.subject));
  if (ready?.verdict !== "pass" || !/^kernel answered "ready"/.test(ready.detail)) fail(`home offline: decision 28 did not answer ready on the kept tensors: ${ready?.detail}`);
  const recordStop = stoppedAt(record);
  if (record.phase !== "failed" || recordStop?.subject !== "hub record matches the index root" || !/^RecordMismatch/.test(recordStop?.detail ?? "")) fail(`home record: expected a changed kept record refused as RecordMismatch, stopped at ${recordStop?.subject}: ${recordStop?.detail} (${record.phase})`);
  const noneStop = stoppedAt(none);
  if (none.phase !== "failed" || noneStop?.subject !== "index root signature" || !/^RootUnreachable/.test(noneStop?.detail ?? "")) fail(`home none: expected the page stopped at the index root, unreachable, stopped at ${noneStop?.subject}: ${noneStop?.detail} (${none.phase})`);
  for (const [name, run] of [["home record", record], ["home none", none]]) if (run.checks.some((c) => ["file", "import"].includes(c.step))) fail(`${name}: the load went on past the record`);
}

// GP-31: tensors kept once by address, only what is not held downloaded.
{
  const { first, twin, q4, tampered, again, unlisted, lie, victim } = results.store;
  const graph = onnx.files.get(onnx.path);
  const tensors16 = apart(graph);
  const tensors4 = apart(onnx.q4.bytes);
  const held16 = new Set(tensors16.map((t) => t.address));
  const shared = tensors4.filter((t) => held16.has(t.address));
  // Whether any range the server sent overlaps a tensor of `list`.
  const overlaps = (ranges, list) => ranges.some((r) => list.some((t) => r.from < t.offset + t.length && t.offset < r.to));
  answered("store first", first);
  const f = storeSaid(first);
  if (!f || f[0] !== 0 || f[2] !== tensors16.length || f[4] !== tensors16.length) fail(`store first: expected every tensor downloaded and kept, saw ${detail(first, "store")}`);
  answered("store twin", twin);
  const t = storeSaid(twin);
  if (!t || t[0] !== tensors16.length || t[2] !== 0 || t[3] !== 0) fail(`store twin: expected every tensor read here and none downloaded, saw ${detail(twin, "store")}`);
  const twinSent = twin.sent.filter((r) => r.path === onnx.path);
  if (overlaps(twinSent, tensors16)) fail(`store twin: a tensor already held was downloaded again: ${JSON.stringify(twinSent)}`);
  // The same graph file under another repository: its rest is held under the
  // file's BLAKE3 too, so nothing of it is downloaded.
  if (twinSent.length || t?.[6] !== 0) fail(`store twin: expected nothing of the graph downloaded, its tensors and rest held; saw ${JSON.stringify(twinSent)}, ${detail(twin, "store")}`);
  if (twin.sent.some((r) => r.path !== onnx.path)) fail(`store twin: a small file already held was downloaded: ${JSON.stringify(twin.sent)}`);
  answered("store q4", q4, onnx.q4.model);
  if (!/kernel answered "q4"/.test(q4.checks.find((c) => c.step === "precision")?.detail ?? "")) fail("store q4: the kernel did not choose the q4 graph");
  const q = storeSaid(q4);
  const fresh4 = tensors4.filter((x) => !held16.has(x.address));
  if (!q || q[0] !== shared.length || q[2] !== fresh4.length || q[3] !== fresh4.reduce((n, x) => n + x.length, 0)) fail(`store q4: expected the ${shared.length} tensors the q4f16 graph shares read here and only the ${fresh4.length} others downloaded, saw ${detail(q4, "store")}`);
  if (overlaps(q4.sent.filter((r) => r.path === onnx.q4.path), shared)) fail("store q4: a tensor the q4f16 graph shares was downloaded again");
  if (!shared.length) fail("store q4: the fixture's two precisions share no tensor, so nothing is tested");
  if (tampered.phase !== "failed" || !/FileRefused/.test(tampered.error ?? "")) fail(`store tampered: expected the graph read back refused as not its BLAKE3, saw ${tampered.phase}: ${tampered.error}`);
  if (tampered.held.some(([n]) => n === victim)) fail("store tampered: the changed tensor is still kept");
  answered("store again", again);
  const a = storeSaid(again);
  if (!a || a[2] < 1) fail(`store again: expected the forgotten tensors downloaded again, saw ${detail(again, "store")}`);
  // Short of room: what the q4 load reads stays pinned; it answers.
  const { short } = results.store;
  answered("store short", short, onnx.q4.model);
  const dropped = /, (\d+) dropped for room/.exec(detail(short, "kept"))?.[1];
  if (!(Number(dropped) > 0)) fail(`store short: expected kept tensors the load does not read dropped for room, saw ${detail(short, "kept")}`);
  const heldNames = new Set(short.held.map(([n]) => n));
  for (const x of shared) if (!heldNames.has(x.address)) fail(`store short: ${x.name}, read by this load, was dropped for room`);
  if (unlisted.phase !== "failed" || !new RegExp(`^no tensor manifest: the hub's UOR index lists no tensors for ${onnx.repo}`).test(unlisted.error ?? "")) fail(`store unlisted: expected the model refused by name for want of a manifest, saw ${unlisted.phase}: ${unlisted.error}`);
  if (unlisted.checks.some((c) => ["file", "kept", "import"].includes(c.step)) || unlisted.sent.length) fail("store unlisted: the load read a file of a model the index does not list");
  if (lie.phase !== "failed" || !/decision 28 answered "refused"/.test(lie.error ?? "")) fail(`store lie: expected decision 28 to refuse an index whose address is not the bytes', saw ${lie.phase}: ${lie.error}`);
  if (lie.held.length) fail(`store lie: bytes were kept from a load decision 28 refused: ${JSON.stringify(lie.held)}`);
}

// GP-32: the same text again reuses the head and gives the same tokens.
for (const [name, run] of Object.entries(results.reuse)) {
  answered(`reuse ${name}`, run);
  if (run.device !== name) fail(`reuse ${name}: ran on ${run.device}`);
  const [one, two] = run.turns;
  if (!one || !two) { fail(`reuse ${name}: expected two turns`); continue; }
  if (one.reuse) fail(`reuse ${name}: the first turn asked to reuse a prefix`);
  if (two.reuse?.verdict !== "reuse") fail(`reuse ${name}: the second turn was not answered reuse: ${JSON.stringify(two.reuse)}`);
  else if (two.computed !== two.promptIds.length - two.reuse.length) fail(`reuse ${name}: the second turn computed ${two.computed} positions, not the ${two.promptIds.length - two.reuse.length} after the head`);
  if (JSON.stringify(two.promptIds) !== JSON.stringify(one.promptIds)) fail(`reuse ${name}: the two turns' prompts differ`);
  if (JSON.stringify(two.tokens) !== JSON.stringify(one.tokens)) fail(`reuse ${name}: reusing the head changed the tokens: ${one.tokens} then ${two.tokens}`);
  if (!(one.stats?.timeToFirstToken > 0 && two.stats?.timeToFirstToken > 0)) fail(`reuse ${name}: a turn reported no time to first token`);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.error("journey: the page loads through the kernel, answers as the reference does, and stops where tampering or a stale kernel is, and replaces an earlier deploy's files; and reads an ONNX export from its own files on WebGPU and on the CPU, refusing one whose files are not the record's");
