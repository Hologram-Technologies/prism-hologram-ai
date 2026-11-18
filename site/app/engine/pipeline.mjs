// One call loads and runs a model from the hub, the shape of transformers.js's
// pipeline():
//
//   import { pipeline } from "./engine/pipeline.mjs";
//   const generate = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", { device: "auto", dtype: "auto" });
//   const { text, tokens } = await generate("What is the capital of France?", { max_new_tokens: 64 });
//
// Everything the page does to load an ONNX export happens here, each step put
// to the verified kernel: the kernel holo checked in this browser, the hub's
// record checked against its root, what each file is called (decision 24), the
// task (decision 24 again), which precision fits (21), the files kept in this
// browser by BLAKE3 and dropped only on decision 26, every span, operator, file
// and 4-bit weight admitted as the bytes arrive, and the device (22). A turn is
// framed by the model's own chat template, its budget is decision 9, it stops
// on the model's own ids, and the template's head is reused only on decision
// 27. The page is one caller of this module; nothing here reads the page.

import { createIntegerEngine } from "./integer.mjs";
import { createEngine } from "./gpu.mjs";
import { asker, checklist, hubRecord, openModule } from "./loader.mjs";
import { strftime } from "./jinja.mjs";
import { fetched, loadOnnx } from "./onnx-load.mjs";
import { keepFiles } from "./files.mjs";

// The precisions this engine runs, in the kernel's code for decision 24.
export const PRECISIONS = { q4f16: 0, q4: 1 };
// The tasks, in decision 24's code. The kernel answers unsupported task for
// any but text generation; the module refuses nothing itself.
export const TASKS = { "text-generation": 0, "feature-extraction": 1, "text-classification": 2 };
// What the kernel's choice of precision and device is weighed against. No
// browser reports how much memory a GPU has, so its budget is a stated 4 GiB;
// the CPU engine, single-threaded on a static host, takes models up to 1 GiB.
export const GPU_BUDGET = 4 * 2 ** 30;
export const CPU_CEILING = 2 ** 30;
export const CAPACITY = 2048;
const INDEX = "https://hologram-technologies.github.io/hub/data/index";

// A number as `width` big-endian bytes, and byte arrays joined: the kernel's wire.
export const be = (value, width) => {
  const out = new Uint8Array(width);
  let v = BigInt(value);
  for (let i = width - 1; i >= 0; i--) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
};
export const join = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};

// The kernel holo, verified in this browser.
export async function verifiedKernel({ holo = new URL("../../app.holo", import.meta.url), checks = checklist() } = {}) {
  const { loadArchive } = await import("../../vendor/prism-code/src/index.mjs");
  const response = await fetch(holo);
  if (!response.ok) throw new Error(`${holo}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const kernel = await loadArchive(bytes);
  const verified = kernel.identities.verified;
  checks.add("kernel", "kernel holo verified in this browser", verified ? "pass" : "fail", kernel.identities.archiveFingerprint);
  if (!verified) throw new Error("the kernel did not verify");
  return { kernel, bytes };
}

// A WebGPU device with the adapter's largest bindings, or null where there is none.
export async function openGpu() {
  const adapter = globalThis.navigator?.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) : null;
  if (!adapter) return null;
  return adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
}

// Load `model` for `task` and answer a function that runs one turn.
//   device: "auto" (the kernel decides), "webgpu" or "cpu"
//   dtype:  "auto" (the kernel decides among those the record offers), "q4f16" or "q4"
//   index:  the hub index the record is read from
//   kernel, ask, gpu: a verified kernel, its asker and a GPUDevice, when the caller holds them already
//   checks: where each verdict is written; onStatus, onProgress: what the caller shows
export async function pipeline(task, model, options = {}) {
  const {
    device: want = "auto",
    dtype = "auto",
    index = INDEX,
    checks = checklist(),
    onStatus = () => {},
    onProgress = () => {},
  } = options;
  const kernel = options.kernel ?? (await verifiedKernel({ checks })).kernel;
  const ask = options.ask ?? asker(kernel, checks);
  const gpu = want === "cpu" ? null : options.gpu !== undefined ? options.gpu : await openGpu();
  if (want === "webgpu" && !gpu) throw new Error("device webgpu: this browser offers no WebGPU adapter");
  const { compileGuest, resolveArchive, runGuest } = await import("../../vendor/prism-code/src/index.mjs");
  const guest = await compileGuest(resolveArchive(kernel.archive).guest.content);
  const answer = (request) => runGuest(guest, request, kernel.limits);

  onStatus("Reading the model's record…");
  const record = await hubRecord({ indexBase: index, repo: model, checks });
  // What each precision's graph is called is the kernel's decision 24, for this
  // task, and which precision is read its decision 21, weighed on what the
  // record offers.
  const named = dtype === "auto" ? Object.keys(PRECISIONS) : [dtype];
  if (named.some((n) => !(n in PRECISIONS))) throw new Error(`dtype ${dtype}: this engine reads q4f16 and q4 exports`);
  const code = TASKS[task] ?? 255;
  const sizes = {};
  for (const name of named) {
    const graphPath = await ask("source", `the ${name} graph's file for ${task}`, new Uint8Array([24, code, PRECISIONS[name], 0, 0]), (a) => a.startsWith("onnx/"));
    const parts = record.files.filter((f) => f.path === graphPath || f.path.startsWith(`${graphPath}_data`));
    if (parts.some((f) => f.path === graphPath)) sizes[name] = { path: graphPath, bytes: parts.reduce((n, f) => n + f.size, 0), data: parts.length - 1 };
  }
  const offer = (name) => new Uint8Array([sizes[name] ? 1 : 0]);
  const amount = (name) => be(sizes[name]?.bytes ?? 0, 8);
  const chosen = await ask("precision", "precision to read", join(new Uint8Array([21]), offer("q4f16"), offer("q4"), new Uint8Array([0]), be(gpu ? GPU_BUDGET : CPU_CEILING, 8), amount("q4f16"), amount("q4"), be(0, 8)), (a) => a in sizes);
  const path = sizes[chosen].path;
  // Each data file the record lists beside the graph is one the kernel names.
  const folder = path.slice(0, path.lastIndexOf("/") + 1);
  for (let part = 1; part <= sizes[chosen].data; part++) {
    await ask("source", `the ${chosen} data file ${part}`, new Uint8Array([24, code, PRECISIONS[chosen], sizes[chosen].data, part]), (a) => record.files.some((f) => f.path === a && a.startsWith(folder)));
  }
  const base = record.origin?.resolve;
  if (!base) throw new Error("the record names no origin");
  const graph = record.files.find((f) => f.path === path);
  if (!graph) throw new Error(`the record names no ${path}`);

  const module = await openModule(new URL("./engine.wasm", import.meta.url));
  // The files kept from an earlier visit are read from this browser and
  // checked exactly as a download is; where the browser keeps none, every
  // file comes from the origin.
  let files = null;
  try {
    files = await keepFiles({ record, open: fetched(base), module, kernel: async (request) => new TextDecoder().decode(await kernel.dispatch(request)) });
  } catch (e) {
    checks.add("kept", "model files kept in this browser", "not checked", `none kept: ${e.message}`);
  }
  onStatus("Downloading and checking every tensor…");
  let on = null;
  const loading = loadOnnx({
    record,
    path,
    open: files ? files.open : fetched(base),
    module,
    kernel: answer,
    checks,
    onProgress: (file, done) => { if (file === 0) onProgress(done / graph.size); },
    build: async (dims, tensors, raw) => {
      // The device is the kernel's decision 22, on the largest buffer the GPU
      // engine would bind: a dense embedding or head is tiled by rows, so one
      // row; a 4-bit weight binds its codes and its scales apart.
      const bound = (t) => (t.fmt === "f32" || t.fmt === "f16") && t.K ? t.K * (t.fmt === "f32" ? 4 : 2)
        : t.parts ? Math.max(...t.parts.map((b) => b.byteLength))
        : t.fmt?.startsWith("q4") ? (t.N * t.K) / 2 : t.bytes.byteLength;
      const largest = Math.max(...[...tensors.values()].map(bound));
      const limit = gpu ? Math.min(gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize) : 0;
      on = await ask("device", "device to run on", join(new Uint8Array([22, gpu ? 1 : 0]), be(limit, 8), be(largest, 8), be(sizes[chosen].bytes, 8), be(CPU_CEILING, 8)), (a) => (want === "auto" ? a === "webgpu" || a === "cpu" : a === want));
      onStatus(on === "webgpu" ? "Uploading to the GPU…" : "Preparing the CPU engine…");
      return on === "webgpu" ? createEngine({ device: gpu, dims, tensors, capacity: CAPACITY }) : createIntegerEngine({ module, ...raw, capacity: CAPACITY });
    },
  });
  // A failed load forgets every file it read or kept here, since one of them
  // may be the file that was not the record's; the next load downloads them.
  const { engine, stops, chat } = await loading.catch(async (e) => {
    if (files) await files.forget();
    throw e;
  });
  if (files) {
    await files.settled();
    const c = files.counts;
    const mb = (n) => (n / 1048576).toFixed(1);
    checks.add("kept", "model files read from this browser are the record's, checked as a download is", "pass", `${c.read} read here (${mb(c.readBytes)} MB), ${c.fetched} downloaded and kept (${mb(c.fetchedBytes)} MB), ${c.notKept} not kept for want of room, ${c.dropped} dropped for room (${c.droppedBytes} bytes), each named by its BLAKE3: ${files.kept.map((k) => `${k.path} ${k.address}`).join(", ")}`);
  }
  const generate = turner({ engine, module, ask, stops, chat });
  return Object.assign(generate, { task, model, dtype: chosen, device: on, engine, module, record });
}

// One turn of a loaded model: `generate(text, { max_new_tokens, onToken })`.
// A model with no chat template of its own (a weight container) is framed by
// what its vocabulary knows.
export function turner(session) {
  const { engine, module: m, ask } = session;
  const encode = (text) => {
    const bytes = m.callWith("tokeniser_encode", new TextEncoder().encode(text));
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4));
  };
  const decode = (ids) => new TextDecoder().decode(m.callWith("tokeniser_decode", new Uint8Array(new Uint32Array(ids).buffer)));
  // The chat frame by what the vocabulary knows. Qwen3 reads <think> as one
  // token, so it gets an empty thinking block and answers directly; Qwen2.5
  // does not, and gets the system line its own template writes when none is given.
  const frame = (prompt) => encode("<think>").length === 1
    ? `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
    : `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
  // An ONNX export's turn is written by its own chat template (Qwen3's with
  // thinking off, as the engine answers directly).
  const framing = (content) => session.chat
    ? session.chat.render({ ...session.chat.specials, messages: [{ role: "user", content }], add_generation_prompt: true, enable_thinking: false, strftime_now: (format) => strftime(new Date(), format) })
    : frame(content);
  // How many ids two prompts open with in common.
  const opening = (a, b) => {
    let n = 0;
    while (n < a.length && n < b.length && a[n] === b[n]) n++;
    return n;
  };
  // The BLAKE3 of token ids, four bytes each, little-endian, by the module.
  const label = (ids) => {
    const bytes = new Uint8Array(4 * ids.length);
    const view = new DataView(bytes.buffer);
    ids.forEach((id, i) => view.setUint32(4 * i, id, true));
    const id = new DataView(m.call("hash_start").slice().buffer).getUint32(0, true);
    m.callWith("hash_take", bytes, id);
    return m.call("hash_finish", id).slice();
  };
  const hexOf = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  let head = null;
  let held = null;

  async function generate(text, { max_new_tokens = 256, onToken = () => {} } = {}) {
    const prompt = encode(framing(text));
    // The model stops on the ids its own files name (EN-28); a file naming
    // none runs to the turn's budget.
    const said = session.stops ? null : m.call("tokeniser_stops_at");
    const stops = session.stops ?? new Set(said.length === 4 ? [new DataView(said.buffer, said.byteOffset, 4).getUint32(0, true)] : []);
    engine.reset();
    const budget = Number(await ask("turn", "token budget for this turn", join(new Uint8Array([9]), be(CAPACITY, 8), be(prompt.length, 8), be(max_new_tokens, 8)), (a) => /^\d+$/.test(a)));
    const t0 = performance.now();
    let firstAt = t0;
    // The frame's head, the ids every turn opens with, is computed once and
    // kept under the BLAKE3 of those ids. A later turn asks the kernel
    // (decision 27) whether the held label is the label of its own opening
    // ids, so a turn whose text merges into the head, or a template whose head
    // changed (a date), is computed from the start.
    head ??= opening(encode(framing("a")), encode(framing("b")));
    let from = 0;
    let reuse = null;
    if (held) {
      const asked = label(prompt.slice(0, held.length));
      const verdict = await ask("reuse", "computed prefix to reuse", join(new Uint8Array([27]), asked, held.label, be(held.length, 4), be(prompt.length, 4)), (a) => a === "reuse" || a === "recompute");
      reuse = { length: held.length, asked: hexOf(asked), held: hexOf(held.label), verdict };
      if (verdict === "reuse") {
        engine.restoreState(held.state);
        from = held.length;
      }
    }
    const tokens = await engine.run(prompt.slice(from), budget, {
      stops,
      onFirst: () => { firstAt = performance.now(); },
      onToken: (ids) => onToken(decode(ids), ids),
    });
    const end = performance.now();
    if (!held && head > 0 && head < prompt.length) {
      held = { length: head, label: label(prompt.slice(0, head)), state: engine.saveState(head) };
    }
    return {
      prompt: text,
      promptIds: prompt,
      promptTokens: prompt.length,
      tokens,
      text: decode(tokens),
      reuse,
      computed: prompt.length - from,
      timeToFirstToken: (firstAt - t0) / 1000,
      tokensPerSecond: tokens.length > 1 ? (tokens.length - 1) / ((end - firstAt) / 1000) : 0,
    };
  }
  return generate;
}
