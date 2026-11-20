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

import { startIntegerEngine } from "./integer.mjs";
import { TYPES, startStreamingEngine } from "./integer.mjs";
import { startIntegerEncoder } from "./integer-encoder.mjs";
import { PRINTED, printedKernels } from "./integer-gpu.mjs";
import { asker, checklist, hubRecord, keptFetch, openModule, tensorManifest } from "./loader.mjs";
import { strftime } from "./jinja.mjs";
import { ropeCandidates } from "./onnx-roles.mjs";
import { fetched, loadOnnx } from "./onnx-load.mjs";
import { keepFiles, manifestReady } from "./files.mjs";

// The precisions this engine runs, in the kernel's code for decision 24.
export const PRECISIONS = { q4f16: 0, q4: 1 };
// The tasks, in decision 24's code. The kernel answers unsupported task for
// any other; the module refuses nothing itself.
export const TASKS = { "text-generation": 0, "feature-extraction": 1, "text-classification": 2 };
// The tasks an encoder answers, on the module's integer engine (row 9).
const ENCODER_TASKS = new Set(["feature-extraction", "text-classification"]);
// What the kernel's choice of precision and device is weighed against. No
// browser reports how much memory a GPU has, so its budget is a stated 4 GiB;
// the CPU engine, single-threaded on a static host, takes models up to 2 GiB,
// so milestone 2's three text models all run on it (Ilya, 2026-11-20).
export const GPU_BUDGET = 4 * 2 ** 30;
export const CPU_CEILING = 2 * 2 ** 30;
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
  // The hub's index files are kept in this browser as they are read, and read
  // from here only when the network is off; they are checked again either way.
  const hubFetch = keptFetch();
  const record = await hubRecord({ indexBase: index, repo: model, checks, fetch: hubFetch });
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
  const dataPaths = [];
  for (let part = 1; part <= sizes[chosen].data; part++) {
    dataPaths.push(await ask("source", `the ${chosen} data file ${part}`, new Uint8Array([24, code, PRECISIONS[chosen], sizes[chosen].data, part]), (a) => record.files.some((f) => f.path === a && a.startsWith(folder))));
  }
  const base = record.origin?.resolve;
  if (!base) throw new Error("the record names no origin");
  const graph = record.files.find((f) => f.path === path);
  if (!graph) throw new Error(`the record names no ${path}`);
  // Every tensor's BLAKE3 address, from the hub's UOR index; a model it does
  // not list is refused by name before any byte of it is read.
  const manifest = await tensorManifest({ indexBase: index, record, paths: [path, ...dataPaths], checks, fetch: hubFetch });

  const module = await openModule(new URL("./engine.wasm", import.meta.url), ask);
  const decide = async (request) => new TextDecoder().decode(await kernel.dispatch(request));
  // The files kept from an earlier visit are read from this browser and
  // checked exactly as a download is; where the browser keeps none, every
  // file comes from the origin. The graph and data files are kept as their
  // tensors, once each by address, and only what is not held is downloaded.
  let files = null;
  try {
    files = await keepFiles({ record, open: fetched(base), module, kernel: decide, manifest });
  } catch (e) {
    checks.add("kept", "model files kept in this browser", "not checked", `none kept: ${e.message}`);
  }
  onStatus("Downloading and checking every tensor…");
  let on = null;
  const encoding = ENCODER_TASKS.has(task);
  // A turn given while the model is still arriving (GP-70): `options.early`
  // answers the text the caller has been given so far, or null. The model's
  // forward pass opens in a worker once its embedding and first layer are
  // in, each later tensor is handed to it as the importer admits it
  // (EN-111), and the turn's prompt goes through each layer as soon as that
  // layer is held (EN-110, EN-112). Its first token is still made only after
  // the last byte, the whole file's BLAKE3 and decision 28.
  const session = { module, ask, engine: null, stops: null, chat: null, kernels: null, pending: null };
  const generate = turner(session);
  const streaming = !encoding && typeof options.early === "function";
  // A layer is one the decoder reads a query projection of; the final norm
  // of a fused export is named under the layer after the last.
  const layerOf = (name) => Number(/(?:^|\.)layers\.(\d+)\./.exec(name)?.[1] ?? -1);
  const listed = [...(manifest.get(path)?.tensors ?? [])];
  const layers = new Set(listed.filter((t) => /\.q_proj\./.test(t.name)).map((t) => layerOf(t.name))).size;
  const needs = Array.from({ length: layers }, (_, l) => listed.filter((t) => layerOf(t.name) === l).map((t) => t.name));
  const embedding = listed.find((t) => /embed_tokens/.test(t.name))?.name;
  const got = new Set();
  const handed = new Set();
  let fresh = [];
  let held = 0;
  let rope = null;
  let adding = Promise.resolve();
  let earlyFailure = null;
  const deviceFor = async ({ d, heads, hd, ff, largest }) => {
    const usable = gpu && globalThis.crossOriginIsolated === true && [d, heads * hd, ff].every((w) => PRINTED.includes(w));
    if (want === "webgpu" && !usable) throw new Error(`device webgpu: ${globalThis.crossOriginIsolated !== true ? "this page is not cross-origin isolated, so the integer engine cannot wait on the GPU" : `the guest prints no group-sum kernel for every width of this model (${d}, ${heads * hd}, ${ff})`}`);
    const limit = usable ? Math.min(gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize) : 0;
    return ask("device", "device to run on", join(new Uint8Array([22, usable ? 1 : 0]), be(limit, 8), be(largest, 8), be(sizes[chosen].bytes, 8), be(CPU_CEILING, 8)), (a) => (want === "auto" ? a === "webgpu" || a === "cpu" : a === want));
  };
  const onHeader = async ({ config, configText, chat, stops }) => {
    session.configText = configText;
    session.chat = chat;
    session.stops = stops;
    const candidates = ropeCandidates(config);
    // A model whose rotary parameters are checked against its own cos cache
    // waits for the whole file, as before.
    rope = candidates.length === 1 ? (candidates[0].kind === "llama3" ? { llama3: candidates[0].llama3 } : {}) : null;
    // The device is asked before any of the model is computed: decision 22 on
    // the widths the config states and the largest 4-bit codes the UOR index
    // lists, as the built engine would put them.
    const heads = config.num_attention_heads;
    const hd = config.head_dim ?? config.hidden_size / heads;
    const largest = Math.max(0, ...listed.filter((t) => /_Q4$/.test(t.name)).map((t) => t.length));
    on = await deviceFor({ d: config.hidden_size, heads, hd, ff: config.intermediate_size, largest });
  };
  const onArrived = async (tensors, bytesAt) => {
    if (!rope || earlyFailure) return;
    for (const t of tensors) {
      got.add(t.name);
      fresh.push([t.name, t.type, bytesAt(t.offset, t.length)]);
    }
    while (held < layers && needs[held].every((n) => got.has(n))) held++;
    try {
      if (!session.engine) {
        if (held < 1 || !got.has(embedding)) return;
        session.engine = await startStreamingEngine({ module, config: session.configText, rope, capacity: CAPACITY, held, tensors: fresh });
      } else if (fresh.length) {
        const more = fresh;
        const now = held;
        // Handed in order, never awaited here, so the download is not held up
        // by the layers the worker runs meanwhile.
        adding = adding.then(() => session.engine.add(more, now));
      }
      for (const [name] of fresh) handed.add(name);
      fresh = [];
      const input = options.early();
      if (input && !session.pending) await generate.early(input.text, input);
    } catch (e) {
      earlyFailure = e;
    }
  };
  const loading = loadOnnx({
    encoder: encoding,
    record,
    path,
    open: files ? files.open : fetched(base),
    module,
    kernel: answer,
    checks,
    // Decision 28: the model is ready only when every address the UOR index
    // lists is the one the importer derived from the bytes, wherever they came
    // from; only then are newly downloaded tensors kept under their addresses.
    admitted: async (table) => {
      const subject = "every tensor the hub's UOR index lists is the address the importer derived (decision 28)";
      try {
        const verdict = files ? await files.admit(table) : await manifestReady({ manifest, table, module, kernel: decide });
        if (verdict.answer !== "ready") throw new Error(`the model is not ready: decision 28 answered "${verdict.answer}"`);
        checks.add("manifest", subject, "pass", `kernel answered "ready": ${verdict.derived} of ${verdict.listed} addresses`);
      } catch (e) {
        checks.add("manifest", subject, "fail", e.message);
        throw e;
      }
    },
    onProgress: (file, done) => { if (file === 0) onProgress(done / graph.size); },
    onHeader: streaming ? onHeader : null,
    onArrived: streaming ? onArrived : null,
    build: async (dims, tensors, raw) => {
      // A streamed engine has every weight the graph file held already; the
      // rest (a data file's) is handed in now, and the device was decision
      // 22's before any of it was computed.
      if (session.engine) {
        if (earlyFailure) throw earlyFailure;
        const rest = Object.keys(raw.table.places).filter((name) => !handed.has(name) && TYPES[raw.table.tensors[name]?.[0]] !== undefined);
        if (rest.length) adding = adding.then(() => session.engine.add(rest.map((name) => [name, TYPES[raw.table.tensors[name][0]], raw.read(name)]), layers));
        await adding;
        if (session.pending) await session.pending.started;
        onStatus(on === "webgpu" ? "Uploading to the GPU…" : "Preparing the CPU engine…");
        session.kernels = on === "webgpu" ? await printedKernels((request) => kernel.dispatch(request)) : null;
        // With no turn waiting, every product goes to WebGPU now; a waiting
        // turn's first token is made on the module and WebGPU takes the rest.
        if (session.kernels && !session.pending) await session.engine.attach(session.kernels);
        return session.engine;
      }
      // The device is the kernel's decision 22, on the largest buffer WebGPU
      // would bind: a 4-bit weight binds its codes and its scales apart, and
      // the module holds the dense ones. WebGPU is usable only where its
      // products can be handed back while the module waits, which takes
      // shared memory, and where the guest prints the group-sum kernel for
      // every width a product reads (GP-56).
      const bound = (t) => t.parts ? Math.max(...t.parts.map((b) => b.byteLength))
        : t.fmt?.startsWith("q4") ? (t.N * t.K) / 2 : 0;
      const largest = Math.max(...[...tensors.values()].map(bound));
      const usable = gpu && globalThis.crossOriginIsolated === true && [dims.d, dims.n_heads * dims.hd, dims.ff].every((w) => PRINTED.includes(w));
      if (want === "webgpu" && !usable) throw new Error(`device webgpu: ${globalThis.crossOriginIsolated !== true ? "this page is not cross-origin isolated, so the integer engine cannot wait on the GPU" : `the guest prints no group-sum kernel for every width of this model (${dims.d}, ${dims.n_heads * dims.hd}, ${dims.ff})`}`);
      const limit = usable ? Math.min(gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize) : 0;
      on = await ask("device", "device to run on", join(new Uint8Array([22, usable ? 1 : 0]), be(limit, 8), be(largest, 8), be(sizes[chosen].bytes, 8), be(CPU_CEILING, 8)), (a) => (want === "auto" ? a === "webgpu" || a === "cpu" : a === want));
      onStatus(on === "webgpu" ? "Uploading to the GPU…" : "Preparing the CPU engine…");
      const kernels = on === "webgpu" ? await printedKernels((request) => kernel.dispatch(request)) : null;
      return startIntegerEngine({ module, ...raw, capacity: CAPACITY, helpers: options.helpers, gpu: kernels });
    },
  });
  // A failed load forgets every file it read or kept here, since one of them
  // may be the file that was not the record's; the next load downloads them.
  const loaded = await loading.catch(async (e) => {
    if (files) await files.forget();
    throw e;
  });
  if (files) {
    await files.settled();
    const c = files.counts;
    const mb = (n) => (n / 1048576).toFixed(1);
    checks.add("kept", "model files read from this browser are the record's, checked as a download is", "pass", `${c.read} read here (${mb(c.readBytes)} MB), ${c.fetched} downloaded and kept (${mb(c.fetchedBytes)} MB), ${c.notKept} not kept for want of room, ${c.dropped} dropped for room (${c.droppedBytes} bytes), each named by its BLAKE3: ${files.kept.map((k) => `${k.path} ${k.address}`).join(", ")}`);
    const t = files.tensors;
    checks.add("store", "tensors kept once by their BLAKE3 address, only those not held downloaded", "pass", `${t.read} tensors read here (${t.readBytes} bytes), ${t.fetched} downloaded (${t.fetchedBytes} bytes), ${t.kept} newly kept (${t.keptBytes} bytes), ${t.restFetchedBytes} bytes of the rest downloaded, in ${t.reads} reads`);
  }
  if (encoding) return encoderPipeline({ task, model, dtype: chosen, record, module, ask, checks, loaded, size: sizes[chosen].bytes });
  const { engine, stops, chat } = loaded;
  // GP-58: what this run promises (decision 30). Every product and every other
  // stage is exact on integers, on the CPU and on WebGPU alike, and no float
  // is summed in any order, so the kernel's class is "everywhere".
  const promised = await ask("class", "reproducibility class of this run (decision 30)", new Uint8Array([30, 1, 1, 0]), (a) => a === "everywhere");
  const identity = modelIdentity({ record, model, dtype: chosen, paths: [path, ...dataPaths], module });
  checks.add("receipt", "the model named by the BLAKE3 of the files that decide its answers", "pass", `${identity.label}: ${identity.files.map((f) => `${f.path} ${f.blake3.slice(0, 12)}…`).join(", ")}`);
  // The turn exists before the model does (GP-70); what the load found joins it.
  Object.assign(session, { engine, stops, chat, identity, promised, device: on, adapter: engine.gpu?.adapter ?? null });
  return Object.assign(generate, { task, model, dtype: chosen, device: on, engine, module, record, identity, promised });
}

// The BLAKE3 of `bytes`, as hex, by the module.
function blake3Of(m, bytes) {
  const id = new DataView(m.call("hash_start").slice().buffer).getUint32(0, true);
  m.callWith("hash_take", bytes, id);
  return Array.from(m.call("hash_finish", id).slice(), (b) => b.toString(16).padStart(2, "0")).join("");
}

// The files whose bytes decide a run's answers, given the token ids: the
// graph and its data files (every weight), config.json (the shapes and the
// rotary parameters) and generation_config.json (the stop ids). The tokenizer
// and the chat template only turn text into ids, and a receipt carries the
// ids. Each is named by the BLAKE3 the hub's record gives it, which the load
// checked; the model's label is the BLAKE3 of those lines, sorted by path.
function modelIdentity({ record, model, dtype, paths, module }) {
  const named = new Set([...paths, "config.json", "generation_config.json"]);
  const files = record.files
    .filter((f) => named.has(f.path) && f.blake3)
    .map((f) => ({ path: f.path, blake3: f.blake3 }))
    .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  const lines = new TextEncoder().encode(files.map((f) => `${f.path}\t${f.blake3}\n`).join(""));
  return { repo: model, revision: record.revision ?? null, dtype, files, label: blake3Of(module, lines) };
}

// A receipt's labels: the input is its token ids and its budget, the output
// the token ids it produced, each as four little-endian bytes; the budget
// follows the input's ids.
const idBytes = (ids, extra = []) => {
  const all = [...ids, ...extra];
  const bytes = new Uint8Array(4 * all.length);
  const view = new DataView(bytes.buffer);
  all.forEach((id, i) => view.setUint32(4 * i, id, true));
  return bytes;
};
const unhexLabel = (hex) => Uint8Array.from(hex.match(/../g) ?? [], (h) => parseInt(h, 16));

// A receipt as text for a link (`#receipt=…`), and back.
export const receiptText = (receipt) => btoa(JSON.stringify(receipt)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
export const receiptFrom = (text) => JSON.parse(atob(text.replace(/-/g, "+").replace(/_/g, "/")));

// An encoder's task on the module's integer engine: `embed(text)` answers
// the sentence embedding's bits and their Float32Array view, `classify(text)`
// the label the module chose and every score. The device is decision 22's,
// asked as for a browser with no GPU, since only the module runs encoders.
async function encoderPipeline({ task, model, dtype, record, module, ask, checks, loaded, size }) {
  const largest = Math.max(...Object.values(loaded.table.places).map((place) => place[2]));
  const device = await ask("device", "device to run on", join(new Uint8Array([22, 0]), be(0, 8), be(largest, 8), be(size, 8), be(CPU_CEILING, 8)), (a) => a === "cpu");
  const encoder = await startIntegerEncoder({ module, ...loaded });
  checks.add("engine", "the encoder runs on the module's integer engine", "pass", `${encoder.dims.n_layers} layers of width ${encoder.dims.d}, every stage on integers (EN-62${task === "text-classification" ? ", EN-63" : ""})`);
  const run = task === "feature-extraction" ? (text) => encoder.embed(text) : (text) => encoder.classify(text);
  return Object.assign(async (text) => run(text), { task, model, dtype, device, engine: encoder, module, record });
}

// One turn of a loaded model: `generate(text, { max_new_tokens, onToken })`.
// A model with no chat template of its own (a weight container) is framed by
// what its vocabulary knows.
export function turner(session) {
  const { module: m, ask } = session;
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
  // A turn is a user's text or, as transformers.js takes it, a list of messages.
  const render = (messages, add_generation_prompt = true) => session.chat
    ? session.chat.render({ ...session.chat.specials, messages, add_generation_prompt, enable_thinking: false, strftime_now: (format) => strftime(new Date(), format) })
    : frame(messages.at(-1).content);
  const framing = (input) => render(Array.isArray(input) ? input : [{ role: "user", content: input }]);
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

  // The ids a run stops on: the model's own (EN-28), or the vocabulary's
  // single stop id for a file naming none.
  const stopIds = () => {
    if (session.stops) return session.stops;
    const said = m.call("tokeniser_stops_at");
    return new Set(said.length === 4 ? [new DataView(said.buffer, said.byteOffset, 4).getUint32(0, true)] : []);
  };

  // GP-58: a run's receipt. Everything needed to run it again and to check
  // the answer byte for byte: the model's label, the input's ids and budget
  // and their label, the output's label, and the class the kernel gave.
  const receiptOf = (ids, budget, tokens) => ({
    v: 1,
    model: { repo: session.identity.repo, revision: session.identity.revision, dtype: session.identity.dtype, label: session.identity.label },
    input: { ids: Array.from(ids), budget, label: blake3Of(m, idBytes(ids, [budget])) },
    output: { tokens: tokens.length, label: blake3Of(m, idBytes(tokens)) },
    class: session.promised,
    device: session.device,
  });

  // GP-59: run a receipt again in this tab, on this device, from its own ids,
  // and put the two output labels to the kernel (decision 31). The model is
  // ready for it only when this tab's model has the receipt's label; a run
  // from another device is comparable because the class is "everywhere".
  async function check(receipt) {
    const { engine } = session;
    const ready = receipt?.model?.label === session.identity.label ? 1 : 0;
    const stops = stopIds();
    let replayed = { label: "0".repeat(64), tokens: [] };
    if (ready) {
      await engine.reset();
      const tokens = await engine.run(receipt.input.ids, receipt.input.budget, { stops });
      replayed = { label: blake3Of(m, idBytes(tokens)), tokens };
    }
    const classCode = { everywhere: 2, "same hardware": 1, none: 0 }[receipt.class] ?? 0;
    const sameAdapter = receipt.device === session.device ? 1 : 0;
    const verdict = await ask("receipt", "a receipt replayed and its output label compared (decision 31)", join(new Uint8Array([31]), unhexLabel(receipt.output.label), unhexLabel(replayed.label), new Uint8Array([ready, classCode, sameAdapter])), (a) => ["confirmed", "mismatch", "not comparable"].includes(a));
    return { verdict, ready: ready === 1, claimed: receipt.output.label, replayed: replayed.label, tokens: replayed.tokens };
  }

  // GP-60: open a past answer at one layer, position and component. The
  // receipt is run again to recover its output ids (and checked, decision
  // 31), the kernel says whether the replay is exact (decision 32), and the
  // engine replays the ids to that position and reads the component there:
  // the value that produced the answer, named by its BLAKE3.
  const COMPONENTS = ["residual", "attention", "feed-forward"];
  async function probe(receipt, { layer, position, component = "residual" }) {
    const { engine } = session;
    const checked = await check(receipt);
    if (checked.verdict !== "confirmed") return { verdict: "not checked", checked };
    const ids = [...receipt.input.ids, ...checked.tokens];
    const positions = ids.length - 1;
    const layers = engine.sizes.layers;
    const index = COMPONENTS.indexOf(component);
    const classCode = { everywhere: 2, "same hardware": 1, none: 0 }[receipt.class] ?? 0;
    const sameAdapter = receipt.device === session.device ? 1 : 0;
    const verdict = await ask("probe", "a probe of a past answer (decision 32)", join(new Uint8Array([32]), be(layer, 4), be(layers, 4), be(position, 4), be(positions, 4), new Uint8Array([index < 0 ? 255 : index, COMPONENTS.length, classCode, sameAdapter])), (a) => ["exact replay", "approximate replay", "out of range"].includes(a));
    if (verdict === "out of range") return { verdict, checked };
    const bits = await engine.probe(ids, position, layer, index);
    return { verdict, layer, position, component, label: blake3Of(m, bits), values: Array.from(new Uint32Array(bits.buffer, bits.byteOffset, bits.length / 4)), checked };
  }

  // A turn's budget: decision 9.
  const budgetOf = async (prompt, max_new_tokens) => Number(await ask("turn", "token budget for this turn", join(new Uint8Array([9]), be(CAPACITY, 8), be(prompt.length, 8), be(max_new_tokens, 8)), (a) => /^\d+$/.test(a)));

  // A turn given while the model is still arriving (GP-70): its ids and its
  // budget (decision 9) now, and its prompt handed to the streamed engine,
  // which runs each layer as soon as that layer is held. generate(text)
  // finishes it once the load has ended.
  async function early(text, { max_new_tokens = 256 } = {}) {
    const prompt = encode(framing(text));
    const budget = await budgetOf(prompt, max_new_tokens);
    session.pending = { text, prompt, budget, at: performance.now(), started: session.engine.prefill(prompt) };
  }

  // `template: false` runs the text as it is, as transformers.js runs a string.
  async function generate(text, { max_new_tokens = 256, onToken = () => {}, template = true } = {}) {
    const { engine } = session;
    // The model stops on the ids its own files name (EN-28); a file naming
    // none runs to the turn's budget.
    const stops = stopIds();
    const pending = template && session.pending?.text === text ? session.pending : null;
    session.pending = null;
    if (pending) {
      // Its layers ran as they arrived; the final norm and the head now,
      // then the rest of the turn.
      await pending.started;
      let firstAt = pending.at;
      const tokens = await engine.finish(pending.budget, {
        stops,
        gpu: session.kernels,
        onFirst: () => { firstAt = performance.now(); },
        onToken: (ids) => onToken(decode(ids), ids),
      });
      const end = performance.now();
      return {
        receipt: session.identity ? receiptOf(pending.prompt, pending.budget, tokens) : null,
        prompt: text,
        promptIds: pending.prompt,
        promptTokens: pending.prompt.length,
        tokens,
        text: decode(tokens),
        reuse: null,
        computed: pending.prompt.length,
        early: true,
        timeToFirstToken: (firstAt - pending.at) / 1000,
        tokensPerSecond: tokens.length > 1 ? (tokens.length - 1) / ((end - firstAt) / 1000) : 0,
      };
    }
    const prompt = encode(template ? framing(text) : text);
    await engine.reset();
    const budget = await budgetOf(prompt, max_new_tokens);
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
        await engine.restoreState(held.state);
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
      held = { length: head, label: label(prompt.slice(0, head)), state: await engine.saveState(head) };
    }
    const receipt = session.identity ? receiptOf(prompt, budget, tokens) : null;
    return {
      receipt,
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
  return Object.assign(generate, { render, decode, check, probe, early });
}
