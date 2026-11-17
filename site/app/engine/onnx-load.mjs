// Load an ONNX export from its own files, as the hub's record names them: the
// small files checked against the record's SHA-256, the graph and any data
// files streamed through the module's importer with the kernel asked about
// every span, operator, file and 4-bit weight, every weight placed by the
// node that reads it, and the engine built from the bytes as they arrived.
// Nothing is repacked. The decisions are the kernel's; this module moves
// bytes and keeps the checks a reader sees.

import { compileTemplate } from "./jinja.mjs";
import { confirmRope, engineTensors, ropeCandidates, roles } from "./onnx-roles.mjs";

const text = new TextDecoder();
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
const unhex = (h) => Uint8Array.from(h.match(/../g), (b) => parseInt(b, 16));
const sha256 = async (bytes) => hex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
const half = (h) => {
  const e = (h >> 10) & 31;
  const m = h & 1023;
  const v = e === 0 ? m * 2 ** -24 : (1024 + m) * 2 ** (e - 25);
  return h & 0x8000 ? -v : v;
};

// open(path) answers a ReadableStream of the file's bytes: fetch in the page,
// local bytes in a test.
export const fetched = (base) => async (path) => {
  const response = await fetch(base + path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.body;
};

async function whole(stream) {
  const parts = [];
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

export async function loadOnnx({ record, path, open, module: m, kernel, checks, build, onProgress = () => {} }) {
  // A file the record names, with a SHA-256 when one is needed: the kernel
  // admits a model file against it.
  const entry = (p, needsSha256 = true) => {
    const file = record.files.find((f) => f.path === p);
    if (!file || (needsSha256 && !file.sha256)) throw new Error(`the record names no SHA-256 for ${p}`);
    return file;
  };
  // A small file of the repository, checked against the record before use:
  // by its SHA-256 where the record gives one (a file Git LFS stores), and
  // otherwise by its Git blob SHA-1, which is how Git names a file it holds
  // itself. SHA-1 is the weaker of the two, and the check says which it was.
  const small = async (p, optional = false) => {
    if (optional && !record.files.some((f) => f.path === p)) return null;
    const file = entry(p, false);
    const bytes = await whole(await open(p));
    let ok;
    let detail;
    if (file.sha256) {
      const digest = await sha256(bytes);
      ok = digest === file.sha256;
      detail = `sha256 ${digest}`;
    } else if (file.gitSha1) {
      const head = new TextEncoder().encode(`blob ${bytes.length}\0`);
      const blob = new Uint8Array(head.length + bytes.length);
      blob.set(head);
      blob.set(bytes, head.length);
      const digest = hex(new Uint8Array(await crypto.subtle.digest("SHA-1", blob)));
      ok = digest === file.gitSha1;
      detail = `git blob sha1 ${digest} (the record gives no SHA-256)`;
    } else {
      throw new Error(`the record names no hash for ${p}`);
    }
    ok = ok && bytes.length === file.size;
    checks.add("file", `${p} is the record's`, ok ? "pass" : "fail", `${bytes.length} bytes, ${detail}`);
    if (!ok) throw new Error(`${p} is not the file the record names`);
    return bytes;
  };
  const config = JSON.parse(text.decode(await small("config.json")));
  const generation = await small("generation_config.json", true);
  m.callWith("tokeniser_read", await small("tokenizer.json"));
  // The chat template is the model's own, from tokenizer_config.json, with the
  // special tokens it names; a list of templates gives the one named default.
  const tokenizerConfig = await small("tokenizer_config.json", true);
  let chat = null;
  if (tokenizerConfig) {
    const tc = JSON.parse(text.decode(tokenizerConfig));
    const source = Array.isArray(tc.chat_template) ? tc.chat_template.find((t) => t.name === "default")?.template : tc.chat_template;
    if (typeof source === "string") {
      const specials = Object.fromEntries(["bos_token", "eos_token", "pad_token", "unk_token"].filter((k) => typeof tc[k] === "string").map((k) => [k, tc[k]]));
      chat = { render: compileTemplate(source), specials };
    }
  }
  checks.add("template", "the chat template is the model's own", chat ? "pass" : "not checked", chat ? `from tokenizer_config.json, ${Object.keys(chat.specials).join(", ") || "no special tokens"}` : "the model's files carry none");

  // Every question the importer asks goes to the kernel, counted.
  let asked = 0;
  m.setKernel((request) => {
    asked++;
    return kernel(request);
  });

  // The bytes of every file, kept as they arrive, for the engine to bind.
  const kept = new Map();
  const stream = async (index, p, take) => {
    const parts = [];
    let got = 0;
    const reader = (await open(p)).getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      take(value);
      parts.push({ at: got, bytes: value });
      got += value.length;
      onProgress(index, got);
    }
    kept.set(index, parts);
    return got;
  };
  const graph = entry(path);
  const folder = path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";
  const declared = record.files.filter((f) => f.path.startsWith(`${path}_data`)).length;
  m.callWith("onnx_start", unhex(graph.sha256), BigInt(graph.size), declared);
  await stream(0, path, (chunk) => m.callWith("onnx_take", chunk));
  const dataFiles = text.decode(m.call("onnx_graph_finish")).split("\n").filter(Boolean);
  for (const [i, name] of dataFiles.entries()) {
    const file = entry(folder + name);
    m.callWith("onnx_data_start", unhex(file.sha256), i + 1, BigInt(file.size));
    await stream(i + 1, folder + name, (chunk) => m.callWith("onnx_data_take", chunk));
    m.call("onnx_data_finish");
  }
  const table = JSON.parse(text.decode(m.call("onnx_finish")));
  const count = Object.keys(table.places).length;
  checks.add("import", "every span, operator, file and 4-bit weight admitted", "pass", `${count} tensors named by their BLAKE3, ${asked} questions to the kernel, ${dataFiles.length} data files`);

  // A tensor's bytes, from where the importer says they lie.
  const read = (name) => {
    const [file, offset, length] = table.places[name];
    const out = new Uint8Array(length);
    for (const { at, bytes } of kept.get(file)) {
      const from = Math.max(offset, at);
      const to = Math.min(offset + length, at + bytes.length);
      if (from < to) out.set(bytes.subarray(from - at, to - at), from - offset);
    }
    return out;
  };

  const { dims, weights } = roles(table, config);
  checks.add("roles", "every weight placed by the node that reads it", "pass", `${weights.size} tensors for ${config.model_type}, ${dims.n_layers} layers`);
  // The rotary parameters: the config's when it states them; otherwise the
  // candidates are tried against the export's own cos cache.
  const candidates = ropeCandidates(config);
  let rope = {};
  if (candidates.length === 1) {
    rope = candidates[0].kind === "llama3" ? { llama3: candidates[0].llama3 } : {};
  } else if (candidates.length > 1) {
    if (!table.places.cos_cache || !table.places.sin_cache) throw new Error("rope: the config names no scaling and the export has no cos and sin cache to check one against");
    const [cosType, [cacheRows, width]] = table.tensors.cos_cache;
    if (cosType !== "f16") throw new Error(`rope: the cos cache is ${cosType}, read here as f16 only`);
    const cos = new Uint16Array(read("cos_cache").slice().buffer);
    const sin = new Uint16Array(read("sin_cache").slice().buffer);
    const rows = [1, 7, 1000, 8191].filter((p) => p < cacheRows).map((position) => ({
      position,
      cos: Array.from(cos.subarray(position * width, (position + 1) * width), half),
      sin: Array.from(sin.subarray(position * width, (position + 1) * width), half),
    }));
    const confirmed = confirmRope(candidates, dims, rows);
    rope = confirmed.dims;
    checks.add("rope", "rotary parameters agree with the export's own cache", "pass", `${JSON.stringify(rope)}, worst ${confirmed.worst.toExponential(1)}`);
  } else {
    throw new Error(`rope: the config's scaling ${JSON.stringify(config.rope_scaling)} is not one this engine runs`);
  }

  const engine = await build({ ...dims, ...rope }, engineTensors(weights, read));
  // The ids the model stops on, as its own files name them.
  const named = (generation ? JSON.parse(text.decode(generation)).eos_token_id : null) ?? config.eos_token_id;
  const stops = new Set([named].flat().filter((id) => Number.isInteger(id)));
  checks.add("stop", "the ids the model stops on", stops.size ? "pass" : "not checked", stops.size ? [...stops].join(", ") : "the model's files name none");
  return { engine, dims: { ...dims, ...rope }, stops, table, asked, chat };
}
