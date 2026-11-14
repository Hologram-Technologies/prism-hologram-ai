// Load a model's weight container into the engine, asking the kernel before
// every step it decides and recording each answer. The page never parses a
// model file: the header, the per-block labels, the whole-file root and the
// tokeniser are the wasm module's (crates/wasm); the verdicts are the kernel's
// (app.holo). This module moves bytes between them and keeps the record.

const text = new TextDecoder();

// The wasm module's calls, with bytes in and out through its linear memory.
// Every call answers 0 for yes; a no leaves the failure's name as the result.
export async function openModule(url) {
  const { instance } = await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {});
  const x = instance.exports;
  const result = () => new Uint8Array(x.memory.buffer).slice(x.result_at(), x.result_at() + x.result_len());
  const check = (name, status) => {
    if (status !== 0) throw new Error(`${name}: ${text.decode(result())}`);
    return result();
  };
  const withBytes = (bytes, run) => {
    const at = x.allocate(bytes.length);
    new Uint8Array(x.memory.buffer).set(bytes, at);
    try {
      return run(at, bytes.length);
    } finally {
      x.release(at);
    }
  };
  const call = (name, ...args) => check(name, x[name](...args));
  const callWith = (name, bytes, ...rest) => withBytes(bytes, (at, len) => check(name, x[name](at, len, ...rest)));
  return { x, call, callWith };
}

const u64 = (bytes, at) => Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(at, true));
const span = (bytes) => ({ off: u64(bytes, 0), len: u64(bytes, 8) });
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
const unhex = (text) => Uint8Array.from(text.match(/../g), (h) => parseInt(h, 16));
const be = (value, width) => {
  const out = new Uint8Array(width);
  let v = BigInt(value);
  for (let i = width - 1; i >= 0; i--) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
};
const join = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};

async function sha256(bytes) {
  return hex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
}

async function rangeRead(url, start, length) {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${start + length - 1}` } });
  if (response.status !== 206) throw new Error(`range ${start}+${length}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length !== length) throw new Error(`range ${start}+${length}: ${bytes.length} bytes`);
  return bytes;
}

async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// The record of what was checked, in order, as the page shows it and as a
// test reads it from window.__qengine.
export function checklist(onChange = () => {}) {
  const items = [];
  return {
    items,
    add(step, subject, verdict, detail = "") {
      const item = { step, subject, verdict, detail, at: Math.round(performance.now()) };
      items.push(item);
      onChange(item, items);
      return item;
    },
  };
}

const JOURNEY = { idle: 0, preflight: 1, download: 2, compile: 3, materialize: 4, ready: 5 };
const EVENT = { start: 1, preflightPassed: 2, downloadDone: 3, compileDone: 4, labelsVerified: 5 };

// Ask the kernel; answer its text. `kernel.dispatch` is the verified guest.
export function asker(kernel, checks) {
  return async (step, subject, request, expect) => {
    const answer = text.decode(await kernel.dispatch(request));
    const ok = typeof expect === "function" ? expect(answer) : answer === expect;
    checks.add(step, subject, ok ? "pass" : "fail", `kernel answered "${answer}"`);
    if (!ok) throw new Error(`${subject}: the kernel answered "${answer}"`);
    return answer;
  };
}

// The hub's record for a model, checked against the index root by the record's
// SHA-256. The root's own signature is not checked here yet: that waits on the
// hub's importable client.
export async function hubRecord({ indexBase, repo, checks }) {
  const [owner, name] = repo.split("/");
  const recordBytes = new Uint8Array(await (await fetch(`${indexBase}/models/${owner}/${name}.json`)).arrayBuffer());
  const root = await (await fetch(`${indexBase}/root.json`)).json();
  const entry = (root.records ?? []).find((r) => r.repo === repo);
  const record = JSON.parse(text.decode(recordBytes));
  const digest = await sha256(recordBytes);
  const ok = !!entry && entry.sha256 === digest && entry.revision === record.revision;
  checks.add("record", "hub record matches the index root", ok ? "pass" : "fail", `record sha256 ${digest}${entry ? "" : " (not in root)"}`);
  checks.add("record", "index root signature", "not checked", "waits on the hub's client module");
  if (!ok) throw new Error("the hub record does not match its index root");
  return record;
}

// Load the container at `url` (size and whole-file SHA-256 from the record)
// into an engine built by `build(dims, tensors)`. Answers the engine and the
// tokeniser call.
export async function loadContainer({ url, size, sha256: wantRoot, module: m, ask, checks, build, preflight = async () => {}, onProgress = () => {} }) {
  let state = "idle";
  const journey = async (event, next) => {
    await ask("journey", `${state} → ${next}`, new Uint8Array([1, JOURNEY[state], EVENT[event]]), next);
    state = next;
  };
  await journey("start", "preflight");

  // The header: ten bytes say where it lies; the reader reads it and nothing past it.
  const first = await rangeRead(url, 0, 10);
  const where = span(m.callWith("container_header", first));
  const header = await rangeRead(url, 0, where.off + where.len);
  m.callWith("container_read", header);
  const count = m.x.container_entries();
  const tokenizerAt = span(m.call("container_tokeniser"));
  checks.add("header", "container header read", "pass", `${count} entries, header ${where.len} bytes, tokenizer ${tokenizerAt.len} bytes`);
  const meta = JSON.parse(text.decode(header.subarray(where.off, where.off + where.len)).trimEnd());
  const dims = meta.manifest;
  const shape = new Map(meta.dir.map((d) => [d.name, d]));

  // Before any weight moves: every entry's byte range and block length.
  const entries = [];
  const aliases = [];
  const label = (name) => shape.get(name).kappa.replace(/^blake3:/, "");
  let previousEnd = tokenizerAt.off + tokenizerAt.len;
  for (let i = 0; i < count; i++) {
    const name = text.decode(m.call("container_name", i));
    const at = span(m.call("container_span", i));
    // A container stores tied tensors once: a later entry may name exactly the
    // bytes of an earlier one. That is not a new range; the kernel confirms the
    // two are the same buffer by its admission decision on their labels and lengths.
    const same = entries.find((e) => e.off === at.off && e.len === at.len);
    if (same) {
      const request = join(new Uint8Array([5]), unhex(label(name)), unhex(label(same.name)), be(at.len, 8), be(same.len, 8));
      await ask("alias", `${name} is the same bytes as ${same.name}`, request, "admit");
      aliases.push({ name, of: same.name });
      continue;
    }
    await ask("range", `${name} byte range`, m.call("container_range_question", i, BigInt(size), BigInt(previousEnd)), "accept");
    const length = Number(await ask("block", `${name} block length`, m.call("container_block_question", i), (a) => /^\d+$/.test(a)));
    entries.push({ index: i, name, ...at, length });
    previousEnd = at.off + at.len;
  }
  await preflight(dims, entries);
  await journey("preflightPassed", "download");

  // One stream from byte 0: every byte into the whole-file root, each entry's
  // bytes into its own label, each entry admitted, unpacked and handed over.
  const tensors = new Map();
  const response = await fetch(url, { headers: { Range: `bytes=0-${size - 1}` } });
  if (response.status !== 206 && response.status !== 200) throw new Error(`download: HTTP ${response.status}`);
  const reader = response.body.getReader();
  m.call("download_start");
  let offset = 0;
  let current = 0; // the entry whose bytes come next
  let held = [];   // the current entry's bytes so far
  let heldLength = 0;
  let tokenizerBytes = null;
  const tokenizerParts = [];
  const finishEntry = async (e) => {
    const stored = new Uint8Array(heldLength);
    let p = 0;
    for (const part of held) { stored.set(part, p); p += part.length; }
    held = []; heldLength = 0;
    await ask("admit", `${e.name} label and length`, m.call("container_admission_question", e.index), "admit");
    const unpacked = await gunzip(stored);
    const ok = unpacked.length === e.length;
    checks.add("unpack", `${e.name} unpacked length`, ok ? "pass" : "fail", `${unpacked.length} bytes, kernel says ${e.length}`);
    if (!ok) throw new Error(`${e.name}: unpacked to ${unpacked.length} bytes, not ${e.length}`);
    const d = shape.get(e.name);
    tensors.set(e.name, { name: e.name, fmt: d.fmt, N: d.N, K: d.K, bytes: unpacked });
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    let chunk = value;
    while (chunk.length) {
      const e = entries[current];
      // The next boundary: where the tokenizer or the current entry starts or ends.
      let boundary = size;
      if (e) boundary = offset < e.off ? e.off : e.off + e.len;
      const tStart = tokenizerAt.off, tEnd = tokenizerAt.off + tokenizerAt.len;
      if (offset < tStart) boundary = Math.min(boundary, tStart);
      else if (offset < tEnd) boundary = Math.min(boundary, tEnd);
      const take = chunk.subarray(0, Math.min(chunk.length, boundary - offset));
      m.callWith("download_take", take);
      if (offset >= tStart && offset < tEnd) tokenizerParts.push(take.slice());
      if (e && offset >= e.off) { held.push(take.slice()); heldLength += take.length; }
      offset += take.length;
      chunk = chunk.subarray(take.length);
      if (e && offset === e.off) m.call("download_block");
      if (e && offset === e.off + e.len) {
        await finishEntry(e);
        current++;
        const next = entries[current];
        if (next && offset === next.off) m.call("download_block");
      }
      onProgress(offset, size);
    }
  }
  if (offset !== size) throw new Error(`download ended at ${offset} of ${size} bytes`);
  for (const { name, of } of aliases) tensors.set(name, tensors.get(of));
  if (current !== entries.length) throw new Error(`only ${current} of ${entries.length} entries arrived`);
  const root = hex(m.call("download_root"));
  const rootOk = root === wantRoot;
  checks.add("root", "whole-file SHA-256 matches the hub record", rootOk ? "pass" : "fail", root);
  if (!rootOk) throw new Error("the whole file's SHA-256 is not the record's");
  await journey("downloadDone", "compile");

  tokenizerBytes = new Uint8Array(tokenizerAt.len);
  { let p = 0; for (const part of tokenizerParts) { tokenizerBytes.set(part, p); p += part.length; } }
  m.callWith("tokeniser_read_gguf", tokenizerBytes);
  checks.add("tokenizer", "tokenizer read from the container", "pass", `${m.x.tokeniser_size()} tokens`);

  const engine = await build(dims, tensors);
  await journey("compileDone", "materialize");
  await journey("labelsVerified", "ready");
  return { engine, dims };
}
