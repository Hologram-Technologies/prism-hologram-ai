// Load a model's weight container into the engine, asking the kernel before
// every step it decides and recording each answer. The page never parses a
// model file: the header, the per-block labels, the whole-file root and the
// tokeniser are the wasm module's (crates/wasm); the verdicts are the kernel's
// (app.holo). This module moves bytes between them and keeps the record.

import { HubClientError, openIndex, readRecord } from "./hub/hub-client.mjs";
import { uorShardOf } from "./hub/uorkey.mjs";

const text = new TextDecoder();

// The hub's index files (its signed root, the signature, a model's record, the
// UOR index and its shards), kept in this browser as they are read, so a
// return visit with the network off reads them from here. Nothing kept is
// trusted: the hub's client checks the kept root by its signature and the
// record by the root's BLAKE3 exactly as it checks a download, and the
// manifest is checked by the importer's own addresses (decision 28). Only a
// network failure falls back to the kept copy; an answer from the hub, even a
// refusal, is the hub's. `offline` counts the files read from here.
export function keptFetch(name = "hologram-hub") {
  const offline = [];
  const kept = async (input) => {
    const url = String(input);
    const store = globalThis.caches ? await caches.open(name) : null;
    let response;
    try {
      response = await fetch(url);
    } catch (error) {
      const hit = store && (await store.match(url));
      if (!hit) throw error;
      offline.push(url);
      return hit;
    }
    if (response.ok && store) await store.put(url, response.clone());
    return response;
  };
  return Object.assign(kept, { offline });
}

// The wasm module's calls, with bytes in and out through its linear memory.
// Every call answers 0 for yes; a no leaves the failure's name as the result.
export async function openModule(url) {
  // The module reaches the kernel through one import, `kernel.ask`: the
  // request at `at`, `len` bytes, and the answer written at `out`, at most
  // `cap` bytes; it answers the answer's whole length, so a longer one is cut
  // and refused. Until setKernel names the kernel, every question is answered
  // with nothing, which the importer refuses.
  let kernel = null;
  let x = null;
  const imports = {
    kernel: {
      ask: (at, len, out, cap) => {
        if (!kernel) return 0;
        const answer = kernel(new Uint8Array(x.memory.buffer).slice(at, at + len));
        new Uint8Array(x.memory.buffer).set(answer.subarray(0, Math.min(answer.length, cap)), out);
        return answer.length;
      },
    },
  };
  const { instance } = await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), imports);
  x = instance.exports;
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
  // `ask(request) -> answer`, both bytes, answered synchronously: the holo's
  // guest run on the request, as `runGuest` runs it.
  const setKernel = (ask) => {
    kernel = ask;
  };
  return { x, call, callWith, setKernel };
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

// The hub's record for a model, by the hub's own client (hub/VENDOR.md): the
// index root checked by its Sigstore signature against the trust policy this
// page pins (hub/policy.json, the hub model's own anchors), and the record by
// the BLAKE3 the root names. A policy fetched from the hub would be vouched for
// by the thing it is meant to check, so the page carries its own.
export async function hubRecord({ indexBase, repo, checks, fetch: hubFetch = keptFetch() }) {
  const policy = await (await fetch(new URL("./hub/policy.json", import.meta.url))).json();
  const { blake3Hex } = await import("../../vendor/prism-code/src/blake3.mjs");
  const hash = async (input) => {
    if (input instanceof Uint8Array) return blake3Hex(input);
    const parts = [];
    for await (const chunk of input) parts.push(chunk);
    const whole = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    parts.reduce((at, p) => (whole.set(p, at), at + p.length), 0);
    return blake3Hex(whole);
  };
  const base = new URL(`${indexBase.replace(/\/$/, "")}/`, location.href);
  let index = null;
  try {
    index = await openIndex(base, policy, { hash, fetch: hubFetch });
    checks.add("record", "index root signature", "pass", `signed by ${policy.identity}, logged at ${index.facts.signedAt}`);
    const record = await readRecord(index, repo);
    const entry = index.root.records.find((r) => r.repo.toLowerCase() === repo.toLowerCase());
    checks.add("record", "hub record matches the index root", "pass", `record blake3 ${entry.blake3}`);
    if (hubFetch.offline?.length) checks.add("record", "hub index read from this browser, checked again", "pass", `the network is off: ${hubFetch.offline.length} files kept on an earlier visit, the root proved by its signature and the record by its BLAKE3 as a download is`);
    return record;
  } catch (error) {
    if (!(error instanceof HubClientError)) throw error;
    const subject = index ? "hub record matches the index root" : "index root signature";
    checks.add("record", subject, "fail", `${error.code}: ${error.message}`);
    throw new Error(`${error.code}: ${error.message}`);
  }
}

// Every tensor of a model's graph and data files, with its BLAKE3 address,
// from the hub's UOR index (`data/uor` beside `data/index`): the shard the
// hub's own rule names (hub/uorkey.mjs), checked against the BLAKE3 the UOR
// root gives it, and the row for this record's revision and file. The index
// is not signed; an address in it is only where to look. What vouches for
// the bytes is the record's whole-file BLAKE3, checked by the importer, and
// decision 28 then asks that every listed address is the one the importer
// derived. A model the index does not list is refused by name: no address is
// invented in the browser (Q_ENGINE_PLAN.md, answered 2026-11-18).
// Answers a Map from each file's path to { size, blake3, tensors }.
export async function tensorManifest({ indexBase, record, paths, checks, fetch: hubFetch = keptFetch() }) {
  const { blake3Hex } = await import("../../vendor/prism-code/src/blake3.mjs");
  const base = new URL(`${indexBase.replace(/\/$/, "")}/../uor/`, location.href);
  const refuse = (why) => {
    checks.add("manifest", "tensor addresses from the hub's UOR index", "fail", why);
    throw new Error(`no tensor manifest: ${why}`);
  };
  const json = async (url, what) => {
    let response;
    try {
      response = await hubFetch(url);
    } catch (error) {
      return refuse(`${what}: ${error.message}`);
    }
    if (!response.ok) return refuse(`${what}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const root = JSON.parse(text.decode(await json(new URL("root.json", base), "the UOR index root")));
  const shardPath = await uorShardOf(record.repo, root.shardBits);
  const listed = root.shards?.find((s) => s.path === shardPath);
  if (!listed) refuse(`the hub's UOR index lists no tensors for ${record.repo}: it has no ${shardPath}`);
  const bytes = await json(new URL(shardPath, base), shardPath);
  if (blake3Hex(bytes) !== listed.blake3) refuse(`${shardPath} is not the bytes the UOR root names`);
  const rows = JSON.parse(text.decode(bytes));
  const manifest = new Map();
  for (const path of paths) {
    const file = record.files.find((f) => f.path === path);
    const row = rows.find((r) => r.repo.toLowerCase() === record.repo.toLowerCase() && r.revision === record.revision && r.files?.some((f) => f.path === path));
    if (!row) refuse(`the hub's UOR index lists no tensors for ${record.repo} ${path} at ${record.revision}`);
    const named = row.files.find((f) => f.path === path);
    if (named.blake3 !== file?.blake3 || named.size !== file?.size) refuse(`the hub's UOR index row for ${path} names another file than the record does`);
    const tensors = row.tensors.filter((t) => t.file === path).map((t) => ({ name: t.name, offset: t.offset, length: t.length, address: t.address }));
    if (!tensors.length || tensors.some((t) => !/^blake3:[0-9a-f]{64}$/.test(t.address))) refuse(`the hub's UOR index row for ${path} gives no BLAKE3 address to some tensor`);
    manifest.set(path, { size: file.size, blake3: file.blake3, tensors });
  }
  const count = [...manifest.values()].reduce((n, m) => n + m.tensors.length, 0);
  checks.add("manifest", "tensor addresses from the hub's UOR index", "pass", `${count} tensors in ${manifest.size} files, from ${shardPath}`);
  return manifest;
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
    entries.push({ index: i, name, fmt: shape.get(name).fmt, ...at, length });
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
