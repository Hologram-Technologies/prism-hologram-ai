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
//
// Decision 80 first (GP-50): the bytes fetched are compiled only if their
// BLAKE3 is the engine address the kernel seals, so a module the build did not
// make, a stale cache's or one altered on the way, is refused by name before
// any of it runs. `ask` is the page's asker; there is no way in without it.
// The integer worker is handed the module compiled here, never bytes of its own.
export async function openModule(url, ask) {
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const { blake3 } = await import("../../vendor/prism-code/src/blake3.mjs");
  await ask("engine", "engine.wasm is the engine the kernel seals (decision 80)", join(new Uint8Array([80]), blake3(bytes)), "sealed");
  return moduleOf(await WebAssembly.compile(bytes));
}

// The module's calls on a fresh instance of `compiled`. Given `memory`, the
// bytes of another instance's linear memory taken between calls, the new
// instance starts as that one stands: the same graph read, the same heap.
// Between calls the stack pointer is back at its start, so the memory is all
// of the module's state; the integer worker (integer-worker.mjs) runs the
// page's module this way off the page's thread.
export function moduleOf(compiled, memory = null) {
  // The module reaches the kernel through one import, `kernel.ask`: the
  // request at `at`, `len` bytes, and the answer written at `out`, at most
  // `cap` bytes; it answers the answer's whole length, so a longer one is cut
  // and refused. Until setKernel names the kernel, every question is answered
  // with nothing, which the importer refuses.
  let kernel = null;
  let x = null;
  // A wasm32 address or length is an unsigned 32-bit number, and reaches
  // JavaScript as a signed one: past 2 GiB of memory it is negative, and a
  // view or a slice at it reads other bytes (GP-61). Every address is read
  // unsigned where it enters.
  const u = (n) => n >>> 0;
  const imports = {
    kernel: {
      ask: (at, len, out, cap) => {
        [at, len, out, cap] = [u(at), u(len), u(out), u(cap)];
        if (!kernel) return 0;
        const answer = kernel(new Uint8Array(x.memory.buffer).slice(at, at + len));
        new Uint8Array(x.memory.buffer).set(answer.subarray(0, Math.min(answer.length, cap)), out);
        return answer.length;
      },
      // A product handed to the engine's helpers (GP-52), once
      // setProducts names them; until then every product is refused.
      product: (number, at, len, out, cap) => (products ? products(number, u(at), u(len), u(out), u(cap)) : 1),
    },
  };
  let products = null;
  x = new WebAssembly.Instance(compiled, imports).exports;
  if (memory) {
    const pages = memory.byteLength / 65536 - x.memory.buffer.byteLength / 65536;
    if (pages > 0) x.memory.grow(pages);
    new Uint8Array(x.memory.buffer).set(memory);
  }
  const result = () => {
    const at = u(x.result_at());
    return new Uint8Array(x.memory.buffer).slice(at, at + u(x.result_len()));
  };
  const check = (name, status) => {
    if (status !== 0) throw new Error(`${name}: ${text.decode(result())}`);
    return result();
  };
  const withBytes = (bytes, run) => {
    const at = u(x.allocate(bytes.length));
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
  // This instance's memory as it stands, for `moduleOf(compiled, memory)`.
  const snapshot = () => new Uint8Array(x.memory.buffer).slice();
  const setProducts = (answer) => {
    products = answer;
  };
  return { x, call, callWith, setKernel, setProducts, compiled, snapshot };
}

const join = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};

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
    if (tensors.some((t) => !/^blake3:[0-9a-f]{64}$/.test(t.address))) refuse(`the hub's UOR index row for ${path} gives no BLAKE3 address to some tensor`);
    manifest.set(path, { size: file.size, blake3: file.blake3, tensors });
  }
  // A graph may hold no tensor of its own, every one in its data files, as
  // Llama 3.2 1B's export keeps them (GP-57); a model none of whose files
  // lists a tensor is refused.
  const count = [...manifest.values()].reduce((n, m) => n + m.tensors.length, 0);
  if (!count) refuse(`the hub's UOR index lists no tensor in ${paths.join(" or ")}`);
  checks.add("manifest", "tensor addresses from the hub's UOR index", "pass", `${count} tensors in ${manifest.size} files, from ${shardPath}`);
  return manifest;
}
