// GP-03's fixture: a weight container in Bonsai's layout, built from the tiny
// generated model, with the hub record and index root that name it. Nothing
// here is downloaded; every byte is written from the formats' own rules.
//
// Container: "HOLK1\n", a u32 little-endian header length, a space-padded JSON
// header (manifest, tokenizer span, directory), the tokenizer (a GGUF header),
// then each buffer gzip'd and named by the BLAKE3 of its stored bytes. The
// output head names exactly the embedding's bytes, as Bonsai's does.

import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { blake3Hex } from "../../../../try/vendor/prism-code/src/blake3.mjs";
import { referenceModel, tinyModel, tinyQwen2 } from "./reference.mjs";
import { onnxFile } from "./onnx-write.mjs";

const text = new TextEncoder();
const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; };
const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
const str = (s) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);

// GPT-2's map from bytes to printable characters, which a byte-level BPE
// vocabulary is written in.
function byteAlphabet() {
  const keep = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const chars = new Map(keep.map((b) => [b, String.fromCodePoint(b)]));
  let extra = 0;
  for (let b = 0; b < 256; b++) if (!chars.has(b)) chars.set(b, String.fromCodePoint(256 + extra++));
  return Array.from({ length: 256 }, (_, b) => chars.get(b));
}

// A GGUF header holding a byte-level vocabulary of `size` tokens: the 256
// bytes, then pairs of them, as many as the model's vocabulary needs. No
// merges, so text encodes byte by byte; every id the model can pick decodes.
function gguf(size) {
  const alphabet = byteAlphabet();
  const tokens = [...alphabet];
  for (let i = 0; tokens.length < size; i++) tokens.push(alphabet[65 + (i % 26)] + alphabet[97 + Math.floor(i / 26)]);
  const entries = [
    ["general.architecture", 8, str("qwen3")],
    ["general.file_type", 4, u32(7)],
    ["tokenizer.ggml.model", 8, str("gpt2")],
    ["tokenizer.ggml.tokens", 9, Buffer.concat([u32(8), u64(tokens.length), ...tokens.map(str)])],
    ["tokenizer.ggml.merges", 9, Buffer.concat([u32(8), u64(0)])],
    ["tokenizer.ggml.add_space_prefix", 7, Buffer.from([0])],
  ];
  return Buffer.concat([
    Buffer.from("GGUF"), u32(3), u64(0), u64(entries.length),
    ...entries.map(([key, type, value]) => Buffer.concat([str(key), u32(type), value])),
  ]);
}

export function buildFixture() {
  const model = tinyModel(1);
  // The output head is the embedding, stored once.
  model.tensors.set("lm_head", model.tensors.get("embed"));
  const { dims } = model;
  const tokenizer = gguf(dims.vocab);
  const order = [...model.tensors.keys()].filter((n) => n !== "lm_head");
  const stored = order.map((name) => {
    const t = model.tensors.get(name);
    const bytes = gzipSync(Buffer.from(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength));
    return { name, t, bytes, kappa: `blake3:${blake3Hex(bytes)}` };
  });

  const manifest = {
    format: "holo-2bit/1", mode: "q1", model: "tiny", bits: 1, layout: "q1", tern: true,
    d: dims.d, n_heads: dims.n_heads, n_kv_heads: dims.n_kv_heads, ff: dims.ff, vocab: dims.vocab,
    n_layers: dims.n_layers, hd: dims.hd, rope_base: dims.rope_base, qk_norm: true, qk_norm_dim: dims.hd,
    tied: true, yarn: dims.yarn, root: "sha256:" + "0".repeat(64),
  };
  // The header's length is fixed before the offsets are known: room enough,
  // padded with spaces as Bonsai's is.
  const headerLength = 16384;
  let offset = 10 + headerLength;
  const tokenizerSpan = { off: offset, len: tokenizer.length };
  offset += tokenizer.length;
  const dir = [];
  for (const s of stored) {
    dir.push({ name: s.name, fmt: s.t.fmt, N: s.t.N, K: s.t.K, kappa: s.kappa, off: offset, len: s.bytes.length });
    offset += s.bytes.length;
  }
  const embed = dir.find((e) => e.name === "embed");
  dir.push({ ...embed, name: "lm_head" });
  const json = Buffer.from(JSON.stringify({ format: "holo-kappa/1", manifest, tokenizer: tokenizerSpan, dir }));
  if (json.length > headerLength) throw new Error("the fixture header outgrew its room");
  const container = Buffer.concat([
    Buffer.from("HOLK1\n"), u32(headerLength), json, Buffer.alloc(headerLength - json.length, 0x20),
    tokenizer, ...stored.map((s) => s.bytes),
  ]);
  const sha256 = createHash("sha256").update(container).digest("hex");
  return { model, container, sha256, dir, path: "tiny.holk", repo: "fixture/tiny", revision: "0".repeat(40) };
}

// The hub's record for the fixture and the index root that names the record.
export function hubFiles(fixture, { resolve, sha256 = fixture.sha256 }) {
  const record = Buffer.from(JSON.stringify({
    format: "hologram.hub.release/v4",
    repo: fixture.repo,
    revision: fixture.revision,
    origin: { kind: "fixture", resolve },
    files: [{ path: fixture.path, size: fixture.container.length, gitSha1: "0".repeat(40), sha256 }],
    sources: [],
    witnesses: [],
  }));
  const root = Buffer.from(JSON.stringify({
    format: "hologram.hub.index/v2",
    records: [{ repo: fixture.repo, revision: fixture.revision, sha256: createHash("sha256").update(record).digest("hex") }],
  }));
  return { record, root };
}


// The prompt the page frames "hi" in for a model whose vocabulary has no
// <think> token, as site/app/app.mjs writes it.
export const ONNX_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

// GP-19's fixture: GP-18's Qwen3-shaped model written as onnx-community's
// fused q4f16 export writes one, its config.json, generation_config.json and a
// byte-level tokenizer.json, and the hub record that names all four by their
// SHA-256. The model stops on an id the reference never picks in its first
// twenty steps after the prompt, so the page's first sixteen are comparable.
export function buildOnnxFixture() {
  // Seed 16: its first sixteen picks after the prompt are six different ids,
  // the best two logits never closer than 0.0235.
  const model = tinyQwen2(16, "onnx", "f16", "dense", { vectors: "f16", qkNorm: true });
  const { bytes: graph, config } = onnxFile(model, "qwen3");
  const alphabet = byteAlphabet();
  const vocab = {};
  alphabet.forEach((c, id) => { vocab[c] = id; });
  for (let i = 0; Object.keys(vocab).length < model.dims.vocab; i++) vocab[alphabet[65 + (i % 26)] + alphabet[97 + Math.floor(i / 26)]] = 256 + i;
  const tokenizer = Buffer.from(JSON.stringify({
    version: "1.0",
    added_tokens: [],
    normalizer: null,
    pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
    model: { type: "BPE", vocab, merges: [] },
  }));
  const prompt = Array.from(Buffer.from(ONNX_PROMPT, "utf8"));
  const reference = referenceModel(model);
  let next;
  for (const id of prompt) next = reference.step(id).token;
  const picked = [];
  for (let i = 0; i < 20; i++) {
    picked.push(next);
    next = reference.step(next).token;
  }
  let eos = model.dims.vocab - 1;
  while (picked.includes(eos)) eos--;
  const files = new Map([
    ["config.json", Buffer.from(JSON.stringify({ ...config, eos_token_id: eos }))],
    ["generation_config.json", Buffer.from(JSON.stringify({ eos_token_id: [eos] }))],
    ["tokenizer.json", tokenizer],
    ["onnx/model_q4f16.onnx", Buffer.from(graph)],
  ]);
  const sha = (b) => createHash("sha256").update(b).digest("hex");
  return { model, files, sha, prompt, expected: picked.slice(0, 16), eos, path: "onnx/model_q4f16.onnx", repo: "fixture/tiny-onnx", revision: "1".repeat(40) };
}

// The hub's record for the ONNX fixture, with a whole-file SHA-256 for each
// file (the graph's may be given wrong), and the index root that names it.
const gitBlob = (bytes) => createHash("sha1").update(Buffer.concat([Buffer.from(`blob ${bytes.length}\0`), bytes])).digest("hex");

export function onnxHubFiles(fixture, { resolve, sha256 = {}, gitOnly = [], gitSha1 = {} }) {
  const record = Buffer.from(JSON.stringify({
    format: "hologram.hub.release/v4",
    repo: fixture.repo,
    revision: fixture.revision,
    origin: { kind: "fixture", resolve },
    files: [...fixture.files].map(([path, bytes]) => gitOnly.includes(path)
      ? { path, size: bytes.length, gitSha1: gitSha1[path] ?? gitBlob(bytes) }
      : { path, size: bytes.length, gitSha1: "0".repeat(40), sha256: sha256[path] ?? fixture.sha(bytes) }),
    sources: [],
    witnesses: [],
  }));
  const root = Buffer.from(JSON.stringify({
    format: "hologram.hub.index/v2",
    records: [{ repo: fixture.repo, revision: fixture.revision, sha256: createHash("sha256").update(record).digest("hex") }],
  }));
  return { record, root };
}

export { text };
