// Row 9 of milestone 2 on the real exports: Xenova's all-MiniLM-L6-v2 and
// DistilBERT SST-2 through a deployed site's own encoder pipeline, run in Node
// on the page's module, and the same weights in the double-precision float
// reference (crates/conformance/tests/gpu/encoder-reference.mjs).
//
//   node encoders.mjs <site> <cache> <out dir>
//
// Why each piece is taken as it is:
//   the weights are the page's own: the pipeline places them (encoderRoles,
//   GP-17) and hands them to its integer worker; this harness copies that
//   hand-over, so nothing here places a weight;
//   out/<key>/given.bin holds each weight as the module's encode_tensor reads
//   it, and config.txt the config's text, so the native build of the module
//   (engine_row9_real_exports, cargo test -- --ignored) runs them and must
//   answer the same bits;
//   the float reference reads the same bytes decoded exactly to numbers.
// Prints one JSON line per model.

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { installShims } from "./node-shims.mjs";

export const TEXTS = [
  "The weather is lovely today.",
  "A content address names a piece of data by a hash of its bytes rather than by where it is kept.",
  "I did not enjoy this film at all; the plot was dull and the acting was worse.",
  "This is the best meal I have had in years, and the service was wonderful too.",
  "Two copies of the same file fetched from two servers have the same address, and a copy that differs in one byte has another.",
];
export const MODELS = {
  minilm: { repo: "Xenova/all-MiniLM-L6-v2", task: "feature-extraction" },
  sst2: { repo: "Xenova/distilbert-base-uncased-finetuned-sst-2-english", task: "text-classification" },
};

const [site, cache, out] = process.argv.slice(2);
if (!site || !cache || !out) throw new Error("usage: node encoders.mjs <site> <cache> <out dir>");
const app = pathToFileURL(join(site, "app") + "/").href;
installShims({ cacheDir: cache, href: app });

// The page's hand-over to its integer worker: table, placed tensors and the
// config's text, copied before the buffers are transferred.
let handed = null;
const PageWorker = globalThis.Worker;
globalThis.Worker = class extends PageWorker {
  postMessage(message, transfer) {
    if (message?.engine === "encoder") handed = { table: message.table, config: message.config, tensors: message.tensors.map(([n, b]) => [n, new Uint8Array(b).slice()]) };
    super.postMessage(message, transfer);
  }
};

const { pipeline } = await import(new URL("engine/pipeline.mjs", app).href);
const { checklist } = await import(new URL("engine/loader.mjs", app).href);
const { encoderRoles } = await import(new URL("engine/onnx-roles.mjs", app).href);
const { referenceEncoder } = await import(pathToFileURL(join(import.meta.dirname, "..", "tests", "gpu", "encoder-reference.mjs")).href);

const KINDS = { q4f16: 0, q4: 1, f16: 2, f32: 3, bf16: 4 };
const le = (n) => new Uint8Array(new Uint32Array([n]).buffer);
const hex = (words) => Array.from(new Uint8Array(new Uint32Array(words).buffer), (b) => b.toString(16).padStart(2, "0")).join("");
const half = (h) => {
  const s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
};
const numbers = (bytes, fmt) => {
  if (fmt === "f16") return Float64Array.from(new Uint16Array(bytes.slice().buffer), half);
  if (fmt === "f32") return Float64Array.from(new Float32Array(bytes.slice().buffer));
  throw new Error(`float reference: ${fmt} is not read here`);
};

for (const [key, { repo, task }] of Object.entries(MODELS)) {
  handed = null;
  const checks = checklist();
  const run = await pipeline(task, repo, { checks });
  const cases = [];
  for (const text of TEXTS) {
    const said = await run(text);
    cases.push({ text, ids: Array.from(said.ids), label: said.index ?? null, bits: hex(said.bits) });
  }
  await run.dispose?.();
  if (!handed) throw new Error(`${key}: the page handed no encoder to its worker`);

  // The page's weights, as the module's encode_tensor reads each one.
  const config = JSON.parse(handed.config);
  const { dims, weights } = encoderRoles(handed.table, config);
  const bytes = new Map(handed.tensors);
  const named = new TextEncoder();
  const records = [];
  const floats = new Map();
  for (const [name, w] of weights) {
    const [first, second] = w.parts ? w.parts.map((p) => bytes.get(p)) : [bytes.get(w.from), new Uint8Array(0)];
    const n = named.encode(name);
    const record = new Uint8Array(4 + n.length + 13 + first.length + second.length);
    let at = 0;
    const put = (b) => { record.set(b, at); at += b.length; };
    put(le(n.length)); put(n); put([KINDS[w.fmt]]); put(le(w.N)); put(le(w.K)); put(le(first.length)); put(first); put(second);
    records.push(le(record.length), record);
    if (w.parts) {
      // MatMulNBits: two 4-bit codes a byte, low nibble first, zero point 8,
      // one scale per 32 codes of a row.
      const codes = new Int8Array(2 * first.length);
      first.forEach((b, i) => { codes[2 * i] = (b & 15) - 8; codes[2 * i + 1] = (b >> 4) - 8; });
      floats.set(name, { codes, scales: numbers(second, w.fmt === "q4f16" ? "f16" : "f32") });
    } else {
      floats.set(name, { values: numbers(first, w.fmt) });
    }
  }
  const dir = join(out, key);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "given.bin"), Buffer.concat(records));
  await writeFile(join(dir, "config.txt"), handed.config);

  // The float reference on the same weights and ids.
  const reference = referenceEncoder({ dims: { ...dims, eps: config.layer_norm_eps ?? 1e-12 }, tensors: floats });
  for (const c of cases) {
    const got = new Float32Array(Uint8Array.from(c.bits.match(/../g), (h) => parseInt(h, 16)).buffer);
    if (task === "feature-extraction") {
      const want = reference.embed(c.ids);
      let dot = 0, gg = 0, ww = 0, gap = 0;
      for (let i = 0; i < want.length; i++) { dot += got[i] * want[i]; gg += got[i] * got[i]; ww += want[i] * want[i]; gap = Math.max(gap, Math.abs(got[i] - want[i])); }
      c.float = { largestGap: gap, cosine: dot / Math.sqrt(gg * ww) };
    } else {
      const { logits } = reference.classify(c.ids);
      const top = logits.indexOf(Math.max(...logits));
      c.float = { label: top, sameLabel: top === c.label, largestScoreGap: Math.max(...Array.from(logits, (v, i) => Math.abs(got[i] - v))) };
    }
  }
  const record = { key, repo, task, revision: checks.items.find((c) => c.step === "record")?.detail ?? null, dims, labels: config.id2label ?? null, cases, checks: checks.items.map((c) => `${c.step}: ${c.verdict}: ${c.detail}`) };
  await writeFile(join(dir, "cases.json"), JSON.stringify(record, null, 1) + "\n");
  console.log(JSON.stringify({ key, tokens: cases.map((c) => c.ids.length), float: cases.map((c) => c.float) }));
}
process.exit(0);
