// An encoder on the page's integer engine: a sentence embedding or a
// classifier's label from the module's hologram-ai-cpu-encode (EN-62, EN-63).
//
// Why: row 9 of milestone 2 runs one embedding model and one classifier on
// the integer engine, so their answers are the same bits on every machine
// and equal the CPU reference's. The page computes no number of its own: it
// places each weight by the node that reads it (encoderRoles, GP-17), hands
// the module the bytes, and reads back binary32 bits and a label the module
// chose by comparing those bits as integers.

import { encoderRoles } from "./onnx-roles.mjs";
import { openWorker, placed } from "./integer.mjs";

const KINDS = { q4f16: 0, q4: 1, f16: 2, f32: 3, bf16: 4 };
const le = (n) => new Uint8Array(new Uint32Array([n]).buffer);
const words = (bytes) => new Uint32Array(bytes.slice().buffer);

// `module` is the page's module after the graph was imported through it with
// onnx_start_encoder and the tokenizer read with wordpiece_read; `table` and
// `read` are the import's tensors and their bytes; `config` the text of
// config.json, whose numbers the module reads exactly.
export function createIntegerEncoder({ module: m, table, read, config }) {
  const parsed = JSON.parse(config);
  const { dims, weights, labels } = encoderRoles(table, parsed);
  const named = new TextEncoder();
  for (const [name, w] of weights) {
    if (!(w.fmt in KINDS)) throw new Error(`precision: ${name} is ${w.fmt}`);
    const [first, second] = w.parts ? w.parts.map(read) : [read(w.from), new Uint8Array(0)];
    const n = named.encode(name);
    const given = new Uint8Array(4 + n.length + 13 + first.length + second.length);
    let at = 0;
    const put = (bytes) => { given.set(bytes, at); at += bytes.length; };
    put(le(n.length)); put(n); put([KINDS[w.fmt]]); put(le(w.N)); put(le(w.K)); put(le(first.length)); put(first); put(second);
    m.callWith("encode_tensor", given);
  }
  m.callWith("encode_open", named.encode(config));
  const ids = (text) => words(m.callWith("wordpiece_encode", named.encode(text)));
  const run = (call, tokens) => words(m.callWith(call, new Uint8Array(tokens.buffer, tokens.byteOffset, tokens.byteLength)));
  return {
    dims,
    labels,
    ids,
    // The sentence's embedding at unit length: its binary32 bits, and the
    // same bytes viewed as a Float32Array, as transformers.js answers.
    embed(text) {
      const tokens = ids(text);
      const bits = run("encode_embed", tokens);
      return { ids: tokens, bits, data: new Float32Array(bits.buffer) };
    },
    // The label the module chose and every label's score, bits and view.
    classify(text) {
      const tokens = ids(text);
      const said = run("encode_classify", tokens);
      const bits = said.slice(1);
      const index = said[0];
      return { ids: tokens, index, label: labels?.[index] ?? String(index), bits, scores: new Float32Array(bits.buffer) };
    },
  };
}

// The encoder in the integer worker (integer.mjs says why), answering what
// createIntegerEncoder answers, each call awaited.
export async function startIntegerEncoder({ module: m, table, read, config }) {
  const tensors = placed(table, read);
  const { ask, opened } = await openWorker(m, { engine: "encoder", table, tensors, config }, tensors.map(([, bytes]) => bytes.buffer));
  const call = async (method, text) => (await ask({ kind: "call", method, args: [text] })).value;
  return {
    dims: opened.dims,
    labels: opened.labels,
    ids: (text) => call("ids", text),
    embed: (text) => call("embed", text),
    classify: (text) => call("classify", text),
  };
}
