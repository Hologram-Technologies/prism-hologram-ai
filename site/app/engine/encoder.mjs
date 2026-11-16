// The WebGPU engine for BERT-style encoders: a sentence in, its hidden states
// out, and from them a sentence embedding (mean over tokens, unit length, as
// sentence-transformers and transformers.js pool all-MiniLM-L6-v2) or a
// classifier's label scores (DistilBERT's: the first token through a dense
// layer, ReLU and a second dense layer). Decisions about what may be loaded and
// run are the kernel's; this module only computes.
//
// Tensors, by the names this engine reads:
//   word, position, type (optional)  dense f32 or f16 tables, [rows][d]
//   emb_norm, emb_norm_b              the embedding LayerNorm
//   l<i>.wq .wk .wv .wo .w1 .w2       q4 or q4f16 matrices, as in gpu.mjs
//   l<i>.bq .bk .bv .bo .b1 .b2       their biases
//   l<i>.attn_norm(_b) .out_norm(_b)  the LayerNorms after attention and FFN
//   pre, pre_b, cls, cls_b            a classifier's head (optional)

import { matvecDense } from "./wgsl-dense.mjs";
import { ADD_BIAS } from "./wgsl-q4.mjs";
import { ATTENTION_ALL, GELU, LAYER_NORM_ROWS, RELU, embedSum, matmulRowsQ4 } from "./wgsl-encoder.mjs";

const U = typeof GPUBufferUsage === "undefined" ? {} : GPUBufferUsage;
const f32bits = (value) => new Uint32Array(new Float32Array([value]).buffer)[0];
const half = (h) => {
  const e = (h >> 10) & 31;
  const m = h & 1023;
  const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1024 + m) * 2 ** (e - 25);
  return h & 0x8000 ? -v : v;
};
const SIZE = { f32: 4, f16: 2 };
const Q4 = { q4: 4, q4f16: 2 };

export async function createEncoder({ device, dims, tensors, capacity = 128 }) {
  const { d, n_heads: heads, ff, n_layers: layers, eps } = dims;
  const hd = d / heads;
  if (capacity > 512) throw new Error(`capacity ${capacity}: an encoder attends over at most 512 rows`);
  const storage = (size) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const upload = (array) => {
    const buffer = storage(array.byteLength);
    if (array.byteLength % 4) array = Uint8Array.from({ length: Math.ceil(array.byteLength / 4) * 4 }, (_, i) => (i < array.byteLength ? array[i] : 0));
    device.queue.writeBuffer(buffer, 0, array.buffer, array.byteOffset, array.byteLength);
    return buffer;
  };
  const uniform = (values) => {
    const buffer = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
    return buffer;
  };
  const tensor = (name) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`model has no tensor ${name}`);
    return t;
  };
  const vector = (name, n) => {
    const t = tensor(name);
    if (t.K !== n) throw new Error(`${name} has ${t.K} numbers, the model needs ${n}`);
    if (t.fmt === "f16") {
      const h = new Uint16Array(t.bytes.byteLength / 2);
      new Uint8Array(h.buffer).set(t.bytes);
      return upload(Float32Array.from(h, half));
    }
    if (t.fmt !== "f32") throw new Error(`${name} is ${t.fmt}, not f32 or f16`);
    return upload(new Float32Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 4));
  };
  const matrix = (name, N, K) => {
    const t = tensor(name);
    if (!Q4[t.fmt]) throw new Error(`${name} is ${t.fmt}; this encoder runs q4 and q4f16 matrices`);
    if (t.N !== N || t.K !== K) throw new Error(`${name} is ${t.N} by ${t.K}, the model needs ${N} by ${K}`);
    const [codes, scales] = t.parts ?? [t.bytes.subarray(0, (N * K) / 2), t.bytes.subarray((N * K) / 2)];
    if (codes.byteLength !== (N * K) / 2 || scales.byteLength !== (N * K * Q4[t.fmt]) / 32) throw new Error(`${name} holds the wrong number of bytes for ${t.fmt}`);
    return { fmt: t.fmt, N, K, buffers: [upload(codes), upload(scales)] };
  };
  const table = (name, rows) => {
    const t = tensor(name);
    if (!SIZE[t.fmt]) throw new Error(`${name} is ${t.fmt}; a table is f32 or f16`);
    if (t.K !== d || t.N < rows) throw new Error(`${name} is ${t.N} by ${t.K}, the model needs ${rows} by ${d}`);
    if (t.bytes.byteLength !== t.N * t.K * SIZE[t.fmt]) throw new Error(`${name} holds the wrong number of bytes for ${t.fmt}`);
    return { fmt: t.fmt, buffer: upload(t.bytes) };
  };

  const word = table("word", dims.vocab);
  const position = table("position", capacity);
  const kind = tensors.has("type") ? table("type", 1) : null;
  if (position.fmt !== word.fmt || (kind && kind.fmt !== word.fmt)) throw new Error("the embedding tables differ in format");
  const embNorm = [vector("emb_norm", d), vector("emb_norm_b", d)];
  const layer = [];
  for (let l = 0; l < layers; l++) {
    const p = `l${l}.`;
    layer.push({
      wq: matrix(`${p}wq`, d, d), bq: vector(`${p}bq`, d),
      wk: matrix(`${p}wk`, d, d), bk: vector(`${p}bk`, d),
      wv: matrix(`${p}wv`, d, d), bv: vector(`${p}bv`, d),
      wo: matrix(`${p}wo`, d, d), bo: vector(`${p}bo`, d),
      attnNorm: [vector(`${p}attn_norm`, d), vector(`${p}attn_norm_b`, d)],
      w1: matrix(`${p}w1`, ff, d), b1: vector(`${p}b1`, ff),
      w2: matrix(`${p}w2`, d, ff), b2: vector(`${p}b2`, d),
      outNorm: [vector(`${p}out_norm`, d), vector(`${p}out_norm_b`, d)],
    });
  }
  let head = null;
  if (tensors.has("cls")) {
    const dense = (name, N) => {
      const t = tensor(name);
      if (!SIZE[t.fmt] || t.N !== N || t.K !== d || t.bytes.byteLength !== N * d * SIZE[t.fmt]) throw new Error(`${name} is not an f32 or f16 ${N} by ${d} matrix`);
      return { fmt: t.fmt, N, buffer: upload(t.bytes) };
    };
    const labels = tensor("cls").N;
    head = { pre: dense("pre", d), preB: vector("pre_b", d), cls: dense("cls", labels), clsB: vector("cls_b", labels), labels };
  }

  const B = {
    x: storage(capacity * d * 4), r: storage(capacity * d * 4),
    q: storage(capacity * d * 4), k: storage(capacity * d * 4), v: storage(capacity * d * 4),
    att: storage(capacity * d * 4), h: storage(capacity * ff * 4),
    tokens: storage(capacity * 4), zero: storage(16),
    pre: storage(d * 4), logits: storage(Math.max(4, (head?.labels ?? 1)) * 4),
  };
  const readHidden = device.createBuffer({ size: capacity * d * 4, usage: U.MAP_READ | U.COPY_DST });
  const readLogits = device.createBuffer({ size: Math.max(16, (head?.labels ?? 1) * 4), usage: U.MAP_READ | U.COPY_DST });

  const made = new Map();
  const pipeline = (code) => {
    if (!made.has(code)) made.set(code, device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }));
    return made.get(code);
  };
  const bind = (p, buffers) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });

  // Run a sentence of `rows` tokens; the hidden states are left in B.x.
  const encode = (encoder, rows) => {
    const pass = encoder.beginComputePass();
    const run = (code, buffers, x, y = 1) => {
      const p = pipeline(code);
      pass.setPipeline(p);
      pass.setBindGroup(0, bind(p, buffers));
      pass.dispatchWorkgroups(x, y);
    };
    const mm = (w, input, output) => run(matmulRowsQ4(w.fmt === "q4f16" ? "f16" : "f32"), [input, ...w.buffers, output, uniform([w.K, w.N, 0, 0])], w.N, rows);
    const bias = (x, b, n) => run(ADD_BIAS, [x, b, uniform([n, rows, 0, 0])], Math.ceil((n * rows) / 64));
    const norm = (x, r, [g, b], residual) => run(LAYER_NORM_ROWS, [x, r, g, b, uniform([d, rows, f32bits(eps), residual ? 1 : 0])], rows);
    run(embedSum(word.fmt), [word.buffer, position.buffer, kind ? kind.buffer : B.zero, B.x, B.tokens, uniform([d, rows, kind ? 1 : 0, 0])], Math.ceil((d * rows) / 64));
    norm(B.x, B.zero, embNorm, false);
    for (const L of layer) {
      mm(L.wq, B.x, B.q); bias(B.q, L.bq, d);
      mm(L.wk, B.x, B.k); bias(B.k, L.bk, d);
      mm(L.wv, B.x, B.v); bias(B.v, L.bv, d);
      run(ATTENTION_ALL, [B.q, B.k, B.v, B.att, uniform([heads, hd, rows, f32bits(1 / Math.sqrt(hd))])], heads, rows);
      mm(L.wo, B.att, B.r); bias(B.r, L.bo, d);
      norm(B.x, B.r, L.attnNorm, true);
      mm(L.w1, B.x, B.h); bias(B.h, L.b1, ff);
      run(GELU, [B.h, uniform([ff * rows, 0, 0, 0])], Math.ceil((ff * rows) / 64));
      mm(L.w2, B.h, B.r); bias(B.r, L.b2, d);
      norm(B.x, B.r, L.outNorm, true);
    }
    if (head) {
      // The first token's state: row 0 of B.x.
      run(matvecDense(head.pre.fmt), [B.x, head.pre.buffer, B.pre, uniform([d, d, 0, 0])], d);
      run(ADD_BIAS, [B.pre, head.preB, uniform([d, 1, 0, 0])], Math.ceil(d / 64));
      run(RELU, [B.pre, uniform([d, 0, 0, 0])], Math.ceil(d / 64));
      run(matvecDense(head.cls.fmt), [B.pre, head.cls.buffer, B.logits, uniform([d, head.labels, 0, 0])], head.labels);
      run(ADD_BIAS, [B.logits, head.clsB, uniform([head.labels, 1, 0, 0])], 1);
    }
    pass.end();
  };

  // Hidden states, [rows][d], for a sentence of token ids.
  async function hidden(tokens) {
    if (tokens.length < 1 || tokens.length > capacity) throw new Error(`a sentence of ${tokens.length} tokens; this encoder takes 1 to ${capacity}`);
    for (const t of tokens) if (!(t >= 0 && t < dims.vocab)) throw new Error(`token ${t} is outside the vocabulary of ${dims.vocab}`);
    device.queue.writeBuffer(B.tokens, 0, new Uint32Array(tokens));
    const encoder = device.createCommandEncoder();
    encode(encoder, tokens.length);
    encoder.copyBufferToBuffer(B.x, 0, readHidden, 0, tokens.length * d * 4);
    if (head) encoder.copyBufferToBuffer(B.logits, 0, readLogits, 0, head.labels * 4);
    device.queue.submit([encoder.finish()]);
    await readHidden.mapAsync(GPUMapMode.READ, 0, tokens.length * d * 4);
    const states = new Float32Array(readHidden.getMappedRange(0, tokens.length * d * 4).slice(0));
    readHidden.unmap();
    let logits = null;
    if (head) {
      await readLogits.mapAsync(GPUMapMode.READ, 0, head.labels * 4);
      logits = new Float32Array(readLogits.getMappedRange(0, head.labels * 4).slice(0));
      readLogits.unmap();
    }
    return { states, logits };
  }

  // The sentence embedding: the mean of the token states, scaled to length 1.
  async function embed(tokens) {
    const { states } = await hidden(tokens);
    const out = new Float64Array(d);
    for (let m = 0; m < tokens.length; m++) for (let k = 0; k < d; k++) out[k] += states[m * d + k];
    let length = 0;
    for (let k = 0; k < d; k++) length += (out[k] / tokens.length) ** 2;
    length = Math.sqrt(length);
    return Float32Array.from(out, (v) => v / tokens.length / length);
  }

  // A classifier's label scores and their probabilities.
  async function classify(tokens) {
    if (!head) throw new Error("this model has no classifier head");
    const { logits } = await hidden(tokens);
    const top = Math.max(...logits);
    const e = Array.from(logits, (v) => Math.exp(v - top));
    const sum = e.reduce((a, b) => a + b, 0);
    return { logits, probabilities: e.map((v) => v / sum) };
  }

  return { hidden, embed, classify, dims };
}
