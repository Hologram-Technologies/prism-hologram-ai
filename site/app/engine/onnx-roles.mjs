// What each tensor of an ONNX export is to the engine. The importer reads the
// graph and gives its nodes (operator, name, the tensors each reads) and its
// tensors (type and shape); this module names the engine role of every weight
// by the node that reads it, checks every shape against the model's config,
// and refuses by name what it cannot place. It reads no tensor bytes.
//
// Node names, not tensor names, carry the role: Qwen2.5's export names its
// weights onnx::MatMul_9333 and the like, while the node reading one is
// /model/layers.0/self_attn/q_proj/MatMul_Q4. Read on 2026-11-16 from
// onnx-community's q4f16 exports of Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B.

import { llama3, plain } from "./yarn.mjs";

const LAYER = String.raw`^/model/layers\.(\d+)/`;

// [operator, name pattern, engine role of the layer (or of the model when the
// pattern has no layer), the input the role is read from].
const RULES = [
  ["Gather", /^\/model\/embed_tokens\/Gather$/, () => "embed", 0],
  ["MatMulNBits", new RegExp(LAYER + String.raw`(?:self_attn|attn)/([qkvo])_proj/MatMul_Q4$`), (m) => `w${m[2]}`, 1],
  ["MatMulNBits", new RegExp(LAYER + String.raw`mlp/(gate|up|down)_proj/MatMul_Q4$`), (m) => `w_${m[2]}`, 1],
  ["MatMulNBits", /^\/lm_head\/MatMul_Q4$/, () => "lm_head", 1],
  ["Add", new RegExp(LAYER + String.raw`self_attn/([qkv])_proj/Add$`), (m) => `b${m[2]}`, null],
  ["SimplifiedLayerNormalization", new RegExp(LAYER + String.raw`attn/([qk])_norm/SimplifiedLayerNormalization$`), (m) => `${m[2]}_norm`, null],
  ["SimplifiedLayerNormalization", new RegExp(LAYER + String.raw`input_layernorm/LayerNorm$`), () => "attn_norm", null],
  ["SkipSimplifiedLayerNormalization", new RegExp(LAYER + String.raw`input_layernorm/SkipLayerNorm$`), () => "attn_norm", null],
  ["Mul", new RegExp(LAYER + String.raw`input_layernorm/Mul_1$`), () => "attn_norm", null],
  ["SkipSimplifiedLayerNormalization", new RegExp(LAYER + String.raw`post_attention_layernorm/SkipLayerNorm$`), () => "ffn_norm", null],
  ["Mul", new RegExp(LAYER + String.raw`post_attention_layernorm/Mul_1$`), () => "ffn_norm", null],
  // The fused export puts the final norm after the last layer, at index n.
  ["SkipSimplifiedLayerNormalization", new RegExp(LAYER + String.raw`final_norm_layernorm/SkipLayerNorm$`), () => "final_norm", null],
  ["Mul", /^\/model\/norm\/Mul_1$/, () => "final_norm", null],
];

// Operators that read a weight; one of these that no rule places is refused,
// so no weight is silently left out.
const WEIGHT_READERS = new Set(["MatMulNBits", "MatMul", "Gather", "Gemm"]);

const FAMILIES = new Set(["qwen2", "qwen3", "llama"]);

// The rotary parameters a model may use, in the order they are tried against
// its own cos and sin cache. A config that states its scaling is taken at its
// word; Llama 3.2's export states none (its cache is Llama 3 scaling at factor
// 32), so for a Llama config without one, Llama 3's published settings and
// plain frequencies are candidates, and the cache decides.
export function ropeCandidates(config) {
  const s = config.rope_scaling;
  if (s && (s.rope_type ?? s.type) === "llama3") {
    return [{ kind: "llama3", llama3: { factor: s.factor, low: s.low_freq_factor, high: s.high_freq_factor, orig: s.original_max_position_embeddings } }];
  }
  if (s) return [];
  const plain = { kind: "plain" };
  if (config.model_type !== "llama") return [plain];
  const orig = 8192;
  return [
    { kind: "llama3", llama3: { factor: 32, low: 1, high: 4, orig } },
    { kind: "llama3", llama3: { factor: 8, low: 1, high: 4, orig } },
    plain,
  ];
}

// Which candidate the export's own cache rows agree with, as the dims the
// engine takes. rows is a list of { position, cos, sin }, each holding one
// number per pair; Llama 3.2's f16 cache agrees with its parameters within
// 2.4e-4 at positions up to 8191 and with plain frequencies only to 1.97.
export function confirmRope(candidates, { hd, rope_base: base }, rows, tolerance = 2e-3) {
  if (!rows.length) throw new Error("rope: no rows of the cos and sin cache to check against");
  for (const candidate of candidates) {
    const f = candidate.kind === "llama3" ? llama3({ hd, base, ...candidate.llama3 }).freqs : plain({ hd, base }).freqs;
    let worst = 0;
    for (const { position, cos, sin } of rows) {
      for (let i = 0; i < f.length; i++) {
        worst = Math.max(worst, Math.abs(cos[i] - Math.cos(position * f[i])), Math.abs(sin[i] - Math.sin(position * f[i])));
      }
    }
    if (worst <= tolerance) return { dims: candidate.kind === "llama3" ? { llama3: candidate.llama3 } : {}, worst };
  }
  throw new Error("rope: no rotary parameters agree with the model's cos and sin cache");
}

const product = (dims) => dims.reduce((a, b) => a * b, 1);
const FORMAT = { f16: "f16", f32: "f32" };

// The engine's model from an export's graph: dims, and each engine tensor as
// the names of the export's tensors it is read from, in order.
export function roles({ nodes, tensors }, config) {
  if (!FAMILIES.has(config.model_type)) throw new Error(`family: ${config.model_type} is not one this engine runs`);
  const d = config.hidden_size;
  const heads = config.num_attention_heads;
  const kv = config.num_key_value_heads ?? heads;
  const hd = config.head_dim ?? d / heads;
  const dims = {
    d, n_heads: heads, n_kv_heads: kv, hd, ff: config.intermediate_size, vocab: config.vocab_size,
    n_layers: config.num_hidden_layers, rope_base: config.rope_theta, eps: config.rms_norm_eps,
  };
  const L = dims.n_layers;
  const found = new Map();
  const place = (name, value, node) => {
    if (found.has(name)) throw new Error(`duplicate: ${name} is read by ${found.get(name).node} and ${node}`);
    found.set(name, { ...value, node });
  };
  const tensor = (name, node) => {
    const t = tensors[name];
    if (!t) throw new Error(`missing: ${node} reads ${name}, which the graph does not hold`);
    return { dtype: t[0], dims: t[1] };
  };
  const want = (what, got, expected) => {
    if (got !== expected) throw new Error(`shape: ${what} is ${got}, the config gives ${expected}`);
  };
  const width = { wq: [heads * hd, d], wk: [kv * hd, d], wv: [kv * hd, d], wo: [d, heads * hd], w_gate: [dims.ff, d], w_up: [dims.ff, d], w_down: [d, dims.ff], lm_head: [dims.vocab, d] };
  const vectorLength = { bq: heads * hd, bk: kv * hd, bv: kv * hd, q_norm: hd, k_norm: hd, attn_norm: d, ffn_norm: d, final_norm: d };

  for (const [op, name, inputs, attr = {}] of nodes) {
    if (op === "GroupQueryAttention") {
      want(`${name} heads`, attr.num_heads, heads);
      want(`${name} key and value heads`, attr.kv_num_heads, kv);
      if (attr.rotary_interleaved) throw new Error(`rope: ${name} rotates interleaved pairs; this engine rotates halves`);
      continue;
    }
    if (op === "RotaryEmbedding") {
      if (attr.interleaved) throw new Error(`rope: ${name} rotates interleaved pairs; this engine rotates halves`);
      continue;
    }
    const rule = RULES.find(([o, pattern]) => o === op && pattern.test(name));
    if (!rule) {
      if (WEIGHT_READERS.has(op) && inputs.some(Boolean)) throw new Error(`unplaced: ${op} ${name} reads a weight no engine role takes`);
      continue;
    }
    const m = name.match(rule[1]);
    const role = rule[2](m);
    const layer = m[1] === undefined || !/^\d+$/.test(m[1]) ? null : Number(m[1]);
    // The fused final norm sits at layer index n; any other layer must exist.
    if (layer !== null && role !== "final_norm" && layer >= L) throw new Error(`layer: ${name} is past the config's ${L} layers`);
    if (role === "final_norm" && layer !== null && layer !== L) throw new Error(`layer: ${name} is the final norm at layer ${layer}, not ${L}`);
    const engineName = layer === null || role === "final_norm" ? role : `l${layer}.${role}`;

    if (op === "MatMulNBits") {
      if (attr.bits !== 4 || attr.block_size !== 32) throw new Error(`precision: ${name} is ${attr.bits} bits in blocks of ${attr.block_size}; this engine runs 4 in 32`);
      if (inputs[3]) throw new Error(`precision: ${name} carries zero points; this engine runs the default 8`);
      const [N, K] = width[role];
      want(`${name} rows`, attr.N, N);
      want(`${name} columns`, attr.K, K);
      const codes = tensor(inputs[1], name);
      const scales = tensor(inputs[2], name);
      want(`${name} code bytes`, product(codes.dims), (N * K) / 2);
      want(`${name} scales`, product(scales.dims), (N * K) / 32);
      const fmt = { f16: "q4f16", f32: "q4" }[scales.dtype];
      if (!fmt) throw new Error(`precision: ${name} has ${scales.dtype} scales`);
      place(engineName, { fmt, N, K, parts: [inputs[1], inputs[2]] }, name);
    } else if (op === "Gather") {
      const t = tensor(inputs[0], name);
      if (!FORMAT[t.dtype]) throw new Error(`precision: ${name} is ${t.dtype}; this engine runs an f32 or f16 embedding`);
      want(`${name} rows`, t.dims[0], dims.vocab);
      want(`${name} width`, t.dims[1], d);
      place(engineName, { fmt: t.dtype, N: dims.vocab, K: d, from: inputs[0] }, name);
    } else {
      const held = inputs.filter(Boolean);
      if (held.length !== 1) throw new Error(`unplaced: ${name} reads ${held.length} weights, not one`);
      const t = tensor(held[0], name);
      if (!FORMAT[t.dtype]) throw new Error(`precision: ${name} is ${t.dtype}; a vector is f32 or f16`);
      want(`${name} length`, product(t.dims), vectorLength[role]);
      place(engineName, { fmt: t.dtype, N: 1, K: product(t.dims), from: held[0] }, name);
    }
  }

  // Every role the family needs, exactly once.
  const needed = ["embed", "final_norm"];
  for (let l = 0; l < L; l++) {
    for (const r of ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w_gate", "w_up", "w_down"]) needed.push(`l${l}.${r}`);
    if (config.model_type === "qwen2") for (const r of ["bq", "bk", "bv"]) needed.push(`l${l}.${r}`);
    if (config.model_type === "qwen3") for (const r of ["q_norm", "k_norm"]) needed.push(`l${l}.${r}`);
  }
  for (const name of needed) if (!found.has(name)) throw new Error(`missing: the graph has no ${name}`);
  const extra = [...found.keys()].filter((n) => !needed.includes(n) && n !== "lm_head");
  if (extra.length) throw new Error(`unplaced: ${extra[0]} is not part of a ${config.model_type} model`);
  if (!found.has("lm_head") && config.tie_word_embeddings === false) throw new Error("missing: the config unties the head and the graph has no lm_head");
  return { dims, weights: found };
}

// The engine's tensors for a placed model: read(name) gives the bytes of one
// of the export's tensors, as the importer admitted them. A MatMulNBits weight
// stays as its two tensors; nothing is copied or repacked.
export function engineTensors(weights, read) {
  const out = new Map();
  for (const [name, w] of weights) {
    out.set(name, w.parts ? { fmt: w.fmt, N: w.N, K: w.K, parts: w.parts.map(read) } : { fmt: w.fmt, N: w.N, K: w.K, bytes: read(w.from) });
  }
  return out;
}
