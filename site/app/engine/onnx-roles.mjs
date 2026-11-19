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

// Encoders: BERT (all-MiniLM-L6-v2) and DistilBERT (its SST-2 classifier),
// read on 2026-11-16 from Xenova's q4f16 exports at 751bff3 and 0b6928e. Each
// weight is placed by the node that reads it, as for the chat families; a
// LayerNorm is two nodes, its weight read by Mul and its bias by Add_1.
const BERT_LAYER = String.raw`^/encoder/layer\.(\d+)/`;
const DISTIL_LAYER = String.raw`^/distilbert/transformer/layer\.(\d+)/`;
const ENCODER_RULES = {
  bert: [
    ["Gather", /^\/embeddings\/word_embeddings\/Gather$/, "word"],
    ["Gather", /^\/embeddings\/position_embeddings\/Gather$/, "position"],
    ["Gather", /^\/embeddings\/token_type_embeddings\/Gather$/, "type"],
    ["Mul", /^\/embeddings\/LayerNorm\/Mul$/, "emb_norm"],
    ["Add", /^\/embeddings\/LayerNorm\/Add_1$/, "emb_norm_b"],
    ["MatMulNBits", new RegExp(BERT_LAYER + String.raw`attention/self/(query|key|value)/MatMul_Q4$`), (m) => `w${m[2][0]}`],
    ["Add", new RegExp(BERT_LAYER + String.raw`attention/self/(query|key|value)/Add$`), (m) => `b${m[2][0]}`],
    ["MatMulNBits", new RegExp(BERT_LAYER + String.raw`attention/output/dense/MatMul_Q4$`), "wo"],
    ["Add", new RegExp(BERT_LAYER + String.raw`attention/output/dense/Add$`), "bo"],
    ["Mul", new RegExp(BERT_LAYER + String.raw`attention/output/LayerNorm/Mul$`), "attn_norm"],
    ["Add", new RegExp(BERT_LAYER + String.raw`attention/output/LayerNorm/Add_1$`), "attn_norm_b"],
    ["MatMulNBits", new RegExp(BERT_LAYER + String.raw`intermediate/dense/MatMul_Q4$`), "w1"],
    ["Add", new RegExp(BERT_LAYER + String.raw`intermediate/dense/Add$`), "b1"],
    ["MatMulNBits", new RegExp(BERT_LAYER + String.raw`output/dense/MatMul_Q4$`), "w2"],
    ["Add", new RegExp(BERT_LAYER + String.raw`output/dense/Add$`), "b2"],
    ["Mul", new RegExp(BERT_LAYER + String.raw`output/LayerNorm/Mul$`), "out_norm"],
    ["Add", new RegExp(BERT_LAYER + String.raw`output/LayerNorm/Add_1$`), "out_norm_b"],
  ],
  distilbert: [
    ["Gather", /^\/distilbert\/embeddings\/word_embeddings\/Gather$/, "word"],
    ["Gather", /^\/distilbert\/embeddings\/position_embeddings\/Gather$/, "position"],
    ["Mul", /^\/distilbert\/embeddings\/LayerNorm\/Mul$/, "emb_norm"],
    ["Add", /^\/distilbert\/embeddings\/LayerNorm\/Add_1$/, "emb_norm_b"],
    ["MatMulNBits", new RegExp(DISTIL_LAYER + String.raw`attention/([qkv])_lin/MatMul_Q4$`), (m) => `w${m[2]}`],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`attention/([qkv])_lin/Add$`), (m) => `b${m[2]}`],
    ["MatMulNBits", new RegExp(DISTIL_LAYER + String.raw`attention/out_lin/MatMul_Q4$`), "wo"],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`attention/out_lin/Add$`), "bo"],
    ["Mul", new RegExp(DISTIL_LAYER + String.raw`sa_layer_norm/Mul$`), "attn_norm"],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`sa_layer_norm/Add_1$`), "attn_norm_b"],
    ["MatMulNBits", new RegExp(DISTIL_LAYER + String.raw`ffn/lin1/MatMul_Q4$`), "w1"],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`ffn/lin1/Add$`), "b1"],
    ["MatMulNBits", new RegExp(DISTIL_LAYER + String.raw`ffn/lin2/MatMul_Q4$`), "w2"],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`ffn/lin2/Add$`), "b2"],
    ["Mul", new RegExp(DISTIL_LAYER + String.raw`output_layer_norm/Mul$`), "out_norm"],
    ["Add", new RegExp(DISTIL_LAYER + String.raw`output_layer_norm/Add_1$`), "out_norm_b"],
    ["Gemm", /^\/pre_classifier\/Gemm$/, "pre"],
    ["Gemm", /^\/classifier\/Gemm$/, "cls"],
  ],
};

// The encoder's dims and each engine tensor as the export's tensor names, for
// site/app/engine/encoder.mjs. DistilBERT's config has no LayerNorm epsilon;
// transformers fixes it at 1e-12 in DistilBERT's code, the value BERT's config
// gives.
export function encoderRoles({ nodes, tensors }, config) {
  const family = config.model_type;
  const rules = ENCODER_RULES[family];
  if (!rules) throw new Error(`family: ${family} is not an encoder this engine runs`);
  const bert = family === "bert";
  const act = bert ? config.hidden_act : config.activation;
  if (act !== "gelu") throw new Error(`activation: ${act} is not the exact GELU this engine runs`);
  const d = bert ? config.hidden_size : config.dim;
  const dims = {
    d, n_heads: bert ? config.num_attention_heads : config.n_heads, ff: bert ? config.intermediate_size : config.hidden_dim,
    n_layers: bert ? config.num_hidden_layers : config.n_layers, vocab: config.vocab_size, positions: config.max_position_embeddings,
    eps: bert ? config.layer_norm_eps : 1e-12,
  };
  if (d % dims.n_heads) throw new Error(`shape: width ${d} is not a whole number of ${dims.n_heads} heads`);
  const L = dims.n_layers;
  const labels = config.id2label ? Object.keys(config.id2label).length : 0;
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
  const width = { wq: [d, d], wk: [d, d], wv: [d, d], wo: [d, d], w1: [dims.ff, d], w2: [d, dims.ff] };
  const length = { bq: d, bk: d, bv: d, bo: d, b1: dims.ff, b2: d, attn_norm: d, attn_norm_b: d, out_norm: d, out_norm_b: d, emb_norm: d, emb_norm_b: d };
  const rows = { word: dims.vocab, position: dims.positions, type: config.type_vocab_size };

  for (const [op, name, inputs, attr = {}] of nodes) {
    const rule = rules.find(([o, pattern]) => o === op && pattern.test(name));
    if (!rule) {
      if (WEIGHT_READERS.has(op) && inputs.some(Boolean)) throw new Error(`unplaced: ${op} ${name} reads a weight no engine role takes`);
      continue;
    }
    const m = name.match(rule[1]);
    const role = typeof rule[2] === "function" ? rule[2](m) : rule[2];
    const layer = m[1] !== undefined && /^\d+$/.test(m[1]) ? Number(m[1]) : null;
    if (layer !== null && layer >= L) throw new Error(`layer: ${name} is past the config's ${L} layers`);
    const engineName = layer === null ? role : `l${layer}.${role}`;
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
      if (!FORMAT[t.dtype]) throw new Error(`precision: ${name} is ${t.dtype}; a table is f32 or f16`);
      want(`${name} rows`, t.dims[0], rows[role]);
      want(`${name} width`, t.dims[1], d);
      place(engineName, { fmt: t.dtype, N: t.dims[0], K: d, from: inputs[0] }, name);
    } else if (op === "Gemm") {
      if (attr.transB !== 1 || attr.transA || (attr.alpha ?? 1) !== 1 || (attr.beta ?? 1) !== 1) throw new Error(`shape: ${name} is not y = x W^T + b with W as [out][in]`);
      const w = tensor(inputs[1], name);
      const b = tensor(inputs[2], name);
      const out = role === "pre" ? d : labels;
      if (!FORMAT[w.dtype] || !FORMAT[b.dtype]) throw new Error(`precision: ${name} is ${w.dtype}; the head is f32 or f16`);
      want(`${name} rows`, w.dims[0], out);
      want(`${name} columns`, w.dims[1], d);
      want(`${name} bias`, product(b.dims), out);
      place(engineName, { fmt: w.dtype, N: out, K: d, from: inputs[1] }, name);
      place(`${engineName}_b`, { fmt: b.dtype, N: 1, K: out, from: inputs[2] }, name);
    } else {
      const held = inputs.filter(Boolean);
      if (held.length !== 1) throw new Error(`unplaced: ${name} reads ${held.length} weights, not one`);
      const t = tensor(held[0], name);
      if (!FORMAT[t.dtype]) throw new Error(`precision: ${name} is ${t.dtype}; a vector is f32 or f16`);
      want(`${name} length`, product(t.dims), length[role]);
      place(engineName, { fmt: t.dtype, N: 1, K: product(t.dims), from: held[0] }, name);
    }
  }
  const needed = ["word", "position", "emb_norm", "emb_norm_b"];
  if (bert && config.type_vocab_size) needed.push("type");
  for (let l = 0; l < L; l++) for (const r of ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "attn_norm", "attn_norm_b", "w1", "b1", "w2", "b2", "out_norm", "out_norm_b"]) needed.push(`l${l}.${r}`);
  if (!bert && labels) needed.push("pre", "pre_b", "cls", "cls_b");
  for (const name of needed) if (!found.has(name)) throw new Error(`missing: the graph has no ${name}`);
  const extra = [...found.keys()].filter((n) => !needed.includes(n));
  if (extra.length) throw new Error(`unplaced: ${extra[0]} is not part of a ${family} model`);
  return { dims, weights: found, labels: labels ? { ...config.id2label } : null };
}
