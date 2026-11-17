// A generated model written as onnx-community's exports write theirs, for the
// role map to read back: the graph's nodes, its tensors' types and shapes, and
// their bytes. "decomposed" names as Qwen2.5's export does (weights named
// onnx::MatMul_N, norms as Mul_1, biases as Add); "fused" as Qwen3's and
// Llama's do. Written from the recorded exports, not from the role map.

export function asExport(model, style, family) {
  const { dims, tensors } = model;
  const nodes = [];
  const table = {};
  const bytes = new Map();
  let counter = 9333;
  const hold = (name, dtype, shape, data) => {
    table[name] = [dtype, shape];
    bytes.set(name, data);
    return name;
  };
  const vector = (node, op, source, name) => {
    const t = tensors.get(source);
    nodes.push([op, node, op === "SkipSimplifiedLayerNormalization" ? [null, null, hold(name, t.fmt, [t.K], t.bytes)] : [hold(name, t.fmt, [t.K], t.bytes), null]]);
  };
  const nbits = (node, source, prefix) => {
    const t = tensors.get(source);
    const [codes, scales] = t.parts;
    const c = hold(`${prefix}_Q4`, "u8", [t.N, t.K / 32, 16], codes);
    const s = hold(`${prefix}_scales`, t.fmt === "q4f16" ? "f16" : "f32", style === "decomposed" ? [(t.N * t.K) / 32] : [t.N, t.K / 32], scales);
    nodes.push(["MatMulNBits", node, [null, c, s], { K: t.K, N: t.N, bits: 4, block_size: 32 }]);
  };
  const e = tensors.get("embed");
  nodes.push(["Gather", "/model/embed_tokens/Gather", [hold("model.embed_tokens.weight", e.fmt, [e.N, e.K], e.bytes), null]]);
  for (let l = 0; l < dims.n_layers; l++) {
    const at = `/model/layers.${l}/`;
    const attn = style === "decomposed" ? "self_attn" : "attn";
    if (style === "decomposed") vector(`${at}input_layernorm/Mul_1`, "Mul", `l${l}.attn_norm`, `model.layers.${l}.input_layernorm.weight`);
    else if (l === 0) vector(`${at}input_layernorm/LayerNorm`, "SimplifiedLayerNormalization", `l${l}.attn_norm`, `model.layers.${l}.input_layernorm.weight`);
    else vector(`${at}input_layernorm/SkipLayerNorm`, "SkipSimplifiedLayerNormalization", `l${l}.attn_norm`, `model.layers.${l}.input_layernorm.weight`);
    for (const p of ["q", "k", "v"]) {
      const named = style === "decomposed" ? `onnx::MatMul_${counter++}` : `model.layers.${l}.attn.${p}_proj.MatMul.weight`;
      nbits(`${at}${attn}/${p}_proj/MatMul_Q4`, `l${l}.w${p}`, named);
      if (tensors.has(`l${l}.b${p}`)) vector(`${at}${attn}/${p}_proj/Add`, "Add", `l${l}.b${p}`, `model.layers.${l}.self_attn.${p}_proj.bias`);
    }
    // Qwen3's export: a norm on each query and key head, before the rotation.
    for (const p of ["q", "k"]) {
      if (tensors.has(`l${l}.${p}_norm`)) vector(`${at}attn/${p}_norm/SimplifiedLayerNormalization`, "SimplifiedLayerNormalization", `l${l}.${p}_norm`, `model.layers.${l}.attn.${p}_norm.weight`);
    }
    if (style === "fused") nodes.push(["GroupQueryAttention", `${at}attn/GroupQueryAttention`, [null, null, null], { num_heads: dims.n_heads, kv_num_heads: dims.n_kv_heads, do_rotary: 1, rotary_interleaved: 0 }]);
    nbits(`${at}${attn}/o_proj/MatMul_Q4`, `l${l}.wo`, style === "decomposed" ? `onnx::MatMul_${counter++}` : `model.layers.${l}.attn.o_proj.MatMul.weight`);
    if (style === "decomposed") vector(`${at}post_attention_layernorm/Mul_1`, "Mul", `l${l}.ffn_norm`, `model.layers.${l}.post_attention_layernorm.weight`);
    else vector(`${at}post_attention_layernorm/SkipLayerNorm`, "SkipSimplifiedLayerNormalization", `l${l}.ffn_norm`, `model.layers.${l}.post_attention_layernorm.weight`);
    for (const p of ["gate", "up", "down"]) nbits(`${at}mlp/${p}_proj/MatMul_Q4`, `l${l}.w_${p}`, style === "decomposed" ? `onnx::MatMul_${counter++}` : `model.layers.${l}.mlp.${p}_proj.MatMul.weight`);
  }
  if (style === "decomposed") vector("/model/norm/Mul_1", "Mul", "final_norm", "model.norm.weight");
  else vector(`/model/layers.${dims.n_layers}/final_norm_layernorm/SkipLayerNorm`, "SkipSimplifiedLayerNormalization", "final_norm", "model.norm.weight");
  const config = {
    model_type: family, hidden_size: dims.d, num_attention_heads: dims.n_heads, num_key_value_heads: dims.n_kv_heads, head_dim: dims.hd,
    intermediate_size: dims.ff, vocab_size: dims.vocab, num_hidden_layers: dims.n_layers, rope_theta: dims.rope_base, rms_norm_eps: dims.eps,
    tie_word_embeddings: !tensors.has("lm_head"),
    rope_scaling: dims.llama3 ? { rope_type: "llama3", factor: dims.llama3.factor, low_freq_factor: dims.llama3.low, high_freq_factor: dims.llama3.high, original_max_position_embeddings: dims.llama3.orig } : null,
  };
  return { nodes, tensors: table, bytes, config };
}

// A generated encoder written as Xenova's exports write theirs: "bert" as
// all-MiniLM-L6-v2 names its nodes, "distilbert" as the SST-2 classifier.
// Weights carry the onnx::MatMul_N names those exports give them.
export function asEncoderExport(model, family) {
  const { dims, tensors } = model;
  const nodes = [];
  const table = {};
  const bytes = new Map();
  let counter = 787;
  const hold = (name, dtype, shape, data) => {
    table[name] = [dtype, shape];
    bytes.set(name, data);
    return name;
  };
  const bert = family === "bert";
  const top = bert ? "" : "/distilbert";
  const gather = (node, source, name) => {
    const t = tensors.get(source);
    nodes.push(["Gather", node, [hold(name, t.fmt, [t.N, t.K], t.bytes), null]]);
  };
  const vector = (node, op, source, name) => {
    const t = tensors.get(source);
    nodes.push([op, node, [hold(name, t.fmt, [t.K], t.bytes), null]]);
  };
  const nbits = (node, source) => {
    const t = tensors.get(source);
    const n = counter++;
    const [codes, scales] = t.parts;
    nodes.push(["MatMulNBits", node, [null, hold(`onnx::MatMul_${n}_Q4`, "u8", [t.N, t.K / 32, 16], codes), hold(`onnx::MatMul_${n}_scales`, "f16", [(t.N * t.K) / 32], scales)], { K: t.K, N: t.N, bits: 4, block_size: 32 }]);
  };
  gather(`${top}/embeddings/word_embeddings/Gather`, "word", "embeddings.word_embeddings.weight");
  gather(`${top}/embeddings/position_embeddings/Gather`, "position", "embeddings.position_embeddings.weight");
  if (tensors.has("type")) gather("/embeddings/token_type_embeddings/Gather", "type", "embeddings.token_type_embeddings.weight");
  vector(`${top}/embeddings/LayerNorm/Mul`, "Mul", "emb_norm", "embeddings.LayerNorm.weight");
  vector(`${top}/embeddings/LayerNorm/Add_1`, "Add", "emb_norm_b", "embeddings.LayerNorm.bias");
  const names = bert
    ? { q: "attention/self/query", k: "attention/self/key", v: "attention/self/value", o: "attention/output/dense", attn: "attention/output/LayerNorm", 1: "intermediate/dense", 2: "output/dense", out: "output/LayerNorm" }
    : { q: "attention/q_lin", k: "attention/k_lin", v: "attention/v_lin", o: "attention/out_lin", attn: "sa_layer_norm", 1: "ffn/lin1", 2: "ffn/lin2", out: "output_layer_norm" };
  for (let l = 0; l < dims.n_layers; l++) {
    const at = bert ? `/encoder/layer.${l}/` : `/distilbert/transformer/layer.${l}/`;
    const p = `l${l}.`;
    for (const w of ["q", "k", "v", "o"]) {
      nbits(`${at}${names[w]}/MatMul_Q4`, `${p}w${w}`);
      vector(`${at}${names[w]}/Add`, "Add", `${p}b${w}`, `layer.${l}.${w}.bias`);
    }
    vector(`${at}${names.attn}/Mul`, "Mul", `${p}attn_norm`, `layer.${l}.attn_norm.weight`);
    vector(`${at}${names.attn}/Add_1`, "Add", `${p}attn_norm_b`, `layer.${l}.attn_norm.bias`);
    nbits(`${at}${names[1]}/MatMul_Q4`, `${p}w1`);
    vector(`${at}${names[1]}/Add`, "Add", `${p}b1`, `layer.${l}.1.bias`);
    nbits(`${at}${names[2]}/MatMul_Q4`, `${p}w2`);
    vector(`${at}${names[2]}/Add`, "Add", `${p}b2`, `layer.${l}.2.bias`);
    vector(`${at}${names.out}/Mul`, "Mul", `${p}out_norm`, `layer.${l}.out_norm.weight`);
    vector(`${at}${names.out}/Add_1`, "Add", `${p}out_norm_b`, `layer.${l}.out_norm.bias`);
  }
  if (tensors.has("cls")) {
    for (const [node, w, b] of [["/pre_classifier/Gemm", "pre", "pre_b"], ["/classifier/Gemm", "cls", "cls_b"]]) {
      const tw = tensors.get(w);
      const tb = tensors.get(b);
      nodes.push(["Gemm", node, [null, hold(`${w}.weight`, tw.fmt, [tw.N, tw.K], tw.bytes), hold(`${w}.bias`, tb.fmt, [tb.K], tb.bytes)], { alpha: 1, beta: 1, transB: 1 }]);
    }
  }
  const config = bert
    ? { model_type: "bert", hidden_size: dims.d, num_attention_heads: dims.n_heads, intermediate_size: dims.ff, num_hidden_layers: dims.n_layers, vocab_size: dims.vocab, max_position_embeddings: dims.positions, layer_norm_eps: dims.eps, hidden_act: "gelu", type_vocab_size: tensors.has("type") ? tensors.get("type").N : 0 }
    : { model_type: "distilbert", dim: dims.d, n_heads: dims.n_heads, hidden_dim: dims.ff, n_layers: dims.n_layers, vocab_size: dims.vocab, max_position_embeddings: dims.positions, activation: "gelu", id2label: tensors.has("cls") ? { 0: "NEGATIVE", 1: "POSITIVE" } : undefined };
  return { nodes, tensors: table, bytes, config };
}
