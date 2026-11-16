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
