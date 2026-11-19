// GP-35 and GP-36's fixtures: GP-13 and GP-14's encoders (seed 5) written as
// Xenova's q4f16 exports of all-MiniLM-L6-v2 and SST-2's DistilBERT write
// them, node names and all, with a config.json, a WordPiece tokenizer.json
// and the operators around the weights the real graphs hold, so the page
// imports, admits and places them exactly as it would the real files. MiniLM's
// int64 position ids are written too, read by /embeddings/Gather as there.
// Nothing here is read from a real export.

import { createHash } from "node:crypto";
import { tinyEncoder } from "./encoder-reference.mjs";
import { blake3Hex } from "../../../../try/vendor/prism-code/src/blake3.mjs";

const varint = (value) => {
  const out = [];
  let v = BigInt(value);
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v) byte |= 0x80;
    out.push(byte);
  } while (v);
  return Uint8Array.from(out);
};
const join = (parts) => Buffer.concat(parts.map((p) => Buffer.from(p)));
const number = (field, value) => join([varint(field << 3), varint(value)]);
const bytes = (field, payload) => {
  const body = typeof payload === "string" ? Buffer.from(payload) : payload;
  return join([varint((field << 3) | 2), varint(body.length), body]);
};
const TYPES = { f16: 10, u8: 2, i64: 7 };

// The vocabulary: the four specials, the letters, a few words, and pieces.
export const ENCODER_WORDS = ["the", "cat", "sat", "on", "mat", "a", "good", "bad", "film", "day"];
function tokenizer(size) {
  const vocab = { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3 };
  const add = (t) => { if (!(t in vocab) && Object.keys(vocab).length < size) vocab[t] = Object.keys(vocab).length; };
  for (const w of ENCODER_WORDS) add(w);
  for (let c = 97; c <= 122; c++) add(String.fromCharCode(c));
  for (let c = 97; c <= 122; c++) add(`##${String.fromCharCode(c)}`);
  for (let i = 0; Object.keys(vocab).length < size; i++) add(`w${i}`);
  return Buffer.from(JSON.stringify({
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [],
    normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
    pre_tokenizer: { type: "BertPreTokenizer" },
    post_processor: {
      type: "TemplateProcessing",
      single: [{ SpecialToken: { id: "[CLS]", type_id: 0 } }, { Sequence: { id: "A", type_id: 0 } }, { SpecialToken: { id: "[SEP]", type_id: 0 } }],
      pair: [],
      special_tokens: { "[CLS]": { id: "[CLS]", ids: [2], tokens: ["[CLS]"] }, "[SEP]": { id: "[SEP]", ids: [3], tokens: ["[SEP]"] } },
    },
    decoder: { type: "WordPiece", prefix: "##", cleanup: true },
    model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100, vocab },
  }));
}

// `kind` is "bert" or "distil": the model, its graph's bytes and its config.
function encoderGraph(model, kind) {
  const { dims, tensors } = model;
  const nodes = [];
  const initializers = [];
  let activation = 0;
  const hold = (name, dtype, shape, data) => {
    initializers.push(bytes(5, join([...shape.map((d) => number(1, d)), number(2, TYPES[dtype]), bytes(8, name), bytes(9, data)])));
    return name;
  };
  const node = (op, name, inputs, attributes = {}) => {
    const parts = inputs.map((input) => bytes(1, input ?? `x${activation++}`));
    parts.push(bytes(2, `${name}/output_0`), bytes(3, name), bytes(4, op));
    for (const [key, value] of Object.entries(attributes)) parts.push(bytes(5, join([bytes(1, key), number(3, value), number(20, 2)])));
    parts.push(bytes(7, op === "MatMulNBits" ? "com.microsoft" : ""));
    nodes.push(bytes(1, join(parts)));
  };
  const t = (role) => tensors.get(role);
  const vector = (role, name) => hold(name, "f16", [t(role).K], t(role).bytes);
  const table = (role, name) => hold(name, "f16", [t(role).N, t(role).K], t(role).bytes);
  let serial = 700;
  const nbits = (role, nodeName) => {
    const w = t(role);
    const prefix = `onnx::MatMul_${serial++}`;
    const c = hold(`${prefix}_Q4`, "u8", [w.N, w.K / 32, 16], w.parts[0]);
    const s = hold(`${prefix}_scales`, "f16", [(w.N * w.K) / 32], w.parts[1]);
    node("MatMulNBits", nodeName, [null, c, s], { K: w.K, N: w.N, bits: 4, block_size: 32 });
  };
  const bert = kind === "bert";
  const top = bert ? "" : "/distilbert";
  const pre = bert ? "" : "distilbert.";
  node("Gather", `${top}/embeddings/word_embeddings/Gather`, [table("word", `${pre}embeddings.word_embeddings.weight`), null]);
  if (bert) {
    node("Gather", "/embeddings/token_type_embeddings/Gather", [table("type", "embeddings.token_type_embeddings.weight"), null]);
    // The position ids MiniLM's export keeps as an int64 constant.
    const ids = Buffer.alloc(8 * dims.positions);
    for (let i = 0; i < dims.positions; i++) ids.writeBigInt64LE(BigInt(i), 8 * i);
    node("Gather", "/embeddings/Gather", [hold("embeddings.position_ids", "i64", [1, dims.positions], ids), null]);
  }
  node("Gather", `${top}/embeddings/position_embeddings/Gather`, [table("position", `${pre}embeddings.position_embeddings.weight`), null]);
  for (const op of ["ReduceMean", "Sub", "Pow", "Sqrt", "Div"]) node(op, `${top}/embeddings/LayerNorm/${op}`, [null, null]);
  node("Mul", `${top}/embeddings/LayerNorm/Mul`, [null, vector("emb_norm", `${pre}embeddings.LayerNorm.weight`)]);
  node("Add", `${top}/embeddings/LayerNorm/Add_1`, [null, vector("emb_norm_b", `${pre}embeddings.LayerNorm.bias`)]);
  for (let l = 0; l < dims.n_layers; l++) {
    const p = `l${l}.`;
    const at = bert ? `/encoder/layer.${l}/` : `/distilbert/transformer/layer.${l}/`;
    const w = bert ? `encoder.layer.${l}.` : `distilbert.transformer.layer.${l}.`;
    for (const q of ["q", "k", "v"]) {
      const lin = bert ? `attention/self/${{ q: "query", k: "key", v: "value" }[q]}` : `attention/${q}_lin`;
      nbits(`${p}w${q}`, `${at}${lin}/MatMul_Q4`);
      node("Add", `${at}${lin}/Add`, [vector(`${p}b${q}`, `${w}${lin.replaceAll("/", ".")}.bias`), null]);
    }
    for (const op of ["Reshape", "Transpose", "MatMul", "Div", "Softmax"]) node(op, `${at}attention/${op}`, [null]);
    const out = bert ? "attention/output/dense" : "attention/out_lin";
    nbits(`${p}wo`, `${at}${out}/MatMul_Q4`);
    node("Add", `${at}${out}/Add`, [vector(`${p}bo`, `${w}${out.replaceAll("/", ".")}.bias`), null]);
    const n1 = bert ? "attention/output/LayerNorm" : "sa_layer_norm";
    node("Mul", `${at}${n1}/Mul`, [null, vector(`${p}attn_norm`, `${w}${n1.replaceAll("/", ".")}.weight`)]);
    node("Add", `${at}${n1}/Add_1`, [null, vector(`${p}attn_norm_b`, `${w}${n1.replaceAll("/", ".")}.bias`)]);
    const up = bert ? "intermediate/dense" : "ffn/lin1";
    nbits(`${p}w1`, `${at}${up}/MatMul_Q4`);
    node("Add", `${at}${up}/Add`, [vector(`${p}b1`, `${w}${up.replaceAll("/", ".")}.bias`), null]);
    node("Erf", `${at}${bert ? "intermediate/intermediate_act_fn" : "ffn/activation"}/Erf`, [null]);
    const down = bert ? "output/dense" : "ffn/lin2";
    nbits(`${p}w2`, `${at}${down}/MatMul_Q4`);
    node("Add", `${at}${down}/Add`, [vector(`${p}b2`, `${w}${down.replaceAll("/", ".")}.bias`), null]);
    const n2 = bert ? "output/LayerNorm" : "output_layer_norm";
    node("Mul", `${at}${n2}/Mul`, [null, vector(`${p}out_norm`, `${w}${n2.replaceAll("/", ".")}.weight`)]);
    node("Add", `${at}${n2}/Add_1`, [null, vector(`${p}out_norm_b`, `${w}${n2.replaceAll("/", ".")}.bias`)]);
  }
  if (!bert) {
    node("Gather", "/Gather", [null, null]);
    node("Gemm", "/pre_classifier/Gemm", [null, hold("pre_classifier.weight", "f16", [dims.d, dims.d], t("pre").bytes), vector("pre_b", "pre_classifier.bias")], { transB: 1 });
    node("Relu", "/Relu", [null]);
    node("Gemm", "/classifier/Gemm", [null, hold("classifier.weight", "f16", [2, dims.d], t("cls").bytes), vector("cls_b", "classifier.bias")], { transB: 1 });
  }
  const graph = join([...nodes, bytes(2, "fixture"), ...initializers]);
  const opset = (domain, version) => bytes(8, join([bytes(1, domain), number(2, version)]));
  const config = bert
    ? { model_type: "bert", architectures: ["BertModel"], hidden_act: "gelu", hidden_size: dims.d, intermediate_size: dims.ff, num_attention_heads: dims.n_heads, num_hidden_layers: dims.n_layers, layer_norm_eps: 1e-12, max_position_embeddings: dims.positions, type_vocab_size: 2, vocab_size: dims.vocab }
    : { model_type: "distilbert", architectures: ["DistilBertForSequenceClassification"], activation: "gelu", dim: dims.d, hidden_dim: dims.ff, n_heads: dims.n_heads, n_layers: dims.n_layers, max_position_embeddings: dims.positions, vocab_size: dims.vocab, id2label: { 0: "NEGATIVE", 1: "POSITIVE" }, label2id: { NEGATIVE: 0, POSITIVE: 1 } };
  return { graph: join([number(1, 8), opset("", 14), opset("com.microsoft", 1), bytes(7, graph)]), config };
}

// One encoder's repository as the hub would record it: its files, and how the
// journey names them.
export function buildEncoderFixture(kind) {
  const model = tinyEncoder(5, kind);
  const { graph, config } = encoderGraph(model, kind);
  const files = new Map([
    ["config.json", Buffer.from(JSON.stringify(config))],
    ["tokenizer.json", tokenizer(model.dims.vocab)],
    ["tokenizer_config.json", Buffer.from(JSON.stringify({ do_lower_case: true, model_max_length: 512 }))],
    ["onnx/model_q4f16.onnx", graph],
  ]);
  return {
    kind,
    model,
    files,
    sha: (b) => createHash("sha256").update(b).digest("hex"),
    b3: (b) => blake3Hex(b),
    path: "onnx/model_q4f16.onnx",
    repo: kind === "bert" ? "fixture/tiny-minilm" : "fixture/tiny-sst2",
    revision: "2".repeat(40),
  };
}

// The sentences GP-35 and GP-36 put to the page, and what EN-64 recomputes.
export const ENCODER_TEXTS = { bert: ["the cat sat on the mat", "a bad day"], distil: ["a good film", "the bad film"] };

// Run directly, it prints each encoder's tokenizer.json and config.json and
// its sentences, for the Rust suite:
//
//   node crates/conformance/tests/gpu/encoder-export.mjs
if (import.meta.url === `file://${process.argv[1].replaceAll("\\", "/").replace(/^(?=[A-Za-z]:)/, "/")}`) {
  const out = {};
  for (const kind of ["bert", "distil"]) {
    const f = buildEncoderFixture(kind);
    out[kind] = { tokenizer: f.files.get("tokenizer.json").toString(), config: f.files.get("config.json").toString(), texts: ENCODER_TEXTS[kind] };
  }
  console.log(JSON.stringify(out));
}
