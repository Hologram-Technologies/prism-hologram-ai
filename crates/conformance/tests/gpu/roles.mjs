// GP-12's table half: the role map over the graphs of onnx-community's three
// q4f16 exports, as recorded in exports/ (nodes and tensor shapes, no bytes),
// and over copies of them with one thing wrong. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/roles.mjs <repository root>

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.argv[2];
const { roles } = await import(pathToFileURL(join(root, "site/app/engine/onnx-roles.mjs")).href);
const load = (name) => JSON.parse(readFileSync(join(root, "crates/conformance/tests/gpu/exports", `${name}.json`), "utf8"));
const refusal = (f) => {
  try {
    f();
    return null;
  } catch (e) {
    return e.message;
  }
};

const exports = {};
for (const name of ["Qwen2.5-1.5B-Instruct", "Qwen3-0.6B-ONNX", "Llama-3.2-1B-Instruct-ONNX"]) {
  const x = load(name);
  const { dims, weights } = roles(x, x.config);
  const formats = {};
  for (const w of weights.values()) formats[w.fmt] = (formats[w.fmt] ?? 0) + 1;
  exports[name] = {
    revision: x.revision, dims, placed: weights.size, formats,
    // Qwen2.5's layer 3 query weight, by its export's own name.
    q3: weights.get("l3.wq")?.parts, embed: weights.get("embed")?.from, head: weights.has("lm_head"),
  };
}

// One thing wrong at a time, each on a fresh copy.
const qwen25 = () => load("Qwen2.5-1.5B-Instruct");
const qwen3 = () => load("Qwen3-0.6B-ONNX");
const llama = () => load("Llama-3.2-1B-Instruct-ONNX");
const without = (x, test) => ({ ...x, nodes: x.nodes.filter((n) => !test(n)) });
const node = (x, test) => x.nodes.find(test);
const refusals = {
  missing: refusal(() => { const x = qwen25(); roles(without(x, (n) => n[1] === "/model/layers.3/self_attn/q_proj/MatMul_Q4"), x.config); }),
  duplicate: refusal(() => { const x = qwen3(); x.nodes.push(node(x, (n) => n[1].endsWith("layers.5/mlp/up_proj/MatMul_Q4"))); roles(x, x.config); }),
  unplaced: refusal(() => { const x = llama(); const n = structuredClone(node(x, (n) => n[1].endsWith("layers.2/mlp/up_proj/MatMul_Q4"))); n[1] = "/model/layers.2/mlp/extra_proj/MatMul_Q4"; x.nodes.push(n); roles(x, x.config); }),
  zeroPoints: refusal(() => { const x = qwen3(); node(x, (n) => n[1].endsWith("layers.0/attn/k_proj/MatMul_Q4"))[2].push("model.layers.0.attn.q_proj.MatMul.weight_scales"); roles(x, x.config); }),
  block: refusal(() => { const x = llama(); node(x, (n) => n[1].endsWith("layers.1/attn/v_proj/MatMul_Q4"))[3].block_size = 64; roles(x, x.config); }),
  heads: refusal(() => { const x = qwen3(); x.config.num_key_value_heads = 4; roles(x, x.config); }),
  family: refusal(() => { const x = qwen25(); x.config.model_type = "gemma"; roles(x, x.config); }),
  layers: refusal(() => { const x = qwen25(); x.config.num_hidden_layers = 27; roles(x, x.config); }),
  interleaved: refusal(() => { const x = qwen3(); node(x, (n) => n[0] === "RotaryEmbedding")[3].interleaved = 1; roles(x, x.config); }),
  // A float tensor read by an unplaced Gather is refused, as any weight is.
  floatIndex: refusal(() => { const x = qwen25(); x.tensors["/model/Constant_4_output_0"] = ["f16", []]; x.nodes.push(["Gather", "/model/Gather_2", [null, "/model/Constant_4_output_0"]]); roles(x, x.config); }),
};
// Qwen2.5's decomposed export gathers from its shapes by int64 constants, as
// /model/Gather_2 does (read 2026-11-20 at d54957e); the importer holds each
// such constant as a tensor, which indexes and is not a weight, so it is placed
// nowhere and refused never. The recorded graph keeps only the nodes that read
// weights, so the node is added here as the importer gives it.
const indexed = qwen25();
indexed.tensors["/model/Constant_4_output_0"] = ["i64", []];
indexed.nodes.push(["Gather", "/model/Gather_2", [null, "/model/Constant_4_output_0"]]);
const indexConstant = refusal(() => roles(indexed, indexed.config));
console.log(JSON.stringify({ exports, refusals, indexConstant }));
