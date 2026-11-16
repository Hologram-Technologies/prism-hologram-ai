// GP-17's table half: the encoder role map over the graphs of Xenova's q4f16
// exports of all-MiniLM-L6-v2 and distilbert-base-uncased-finetuned-sst-2-english,
// as recorded in exports/ (nodes and tensor shapes, no bytes), and over copies
// of them with one thing wrong. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/encoder-roles.mjs <repository root>

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.argv[2];
const { encoderRoles } = await import(pathToFileURL(join(root, "site/app/engine/onnx-roles.mjs")).href);
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
for (const name of ["all-MiniLM-L6-v2", "distilbert-base-uncased-finetuned-sst-2-english"]) {
  const x = load(name);
  const { dims, weights, labels } = encoderRoles(x, x.config);
  const formats = {};
  for (const w of weights.values()) formats[w.fmt] = (formats[w.fmt] ?? 0) + 1;
  exports[name] = { revision: x.revision, dims, placed: weights.size, formats, labels, wq3: weights.get("l3.wq")?.parts, head: weights.get("cls")?.from ?? null };
}

const minilm = () => load("all-MiniLM-L6-v2");
const distil = () => load("distilbert-base-uncased-finetuned-sst-2-english");
const node = (x, test) => x.nodes.find(test);
const refusals = {
  activation: refusal(() => { const x = minilm(); x.config.hidden_act = "gelu_new"; encoderRoles(x, x.config); }),
  family: refusal(() => { const x = minilm(); x.config.model_type = "roberta"; encoderRoles(x, x.config); }),
  missing: refusal(() => { const x = distil(); x.nodes = x.nodes.filter((n) => n[1] !== "/distilbert/transformer/layer.4/ffn/lin2/Add"); encoderRoles(x, x.config); }),
  transB: refusal(() => { const x = distil(); node(x, (n) => n[1] === "/classifier/Gemm")[3].transB = 0; encoderRoles(x, x.config); }),
  labels: refusal(() => { const x = distil(); x.config.id2label = { 0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE" }; encoderRoles(x, x.config); }),
  unplaced: refusal(() => { const x = minilm(); const n = structuredClone(node(x, (n) => n[1] === "/encoder/layer.1/intermediate/dense/MatMul_Q4")); n[1] = "/encoder/layer.1/extra/dense/MatMul_Q4"; x.nodes.push(n); encoderRoles(x, x.config); }),
  width: refusal(() => { const x = minilm(); x.config.intermediate_size = 1024; encoderRoles(x, x.config); }),
};
console.log(JSON.stringify({ exports, refusals }));
