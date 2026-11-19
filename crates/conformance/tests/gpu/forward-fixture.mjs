// The forward pass's fixtures (EN-49 to EN-51): a generated model written as
// onnx-community's export of its family writes one, with the float
// reference's greedy tokens and logits after its prompt. Prints one JSON
// line; the graph is hex.
//
//   node crates/conformance/tests/gpu/forward-fixture.mjs <repository root> <family>
//
// qwen3: GP-19's fixture, fused, with query and key norms. qwen2: Qwen2.5's
// decomposed export, with query, key and value biases. llama: Llama's fused
// export, with Llama 3 rotary scaling.

import { buildOnnxFixture } from "./fixture.mjs";
import { onnxFile } from "./onnx-write.mjs";
import { referenceModel, tinyQwen2 } from "./reference.mjs";

const family = process.argv[3] ?? "qwen3";
let model, graph, config, prompt;
if (family === "qwen3") {
  const fixture = buildOnnxFixture();
  model = fixture.model;
  graph = fixture.files.get(fixture.path);
  config = JSON.parse(fixture.files.get("config.json").toString("utf8"));
  prompt = fixture.prompt;
} else {
  const llama3 = { factor: 4, low: 1, high: 4, orig: 64 };
  model = tinyQwen2(16, "onnx", "f16", "dense", family === "llama" ? { vectors: "f16", llama3 } : { vectors: "f16" });
  ({ bytes: graph, config } = onnxFile(model, family, family === "qwen2" ? "decomposed" : "fused"));
  prompt = model.prompt;
}
const reference = referenceModel(model);
let last;
for (const id of prompt) last = reference.step(id);
const tokens = [];
const logits = [];
let margin = Infinity;
for (let i = 0; i < 16; i++) {
  const sorted = Array.from(last.logits).sort((a, b) => b - a);
  margin = Math.min(margin, sorted[0] - sorted[1]);
  tokens.push(last.token);
  logits.push(Array.from(last.logits));
  last = reference.step(last.token);
}
console.log(JSON.stringify({
  graph: Buffer.from(graph).toString("hex"),
  config,
  prompt,
  tokens,
  logits,
  margin,
}));
