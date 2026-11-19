// EN-49's fixture: GP-19's Qwen3-shaped model, written as onnx-community's
// fused q4f16 export writes one, with the float reference's greedy tokens and
// logits after its prompt. Prints one JSON line; the graph is hex.
//
//   node crates/conformance/tests/gpu/forward-fixture.mjs <repository root>

import { buildOnnxFixture } from "./fixture.mjs";
import { referenceModel } from "./reference.mjs";

const fixture = buildOnnxFixture();
const config = JSON.parse(fixture.files.get("config.json").toString("utf8"));
const reference = referenceModel(fixture.model);
let last;
for (const id of fixture.prompt) last = reference.step(id);
const tokens = [];
const logits = [];
for (let i = 0; i < 16; i++) {
  tokens.push(last.token);
  logits.push(Array.from(last.logits));
  last = reference.step(last.token);
}
console.log(JSON.stringify({
  graph: Buffer.from(fixture.files.get(fixture.path)).toString("hex"),
  rope_theta: config.rope_theta,
  rms_norm_eps: config.rms_norm_eps,
  prompt: fixture.prompt,
  expected: fixture.expected,
  tokens,
  logits,
}));
