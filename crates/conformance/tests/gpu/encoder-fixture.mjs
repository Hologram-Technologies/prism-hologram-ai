// EN-62 and EN-63's fixture: GP-13 and GP-14's encoders (seed 5), a BERT
// shaped as all-MiniLM-L6-v2's export and a DistilBERT classifier shaped as
// SST-2's, with the float reference's outputs for each sentence. Prints one
// JSON line; bytes are hex.
//
//   node crates/conformance/tests/gpu/encoder-fixture.mjs

import { referenceEncoder, tinyEncoder } from "./encoder-reference.mjs";

const hex = (bytes) => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("hex");

function encoder(kind) {
  const model = tinyEncoder(5, kind);
  const reference = referenceEncoder(model);
  const tensors = {};
  for (const [name, t] of model.tensors) {
    tensors[name] = t.parts
      ? { kind: "q4", rows: t.N, columns: t.K, codes: hex(t.parts[0]), scales: hex(t.parts[1]) }
      : { kind: "f16", rows: t.N, columns: t.K, bytes: hex(t.bytes) };
  }
  return {
    dims: model.dims,
    tensors,
    sentences: model.sentences,
    embed: kind === "bert" ? model.sentences.map((s) => Array.from(reference.embed(s))) : null,
    logits: kind === "distil" ? model.sentences.map((s) => Array.from(reference.classify(s).logits)) : null,
  };
}

console.log(JSON.stringify({ bert: encoder("bert"), distil: encoder("distil") }));
