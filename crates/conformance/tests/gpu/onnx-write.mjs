// An ONNX file written from a generated model, field by field from onnx.proto:
// ModelProto (ir_version 1, opset_import 8, graph 7); GraphProto (node 1,
// name 2, initializer 5); NodeProto (input 1, output 2, name 3, op_type 4,
// attribute 5, domain 7); AttributeProto (name 1, i 3, type 20 = INT);
// TensorProto (dims 1, data_type 2, name 8, raw_data 9). Every weight is
// inline. Nothing here is read from a real export or from the importer.

import { asExport } from "./export.mjs";

const varint = (value) => {
  const out = [];
  let v = BigInt(value);
  if (v < 0n) v += 1n << 64n;
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v) byte |= 0x80;
    out.push(byte);
  } while (v);
  return Uint8Array.from(out);
};
const join = (parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};
const text = new TextEncoder();
const number = (field, value) => join([varint(field << 3), varint(value)]);
const bytes = (field, payload) => {
  const body = typeof payload === "string" ? text.encode(payload) : payload;
  return join([varint((field << 3) | 2), varint(body.length), body]);
};

// The operators that live in Microsoft's domain in an onnxruntime export.
const MICROSOFT = new Set(["MatMulNBits", "GroupQueryAttention", "RotaryEmbedding", "SkipSimplifiedLayerNormalization"]);
const TYPES = { f32: 1, u8: 2, f16: 10 };

// The model written as onnx-community's fused exports are, and its bytes.
export function onnxFile(model, family) {
  const exp = asExport(model, "fused", family);
  let activation = 0;
  const nodes = exp.nodes.map(([op, name, inputs, attributes = {}]) => {
    const parts = [];
    for (const input of inputs) parts.push(bytes(1, input ?? `x${activation++}`));
    parts.push(bytes(2, `${name}/output_0`), bytes(3, name), bytes(4, op));
    for (const [key, value] of Object.entries(attributes)) {
      parts.push(bytes(5, join([bytes(1, key), number(3, value), number(20, 2)])));
    }
    parts.push(bytes(7, MICROSOFT.has(op) ? "com.microsoft" : ""));
    return bytes(1, join(parts));
  });
  const initializers = Object.entries(exp.tensors).map(([name, [dtype, dims]]) => {
    const parts = dims.map((d) => number(1, d));
    parts.push(number(2, TYPES[dtype]), bytes(8, name), bytes(9, exp.bytes.get(name)));
    return bytes(5, join(parts));
  });
  const graph = join([...nodes, bytes(2, "fixture"), ...initializers]);
  const opset = (domain, version) => bytes(8, join([bytes(1, domain), number(2, version)]));
  return { bytes: join([number(1, 8), opset("", 21), opset("com.microsoft", 1), bytes(7, graph)]), config: exp.config };
}
