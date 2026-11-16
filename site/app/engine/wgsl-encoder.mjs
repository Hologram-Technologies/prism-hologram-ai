// Kernels for BERT-style encoders: the embedding model and the classifier of
// milestone 2. An encoder reads a whole sentence at once and every token
// attends to every other, so these work on S rows of the sentence together.
// Read on 2026-11-16 from Xenova's q4f16 exports of all-MiniLM-L6-v2 (BERT)
// and distilbert-base-uncased-finetuned-sst-2-english: 4-bit weights with f16
// biases, LayerNorm with weight and bias after each block, exact GELU (the
// graph's Erf), word and position embeddings and, for BERT, token types.

// The read of one number from a dense f32 or f16 table, element e.
const TABLE = {
  f32: (v) => `array<f32>;\nfn ${v}At(e: u32) -> f32 { return ${v}[e]; }`,
  f16: (v) => `array<u32>;\nfn ${v}At(e: u32) -> f32 { return unpack2x16float(${v}[e >> 1u])[e & 1u]; }`,
};

// One output row n of one sentence row m per workgroup: the q4 matrix-vector
// product of wgsl-q4.mjs, row by row. P = (K, N, accumulate, 0).
export const matmulRowsQ4 = (s) => `
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: ${s === "f16" ? "array<u32>;\nfn scale(i: u32) -> f32 { return unpack2x16float(scales[i >> 1u])[i & 1u]; }" : "array<f32>;\nfn scale(i: u32) -> f32 { return scales[i]; }"}
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x;
  let m = wg.y;
  let t = lid.x;
  let words = P.x >> 3u;
  let groups = P.x >> 5u;
  let row4 = (P.x >> 2u) * m;
  var acc = 0.0;
  for (var w = t; w < words; w += 64u) {
    let c = codes[n * words + w];
    let lo = vec4<f32>(f32(c & 0xfu), f32((c >> 4u) & 0xfu), f32((c >> 8u) & 0xfu), f32((c >> 12u) & 0xfu)) - 8.0;
    let hi = vec4<f32>(f32((c >> 16u) & 0xfu), f32((c >> 20u) & 0xfu), f32((c >> 24u) & 0xfu), f32(c >> 28u)) - 8.0;
    acc += (dot(x4[row4 + 2u * w], lo) + dot(x4[row4 + 2u * w + 1u], hi)) * scale(n * groups + (w >> 2u));
  }
  red[t] = acc;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  if (t == 0u) {
    let at = m * P.y + n;
    if (P.z == 1u) { o[at] = o[at] + red[0]; } else { o[at] = red[0]; }
  }
}`;

// x[m][k] = word[tokens[m]][k] + position[m][k] (+ type[0][k] when P.z = 1).
// P = (width, rows, has types, 0).
export const embedSum = (s) => `
@group(0) @binding(0) var<storage, read> word: ${TABLE[s]("word")}
@group(0) @binding(1) var<storage, read> position: ${TABLE[s]("position")}
@group(0) @binding(2) var<storage, read> kind: ${TABLE[s]("kind")}
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> tokens: array<u32>;
@group(0) @binding(5) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * P.y) { return; }
  let m = i / P.x;
  let k = i % P.x;
  var v = wordAt(tokens[m] * P.x + k) + positionAt(m * P.x + k);
  if (P.z == 1u) { v += kindAt(k); }
  x[i] = v;
}`;

// One row per workgroup: x = LayerNorm(x + r) * g + b, the residual added
// first when P.w = 1. P = (width, rows, eps as f32 bits, residual).
export const LAYER_NORM_ROWS = `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x * P.x;
  let t = lid.x;
  if (P.w == 1u) {
    for (var k = t; k < P.x; k += 64u) { x[row + k] = x[row + k] + r[row + k]; }
    workgroupBarrier();
  }
  var s = 0.0;
  for (var k = t; k < P.x; k += 64u) { s += x[row + k]; }
  red[t] = s;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let mean = red[0] / f32(P.x);
  workgroupBarrier();
  var q = 0.0;
  for (var k = t; k < P.x; k += 64u) { let c = x[row + k] - mean; q += c * c; }
  red[t] = q;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(P.x) + bitcast<f32>(P.z));
  for (var k = t; k < P.x; k += 64u) { x[row + k] = (x[row + k] - mean) * inv * g[k] + b[k]; }
}`;

// x = x * (1 + erf(x / sqrt 2)) / 2 over P.x numbers. WGSL has no erf; this is
// Abramowitz and Stegun 7.1.26, within 1.5e-7 of it.
export const GELU = `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> P: vec4<u32>;
fn erf(z: f32) -> f32 {
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, z >= 0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x) { return; }
  let v = x[i];
  x[i] = 0.5 * v * (1.0 + erf(v * 0.70710678118654752));
}`;

// x = max(x, 0) over P.x numbers.
export const RELU = `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x) { return; }
  x[i] = max(x[i], 0.0);
}`;

// Every row attends to every row: one workgroup per (head, query row). q, k
// and v are [rows][heads * hd]; out the same. P = (heads, hd, rows, scale bits).
// At most 512 rows, a BERT model's positions.
export const ATTENTION_ALL = `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> score: array<f32, 512>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let m = wg.y;
  let t = lid.x;
  let hd = P.y;
  let rows = P.z;
  let width = P.x * hd;
  let scale = bitcast<f32>(P.w);
  for (var j = t; j < rows; j += 64u) {
    var s = 0.0;
    for (var i = 0u; i < hd; i++) { s += q[m * width + h * hd + i] * k[j * width + h * hd + i]; }
    score[j] = s * scale;
  }
  workgroupBarrier();
  var best = -3.0e38;
  for (var j = t; j < rows; j += 64u) { best = max(best, score[j]); }
  red[t] = best;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
  }
  let top = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = t; j < rows; j += 64u) { let e = exp(score[j] - top); score[j] = e; sum += e; }
  red[t] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let total = red[0];
  for (var i = t; i < hd; i += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < rows; j++) { acc += score[j] * v[j * width + h * hd + i]; }
    out[m * width + h * hd + i] = acc / total;
  }
}`;
