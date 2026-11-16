// Kernels for q4 weights and for the attention biases Qwen2 has.
//
// q4 is planar per tensor: N*K/2 bytes of codes, eight to a u32, the weight at
// index g in bits 4*(g % 8) of word g / 8 (low nibble first); then one f32
// scale per 32 weights along a row. A weight is (code - 8) * scale. KN-13 gives
// a block's length (20 bytes per 32 weights) and KN-17 the unpacking.
//
// ONNX's MatMulNBits at 4 bits and block 32, with no zero points, writes the
// same bytes: its B input is [N][K / 32][16], two codes to a byte, low nibble
// first, so row n's codes start at byte n * K / 2; its scales are [N][K / 32];
// a missing zero point is 8. So an exported weight runs from the file's own
// bytes, with no repack. Its scales are f32 in a q4 export and f16 in a q4f16
// one; f16 scales are read two to a u32 and widened in the kernel, which is
// exact, so both read the same numbers and the f16 file moves fewer bytes.
//
// The batched kernel repeats the one-position arithmetic in the same order row
// by row, so a prompt run in blocks gives the same bits as one position at a
// time.

// The scales binding and the read of scale i, for f32 or f16 scales.
const SCALES = {
  f32: `array<f32>;
fn scale(i: u32) -> f32 { return scales[i]; }`,
  f16: `array<u32>;
fn scale(i: u32) -> f32 { return unpack2x16float(scales[i >> 1u])[i & 1u]; }`,
};

// One output row per workgroup. P = (K, N, accumulate, 0).
export const matvecQ4 = (s) => `
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: ${SCALES[s]}
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * 65535u + wg.x;
  if (n >= P.y) { return; }
  let t = lid.x;
  let words = P.x >> 3u;
  let groups = P.x >> 5u;
  var acc = 0.0;
  for (var w = t; w < words; w += 64u) {
    let c = codes[n * words + w];
    let lo = vec4<f32>(f32(c & 0xfu), f32((c >> 4u) & 0xfu), f32((c >> 8u) & 0xfu), f32((c >> 12u) & 0xfu)) - 8.0;
    let hi = vec4<f32>(f32((c >> 16u) & 0xfu), f32((c >> 20u) & 0xfu), f32((c >> 24u) & 0xfu), f32(c >> 28u)) - 8.0;
    let part = dot(x4[2u * w], lo) + dot(x4[2u * w + 1u], hi);
    acc += part * scale(n * groups + (w >> 2u));
  }
  red[t] = acc;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  if (t == 0u) {
    if (P.z == 1u) { o[n] = o[n] + red[0]; } else { o[n] = red[0]; }
  }
}`;

// Up to M rows of input at once, each weight word read once. P = (K, N,
// accumulate, rows).
export const matmulQ4 = (s, M) => `
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: ${SCALES[s]}
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, ${64 * M}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * 65535u + wg.x;
  if (n >= P.y) { return; }
  let t = lid.x;
  let words = P.x >> 3u;
  let groups = P.x >> 5u;
  let row4 = P.x >> 2u;
  let count = P.w;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < ${M}u; m++) { acc[m] = 0.0; }
  for (var w = t; w < words; w += 64u) {
    let c = codes[n * words + w];
    let lo = vec4<f32>(f32(c & 0xfu), f32((c >> 4u) & 0xfu), f32((c >> 8u) & 0xfu), f32((c >> 12u) & 0xfu)) - 8.0;
    let hi = vec4<f32>(f32((c >> 16u) & 0xfu), f32((c >> 20u) & 0xfu), f32((c >> 24u) & 0xfu), f32(c >> 28u)) - 8.0;
    let sc = scale(n * groups + (w >> 2u));
    for (var m = 0u; m < count; m++) {
      let part = dot(x4[m * row4 + 2u * w], lo) + dot(x4[m * row4 + 2u * w + 1u], hi);
      acc[m] += part * sc;
    }
  }
  for (var m = 0u; m < ${M}u; m++) { red[m * 64u + t] = acc[m]; }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      for (var m = 0u; m < ${M}u; m++) { red[m * 64u + t] += red[m * 64u + t + stride]; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    for (var m = 0u; m < count; m++) {
      if (P.z == 1u) { o[m * P.y + n] = o[m * P.y + n] + red[m * 64u]; } else { o[m * P.y + n] = red[m * 64u]; }
    }
  }
}`;

// x = the embedding rows of tokens ring[P.y .. P.y + P.z); P.x is the width.
// A position run alone is P.z = 1.
export const embedQ4 = (s) => `
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: ${SCALES[s]}
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> ring: array<u32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * max(P.z, 1u)) { return; }
  let m = i / P.x;
  let k = i % P.x;
  let g = ring[P.y + m] * P.x + k;
  let c = (codes[g >> 3u] >> ((g & 7u) * 4u)) & 0xfu;
  x[i] = (f32(c) - 8.0) * scale(g >> 5u);
}`;

export const MATVEC_Q4 = matvecQ4("f32");
export const MATMUL_Q4 = (M) => matmulQ4("f32", M);
export const EMBED_Q4 = embedQ4("f32");

// x[r * P.x + i] += b[i] for rows r < P.y.
export const ADD_BIAS = `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * P.y) { return; }
  x[i] = x[i] + b[i % P.x];
}`;
