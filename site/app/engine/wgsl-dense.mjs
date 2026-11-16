// Kernels for a dense f32 or f16 matrix: the embedding of an ONNX export,
// which the export's output head reads too (its Transpose feeds a MatMul), so
// one tensor serves both. Its rows are [N][K] in the file, the order the
// engine's matrices already use.
//
// A head this size (151936 by 1536 f16 is 467 MB) is larger than one binding
// on many devices, so the engine splits it into tiles of whole rows, one buffer
// each. A tile names its first row; the matrix-vector kernel writes its rows
// from there, and the embedding kernel copies a token's row only from the tile
// that holds it.

// The binding for the weights and the read of four along a row from vec4 i.
// f16 is read as two u32 per four numbers and widened, which is exact.
const WEIGHTS = {
  f32: `array<vec4<f32>>;
fn four(i: u32) -> vec4<f32> { return w[i]; }
fn one(e: u32) -> f32 { return w[e >> 2u][e & 3u]; }`,
  f16: `array<vec2<u32>>;
fn four(i: u32) -> vec4<f32> { let p = w[i]; return vec4<f32>(unpack2x16float(p.x), unpack2x16float(p.y)); }
fn one(e: u32) -> f32 { return unpack2x16float(w[e >> 2u][(e >> 1u) & 1u])[e & 1u]; }`,
};

// One output row per workgroup. P = (K, rows in this tile, accumulate, first row).
export const matvecDense = (s) => `
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: ${WEIGHTS[s]}
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * 65535u + wg.x;
  if (n >= P.y) { return; }
  let t = lid.x;
  let quads = P.x >> 2u;
  var acc = 0.0;
  for (var j = t; j < quads; j += 64u) {
    acc += dot(x4[j], four(n * quads + j));
  }
  red[t] = acc;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  if (t == 0u) {
    let at = P.w + n;
    if (P.z == 1u) { o[at] = o[at] + red[0]; } else { o[at] = red[0]; }
  }
}`;

// x = the embedding rows of tokens ring[P.y .. P.y + P.z), P.x wide, taken
// from the tile whose rows are T.x .. T.x + T.y. A position run alone is P.z = 1.
export const embedDense = (s) => `
@group(0) @binding(0) var<storage, read> w: ${WEIGHTS[s]}
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> ring: array<u32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
@group(0) @binding(4) var<uniform> T: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * max(P.z, 1u)) { return; }
  let m = i / P.x;
  let k = i % P.x;
  let token = ring[P.y + m];
  if (token < T.x || token >= T.x + T.y) { return; }
  x[i] = one((token - T.x) * P.x + k);
}`;
