// The compute shaders of one decode step. Adapted from Holo Q's WebGPU engine
// (Hologram-Technologies/hologram-apps, apps/q/qvac-gpu.js, MIT); the ternary
// matrix-vector product, YaRN rotation and per-head norms are new here.
//
// A ternary weight block is three arrays: a sign plane and a nonzero plane of
// one bit per weight (bit g = row * K + column, least significant first), and
// one f32 scale per 128 weights. A weight is +scale or -scale where its nonzero
// bit is set, by its sign bit, and 0 otherwise.

// out[n] = (+)= sum_k x[k] * w[n][k]. One workgroup per output row; 64 threads
// stride over the row's 32-weight words. P = (K, N, accumulate, 0).
export const MATVEC_TERNARY = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<u32>;
@group(0) @binding(2) var<storage, read> nonzeros: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * 65535u + wg.x;
  if (n >= P.y) { return; }
  let t = lid.x;
  let words = P.x >> 5u;
  let groups = P.x >> 7u;
  var acc = 0.0;
  for (var w = t; w < words; w += 64u) {
    let s = signs[n * words + w];
    let z = nonzeros[n * words + w];
    let k0 = w << 5u;
    var part = 0.0;
    for (var j = 0u; j < 32u; j++) {
      let bit = 1u << j;
      if ((z & bit) != 0u) {
        let v = x[k0 + j];
        part += select(-v, v, (s & bit) != 0u);
      }
    }
    acc += part * scales[n * groups + (w >> 2u)];
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

// o = x / rms(x) * gamma over n values. P = (n, bits of eps, 0, 0).
export const RMS_NORM = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let n = P.x;
  var s = 0.0;
  for (var i = t; i < n; i += 256u) { s += x[i] * x[i]; }
  red[t] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(n) + bitcast<f32>(P.y));
  for (var i = t; i < n; i += 256u) { o[i] = x[i] * r * gamma[i]; }
}`;

// The same norm applied to each head of v in place, with one gamma shared by
// every head (Qwen3's q_norm and k_norm). One workgroup per head. P = (hd, eps).
export const HEAD_NORM = `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let hd = P.x;
  let base = wg.x * hd;
  var s = 0.0;
  for (var i = t; i < hd; i += 64u) { s += v[base + i] * v[base + i]; }
  red[t] = s;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(hd) + bitcast<f32>(P.y));
  for (var i = t; i < hd; i += 64u) { v[base + i] = v[base + i] * r * gamma[i]; }
}`;

// Rotate each head of v in place by position, pairing element i with i + hd/2
// (the NeoX layout Qwen uses), at the given frequencies, scaled by mscale.
// One thread per (head, pair). P = (heads, hd, position, bits of mscale).
export const ROPE = `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = P.y / 2u;
  let idx = gid.x;
  if (idx >= P.x * half) { return; }
  let h = idx / half;
  let i = idx % half;
  let base = h * P.y;
  let angle = f32(P.z) * freqs[i];
  let m = bitcast<f32>(P.w);
  let c = cos(angle) * m;
  let s = sin(angle) * m;
  let a = v[base + i];
  let b = v[base + i + half];
  v[base + i] = a * c - b * s;
  v[base + i + half] = a * s + b * c;
}`;

// Write this position's key and value into the layer's cache. P = (kvdim, position).
export const KV_WRITE = `
@group(0) @binding(0) var<storage, read> k: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x) { return; }
  kc[P.y * P.x + i] = k[i];
  vc[P.y * P.x + i] = v[i];
}`;

// Causal attention for one query position over the cache, grouped-query: query
// head h reads key/value head h / (heads / kvheads). One workgroup per query
// head. P = (heads, kvheads, hd, length).
export const ATTENTION = (capacity) => `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> score: array<f32, ${capacity}>;
var<workgroup> top: array<f32, 64>;
var<workgroup> total: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let h = wg.x;
  let hd = P.z;
  let len = P.w;
  let kvdim = P.y * hd;
  let kvh = h / (P.x / P.y);
  let qb = h * hd;
  let kb = kvh * hd;
  let scale = inverseSqrt(f32(hd));
  var m = -3.4e38;
  for (var p = t; p < len; p += 64u) {
    var d = 0.0;
    for (var i = 0u; i < hd; i++) { d += q[qb + i] * kc[p * kvdim + kb + i]; }
    score[p] = d * scale;
    m = max(m, score[p]);
  }
  top[t] = m;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { top[t] = max(top[t], top[t + stride]); }
    workgroupBarrier();
  }
  let best = top[0];
  var s = 0.0;
  for (var p = t; p < len; p += 64u) {
    let e = exp(score[p] - best);
    score[p] = e;
    s += e;
  }
  total[t] = s;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) { total[t] += total[t + stride]; }
    workgroupBarrier();
  }
  let inv = 1.0 / total[0];
  for (var i = t; i < hd; i += 64u) {
    var acc = 0.0;
    for (var p = 0u; p < len; p++) { acc += score[p] * vc[p * kvdim + kb + i]; }
    o[qb + i] = acc * inv;
  }
}`;

// g = silu(g) * u, in place. P = (n).
export const SILU_MUL = `
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x) { return; }
  let a = g[i];
  g[i] = a / (1.0 + exp(-a)) * u[i];
}`;

// x = the embedding row of token, decoded from the ternary planes. P = (K, token).
export const EMBED_TERNARY = `
@group(0) @binding(0) var<storage, read> signs: array<u32>;
@group(0) @binding(1) var<storage, read> nonzeros: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= P.x) { return; }
  let g = P.y * P.x + k;
  let bit = 1u << (g & 31u);
  let z = (nonzeros[g >> 5u] & bit) != 0u;
  let s = (signs[g >> 5u] & bit) != 0u;
  let w = scales[g >> 7u];
  x[k] = select(0.0, select(-w, w, s), z);
}`;

// The index of the first maximum of n logits. One workgroup. P = (n).
export const ARGMAX = `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
var<workgroup> value: array<f32, 256>;
var<workgroup> slot: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var bv = -3.4e38;
  var bi = 0xffffffffu;
  for (var i = t; i < P.x; i += 256u) {
    if (logits[i] > bv) { bv = logits[i]; bi = i; }
  }
  value[t] = bv;
  slot[t] = bi;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      let ov = value[t + stride];
      let oi = slot[t + stride];
      if (ov > value[t] || (ov == value[t] && oi < slot[t])) { value[t] = ov; slot[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { out[0] = slot[0]; }
}`;
