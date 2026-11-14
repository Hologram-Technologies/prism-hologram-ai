// The prompt's kernels: a block of M positions at a time. Each kernel does, for
// each position, exactly the arithmetic its one-position counterpart in
// wgsl.mjs does, in the same order, so a prompt run in blocks gives the same
// bits as one run position by position. What changes is how often a weight is
// read: once per block instead of once per position.
//
// Rows are laid out position after position: row m of a d-wide buffer is
// [m * d, (m + 1) * d).

// o[m][n] = (+)= sum_k x[m][k] * w[n][k] for m < count. One workgroup per output
// row n, 64 lanes, as MATVEC_TERNARY; each weight word is decoded once and used
// for every position. P = (K, N, accumulate, count).
export const MATMUL_TERNARY = (M) => `
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> signs: array<u32>;
@group(0) @binding(2) var<storage, read> nonzeros: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, ${64 * M}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * 65535u + wg.x;
  if (n >= P.y) { return; }
  let t = lid.x;
  let words = P.x >> 5u;
  let groups = P.x >> 7u;
  let row4 = P.x >> 2u;
  let count = P.w;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < ${M}u; m++) { acc[m] = 0.0; }
  for (var w = t; w < words; w += 64u) {
    let s = signs[n * words + w];
    let z = nonzeros[n * words + w];
    let x0 = w << 3u;
    var part: array<f32, ${M}>;
    for (var m = 0u; m < ${M}u; m++) { part[m] = 0.0; }
    for (var q = 0u; q < 8u; q++) {
      let zz = (z >> (q * 4u)) & 0xfu;
      let ss = (s >> (q * 4u)) & 0xfu;
      let nz = vec4<f32>(f32(zz & 1u), f32((zz >> 1u) & 1u), f32((zz >> 2u) & 1u), f32((zz >> 3u) & 1u));
      let sg = vec4<f32>(f32(ss & 1u), f32((ss >> 1u) & 1u), f32((ss >> 2u) & 1u), f32((ss >> 3u) & 1u));
      let wv = nz * (sg * 2.0 - 1.0);
      for (var m = 0u; m < count; m++) { part[m] += dot(x4[m * row4 + x0 + q], wv); }
    }
    let sc = scales[n * groups + (w >> 2u)];
    for (var m = 0u; m < count; m++) { acc[m] += part[m] * sc; }
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

// Row m of o = row m of x / rms * gamma. One workgroup per position, as RMS_NORM.
// P = (n, bits of eps, 0, 0).
export const RMS_NORM_ROWS = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> P: vec4<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let n = P.x;
  let base = wg.x * n;
  var s = 0.0;
  for (var i = t; i < n; i += 256u) { s += x[base + i] * x[base + i]; }
  red[t] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(n) + bitcast<f32>(P.y));
  for (var i = t; i < n; i += 256u) { o[base + i] = x[base + i] * r * gamma[i]; }
}`;

// Row m of x = the embedding of the token at ring[first + m]. P = (K, first, count).
export const EMBED_ROWS = `
@group(0) @binding(0) var<storage, read> signs: array<u32>;
@group(0) @binding(1) var<storage, read> nonzeros: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> ring: array<u32>;
@group(0) @binding(5) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * P.z) { return; }
  let m = i / P.x;
  let k = i % P.x;
  let g = ring[P.y + m] * P.x + k;
  let bit = 1u << (g & 31u);
  let z = (nonzeros[g >> 5u] & bit) != 0u;
  let s = (signs[g >> 5u] & bit) != 0u;
  let w = scales[g >> 7u];
  x[i] = select(0.0, select(-w, w, s), z);
}`;

// ROPE for each position of the block: row m turns at position first + m.
// P = (heads, hd, first, bits of mscale); count rows dispatched.
export const ROPE_ROWS = `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = P.y / 2u;
  let perRow = P.x * half;
  let m = gid.y;
  let idx = gid.x;
  if (idx >= perRow) { return; }
  let h = idx / half;
  let i = idx % half;
  let base = m * P.x * P.y + h * P.y;
  let angle = f32(P.z + m) * freqs[i];
  let mm = bitcast<f32>(P.w);
  let c = cos(angle) * mm;
  let s = sin(angle) * mm;
  let a = v[base + i];
  let b = v[base + i + half];
  v[base + i] = a * c - b * s;
  v[base + i + half] = a * s + b * c;
}`;

// Each position's key and value into the cache. P = (kvdim, first, count).
export const KV_WRITE_ROWS = `
@group(0) @binding(0) var<storage, read> k: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.x * P.z) { return; }
  kc[P.y * P.x + i] = k[i];
  vc[P.y * P.x + i] = v[i];
}`;

// Causal attention for each position of the block, as ATTENTION: position
// first + m sees the cache up to and including itself. One workgroup per
// (head, position). P = (heads, kvheads, hd, first + 1).
export const ATTENTION_ROWS = (capacity) => `
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
  let m = wg.y;
  let hd = P.z;
  let len = P.w + m;
  let kvdim = P.y * hd;
  let kvh = h / (P.x / P.y);
  let qb = m * P.x * hd + h * hd;
  let kb = kvh * hd;
  let scale = inverseSqrt(f32(hd));
  var mx = -3.4e38;
  for (var p = t; p < len; p += 64u) {
    var d = 0.0;
    for (var i = 0u; i < hd; i++) { d += q[qb + i] * kc[p * kvdim + kb + i]; }
    score[p] = d * scale;
    mx = max(mx, score[p]);
  }
  top[t] = mx;
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
