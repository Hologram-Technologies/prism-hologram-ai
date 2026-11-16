// Attention split across the cache. One workgroup per (head, 64-position chunk,
// row) scores its chunk and keeps a partial: the chunk's largest score, the sum
// of its exponentials and its weighted values. A second kernel combines the
// partials per (head, row). The chunks are fixed by position, so the same
// position gives the same bits whether it runs alone or in a prompt's block.
//
// Rows: row r of a block is position first + r, so it sees len = first + 1 + r
// cached positions. A position run alone is a block of one row.
//
// part holds, per (row, head, chunk), [max, sum, value[hd]]: stride hd + 2.

const CHUNK = 64;

// P = (heads, kvheads, hd, first + 1). Dispatch (heads, chunks of the last
// row, rows).
export const ATTENTION_PARTIAL = (maxChunks) => `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<u32>;
var<workgroup> weight: array<f32, ${CHUNK}>;
var<workgroup> red: array<f32, ${CHUNK}>;
@compute @workgroup_size(${CHUNK})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let c = wg.y;
  let r = wg.z;
  let t = lid.x;
  let hd = P.z;
  let len = P.w + r;
  let kvdim = P.y * hd;
  let kvh = h / (P.x / P.y);
  let qb = (r * P.x + h) * hd;
  let kb = kvh * hd;
  let slot = ((r * P.x + h) * ${maxChunks}u + c) * (hd + 2u);
  let start = c * ${CHUNK}u;
  let p = start + t;
  var s = -3.4e38;
  if (p < len) {
    var d = 0.0;
    for (var i = 0u; i < hd; i++) { d += q[qb + i] * kc[p * kvdim + kb + i]; }
    s = d * inverseSqrt(f32(hd));
  }
  red[t] = s;
  workgroupBarrier();
  for (var stride = ${CHUNK / 2}u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  let e = select(0.0, exp(s - m), p < len);
  weight[t] = e;
  red[t] = e;
  workgroupBarrier();
  for (var stride = ${CHUNK / 2}u; stride > 0u; stride >>= 1u) {
    if (t < stride) { red[t] += red[t + stride]; }
    workgroupBarrier();
  }
  let n = select(0u, min(${CHUNK}u, len - start), start < len);
  for (var i = t; i < hd; i += ${CHUNK}u) {
    var acc = 0.0;
    for (var j = 0u; j < n; j++) { acc += weight[j] * vc[(start + j) * kvdim + kb + i]; }
    part[slot + 2u + i] = acc;
  }
  if (t == 0u) {
    part[slot] = m;
    part[slot + 1u] = red[0];
  }
}`;

// o = the partials of each (head, row) combined. P as ATTENTION_PARTIAL.
// Dispatch (heads, rows).
export const ATTENTION_COMBINE = (maxChunks) => `
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(${CHUNK})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let r = wg.y;
  let t = lid.x;
  let hd = P.z;
  let len = P.w + r;
  let chunks = (len + ${CHUNK - 1}u) / ${CHUNK}u;
  let base = (r * P.x + h) * ${maxChunks}u;
  var top = -3.4e38;
  for (var c = 0u; c < chunks; c++) { top = max(top, part[(base + c) * (hd + 2u)]); }
  var total = 0.0;
  for (var c = 0u; c < chunks; c++) {
    let at = (base + c) * (hd + 2u);
    total += exp(part[at] - top) * part[at + 1u];
  }
  let inv = 1.0 / total;
  for (var i = t; i < hd; i += ${CHUNK}u) {
    var acc = 0.0;
    for (var c = 0u; c < chunks; c++) {
      let at = (base + c) * (hd + 2u);
      acc += exp(part[at] - top) * part[at + 2u + i];
    }
    o[(r * P.x + h) * hd + i] = acc * inv;
  }
}`;

export const ATTENTION_CHUNK = CHUNK;
