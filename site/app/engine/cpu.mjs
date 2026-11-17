// The CPU decode engine: the same models as gpu.mjs, run with WebGPU off.
// It reads the same tensors in the same layouts (q4 and q4f16 matrices as
// MatMulNBits holds them, dense f32 or f16 embeddings and heads, f32 or f16
// vectors), repacks none of them, and answers with the same interface:
// step, run, generate, reset, position, saveState, restoreState.
//
// Every dot product is summed in double precision in a fixed order, so a step
// gives the same numbers on every machine; the activations between steps are
// kept as single precision, as the GPU keeps them. GitHub Pages cannot enable
// threads, so this is one thread and is for small models; which models may
// run here is the kernel's decision 22 (the CPU ceiling), not this module's.

import { llama3, plain, yarn } from "./yarn.mjs";

// Every f16 as the f32 that holds it exactly, by its bits.
const HALF = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const e = (h >> 10) & 31;
    const m = h & 1023;
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1024 + m) * 2 ** (e - 25);
    table[h] = h & 0x8000 ? -v : v;
  }
  return table;
})();

const halves = (bytes) => {
  const h = new Uint16Array(bytes.byteLength / 2);
  new Uint8Array(h.buffer).set(bytes);
  return h;
};

export function createCpuEngine({ dims, tensors, capacity = 1024 }) {
  const { d, n_heads: heads, n_kv_heads: kvHeads, hd, ff, vocab, n_layers: layers } = dims;
  const eps = dims.eps ?? 1e-6;
  const qdim = heads * hd;
  const kvdim = kvHeads * hd;
  const group = heads / kvHeads;
  const rope = dims.yarn
    ? yarn({ hd, base: dims.rope_base, factor: dims.yarn.factor, orig: dims.yarn.orig })
    : dims.llama3
      ? llama3({ hd, base: dims.rope_base, ...dims.llama3 })
      : plain({ hd, base: dims.rope_base });

  const tensor = (name) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`model has no tensor ${name}`);
    return t;
  };

  // A q4 or q4f16 matrix: its codes as bytes, two to a byte, low nibble
  // first, and its scales widened to f32 (exact). A dense f32 or f16 matrix:
  // its numbers, read by row.
  const matrix = (name) => {
    const t = tensor(name);
    if (t.K % 32) throw new Error(`${name} has ${t.K} columns; this engine runs whole blocks of 32`);
    if (t.fmt === "q4" || t.fmt === "q4f16") {
      const codesLength = (t.N * t.K) / 2;
      const [codes, scaleBytes] = t.parts ?? [t.bytes.subarray(0, codesLength), t.bytes.subarray(codesLength)];
      const blocks = (t.N * t.K) / 32;
      const wide = t.fmt === "q4f16" ? 2 : 4;
      if (codes.byteLength !== codesLength || scaleBytes.byteLength !== blocks * wide) throw new Error(`${name} holds the wrong number of bytes for ${t.fmt}`);
      const scales = t.fmt === "q4f16" ? Float32Array.from(halves(scaleBytes), (h) => HALF[h]) : new Float32Array(scaleBytes.slice().buffer);
      return { kind: "q4", N: t.N, K: t.K, codes, scales };
    }
    if (t.fmt === "f32" || t.fmt === "f16") {
      const size = t.fmt === "f32" ? 4 : 2;
      if (t.bytes.byteLength !== t.N * t.K * size) throw new Error(`${name} holds the wrong number of bytes for ${t.fmt}`);
      return { kind: t.fmt, N: t.N, K: t.K, values: t.fmt === "f32" ? new Float32Array(t.bytes.slice().buffer) : halves(t.bytes) };
    }
    throw new Error(`${name} is ${t.fmt}; this engine runs q4, q4f16, f32 and f16 matrices`);
  };
  const vector = (name) => {
    const t = tensor(name);
    if (t.fmt === "f16") return Float32Array.from(halves(t.bytes), (h) => HALF[h]);
    if (t.fmt !== "f32") throw new Error(`${name} is ${t.fmt}, not f32 or f16`);
    return new Float32Array(t.bytes.slice().buffer);
  };

  // out[n] = sum_k w[n][k] x[k] for rows first .. first + count of w.
  const matvec = (w, x, out, accumulate = false) => {
    const { N, K } = w;
    if (w.kind === "q4") {
      const rowBytes = K / 2;
      const blocks = K / 32;
      for (let n = 0; n < N; n++) {
        let acc = 0;
        const row = n * rowBytes;
        for (let b = 0; b < blocks; b++) {
          let s = 0;
          const at = row + b * 16;
          const k0 = b * 32;
          for (let j = 0; j < 16; j++) {
            const byte = w.codes[at + j];
            s += x[k0 + 2 * j] * ((byte & 15) - 8) + x[k0 + 2 * j + 1] * ((byte >> 4) - 8);
          }
          acc += s * w.scales[n * blocks + b];
        }
        out[n] = accumulate ? out[n] + acc : acc;
      }
      return;
    }
    const v = w.values;
    const f16 = w.kind === "f16";
    for (let n = 0; n < N; n++) {
      let acc = 0;
      const row = n * K;
      if (f16) for (let k = 0; k < K; k++) acc += x[k] * HALF[v[row + k]];
      else for (let k = 0; k < K; k++) acc += x[k] * v[row + k];
      out[n] = accumulate ? out[n] + acc : acc;
    }
  };
  // Row `r` of an embedding, written into x.
  const embedRow = (w, r, x) => {
    if (r < 0 || r >= w.N) throw new Error(`token ${r} is outside the vocabulary of ${w.N}`);
    if (w.kind === "q4") {
      const rowBytes = w.K / 2;
      for (let k = 0; k < w.K; k++) {
        const byte = w.codes[r * rowBytes + (k >> 1)];
        const code = k & 1 ? byte >> 4 : byte & 15;
        x[k] = (code - 8) * w.scales[(r * w.K + k) >> 5];
      }
    } else if (w.kind === "f16") for (let k = 0; k < w.K; k++) x[k] = HALF[w.values[r * w.K + k]];
    else for (let k = 0; k < w.K; k++) x[k] = w.values[r * w.K + k];
  };
  const rmsNorm = (x, gamma, out, n = x.length, at = 0) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[at + i] * x[at + i];
    const r = 1 / Math.sqrt(s / n + eps);
    for (let i = 0; i < n; i++) out[at + i] = x[at + i] * r * gamma[i];
  };
  const rotate = (v, count, pos) => {
    const half = hd / 2;
    for (let h = 0; h < count; h++) {
      const base = h * hd;
      for (let i = 0; i < half; i++) {
        const angle = pos * rope.freqs[i];
        const c = Math.cos(angle) * rope.mscale;
        const s = Math.sin(angle) * rope.mscale;
        const a = v[base + i];
        const b = v[base + i + half];
        v[base + i] = a * c - b * s;
        v[base + i + half] = a * s + b * c;
      }
    }
  };

  const embed = matrix("embed");
  if (embed.K !== d || embed.N !== vocab) throw new Error(`embed is ${embed.N} by ${embed.K}, the model needs ${vocab} by ${d}`);
  const embedTensor = tensor("embed");
  const headTensor = tensors.get("lm_head");
  const tied = !headTensor || (headTensor.bytes ?? headTensor.parts?.[0]) === (embedTensor.bytes ?? embedTensor.parts?.[0]);
  const head = tied ? embed : matrix("lm_head");
  const qkNorm = tensors.has("l0.q_norm");
  const bias = tensors.has("l0.bq");
  const layer = [];
  for (let l = 0; l < layers; l++) {
    const p = `l${l}.`;
    const L = {
      attnNorm: vector(`${p}attn_norm`), wq: matrix(`${p}wq`), wk: matrix(`${p}wk`), wv: matrix(`${p}wv`), wo: matrix(`${p}wo`),
      ffnNorm: vector(`${p}ffn_norm`), gate: matrix(`${p}w_gate`), up: matrix(`${p}w_up`), down: matrix(`${p}w_down`),
      qNorm: qkNorm ? vector(`${p}q_norm`) : null, kNorm: qkNorm ? vector(`${p}k_norm`) : null,
      bq: bias ? vector(`${p}bq`) : null, bk: bias ? vector(`${p}bk`) : null, bv: bias ? vector(`${p}bv`) : null,
      kc: new Float32Array(capacity * kvdim), vc: new Float32Array(capacity * kvdim),
    };
    for (const [w, N, K] of [[L.wq, qdim, d], [L.wk, kvdim, d], [L.wv, kvdim, d], [L.wo, d, qdim], [L.gate, ff, d], [L.up, ff, d], [L.down, d, ff]]) {
      if (w.N !== N || w.K !== K) throw new Error(`layer ${l}: a matrix is ${w.N} by ${w.K}, the model needs ${N} by ${K}`);
    }
    layer.push(L);
  }
  const finalNorm = vector("final_norm");

  const x = new Float32Array(d);
  const h = new Float32Array(d);
  const q = new Float32Array(qdim);
  const k = new Float32Array(kvdim);
  const v = new Float32Array(kvdim);
  const att = new Float32Array(qdim);
  const g = new Float32Array(ff);
  const u = new Float32Array(ff);
  const logits = new Float32Array(vocab);
  const scores = new Float64Array(capacity);
  const scale = 1 / Math.sqrt(hd);
  let position = 0;

  // One token at the next position: the greedy pick, the lowest id among
  // equal best logits as the GPU's argmax takes it, and the logits if asked.
  function stepSync(token, wantLogits = false) {
    if (position >= capacity) throw new Error(`context full: ${capacity} positions`);
    const pos = position;
    embedRow(embed, token, x);
    for (const L of layer) {
      rmsNorm(x, L.attnNorm, h);
      matvec(L.wq, h, q);
      matvec(L.wk, h, k);
      matvec(L.wv, h, v);
      if (bias) {
        for (let i = 0; i < qdim; i++) q[i] += L.bq[i];
        for (let i = 0; i < kvdim; i++) { k[i] += L.bk[i]; v[i] += L.bv[i]; }
      }
      if (qkNorm) {
        for (let hh = 0; hh < heads; hh++) rmsNorm(q, L.qNorm, q, hd, hh * hd);
        for (let hh = 0; hh < kvHeads; hh++) rmsNorm(k, L.kNorm, k, hd, hh * hd);
      }
      rotate(q, heads, pos);
      rotate(k, kvHeads, pos);
      L.kc.set(k, pos * kvdim);
      L.vc.set(v, pos * kvdim);
      for (let hh = 0; hh < heads; hh++) {
        const kvh = Math.floor(hh / group);
        let top = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += q[hh * hd + i] * L.kc[j * kvdim + kvh * hd + i];
          scores[j] = s * scale;
          if (scores[j] > top) top = scores[j];
        }
        let total = 0;
        for (let j = 0; j <= pos; j++) { scores[j] = Math.exp(scores[j] - top); total += scores[j]; }
        for (let i = 0; i < hd; i++) {
          let s = 0;
          for (let j = 0; j <= pos; j++) s += scores[j] * L.vc[j * kvdim + kvh * hd + i];
          att[hh * hd + i] = s / total;
        }
      }
      matvec(L.wo, att, x, true);
      rmsNorm(x, L.ffnNorm, h);
      matvec(L.gate, h, g);
      matvec(L.up, h, u);
      for (let i = 0; i < ff; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      matvec(L.down, g, x, true);
    }
    rmsNorm(x, finalNorm, h);
    matvec(head, h, logits);
    let best = 0;
    for (let i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    position = pos + 1;
    return { token: best, logits: wantLogits ? logits.slice() : null };
  }

  async function step(token, { logits: wantLogits = false } = {}) {
    return stepSync(token, wantLogits);
  }
  // The prompt, then `count` greedy picks, stopping at any id in `stops`.
  async function run(prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) {
    if (position + prompt.length + count > capacity) throw new Error(`context full: ${capacity} positions`);
    let last = null;
    for (const t of prompt) last = stepSync(t).token;
    onFirst();
    const out = [];
    while (out.length < count) {
      if (stops.has(last)) break;
      out.push(last);
      onToken(out.slice());
      if (out.length < count) last = stepSync(last).token;
    }
    return out;
  }
  async function generate(prompt, count) {
    return run(prompt, count);
  }
  function saveState(length = position) {
    if (!(length >= 1 && length <= position)) throw new Error(`prefix: ${length} positions asked, ${position} computed`);
    return { length, shape: { layers, kvdim, capacity }, kept: layer.map((L) => ({ k: L.kc.slice(0, length * kvdim), v: L.vc.slice(0, length * kvdim) })) };
  }
  function restoreState(state) {
    if (state.shape.layers !== layers || state.shape.kvdim !== kvdim) throw new Error(`prefix: saved from ${state.shape.layers} layers of ${state.shape.kvdim}, this model has ${layers} of ${kvdim}`);
    if (state.length > capacity) throw new Error(`prefix: ${state.length} positions, this engine holds ${capacity}`);
    state.kept.forEach((saved, l) => {
      layer[l].kc.set(saved.k);
      layer[l].vc.set(saved.v);
    });
    position = state.length;
  }

  return {
    step, run, generate, saveState, restoreState,
    reset: () => { position = 0; },
    get position() { return position; },
    device: "cpu",
    rope,
  };
}
