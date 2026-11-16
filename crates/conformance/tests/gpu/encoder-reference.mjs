// The CPU reference GP-13 and GP-14 compare the WebGPU encoder with: a BERT-
// style encoder computed in double precision from weights decoded to numbers,
// written from the published description (Devlin et al. 2019; DistilBERT,
// Sanh et al. 2019) and transformers' modeling code, not from site/app/engine.

import { f16bits, f16value, generator, packMatMulNBits } from "./reference.mjs";

// erf by its Taylor series, which in double precision is within 1e-12 for
// |x| < 5; beyond, erf is within 1e-11 of +-1.
function erf(x) {
  if (Math.abs(x) >= 5) return Math.sign(x);
  let term = x;
  let sum = x;
  for (let n = 1; n < 120; n++) {
    term *= (-x * x) / n;
    sum += term / (2 * n + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}

// A small encoder with the exports' formats: q4f16 matrices as MatMulNBits'
// two tensors, f16 biases and LayerNorm weights, f16 embedding tables. "bert"
// has token types (all-MiniLM-L6-v2's shape); "distil" has none and a
// classifier head of two labels (DistilBERT SST-2's shape).
export function tinyEncoder(seed, kind) {
  const random = generator(seed);
  const dims = { d: 128, n_heads: 4, ff: 256, n_layers: 2, vocab: 320, positions: 64, eps: 1e-12 };
  const tensors = new Map();
  const halves = (n, draw) => {
    const h = new Uint16Array(n);
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      h[i] = f16bits(draw());
      values[i] = f16value(h[i]);
    }
    return { bytes: new Uint8Array(h.buffer), values };
  };
  const signed = (low, high) => () => (random() < 0.5 ? -1 : 1) * (low + (high - low) * random());
  const matrix = (name, n, k) => {
    const codes = new Int8Array(n * k);
    for (let i = 0; i < n * k; i++) codes[i] = Math.floor(random() * 16) - 8;
    const scales = new Float32Array((n * k) / 32);
    for (let i = 0; i < scales.length; i++) scales[i] = f16value(f16bits(0.004 + 0.01 * random()));
    tensors.set(name, { fmt: "q4f16", N: n, K: k, parts: packMatMulNBits(codes, scales, n, k, "f16"), codes, scales });
  };
  const vector = (name, n, low, span) => {
    const { bytes, values } = halves(n, () => low + span * random());
    tensors.set(name, { fmt: "f16", N: 1, K: n, bytes, values });
  };
  const table = (name, rows) => {
    const { bytes, values } = halves(rows * dims.d, signed(0.01, 0.2));
    tensors.set(name, { fmt: "f16", N: rows, K: dims.d, bytes, values });
  };
  const { d, ff } = dims;
  table("word", dims.vocab);
  table("position", dims.positions);
  if (kind === "bert") table("type", 2);
  vector("emb_norm", d, 0.8, 0.4);
  vector("emb_norm_b", d, -0.1, 0.2);
  for (let l = 0; l < dims.n_layers; l++) {
    const p = `l${l}.`;
    for (const w of ["q", "k", "v", "o"]) {
      matrix(`${p}w${w}`, d, d);
      vector(`${p}b${w}`, d, -0.2, 0.4);
    }
    vector(`${p}attn_norm`, d, 0.8, 0.4);
    vector(`${p}attn_norm_b`, d, -0.1, 0.2);
    matrix(`${p}w1`, ff, d);
    vector(`${p}b1`, ff, -0.2, 0.4);
    matrix(`${p}w2`, d, ff);
    vector(`${p}b2`, d, -0.2, 0.4);
    vector(`${p}out_norm`, d, 0.8, 0.4);
    vector(`${p}out_norm_b`, d, -0.1, 0.2);
  }
  if (kind === "distil") {
    const dense = (name, n) => {
      const { bytes, values } = halves(n * d, signed(0.005, 0.12));
      tensors.set(name, { fmt: "f16", N: n, K: d, bytes, values });
    };
    dense("pre", d);
    vector("pre_b", d, -0.1, 0.2);
    dense("cls", 2);
    vector("cls_b", 2, -0.1, 0.2);
  }
  const sentence = (n) => Array.from({ length: n }, () => Math.floor(random() * dims.vocab));
  return { dims, tensors, sentences: [sentence(6), sentence(60)] };
}

export function referenceEncoder({ dims, tensors }) {
  const { d, n_heads: heads, ff, n_layers: layers, eps } = dims;
  const hd = d / heads;
  const values = (name) => {
    const t = tensors.get(name);
    if (t.codes) return Float64Array.from(t.codes, (c, g) => c * t.scales[Math.floor(g / 32)]);
    return Float64Array.from(t.values);
  };
  // y[n] = sum_k w[n][k] x[k] + b[n], for each row of x.
  const linear = (rows, w, b, n, k) => rows.map((x) => {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b ? b[i] : 0;
      for (let j = 0; j < k; j++) s += w[i * k + j] * x[j];
      y[i] = s;
    }
    return y;
  });
  const layerNorm = (x, g, b) => {
    const mean = x.reduce((a, v) => a + v, 0) / x.length;
    const variance = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
    return x.map((v, i) => ((v - mean) / Math.sqrt(variance + eps)) * g[i] + b[i]);
  };
  const W = Object.fromEntries([...tensors.keys()].map((name) => [name, values(name)]));

  function hidden(tokens) {
    let x = tokens.map((t, m) => {
      const row = new Float64Array(d);
      for (let k = 0; k < d; k++) row[k] = W.word[t * d + k] + W.position[m * d + k] + (W.type ? W.type[k] : 0);
      return layerNorm(row, W.emb_norm, W.emb_norm_b);
    });
    for (let l = 0; l < layers; l++) {
      const p = (n) => W[`l${l}.${n}`];
      const q = linear(x, p("wq"), p("bq"), d, d);
      const k = linear(x, p("wk"), p("bk"), d, d);
      const v = linear(x, p("wv"), p("bv"), d, d);
      const att = x.map((_, m) => {
        const out = new Float64Array(d);
        for (let h = 0; h < heads; h++) {
          const scores = k.map((kj) => {
            let s = 0;
            for (let i = 0; i < hd; i++) s += q[m][h * hd + i] * kj[h * hd + i];
            return s / Math.sqrt(hd);
          });
          const top = Math.max(...scores);
          const e = scores.map((s) => Math.exp(s - top));
          const total = e.reduce((a, b) => a + b, 0);
          for (let i = 0; i < hd; i++) {
            let s = 0;
            for (let j = 0; j < e.length; j++) s += e[j] * v[j][h * hd + i];
            out[h * hd + i] = s / total;
          }
        }
        return out;
      });
      const o = linear(att, p("wo"), p("bo"), d, d);
      x = x.map((row, m) => layerNorm(row.map((val, i) => val + o[m][i]), p("attn_norm"), p("attn_norm_b")));
      const h1 = linear(x, p("w1"), p("b1"), ff, d).map((row) => row.map((val) => 0.5 * val * (1 + erf(val / Math.SQRT2))));
      const h2 = linear(h1, p("w2"), p("b2"), d, ff);
      x = x.map((row, m) => layerNorm(row.map((val, i) => val + h2[m][i]), p("out_norm"), p("out_norm_b")));
    }
    return x;
  }

  function embed(tokens) {
    const x = hidden(tokens);
    const mean = new Float64Array(d);
    for (const row of x) for (let k = 0; k < d; k++) mean[k] += row[k] / x.length;
    const length = Math.sqrt(mean.reduce((a, v) => a + v * v, 0));
    return mean.map((v) => v / length);
  }

  function classify(tokens) {
    const first = hidden(tokens)[0];
    const pre = linear([first], W.pre, W.pre_b, d, d)[0].map((v) => Math.max(0, v));
    const logits = linear([pre], W.cls, W.cls_b, W.cls_b.length, d)[0];
    const top = Math.max(...logits);
    const e = Array.from(logits, (v) => Math.exp(v - top));
    const sum = e.reduce((a, b) => a + b, 0);
    return { logits, probabilities: e.map((v) => v / sum) };
  }

  return { hidden, embed, classify };
}
