// The CPU reference GP-01 compares the WebGPU engine with: the same model
// computed in double precision from weights decoded to numbers, with its own
// rotary frequencies. It shares no code with site/app/engine, so a mistake in
// one is not repeated in the other.

// A deterministic generator, so the model and prompt are the same every run.
export function generator(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Pack an N by K matrix of codes in {-1, 0, +1}, with one scale per 128
// weights, into the ternary block layout: sign plane, nonzero plane, scales.
export function packTernary(codes, scales, n, k) {
  const plane = (n * k) / 8;
  const bytes = new Uint8Array(2 * plane + ((n * k) / 128) * 4);
  for (let g = 0; g < n * k; g++) {
    if (codes[g] !== 0) bytes[plane + (g >> 3)] |= 1 << (g & 7);
    if (codes[g] > 0) bytes[g >> 3] |= 1 << (g & 7);
  }
  new Float32Array(bytes.buffer, 2 * plane).set(scales);
  return bytes;
}

// A small model with Bonsai's shape of layers and its tensor names.
export function tinyModel(seed = 1) {
  const random = generator(seed);
  const dims = { d: 128, n_heads: 4, n_kv_heads: 2, hd: 32, ff: 256, vocab: 320, n_layers: 2, rope_base: 10000, yarn: { factor: 4, orig: 64 }, eps: 1e-6 };
  const tensors = new Map();
  const matrix = (name, n, k) => {
    const codes = new Int8Array(n * k);
    for (let i = 0; i < n * k; i++) codes[i] = Math.floor(random() * 3) - 1;
    const scales = new Float32Array((n * k) / 128);
    for (let i = 0; i < scales.length; i++) scales[i] = Math.fround(0.05 + 0.1 * random());
    tensors.set(name, { fmt: "q1", N: n, K: k, bytes: packTernary(codes, scales, n, k), codes, scales });
  };
  const vector = (name, n) => {
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = Math.fround(0.8 + 0.4 * random());
    tensors.set(name, { fmt: "f32", N: 1, K: n, bytes: new Uint8Array(values.buffer), values });
  };
  const { d, n_heads: h, n_kv_heads: kv, hd, ff, vocab } = dims;
  matrix("embed", vocab, d);
  for (let l = 0; l < dims.n_layers; l++) {
    vector(`l${l}.attn_norm`, d);
    matrix(`l${l}.wq`, h * hd, d);
    matrix(`l${l}.wk`, kv * hd, d);
    matrix(`l${l}.wv`, kv * hd, d);
    vector(`l${l}.q_norm`, hd);
    vector(`l${l}.k_norm`, hd);
    matrix(`l${l}.wo`, d, h * hd);
    vector(`l${l}.ffn_norm`, d);
    matrix(`l${l}.w_gate`, ff, d);
    matrix(`l${l}.w_up`, ff, d);
    matrix(`l${l}.w_down`, d, ff);
  }
  vector("final_norm", d);
  matrix("lm_head", vocab, d);
  const prompt = Array.from({ length: 6 }, () => Math.floor(random() * vocab));
  return { dims, tensors, prompt };
}

// YaRN, written from the published description rather than from the engine.
function rotary(dims) {
  const { hd, rope_base: base } = dims;
  const { factor, orig } = dims.yarn;
  const correction = (turns) => (hd * Math.log(orig / (turns * 2 * Math.PI))) / (2 * Math.log(base));
  const low = Math.max(0, Math.floor(correction(32)));
  const high = Math.min(hd - 1, Math.ceil(correction(1)));
  const theta = [];
  for (let i = 0; i < hd / 2; i++) {
    const original = base ** (-(2 * i) / hd);
    const y = (i - low) / Math.max(0.001, high - low);
    const extrapolate = 1 - Math.min(1, Math.max(0, y));
    theta.push((original / factor) * (1 - extrapolate) + original * extrapolate);
  }
  return { theta, mscale: 1 + 0.1 * Math.log(factor) };
}

export function referenceModel({ dims, tensors }) {
  const { d, n_heads: heads, n_kv_heads: kvHeads, hd, ff, vocab, n_layers: layers, eps } = dims;
  const weight = (name) => {
    const t = tensors.get(name);
    const w = new Float64Array(t.N * t.K);
    for (let g = 0; g < t.N * t.K; g++) w[g] = t.codes[g] * t.scales[Math.floor(g / 128)];
    return { w, N: t.N, K: t.K };
  };
  const gamma = (name) => Float64Array.from(tensors.get(name).values);
  const mul = ({ w, N, K }, x) => {
    const out = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += w[n * K + k] * x[k];
      out[n] = s;
    }
    return out;
  };
  const norm = (x, g) => {
    let s = 0;
    for (const v of x) s += v * v;
    const r = 1 / Math.sqrt(s / x.length + eps);
    return x.map((v, i) => v * r * g[i]);
  };
  const { theta, mscale } = rotary(dims);
  const rotate = (v, count, pos) => {
    const half = hd / 2;
    for (let h = 0; h < count; h++) {
      for (let i = 0; i < half; i++) {
        const a = v[h * hd + i];
        const b = v[h * hd + i + half];
        const c = Math.cos(pos * theta[i]) * mscale;
        const s = Math.sin(pos * theta[i]) * mscale;
        v[h * hd + i] = a * c - b * s;
        v[h * hd + i + half] = a * s + b * c;
      }
    }
  };
  const headNorm = (v, count, g) => {
    for (let h = 0; h < count; h++) v.set(norm(v.subarray(h * hd, (h + 1) * hd), g), h * hd);
  };

  const embed = weight("embed");
  const head = weight("lm_head");
  const L = [];
  for (let l = 0; l < layers; l++) {
    L.push({
      attnNorm: gamma(`l${l}.attn_norm`), wq: weight(`l${l}.wq`), wk: weight(`l${l}.wk`), wv: weight(`l${l}.wv`),
      qNorm: gamma(`l${l}.q_norm`), kNorm: gamma(`l${l}.k_norm`), wo: weight(`l${l}.wo`), ffnNorm: gamma(`l${l}.ffn_norm`),
      gate: weight(`l${l}.w_gate`), up: weight(`l${l}.w_up`), down: weight(`l${l}.w_down`), keys: [], values: [],
    });
  }
  const finalNorm = gamma("final_norm");
  let position = 0;

  function step(token) {
    let x = Float64Array.from(embed.w.subarray(token * d, (token + 1) * d));
    for (const layer of L) {
      const h = norm(x, layer.attnNorm);
      const q = mul(layer.wq, h);
      const k = mul(layer.wk, h);
      const v = mul(layer.wv, h);
      headNorm(q, heads, layer.qNorm);
      headNorm(k, kvHeads, layer.kNorm);
      rotate(q, heads, position);
      rotate(k, kvHeads, position);
      layer.keys.push(k);
      layer.values.push(v);
      const att = new Float64Array(heads * hd);
      for (let hq = 0; hq < heads; hq++) {
        const kh = Math.floor(hq / (heads / kvHeads));
        const scores = layer.keys.map((key) => {
          let s = 0;
          for (let i = 0; i < hd; i++) s += q[hq * hd + i] * key[kh * hd + i];
          return s / Math.sqrt(hd);
        });
        const top = Math.max(...scores);
        const e = scores.map((s) => Math.exp(s - top));
        const total = e.reduce((a, b) => a + b, 0);
        for (let i = 0; i < hd; i++) {
          let s = 0;
          for (let p = 0; p < e.length; p++) s += e[p] * layer.values[p][kh * hd + i];
          att[hq * hd + i] = s / total;
        }
      }
      const o = mul(layer.wo, att);
      x = x.map((value, i) => value + o[i]);
      const h2 = norm(x, layer.ffnNorm);
      const g = mul(layer.gate, h2);
      const u = mul(layer.up, h2);
      const a = g.map((value, i) => (value / (1 + Math.exp(-value))) * u[i]);
      const down = mul(layer.down, a);
      x = x.map((value, i) => value + down[i]);
    }
    const logits = mul(head, norm(x, finalNorm));
    position++;
    let best = 0;
    for (let i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return { token: best, logits };
  }

  return { step };
}
