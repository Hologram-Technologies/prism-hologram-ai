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

// Pack an N by K matrix of codes in -8..7, one scale per 32 weights, into q4:
// four-bit codes offset by 8, weight g at bits 4 * (g % 2) of byte g / 2, then
// the scales.
export function packQ4(codes, scales, n, k) {
  const bytes = new Uint8Array((n * k) / 2 + ((n * k) / 32) * 4);
  for (let g = 0; g < n * k; g++) bytes[g >> 1] |= ((codes[g] + 8) & 15) << ((g & 1) * 4);
  new Float32Array(bytes.buffer, (n * k) / 2).set(scales);
  return bytes;
}

// IEEE half precision, as an f16 file holds numbers: the nearest f16 (ties to
// even) of a number below f16's largest, and the number an f16 holds.
export function f16bits(x) {
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  // Below 2^-14 an f16 is subnormal: a count of 2^-24 steps.
  if (Math.abs(x) < 2 ** -14) {
    const steps = Math.abs(x) / 2 ** -24;
    let m = Math.floor(steps);
    if (steps - m > 0.5 || (steps - m === 0.5 && m & 1)) m++;
    return sign | m;
  }
  const bits = new Uint32Array(new Float32Array([x]).buffer)[0];
  let exponent = ((bits >>> 23) & 255) - 127 + 15;
  let mantissa = (bits & 0x7fffff) >>> 13;
  const rest = bits & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && mantissa & 1)) mantissa++;
  if (mantissa === 0x400) (mantissa = 0), exponent++;
  if (exponent > 30) throw new Error(`${x} is above f16's range`);
  return ((bits >>> 16) & 0x8000) | (exponent << 10) | mantissa;
}
export const f16value = (h) => (h & 0x8000 ? -1 : 1) * (((h >> 10) & 31) === 0 ? (h & 1023) * 2 ** -24 : (1 + (h & 1023) / 1024) * 2 ** (((h >> 10) & 31) - 15));

// An N by K matrix of codes in -8..7 as ONNX's MatMulNBits holds it at 4 bits,
// block size 32 and no zero points, written from the operator's definition:
// B is [N][K / 32][16] bytes, element 2j of a block in the low nibble of byte
// j and element 2j + 1 in the high one, each stored plus the default zero
// point 8; scales are [N][K / 32], f32 or f16.
export function packMatMulNBits(codes, scales, n, k, scaleType) {
  const blocks = k / 32;
  const b = new Uint8Array(n * blocks * 16);
  for (let row = 0; row < n; row++)
    for (let block = 0; block < blocks; block++)
      for (let j = 0; j < 16; j++) {
        const at = row * k + block * 32 + 2 * j;
        b[(row * blocks + block) * 16 + j] = (codes[at] + 8) | ((codes[at + 1] + 8) << 4);
      }
  const s = scaleType === "f16" ? new Uint8Array(Uint16Array.from(scales, f16bits).buffer) : new Uint8Array(Float32Array.from(scales).buffer);
  return [b, s];
}

// A small model with Qwen2's shape: q4 weights, biases on q, k and v, no
// per-head norm, plain rotary, and an output head of its own. With layout
// "onnx" each weight is given as MatMulNBits' two tensors, with f32 scales,
// or with f16 ones when scaleType is "f16". With embedding "dense" the
// embedding is a plain matrix in the scales' type, f32 or f16, and the head is
// tied to it with no tensor of its own, as in onnx-community's exports.
// shape.vectors "f16" holds the norm weights and biases in f16, as a q4f16
// export does; shape.llama3 gives Llama's shape instead: no biases, and Llama
// 3's rotary scaling with the parameters given; shape.qkNorm gives Qwen3's:
// no biases, and a norm on each query and key head.
export function tinyQwen2(seed = 1, layout = "q4", scaleType = "f32", embedding = "q4", shape = {}) {
  const random = generator(seed);
  const dims = { d: 128, n_heads: 4, n_kv_heads: 2, hd: 32, ff: 256, vocab: 320, n_layers: 2, rope_base: 10000, eps: 1e-6 };
  if (shape.llama3) dims.llama3 = shape.llama3;
  const tensors = new Map();
  const matrix = (name, n, k) => {
    const codes = new Int8Array(n * k);
    for (let i = 0; i < n * k; i++) codes[i] = Math.floor(random() * 16) - 8;
    const scales = new Float32Array((n * k) / 32);
    for (let i = 0; i < scales.length; i++) scales[i] = Math.fround(0.006 + 0.012 * random());
    if (scaleType === "f16") for (let i = 0; i < scales.length; i++) scales[i] = f16value(f16bits(scales[i]));
    if (layout === "onnx") {
      const fmt = scaleType === "f16" ? "q4f16" : "q4";
      tensors.set(name, { fmt, N: n, K: k, parts: packMatMulNBits(codes, scales, n, k, scaleType), codes, scales });
    } else tensors.set(name, { fmt: "q4", N: n, K: k, bytes: packQ4(codes, scales, n, k), codes, scales });
  };
  const vector = (name, n, low, span) => {
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = Math.fround(low + span * random());
    if (shape.vectors === "f16") {
      const halves = Uint16Array.from(values, f16bits);
      values.set(Array.from(halves, f16value));
      tensors.set(name, { fmt: "f16", N: 1, K: n, bytes: new Uint8Array(halves.buffer), values });
    } else tensors.set(name, { fmt: "f32", N: 1, K: n, bytes: new Uint8Array(values.buffer), values });
  };
  const { d, n_heads: h, n_kv_heads: kv, hd, ff, vocab } = dims;
  if (embedding === "dense") {
    const values = new Float32Array(vocab * d);
    // Magnitudes 0.005 to 0.1, either sign: inside f16's normal range.
    for (let i = 0; i < values.length; i++) values[i] = Math.fround((random() < 0.5 ? -1 : 1) * (0.005 + 0.095 * random()));
    if (scaleType === "f16") {
      const halves = Uint16Array.from(values, f16bits);
      values.set(Array.from(halves, f16value));
      tensors.set("embed", { fmt: "f16", N: vocab, K: d, bytes: new Uint8Array(halves.buffer), values });
    } else tensors.set("embed", { fmt: "f32", N: vocab, K: d, bytes: new Uint8Array(values.buffer), values });
  } else matrix("embed", vocab, d);
  for (let l = 0; l < dims.n_layers; l++) {
    vector(`l${l}.attn_norm`, d, 0.8, 0.4);
    matrix(`l${l}.wq`, h * hd, d);
    matrix(`l${l}.wk`, kv * hd, d);
    matrix(`l${l}.wv`, kv * hd, d);
    if (shape.qkNorm) {
      vector(`l${l}.q_norm`, hd, 0.8, 0.4);
      vector(`l${l}.k_norm`, hd, 0.8, 0.4);
    } else if (!shape.llama3) {
      vector(`l${l}.bq`, h * hd, -0.3, 0.6);
      vector(`l${l}.bk`, kv * hd, -0.3, 0.6);
      vector(`l${l}.bv`, kv * hd, -0.3, 0.6);
    }
    matrix(`l${l}.wo`, d, h * hd);
    vector(`l${l}.ffn_norm`, d, 0.8, 0.4);
    matrix(`l${l}.w_gate`, ff, d);
    matrix(`l${l}.w_up`, ff, d);
    matrix(`l${l}.w_down`, d, ff);
  }
  vector("final_norm", d, 0.8, 0.4);
  if (embedding !== "dense") matrix("lm_head", vocab, d);
  const prompt = Array.from({ length: 6 }, () => Math.floor(random() * vocab));
  return { dims, tensors, prompt };
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
export function rotary(dims) {
  const { hd, rope_base: base } = dims;
  // Llama 3, from transformers' description of rope type "llama3": by each
  // pair's wavelength 2 pi / theta against the original context divided by the
  // low and high frequency factors.
  if (dims.llama3) {
    const { factor, low, high, orig } = dims.llama3;
    const theta = Array.from({ length: hd / 2 }, (_, i) => {
      const t = base ** (-(2 * i) / hd);
      const wavelength = (2 * Math.PI) / t;
      if (wavelength < orig / high) return t;
      if (wavelength > orig / low) return t / factor;
      const smooth = (orig / wavelength - low) / (high - low);
      return (1 - smooth) * (t / factor) + smooth * t;
    });
    return { theta, mscale: 1 };
  }
  if (!dims.yarn) return { theta: Array.from({ length: hd / 2 }, (_, i) => base ** (-(2 * i) / hd)), mscale: 1 };
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
    if (t.fmt === "f32" || t.fmt === "f16") return { w: Float64Array.from(t.values), N: t.N, K: t.K };
    const w = new Float64Array(t.N * t.K);
    const group = { q1: 128, q4: 32, q4f16: 32 }[t.fmt];
    for (let g = 0; g < t.N * t.K; g++) w[g] = t.codes[g] * t.scales[Math.floor(g / group)];
    return { w, N: t.N, K: t.K };
  };
  const gamma = (name) => Float64Array.from(tensors.get(name).values);
  const optional = (name) => (tensors.has(name) ? gamma(name) : null);
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
  const head = tensors.has("lm_head") ? weight("lm_head") : embed;
  const L = [];
  for (let l = 0; l < layers; l++) {
    L.push({
      attnNorm: gamma(`l${l}.attn_norm`), wq: weight(`l${l}.wq`), wk: weight(`l${l}.wk`), wv: weight(`l${l}.wv`),
      qNorm: optional(`l${l}.q_norm`), kNorm: optional(`l${l}.k_norm`), wo: weight(`l${l}.wo`), ffnNorm: gamma(`l${l}.ffn_norm`),
      bq: optional(`l${l}.bq`), bk: optional(`l${l}.bk`), bv: optional(`l${l}.bv`),
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
      for (const [x, b] of [[q, layer.bq], [k, layer.bk], [v, layer.bv]]) if (b) for (let i = 0; i < x.length; i++) x[i] += b[i];
      if (layer.qNorm) headNorm(q, heads, layer.qNorm);
      if (layer.kNorm) headNorm(k, kvHeads, layer.kNorm);
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
