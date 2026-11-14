// Rotary frequencies with YaRN scaling, as transformers and llama.cpp compute
// them. A model trained at `orig` positions and stretched by `factor` keeps the
// frequencies of its fast-turning pairs, divides the slow ones by `factor`, and
// blends the pairs between; the rotated vectors are scaled by `mscale` so that
// attention's softmax keeps its sharpness at the longer context.

const BETA_FAST = 32;
const BETA_SLOW = 1;

// The pair index at which a rotation makes `turns` full turns over `orig`
// positions, read back from the frequency formula.
function pairFor(turns, hd, base, orig) {
  return (hd * Math.log(orig / (turns * 2 * Math.PI))) / (2 * Math.log(base));
}

export function yarn({ hd, base, factor, orig }) {
  const half = hd / 2;
  const low = Math.max(0, Math.floor(pairFor(BETA_FAST, hd, base, orig)));
  const high = Math.min(hd - 1, Math.ceil(pairFor(BETA_SLOW, hd, base, orig)));
  const freqs = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const extrapolated = Math.pow(base, (-2 * i) / hd);
    const interpolated = extrapolated / factor;
    const ramp = (i - low) / Math.max(0.001, high - low);
    const keep = 1 - Math.min(1, Math.max(0, ramp));
    freqs[i] = interpolated * (1 - keep) + extrapolated * keep;
  }
  const mscale = factor > 1 ? 1 + 0.1 * Math.log(factor) : 1;
  return { freqs, low, high, mscale };
}

// Plain rotary frequencies, for a model with no YaRN in its manifest.
export function plain({ hd, base }) {
  const freqs = new Float32Array(hd / 2);
  for (let i = 0; i < hd / 2; i++) freqs[i] = Math.pow(base, (-2 * i) / hd);
  return { freqs, low: hd / 2, high: hd / 2, mscale: 1 };
}
