// The WebGPU decode engine for a ternary Qwen3 model (Bonsai's layout). One
// step is one command buffer: every layer's kernels are recorded back to back
// and the only read back is the chosen token (and the logits when asked for).
// Decisions about what may be loaded and run are the kernel's; this module only
// computes.

import { ARGMAX, ATTENTION, EMBED_TERNARY, HEAD_NORM, KV_WRITE, MATVEC_TERNARY, RMS_NORM, ROPE, SILU_MUL } from "./wgsl.mjs";
import { plain, yarn } from "./yarn.mjs";

const U = typeof GPUBufferUsage === "undefined" ? {} : GPUBufferUsage;
const f32bits = (value) => new Uint32Array(new Float32Array([value]).buffer)[0];

// Bytes of one ternary block: two bit planes and one f32 scale per 128 weights.
export const ternaryLength = (n, k) => (n * k) / 4 + ((n * k) / 128) * 4;

export async function createEngine({ device, dims, tensors, capacity = 1024 }) {
  const { d, n_heads: heads, n_kv_heads: kvHeads, hd, ff, vocab, n_layers: layers } = dims;
  const eps = dims.eps ?? 1e-6;
  const qdim = heads * hd;
  const kvdim = kvHeads * hd;
  const rope = dims.yarn ? yarn({ hd, base: dims.rope_base, factor: dims.yarn.factor, orig: dims.yarn.orig }) : plain({ hd, base: dims.rope_base });

  let gpuBytes = 0;
  const storage = (size, extra = 0) => {
    gpuBytes += size;
    return device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC | extra });
  };
  const upload = (array) => {
    const buffer = storage(array.byteLength);
    device.queue.writeBuffer(buffer, 0, array.buffer, array.byteOffset, array.byteLength);
    return buffer;
  };
  const uniform = (values) => {
    const buffer = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
    return buffer;
  };

  const tensor = (name) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`model has no tensor ${name}`);
    return t;
  };
  const ternary = (name) => {
    const t = tensor(name);
    if (t.fmt !== "q1") throw new Error(`${name} is ${t.fmt}, not the ternary format`);
    const plane = (t.N * t.K) / 8;
    const want = ternaryLength(t.N, t.K);
    if (t.bytes.byteLength !== want) throw new Error(`${name} holds ${t.bytes.byteLength} bytes, its format needs ${want}`);
    return {
      N: t.N,
      K: t.K,
      signs: upload(t.bytes.subarray(0, plane)),
      nonzeros: upload(t.bytes.subarray(plane, 2 * plane)),
      scales: upload(t.bytes.subarray(2 * plane)),
    };
  };
  const vector = (name) => {
    const t = tensor(name);
    if (t.fmt !== "f32") throw new Error(`${name} is ${t.fmt}, not f32`);
    return upload(new Float32Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 4));
  };

  const embed = ternary("embed");
  const head = tensors.has("lm_head") && tensors.get("lm_head").bytes !== tensor("embed").bytes ? ternary("lm_head") : embed;
  const layer = [];
  for (let l = 0; l < layers; l++) {
    layer.push({
      attnNorm: vector(`l${l}.attn_norm`),
      wq: ternary(`l${l}.wq`),
      wk: ternary(`l${l}.wk`),
      wv: ternary(`l${l}.wv`),
      qNorm: vector(`l${l}.q_norm`),
      kNorm: vector(`l${l}.k_norm`),
      wo: ternary(`l${l}.wo`),
      ffnNorm: vector(`l${l}.ffn_norm`),
      gate: ternary(`l${l}.w_gate`),
      up: ternary(`l${l}.w_up`),
      down: ternary(`l${l}.w_down`),
      kc: storage(capacity * kvdim * 4),
      vc: storage(capacity * kvdim * 4),
    });
  }
  const finalNorm = vector("final_norm");
  const freqs = upload(rope.freqs);

  const B = {
    x: storage(d * 4),
    h: storage(d * 4),
    q: storage(qdim * 4),
    k: storage(kvdim * 4),
    v: storage(kvdim * 4),
    att: storage(qdim * 4),
    g: storage(ff * 4),
    u: storage(ff * 4),
    logits: storage(vocab * 4),
    chosen: storage(16),
  };
  const readChosen = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
  const readLogits = device.createBuffer({ size: vocab * 4, usage: U.MAP_READ | U.COPY_DST });

  const pipeline = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const P = {
    matvec: pipeline(MATVEC_TERNARY),
    norm: pipeline(RMS_NORM),
    headNorm: pipeline(HEAD_NORM),
    rope: pipeline(ROPE),
    kv: pipeline(KV_WRITE),
    attention: pipeline(ATTENTION(capacity)),
    silu: pipeline(SILU_MUL),
    embed: pipeline(EMBED_TERNARY),
    argmax: pipeline(ARGMAX),
  };
  const bind = (p, buffers) =>
    device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });

  // Uniforms whose values change with the position are written each step.
  const at = {
    embed: uniform([d, 0, 0, 0]),
    ropeQ: uniform([heads, hd, 0, f32bits(rope.mscale)]),
    ropeK: uniform([kvHeads, hd, 0, f32bits(rope.mscale)]),
    kv: uniform([kvdim, 0, 0, 0]),
    attention: uniform([heads, kvHeads, hd, 1]),
  };
  const fixed = {
    normD: uniform([d, f32bits(eps), 0, 0]),
    normHead: uniform([hd, f32bits(eps), 0, 0]),
    silu: uniform([ff, 0, 0, 0]),
    argmax: uniform([vocab, 0, 0, 0]),
  };
  const matvecUniform = new Map();
  const matvecParams = (K, N, accumulate) => {
    const key = `${K}:${N}:${accumulate}`;
    if (!matvecUniform.has(key)) matvecUniform.set(key, uniform([K, N, accumulate ? 1 : 0, 0]));
    return matvecUniform.get(key);
  };

  // Every dispatch of a step, recorded once: [pipeline, bind group, x, y].
  const plan = [];
  const grid = (n) => [Math.min(n, 65535), Math.ceil(n / 65535)];
  const matvec = (w, input, output, accumulate = false) =>
    plan.push([P.matvec, bind(P.matvec, [input, w.signs, w.nonzeros, w.scales, output, matvecParams(w.K, w.N, accumulate)]), ...grid(w.N)]);

  plan.push([P.embed, bind(P.embed, [embed.signs, embed.nonzeros, embed.scales, B.x, at.embed]), Math.ceil(d / 64), 1]);
  for (const L of layer) {
    plan.push([P.norm, bind(P.norm, [B.x, L.attnNorm, B.h, fixed.normD]), 1, 1]);
    matvec(L.wq, B.h, B.q);
    matvec(L.wk, B.h, B.k);
    matvec(L.wv, B.h, B.v);
    plan.push([P.headNorm, bind(P.headNorm, [B.q, L.qNorm, fixed.normHead]), heads, 1]);
    plan.push([P.headNorm, bind(P.headNorm, [B.k, L.kNorm, fixed.normHead]), kvHeads, 1]);
    plan.push([P.rope, bind(P.rope, [B.q, freqs, at.ropeQ]), Math.ceil((heads * hd) / 2 / 64), 1]);
    plan.push([P.rope, bind(P.rope, [B.k, freqs, at.ropeK]), Math.ceil((kvHeads * hd) / 2 / 64), 1]);
    plan.push([P.kv, bind(P.kv, [B.k, B.v, L.kc, L.vc, at.kv]), Math.ceil(kvdim / 64), 1]);
    plan.push([P.attention, bind(P.attention, [B.q, L.kc, L.vc, B.att, at.attention]), heads, 1]);
    matvec(L.wo, B.att, B.x, true);
    plan.push([P.norm, bind(P.norm, [B.x, L.ffnNorm, B.h, fixed.normD]), 1, 1]);
    matvec(L.gate, B.h, B.g);
    matvec(L.up, B.h, B.u);
    plan.push([P.silu, bind(P.silu, [B.g, B.u, fixed.silu]), Math.ceil(ff / 64), 1]);
    matvec(L.down, B.g, B.x, true);
  }
  plan.push([P.norm, bind(P.norm, [B.x, finalNorm, B.h, fixed.normD]), 1, 1]);
  matvec(head, B.h, B.logits);
  plan.push([P.argmax, bind(P.argmax, [B.logits, B.chosen, fixed.argmax]), 1, 1]);

  let position = 0;

  // Run one token at the next position; answer the greedy pick, and the logits
  // when asked (tests compare them; decoding does not need them).
  async function step(token, { logits = false } = {}) {
    if (position >= capacity) throw new Error(`context full: ${capacity} positions`);
    const pos = position;
    device.queue.writeBuffer(at.embed, 0, new Uint32Array([d, token, 0, 0]));
    device.queue.writeBuffer(at.ropeQ, 0, new Uint32Array([heads, hd, pos, f32bits(rope.mscale)]));
    device.queue.writeBuffer(at.ropeK, 0, new Uint32Array([kvHeads, hd, pos, f32bits(rope.mscale)]));
    device.queue.writeBuffer(at.kv, 0, new Uint32Array([kvdim, pos, 0, 0]));
    device.queue.writeBuffer(at.attention, 0, new Uint32Array([heads, kvHeads, hd, pos + 1]));
    const encoder = device.createCommandEncoder();
    for (const [p, group, x, y] of plan) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
      pass.end();
    }
    encoder.copyBufferToBuffer(B.chosen, 0, readChosen, 0, 16);
    if (logits) encoder.copyBufferToBuffer(B.logits, 0, readLogits, 0, vocab * 4);
    device.queue.submit([encoder.finish()]);
    position = pos + 1;
    await readChosen.mapAsync(GPUMapMode.READ);
    const chosen = new Uint32Array(readChosen.getMappedRange().slice(0))[0];
    readChosen.unmap();
    let values = null;
    if (logits) {
      await readLogits.mapAsync(GPUMapMode.READ);
      values = new Float32Array(readLogits.getMappedRange().slice(0));
      readLogits.unmap();
    }
    return { token: chosen, logits: values };
  }

  // Greedy continuation of a prompt; prompt tokens run one at a time for now.
  async function generate(prompt, count) {
    let last = null;
    for (const token of prompt) last = await step(token);
    const out = [];
    for (let i = 0; i < count && last; i++) {
      out.push(last.token);
      last = await step(last.token);
    }
    return out;
  }

  return {
    step,
    generate,
    reset: () => { position = 0; },
    get position() { return position; },
    get gpuBytes() { return gpuBytes; },
    rope,
  };
}
