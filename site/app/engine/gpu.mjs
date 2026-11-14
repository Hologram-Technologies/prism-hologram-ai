// The WebGPU decode engine for a ternary Qwen3 model (Bonsai's layout). One
// step is one command buffer: every layer's kernels are recorded back to back
// and the only read back is the chosen token (and the logits when asked for).
// Decisions about what may be loaded and run are the kernel's; this module only
// computes.

import { APPEND, ARGMAX, ATTENTION, EMBED_TERNARY, HEAD_NORM, KV_WRITE, MATVEC_TERNARY, RMS_NORM, ROPE, SILU_MUL } from "./wgsl.mjs";
import { plain, yarn } from "./yarn.mjs";
import { ATTENTION_ROWS, EMBED_ROWS, KV_WRITE_ROWS, MATMUL_TERNARY, RMS_NORM_ROWS, ROPE_ROWS } from "./wgsl-batch.mjs";

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

  // The sequence lives on the GPU: position p's token is ring[p]. A position
  // reads its token from the ring and, when it generates, writes its pick to
  // the next slot, so many positions run in one submission.
  const ring = storage(capacity * 4);
  const pipeAppend = pipeline(APPEND);

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

  // A slot is one position's uniforms and the bind groups that read them.
  // Passes that do not depend on the position share one bind group.
  const SLOTS = 32;
  const grid = (n) => [Math.min(n, 65535), Math.ceil(n / 65535)];
  const shared = new Map();
  const once = (key, make) => {
    if (!shared.has(key)) shared.set(key, make());
    return shared.get(key);
  };
  const slots = [];
  for (let s = 0; s < SLOTS; s++) {
    const u = {
      embed: uniform([d, 0, 0, 0]),
      ropeQ: uniform([heads, hd, 0, f32bits(rope.mscale)]),
      ropeK: uniform([kvHeads, hd, 0, f32bits(rope.mscale)]),
      kv: uniform([kvdim, 0, 0, 0]),
      attention: uniform([heads, kvHeads, hd, 1]),
      append: uniform([0, 0, 0, 0]),
    };
    const plan = [];
    const matvec = (key, w, input, output, accumulate = false) =>
      plan.push([P.matvec, once(key, () => bind(P.matvec, [input, w.signs, w.nonzeros, w.scales, output, matvecParams(w.K, w.N, accumulate)])), ...grid(w.N)]);
    plan.push([P.embed, bind(P.embed, [embed.signs, embed.nonzeros, embed.scales, B.x, ring, u.embed]), Math.ceil(d / 64), 1]);
    layer.forEach((L, l) => {
      plan.push([P.norm, once(`an${l}`, () => bind(P.norm, [B.x, L.attnNorm, B.h, fixed.normD])), 1, 1]);
      matvec(`q${l}`, L.wq, B.h, B.q);
      matvec(`k${l}`, L.wk, B.h, B.k);
      matvec(`v${l}`, L.wv, B.h, B.v);
      plan.push([P.headNorm, once(`qn${l}`, () => bind(P.headNorm, [B.q, L.qNorm, fixed.normHead])), heads, 1]);
      plan.push([P.headNorm, once(`kn${l}`, () => bind(P.headNorm, [B.k, L.kNorm, fixed.normHead])), kvHeads, 1]);
      plan.push([P.rope, bind(P.rope, [B.q, freqs, u.ropeQ]), Math.ceil((heads * hd) / 2 / 64), 1]);
      plan.push([P.rope, bind(P.rope, [B.k, freqs, u.ropeK]), Math.ceil((kvHeads * hd) / 2 / 64), 1]);
      plan.push([P.kv, bind(P.kv, [B.k, B.v, L.kc, L.vc, u.kv]), Math.ceil(kvdim / 64), 1]);
      plan.push([P.attention, bind(P.attention, [B.q, L.kc, L.vc, B.att, u.attention]), heads, 1]);
      matvec(`o${l}`, L.wo, B.att, B.x, true);
      plan.push([P.norm, once(`fn${l}`, () => bind(P.norm, [B.x, L.ffnNorm, B.h, fixed.normD])), 1, 1]);
      matvec(`g${l}`, L.gate, B.h, B.g);
      matvec(`u${l}`, L.up, B.h, B.u);
      plan.push([P.silu, once(`s${l}`, () => bind(P.silu, [B.g, B.u, fixed.silu])), Math.ceil(ff / 64), 1]);
      matvec(`d${l}`, L.down, B.g, B.x, true);
    });
    plan.push([P.norm, once("final", () => bind(P.norm, [B.x, finalNorm, B.h, fixed.normD])), 1, 1]);
    matvec("head", head, B.h, B.logits);
    plan.push([P.argmax, once("argmax", () => bind(P.argmax, [B.logits, B.chosen, fixed.argmax])), 1, 1]);
    slots.push({ u, plan, append: bind(pipeAppend, [ring, B.chosen, u.append]) });
  }

  const record = (encoder, s, pos, generates) => {
    const { u, plan, append } = slots[s];
    device.queue.writeBuffer(u.embed, 0, new Uint32Array([d, pos, 0, 0]));
    device.queue.writeBuffer(u.ropeQ, 0, new Uint32Array([heads, hd, pos, f32bits(rope.mscale)]));
    device.queue.writeBuffer(u.ropeK, 0, new Uint32Array([kvHeads, hd, pos, f32bits(rope.mscale)]));
    device.queue.writeBuffer(u.kv, 0, new Uint32Array([kvdim, pos, 0, 0]));
    device.queue.writeBuffer(u.attention, 0, new Uint32Array([heads, kvHeads, hd, pos + 1]));
    device.queue.writeBuffer(u.append, 0, new Uint32Array([pos + 1, 0, 0, 0]));
    // One compute pass per position: dispatches within a pass run in order and
    // see each other's writes, without a pass's start-up cost each time.
    const pass = encoder.beginComputePass();
    for (const [p, group, x, y] of generates ? [...plan, [pipeAppend, append, 1, 1]] : plan) {
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
    }
    pass.end();
  };

  // The prompt, BLOCK positions at a time: each weight is read once per block.
  // The kernels repeat the one-position arithmetic row by row, so the cache and
  // every later pick are the same as running the prompt position by position.
  const BLOCK = 16;
  const PB = {
    x: storage(BLOCK * d * 4), h: storage(BLOCK * d * 4),
    q: storage(BLOCK * qdim * 4), k: storage(BLOCK * kvdim * 4), v: storage(BLOCK * kvdim * 4),
    att: storage(BLOCK * qdim * 4), g: storage(BLOCK * ff * 4), u: storage(BLOCK * ff * 4),
  };
  const PP = {
    matmul: pipeline(MATMUL_TERNARY(BLOCK)), norm: pipeline(RMS_NORM_ROWS), embed: pipeline(EMBED_ROWS),
    rope: pipeline(ROPE_ROWS), kv: pipeline(KV_WRITE_ROWS), attention: pipeline(ATTENTION_ROWS(capacity)),
  };
  const pu = {
    embed: uniform([d, 0, 0, 0]),
    ropeQ: uniform([heads, hd, 0, f32bits(rope.mscale)]),
    ropeK: uniform([kvHeads, hd, 0, f32bits(rope.mscale)]),
    kv: uniform([kvdim, 0, 0, 0]),
    attention: uniform([heads, kvHeads, hd, 1]),
    silu: uniform([0, 0, 0, 0]),
  };
  const mmUniforms = new Map();
  const mmParams = (K, N, accumulate) => {
    const key = `${K}:${N}:${accumulate}`;
    if (!mmUniforms.has(key)) mmUniforms.set(key, { buffer: uniform([K, N, accumulate ? 1 : 0, 0]), values: [K, N, accumulate ? 1 : 0] });
    return mmUniforms.get(key).buffer;
  };
  // [pipeline, bind group, x(count), y(count)]
  const prefillPlan = [];
  const mm = (w, input, output, accumulate = false) =>
    prefillPlan.push([PP.matmul, bind(PP.matmul, [input, w.signs, w.nonzeros, w.scales, output, mmParams(w.K, w.N, accumulate)]), () => Math.min(w.N, 65535), () => Math.ceil(w.N / 65535)]);
  prefillPlan.push([PP.embed, bind(PP.embed, [embed.signs, embed.nonzeros, embed.scales, PB.x, ring, pu.embed]), (c) => Math.ceil((c * d) / 64), () => 1]);
  for (const L of layer) {
    prefillPlan.push([PP.norm, bind(PP.norm, [PB.x, L.attnNorm, PB.h, fixed.normD]), (c) => c, () => 1]);
    mm(L.wq, PB.h, PB.q);
    mm(L.wk, PB.h, PB.k);
    mm(L.wv, PB.h, PB.v);
    prefillPlan.push([P.headNorm, bind(P.headNorm, [PB.q, L.qNorm, fixed.normHead]), (c) => c * heads, () => 1]);
    prefillPlan.push([P.headNorm, bind(P.headNorm, [PB.k, L.kNorm, fixed.normHead]), (c) => c * kvHeads, () => 1]);
    prefillPlan.push([PP.rope, bind(PP.rope, [PB.q, freqs, pu.ropeQ]), () => Math.ceil((heads * hd) / 2 / 64), (c) => c]);
    prefillPlan.push([PP.rope, bind(PP.rope, [PB.k, freqs, pu.ropeK]), () => Math.ceil((kvHeads * hd) / 2 / 64), (c) => c]);
    prefillPlan.push([PP.kv, bind(PP.kv, [PB.k, PB.v, L.kc, L.vc, pu.kv]), (c) => Math.ceil((c * kvdim) / 64), () => 1]);
    prefillPlan.push([PP.attention, bind(PP.attention, [PB.q, L.kc, L.vc, PB.att, pu.attention]), () => heads, (c) => c]);
    mm(L.wo, PB.att, PB.x, true);
    prefillPlan.push([PP.norm, bind(PP.norm, [PB.x, L.ffnNorm, PB.h, fixed.normD]), (c) => c, () => 1]);
    mm(L.gate, PB.h, PB.g);
    mm(L.up, PB.h, PB.u);
    prefillPlan.push([P.silu, bind(P.silu, [PB.g, PB.u, pu.silu]), (c) => Math.ceil((c * ff) / 64), () => 1]);
    mm(L.down, PB.g, PB.x, true);
  }

  // Run prompt positions first .. first + count - 1 (count <= BLOCK) as one block.
  const prefill = (first, count) => {
    device.queue.writeBuffer(pu.embed, 0, new Uint32Array([d, first, count, 0]));
    device.queue.writeBuffer(pu.ropeQ, 0, new Uint32Array([heads, hd, first, f32bits(rope.mscale)]));
    device.queue.writeBuffer(pu.ropeK, 0, new Uint32Array([kvHeads, hd, first, f32bits(rope.mscale)]));
    device.queue.writeBuffer(pu.kv, 0, new Uint32Array([kvdim, first, count, 0]));
    device.queue.writeBuffer(pu.attention, 0, new Uint32Array([heads, kvHeads, hd, first + 1]));
    device.queue.writeBuffer(pu.silu, 0, new Uint32Array([count * ff, 0, 0, 0]));
    for (const { buffer, values } of mmUniforms.values()) device.queue.writeBuffer(buffer, 0, new Uint32Array([...values, count]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const [p, group, x, y] of prefillPlan) {
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x(count), y(count));
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  const readRing = device.createBuffer({ size: SLOTS * 4 + 4, usage: U.MAP_READ | U.COPY_DST });
  let position = 0;

  // Run one token at the next position; answer the greedy pick, and the logits
  // when asked (tests compare them; decoding does not need them).
  async function step(token, { logits = false } = {}) {
    if (position >= capacity) throw new Error(`context full: ${capacity} positions`);
    const pos = position;
    device.queue.writeBuffer(ring, pos * 4, new Uint32Array([token]));
    const encoder = device.createCommandEncoder();
    record(encoder, 0, pos, false);
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

  // Greedy continuation of a prompt, many positions per submission: the
  // prompt's tokens are written to the ring and run in chunks; generation picks
  // on the GPU and appends to the ring, and only the new tokens are read back.
  // Stops after `count` new tokens or at a stop token (which is not returned).
  // `onToken(ids)` sees the tokens so far after each submission.
  async function run(prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) {
    if (position + prompt.length + count > capacity) throw new Error(`context full: ${capacity} positions`);
    device.queue.writeBuffer(ring, position * 4, new Uint32Array(prompt));
    let p = 0;
    // All but the last prompt token, in blocks: nothing to read back. The last
    // goes through the one-position path below, which makes the first pick.
    while (p < prompt.length - 1) {
      const n = Math.min(BLOCK, prompt.length - 1 - p);
      prefill(position, n);
      position += n;
      p += n;
    }
    // The last prompt token and every generated one pick and append.
    const out = [];
    let first = true;
    while (out.length < count) {
      // The first pick alone, so the time to the first token is that and no more.
      const n = Math.min(first ? 1 : SLOTS, count - out.length);
      const encoder = device.createCommandEncoder();
      for (let s = 0; s < n; s++) record(encoder, s, position + s, true);
      encoder.copyBufferToBuffer(ring, (position + 1) * 4, readRing, 0, n * 4);
      device.queue.submit([encoder.finish()]);
      await readRing.mapAsync(GPUMapMode.READ, 0, n * 4);
      const picked = Array.from(new Uint32Array(readRing.getMappedRange(0, n * 4).slice(0)));
      readRing.unmap();
      if (first) { onFirst(); first = false; }
      let stopAt = picked.findIndex((t) => stops.has(t));
      const take = stopAt < 0 ? picked : picked.slice(0, stopAt);
      out.push(...take);
      // Positions past a stop were computed but are not part of the sequence.
      position += stopAt < 0 ? n : stopAt + 1;
      onToken(out.slice());
      if (stopAt >= 0) break;
    }
    return out;
  }

  // Greedy continuation, one position per submission: the reference run is
  // compared with.
  async function generate(prompt, count) {
    let last = null;
    for (const token of prompt) last = await step(token);
    const out = [];
    for (let i = 0; i < count && last; i++) {
      out.push(last.token);
      if (i + 1 < count) last = await step(last.token);
    }
    return out;
  }

  return {
    step,
    run,
    generate,
    reset: () => { position = 0; },
    get position() { return position; },
    get gpuBytes() { return gpuBytes; },
    rope,
  };
}
