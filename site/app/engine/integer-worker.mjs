// The integer engines off the page's thread (integer.mjs says why). The page
// sends a copy of its module as it stands after the import and every weight's
// bytes; this worker opens the engine the page names on them, exactly as the
// page opened it before, and answers each call. A run's tokens go back one
// message per step, so the page shows each as it is made. A streamed engine
// (GP-70) is opened on the weights that have arrived and handed the rest as
// they come; its products move to WebGPU once its first token is out.
import { moduleOf } from "./loader.mjs";
import { createIntegerEngine, createStreamingEngine } from "./integer.mjs";
import { createIntegerEncoder } from "./integer-encoder.mjs";
import { helperCount, useHelpers } from "./integer-helpers.mjs";
import { useGpu } from "./integer-gpu.mjs";

let engine = null;
// WebGPU, once it takes the streamed engine's products (GP-71).
let streamedGpu = null;
// The module the engine runs on, kept for WebGPU to take its products later.
let current = null;

self.onmessage = async ({ data }) => {
  const { id } = data;
  const say = (kind, value) => self.postMessage({ id, kind, value });
  try {
    if (data.kind === "open") {
      const module = moduleOf(data.compiled, data.memory);
      current = module;
      if (data.engine === "stream") {
        const onHeld = async (before, now) => {
          if (streamedGpu) await streamedGpu.more(7 * before, 7 * (now - before), now);
        };
        engine = createStreamingEngine({ module, config: data.config, rope: data.rope, capacity: data.capacity, held: data.held, tensors: data.tensors, onHeld });
        // The layers held at open and the head on WebGPU now, when the page
        // chose it; the products of each later layer before it runs.
        streamedGpu = data.gpu ? await useGpu(module, data.gpu, engine.sizes) : null;
        const gpu = streamedGpu && { products: streamedGpu.gpu, cpu: streamedGpu.cpu, adapter: streamedGpu.adapter };
        self.postMessage({ id, kind: "done", ok: true, value: { sizes: engine.sizes, gpu } });
        return;
      }
      const held = new Map(data.tensors);
      const given = { module, table: data.table, read: (name) => held.get(name), config: data.config };
      engine = data.engine === "encoder" ? createIntegerEncoder(given) : createIntegerEngine({ ...given, rope: data.rope, capacity: data.capacity });
      // The decoder's products on WebGPU when the page chose that device
      // (GP-56), else on helpers when this page can share memory (GP-52);
      // the same bits every way.
      const gpu = data.engine === "encoder" || !data.gpu ? null : await useGpu(module, data.gpu, engine.sizes);
      const helpers = data.engine === "encoder" || gpu ? 0 : helperCount(data.helpers);
      if (helpers > 0) await useHelpers(module, helpers, engine.sizes);
      const value = data.engine === "encoder" ? { dims: engine.dims, labels: engine.labels } : { sizes: engine.sizes, helpers, gpu: gpu && { products: gpu.gpu, cpu: gpu.cpu, adapter: gpu.adapter } };
      self.postMessage({ id, kind: "done", ok: true, value });
      return;
    }
    const hooks = { onFirst: () => say("first"), onToken: (out) => say("token", out) };
    let value;
    if (data.method === "run") {
      value = await engine.run(data.args[0], data.args[1], { stops: data.args[2], ...hooks });
    } else if (data.method === "finish") {
      // The first token is computed on the module, whose weights are all in
      // by now; WebGPU, when the page chose it, takes every product after.
      let gpu = null;
      const attach = async () => {
        if (data.args[2] && !streamedGpu) gpu = await useGpu(current, data.args[2], engine.sizes);
      };
      const tokens = await engine.finish(data.args[0], { stops: data.args[1], ...hooks, attach });
      value = { tokens, gpu: gpu && { products: gpu.gpu, cpu: gpu.cpu, adapter: gpu.adapter } };
    } else if (data.method === "attach") {
      const gpu = streamedGpu ?? await useGpu(current, data.args[0], engine.sizes);
      value = { products: gpu.gpu, cpu: gpu.cpu, adapter: gpu.adapter };
    } else {
      value = await engine[data.method](...data.args);
    }
    self.postMessage({ id, kind: "done", ok: true, value, position: engine.position });
  } catch (error) {
    self.postMessage({ id, kind: "done", ok: false, error: String(error?.message ?? error) });
  }
};
