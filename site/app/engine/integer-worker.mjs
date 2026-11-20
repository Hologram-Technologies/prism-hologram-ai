// The integer engines off the page's thread (integer.mjs says why). The page
// sends a copy of its module as it stands after the import and every weight's
// bytes; this worker opens the engine the page names on them, exactly as the
// page opened it before, and answers each call. A run's tokens go back one
// message per step, so the page shows each as it is made.
import { moduleOf } from "./loader.mjs";
import { createIntegerEngine } from "./integer.mjs";
import { createIntegerEncoder } from "./integer-encoder.mjs";
import { helperCount, useHelpers } from "./integer-helpers.mjs";

let engine = null;

self.onmessage = async ({ data }) => {
  const { id } = data;
  const say = (kind, value) => self.postMessage({ id, kind, value });
  try {
    if (data.kind === "open") {
      const module = moduleOf(data.compiled, data.memory);
      const held = new Map(data.tensors);
      const given = { module, table: data.table, read: (name) => held.get(name), config: data.config };
      engine = data.engine === "encoder" ? createIntegerEncoder(given) : createIntegerEngine({ ...given, rope: data.rope, capacity: data.capacity });
      // The decoder's products on helpers when this page can share memory
      // (GP-52); the same bits either way.
      const helpers = data.engine === "encoder" ? 0 : helperCount(data.helpers);
      if (helpers > 0) await useHelpers(module, helpers, engine.sizes);
      const value = data.engine === "encoder" ? { dims: engine.dims, labels: engine.labels } : { sizes: engine.sizes, helpers };
      self.postMessage({ id, kind: "done", ok: true, value });
      return;
    }
    const args = data.method === "run"
      ? [data.args[0], data.args[1], { stops: data.args[2], onFirst: () => say("first"), onToken: (out) => say("token", out) }]
      : data.args;
    const value = await engine[data.method](...args);
    self.postMessage({ id, kind: "done", ok: true, value, position: engine.position });
  } catch (error) {
    self.postMessage({ id, kind: "done", ok: false, error: String(error?.message ?? error) });
  }
};
