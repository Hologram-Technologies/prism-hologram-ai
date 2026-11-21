// The integer engine on the page: the module's forward pass (EN-52), answering
// with the interface the float engines answer: step, run, generate, reset,
// position, saveState, restoreState.
//
// Why: Hologram AI's inference uses integers only (Q_ENGINE_PLAN.md), so the
// page computes no number of its own. Every step is `hologram-ai-cpu-decode`
// in the module: the residual stream as binary32 bit patterns touched only by
// integer code, each stage rounded once, half to even. The page hands the
// module each weight's bytes once, opens the model with what its config says,
// and reads back the greedy token and the logits' bits.

export const TYPES = { f32: 1, u8: 2, i8: 3, i32: 6, i64: 7, f16: 10, bf16: 16 };
const le = (n) => new Uint8Array(new Uint32Array([n]).buffer);
const word = (bytes, at = 0) => new DataView(bytes.buffer, bytes.byteOffset, bytes.length).getUint32(at, true);

// The page's integer engines run in a worker (integer-worker.mjs): a forward
// pass takes seconds a token on a phone, and on the page's thread it would
// hold every paint and tap until the reply ended. The worker starts from a
// copy of the page's module as it stands after the import, so it computes on
// the same graph with the same code; the page keeps its own module for the
// tokeniser and the hashes. `open` names the engine and what it is opened
// with; `transfer` the buffers handed over rather than copied.
export async function openWorker(m, open, transfer = []) {
  const worker = new Worker(new URL("./integer-worker.mjs", import.meta.url), { type: "module" });
  const waiting = new Map();
  let next = 0;
  worker.onmessage = ({ data }) => {
    const asked = waiting.get(data.id);
    if (!asked) return;
    if (data.kind === "first") return asked.onFirst();
    if (data.kind === "token") return asked.onToken(data.value);
    waiting.delete(data.id);
    if (data.ok) asked.resolve(data);
    else asked.reject(new Error(data.error));
  };
  worker.onerror = (event) => {
    for (const asked of waiting.values()) asked.reject(new Error(`the integer worker stopped: ${event.message ?? "it did not load"}`));
    waiting.clear();
  };
  const ask = (message, given = [], hooks = {}) => new Promise((resolve, reject) => {
    const id = next++;
    waiting.set(id, { resolve, reject, onFirst: () => {}, onToken: () => {}, ...hooks });
    worker.postMessage({ id, ...message }, given);
  });
  const opened = await ask({ kind: "open", compiled: m.compiled, memory: m.snapshot(), ...open }, transfer);
  return { ask, opened: opened.value, close: () => worker.terminate() };
}

// Every weight the import placed, by name, read once for the worker.
export const placed = (table, read) => Object.keys(table.places).map((name) => [name, read(name)]);

// The forward pass in the worker, with the interface createIntegerEngine
// answers; each call is awaited, and `run` hands `onToken` each token as the
// worker makes it.
// `gpu`, the kernels decision 48 printed, puts its products on WebGPU (GP-56).
export async function startIntegerEngine({ module: m, table, read, config, rope = {}, capacity, helpers, gpu = null }) {
  const tensors = placed(table, read);
  const { ask, opened } = await openWorker(m, { engine: "decoder", table, tensors, config, rope, capacity, helpers, gpu }, tensors.map(([, bytes]) => bytes.buffer));
  let position = 0;
  const call = async (method, args, hooks) => {
    const said = await ask({ kind: "call", method, args }, [], hooks);
    position = said.position;
    return said.value;
  };
  return {
    step: (token, options) => call("step", [token, options]),
    run: (prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) => call("run", [prompt, count, stops], { onToken, onFirst }),
    generate: (prompt, count) => call("generate", [prompt, count]),
    probe: (ids, at, layer, component) => call("probe", [ids, at, layer, component]),
    saveState: (length) => call("saveState", [length]),
    restoreState: (state) => call("restoreState", [state]),
    reset: () => call("reset", []),
    get position() { return position; },
    device: opened.gpu ? "webgpu" : "cpu",
    integer: true,
    sizes: opened.sizes,
    // What WebGPU holds (GP-56): how many products its kernels answer, how
    // many the module answers beside them, and the adapter.
    gpu: opened.gpu,
    // How many helpers answer its products (GP-52); 0 computes them all in
    // the worker.
    helpers: opened.helpers,
  };
}

// The forward pass in the worker while the model's file is still arriving
// (GP-70): opened on the weights held so far, `held` layers of them; `add`
// hands it more as they arrive, `prefill` starts a prompt that runs each
// layer as soon as it is held, and `finish` answers the prompt's first token
// once every weight is in, then the rest of the turn, its products on WebGPU
// from then on when `gpu` names the kernels (GP-56). Otherwise it answers as
// startIntegerEngine's engine does.
export async function startStreamingEngine({ module: m, config, rope = {}, capacity, held, tensors }) {
  const { ask, opened } = await openWorker(m, { engine: "stream", config, rope, capacity, held, tensors }, tensors.map(([, , bytes]) => bytes.buffer));
  let position = 0;
  let gpu = null;
  const call = async (method, args, hooks, given = []) => {
    const said = await ask({ kind: "call", method, args }, given, hooks);
    position = said.position;
    return said.value;
  };
  return {
    add: (more, layers) => call("add", [more, layers], {}, more.map(([, , bytes]) => bytes.buffer)),
    prefill: (prompt) => call("prefill", [prompt]),
    complete: () => call("complete", []),
    probe: (ids, at, layer, component) => call("probe", [ids, at, layer, component]),
    // Every product on WebGPU from now on (GP-56), for a load no early turn
    // is waiting on.
    attach: async (kernels) => { gpu = await call("attach", [kernels]); return gpu; },
    finish: async (count, { stops = new Set(), onToken = () => {}, onFirst = () => {}, gpu: kernels = null } = {}) => {
      const said = await call("finish", [count, stops, kernels], { onToken, onFirst });
      gpu = said.gpu;
      return said.tokens;
    },
    step: (token, options) => call("step", [token, options]),
    run: (prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) => call("run", [prompt, count, stops], { onToken, onFirst }),
    saveState: (length) => call("saveState", [length]),
    restoreState: (state) => call("restoreState", [state]),
    reset: () => call("reset", []),
    get position() { return position; },
    get device() { return gpu ? "webgpu" : "cpu"; },
    get gpu() { return gpu; },
    integer: true,
    streamed: true,
    sizes: opened.sizes,
    helpers: 0,
  };
}

// `module` is the page's module after the graph was imported through it;
// `table` and `read` are the import's tensors and their bytes; `config` the
// text of config.json, whose numbers the module reads exactly (KN-50, EN-58);
// `rope` the rotary parameters the loader confirmed.
export function createIntegerEngine({ module: m, table, read, config, rope = {}, capacity }) {
  for (const [name, [element]] of Object.entries(table.tensors)) {
    if (!(name in table.places) || !(element in TYPES)) continue;
    handTensor(m, name, TYPES[element], read(name));
  }
  return engineOver(m, openDecode(m, config, rope, null), capacity);
}

// One weight for the module's forward pass: its name, its element type as
// onnx.proto numbers it, its bytes.
function handTensor(m, name, type, bytes) {
  const named = new TextEncoder().encode(name);
  const given = new Uint8Array(8 + named.length + bytes.length);
  given.set(le(named.length), 0);
  given.set(named, 4);
  given.set(le(type), 4 + named.length);
  given.set(bytes, 8 + named.length);
  m.callWith("decode_tensor", given);
}

// The module's forward pass opened with the config's text and the rotary
// parameters, as far as `held` layers when a streamed load says how many
// (EN-112); its sizes.
function openDecode(m, config, rope, held) {
  const opened = {
    config,
    scaling: rope.llama3 ? { factor: rope.llama3.factor, low: rope.llama3.low, high: rope.llama3.high, original: rope.llama3.orig } : null,
    ...(held === null ? {} : { held }),
  };
  return JSON.parse(new TextDecoder().decode(m.callWith("decode_open", new TextEncoder().encode(JSON.stringify(opened)))));
}

// The engine in the worker while the file arrives (GP-70): `tensors` are
// [name, type, bytes] held so far and `held` the layers whole among them.
// A prompt given to `prefill` goes through each layer as soon as it is held
// (EN-110, EN-112), the layers it can before the rest arrive.
export function createStreamingEngine({ module: m, config, rope = {}, capacity, held, tensors }) {
  for (const [name, type, bytes] of tensors) handTensor(m, name, type, bytes);
  let layers = held;
  const engine = engineOver(m, openDecode(m, config, rope, held), capacity);
  let prompt = null;
  let ran = 0;
  const runHeld = () => {
    while (prompt && ran < layers) ran = word(m.call("decode_prefill_layer", layers));
  };
  return Object.assign(engine, {
    add(more, now) {
      for (const [name, type, bytes] of more) handTensor(m, name, type, bytes);
      layers = now;
      runHeld();
      return { ran, layers };
    },
    // Every weight is in and no prompt waits: the model is read whole from
    // now on, as an engine opened after the load reads it.
    complete() {
      if (!prompt) openDecode(m, config, rope, null);
      return { layers };
    },
    prefill(ids) {
      if (engine.position + ids.length > capacity) throw new Error(`context full: ${capacity} positions`);
      m.callWith("decode_prefill", new Uint8Array(new Uint32Array(ids).buffer));
      prompt = ids;
      ran = 0;
      runHeld();
      return { ran, layers };
    },
    // Every weight in: the prompt's first token, then up to `count` in all,
    // stopping at any id in `stops`; `attach` runs once the first token is
    // out, before the next is computed.
    async finish(count, { stops = new Set(), onToken = () => {}, onFirst = () => {}, attach = async () => {} } = {}) {
      if (!prompt) throw new Error("no prompt was prefilled");
      runHeld();
      const said = m.call("decode_prefill_finish");
      engine.moved(prompt.length);
      prompt = null;
      let last = word(said);
      onFirst();
      await attach();
      const out = [];
      while (out.length < count) {
        if (stops.has(last)) break;
        out.push(last);
        onToken(out.slice());
        if (out.length < count) last = (await engine.step(last)).token;
      }
      return out;
    },
  });
}

// What every integer engine answers, over the module's open forward pass.
function engineOver(m, sizes, capacity) {
  let position = 0;

  // One token through the model: the greedy token, and the logits if asked,
  // read from their binary32 bits.
  function stepSync(token, wantLogits = false) {
    const said = m.call("decode_step", token);
    position += 1;
    const logits = wantLogits ? new Float32Array(said.slice(4).buffer) : null;
    return { token: word(said), logits };
  }
  async function step(token, { logits: wantLogits = false } = {}) {
    return stepSync(token, wantLogits);
  }
  // The prompt, then `count` greedy picks, stopping at any id in `stops`.
  async function run(prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) {
    if (position + prompt.length + count > capacity) throw new Error(`context full: ${capacity} positions`);
    // Every prompt id but the last goes through without the output head
    // (GP-51): nothing reads those logits, and the head is a quarter of a
    // token. The cache, and so every token after, is the same bits.
    let last = null;
    prompt.forEach((t, i) => {
      if (i < prompt.length - 1) {
        m.call("decode_advance", t);
        position += 1;
      } else last = stepSync(t).token;
    });
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
  // EN-97: component `component` of layer `layer` at position `at` of the
  // run `ids`, replayed from the start: its binary32 bit patterns, four bytes
  // each. The engine is left with no run, as reset leaves it.
  async function probe(ids, at, layer, component) {
    m.call("decode_reset");
    position = 0;
    try {
      for (const id of ids.slice(0, at)) m.call("decode_advance", id);
      m.call("decode_probe", layer, component);
      m.call("decode_advance", ids[at]);
      return m.call("decode_probed").slice();
    } finally {
      m.call("decode_reset");
      position = 0;
    }
  }
  // The run after its first `length` tokens, kept in the module.
  function saveState(length = position) {
    if (!(length >= 1 && length <= position)) throw new Error(`prefix: ${length} positions asked, ${position} computed`);
    return { length, id: word(m.call("decode_save", length)) };
  }
  function restoreState(state) {
    m.call("decode_restore", state.id);
    position = state.length;
  }

  return {
    step, run, generate, probe, saveState, restoreState,
    reset: () => { m.call("decode_reset"); position = 0; },
    // A prefill moved the run past its prompt (GP-70).
    moved: (n) => { position += n; },
    get position() { return position; },
    device: "cpu",
    integer: true,
    sizes,
  };
}
