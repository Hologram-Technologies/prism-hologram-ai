// The integer engine's products on several threads (GP-52).
//
// Why: a token's time is its products, and each output row of one is that
// row's weights times an input put on its integers the same way for every row
// (EN-92). So the worker that runs the model hands each product to helpers,
// fresh instances of the same engine.wasm each holding one slice of every
// matrix's rows, and the module joins their rows: the same bits, at once.
//
// How: the model's instance asks for a product through its `kernel.product`
// import, which must answer before it returns. The input goes into a
// SharedArrayBuffer, the helpers are woken, each writes its rows where they
// belong, and this worker waits until all have. That needs shared memory,
// which a browser gives only a cross-origin-isolated page (the engine page's
// service worker makes a return visit one); without it there are no helpers
// and every product is computed in the model's instance, the same bits.

// The control words: a turn number the helpers wait on, how many answered,
// the product asked, the input's length, and whether any helper failed.
const TURN = 0, ANSWERED = 1, NUMBER = 2, LENGTH = 3, FAILED = 4;
const CONTROL = 32;

// How many helpers to start: `asked` when the page names a number, else one
// fewer than the cores, at most eight, and none on two cores or fewer, where a
// helper only adds its messages. None without shared memory.
export function helperCount(asked) {
  if (typeof SharedArrayBuffer !== "function" || globalThis.crossOriginIsolated === false) return 0;
  if (Number.isInteger(asked) && asked >= 0) return asked < 16 ? asked : 16;
  const cores = globalThis.navigator?.hardwareConcurrency ?? 1;
  return cores > 2 ? (cores - 1 < 8 ? cores - 1 : 8) : 0;
}

// Start `count` helpers for module `m`, whose model is open, and route its
// products to them. Answers a function that stops them.
export async function useHelpers(m, count, sizes) {
  const most = (...n) => n.reduce((a, b) => (b > a ? b : a), 0);
  const widest = 4 * most(sizes.hidden, sizes.heads * sizes.head_dim, sizes.ff);
  const tallest = 4 * most(sizes.vocab, sizes.hidden, sizes.ff, sizes.heads * sizes.head_dim);
  const shared = new SharedArrayBuffer(CONTROL + widest + tallest);
  const control = new Int32Array(shared, 0, CONTROL / 4);
  const bytes = new Uint8Array(shared);
  const rows = CONTROL + widest;
  const helpers = [];
  try {
    for (let i = 0; i < count; i++) {
      const slices = m.call("decode_slices", i, count);
      const helper = new Worker(new URL("./integer-helper.mjs", import.meta.url), { type: "module" });
      helpers.push(helper);
      await new Promise((ready, refused) => {
        helper.onmessage = ({ data }) => (data === "ready" ? ready() : refused(new Error(`helper ${i}: ${data}`)));
        helper.onerror = (event) => refused(new Error(`helper ${i} did not start: ${event.message ?? "no reason given"}`));
        helper.postMessage({ compiled: m.compiled, slices, shared, input: CONTROL, rows, widest }, [slices.buffer]);
      });
    }
  } catch (error) {
    for (const helper of helpers) helper.terminate();
    throw error;
  }
  m.setProducts((number, at, len, out, cap) => {
    if (len > widest || cap > bytes.length - rows) return 1;
    bytes.set(new Uint8Array(m.x.memory.buffer, at, len), CONTROL);
    Atomics.store(control, NUMBER, number);
    Atomics.store(control, LENGTH, len);
    Atomics.store(control, ANSWERED, 0);
    Atomics.store(control, FAILED, 0);
    Atomics.add(control, TURN, 1);
    Atomics.notify(control, TURN);
    for (;;) {
      const answered = Atomics.load(control, ANSWERED);
      if (answered >= count) break;
      Atomics.wait(control, ANSWERED, answered);
    }
    if (Atomics.load(control, FAILED)) return 1;
    new Uint8Array(m.x.memory.buffer, out, cap).set(bytes.subarray(rows, rows + cap));
    return 0;
  });
  m.call("decode_helpers", count);
  return () => {
    m.call("decode_helpers", 0);
    m.setProducts(null);
    for (const helper of helpers) helper.terminate();
  };
}
