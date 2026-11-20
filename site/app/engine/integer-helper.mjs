// One helper of the integer engine (GP-52, integer-helpers.mjs says why): a
// fresh instance of the engine's module holding one slice of every matrix's
// rows. It waits on the shared turn number, computes its rows of the product
// asked, writes them where they belong and says it answered.
import { moduleOf } from "./loader.mjs";

const TURN = 0, ANSWERED = 1, NUMBER = 2, LENGTH = 3, FAILED = 4;

self.onmessage = ({ data }) => {
  self.onmessage = null;
  let m;
  try {
    m = moduleOf(data.compiled);
    m.callWith("slices_hold", new Uint8Array(data.slices));
  } catch (error) {
    self.postMessage(String(error?.message ?? error));
    return;
  }
  const control = new Int32Array(data.shared, 0, 8);
  const bytes = new Uint8Array(data.shared);
  let seen = Atomics.load(control, TURN);
  self.postMessage("ready");
  for (;;) {
    Atomics.wait(control, TURN, seen);
    seen = Atomics.load(control, TURN);
    try {
      const input = bytes.slice(data.input, data.input + Atomics.load(control, LENGTH));
      const out = m.callWith("slices_run", input, Atomics.load(control, NUMBER));
      const first = new DataView(out.buffer, out.byteOffset, 4).getUint32(0, true);
      const at = data.rows + 4 * first;
      if (at + out.length - 4 > bytes.length) throw new Error("rows past the shared buffer");
      bytes.set(out.subarray(4), at);
    } catch {
      Atomics.store(control, FAILED, 1);
    }
    Atomics.add(control, ANSWERED, 1);
    Atomics.notify(control, ANSWERED);
  }
};
