// The integer engine's products on WebGPU (GP-56).
//
// Why: a token's time is its products (#106's profile: 86 per cent), and
// Hologram AI's inference is integers only, on the GPU as on the CPU. So the
// worker that runs the model hands each q4 product to this worker, which runs
// it with the four kernels the verified kernel's guest prints (decision 48,
// KN-49, KN-73): the activations put on 16-bit codes, the group sums, each
// row's 384-bit register and its one rounding. Every value is a u32 word and
// no kernel names a float type; the rows are hologram_ai_exact's bits, so the
// tokens are the CPU's.
//
// How: the model's instance asks for a product through its `kernel.product`
// import, which must answer before it returns, and WebGPU answers only on an
// event loop. So the model's worker writes the input into a SharedArrayBuffer,
// posts the product's number here and waits; this worker dispatches, reads
// the rows back, writes them where the model's worker reads them and wakes
// it. A product the printed kernels do not take (the output head, which is
// dense, or a matrix with binary32 scales) is computed here by an instance of
// the same engine.wasm holding only that product, the same bits.
import { moduleOf } from "./loader.mjs";

// The control words, as integer-helpers.mjs has them: how many answered and
// whether this worker failed. The input follows the control words; the rows
// follow the widest input.
export const ANSWERED = 1, FAILED = 4, CONTROL = 32;

// The widths whose group-sum kernel the guest prints (Kernel.Render's
// printedWidth), so a product of another width is not one this worker takes.
export const PRINTED = [128, 256, 1024, 1536, 2048, 3072, 8192, 8960];

// The text decision 48 prints for kernel `kind` with `units` and `width`: its
// pieces of 256 bytes, in order, up to the empty one. `dispatch` is the
// verified kernel's.
export async function printedKernels(dispatch) {
  const text = new TextDecoder();
  const printed = async (kind, units, width) => {
    let out = "";
    for (let piece = 0; ; piece++) {
      const request = new Uint8Array(17);
      const view = new DataView(request.buffer);
      request[0] = 48;
      request[1] = kind;
      view.setUint16(2, piece);
      view.setUint32(4, units);
      view.setUint32(8, width);
      const answer = text.decode(await dispatch(request));
      if (answer === "invalid request") throw new Error(`decision 48 prints no kernel ${kind} for ${units} units of ${width}`);
      if (answer === "") return out;
      out += answer;
    }
  };
  const sums = {};
  for (const width of PRINTED) sums[width] = await printed(1, width / 32, width);
  return { quantise: await printed(3, 0, 0), sums, accumulate: await printed(4, 0, 0), finish: await printed(5, 0, 0), dense: await printed(6, 0, 0) };
}

// The slices decode_slices wrote, by product number: for each, its number,
// first row and length, then the slice as hologram-ai-cpu-decode writes it.
function slicesOf(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = [];
  for (let i = 0; i < bytes.length;) {
    const [number, start, length] = [view.getUint32(i, true), view.getUint32(i + 4, true), view.getUint32(i + 8, true)];
    const record = bytes.subarray(i, i + 12 + length);
    const slice = bytes.subarray(i + 12, i + 12 + length);
    const at = (k) => view.getUint32(i + 12 + 4 * k, true);
    const kind = at(0);
    const one = { number, start, record, kind };
    if (kind === 1) {
      const [rows, columns, block, scaleKind, packed, codes, scales] = [at(1), at(2), at(3), at(4), at(5), at(6), at(7)];
      Object.assign(one, { rows, columns, block, scaleKind, packed, codes: slice.subarray(36, 36 + codes), scales: slice.subarray(36 + codes, 36 + codes + scales) });
    } else if (kind === 2) {
      // The head: its rows of half weights, row after row.
      const [rows, columns, length] = [at(1), at(2), at(3)];
      Object.assign(one, { rows, columns, weights: slice.subarray(16, 16 + length) });
    }
    out.push(one);
    i += 12 + length;
  }
  return out;
}

// Whether the printed kernels take a slice: a q4 matrix of blocks of 32, half
// scales, no zero points, of a printed width, whole; or the dense head, whole,
// whose row the dense kernel walks up to DENSE_WIDTH inputs (Engine.Gpu's
// denseAccumulate).
export const DENSE_WIDTH = 2048;
export const onGpu = (s) => (s.kind === 1 && s.block === 32 && s.scaleKind === 2 && s.packed === 0 && s.start === 0 && PRINTED.includes(s.columns))
  || (s.kind === 2 && s.start === 0 && s.columns % 32 === 0 && s.columns <= DENSE_WIDTH);

// Bytes as little-endian words, the last padded with zeros.
const words = (bytes) => {
  const out = new Uint32Array(((bytes.length + 3) >> 2) || 1);
  new Uint8Array(out.buffer).set(bytes);
  return out;
};

async function open({ compiled, slices, shared, rowsAt, kernels, port }) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("this worker has no WebGPU adapter");
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  const errors = [];
  device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));
  const control = new Int32Array(shared, 0, CONTROL / 4);
  const bytes = new Uint8Array(shared);
  const all = slicesOf(new Uint8Array(slices));

  // The products the printed kernels do not take, held by an instance of the
  // engine's module.
  // A slice whose weights one binding cannot hold stays on the module too.
  const bound = device.limits.maxStorageBufferBindingSize;
  const fits = (s) => (s.kind === 1 ? s.codes.length : s.weights.length) <= bound;
  const gpuSide = all.filter((s) => onGpu(s) && fits(s));
  const rest = all.filter((s) => !gpuSide.includes(s));
  let cpu = null;
  if (rest.length) {
    cpu = moduleOf(compiled);
    const held = new Uint8Array(rest.reduce((n, s) => n + s.record.length, 0));
    let at = 0;
    for (const s of rest) { held.set(s.record, at); at += s.record.length; }
    cpu.callWith("slices_hold", held);
  }

  // Every buffer is at least one word, as WebGPU binds none smaller.
  const empty = (count) => device.createBuffer({ size: 4 * (count > 0 ? count : 1), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const buffer = (data) => {
    const b = empty(data.length);
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  // One pipeline per kernel text and invocation count, every binding the text
  // declares read-only or read-write as it says.
  const pipelines = new Map();
  const pipelineOf = async (text, invocations, count) => {
    const key = `${invocations}:${text}`;
    if (!pipelines.has(key)) {
      const layout = device.createBindGroupLayout({
        entries: Array.from({ length: count }, (_, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: text.includes(`var<storage, read_write> b${i}:`) ? "storage" : "read-only-storage" } })),
      });
      const pipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code: text }), entryPoint: "main", constants: { invocations } },
      });
      pipelines.set(key, { pipeline, layout });
    }
    return pipelines.get(key);
  };
  const step = async (text, invocations, list) => {
    const { pipeline, layout } = await pipelineOf(text, invocations, list.length);
    return { pipeline, group: device.createBindGroup({ layout, entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })) }), workgroups: (invocations + 63) >> 6 };
  };

  // Per width, the input and its codes and powers; per product, its weights,
  // sums, registers and rows, and the four steps bound once.
  const inputs = new Map();
  const inputOf = async (width) => {
    if (!inputs.has(width)) {
      const units = width / 32;
      const input = empty(width), codes = empty(width), powers = empty(units);
      inputs.set(width, { input, codes, powers, quantise: await step(kernels.quantise, units, [input, codes, powers]) });
    }
    return inputs.get(width);
  };
  const products = new Map();
  for (const s of gpuSide) {
    const units = s.columns / 32;
    const shared = await inputOf(s.columns);
    const registers = empty(s.rows * 12), rows = empty(s.rows);
    const read = device.createBuffer({ size: 4 * s.rows, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const finish = await step(kernels.finish, s.rows, [empty(1), empty(1), empty(1), registers, rows]);
    let steps;
    if (s.kind === 1) {
      const codes = buffer(words(s.codes)), scales = buffer(words(s.scales)), sums = empty(s.rows * units);
      steps = [
        shared.quantise,
        await step(kernels.sums[s.columns], s.rows * units, [shared.codes, codes, empty(1), sums]),
        await step(kernels.accumulate, s.rows, [sums, scales, shared.powers, registers, rows, buffer(new Uint32Array([units]))]),
        finish,
      ];
    } else {
      // Each input's code times its half weight, placed in the row's register
      // at its group's power: the dense kernel reads the codes themselves.
      steps = [
        shared.quantise,
        await step(kernels.dense, s.rows, [shared.codes, buffer(words(s.weights)), shared.powers, registers, rows, buffer(new Uint32Array([s.columns]))]),
        finish,
      ];
    }
    products.set(s.number, { rows: s.rows, columns: s.columns, shared, registers, out: rows, read, steps });
  }

  // One product: `number`, its input `length` bytes at CONTROL; its rows go at rowsAt.
  const answer = async (number, length) => {
    const p = products.get(number);
    if (!p) {
      const out = cpu.callWith("slices_run", bytes.slice(CONTROL, CONTROL + length), number);
      bytes.set(out.subarray(4), rowsAt + 4 * new DataView(out.buffer, out.byteOffset, 4).getUint32(0, true));
      return;
    }
    if (length !== 4 * p.columns) throw new Error(`product ${number}: ${length / 4} inputs, not ${p.columns}`);
    device.queue.writeBuffer(p.shared.input, 0, bytes.slice(CONTROL, CONTROL + length));
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(p.registers);
    for (const s of p.steps) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.group);
      pass.dispatchWorkgroups(s.workgroups);
      pass.end();
    }
    encoder.copyBufferToBuffer(p.out, 0, p.read, 0, 4 * p.rows);
    device.queue.submit([encoder.finish()]);
    await p.read.mapAsync(GPUMapMode.READ);
    bytes.set(new Uint8Array(p.read.getMappedRange()), rowsAt);
    p.read.unmap();
    if (errors.length) throw new Error(`WebGPU: ${errors[0]}`);
  };

  port.onmessage = async ({ data }) => {
    try {
      await answer(data.number, data.length);
    } catch (error) {
      Atomics.store(control, FAILED, 1);
      console.error(error);
    }
    Atomics.store(control, ANSWERED, 1);
    Atomics.notify(control, ANSWERED);
  };
  return { gpu: products.size, cpu: rest.length, adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null };
}

// Run as a worker: the first message opens it, and its answer is "ready" with
// how many products each side holds, or the reason it could not open.
if (globalThis.name === "integer-gpu") {
  self.onmessage = async ({ data }) => {
    self.onmessage = null;
    try {
      self.postMessage({ ready: true, ...(await open(data)) });
    } catch (error) {
      self.postMessage({ ready: false, error: String(error?.message ?? error) });
    }
  };
}

// In the model's worker: route module `m`'s products to a WebGPU worker,
// with the kernels the page read from the verified kernel. Answers what the
// GPU worker holds and a function that stops it.
export async function useGpu(m, kernels, sizes) {
  const slices = m.call("decode_slices", 0, 1);
  const most = (...n) => n.reduce((a, b) => (b > a ? b : a), 0);
  const widest = most(sizes.hidden, sizes.heads * sizes.head_dim, sizes.ff);
  const tallest = most(sizes.vocab, sizes.hidden, sizes.ff, sizes.heads * sizes.head_dim);
  const shared = new SharedArrayBuffer(CONTROL + 4 * widest + 4 * tallest);
  const control = new Int32Array(shared, 0, CONTROL / 4);
  const bytes = new Uint8Array(shared);
  const rowsAt = CONTROL + 4 * widest;
  const channel = new MessageChannel();
  const worker = new Worker(new URL("./integer-gpu.mjs", import.meta.url), { type: "module", name: "integer-gpu" });
  const opened = await new Promise((done, refused) => {
    worker.onmessage = ({ data }) => (data.ready ? done(data) : refused(new Error(`the WebGPU worker did not open: ${data.error}`)));
    worker.onerror = (event) => refused(new Error(`the WebGPU worker did not start: ${event.message ?? "no reason given"}`));
    worker.postMessage({ compiled: m.compiled, slices: slices.buffer, shared, rowsAt, kernels, port: channel.port2 }, [slices.buffer, channel.port2]);
  });
  m.setProducts((number, at, len, out, cap) => {
    if (len > rowsAt - CONTROL || cap > bytes.length - rowsAt) return 1;
    bytes.set(new Uint8Array(m.x.memory.buffer, at, len), CONTROL);
    Atomics.store(control, ANSWERED, 0);
    Atomics.store(control, FAILED, 0);
    channel.port1.postMessage({ number, length: len });
    Atomics.wait(control, ANSWERED, 0);
    if (Atomics.load(control, FAILED)) return 1;
    new Uint8Array(m.x.memory.buffer, out, cap).set(bytes.subarray(rowsAt, rowsAt + cap));
    return 0;
  });
  m.call("decode_helpers", 1);
  return { ...opened, stop: () => { m.call("decode_helpers", 0); m.setProducts(null); worker.terminate(); } };
}
