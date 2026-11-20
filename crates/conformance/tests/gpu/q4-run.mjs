// GP-21's browser half: run the q4 product's kernels, as the guest prints them (decision 48's
// answers in the claims the app recipe replays), on WebGPU in headless Chromium, over cases the
// Rust half computed with hologram_ai_exact. Prints one JSON line: each case's quantised codes,
// powers, group sums and rows, as the GPU wrote them. The Rust half compares them.
//
//   node crates/conformance/tests/gpu/q4-run.mjs <repository root> <cases.json>

import http from "node:http";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const root = process.argv[2];
const cases = JSON.parse(await readFile(process.argv[3], "utf8"));
const playwright = process.env.PRISM_PLAYWRIGHT ?? "/opt/prismpm/oracles/node_modules/playwright/index.mjs";
const { chromium } = await import(playwright);

// The guest's text of one kernel: its pieces of 256 bytes, in order, up to the empty one.
const claims = JSON.parse(await readFile(join(root, "crates/conformance/tests/kernel/claims/048.json"), "utf8")).claims;
const answers = new Map(claims.map((one) => [one.request, one.answer]));
const printed = (kind, units, width) => {
  let text = "";
  for (let piece = 0; ; piece++) {
    const request = "30" + kind.toString(16).padStart(2, "0") + piece.toString(16).padStart(4, "0")
      + units.toString(16).padStart(8, "0") + width.toString(16).padStart(8, "0") + "0000000000";
    const answer = answers.get(request);
    if (answer === undefined) throw new Error(`no claim for decision 48's ${request}`);
    if (answer === "") return text;
    text += answer;
  }
};
const kernels = { quantise: printed(3, 0, 0), sums: printed(1, 4, 128), accumulate: printed(4, 0, 0), finish: printed(5, 0, 0) };

const server = http.createServer((request, response) => {
  response.setHeader("content-type", "text/html");
  response.end("<!doctype html><title>q4</title>");
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
const browser = await chromium.launch({ headless: true, executablePath: process.env.PRISM_CHROME, args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async ({ kernels, cases }) => {
    if (!navigator.gpu) return { error: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: "no WebGPU adapter" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));
    // One dispatch: every binding the text declares, read-only or read-write as it says, and the
    // read-write ones read back.
    const run = async (wgsl, invocations, words) => {
      const writable = words.map((_, i) => wgsl.includes(`var<storage, read_write> b${i}:`));
      const layout = device.createBindGroupLayout({
        entries: words.map((_, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: writable[i] ? "storage" : "read-only-storage" } })),
      });
      const pipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main", constants: { invocations } },
      });
      const buffers = words.map((list) => {
        const data = new Uint32Array(list.length ? list : [0]);
        const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      });
      const reads = buffers.map((buffer, i) => writable[i] ? device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout, entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })) }));
      pass.dispatchWorkgroups(Math.ceil(invocations / 64));
      pass.end();
      buffers.forEach((buffer, i) => reads[i] && encoder.copyBufferToBuffer(buffer, 0, reads[i], 0, buffer.size));
      device.queue.submit([encoder.finish()]);
      const out = [];
      for (const read of reads) {
        if (!read) { out.push(null); continue; }
        await read.mapAsync(GPUMapMode.READ);
        out.push(Array.from(new Uint32Array(read.getMappedRange().slice(0))));
        read.unmap();
      }
      return out;
    };
    const packWords = (bytes) => Array.from({ length: Math.ceil(bytes.length / 4) }, (_, i) =>
      (bytes[4 * i] | (bytes[4 * i + 1] << 8) | ((bytes[4 * i + 2] ?? 0) << 16) | ((bytes[4 * i + 3] ?? 0) << 24)) >>> 0);
    const results = [];
    for (const c of cases) {
      const units = c.columns / 32;
      const q = await run(kernels.quantise, units, [c.input, new Array(c.columns).fill(0), new Array(units).fill(0)]);
      const s = await run(kernels.sums, c.rows * units, [q[1], packWords(c.codes), [0], new Array(c.rows * units).fill(0)]);
      const a = await run(kernels.accumulate, c.rows, [s[3], packWords(c.scales), q[2], new Array(c.rows * 12).fill(0), new Array(c.rows).fill(0), [units]]);
      const f = await run(kernels.finish, c.rows, [[0], [0], [0], a[3], new Array(c.rows).fill(0)]);
      results.push({ codes: q[1], powers: q[2], sums: s[3], rows: f[4] });
    }
    return { adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null, errors, results };
  }, { kernels, cases });
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}
