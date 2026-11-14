// The engine page: verify the kernel, load Bonsai through the checks, answer a
// prompt by greedy decoding. Everything shown is also on window.__qengine, so a
// test reads the verdicts and timings rather than the screen.

import { createEngine } from "./engine/gpu.mjs";
import { asker, checklist, hubRecord, loadContainer, openModule } from "./engine/loader.mjs";

// The model to run: Bonsai unless the address names another, so a link can say
// "run this model" (?repo=owner/name&path=file&index=hub index base).
const asked = new URLSearchParams(location.search);
const MODEL = {
  repo: asked.get("repo") ?? "HOLOGRAMTECH/q-ternary-bonsai-1.7b",
  path: asked.get("path") ?? "q-ternary-bonsai-1.7b.v1.holo",
  indexBase: asked.get("index") ?? "https://hologram-technologies.github.io/hub/data/index",
  fallback: asked.get("repo") ? null : "https://huggingface.co/HOLOGRAMTECH/q-ternary-bonsai-1.7b/resolve/c6d6a12384ecc52ae7e006bef263bfedf5e7496c/",
};
const CAPACITY = 2048;

const $ = (id) => document.getElementById(id);
const state = { phase: "idle", checks: [], output: null, stats: null, error: null };
window.__qengine = state;

const checks = checklist((item) => {
  const li = document.createElement("li");
  li.className = item.verdict === "pass" ? "pass" : item.verdict === "fail" ? "fail" : "";
  li.textContent = `${item.verdict} · ${item.subject} `;
  const detail = document.createElement("span");
  detail.className = "detail";
  detail.textContent = item.detail;
  li.append(detail);
  $("checks").append(li);
});
state.checks = checks.items;

const status = (message) => { $("status").textContent = message; };
const be = (value, width) => {
  const out = new Uint8Array(width);
  let v = BigInt(value);
  for (let i = width - 1; i >= 0; i--) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
};
const join = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};

let session = null;

async function load() {
  $("load").disabled = true;
  state.phase = "loading";
  const started = performance.now();
  try {
    status("Verifying the kernel…");
    const { load: loadHolo } = await import("../vendor/prism-code/src/browser.mjs");
    const kernel = await loadHolo("../app.holo");
    const verified = kernel.identities.verified;
    checks.add("kernel", "kernel holo verified in this browser", verified ? "pass" : "fail", kernel.identities.archiveFingerprint);
    if (!verified) throw new Error("the kernel did not verify");
    const ask = asker(kernel, checks);

    status("Opening the GPU…");
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) : null;
    const limit = adapter ? adapter.limits.maxStorageBufferBindingSize : 0;
    const device = adapter
      ? await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: limit, maxBufferSize: adapter.limits.maxBufferSize } })
      : null;

    status("Reading the model's record…");
    const record = await hubRecord({ indexBase: MODEL.indexBase, repo: MODEL.repo, checks });
    const file = record.files.find((f) => f.path === MODEL.path);
    if (!file || !file.sha256) throw new Error("the record names no SHA-256 for the model file");
    const base = record.origin?.resolve ?? MODEL.fallback;
    if (!base) throw new Error("the record names no origin and no fallback is configured");
    checks.add("record", "file named by the record", "pass", `${file.size} bytes, sha256 ${file.sha256}, from ${base}`);

    const module = await openModule("./engine/engine.wasm");
    status("Checking the container…");
    const { engine } = await loadContainer({
      url: base + MODEL.path,
      size: file.size,
      sha256: file.sha256,
      module,
      ask,
      checks,
      preflight: async (dims, entries) => {
        // The largest buffer the engine binds is one bit plane of the biggest matrix.
        // A ternary block is two planes of N*K/8 bytes and N*K/32 bytes of scales,
        // so one plane is 4/9 of the block.
        const largest = Math.max(...entries.map((e) => (e.length * 4) / 9));
        await ask("backend", "backend for this device", join(new Uint8Array([12, device ? 1 : 0]), be(limit, 8), be(Math.ceil(largest), 8)), "webgpu");
        status("Downloading and checking every block…");
        $("progress").hidden = false;
      },
      onProgress: (done, total) => { $("progress").value = done / total; },
      build: async (dims, tensors) => {
        status("Uploading to the GPU…");
        return createEngine({ device, dims, tensors, capacity: CAPACITY });
      },
    });
    session = { engine, module, kernel, ask };
    state.phase = "ready";
    state.loadSeconds = (performance.now() - started) / 1000;
    status(`Ready in ${state.loadSeconds.toFixed(1)} s · ${(engine.gpuBytes / 1048576).toFixed(0)} MB on the GPU`);
    $("send").disabled = false;
  } catch (error) {
    state.phase = "failed";
    state.error = String(error.message ?? error);
    status(`Stopped: ${state.error}`);
    checks.add("stop", "load stopped", "fail", state.error);
  }
}

// Qwen3's chat frame, with an empty thinking block so it answers directly.
const frame = (prompt) => `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;

function encode(text) {
  const bytes = session.module.callWith("tokeniser_encode", new TextEncoder().encode(text));
  return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4));
}
function decode(ids) {
  const bytes = new Uint8Array(new Uint32Array(ids).buffer);
  return new TextDecoder().decode(session.module.callWith("tokeniser_decode", bytes));
}

async function send(event) {
  event.preventDefault();
  if (!session) return;
  $("send").disabled = true;
  try {
    await answer();
  } catch (error) {
    state.error = String(error.message ?? error);
    status(`Stopped: ${state.error}`);
    checks.add("stop", "answer stopped", "fail", state.error);
  }
  $("send").disabled = false;
}

async function answer() {
  const { engine, ask } = session;
  const prompt = encode(frame($("prompt").value));
  // A stop is a special string the tokenizer reads as one token.
  const stops = new Set(["<|im_end|>", "<|endoftext|>"].map(encode).filter((ids) => ids.length === 1).map((ids) => ids[0]));
  engine.reset();
  const budget = Number(await ask("turn", "token budget for this turn", join(new Uint8Array([9]), be(CAPACITY, 8), be(prompt.length, 8), be(256, 8)), (a) => /^\d+$/.test(a)));
  const t0 = performance.now();
  let firstAt = t0;
  const out = await engine.run(prompt, budget, {
    stops,
    onFirst: () => { firstAt = performance.now(); },
    onToken: (ids) => { $("answer").textContent = decode(ids); },
  });
  const end = performance.now();
  state.output = { prompt: $("prompt").value, promptIds: prompt, promptTokens: prompt.length, tokens: out, text: decode(out) };
  state.stats = {
    timeToFirstToken: (firstAt - t0) / 1000,
    tokensPerSecond: out.length > 1 ? (out.length - 1) / ((end - firstAt) / 1000) : 0,
    generated: out.length,
  };
  $("stats").textContent = `${out.length} tokens · first token ${state.stats.timeToFirstToken.toFixed(2)} s · ${state.stats.tokensPerSecond.toFixed(1)} tokens/s (32 positions per submission)`;
}

$("load").addEventListener("click", load);
$("ask").addEventListener("submit", send);
