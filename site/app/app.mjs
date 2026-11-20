// The engine page: verify the kernel, load an ONNX export through the checks,
// answer a prompt by greedy decoding on the integer engine. Everything shown is also on window.__qengine, so a
// test reads the verdicts and timings rather than the screen.

import { asker, checklist } from "./engine/loader.mjs";
import { pipeline } from "./engine/pipeline.mjs";

// The model to run: Qwen3-0.6B's ONNX export, read from its own files with no
// repack, unless the address names another (?onnx=owner/name&precision=q4f16,
// and &index= for another hub index). Milestone 2 runs ONNX exports only; the
// repacked Bonsai container returns in milestone 3, on the integer engine.
const asked = new URLSearchParams(location.search);
const MODEL = {
  onnx: asked.get("onnx") ?? "onnx-community/Qwen3-0.6B-ONNX",
  indexBase: asked.get("index") ?? "https://hologram-technologies.github.io/hub/data/index",
  precision: asked.get("precision"),
};

const $ = (id) => document.getElementById(id);
// A link naming another model says which one the page will load.
if (asked.get("onnx")) $("model-name").textContent = `${MODEL.onnx}, its ${MODEL.precision ?? "chosen"} ONNX export read from its own files, from the index at ${MODEL.indexBase}.`;
const state = { phase: "idle", checks: [], output: null, stats: null, error: null };
window.__qengine = state;

// This page's own worker (sw.js beside it) keeps the page's files, the kernel
// and the engine, so a return visit loads with the network off. The site's
// worker (../sw.js) is the template's frame and is left as it is.
if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});

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

let session = null;

// The kernel this page was deployed with. The site's service worker serves
// app.holo from its cache, and a cache from an earlier deploy can outlive it
// (UOR-Foundation/template#83): an old kernel then refuses requests it never
// knew, and its refusal would read as a verdict. precache.json is written at
// deploy with every served file's BLAKE3 and the worker never answers it, so
// it is the one listing the cache cannot serve stale.
const HOLO = "../app.holo";
const fetchBytes = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
};
// With the network off there is no deploy to compare with: the page and its
// kernel are the ones this page's worker kept, and the check says so.
const OFFLINE = "offline";
async function deployedKernel() {
  let response;
  try {
    response = await fetch("../precache.json", { cache: "no-store" });
  } catch {
    return OFFLINE;
  }
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`precache.json: HTTP ${response.status}`);
  const entry = (await response.json()).files.find((f) => f.path === "./app.holo");
  if (!entry) throw new Error("precache.json lists no app.holo");
  return entry.blake3;
}
async function sameKernel(holo, listed) {
  const subject = "kernel is the one this deploy lists";
  const { blake3Hex } = await import("../vendor/prism-code/src/blake3.mjs");
  const held = blake3Hex(holo);
  // A page served outside a deploy (a local preview) has no listing to compare.
  if (listed === null) return checks.add("kernel", subject, "not checked", `no precache.json; held blake3 ${held}`);
  if (listed === OFFLINE) return checks.add("kernel", subject, "not checked", `the network is off, so no deploy listing to compare; held blake3 ${held}, verified above`);
  if (held === listed) return checks.add("kernel", subject, "pass", `blake3 ${held}`);
  checks.add("kernel", subject, "fail", `deployed blake3 ${listed}, held blake3 ${held}`);
  // Remove the stale copy so the next load fetches this deploy's kernel.
  for (const name of globalThis.caches ? await caches.keys() : []) {
    if (name.startsWith("prism-")) await (await caches.open(name)).delete(new URL(HOLO, location.href).href, { ignoreSearch: true });
  }
  throw new Error(`stale kernel: this page was deployed with kernel ${listed.slice(0, 12)}…, the browser held ${held.slice(0, 12)}… from an old cache. The old copy is removed; reload the page.`);
}

async function load() {
  $("load").disabled = true;
  state.phase = "loading";
  const started = performance.now();
  try {
    status("Verifying the kernel…");
    const { loadArchive } = await import("../vendor/prism-code/src/index.mjs");
    const [holo, listed] = await Promise.all([fetchBytes(HOLO), deployedKernel()]);
    const kernel = await loadArchive(holo);
    const verified = kernel.identities.verified;
    checks.add("kernel", "kernel holo verified in this browser", verified ? "pass" : "fail", kernel.identities.archiveFingerprint);
    if (!verified) throw new Error("the kernel did not verify");
    await sameKernel(holo, listed);
    const ask = asker(kernel, checks);

    status("Opening the GPU…");
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) : null;
    const limit = adapter ? adapter.limits.maxStorageBufferBindingSize : 0;
    const device = adapter
      ? await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: limit, maxBufferSize: adapter.limits.maxBufferSize } })
      : null;

    await loadOnnxModel({ kernel, ask, device, started });
  } catch (error) {
    state.phase = "failed";
    state.error = String(error.message ?? error);
    status(`Stopped: ${state.error}`);
    checks.add("stop", "load stopped", "fail", state.error);
  }
}

// An ONNX export from its own files, by the one call any page can make
// (site/app/engine/pipeline.mjs). The page hands it the kernel it verified
// against its deploy, and shows what the call reports.
async function loadOnnxModel({ kernel, ask, device, started }) {
  $("progress").hidden = false;
  const generate = await pipeline("text-generation", MODEL.onnx, {
    dtype: MODEL.precision ?? "auto",
    index: MODEL.indexBase,
    kernel,
    ask,
    gpu: device,
    checks,
    onStatus: status,
    onProgress: (fraction) => { $("progress").value = fraction; },
  });
  session = { generate };
  state.phase = "ready";
  state.device = generate.device;
  state.integer = generate.engine.integer === true;
  state.gpu = generate.engine.gpu ?? null;
  state.isolated = globalThis.crossOriginIsolated === true;
  state.precision = generate.dtype;
  state.loadSeconds = (performance.now() - started) / 1000;
  status(`Ready in ${state.loadSeconds.toFixed(1)} s · ${generate.device === "webgpu" ? `${generate.engine.gpu.products} products on the GPU, in integers` : "on the CPU"}`);
  $("send").disabled = false;
}


async function send(event) {
  event.preventDefault();
  // Enter reaches here too, so a turn in progress is not started twice.
  if (!session || $("send").disabled) return;
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
  // An empty box asks the example its placeholder shows.
  const text = $("prompt").value.trim() || $("prompt").placeholder;
  const out = await session.generate(text, { max_new_tokens: 256, onToken: (shown) => { $("answer").textContent = shown; } });
  state.output = { prompt: out.prompt, promptIds: out.promptIds, promptTokens: out.promptTokens, tokens: out.tokens, text: out.text, reuse: out.reuse, computed: out.computed };
  state.stats = { timeToFirstToken: out.timeToFirstToken, tokensPerSecond: out.tokensPerSecond, generated: out.tokens.length };
  const reused = out.promptTokens - out.computed;
  $("stats").textContent = `${out.tokens.length} tokens · first token ${out.timeToFirstToken.toFixed(2)} s · ${out.tokensPerSecond.toFixed(1)} tokens/s (32 positions per submission)${reused ? ` · ${reused} of ${out.promptTokens} prompt positions reused` : ""}`;
}

$("load").addEventListener("click", load);
// The button is disabled in the page until this module is listening, so a click is never lost.
$("load").disabled = false;
$("ask").addEventListener("submit", send);
// Enter sends, Shift+Enter starts a new line; Enter that ends an IME composition does neither.
$("prompt").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  $("ask").requestSubmit();
});
