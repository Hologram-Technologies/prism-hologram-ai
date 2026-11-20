// Drives speed-run.mjs's protocol in a browser on the published page, for this
// engine or for transformers.js, and writes the result as JSON.
//
//   node speed.mjs --lib hologram|transformers --model <repo> --machine "<name>" --out <file>
//                  [--devices webgpu:short+long,wasm:short] [--dtype q4f16]
//                  [--page <url>] [--cdp <endpoint>] [--channel chrome|chromium] [--headed]
//                  [--protocol prefix-run.mjs]   (row 7, prefix reuse: this engine only)
//
// Both libraries run on the same published page, so they share its origin,
// network path and browser; the page's CSP is bypassed for the run so
// transformers.js can be imported from its CDN. Nothing here is shipped.
//
// The devices run one after another in one browser profile. The first is the
// cold load: the profile holds nothing, so the model is downloaded and
// checked in full. Each later device loads the model from what the first kept
// in the browser, so a model is downloaded once per library; its load is
// recorded as such, never as cold.
//
// On this machine the browser is the system's Chrome, headless, which runs
// WebGPU on the machine's GPU and is not throttled as a hidden tab is
// (`--headed` shows the window), with a profile made for the run and deleted
// after. On a phone,
// `--cdp` connects to its Chrome (`adb forward tcp:9222
// localabstract:chrome_devtools_remote`, then `--cdp http://127.0.0.1:9222`);
// every byte the page's origin, the CDN and Hugging Face stored is cleared
// before the first load, which is the phone's cold start.

import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

const PUBLISHED = "https://hologram-technologies.github.io/prism-hologram-ai/app/";
// Pinned, and used for the comparison only (docs/Q_ENGINE_PLAN.md: nothing
// from transformers.js or ONNX Runtime ships).
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const lib = option("--lib");
const model = option("--model");
const dtype = option("--dtype", "q4f16");
const machine = option("--machine");
const out = option("--out");
const devices = option("--devices", "webgpu:short+long,wasm:short").split(",").map((d) => {
  const [device, legs = "short+long"] = d.split(":");
  return { device, legs: legs.split("+") };
});
const pageUrl = option("--page", PUBLISHED);
const cdp = option("--cdp");
const channel = option("--channel", "chrome");
if (!["hologram", "transformers"].includes(lib) || !model || !machine || !out) {
  throw new Error("usage: --lib hologram|transformers --model <repo> --machine <name> --out <file> [--devices webgpu:short+long,wasm:short]");
}
const here = dirname(fileURLToPath(import.meta.url));
const protocolFile = option("--protocol", "speed-run.mjs");
const protocol = await readFile(join(here, protocolFile), "utf8");
const { chromium } = await import(process.env.PRISM_PLAYWRIGHT ?? "playwright");

let context;
let profile = null;
if (cdp) {
  const browser = await chromium.connectOverCDP(cdp);
  context = browser.contexts()[0];
} else {
  profile = await mkdtemp(join(tmpdir(), "speed-"));
  const flags = channel === "chromium" ? ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] : ["--enable-unsafe-webgpu"];
  context = await chromium.launchPersistentContext(profile, { channel: channel === "chrome" ? "chrome" : undefined, headless: !args.includes("--headed"), args: flags, viewport: { width: 1200, height: 900 } });
}
let code = 0;
try {
  const page = await context.newPage();
  page.on("console", (m) => { if (!m.text().includes("powerPreference")) console.error(`[page] ${m.text()}`); });
  const session = await context.newCDPSession(page);
  if (cdp) {
    for (const origin of [new URL(pageUrl).origin, new URL(TRANSFORMERS).origin, "https://huggingface.co"]) {
      await session.send("Storage.clearDataForOrigin", { origin, storageTypes: "all" });
    }
    await session.send("Network.clearBrowserCache");
  }
  await session.send("Page.setBypassCSP", { enabled: true });
  await page.goto(pageUrl, { waitUntil: "load" });
  const environment = await page.evaluate(async () => {
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }).catch(() => null) : null;
    return {
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemory: navigator.deviceMemory ?? null,
      adapter: adapter?.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : null,
      crossOriginIsolated: globalThis.crossOriginIsolated,
    };
  });
  // Which deploy was measured: its kernel holo's and engine's SHA-256.
  const deploy = await page.evaluate(async () => {
    const sha = async (path) => {
      const r = await fetch(new URL(path, location.href), { cache: "no-store" });
      const hash = await crypto.subtle.digest("SHA-256", await r.arrayBuffer());
      return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("");
    };
    // Which engine the deploy runs on WebGPU: the integer one is the module
    // row 15 adds (Q_ENGINE_PLAN.md: site/app/engine/integer-gpu.mjs), and
    // the page imports it from pipeline.mjs once it is the WebGPU engine.
    const source = await (await fetch(new URL("engine/pipeline.mjs", location.href), { cache: "no-store" })).text();
    return {
      holoSha256: await sha("../app.holo"),
      engineWasmSha256: await sha("engine/engine.wasm"),
      pipelineSha256: await sha("engine/pipeline.mjs"),
      webgpuEngine: /from "\.\/integer-gpu\.mjs"/.test(source) ? "integer" : "float",
    };
  });
  const runs = [];
  for (const [i, { device, legs }] of devices.entries()) {
    // A prefix sample needs an engine that has held no head, and a second
    // engine in one page can run out of the module's memory: a fresh page each.
    if (i > 0 && protocolFile !== "speed-run.mjs") await page.reload({ waitUntil: "load" });
    const result = await page.evaluate(async ({ protocol, lib, model, device, dtype, legs, transformers }) => {
      const url = URL.createObjectURL(new Blob([protocol], { type: "text/javascript" }));
      const { measure } = await import(url);
      const library = () => import(lib === "hologram" ? new URL("engine/transformers.mjs", location.href).href : transformers);
      const counter = () => import(transformers);
      try {
        return await measure({ library, counter, model, device, dtype, legs, onStatus: (s) => console.log(`${device}: ${s}`) });
      } catch (e) {
        return { device, error: String(e?.message ?? e) };
      }
    }, { protocol, lib, model, device, dtype, legs, transformers: TRANSFORMERS });
    result.load = i === 0 ? "cold" : "from this browser";
    runs.push(result);
    if (result.error) code = 1;
    console.log(JSON.stringify({ lib, model, device, load: result.load, error: result.error ?? null, loadSeconds: result.loadSeconds, cold: result.cold?.firstToken, warm: result.short?.firstToken, tokensPerSecond: result.short?.tokensPerSecond, longFirst: result.long?.firstToken, longTokensPerSecond: result.long?.tokensPerSecond }));
  }
  const guest = lib === "hologram" && protocolFile === "speed-run.mjs"
    ? await page.evaluate(async (protocol) => (await import(URL.createObjectURL(new Blob([protocol], { type: "text/javascript" })))).guestOverhead(new URL("..", location.href).href), protocol)
    : null;
  if (guest) console.log(JSON.stringify({ guest }));
  const record = { lib, library: lib === "transformers" ? TRANSFORMERS : pageUrl, model, dtype, machine, date: new Date().toISOString(), page: pageUrl, deploy, environment, runs, guest };
  await writeFile(out, JSON.stringify(record, null, 1) + "\n");
} finally {
  if (!cdp) await context.close();
  if (profile) await rm(profile, { recursive: true, force: true });
}
process.exit(code);
