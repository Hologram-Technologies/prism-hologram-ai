// Row 15 of milestone 2: the integer engine on WebGPU against the integer
// engine on the CPU, on the published page, held by receipts (GP-58, GP-59).
//
//   node row15.mjs <model repo> <out.json> [--page <url>] [--channel chrome|chromium] [--headed]
//
// Why each step is taken as it is:
//   a first visit installs the page's service worker, and the run happens on
//   the return visit, which the worker makes cross-origin isolated: the
//   integer engine waits on the GPU only there (pipeline.mjs refuses webgpu
//   otherwise), and helpers share memory only there;
//   each device runs in a page of its own, one after the other, so one
//   engine's memory never limits the other's;
//   the three prompts and 256 tokens are rows 2 and 3's (equality-run.mjs);
//   each device then checks the other's three receipts with generate.check,
//   which replays the receipt's own ids on this device and asks the kernel
//   (decision 31) to compare the output labels: "confirmed" is the claim.
// Labels compared here: the model's, each input's and each output's.

import { writeFile, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { PROMPTS, TOKENS } from "./equality-run.mjs";

const PUBLISHED = "https://hologram-technologies.github.io/prism-hologram-ai/app/";
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [model, out] = args;
const pageUrl = option("--page", PUBLISHED);
const channel = option("--channel", "chrome");
if (!model || !out) throw new Error("usage: node row15.mjs <model repo> <out.json>");
const { chromium } = await import(process.env.PRISM_PLAYWRIGHT ?? "playwright");

const profile = await mkdtemp(join(tmpdir(), "row15-"));
const flags = channel === "chromium" ? ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] : ["--enable-unsafe-webgpu"];
const context = await chromium.launchPersistentContext(profile, { channel: channel === "chrome" ? "chrome" : undefined, headless: !args.includes("--headed"), args: flags });
const record = { protocol: "m2-row15/1", model, page: pageUrl, date: new Date().toISOString(), tokens: TOKENS, prompts: PROMPTS.map((p) => p.name) };
let code = 0;
try {
  // The first visit: the worker installs and claims the page.
  const first = await context.newPage();
  first.on("console", (m) => console.error(`[first] ${m.text()}`));
  await first.goto(pageUrl, { waitUntil: "load" });
  await first.evaluate(() => navigator.serviceWorker?.ready.then(() => true));
  await first.close();

  // One device in a fresh page of the return visit: its receipts, and its
  // checks of the receipts given.
  const onDevice = async (device, toCheck) => {
    const page = await context.newPage();
    page.on("console", (m) => { if (!m.text().includes("powerPreference")) console.error(`[${device}] ${m.text()}`); });
    await page.goto(pageUrl, { waitUntil: "load" });
    const result = await page.evaluate(async ({ model, device, prompts, tokens, toCheck }) => {
      const env = { crossOriginIsolated: globalThis.crossOriginIsolated, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency };
      const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }).catch(() => null) : null;
      env.adapter = adapter?.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description } : null;
      try {
        const { pipeline } = await import(new URL("engine/pipeline.mjs", location.href).href);
        const { checklist } = await import(new URL("engine/loader.mjs", location.href).href);
        const checks = checklist();
        const t0 = performance.now();
        const run = await pipeline("text-generation", model, { device, dtype: "q4f16", checks });
        const loadSeconds = (performance.now() - t0) / 1000;
        const turns = [];
        for (const p of prompts) {
          const s = performance.now();
          const turn = await run(p.messages, { max_new_tokens: tokens });
          turns.push({ name: p.name, receipt: turn.receipt, tokens: turn.tokens.length, seconds: (performance.now() - s) / 1000 });
          console.log(`${p.name}: ${turn.tokens.length} tokens in ${((performance.now() - s) / 1000).toFixed(1)} s`);
        }
        const checked = [];
        for (const r of toCheck) {
          const c = await run.check(r);
          checked.push({ verdict: c.verdict, ready: c.ready, claimed: c.claimed, replayed: c.replayed });
          console.log(`check: ${c.verdict}`);
        }
        await run.engine?.dispose?.();
        return { env, device: run.device, dtype: run.dtype, identity: run.identity, promised: run.promised, loadSeconds, turns, checked, gpuEngine: checks.items.filter((c) => c.step === "device").map((c) => c.detail) };
      } catch (e) {
        return { env, error: String(e?.message ?? e) };
      }
    }, { model, device, prompts: PROMPTS, tokens: TOKENS, toCheck });
    await page.close();
    return result;
  };

  record.webgpu = await onDevice("webgpu", []);
  record.cpu = await onDevice("cpu", record.webgpu.turns?.map((t) => t.receipt) ?? []);
  // WebGPU checks the CPU's receipts in a page of its own.
  record.webgpuChecks = record.cpu.turns ? (await onDevice("webgpu", record.cpu.turns.map((t) => t.receipt))).checked ?? null : null;

  const g = record.webgpu, c = record.cpu;
  const same = (f) => g.turns && c.turns && g.turns.every((t, i) => f(t.receipt) === f(c.turns[i].receipt));
  record.summary = {
    error: g.error ?? c.error ?? null,
    isolated: { webgpu: g.env?.crossOriginIsolated, cpu: c.env?.crossOriginIsolated },
    devices: { webgpu: g.device, cpu: c.device },
    adapter: g.env?.adapter ?? null,
    modelLabelEqual: Boolean(g.identity && c.identity && g.identity.label === c.identity.label),
    inputLabelsEqual: Boolean(same((r) => r?.input?.label)),
    outputLabelsEqual: Boolean(same((r) => r?.output?.label)),
    classes: { webgpu: g.promised, cpu: c.promised },
    cpuChecksGpu: c.checked?.map((x) => x.verdict) ?? null,
    gpuChecksCpu: record.webgpuChecks?.map((x) => x.verdict) ?? null,
    tokens: g.turns?.map((t) => t.tokens) ?? null,
  };
  console.log(JSON.stringify(record.summary));
  const s = record.summary;
  if (s.error || !s.modelLabelEqual || !s.inputLabelsEqual || !s.outputLabelsEqual || !s.cpuChecksGpu?.every((v) => v === "confirmed") || !s.gpuChecksCpu?.every((v) => v === "confirmed")) code = 1;
  await writeFile(out, JSON.stringify(record, null, 1) + "\n");
} finally {
  await context.close();
  await rm(profile, { recursive: true, force: true });
}
process.exit(code);
