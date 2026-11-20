// The WebGPU half of rows 2 and 3: the same run as equality.mjs, in a browser
// on device webgpu, held to the CPU references in references/. It refuses to
// compare unless the engine the page answers for webgpu is the integer one: a
// float engine's tokens are not what rows 2 and 3 ask about, and its passing
// or failing would say nothing about them.
//
//   node equality-browser.mjs <site> <model key> <out.json> [--device webgpu|cpu] [--channel chrome|chromium] [--headed]
//
// <site> is a deployed site, served here from a local origin. The browser is
// the system's Chrome by default, headless (`--headed` shows it), which runs
// WebGPU on the machine's GPU;
// `--channel chromium` with SwiftShader is for a machine without one. The
// profile is fresh and deleted after, so the model is downloaded in the run.

import http from "node:http";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { extname, join, normalize, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { compareRuns } from "./equality-run.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [site, key, out] = args;
const device = option("--device", "webgpu");
const channel = option("--channel", "chrome");
const here = dirname(fileURLToPath(import.meta.url));
const reference = JSON.parse(await readFile(join(here, "references", `${key}.json`), "utf8"));
const playwright = process.env.PRISM_PLAYWRIGHT ?? "playwright";
const { chromium } = await import(playwright);

const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".html": "text/html", ".json": "application/json", ".wasm": "application/wasm" };
const server = http.createServer(async (request, response) => {
  const path = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname));
  if (path === "/" || path === "\\") {
    response.setHeader("content-type", "text/html");
    response.end("<!doctype html><title>equality</title>");
    return;
  }
  const file = path.startsWith("/bench/") || path.startsWith("\\bench\\") ? join(here, path.slice(7)) : join(site, path);
  try {
    const body = await readFile(file);
    response.setHeader("content-type", types[extname(file)] ?? "application/octet-stream");
    response.end(body);
  } catch {
    response.statusCode = 404;
    response.end();
  }
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
const origin = `http://127.0.0.1:${server.address().port}`;

const profile = await mkdtemp(join(tmpdir(), "equality-"));
const flags = channel === "chromium" ? ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] : ["--enable-unsafe-webgpu"];
const context = await chromium.launchPersistentContext(profile, { channel: channel === "chrome" ? "chrome" : undefined, headless: !args.includes("--headed"), args: flags });
let code = 1;
try {
  const page = context.pages()[0] ?? (await context.newPage());
  page.on("console", (m) => console.error(`[page] ${m.text()}`));
  await page.goto(`${origin}/`);
  const result = await page.evaluate(async ({ origin, model, device, tokens }) => {
    const { pipeline } = await import(`${origin}/app/engine/pipeline.mjs`);
    const { checklist } = await import(`${origin}/app/engine/loader.mjs`);
    const { referenceRun } = await import(`${origin}/bench/equality-run.mjs`);
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) : null;
    const info = adapter?.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : null;
    try {
      return { adapter: info, run: await referenceRun({ pipeline, checklist, model, device, tokens, onStatus: (s) => console.log(s) }) };
    } catch (e) {
      return { adapter: info, error: String(e?.message ?? e) };
    }
  }, { origin, model: reference.model, device, tokens: reference.tokens });
  if (result.error) throw new Error(result.error);
  const { run } = result;
  run.key = key;
  run.adapter = result.adapter;
  run.runner = `${channel} ${context.browser()?.version() ?? ""}`.trim();
  await writeFile(out, JSON.stringify(run, null, 1) + "\n");
  if (!run.integer) {
    console.log(JSON.stringify({ key, device: run.device, integer: false, verdict: `the page's ${run.device} engine is not the integer engine; rows 2 and 3 compare integer engines only` }));
  } else {
    const verdict = compareRuns(reference, run);
    console.log(JSON.stringify({ key, device: run.device, integer: true, adapter: run.adapter, verdict }));
    code = verdict.every((v) => v.equal) ? 0 : 1;
  }
} finally {
  await context.close();
  server.close();
  await rm(profile, { recursive: true, force: true });
}
process.exit(code);
