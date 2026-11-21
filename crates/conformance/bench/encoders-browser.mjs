// Row 9's browser half: the same two real encoders and five sentences as
// encoders.mjs, run in the published page in Chrome, every id and bit held to
// what encoders.mjs saved (out/<key>/cases.json), which the native module
// matched (engine_row9_real_exports).
//
//   node encoders-browser.mjs <encoders.mjs out dir> <out.json> [--page <url>] [--channel chrome|chromium] [--headed]

import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

const PUBLISHED = "https://hologram-technologies.github.io/prism-hologram-ai/app/";
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [dump, out] = args;
const pageUrl = option("--page", PUBLISHED);
const channel = option("--channel", "chrome");
if (!dump || !out) throw new Error("usage: node encoders-browser.mjs <encoders.mjs out dir> <out.json>");
const { chromium } = await import(process.env.PRISM_PLAYWRIGHT ?? "playwright");

const profile = await mkdtemp(join(tmpdir(), "row9-"));
const context = await chromium.launchPersistentContext(profile, { channel: channel === "chrome" ? "chrome" : undefined, headless: !args.includes("--headed") });
const record = { protocol: "m2-row9-browser/1", page: pageUrl, date: new Date().toISOString(), models: {} };
let code = 0;
try {
  for (const key of ["minilm", "sst2"]) {
    const saved = JSON.parse(await readFile(join(dump, key, "cases.json"), "utf8"));
    const page = await context.newPage();
    page.on("console", (m) => console.error(`[${key}] ${m.text()}`));
    await page.goto(pageUrl, { waitUntil: "load" });
    const run = await page.evaluate(async ({ repo, task, texts }) => {
      const hex = (bits) => Array.from(new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength), (b) => b.toString(16).padStart(2, "0")).join("");
      try {
        const { pipeline } = await import(new URL("engine/pipeline.mjs", location.href).href);
        const r = await pipeline(task, repo, {});
        const cases = [];
        for (const text of texts) {
          const said = await r(text);
          cases.push({ text, ids: Array.from(said.ids), label: said.index ?? null, bits: hex(said.bits) });
        }
        return { device: r.device ?? null, cases, userAgent: navigator.userAgent, crossOriginIsolated };
      } catch (e) {
        return { error: String(e?.message ?? e) };
      }
    }, { repo: saved.repo, task: saved.task, texts: saved.cases.map((c) => c.text) });
    await page.close();
    const same = !run.error && run.cases.every((c, i) => JSON.stringify(c.ids) === JSON.stringify(saved.cases[i].ids) && c.bits === saved.cases[i].bits && c.label === saved.cases[i].label);
    record.models[key] = { repo: saved.repo, task: saved.task, ...run, equalToNodeAndNative: same };
    console.log(JSON.stringify({ key, error: run.error ?? null, device: run.device, sentences: run.cases?.length, equalToNodeAndNative: same }));
    if (!same) code = 1;
  }
  await writeFile(out, JSON.stringify(record, null, 1) + "\n");
} finally {
  await context.close();
  await rm(profile, { recursive: true, force: true });
}
process.exit(code);
