// The CPU half of rows 2 and 3: the integer CPU engine's 256 tokens of the
// three prompts, run in Node on a deployed site's own files (its engine.wasm,
// its kernel holo, its page modules), written as the reference the WebGPU
// half is held to. Given an existing reference, it checks a new run against
// it instead and fails on the first id that differs.
//
//   node equality.mjs <site> <model cache> <model key> <out.json> [--check <reference.json>]
//
// <site> is a deployed site (the Pages artifact, or `just page-journey`'s
// target/page-journey); <model key> one of equality-run.mjs's MODELS.

import { writeFile, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { installShims } from "./node-shims.mjs";
import { MODELS, referenceRun, compareRuns } from "./equality-run.mjs";

const [site, cacheDir, key, out, flag, against] = process.argv.slice(2);
if (!MODELS[key]) throw new Error(`model key ${key}: one of ${Object.keys(MODELS).join(", ")}`);
const net = installShims({ cacheDir });
const engineDir = path.join(site, "app", "engine");
const { pipeline } = await import(pathToFileURL(path.join(engineDir, "pipeline.mjs")).href);
const { checklist } = await import(pathToFileURL(path.join(engineDir, "loader.mjs")).href);
const sha256 = async (file) => createHash("sha256").update(await readFile(file)).digest("hex");

const result = await referenceRun({ pipeline, checklist, model: MODELS[key], device: "cpu", onStatus: (s) => console.error(`[${key}] ${s}`) });
if (!result.integer) throw new Error(`device cpu answered a float engine: the CPU reference must be the integer engine`);
result.key = key;
result.site = {
  engineWasmSha256: await sha256(path.join(engineDir, "engine.wasm")),
  holoSha256: await sha256(path.join(site, "app.holo")),
  pipelineSha256: await sha256(path.join(engineDir, "pipeline.mjs")),
  loaderSha256: await sha256(path.join(engineDir, "loader.mjs")),
};
result.runner = `node ${process.version}, ${process.platform}`;
result.modelFiles = [...new Set(net.filter((n) => n.file).map((n) => n.file))];
await writeFile(out, JSON.stringify(result, null, 1) + "\n");

if (flag === "--check") {
  const reference = JSON.parse(await readFile(against, "utf8"));
  const verdict = compareRuns(reference, result);
  console.log(JSON.stringify({ key, device: result.device, integer: result.integer, verdict }));
  // The engine's worker would keep Node running; the run is over.
  process.exit(verdict.some((v) => !v.equal) ? 1 : 0);
}
console.log(JSON.stringify({ key, device: result.device, integer: result.integer, runs: result.runs.map((r) => ({ prompt: r.prompt, ids: r.ids.length, idsSha256: r.idsSha256, endOfTurnAt: r.endOfTurnAt })) }));
process.exit(0);
