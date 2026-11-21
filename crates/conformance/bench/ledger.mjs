// Writes the measurements of rows 2, 3, 11 and 16 to model/ledger.toml as
// open claims (R2: measured and reported, never asserted). A claim with the
// same id is replaced, so a rerun on the same machine updates its row.
//
//   node ledger.mjs equality <reference.json> <float.json> [--note <how the run departs from the deploy>]
//   node ledger.mjs speed <machine id> <hologram.json> <transformers.json> [--bound <target>=<step> ...]
//
// <machine id> names the device in the claim's id: Z13 for the laptop the
// plan names, PHONE for the phone. Row 16's targets are judged on this
// engine's integer engines only: on WebGPU once the deploy's WebGPU engine is
// the integer one, and on the CPU always. A target that is missed is written
// with the step that bounds it, given by --bound; the writer refuses a miss
// without one, and no target is ever written lower than the plan states.

import { readFile, writeFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const LEDGER = join(here, "..", "..", "..", "model", "ledger.toml");

// Row 16 of milestone 2, as the plan states it.
const TARGETS = {
  "warm-first-token": { model: "onnx-community/Qwen3-0.6B-ONNX", text: "warm time to first token under 300 ms on Qwen3-0.6B", met: (m) => m.warm !== null && m.warm < 0.3, value: (m) => `${ms(m.warm)} ms` },
  "tokens-per-second": { model: "onnx-community/Qwen2.5-1.5B-Instruct", text: "at least 100 tokens per second on Qwen2.5-1.5B at 4 bits", met: (m) => m.tps !== null && m.tps >= 100, value: (m) => `${fixed(m.tps, 1)} tokens per second` },
  "guest-call": { text: "guest call overhead under 50 µs warm", met: (m) => m.guest !== null && m.guest < 50, value: (m) => `${fixed(m.guest, 1)} µs` },
  "holo-size": { text: "the holo under 16 MB", met: (m) => m.holo !== null && m.holo < 16 * 2 ** 20, value: (m) => `${m.holo} bytes` },
};

const ms = (s) => (s === null || s === undefined ? "n/a" : Math.round(s * 1000).toLocaleString("en-US"));
const fixed = (x, d) => (x === null || x === undefined ? "n/a" : x.toFixed(d));
const mb = (bytes) => (bytes / 1e6).toFixed(1);
const short = (hex) => `${hex.slice(0, 8)}...`;
const keyOf = (model) => model.split("/")[1].replace(/-ONNX$/, "").replace(/-Instruct$/, "").toUpperCase();

async function write(claim) {
  const text = await readFile(LEDGER, "utf8");
  const quoted = claim.statement.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  const block = `[[claim]]\nid = "${claim.id}"\nlevel = "open"\nstatement = "${quoted}"\nsample_size = ${claim.sample_size}\n`;
  const blocks = text.split(/\n(?=\[\[claim\]\]\n)/);
  const at = blocks.findIndex((b) => b.startsWith(`[[claim]]\nid = "${claim.id}"\n`));
  if (at >= 0) blocks[at] = block.trimEnd();
  else blocks.push(block.trimEnd());
  await writeFile(LEDGER, blocks.join("\n") + "\n");
  console.log(`${at >= 0 ? "replaced" : "added"} ${claim.id}`);
}

async function equality(referencePath, floatPath, note) {
  const r = JSON.parse(await readFile(referencePath, "utf8"));
  const f = JSON.parse(await readFile(floatPath, "utf8"));
  const graph = r.modelFiles.find((p) => p.endsWith(".onnx"));
  const revision = r.revision.match(/resolve\/([0-9a-f]{7})/)?.[1] ?? r.revision;
  const each = r.runs.map((run) => {
    const a = f.prompts.find((p) => p.prompt === run.prompt);
    const end = run.endOfTurnAt === null ? "no end of turn in 256" : `end of turn at token ${run.endOfTurnAt}`;
    const misses = a.misses.length ? `; where they differ, the float model's best two logits are ${a.misses.map((m) => fixed(m.margin, 3)).join(", ")} apart` : "";
    return `"${run.prompt}" (${run.promptIds.length} prompt ids, ${end}, ids sha256 ${short(run.idsSha256)}): teacher-forced ${a.forced_equal} of ${a.tokens} equal (${a.forced_equal_to_end_of_turn} up to the end of turn), free runs ${a.free_parts_at === null ? "equal throughout" : `part at token ${a.free_parts_at}`}${misses}`;
  });
  const perToken = r.runs.map((x) => fixed(x.secondsPerToken, 2));
  await write({
    id: `OPEN-GREEDY-256-${keyOf(r.model)}-INTEGER`,
    sample_size: r.runs.length * r.tokens,
    statement: `Rows 2 and 3 of milestone 2, the CPU half: the page's integer CPU engine on ${r.model}@${revision} (${graph.split("/").pop()}), loaded through pipeline() with device cpu from the deployed page's own files (engine.wasm sha256 ${short(r.site.engineWasmSha256)}, holo ${short(r.site.holoSha256)}, pipeline.mjs ${short(r.site.pipelineSha256)}${r.site.loaderSha256 ? `, loader.mjs ${short(r.site.loaderSha256)}` : ""}${note ? `; ${note}` : ""}), ${r.runner}, one thread: 256 greedy tokens of each of three fixed prompts (crates/conformance/bench/equality-run.mjs), run past the end of turn, every id recorded in crates/conformance/bench/references/${r.key}.json as the reference the WebGPU integer engine is held to. Against the float model, the same file in ${f.float}: ${each.join("; ")}. ${perToken.join(", ")} s per token. The WebGPU half is row 15 (OPEN-ROW15-*), held to the same prompts by receipts. Measured ${new Date().toISOString().slice(0, 10)}.`,
  });
}

// The numbers one result gives for one device, or null where it did not run.
function numbers(result, device) {
  const run = result.runs.find((x) => x.device === device);
  if (!run || run.error) return { run: run ?? null, error: run?.error ?? "not run" };
  const download = Object.values(run.downloadBytes ?? {}).reduce((a, b) => a + b, 0);
  return { run, warm: run.short?.firstToken ?? null, tps: run.short?.tokensPerSecond ?? null, download };
}

function described(result, device, label) {
  const n = numbers(result, device);
  if (n.error) return `${label}: did not run (${n.error})`;
  const { run } = n;
  const parts = [`load ${fixed(run.loadSeconds, 1)} s ${run.load === "cold" ? `cold, ${mb(n.download)} MB downloaded` : "from this browser's copy"}`];
  if (run.cold) parts.push(`first token ${ms(run.cold.firstToken)} ms on the first turn`);
  if (run.short) parts.push(`warm first token ${ms(run.short.firstToken)} ms and ${fixed(run.short.tokensPerSecond, 1)} tokens per second (median of ${run.short.runs.length} after one warm-up, ${run.promptTokens?.short ?? "?"}-token prompt, up to 128 tokens)`);
  if (run.long) parts.push(`${run.promptTokens?.long ?? "?"}-token prompt: first token ${ms(run.long.firstToken)} ms, ${fixed(run.long.tokensPerSecond, 1)} tokens per second (median of ${run.long.runs.length})`);
  return `${label}: ${parts.join(", ")}`;
}

async function speed(machineId, hologramPath, transformersPath, bounds) {
  const h = JSON.parse(await readFile(hologramPath, "utf8"));
  const t = JSON.parse(await readFile(transformersPath, "utf8"));
  if (h.model !== t.model || h.machine !== t.machine) throw new Error("the two results are not the same model on the same machine");
  const webgpuInteger = h.deploy.webgpuEngine === "integer";
  const webgpuLabel = `this engine on WebGPU (${webgpuInteger ? "the integer engine" : "the float engine, which the page runs on WebGPU until row 15"})`;
  // Row 16 is judged on an integer engine: WebGPU when it is integer, else the CPU.
  const judged = webgpuInteger ? numbers(h, "webgpu") : numbers(h, "wasm");
  const measured = { warm: judged.warm ?? null, tps: judged.tps ?? null, guest: h.guest?.microseconds ?? null, holo: h.guest?.holoBytes ?? null };
  const verdicts = [];
  for (const [name, target] of Object.entries(TARGETS)) {
    if (target.model && target.model !== h.model) continue;
    if (target.met(measured)) {
      verdicts.push(`${target.text}: met, ${target.value(measured)}`);
      continue;
    }
    const bound = bounds[name];
    if (!bound) throw new Error(`target "${target.text}" is missed (${target.value(measured)}): give the step that bounds it with --bound ${name}=<step>`);
    verdicts.push(`${target.text}: missed, ${target.value(measured)}; bounded by ${bound}`);
  }
  const sample = [h, t].flatMap((r) => r.runs).reduce((n, run) => n + (run.short?.runs.length ?? 0) + (run.long?.runs.length ?? 0) + (run.cold ? 1 : 0), 0);
  await write({
    id: `OPEN-SPEED-${keyOf(h.model)}-${machineId}`,
    sample_size: sample,
    statement: [
      `Rows 11 and 16 of milestone 2 on ${h.model} q4f16, ${h.machine}, ${h.date.slice(0, 10)}, on the published page ${h.page} (holo sha256 ${short(h.deploy.holoSha256)}, engine.wasm ${short(h.deploy.engineWasmSha256)}), protocol m2-speed/1 (crates/conformance/bench/speed-run.mjs), one browser profile per library holding nothing at the start.`,
      `${described(h, "webgpu", webgpuLabel)}.`,
      `${described(h, "wasm", "this engine's integer CPU engine (device wasm)")}.`,
      `${described(t, "webgpu", `transformers.js ${t.library.split("@").pop()} on WebGPU, same page, same code`)}.`,
      `${described(t, "wasm", `transformers.js on wasm`)}.`,
      h.guest ? `Guest call overhead ${fixed(h.guest.microseconds, 1)} µs warm (median of ${h.guest.batches} batches of ${h.guest.calls} calls on a ${h.guest.requestBytes}-byte request, page not cross-origin isolated); holo ${h.guest.holoBytes} bytes.` : "",
      `Row 16, judged on ${webgpuInteger ? "the integer WebGPU engine" : "the integer CPU engine, since the page's WebGPU engine is not integer yet"}: ${verdicts.join("; ")}.`,
    ].filter(Boolean).join(" "),
  });
}

// Row 15: the integer engine on WebGPU against the integer engine on the CPU,
// on the published page, held by receipts (row15.mjs).
async function row15(path, note) {
  const r = JSON.parse(await readFile(path, "utf8"));
  const s = r.summary, g = r.webgpu, c = r.cpu;
  const legs = (x, name) => x.turns ? `${name}: ${x.turns.map((t) => `${t.name} ${t.tokens} tokens in ${fixed(t.seconds, 0)} s`).join(", ")}` : `${name}: did not run (${x.error})`;
  const verdict = s.error
    ? `Not passed: ${s.error}; the CPU leg ran, so nothing was compared across devices`
    : `Model label equal on both devices: ${s.modelLabelEqual}; every input label equal: ${s.inputLabelsEqual}; every output label equal: ${s.outputLabelsEqual}; the CPU's generate.check of WebGPU's three receipts: ${s.cpuChecksGpu.join(", ")}; WebGPU's of the CPU's three: ${s.gpuChecksCpu.join(", ")}`;
  await write({
    id: `OPEN-ROW15-${keyOf(r.model)}-Z13`,
    sample_size: (g.turns?.length ?? 0) + (c.turns?.length ?? 0),
    statement: `Row 15 of milestone 2 on ${r.model} q4f16, Z13, ${r.date.slice(0, 10)}, on the published page ${r.page} ${note ? `(${note}) ` : ""}on a cross-origin-isolated return visit (isolated: WebGPU ${s.isolated.webgpu}, CPU ${s.isolated.cpu}), protocol ${r.protocol} (crates/conformance/bench/row15.mjs): rows 2 and 3's three prompts, ${r.tokens} tokens each, on the integer engine on WebGPU and on the CPU, each device in a page of its own, receipt class ${s.classes?.webgpu ?? "n/a"} / ${s.classes?.cpu ?? "n/a"}; adapter ${s.adapter ? `${s.adapter.vendor} ${s.adapter.architecture}` : "none"}. ${legs(g, "WebGPU")}. ${legs(c, "CPU")}. ${verdict}. Model label ${g.identity?.label ? short(g.identity.label) : c.identity?.label ? short(c.identity.label) : "n/a"}.`,
  });
}

// Row 9: the real encoders in the page (encoders.mjs, encoders-browser.mjs,
// engine_row9_real_exports).
async function row9(dump, browserPath) {
  const b = JSON.parse(await readFile(browserPath, "utf8"));
  for (const key of ["minilm", "sst2"]) {
    const s = JSON.parse(await readFile(join(dump, key, "cases.json"), "utf8"));
    const m = b.models[key];
    const each = s.cases.map((c) => s.task === "feature-extraction"
      ? `${c.ids.length} ids: largest component gap ${c.float.largestGap.toExponential(1)}, cosine ${c.float.cosine.toFixed(8)}`
      : `${c.ids.length} ids: label ${s.labels?.[c.label] ?? c.label}, the float reference's ${c.float.sameLabel ? "the same" : "another"}, largest score gap ${c.float.largestScoreGap.toExponential(1)}`);
    await write({
      id: `OPEN-ROW9-${key.toUpperCase()}-PAGE`,
      sample_size: s.cases.length,
      statement: `Row 9 of milestone 2 on the real export ${s.repo} (onnx/model_q4f16.onnx), ${s.task}, ${b.date.slice(0, 10)}: five English sentences (crates/conformance/bench/encoders.mjs). The published page in Chrome (${b.page}, device ${m.device}) answered every id and binary32 bit equal to the deployed page's pipeline run in Node, and the weights that pipeline placed, run through the native build of the module, which is hologram-ai-cpu-encode (engine_row9_real_exports, cargo test --ignored), answered the same bits word for word: equal on all five (equal to Node and native: ${m.equalToNodeAndNative}). Against the double-precision float reference crates/conformance/tests/gpu/encoder-reference.mjs on the same weights decoded exactly: ${each.join("; ")}.`,
    });
  }
}

// Row 7: prefix reuse's gain on a real model (prefix-run.mjs through speed.mjs).
async function prefix(path) {
  const r = JSON.parse(await readFile(path, "utf8"));
  const byDevice = {};
  for (const run of r.runs) (byDevice[run.device] ??= []).push(run);
  const parts = Object.entries(byDevice).map(([device, runs]) => {
    const ok = runs.filter((x) => !x.error && x.samples?.length);
    if (!ok.length) return `${device}: did not run (${runs.find((x) => x.error)?.error ?? "no sample"})`;
    const samples = ok.flatMap((x) => x.samples);
    const whole = samples.map((x) => x.whole.firstToken), reused = samples.map((x) => x.reused.firstToken);
    const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
    const verdicts = [...new Set(samples.flatMap((x) => x.verdictReused))].join("; ");
    return `${device}: ${samples.length} fresh pages; time to first token, whole prompt then head reused, median ${fixed(med(whole), 3)} s then ${fixed(med(reused), 3)} s (each sample ${samples.map((x, i) => `${fixed(whole[i], 2)}/${fixed(reused[i], 2)}`).join(", ")}); the second turn's verdict ${verdicts || "none"}; the two turns' texts equal in ${samples.filter((x) => x.sameText).length} of ${samples.length}`;
  });
  const first = r.runs.find((x) => x.promptTokens);
  await write({
    id: `OPEN-PREFIX-${keyOf(r.model)}-Z13`,
    sample_size: r.runs.reduce((n, x) => n + (x.samples?.length ?? 0), 0),
    statement: `Row 7 of milestone 2 on ${r.model} q4f16, Z13, ${r.date.slice(0, 10)}, on the published page ${r.page}, protocol m2-prefix/1 (crates/conformance/bench/prefix-run.mjs): a fresh page and pipeline per sample, the engine's setup paid on a bare string shorter than the head, then one chat text twice, 32 tokens each; ${first ? `${first.promptTokens} prompt ids, the frame's head ${first.headTokens} of them` : "prompt not counted"}. ${parts.join(". ")}.`,
  });
}

// A row measured nowhere yet, written open with the reason.
async function open(id, statement) {
  await write({ id, sample_size: 0, statement });
}

const [mode, ...rest] = process.argv.slice(2);
if (mode === "equality") await equality(rest[0], rest[1], rest[2] === "--note" ? rest[3] : null);
else if (mode === "speed") {
  const bounds = {};
  rest.forEach((a, i) => { if (a === "--bound") { const [k, ...v] = rest[i + 1].split("="); bounds[k] = v.join("="); } });
  await speed(rest[0], rest[1], rest[2], bounds);
} else if (mode === "row15") await row15(rest[0], rest[1] === "--note" ? rest[2] : null);
else if (mode === "row9") await row9(rest[0], rest[1]);
else if (mode === "prefix") await prefix(rest[0]);
else if (mode === "open") await open(rest[0], rest[1]);
else throw new Error("usage: ledger.mjs equality <reference> <float> | speed <machine id> <hologram> <transformers> [--bound target=step] | row15 <row15.json> | row9 <encoders out dir> <browser.json> | prefix <prefix.json> | open <id> <statement>");
