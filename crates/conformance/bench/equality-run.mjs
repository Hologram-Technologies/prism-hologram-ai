// Rows 2 and 3 of milestone 2 (docs/Q_ENGINE_PLAN.md, "Done when"): the
// integer engine's token ids over 256 tokens of three fixed prompts, the same
// on the CPU and on WebGPU. This file runs in Node (equality.mjs) and in a
// browser (equality-browser.mjs) alike, so both devices are driven by the one
// piece of code and differ only in the device named.
//
// Why 256 tokens with no stop: an engine that differs in one bit anywhere
// usually differs in a token within a few hundred steps, and a reply that
// ended at the model's end of turn would test only as far as that. The run
// goes on past it; the end of turn is recorded so agreement can be read both
// up to it and over all 256.

export const TOKENS = 256;

// Fixed prompts, each a user turn in the model's own chat template. Chosen to
// ask for long answers so most of the 256 tokens are the reply itself.
export const PROMPTS = [
  { name: "law", messages: [{ role: "user", content: "Explain, step by step and in detail, how a bill becomes a law in the United States." }] },
  { name: "lighthouse", messages: [{ role: "user", content: "Write a long story about a lighthouse keeper who finds a message in a bottle." }] },
  { name: "planets", messages: [{ role: "user", content: "Name the planets of the solar system in order and describe each one in a short paragraph." }] },
];

// The three models of rows 2 and 3, at the precision the page reads first.
export const MODELS = {
  "qwen2.5-1.5b": "onnx-community/Qwen2.5-1.5B-Instruct",
  "llama-3.2-1b": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
  "qwen3-0.6b": "onnx-community/Qwen3-0.6B-ONNX",
};

const hex = (bytes) => Array.from(new Uint8Array(bytes), (b) => b.toString(16).padStart(2, "0")).join("");
const idsLE = (ids) => new Uint8Array(new Uint32Array(ids).buffer);
export const idsSha256 = async (ids) => hex(await crypto.subtle.digest("SHA-256", idsLE(ids)));

// Load `model` through the page's one call on `device` and run every prompt
// for `tokens` greedy steps from a reset engine. `pipeline` is the page's
// pipeline.mjs export, imported by the caller from wherever the page is.
export async function referenceRun({ pipeline, checklist, model, device, dtype = "q4f16", tokens = TOKENS, prompts = PROMPTS, onStatus = () => {} }) {
  const checks = checklist();
  const t0 = performance.now();
  const generate = await pipeline("text-generation", model, { device, dtype, checks, onStatus });
  const loadSeconds = (performance.now() - t0) / 1000;
  const { engine, module: m, record } = generate;
  const encode = (text) => {
    const bytes = m.callWith("tokeniser_encode", new TextEncoder().encode(text));
    return Array.from(new Uint32Array(bytes.slice().buffer));
  };
  // The ids the model's own files name as stops; recorded, not obeyed.
  const stopsAt = (ids, stops) => {
    const at = ids.findIndex((id) => stops.has(id));
    return at < 0 ? null : at;
  };
  const stated = checks.items.find((c) => c.step === "stop" && c.verdict === "pass");
  const stops = new Set(stated ? stated.detail.split(", ").map(Number) : []);
  const failed = checks.items.filter((c) => c.verdict === "fail");
  if (failed.length) throw new Error(`the load failed a check: ${failed.map((c) => `${c.step}: ${c.detail}`).join("; ")}`);
  const runs = [];
  for (const p of prompts) {
    const promptIds = encode(generate.render(p.messages, true));
    await engine.reset();
    const started = performance.now();
    let firstAt = started;
    const ids = await engine.run(promptIds, tokens, { stops: new Set(), onFirst: () => { firstAt = performance.now(); } });
    const end = performance.now();
    runs.push({
      prompt: p.name,
      promptIds,
      promptIdsSha256: await idsSha256(promptIds),
      ids,
      idsSha256: await idsSha256(ids),
      text: generate.decode(ids),
      endOfTurnAt: stopsAt(ids, stops),
      prefillSeconds: (firstAt - started) / 1000,
      secondsPerToken: ids.length > 1 ? (end - firstAt) / 1000 / (ids.length - 1) : null,
    });
    onStatus(`${p.name}: ${ids.length} ids, ${runs.at(-1).idsSha256.slice(0, 16)}`);
  }
  return {
    model,
    revision: record.origin?.resolve ?? null,
    dtype: generate.dtype,
    device: generate.device,
    integer: Boolean(engine.integer),
    stops: [...stops],
    // Every verdict of the load, so a reference names the bytes it ran on.
    checks: checks.items.map((c) => `${c.step} ${c.verdict}: ${c.subject}${c.detail ? `: ${c.detail.slice(0, 400)}` : ""}`),
    loadSeconds,
    tokens,
    runs,
  };
}

// Token-by-token comparison of two runs of the same prompts: the first index
// where they part, or null where all are equal.
export function compareRuns(reference, candidate) {
  return reference.runs.map((r) => {
    const c = candidate.runs.find((x) => x.prompt === r.prompt);
    if (!c) return { prompt: r.prompt, equal: false, reason: "not run" };
    if (c.promptIdsSha256 !== r.promptIdsSha256) return { prompt: r.prompt, equal: false, reason: "prompt ids differ" };
    const n = Math.max(r.ids.length, c.ids.length);
    let firstDifference = null;
    for (let i = 0; i < n; i++) if (r.ids[i] !== c.ids[i]) { firstDifference = i; break; }
    return { prompt: r.prompt, equal: firstDifference === null, firstDifference, compared: n };
  });
}
