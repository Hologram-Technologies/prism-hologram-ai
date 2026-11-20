// Row 7 of milestone 2: what prefix reuse (decision 27) saves on a real model,
// with the tokens unchanged. Run in the browser by speed.mjs --protocol prefix-run.mjs,
// this engine only (transformers.js keeps no head between calls).
//
// Why each number is taken as it is:
//   a fresh page and pipeline per sample, so the first chat turn holds no head and
//   computes the whole prompt; before it, one turn on a bare string shorter
//   than the frame's head, which pays the engine's one-time setup without
//   keeping a head (the page keeps one only when the prompt is longer);
//   then the same chat text twice: the first computes every id and keeps the
//   head, the second is answered `reuse` by the kernel and computes the rest;
//   time to first token of each, from the call to the first token handed to
//   the streamer; the two turns' texts compared, and the kernel's verdicts
//   read from the page's own checklist.

export const PROTOCOL = {
  name: "m2-prefix/1",
  prompt: "Write a short paragraph about why the sky is blue.",
  samples: 1, // per page; speed.mjs reloads the page between samples (--devices wasm,wasm,wasm)
  max_new_tokens: 32,
};

const median = (xs) => {
  const s = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function turn(generator, input, max_new_tokens) {
  let first = null;
  const streamer = { skip_prompt: true, put() { first ??= performance.now(); }, end() {} };
  const t0 = performance.now();
  const output = await generator(input, { max_new_tokens, do_sample: false, streamer });
  const end = performance.now();
  const reply = output?.[0]?.generated_text;
  return {
    firstToken: first === null ? null : (first - t0) / 1000,
    seconds: (end - t0) / 1000,
    text: Array.isArray(reply) ? reply.at(-1)?.content : reply,
  };
}

export async function measure({ library, counter, model, device, dtype, onStatus = () => {} }) {
  const lib = await library();
  const { checklist } = await import(new URL("engine/loader.mjs", location.href).href);
  const chat = [{ role: "user", content: PROTOCOL.prompt }];
  const samples = [];
  for (let i = 0; i < PROTOCOL.samples; i++) {
    const checks = checklist();
    const t0 = performance.now();
    const generator = await lib.pipeline("text-generation", model, { device, dtype, checks });
    const loadSeconds = (performance.now() - t0) / 1000;
    await turn(generator, "hi", 2);
    const verdicts = () => checks.items.filter((c) => c.step === "reuse").map((c) => c.detail);
    const whole = await turn(generator, chat, PROTOCOL.max_new_tokens);
    const afterWhole = verdicts().length;
    const reused = await turn(generator, chat, PROTOCOL.max_new_tokens);
    const all = verdicts();
    await generator.dispose?.();
    samples.push({ loadSeconds, whole, reused, verdictWhole: all.slice(0, afterWhole), verdictReused: all.slice(afterWhole), sameText: whole.text === reused.text });
    onStatus(`sample ${i + 1}: first token ${whole.firstToken?.toFixed(3)} s whole, ${reused.firstToken?.toFixed(3)} s reused, same text ${whole.text === reused.text}, ${all.join("; ")}`);
  }
  // Ids counted after every timed turn by transformers.js's tokenizer for the
  // model: the prompt framed by its chat template, and the head every framed
  // prompt opens with, as the page finds it (the common opening of two framings).
  const t = await (await counter()).AutoTokenizer.from_pretrained(model);
  const framed = (content) => t.encode(t.apply_chat_template([{ role: "user", content }], { tokenize: false, add_generation_prompt: true, enable_thinking: false }), { add_special_tokens: false });
  const [a, b, p] = [framed("a"), framed("b"), framed(PROTOCOL.prompt)];
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  return {
    protocol: PROTOCOL.name, model, device, dtype, promptTokens: p.length, headTokens: head, samples,
    wholeFirstToken: median(samples.map((s) => s.whole.firstToken)),
    reusedFirstToken: median(samples.map((s) => s.reused.firstToken)),
    allSameText: samples.every((s) => s.sameText),
  };
}
