// transformers.js's own surface over this engine, so code written for
// @huggingface/transformers runs here unchanged:
//
//   import { pipeline, TextStreamer } from "./engine/transformers.mjs";
//   const generator = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", { device: "webgpu", dtype: "q4f16" });
//   const output = await generator([{ role: "user", content: "Hi" }], { max_new_tokens: 64 });
//   output[0].generated_text.at(-1).content;
//
// Whether a call runs, and on which device, is the kernel's decision 70: the
// names transformers.js gives are sent as codes and the kernel answers the
// engine's device or refuses by name. Nothing here decides; an option this
// engine cannot honour is refused, never ignored.

import { asker, checklist } from "./loader.mjs";
import { pipeline as engine, verifiedKernel } from "./pipeline.mjs";

// transformers.js's task names and its aliases, in decision 70's code.
const TASK = { "text-generation": 0, "feature-extraction": 1, embeddings: 1, "text-classification": 2, "sentiment-analysis": 2 };
const TASK_NAME = ["text-generation", "feature-extraction", "text-classification"];
const DEVICE = { auto: 0, webgpu: 1, gpu: 2, wasm: 3, cpu: 4 };
const DTYPE = { auto: 0, q4f16: 1, q4: 2 };
const code = (table, name) => (name === undefined ? 0 : table[name] ?? 255);

// The call as decision 70's request: tag, task, device, dtype, sampling, beams, penalty.
function callRequest(task, options, generation = {}) {
  return new Uint8Array([
    70,
    TASK[task] ?? 255,
    code(DEVICE, options.device),
    typeof options.dtype === "object" ? 255 : code(DTYPE, options.dtype),
    generation.do_sample ? 1 : 0,
    Math.min(generation.num_beams ?? 1, 255),
    generation.repetition_penalty === undefined || generation.repetition_penalty === 1 ? 1 : 0,
  ]);
}

const RUNS = (answer) => answer === "auto" || answer === "webgpu" || answer === "cpu";

export class TextStreamer {
  constructor(tokenizer, { skip_prompt = false, skip_special_tokens = true, callback_function = null, decode_kwargs = {} } = {}) {
    Object.assign(this, { tokenizer, skip_prompt, skip_special_tokens, callback_function, decode_kwargs });
  }
  on_finalized_text(text, stream_end) {
    if (this.callback_function) this.callback_function(text);
    else if (globalThis.process?.stdout) process.stdout.write(text + (stream_end ? "\n" : ""));
    else if (text || stream_end) console.log(text);
  }
  put(text) { this.on_finalized_text(text, false); }
  end() { this.on_finalized_text("", true); }
}

// transformers.js's Tensor, as far as an encoder's answer needs it.
export class Tensor {
  constructor(type, data, dims) {
    Object.assign(this, { type, data, dims, size: data.length });
  }
  tolist() {
    const [rows, width] = this.dims.length === 2 ? this.dims : [1, this.size];
    return Array.from({ length: rows }, (_, r) => Array.from(this.data.subarray(r * width, (r + 1) * width)));
  }
}

// Decision 82's request: tag, task, pooling, normalize.
const POOLING = { none: 0, mean: 1, cls: 2 };
const encodeRequest = (task, { pooling, normalize } = {}) =>
  new Uint8Array([82, task, pooling === undefined ? 0 : POOLING[pooling] ?? 255, normalize ? 1 : 0]);

// feature-extraction: `extractor(texts, { pooling: "mean", normalize: true })`
// answers a Tensor of dims [texts, width], each row the module's embedding.
function extractor(run, ask) {
  return async (input, options = {}) => {
    await ask("call", "a transformers.js feature-extraction call (decision 82)", encodeRequest(1, options), "run");
    const texts = Array.isArray(input) ? input : [input];
    const rows = [];
    for (const text of texts) rows.push((await run(text)).data);
    const width = rows[0]?.length ?? 0;
    const data = new Float32Array(texts.length * width);
    rows.forEach((row, i) => data.set(row, i * width));
    return new Tensor("float32", data, [texts.length, width]);
  };
}

// text-classification: `classifier(text, { top_k })` answers [{ label, score }],
// the score the module's integer softmax over the labels (EN-95), highest
// first; an array of texts answers one list per text, or one entry each when
// top_k is 1, as transformers.js does.
function classifier(run, ask) {
  const labels = run.engine?.labels ?? [];
  return async (input, options = {}) => {
    const { top_k = 1, ...rest } = options;
    await ask("call", "a transformers.js text-classification call (decision 82)", encodeRequest(2, rest), "run");
    const one = async (text) => {
      const { bits } = await run(text);
      const scores = run.module.callWith("encode_probabilities", new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength));
      const probabilities = new Uint32Array(scores.buffer.slice(scores.byteOffset, scores.byteOffset + scores.byteLength));
      const view = new Float32Array(probabilities.buffer);
      // Non-negative binary32 values order as their bits do.
      const order = Array.from(probabilities.keys()).sort((a, b) => probabilities[b] - probabilities[a] || a - b);
      return order.slice(0, top_k ?? order.length).map((i) => ({ label: labels[i] ?? String(i), score: view[i] }));
    };
    if (!Array.isArray(input)) return one(input);
    const each = [];
    for (const text of input) each.push(await one(text));
    return top_k === 1 ? each.map((list) => list[0]) : each;
  };
}

export const env = Object.freeze({ backends: {}, allowRemoteModels: true, allowLocalModels: false });

export async function pipeline(task = "text-generation", model, options = {}) {
  const { progress_callback, ...rest } = options;
  const checks = rest.checks ?? checklist();
  const kernel = rest.kernel ?? (await verifiedKernel({ checks })).kernel;
  const ask = asker(kernel, checks);
  const device = await ask("call", "a transformers.js pipeline call (decision 70)", callRequest(task, options), RUNS);
  progress_callback?.({ status: "initiate", name: model, file: "" });
  const run = await engine(TASK_NAME[TASK[task]], model, {
    ...rest,
    kernel,
    checks,
    device,
    dtype: options.dtype ?? "auto",
    onProgress: (fraction) => progress_callback?.({ status: "progress", name: model, file: "", progress: 100 * fraction }),
  });
  progress_callback?.({ status: "ready", task, model });
  if (TASK[task] === 1) return Object.assign(extractor(run, ask), { task, model, dispose: async () => run.engine?.dispose?.() });
  if (TASK[task] === 2) return Object.assign(classifier(run, ask), { task, model, dispose: async () => run.engine?.dispose?.() });
  const tokenizer = {
    decode: (ids) => run.decode(ids),
    apply_chat_template: (messages, { add_generation_prompt = true } = {}) => run.render(messages, add_generation_prompt),
  };

  async function generator(input, generation = {}) {
    await ask("call", "a transformers.js generation call (decision 70)", callRequest(task, options, generation), RUNS);
    const { max_new_tokens = 20, streamer, return_full_text = true } = generation;
    const chat = Array.isArray(input) && typeof input[0] === "object";
    if (streamer && !streamer.skip_prompt) streamer.put(chat ? run.render(input, true) : input);
    // The engine reports the text so far; a streamer is given what is new.
    let shown = "";
    const onToken = (text) => {
      if (streamer && text.startsWith(shown) && text.length > shown.length) streamer.put(text.slice(shown.length));
      if (text.startsWith(shown)) shown = text;
    };
    const turn = await run(input, { max_new_tokens, template: chat, onToken });
    if (streamer && turn.text.length > shown.length) streamer.put(turn.text.slice(shown.length));
    streamer?.end();
    if (chat) return [{ generated_text: [...input, { role: "assistant", content: turn.text }] }];
    return [{ generated_text: return_full_text ? input + turn.text : turn.text }];
  }
  return Object.assign(generator, { task, model, tokenizer, dispose: async () => run.engine?.dispose?.() });
}
