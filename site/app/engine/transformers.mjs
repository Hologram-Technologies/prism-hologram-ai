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
