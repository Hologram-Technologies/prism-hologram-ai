// Rows 11 and 16 of milestone 2: one fixed protocol, run in the browser on
// the published page, by the same code for this engine and for transformers.js.
// The only difference between the two runs is where `pipeline` is imported
// from; the device and dtype names are transformers.js's ("webgpu", "wasm",
// "q4f16"), which this engine takes through decision 70 ("wasm" is its
// integer CPU engine).
//
// Why each number is taken as it is:
//   load: from the call to pipeline() to its answer, in a browser profile
//     holding nothing, so the model is downloaded and checked in full;
//   download: the bytes of every response body the page's fetch receives
//     during the load, by host;
//   cold first token: the first turn after the load, from the call to the
//     first token handed to the streamer;
//   warm first token and tokens per second: after one warm-up turn, the median
//     of the timed turns; tokens per second is (n - 1) over the time from the
//     first token to the turn's end, so it leaves the first token's latency
//     out;
//   long: the same over a prompt of about 440 tokens, every turn timed.
// The first token is the first put to a streamer handed to both libraries.
// transformers.js hands a streamer the prompt's ids first and then each
// token's; this engine hands it new text only, and on WebGPU once per
// submission of many positions (the first token alone). So the first put that
// is not text is the prompt and is not timed, and n, the tokens a turn made,
// is counted after the timed turns by one tokenizer for both libraries:
// transformers.js's, read from the model's own files, encoding each reply.

export const PROTOCOL = {
  name: "m2-speed/1",
  short: { prompt: "Write a short paragraph about why the sky is blue.", warmup: 1, runs: 5, max_new_tokens: 128 },
  long: { runs: 3, max_new_tokens: 128 },
  guest: { batches: 20, calls: 200 },
};

// About 440 tokens in the three models' tokenisers once framed by their chat
// templates, with a closing instruction.
const PASSAGE = [
  "A content address names a piece of data by a hash of its bytes rather than by where it is kept.",
  "Two copies of the same file, fetched from two servers, have the same address, and a copy that differs in one byte has another.",
  "This changes what a client has to trust. A client that holds an address does not need to trust the server that answers: it hashes what arrives and compares.",
  "If the hash matches, the bytes are the ones that were named, wherever they came from; if it does not, they are refused, and another source can be asked.",
  "Caches become simpler for the same reason. A browser that has kept a block under its address can answer any later request for that address from its own storage, and it never has to ask whether the block has gone stale, because a changed block would have a different name.",
  "Large models are a good fit for this. A model is mostly a few hundred tensors, each a long run of numbers that does not change between releases unless the weights themselves were retrained.",
  "When two exports of the same model share a tensor byte for byte, a client that already holds it need not download it again, and the saving can be counted exactly before any byte moves.",
  "The cost is that every read must be hashed, which takes time, and that the index mapping names to addresses must itself be trusted, which is usually done by signing it.",
  "A signed index, a hash per block, and a rule that nothing is used until its hash has been checked together let a page load a model from any mirror while running only the bytes its authors named.",
  "Whether this is worth its cost depends on how often the same bytes are fetched twice, how fast the client can hash, and how much the people using the page care that what runs is exactly what was published.",
  "For a model that is downloaded once and used for months, the hashing is paid once; for a page that is visited by many people on slow links, the savings from shared blocks and offline reuse add up quickly.",
  "There are also failure modes to plan for: an index that names a block no mirror still holds, a mirror that answers slowly, and a client whose storage is full and must decide what to drop.",
].join(" ");
export const LONG_PROMPT = `${PASSAGE}\n\nSummarise the passage above in three sentences, then give one example of where content addressing would not help.`;

const median = (xs) => {
  const s = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// Bytes of every response body the page's fetch hands back, by host. Module
// scripts the browser imports itself do not pass through fetch and are not in it.
function countDownloads() {
  const real = globalThis.fetch;
  const bytes = {};
  globalThis.fetch = async (input, init) => {
    const response = await real(input, init);
    if (!response.body) return response;
    const host = new URL(response.url || String(input?.url ?? input), location.href).host;
    const counted = response.body.pipeThrough(new TransformStream({
      transform(chunk, out) { bytes[host] = (bytes[host] ?? 0) + chunk.byteLength; out.enqueue(chunk); },
    }));
    const copy = new Response(counted, { status: response.status, statusText: response.statusText, headers: response.headers });
    Object.defineProperty(copy, "url", { value: response.url });
    Object.defineProperty(copy, "redirected", { value: response.redirected });
    return copy;
  };
  return { bytes, stop: () => { globalThis.fetch = real; } };
}

async function turn(generator, content, max_new_tokens) {
  let first = null;
  let prompt = false;
  const streamer = {
    skip_prompt: true,
    put(value) {
      if (!prompt && typeof value !== "string") { prompt = true; return; }
      prompt = true;
      first ??= performance.now();
    },
    end() {},
  };
  const t0 = performance.now();
  // Thinking off, as this engine always runs Qwen3; transformers.js passes it
  // to the chat template, which this engine already renders with it off.
  const output = await generator([{ role: "user", content }], { max_new_tokens, do_sample: false, streamer, tokenizer_encode_kwargs: { enable_thinking: false } });
  const end = performance.now();
  const reply = output?.[0]?.generated_text;
  return {
    firstToken: first === null ? null : (first - t0) / 1000,
    decodeSeconds: first === null ? null : (end - first) / 1000,
    seconds: (end - t0) / 1000,
    text: Array.isArray(reply) ? reply.at(-1)?.content : reply,
  };
}

// n for each turn, and tokens per second from it.
function counted(run, encode) {
  const tokens = encode(run.text ?? "");
  return { ...run, tokens, tokensPerSecond: tokens > 1 && run.decodeSeconds ? (tokens - 1) / run.decodeSeconds : null };
}

// The whole protocol for one library, model, device and dtype. `library`
// imports the module that exports pipeline(), so its own files are counted
// in the download; `counter` imports transformers.js, whose tokenizer counts
// every reply once the timed turns are over; `legs` names which of short and
// long run.
export async function measure({ library, counter, model, device, dtype, legs = ["short", "long"], onStatus = () => {} }) {
  const downloads = countDownloads();
  const t0 = performance.now();
  const lib = await library();
  const generator = await lib.pipeline("text-generation", model, { device, dtype });
  const loadSeconds = (performance.now() - t0) / 1000;
  downloads.stop();
  const result = { protocol: PROTOCOL.name, model, device, dtype, loadSeconds, downloadBytes: { ...downloads.bytes } };
  onStatus(`loaded in ${loadSeconds.toFixed(1)} s`);
  const { short, long } = PROTOCOL;
  if (legs.includes("short")) {
    result.cold = await turn(generator, short.prompt, short.max_new_tokens);
    onStatus(`cold first token ${result.cold.firstToken?.toFixed(3)} s`);
    for (let i = 0; i < short.warmup; i++) await turn(generator, short.prompt, short.max_new_tokens);
    const runs = [];
    for (let i = 0; i < short.runs; i++) {
      runs.push(await turn(generator, short.prompt, short.max_new_tokens));
      onStatus(`short ${i + 1}: first token ${runs.at(-1).firstToken?.toFixed(3)} s, ${runs.at(-1).seconds.toFixed(2)} s`);
    }
    result.short = { runs };
  }
  if (legs.includes("long")) {
    const runs = [];
    for (let i = 0; i < long.runs; i++) {
      runs.push(await turn(generator, LONG_PROMPT, long.max_new_tokens));
      onStatus(`long ${i + 1}: first token ${runs.at(-1).firstToken?.toFixed(3)} s, ${runs.at(-1).seconds.toFixed(2)} s`);
    }
    result.long = { runs };
  }
  await generator.dispose?.();
  // Counted after every timed turn, so the tokenizer's download and work are
  // in no number above.
  const t = await (await counter()).AutoTokenizer.from_pretrained(model);
  const encode = (text) => t.encode(text, { add_special_tokens: false }).length;
  const framed = (content) => t.encode(t.apply_chat_template([{ role: "user", content }], { tokenize: false, add_generation_prompt: true, enable_thinking: false }), { add_special_tokens: false }).length;
  result.promptTokens = { short: framed(short.prompt), long: framed(LONG_PROMPT) };
  if (result.cold) result.cold = counted(result.cold, encode);
  for (const leg of ["short", "long"]) {
    if (!result[leg]) continue;
    const runs = result[leg].runs.map((r) => counted(r, encode));
    result[leg] = { runs, firstToken: median(runs.map((r) => r.firstToken)), tokensPerSecond: median(runs.map((r) => r.tokensPerSecond)) };
  }
  return result;
}

// One call into the page's kernel guest, warm: the smallest non-empty
// acceptance example's request, answered `calls` times in each of `batches` timed
// batches; the median batch over its calls. Timed in batches because a page
// that is not cross-origin isolated reads the clock to 100 microseconds.
export async function guestOverhead(origin) {
  const { loadArchive, compileGuest, resolveArchive, runGuest } = await import(new URL("vendor/prism-code/src/index.mjs", origin).href);
  const holo = new Uint8Array(await (await fetch(new URL("app.holo", origin))).arrayBuffer());
  const kernel = await loadArchive(holo);
  const guest = await compileGuest(resolveArchive(kernel.archive).guest.content);
  const vector = kernel.vectors.filter((v) => v.request.length > 0).reduce((a, b) => (b.request.length < a.request.length ? b : a));
  for (let i = 0; i < 500; i++) await runGuest(guest, vector.request, kernel.limits);
  const { batches, calls } = PROTOCOL.guest;
  const each = [];
  for (let b = 0; b < batches; b++) {
    const t = performance.now();
    for (let i = 0; i < calls; i++) await runGuest(guest, vector.request, kernel.limits);
    each.push(((performance.now() - t) * 1000) / calls);
  }
  return { holoBytes: holo.length, requestBytes: vector.request.length, microseconds: median(each), batches, calls };
}
