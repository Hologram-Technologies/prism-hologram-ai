// The integer engine on the page: the module's forward pass (EN-52), answering
// with the interface the float engines answer: step, run, generate, reset,
// position, saveState, restoreState.
//
// Why: Hologram AI's inference uses integers only (Q_ENGINE_PLAN.md), so the
// page computes no number of its own. Every step is `hologram-ai-cpu-decode`
// in the module: the residual stream as binary32 bit patterns touched only by
// integer code, each stage rounded once, half to even. The page hands the
// module each weight's bytes once, opens the model with what its config says,
// and reads back the greedy token and the logits' bits.

const TYPES = { f32: 1, u8: 2, i8: 3, i32: 6, i64: 7, f16: 10, bf16: 16 };
const le = (n) => new Uint8Array(new Uint32Array([n]).buffer);
const word = (bytes, at = 0) => new DataView(bytes.buffer, bytes.byteOffset, bytes.length).getUint32(at, true);
// The config's epsilon as the binary32 the export states it in.
const binary32 = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];

// `module` is the page's module after the graph was imported through it;
// `table` and `read` are the import's tensors and their bytes; `config` the
// model's config.json; `rope` the rotary parameters the loader confirmed.
export function createIntegerEngine({ module: m, table, read, config, rope = {}, capacity }) {
  const text = new TextEncoder();
  for (const [name, [element]] of Object.entries(table.tensors)) {
    if (!(name in table.places) || !(element in TYPES)) continue;
    const named = text.encode(name);
    const bytes = read(name);
    const given = new Uint8Array(8 + named.length + bytes.length);
    given.set(le(named.length), 0);
    given.set(named, 4);
    given.set(le(TYPES[element]), 4 + named.length);
    given.set(bytes, 8 + named.length);
    m.callWith("decode_tensor", given);
  }
  const heads = config.num_attention_heads;
  const opened = {
    heads,
    kv_heads: config.num_key_value_heads ?? heads,
    rope_theta: Math.round(config.rope_theta),
    rms_norm_eps: binary32(config.rms_norm_eps),
    scaling: rope.llama3 ? { factor: rope.llama3.factor, low: rope.llama3.low, high: rope.llama3.high, original: rope.llama3.orig } : null,
  };
  const sizes = JSON.parse(new TextDecoder().decode(m.callWith("decode_open", text.encode(JSON.stringify(opened)))));
  let position = 0;

  // One token through the model: the greedy token, and the logits if asked,
  // read from their binary32 bits.
  function stepSync(token, wantLogits = false) {
    const said = m.call("decode_step", token);
    position += 1;
    const logits = wantLogits ? new Float32Array(said.slice(4).buffer) : null;
    return { token: word(said), logits };
  }
  async function step(token, { logits: wantLogits = false } = {}) {
    return stepSync(token, wantLogits);
  }
  // The prompt, then `count` greedy picks, stopping at any id in `stops`.
  async function run(prompt, count, { stops = new Set(), onToken = () => {}, onFirst = () => {} } = {}) {
    if (position + prompt.length + count > capacity) throw new Error(`context full: ${capacity} positions`);
    let last = null;
    for (const t of prompt) last = stepSync(t).token;
    onFirst();
    const out = [];
    while (out.length < count) {
      if (stops.has(last)) break;
      out.push(last);
      onToken(out.slice());
      if (out.length < count) last = stepSync(last).token;
    }
    return out;
  }
  async function generate(prompt, count) {
    return run(prompt, count);
  }
  // The run after its first `length` tokens, kept in the module.
  function saveState(length = position) {
    if (!(length >= 1 && length <= position)) throw new Error(`prefix: ${length} positions asked, ${position} computed`);
    return { length, id: word(m.call("decode_save", length)) };
  }
  function restoreState(state) {
    m.call("decode_restore", state.id);
    position = state.length;
  }

  return {
    step, run, generate, saveState, restoreState,
    reset: () => { m.call("decode_reset"); position = 0; },
    get position() { return position; },
    device: "cpu",
    integer: true,
    sizes,
  };
}
