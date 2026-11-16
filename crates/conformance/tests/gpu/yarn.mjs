// GP-02's script: print the engine's rotary frequencies for a model's
// parameters, so the Rust test can compare them with the formula it computes.
//
//   node crates/conformance/tests/gpu/yarn.mjs <root> <hd> <base> <factor> <orig>

import { pathToFileURL } from "node:url";
import { join } from "node:path";

// GP-11 runs it with "llama3" first: <root> llama3 <hd> <base> <factor> <low> <high> <orig>.
const [root, ...rest] = process.argv.slice(2);
const { llama3, yarn } = await import(pathToFileURL(join(root, "site/app/engine/yarn.mjs")).href);
let r;
if (rest[0] === "llama3") {
  const [hd, base, factor, low, high, orig] = rest.slice(1).map(Number);
  r = llama3({ hd, base, factor, low, high, orig });
} else {
  const [hd, base, factor, orig] = rest.map(Number);
  r = yarn({ hd, base, factor, orig });
}
console.log(JSON.stringify({ low: r.low, high: r.high, mscale: r.mscale, freqs: Array.from(r.freqs) }));
