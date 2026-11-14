// GP-02's script: print the engine's rotary frequencies for a model's
// parameters, so the Rust test can compare them with the formula it computes.
//
//   node crates/conformance/tests/gpu/yarn.mjs <root> <hd> <base> <factor> <orig>

import { pathToFileURL } from "node:url";
import { join } from "node:path";

const [root, hd, base, factor, orig] = process.argv.slice(2);
const { yarn } = await import(pathToFileURL(join(root, "site/app/engine/yarn.mjs")).href);
const r = yarn({ hd: Number(hd), base: Number(base), factor: Number(factor), orig: Number(orig) });
console.log(JSON.stringify({ low: r.low, high: r.high, mscale: r.mscale, freqs: Array.from(r.freqs) }));
