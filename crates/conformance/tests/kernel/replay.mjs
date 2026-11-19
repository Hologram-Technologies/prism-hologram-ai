// Run the compiled guest of a holo on the claims the engine's crates make of
// it, and fail unless each gets the answer it lists.
//
//     node crates/conformance/tests/kernel/replay.mjs <app.holo> <claims.json>
//
// The holo's own examples are replayed by `prismpm --project app verify`, but
// they cannot grow past about 108 (KERNEL-VERIFICATION.md, The limit on
// examples). These claims are not examples: the Rust suites check that each
// request is the crate's own, and this runs each through the guest built from
// the proved definitions, so a crate that disagrees with the kernel fails the
// `app` recipe.

import { readFileSync } from "node:fs";
import { loadArchive } from "../../../../try/vendor/prism-code/src/index.mjs";

const [holoPath, claimsPath] = process.argv.slice(2);
if (!holoPath || !claimsPath) {
  console.error("usage: replay.mjs <app.holo> <claims.json>");
  process.exit(64);
}
const kernel = await loadArchive(new Uint8Array(readFileSync(holoPath)));
const { claims } = JSON.parse(readFileSync(claimsPath, "utf8"));
const text = new TextDecoder("utf-8", { fatal: true });
const wrong = [];
for (const { request, answer } of claims) {
  const bytes = Uint8Array.from(request.match(/../g) ?? [], (pair) => parseInt(pair, 16));
  // A trap names its request and the replay goes on, so one run shows them all.
  let got;
  try {
    got = text.decode(await kernel.dispatch(bytes));
  } catch (error) {
    got = `a trap (${error.message})`;
  }
  if (got !== answer) wrong.push(`${request}: the guest answers \`${got}\`, the claim lists \`${answer}\``);
}
if (claims.length === 0 || wrong.length > 0) {
  console.error(`kernel replay: ${wrong.length} of ${claims.length} claims answered otherwise\n${wrong.join("\n")}`);
  process.exit(1);
}
console.log(`kernel replay: the guest answers all ${claims.length} claims as listed`);
