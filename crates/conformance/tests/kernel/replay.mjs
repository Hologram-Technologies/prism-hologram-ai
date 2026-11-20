// Run the compiled guest of a holo on the claims the engine's crates make of
// it, and fail unless each gets the answer it lists.
//
//     node crates/conformance/tests/kernel/replay.mjs <app.holo> <claims/>
//
// The holo's own examples are replayed by `prismpm --project app verify`, but
// they cannot grow past about 108 (KERNEL-VERIFICATION.md, The limit on
// examples). These claims are not examples: the Rust suites check that each
// request is the crate's own, and this runs each through the guest built from
// the proved definitions, so a crate that disagrees with the kernel fails the
// `app` recipe.
//
// The claims are one file per decision, `claims/NNN.json` (claims/README.md).
// The filing is checked here as the Rust suites check it: a file is NNN.json,
// names decision NNN, holds a claim, and each request opens with that byte.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { loadArchive } from "../../../../try/vendor/prism-code/src/index.mjs";

const [holoPath, claimsDir] = process.argv.slice(2);
if (!holoPath || !claimsDir) {
  console.error("usage: replay.mjs <app.holo> <claims/>");
  process.exit(64);
}
const claims = [];
const misfiled = [];
const files = readdirSync(claimsDir).filter((name) => name.endsWith(".json")).sort();
for (const name of files) {
  const tag = /^\d{3}\.json$/.test(name) ? Number(name.slice(0, 3)) : NaN;
  const filed = JSON.parse(readFileSync(join(claimsDir, name), "utf8"));
  const byte = Number.isInteger(tag) && tag < 256 ? tag.toString(16).padStart(2, "0") : null;
  if (byte === null || filed.decision !== tag || !Array.isArray(filed.claims) || filed.claims.length === 0) {
    misfiled.push(`${name}: a claims file is NNN.json, names decision NNN and holds a claim`);
    continue;
  }
  for (const claim of filed.claims) {
    if (typeof claim.request !== "string" || !claim.request.startsWith(byte)) misfiled.push(`${name}: ${claim.request} is not a decision-${tag} request`);
    else claims.push(claim);
  }
}
if (misfiled.length > 0) {
  console.error(`kernel replay: misfiled claims\n${misfiled.join("\n")}`);
  process.exit(1);
}
const kernel = await loadArchive(new Uint8Array(readFileSync(holoPath)));
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
console.log(`kernel replay: the guest answers all ${claims.length} claims of ${files.length} decisions as listed`);
