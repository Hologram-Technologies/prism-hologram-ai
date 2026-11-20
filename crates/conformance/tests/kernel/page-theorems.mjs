// The page names every theorem the proofs carry. Run by the `page-journey`
// recipe on the page it has just assembled:
//
//     node crates/conformance/tests/kernel/page-theorems.mjs <assembled site>
//
// Why here and not against a list in site/app.json: the list there was a
// second copy every decision edited, and the page never read it while the
// LexLean source was there to read. This checks the page itself: every
// `theorem{}` in app/src/Proofs is named in the assembled spec.html's proof
// section, and the section counts no fewer theorems than the proofs hold.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const site = process.argv[2];
if (!site) {
  console.error("usage: page-theorems.mjs <assembled site>");
  process.exit(64);
}
const root = fileURLToPath(new URL("../../../../", import.meta.url));
const proofs = join(root, "app/src/Proofs");
const proved = [];
for (const file of readdirSync(proofs).filter((f) => f.endsWith(".lex.tex")).sort()) {
  const text = readFileSync(join(proofs, file), "utf8");
  for (const block of text.split("theorem{").slice(1)) {
    const name = /^\s*name:\s*([A-Za-z0-9_']+)/m.exec(block)?.[1];
    if (name) proved.push(name);
  }
}
const spec = readFileSync(join(site, "spec.html"), "utf8");
const shown = [...spec.matchAll(/<div class="thm-name"><span>([^<]*)<\/span>/g)].map((m) => m[1]);
const missing = proved.filter((name) => !shown.includes(name));
if (proved.length < 49 || missing.length > 0 || shown.length < proved.length) {
  console.error(
    `page theorems: the proofs carry ${proved.length} theorems, the page shows ${shown.length}; ` +
      `not on the page: ${missing.join(", ") || "none"}`,
  );
  process.exit(1);
}
console.log(`page theorems: the page names all ${proved.length} theorems the proofs carry`);
