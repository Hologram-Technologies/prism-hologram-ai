// GP-20's check: each export's chat template, rendered by site/app/engine/jinja.mjs,
// against what Python's jinja2 rendered for the same conversations, configured
// as transformers configures it (recorded in templates/, with the template
// itself, at the hub's revisions). And templates outside what the engine
// follows, refused by name. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/templates.mjs <repository root>

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.argv[2];
const { compileTemplate, strftime } = await import(pathToFileURL(join(root, "site/app/engine/jinja.mjs")).href);
const dir = join(root, "crates/conformance/tests/gpu/templates");
const results = {};
for (const file of readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
  const fixture = JSON.parse(readFileSync(join(dir, file), "utf8"));
  const render = compileTemplate(fixture.template);
  const [d, b, y] = fixture.date.split(" ");
  const date = new Date(Number(y), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].indexOf(b), Number(d));
  const differing = [];
  for (const c of fixture.cases) {
    let got;
    try {
      got = { text: render({ ...c.context, strftime_now: (format) => strftime(date, format) }) };
    } catch (e) {
      got = { error: e.message };
    }
    const want = c.error !== undefined ? { error: c.error } : { text: c.text };
    if (JSON.stringify(got) !== JSON.stringify(want)) differing.push({ name: c.name, got, want });
  }
  results[fixture.model] = { cases: fixture.cases.length, differing };
}
const refusal = (source) => {
  try {
    compileTemplate(source)({ messages: [] });
    return null;
  } catch (e) {
    return e.message;
  }
};
const refusals = {
  macro: refusal("{% macro m() %}x{% endmacro %}"),
  include: refusal("{% include 'other' %}"),
  filter: refusal("{{ 'a' | upper }}"),
  method: refusal("{{ 'a'.title() }}"),
  test: refusal("{% if 1 is odd %}x{% endif %}"),
  unclosed: refusal("{% if true %}x"),
  raise: refusal("{{ raise_exception('no tools here') }}"),
};
console.log(JSON.stringify({ results, refusals }));
