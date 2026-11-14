// GP-03's journey: the engine page, served with the holo and the wasm module
// the build made, loads a weight container through the verified kernel, and
// refuses a tampered one where it should. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/journey.mjs <assembled site directory>
//
// The site directory is what Pages publishes: site/ with app.holo, vendor/ and
// app/engine/engine.wasm in place.

import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { buildFixture, hubFiles } from "./fixture.mjs";
import { referenceModel } from "./reference.mjs";

const site = process.argv[2];
const playwright = process.env.PRISM_PLAYWRIGHT ?? "/opt/prismpm/oracles/node_modules/playwright/index.mjs";
const { chromium } = await import(playwright);

const fixture = buildFixture();
// One byte inside a block the loader takes after many others: its label no
// longer matches, so the kernel must refuse that block and nothing after it.
const victim = fixture.dir.find((e) => e.name === "l1.wq");
const flipped = Buffer.from(fixture.container);
flipped[victim.off + Math.floor(victim.len / 2)] ^= 0x01;

const variants = {
  good: { container: fixture.container, sha256: fixture.sha256 },
  flip: { container: flipped, sha256: fixture.sha256 },
  root: { container: fixture.container, sha256: "f".repeat(64) },
};

const types = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm" };
const server = http.createServer(async (request, response) => {
  const path = decodeURIComponent(new URL(request.url, "http://x").pathname);
  const fixtureRoute = path.match(/^\/fixture\/(\w+)\/(.*)$/);
  if (fixtureRoute && variants[fixtureRoute[1]]) {
    const v = variants[fixtureRoute[1]];
    const hub = hubFiles(fixture, { resolve: `/fixture/${fixtureRoute[1]}/`, sha256: v.sha256 });
    const rest = fixtureRoute[2];
    if (rest === "index/root.json") return response.end(hub.root);
    if (rest === `index/models/${fixture.repo}.json`) return response.end(hub.record);
    if (rest === fixture.path) {
      const range = /bytes=(\d+)-(\d+)?/.exec(request.headers.range ?? "");
      if (!range) return response.end(v.container);
      const start = Number(range[1]);
      const end = range[2] === undefined ? v.container.length - 1 : Number(range[2]);
      response.statusCode = 206;
      response.setHeader("content-range", `bytes ${start}-${end}/${v.container.length}`);
      return response.end(v.container.subarray(start, end + 1));
    }
    response.statusCode = 404;
    return response.end();
  }
  const file = path.endsWith("/") ? `${path}index.html` : path;
  try {
    const body = await readFile(join(site, normalize(file)));
    response.setHeader("content-type", types[extname(file)] ?? "application/octet-stream");
    response.end(body);
  } catch {
    response.statusCode = 404;
    response.end();
  }
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"] });
const results = {};
try {
  for (const name of Object.keys(variants)) {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message)));
    await page.goto(`${origin}/app/?repo=${fixture.repo}&path=${fixture.path}&index=/fixture/${name}/index`);
    await page.click("#load");
    page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
    try {
      await page.waitForFunction(() => ["ready", "failed"].includes(window.__qengine?.phase), null, { timeout: 180000 });
    } catch {
      // Say where it stuck, not only that it did.
      const stuck = await page.evaluate(() => ({ phase: window.__qengine?.phase, status: document.getElementById("status")?.textContent, checks: window.__qengine?.checks ?? [] }));
      results[name] = { ...stuck, timeout: true, errors, victim: victim.name };
      await page.close();
      continue;
    }
    let output = null;
    if (name === "good") {
      await page.fill("#prompt", "hi");
      await page.click("#send");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      output = await page.evaluate(() => window.__qengine.output);
    }
    const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, checks: window.__qengine.checks }));
    results[name] = { ...state, output, errors, victim: victim.name };
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}

// The answer the page gave, against the reference run over the same prompt ids.
const good = results.good;
if (good.output) {
  const reference = referenceModel(fixture.model);
  let last;
  for (const id of good.output.promptIds) last = reference.step(id);
  const expected = [];
  while (expected.length < 16) {
    expected.push(last.token);
    last = reference.step(last.token);
  }
  good.expected = expected;
}
console.log(JSON.stringify(results));
