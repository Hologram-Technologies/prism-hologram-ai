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
// No browser is a failure, never a skip: a journey that did not run proves nothing.
let chromium;
try {
  ({ chromium } = await import(playwright));
} catch (error) {
  console.error(`journey: no Playwright at ${playwright} (set PRISM_PLAYWRIGHT): ${error.message}`);
  process.exit(1);
}

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
  // The deploy lists a kernel other than the one the browser holds, as when the
  // service worker serves app.holo from an earlier deploy's cache.
  stale: { container: fixture.container, sha256: fixture.sha256, staleKernel: true },
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
    if (variants[name].staleKernel) {
      await page.route("**/precache.json", async (route) => {
        const listing = await (await route.fetch()).json();
        for (const f of listing.files) if (f.path === "./app.holo") f.sha256 = "0".repeat(64);
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(listing) });
      });
    }
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
    let promptStart = null;
    if (name === "good") {
      promptStart = await page.inputValue("#prompt");
      await page.fill("#prompt", "hi");
      await page.press("#prompt", "Enter");
      await page.waitForFunction(() => window.__qengine.stats || window.__qengine.error, null, { timeout: 180000 }).catch(() => {});
      output = await page.evaluate(() => window.__qengine.output);
    }
    const state = await page.evaluate(() => ({ phase: window.__qengine.phase, error: window.__qengine.error, checks: window.__qengine.checks }));
    results[name] = { ...state, output, promptStart, errors, victim: victim.name };
    await page.close();
  }

  // A cache left by an earlier deploy (GP-07): one file that is not this
  // deploy's, and one that is. The page must remove the first, keep the
  // second, reload once, and then import the engine.
  const earlier = async (context) => {
    const seed = await context.newPage();
    await seed.goto(`${origin}/precache.json`);
    await seed.evaluate(async (origin) => {
      const cache = await caches.open("prism-earlier");
      await cache.put(`${origin}/app/engine/gpu.mjs`, new Response("// an earlier deploy's engine"));
      await cache.put(`${origin}/app/app.css`, await fetch(`${origin}/app/app.css`));
    }, origin);
    await seed.close();
  };
  const held = (page) => page.evaluate(async (origin) => {
    const cache = await caches.open("prism-earlier");
    return { engine: Boolean(await cache.match(`${origin}/app/engine/gpu.mjs`)), css: Boolean(await cache.match(`${origin}/app/app.css`)) };
  }, origin);
  {
    const context = await browser.newContext();
    await earlier(context);
    const page = await context.newPage();
    let loads = 0;
    page.on("framenavigated", (frame) => { if (frame === page.mainFrame()) loads += 1; });
    await page.goto(`${origin}/app/`);
    const imported = await page.waitForFunction(() => window.__qengine && window.__qpage, null, { timeout: 30000 }).then(() => true, () => false);
    results.earlier = { imported, loads, page: await page.evaluate(() => window.__qpage ?? null), status: await page.textContent("#status"), held: await held(page) };
    await context.close();
  }
  // The same cache when one reload has already been spent: the page must stop
  // and not import the engine, or it would reload for ever.
  {
    const context = await browser.newContext();
    await earlier(context);
    const page = await context.newPage();
    await page.addInitScript(() => sessionStorage.setItem("prism-stale-reload", "1"));
    let loads = 0;
    page.on("framenavigated", (frame) => { if (frame === page.mainFrame()) loads += 1; });
    await page.goto(`${origin}/app/`);
    await page.waitForFunction(() => window.__qpage, null, { timeout: 30000 }).catch(() => {});
    // Long enough for an engine imported by mistake to show itself.
    await page.waitForFunction(() => window.__qengine, null, { timeout: 3000 }).catch(() => {});
    results.again = { loads, imported: await page.evaluate(() => Boolean(window.__qengine)), page: await page.evaluate(() => window.__qpage ?? null), status: await page.textContent("#status"), disabled: await page.isDisabled("#load") };
    await context.close();
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

// The verdicts. Each says where the page must have stopped, not only that it
// did: a page that refuses for the wrong reason fails too.
const failures = [];
const fail = (why) => failures.push(why);
const stoppedAt = (r) => r.checks.filter((c) => c.verdict === "fail")[0];

if (good.phase !== "ready") fail(`good: the page did not reach ready (${good.phase}: ${good.error ?? (good.timeout ? "timed out" : "")})`);
if (good.checks.some((c) => c.verdict === "fail")) fail(`good: a check failed: ${stoppedAt(good)?.subject}`);
const alias = good.checks.find((c) => c.step === "alias");
if (!alias || alias.verdict !== "pass") fail("good: the tied output head was not confirmed as the embedding's bytes");
if (!good.checks.some((c) => c.step === "root" && c.verdict === "pass")) fail("good: the whole-file root was not checked");
if (!good.output) fail("good: no answer");
else {
  const got = good.output.tokens.slice(0, 16);
  if (got.length < 16 || got.some((t, i) => t !== good.expected[i])) fail(`good: the page's tokens ${got} are not the reference's ${good.expected}`);
}
const listed = good.checks.find((c) => c.subject === "kernel is the one this deploy lists");
if (listed?.verdict !== "pass") fail(`good: the kernel was not compared with the deploy's listing (${listed?.verdict ?? "no check"})`);
if (good.promptStart !== "") fail(`good: the prompt box started with text, so typing would join it: ${JSON.stringify(good.promptStart)}`);
if (good.errors.length) fail(`good: page errors: ${good.errors.join(" | ")}`);

const flip = results.flip;
const flipStop = stoppedAt(flip);
if (flip.phase !== "failed" || flipStop?.step !== "admit" || flipStop?.subject !== `${victim.name} label and length`)
  fail(`flip: expected to stop at the admission of ${victim.name}, stopped at ${flipStop?.step}: ${flipStop?.subject} (${flip.phase})`);
if (flip.checks.some((c) => c.subject === "download → compile")) fail("flip: the load went on to compile");

const root = results.root;
const rootStop = stoppedAt(root);
if (root.phase !== "failed" || rootStop?.step !== "root") fail(`root: expected to stop at the whole-file root, stopped at ${rootStop?.step}: ${rootStop?.subject} (${root.phase})`);

const stale = results.stale;
const staleStop = stoppedAt(stale);
if (stale.phase !== "failed" || staleStop?.subject !== "kernel is the one this deploy lists" || !/^stale kernel/.test(stale.error ?? ""))
  fail(`stale: expected to stop with "stale kernel" at the deploy comparison, stopped at ${staleStop?.step}: ${staleStop?.subject} (${stale.phase}: ${stale.error})`);
if (stale.checks.some((c) => !["kernel", "stop"].includes(c.step))) fail(`stale: the page went on past the kernel: ${stale.checks.map((c) => c.subject).join(", ")}`);

// GP-07: an earlier deploy's file is removed, this deploy's kept, one reload.
const earlier = results.earlier;
if (!earlier.imported) fail(`earlier: the engine was not imported after the stale file was removed (${earlier.status})`);
if (earlier.loads !== 2) fail(`earlier: the page loaded ${earlier.loads} times, expected one reload`);
if (earlier.page?.removed !== 1 || earlier.page?.stale.length !== 0) fail(`earlier: expected one file removed and none stale after the reload, saw ${JSON.stringify(earlier.page)}`);
if (earlier.held.engine) fail("earlier: the earlier deploy's engine file is still in the cache");
if (!earlier.held.css) fail("earlier: a cached file that is this deploy's was removed");
if (!/^Updated: files from an earlier deploy were replaced \(1\)\.$/.test(earlier.status ?? "")) fail(`earlier: the page did not say what it replaced: ${earlier.status}`);
const again = results.again;
if (again.imported) fail("again: the engine was imported with a stale file still served after a reload");
if (again.loads !== 1) fail(`again: the page loaded ${again.loads} times, expected no further reload`);
if (!/^Stopped: stale page/.test(again.status ?? "") || !again.disabled) fail(`again: expected "Stopped: stale page" and no Load button, saw ${again.status} (disabled: ${again.disabled})`);

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.error("journey: the page loads through the kernel, answers as the reference does, and stops where tampering or a stale kernel is, and replaces an earlier deploy's files");
