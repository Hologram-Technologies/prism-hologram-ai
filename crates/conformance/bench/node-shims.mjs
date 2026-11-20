// What a browser gives the page's engine module that Node does not: a page
// `location`, `fetch` of file: URLs, a Web Worker, and a disk cache for model
// files. Nothing in the engine is changed; these only stand in for the browser.
//
// The cache mirrors Hugging Face's resolve URLs as paths, `<cache>/<repo>/
// resolve/<revision>/<file>`, so the float-model side (float_agreement.py)
// opens the very bytes the integer engine read. A file is downloaded whole
// once, and a Range read is answered from it with 206, as the CDN answers.

import { readFile, mkdir, stat, rename, open } from "node:fs/promises";
import { createWriteStream, createReadStream } from "node:fs";
import { Readable } from "node:stream";
import { pipeline as streamPipe } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { Worker as ThreadWorker } from "node:worker_threads";
import path from "node:path";

const HF = /^https:\/\/huggingface\.co\/(.+\/resolve\/.+)$/;

export function installShims({ cacheDir, href = "https://gethologram.ai/app/" }) {
  globalThis.location = { href };
  const realFetch = globalThis.fetch;
  const log = [];
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url.startsWith("file:")) return new Response(await readFile(fileURLToPath(url)), { status: 200 });
    const hf = HF.exec(url.split("?")[0]);
    if (!hf) {
      log.push({ url, from: "network" });
      return realFetch(input, init);
    }
    const file = path.join(cacheDir, decodeURIComponent(hf[1]));
    let from = "disk";
    try { await stat(file); } catch {
      from = "network";
      await mkdir(path.dirname(file), { recursive: true });
      const r = await realFetch(url);
      if (!r.ok) return r;
      await streamPipe(Readable.fromWeb(r.body), createWriteStream(file + ".part"));
      await rename(file + ".part", file);
    }
    const size = (await stat(file)).size;
    const range = /^bytes=(\d+)-(\d*)$/.exec(new Headers(init.headers).get("range") ?? "");
    log.push({ url, from, file: path.relative(cacheDir, file), range: range?.[0] ?? null });
    if (!range) return new Response(Readable.toWeb(createReadStream(file, { highWaterMark: 1 << 20 })), { status: 200, headers: { "content-length": String(size) } });
    const start = Number(range[1]);
    const end = range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
    const handle = await open(file);
    const bytes = Buffer.alloc(end - start + 1);
    await handle.read(bytes, 0, bytes.length, start);
    await handle.close();
    return new Response(bytes, { status: 206, headers: { "content-range": `bytes ${start}-${end}/${size}`, "content-length": String(bytes.length) } });
  };
  globalThis.Worker = NodeWebWorker;
  return log;
}

// A module Web Worker on worker_threads: the worker's `self.onmessage` and
// `self.postMessage` are wired to the thread's port, and it can start workers
// of its own, as the integer engine's worker starts its helpers (GP-52).
const BOOT = `
import { parentPort, workerData } from "node:worker_threads";
const { NodeWebWorker } = await import(workerData.shims);
globalThis.Worker = NodeWebWorker;
globalThis.location = { href: workerData.href };
globalThis.self = globalThis;
globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
parentPort.on("message", (data) => globalThis.onmessage?.({ data }));
await import(workerData.url);
`;

export class NodeWebWorker {
  constructor(url) {
    this.thread = new ThreadWorker(new URL(`data:text/javascript,${encodeURIComponent(BOOT)}`), { workerData: { url: String(url), href: globalThis.location.href, shims: import.meta.url } });
    this.thread.on("message", (data) => this.onmessage?.({ data }));
    this.thread.on("error", (error) => this.onerror?.({ message: error.message }));
  }
  postMessage(message, transfer = []) { this.thread.postMessage(message, transfer); }
  terminate() { return this.thread.terminate(); }
}
