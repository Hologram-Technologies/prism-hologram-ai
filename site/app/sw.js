// The engine page's own worker, so a return visit loads with the network off.
//
// Why its own: the site's worker (../sw.js) is the template's frame, and it
// keeps every page under one key, so it cannot answer this page offline.
//
// What it keeps: the files this deploy lists in precache.json (the page, its
// engine, the kernel holo, the vendored runtime), all of them on install, so
// the first visit leaves nothing out whatever it fetched before this worker
// took over. How: network first, every time, so an online visit always gets
// this deploy's files and the page's own stale-kernel check (GP-06) is
// unchanged; the kept copy answers only when the network fails. precache.json
// itself is never answered from here, so the page knows when it is offline.
//
// What it does not keep: model bytes and the hub's index. Tensors are kept in
// the private file system by their BLAKE3 address (engine/files.mjs) and the
// hub's index files by the page (engine/loader.mjs); both are checked again
// on every read, which a worker answering in their place would not do.

const CACHE = "hologram-engine-page";
const LISTING = new URL("../precache.json", self.location).href;
const ROOT = new URL("../", self.location).href;

// The address a request is kept under: a directory is its index.html, and a
// query never names another file.
const keyOf = (url) => {
  const u = new URL(url);
  return `${u.origin}${u.pathname.endsWith("/") ? `${u.pathname}index.html` : u.pathname}`;
};

async function listed() {
  const cache = await caches.open(CACHE);
  const kept = await cache.match(LISTING);
  if (!kept) return new Set();
  return new Set((await kept.json()).files.map((f) => keyOf(new URL(f.path, ROOT))));
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const response = await fetch(LISTING, { cache: "no-store" });
    if (!response.ok) throw new Error(`precache.json: HTTP ${response.status}`);
    const listing = await response.clone().json();
    const cache = await caches.open(CACHE);
    // The HTTP cache answers most of these: the page fetched them moments ago.
    await cache.addAll(listing.files.map((f) => new URL(f.path, ROOT).href));
    await cache.put(LISTING, response);
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// The listed files, read once per worker.
let own = null;
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || request.headers.has("range")) return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || url.href.split(/[?#]/)[0] === LISTING) return;
  const key = keyOf(url);
  const navigate = request.mode === "navigate";
  // Not this deploy's file: the browser fetches it as if no worker were here.
  if (own?.size && !navigate && !own.has(key)) return;
  event.respondWith((async () => {
    if (!own?.size) own = await listed();
    if (!navigate && !own.has(key)) return fetch(request);
    const cache = await caches.open(CACHE);
    try {
      const fresh = await fetch(request);
      if (fresh.ok) await cache.put(key, fresh.clone());
      return fresh;
    } catch (error) {
      const kept = await cache.match(key);
      if (kept) return kept;
      throw error;
    }
  })());
});
