// A model's files kept in the browser, each named by its BLAKE3, the address
// Hologram AI names everything by, so a second visit reads them from here
// instead of the network.
//
// The hub's record names each file by its BLAKE3, and a kept file is held
// under that name. The name is not trusted: a kept file's BLAKE3 is derived
// again from its bytes as they are read, and a file that is not its name stops
// the load; and the bytes go to the same loader a download goes to, which
// checks them against the record: the small files by their BLAKE3 and length,
// the graph and data files through the module's importer with the kernel asked
// about each one. A load that fails forgets every file it read or kept, so the
// next load downloads them again.
//
// Room is the kernel's decision 26. Before a downloaded file is kept, each
// file kept here is offered to the kernel, oldest first, with the files this
// load reads or keeps pinned: one is dropped only on evict, and the offers
// stop at enough. On does not fit the file is not kept, and the load reads it
// from the download as before.
//
// The browser's private file system (OPFS) is per origin and may be cleared
// under storage pressure.
//
// A graph or data file the hub's UOR index lists tensor by tensor (`manifest`,
// from loader.mjs) is kept as its tensors, each once under its BLAKE3 address
// whatever model, precision or repository it came with, and the rest of its
// bytes (framing, the graph, small tensors) under `rest-` and the file's
// BLAKE3. A tensor already held is read from here and not downloaded; the
// bytes held nowhere here are range-read, in as few reads as the gaps allow,
// from the first of the record's sources that answers. Each span is asked of
// decision 23 first. Nothing new is kept until decision 28 says every
// listed address is the one the importer derived from the bytes (admit).

const HEX = /^[0-9a-f]{64}$/;
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
const unhex = (h) => Uint8Array.from(h.match(/../g), (b) => parseInt(b, 16));
const be = (value, width) => {
  const out = new Uint8Array(width);
  let v = BigInt(value);
  for (let i = width - 1; i >= 0; i--) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
};

// Tensors at least this long are kept apart under their own address; smaller
// ones (norms, biases, constants) travel with the rest of their file, where
// thousands of tiny files would cost more than the bytes they save.
export const APART = 4096;

// The chunks of a ReadableStream, as an async iterable.
async function* chunksOf(stream) {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

// The places a record lists for a file: its origin, then each source that
// does not say it lacks the file. None is trusted; each byte is checked.
export const sourcesOf = (record) => [record.origin?.resolve, ...(record.sources ?? []).map((s) => s.resolve && { resolve: s.resolve, missing: s.missing ?? [] })]
  .map((s) => (typeof s === "string" ? { resolve: s, missing: [] } : s))
  .filter((s) => s?.resolve);

// Decision 28 on a graph's import: the label of the addresses the importer
// derived, in the index's order, against the label of the addresses the index
// lists, and how many of each. Answers the kernel's verdict and the counts.
export async function manifestReady({ manifest, table, module: m, kernel }) {
  const label = (addresses) => {
    const id = new DataView(m.call("hash_start").slice().buffer).getUint32(0, true);
    for (const a of addresses) m.callWith("hash_take", unhex(a.slice(7)), id);
    return m.call("hash_finish", id).slice();
  };
  const listed = [];
  const derived = [];
  let other = 0;
  for (const plan of manifest.values()) {
    for (const t of plan.tensors) {
      listed.push(t.address);
      const place = table.places[t.name];
      if (!place || place[1] !== t.offset || place[2] !== t.length) continue;
      derived.push(place[3]);
      if (place[3] !== t.address) other++;
    }
  }
  const request = new Uint8Array(81);
  request[0] = 28;
  request.set(label(derived), 1);
  request.set(label(listed), 33);
  request.set(be(listed.length, 8), 65);
  request.set(be(derived.length, 8), 73);
  return { answer: await kernel(request), listed: listed.length, derived: derived.length, other };
}

export async function keepFiles({ record, open, module: m, kernel, root, name = "hologram-files", manifest = new Map(), sources = sourcesOf(record) }) {
  const base = root ?? (await navigator.storage.getDirectory());
  const dir = await base.getDirectoryHandle(name, { create: true });
  const counts = { read: 0, readBytes: 0, fetched: 0, fetchedBytes: 0, notKept: 0, dropped: 0, droppedBytes: 0 };
  // The tensors: read here, downloaded, and newly kept; the bytes downloaded
  // for the rest of a file, and how many ranged reads it took.
  const tensors = { read: 0, readBytes: 0, fetched: 0, fetchedBytes: 0, kept: 0, keptBytes: 0, restFetchedBytes: 0, reads: 0 };
  const touched = [];
  const writing = [];
  const kept = [];
  // Written as `partial-` files while the bytes arrive, named only on admit;
  // and the files kept as tensors (not those there was no room for).
  const pending = [];
  const stored = new Set();

  const recorded = (path) => {
    const file = record.files.find((f) => f.path === path);
    return file && HEX.test(file.blake3 ?? "") ? file : null;
  };
  const missing = (e) => e && e.name === "NotFoundError";

  // The bytes' BLAKE3, taken in the module as a stream passes through.
  const hashing = () => {
    const id = new DataView(m.call("hash_start").slice().buffer).getUint32(0, true);
    let done = null;
    const stream = new TransformStream({
      transform(chunk, controller) {
        m.callWith("hash_take", chunk, id);
        controller.enqueue(chunk);
      },
      flush() {
        done = hex(m.call("hash_finish", id));
      },
    });
    return { stream, digest: () => done };
  };

  // The kept file under the BLAKE3 the record names, when it is held and is
  // the record's length.
  const held = async (file) => {
    try {
      const named = file.blake3;
      const bytes = await (await dir.getFileHandle(named)).getFile();
      return bytes.size === file.size ? { named, bytes } : null;
    } catch (e) {
      if (missing(e)) return null;
      throw e;
    }
  };

  // A stream of the file's bytes: from here when held, its BLAKE3 derived
  // again on the way and the stream failed if it is not the name; otherwise
  // from open(), named by its BLAKE3 as it is written here, and committed
  // under that name only when the stream ends.
  async function openKept(path) {
    const file = recorded(path);
    if (!file) return open(path);
    if (manifest.has(path)) return openTensors(path, file);
    const local = await held(file);
    if (local) {
      touched.push({ dir: local.named });
      counts.read++;
      counts.readBytes += file.size;
      const { stream, digest } = hashing();
      const check = new TransformStream({
        flush(controller) {
          if (digest() !== local.named) controller.error(new Error(`${path}: the file kept as blake3:${local.named} is blake3:${digest()}`));
          else kept.push({ path, address: `blake3:${local.named}` });
        },
      });
      return local.bytes.stream().pipeThrough(stream).pipeThrough(check);
    }
    const body = await ranged(path, 0, file.size, file.size);
    if (!(await room(file.size))) {
      counts.notKept++;
      return body;
    }
    const [mine, keep] = body.tee();
    const partial = `partial-${crypto.randomUUID()}`;
    touched.push({ dir: partial });
    const handle = await dir.getFileHandle(partial, { create: true });
    const { stream, digest } = hashing();
    writing.push(
      keep.pipeThrough(stream).pipeTo(await handle.createWritable()).then(async () => {
        const named = digest();
        // Bytes that are not the record's are not kept: the loader refuses them.
        if (named !== file.blake3) {
          await dir.removeEntry(partial);
          return;
        }
        // The same bytes kept already, for another model: keep one.
        if (await dir.getFileHandle(named).then(() => true, (e) => (missing(e) ? false : Promise.reject(e)))) await dir.removeEntry(partial);
        else await handle.move(named);
        touched.push({ dir: named });
        kept.push({ path, address: `blake3:${named}` });
      }),
    );
    counts.fetched++;
    counts.fetchedBytes += file.size;
    return mine;
  }

  // A kept entry's bytes, when it is held at that length.
  const heldAt = async (named, length) => {
    try {
      const f = await (await dir.getFileHandle(named)).getFile();
      return f.size === length ? f : null;
    } catch (e) {
      if (missing(e)) return null;
      throw e;
    }
  };

  // Bytes [from, to) of a file from the first listed source that answers with
  // exactly them: the whole file by a plain read, a part by a Range read.
  async function ranged(path, from, to, size) {
    const tried = [];
    for (const source of sources) {
      if (source.missing.includes(path)) continue;
      const whole = from === 0 && to === size;
      try {
        const response = await fetch(source.resolve + path, whole ? {} : { headers: { Range: `bytes=${from}-${to - 1}` } });
        const span = /^bytes (\d+)-(\d+)\//.exec(response.headers.get("content-range") ?? "");
        if (whole ? response.ok : response.status === 206 && span && Number(span[1]) === from && Number(span[2]) === to - 1) {
          return response.body;
        }
        tried.push(`${source.resolve}: HTTP ${response.status}`);
        await response.body?.cancel();
      } catch (e) {
        tried.push(`${source.resolve}: ${e.message}`);
      }
    }
    throw new Error(`${path} bytes ${from} to ${to - 1}: no listed source answered (${tried.join("; ") || "none listed"})`);
  }

  // A file as its tensors and the rest: each tensor the index lists, at least
  // APART long, held here or read with its neighbours from a source; the rest
  // held here under the file's name or read with them.
  async function openTensors(path, file) {
    const plan = manifest.get(path);
    const apart = plan.tensors.filter((t) => t.length >= APART).sort((a, b) => a.offset - b.offset);
    // Decision 23: each span inside the file, after the one before it.
    let end = 0;
    for (const t of apart) {
      const request = new Uint8Array(35);
      request[0] = 23;
      request.set(be(file.size, 8), 3);
      request.set(be(t.offset, 8), 11);
      request.set(be(t.length, 8), 19);
      request.set(be(end, 8), 27);
      const answer = await kernel(request);
      if (answer !== "accept") throw new Error(`the hub's UOR index places ${t.name} at ${t.offset} for ${t.length} bytes, which decision 23 answers "${answer}" in ${path} of ${file.size} bytes`);
      end = t.offset + t.length;
    }
    const segments = [];
    let at = 0;
    let restAt = 0;
    const rest = (to) => {
      segments.push({ from: at, to, restAt });
      restAt += to - at;
    };
    for (const t of apart) {
      if (t.offset > at) rest(t.offset);
      segments.push({ from: t.offset, to: t.offset + t.length, named: t.address.slice(7), held: await heldAt(t.address.slice(7), t.length) });
      at = t.offset + t.length;
    }
    if (at < file.size) rest(file.size);
    const restNamed = `rest-${file.blake3}`;
    const restHeld = restAt ? await heldAt(restNamed, restAt) : null;
    const local = (s) => (s.named ? s.held : restHeld);
    // Room for what is new, asked of decision 26 once for the file.
    const fresh = segments.reduce((n, s) => n + (local(s) ? 0 : s.to - s.from), 0);
    const keep = fresh === 0 || (await room(fresh));
    if (!keep) counts.notKept++;
    else stored.add(path);
    for (const s of segments) if (s.held) touched.push({ dir: s.named });
    if (restHeld) touched.push({ dir: restNamed });

    // A writer for new bytes: a partial file named on admit.
    const partial = async (named) => {
      const temp = `partial-${crypto.randomUUID()}`;
      touched.push({ dir: temp });
      const handle = await dir.getFileHandle(temp, { create: true });
      const entry = { temp, handle, named, writer: await handle.createWritable() };
      pending.push(entry);
      return entry;
    };
    const restWriter = keep && !restHeld && restAt ? await partial(restNamed) : null;

    async function* produce() {
      for (let i = 0; i < segments.length;) {
        const s = segments[i];
        if (local(s)) {
          const bytes = s.named ? s.held : restHeld.slice(s.restAt, s.restAt + (s.to - s.from));
          if (s.named) {
            tensors.read++;
            tensors.readBytes += s.to - s.from;
          }
          yield* chunksOf(bytes.stream());
          i++;
          continue;
        }
        // The run of segments held nowhere here, in one read.
        let j = i;
        while (j < segments.length && !local(segments[j])) j++;
        const from = s.from;
        const to = segments[j - 1].to;
        let pos = from;
        let k = i;
        let writer = null;
        tensors.reads++;
        for await (const chunk of chunksOf(await ranged(path, from, to, file.size))) {
          if (pos + chunk.length > to) throw new Error(`${path} bytes ${from} to ${to - 1}: the source sent more`);
          for (let off = 0; off < chunk.length;) {
            const seg = segments[k];
            const take = Math.min(chunk.length - off, seg.to - pos);
            if (seg.named) {
              tensors.fetchedBytes += take;
              if (keep) {
                writer ??= await partial(seg.named);
                await writer.writer.write(chunk.subarray(off, off + take));
              }
            } else {
              tensors.restFetchedBytes += take;
              if (restWriter) await restWriter.writer.write(chunk.subarray(off, off + take));
            }
            off += take;
            pos += take;
            if (pos === seg.to) {
              if (seg.named) tensors.fetched++;
              if (writer) await writer.writer.close();
              writer = null;
              k++;
            }
          }
          yield chunk;
        }
        if (pos !== to) throw new Error(`${path} bytes ${from} to ${to - 1}: the source ended at ${pos}`);
        i = j;
      }
      if (restWriter) await restWriter.writer.close();
      // Counted as a file, as a whole file kept or read is.
      if (fresh === 0) {
        counts.read++;
        counts.readBytes += file.size;
      } else if (keep) {
        counts.fetched++;
        counts.fetchedBytes += file.size;
      }
    }
    const chunks = produce();
    return new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await chunks.next();
          if (done) controller.close();
          else controller.enqueue(value);
        } catch (e) {
          controller.error(e);
        }
      },
      cancel: () => chunks.return(),
    });
  }

  // Decision 28 on the import, then the new bytes named: a tensor under its
  // address unless the same bytes are held already, the rest under its file's
  // name. Throws when the kernel does not answer ready; the caller forgets.
  async function admit(table) {
    if (!manifest.size) return null;
    const verdict = await manifestReady({ manifest, table, module: m, kernel });
    if (verdict.answer !== "ready") {
      throw new Error(`the model is not ready: decision 28 answered "${verdict.answer}": the importer derived ${verdict.derived} of the ${verdict.listed} addresses the hub's UOR index lists, ${verdict.other} of them not the address listed`);
    }
    for (const p of pending) {
      if (await heldAt(p.named, (await p.handle.getFile()).size)) {
        await dir.removeEntry(p.temp);
      } else {
        await p.handle.move(p.named);
        if (!p.named.startsWith("rest-")) {
          tensors.kept++;
          tensors.keptBytes += (await p.handle.getFile()).size;
        }
      }
      touched.push({ dir: p.named });
    }
    pending.length = 0;
    for (const path of manifest.keys()) {
      const file = recorded(path);
      if (file && stored.has(path)) kept.push({ path, address: `blake3:${file.blake3}` });
    }
    return verdict;
  }

  // Decision 26's request: pinned, candidates left, bytes needed, free and freed.
  const join = (tag, pinned, remaining, needed, free, freed) => {
    const out = new Uint8Array(30);
    const view = new DataView(out.buffer);
    out[0] = tag;
    out[1] = pinned ? 1 : 0;
    view.setUint32(2, remaining);
    view.setBigUint64(6, BigInt(needed));
    view.setBigUint64(14, BigInt(free));
    view.setBigUint64(22, BigInt(freed));
    return out;
  };
  // Whether there is room to keep `needed` bytes, dropping what the kernel says.
  async function room(needed) {
    const { quota = 0, usage = 0 } = await navigator.storage.estimate();
    const free = Math.max(0, quota - usage);
    const pinned = new Set(touched.map((t) => t.dir).filter(Boolean));
    const candidates = [];
    for await (const [n, h] of dir.entries()) {
      if (h.kind !== "file" || !HEX.test(n.replace(/^rest-/, ""))) continue;
      const f = await h.getFile();
      candidates.push({ name: n, size: f.size, at: f.lastModified });
    }
    candidates.sort((a, b) => a.at - b.at || (a.name < b.name ? -1 : 1));
    let freed = 0;
    for (let i = 0; ; i++) {
      const c = candidates[i];
      const answer = await kernel(join(26, c ? pinned.has(c.name) : false, candidates.length - i, needed, free, freed));
      if (answer === "enough") return true;
      if (answer === "does not fit") return false;
      if (answer === "evict" && c) {
        await dir.removeEntry(c.name);
        freed += c.size;
        counts.dropped++;
        counts.droppedBytes += c.size;
        continue;
      }
      if (answer === "keep" && c) continue;
      throw new Error(`room for ${needed} bytes: the kernel answered "${answer}"`);
    }
  }

  // Once the writes this load started have ended.
  const settled = () => Promise.allSettled(writing);

  // Remove every file this load read or kept: called when the load failed,
  // since one of them may be what was not the record's.
  async function forget() {
    await settled();
    for (const p of pending.splice(0)) await p.writer.abort().catch(() => {});
    for (const t of touched) {
      try {
        await dir.removeEntry(t.dir);
      } catch (e) {
        if (!missing(e)) throw e;
      }
    }
    touched.length = 0;
  }

  return { open: openKept, admit, forget, settled, counts, tensors, kept };
}
