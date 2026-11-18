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

const HEX = /^[0-9a-f]{64}$/;
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

export async function keepFiles({ record, open, module: m, kernel, root, name = "hologram-files" }) {
  const base = root ?? (await navigator.storage.getDirectory());
  const dir = await base.getDirectoryHandle(name, { create: true });
  const counts = { read: 0, readBytes: 0, fetched: 0, fetchedBytes: 0, notKept: 0, dropped: 0, droppedBytes: 0 };
  const touched = [];
  const writing = [];
  const kept = [];

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
    const body = await open(path);
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
      if (h.kind !== "file" || !HEX.test(n)) continue;
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
    for (const t of touched) {
      try {
        await dir.removeEntry(t.dir);
      } catch (e) {
        if (!missing(e)) throw e;
      }
    }
    touched.length = 0;
  }

  return { open: openKept, forget, settled, counts, kept };
}
