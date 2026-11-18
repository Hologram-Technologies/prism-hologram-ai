// A model's files kept in the browser, each named by its BLAKE3, the address
// Hologram AI names everything by, so a second visit reads them from here
// instead of the network.
//
// The hub's record names a file by the hash its source publishes (Hugging
// Face's SHA-256), so beside each kept file is a pointer from that hash to the
// file's BLAKE3. Neither is trusted. A kept file's BLAKE3 is derived again
// from its bytes as they are read, and a file that is not its name stops the
// load; and the bytes go to the same loader a download goes to, which checks
// them against the record: the small files by their SHA-256 and length, the
// graph and data files through the module's importer with the kernel asked
// about each one. A load that fails forgets every file and pointer it read or
// kept, so the next load downloads them again.
//
// Files the record names no SHA-256 for (the small files Git stores itself)
// are not kept; they are fetched each time. The browser's private file system
// (OPFS) is per origin and may be cleared under storage pressure.

const HEX = /^[0-9a-f]{64}$/;
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

export async function keepFiles({ record, open, module: m, root, name = "hologram-files" }) {
  const base = root ?? (await navigator.storage.getDirectory());
  const dir = await base.getDirectoryHandle(name, { create: true });
  const names = await base.getDirectoryHandle(`${name}-names`, { create: true });
  const counts = { read: 0, readBytes: 0, fetched: 0, fetchedBytes: 0 };
  const touched = [];
  const writing = [];
  const kept = [];

  const recorded = (path) => {
    const file = record.files.find((f) => f.path === path);
    return file && HEX.test(file.sha256 ?? "") ? file : null;
  };
  const pointer = (file) => `sha256-${file.sha256}`;
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

  // The BLAKE3 a pointer names and the kept file under it, when both are held
  // and the file is the record's length.
  const held = async (file) => {
    try {
      const named = (await (await (await names.getFileHandle(pointer(file))).getFile()).text()).trim();
      if (!HEX.test(named)) return null;
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
    touched.push({ names: pointer(file) });
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
    const [mine, keep] = body.tee();
    const partial = `partial-${crypto.randomUUID()}`;
    touched.push({ dir: partial });
    const handle = await dir.getFileHandle(partial, { create: true });
    const { stream, digest } = hashing();
    writing.push(
      keep.pipeThrough(stream).pipeTo(await handle.createWritable()).then(async () => {
        const named = digest();
        // The same bytes kept already, under another record's name: keep one.
        if (await dir.getFileHandle(named).then(() => true, (e) => (missing(e) ? false : Promise.reject(e)))) await dir.removeEntry(partial);
        else await handle.move(named);
        touched.push({ dir: named });
        const w = await (await names.getFileHandle(pointer(file), { create: true })).createWritable();
        await w.write(named);
        await w.close();
        kept.push({ path, address: `blake3:${named}` });
      }),
    );
    counts.fetched++;
    counts.fetchedBytes += file.size;
    return mine;
  }

  // Once the writes this load started have ended.
  const settled = () => Promise.allSettled(writing);

  // Remove every file and pointer this load read or kept: called when the
  // load failed, since one of them may be what was not the record's.
  async function forget() {
    await settled();
    for (const t of touched) {
      try {
        if (t.names) await names.removeEntry(t.names);
        else await dir.removeEntry(t.dir);
      } catch (e) {
        if (!missing(e)) throw e;
      }
    }
    touched.length = 0;
  }

  return { open: openKept, forget, settled, counts, kept };
}
