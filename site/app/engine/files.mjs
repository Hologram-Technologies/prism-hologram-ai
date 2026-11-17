// A model's files kept in the browser under the SHA-256 the hub's record names
// for each, so a second visit reads them from here instead of the network.
//
// Nothing read from here is trusted. A kept file is handed to the same loader
// a download is handed to, and that loader checks every byte against the
// record: the small files by their SHA-256 and length, the graph and data
// files through the module's importer with the kernel asked about each one.
// A file kept here that is not the record's fails exactly as a tampered
// download fails, and forget() then removes every file this load read or
// kept, so the next load downloads them again.
//
// Files the record names no SHA-256 for (the small files Git stores itself)
// are not kept; they are fetched each time. The browser's private file system
// (OPFS) is per origin and may be cleared under storage pressure.

const SHA256 = /^[0-9a-f]{64}$/;

export async function keepFiles({ record, open, root, name = "hologram-files" }) {
  const base = root ?? (await navigator.storage.getDirectory());
  const dir = await base.getDirectoryHandle(name, { create: true });
  const counts = { read: 0, readBytes: 0, fetched: 0, fetchedBytes: 0 };
  const touched = new Set();
  const writing = [];

  const named = (path) => {
    const file = record.files.find((f) => f.path === path);
    return file && SHA256.test(file.sha256 ?? "") ? file : null;
  };

  // The kept file under a SHA-256, when one is held at the record's length.
  const held = async (file) => {
    try {
      const kept = await (await dir.getFileHandle(file.sha256)).getFile();
      return kept.size === file.size ? kept : null;
    } catch (e) {
      if (e && e.name === "NotFoundError") return null;
      throw e;
    }
  };

  // A stream of the file's bytes: from here when held, otherwise from open(),
  // written here as it is read. A write is committed only when the stream
  // ends, so a download cut short leaves nothing behind.
  async function openKept(path) {
    const file = named(path);
    if (!file) return open(path);
    touched.add(file.sha256);
    const kept = await held(file);
    if (kept) {
      counts.read++;
      counts.readBytes += file.size;
      return kept.stream();
    }
    const body = await open(path);
    const [mine, keep] = body.tee();
    const writable = await (await dir.getFileHandle(file.sha256, { create: true })).createWritable();
    writing.push(keep.pipeTo(writable));
    counts.fetched++;
    counts.fetchedBytes += file.size;
    return mine;
  }

  // Once the writes this load started have ended.
  const settled = () => Promise.allSettled(writing);

  // Remove every file this load read or kept: called when the load failed,
  // since one of them may be the file that was not the record's.
  async function forget() {
    await settled();
    for (const n of touched) {
      try {
        await dir.removeEntry(n);
      } catch (e) {
        if (!(e && e.name === "NotFoundError")) throw e;
      }
    }
    touched.clear();
  }

  return { open: openKept, forget, settled, counts };
}
