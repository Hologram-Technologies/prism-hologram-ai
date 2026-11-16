// Tensors kept in the browser by their address, once each. A tensor's address
// is the BLAKE3 of its bytes as a uor-addr label, blake3:<64 hex>, so a tensor
// held under one model is not stored or fetched again for another, and a
// return visit reads every tensor from here with the network off.
//
// Nothing read from here is trusted. get() hands bytes back only after
// admit(address, bytes) says they are the tensor named: in the page that is
// the kernel's admission decision on the label the wasm module computes, the
// same question a downloaded tensor is asked. Bytes not admitted are removed.
// Which tensors to drop when space is short is the kernel's decision 26, asked
// by the caller; this module only stores, reads and removes.
//
// The browser's private file system (OPFS) holds one file per tensor, named
// by the address's hex. It is per origin and survives reloads; the browser
// may clear it under storage pressure unless the page is granted persistence.

const BLAKE3 = /^blake3:([0-9a-f]{64})$/;

// The hex of a BLAKE3 address. Both blake3: and sha256: labels are 71 bytes,
// so which hash a label names is read from its prefix, never its length.
function fileName(address) {
  const m = BLAKE3.exec(address);
  if (!m) throw new Error(`address: ${String(address).slice(0, 80)} is not a blake3: label`);
  return m[1];
}

export async function openStore({ name = "hologram-tensors", admit, root } = {}) {
  if (typeof admit !== "function") throw new Error("store: no admission check given; nothing is read back unchecked");
  const base = root ?? (await navigator.storage.getDirectory());
  const dir = await base.getDirectoryHandle(name, { create: true });
  const counts = { stored: 0, storedBytes: 0, held: 0, heldBytes: 0, read: 0, readBytes: 0, refused: 0 };

  const handle = async (address) => {
    try {
      return await dir.getFileHandle(fileName(address));
    } catch (e) {
      if (e && e.name === "NotFoundError") return null;
      throw e;
    }
  };

  // The length held under an address, or null.
  async function has(address) {
    const h = await handle(address);
    return h ? (await h.getFile()).size : null;
  }

  // Keep bytes under their address. A tensor already held at that length is
  // not written again; that is the saving, counted in held and heldBytes.
  async function put(address, bytes) {
    const n = fileName(address);
    const length = await has(address);
    if (length === bytes.byteLength) {
      counts.held++;
      counts.heldBytes += bytes.byteLength;
      return { stored: false };
    }
    const h = await dir.getFileHandle(n, { create: true });
    const w = await h.createWritable();
    await w.write(bytes);
    await w.close();
    counts.stored++;
    counts.storedBytes += bytes.byteLength;
    return { stored: true };
  }

  // The bytes under an address, admitted again, or null: when nothing is held,
  // when the length is not the one expected, or when admission refuses them,
  // in which case the file is removed.
  async function get(address, length) {
    const h = await handle(address);
    if (!h) return null;
    const file = await h.getFile();
    const bytes = new Uint8Array(await file.arrayBuffer());
    if ((length !== undefined && bytes.byteLength !== length) || !(await admit(address, bytes))) {
      counts.refused++;
      await remove(address);
      return null;
    }
    counts.read++;
    counts.readBytes += bytes.byteLength;
    return bytes;
  }

  async function remove(address) {
    try {
      await dir.removeEntry(fileName(address));
      return true;
    } catch (e) {
      if (e && e.name === "NotFoundError") return false;
      throw e;
    }
  }

  // Every address held and its length, for a caller deciding what to drop.
  async function list() {
    const out = [];
    for await (const [n, h] of dir.entries()) {
      if (h.kind === "file" && /^[0-9a-f]{64}$/.test(n)) out.push({ address: `blake3:${n}`, length: (await h.getFile()).size });
    }
    return out;
  }

  return { has, put, get, remove, list, counts };
}
