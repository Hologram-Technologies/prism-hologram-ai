// The hub's index, checked by a reader with no server of its own (HK-54).
//
// Why: a model's bytes are only as trustworthy as the record that names them,
// and a record only as trustworthy as the root that names the record. This is
// the smallest piece a page needs to hold all three to account: the root to
// its signature, a record to the root by BLAKE3, a file to its record by
// BLAKE3. Nothing here asks gethologram.ai to vouch for anything.
//
// How the root is signed. There is no long-lived signing key. pages.yml signs
// `data/index/root.json` with Sigstore keyless signing (GitHub's OIDC token,
// a Fulcio certificate, an entry in the Rekor transparency log), published as
// `data/index/index.sigstore.json`. The trust anchors are public and pinned
// in the hub's Prism model, screen 3 (`app/tools/site_spec.py`, `trust.policy`):
//   identity            https://github.com/Hologram-Technologies/hub/.github/workflows/pages.yml@refs/heads/main
//   indexPredicateType  https://hologram.foundation/hub/index/v1
//   fulcioIntermediate  the sigstore-intermediate certificate, P-384
//   rekorKey            Rekor's ECDSA P-256 public key
// plus the OIDC issuer. `openIndex` takes that policy as an argument rather
// than carrying a copy: the model is its one source here, and a reader pins
// its own copy in its own model. A policy fetched from the hub would be
// vouched for by the thing it is meant to check.
//
// The one SHA-256 left. The signature's subject is the SHA-256 of root.json,
// because Sigstore's GitHub action attests SHA-256 digests only. Everything
// past the root is BLAKE3: the root names each record by `blake3`, and a
// record names each file it addresses by `blake3`, 64 lowercase hex.
//
// What: openIndex, readRecord, fileBlake3, checkFile, and HubClientError, whose
// `code` is one of CODES and never changes meaning.
//
// Self-contained for vendoring: the only static import is ./sigstore.mjs, at
// the BLAKE3 named by SIGSTORE_BLAKE3 below. Pass `hash` to use your own
// BLAKE3; only without it does this load ./piece-hash.mjs, ./piece-hash.wasm
// and ./pieces.mjs beside it.
import { verifyBundle } from "./sigstore.mjs";

// The web/sigstore.mjs this module was written and tested against.
export const SIGSTORE_BLAKE3 = "0f25ca52fed3967f36d291179b35b36f10ed39f60c46be948b94037a369f3afa";

export const CODES = Object.freeze({
  // root.json or its bundle could not be fetched.
  RootUnreachable: "RootUnreachable",
  // The bundle does not prove root.json; `facts` says which checks failed.
  RootUnverified: "RootUnverified",
  // The signed root does not name the repository.
  RecordUnknown: "RecordUnknown",
  // The signed root names the record without a BLAKE3.
  RecordUnaddressed: "RecordUnaddressed",
  // The record could not be fetched.
  RecordUnreachable: "RecordUnreachable",
  // The record's bytes are not the BLAKE3 the signed root names.
  RecordMismatch: "RecordMismatch",
  // The record does not give the file a BLAKE3.
  FileUnaddressed: "FileUnaddressed",
  // The file's bytes are not the BLAKE3 and length its record names.
  FileMismatch: "FileMismatch",
});

export class HubClientError extends Error {
  constructor(code, message, detail = {}) {
    super(message);
    this.name = "HubClientError";
    this.code = code;
    Object.assign(this, detail);
  }
}

const FACTS = ["chain", "time", "log", "signature", "identity", "subject", "predicate"];
const HEX64 = /^[0-9a-f]{64}$/;
const toHex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

// BLAKE3 by the hub's own hasher, loaded only when a caller brings none.
let defaultHash = null;
async function hubHash() {
  defaultHash ??= (async () => {
    const [{ instantiate }, { streamHasher }] = await Promise.all([import("./piece-hash.mjs"), import("./pieces.mjs")]);
    const url = new URL("./piece-hash.wasm", import.meta.url);
    const wasm = url.protocol === "file:"
      ? await (await import("node:fs/promises")).readFile(url)
      : fetch(url);
    const hasher = await instantiate(wasm);
    return async (input) => {
      const h = streamHasher(hasher);
      if (input instanceof Uint8Array) h.update(input);
      else for await (const chunk of input) h.update(chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
      return toHex(h.digest());
    };
  })();
  return defaultHash;
}

// Counts bytes as they pass to the hasher, so a file is checked for length too.
async function* counted(input, seen) {
  if (input instanceof Uint8Array) { seen.length += input.length; yield input; return; }
  for await (const chunk of input) {
    const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
    seen.length += bytes.length;
    yield bytes;
  }
}

async function bytesOf(fetchImpl, url, code, what) {
  let response;
  try {
    response = await fetchImpl(url);
  } catch (error) {
    throw new HubClientError(code, `${what}: ${error?.message ?? error}`, { url: String(url) });
  }
  if (!response.ok) throw new HubClientError(code, `${what}: ${response.status}`, { url: String(url), status: response.status });
  return new Uint8Array(await response.arrayBuffer());
}

// The signed root at `base` (e.g. https://gethologram.ai/data/index/), or a
// RootUnverified error naming every check that failed.
export async function openIndex(base, policy, { fetch: fetchImpl = globalThis.fetch, hash } = {}) {
  const rootBytes = await bytesOf(fetchImpl, new URL("root.json", base), CODES.RootUnreachable, "root.json");
  const bundleBytes = await bytesOf(fetchImpl, new URL("index.sigstore.json", base), CODES.RootUnreachable, "index.sigstore.json");
  let facts;
  try {
    facts = await verifyBundle(JSON.parse(new TextDecoder().decode(bundleBytes)), rootBytes, { ...policy, predicateType: policy.indexPredicateType });
  } catch {
    facts = Object.fromEntries(FACTS.map((k) => [k, false]));
  }
  const failed = FACTS.filter((k) => !facts?.[k]);
  if (failed.length) throw new HubClientError(CODES.RootUnverified, `the index root is not proven by its signature: ${failed.join(", ")}`, { facts, failed });
  const root = JSON.parse(new TextDecoder().decode(rootBytes));
  return { base: String(base), root, facts, fetch: fetchImpl, hash: hash ?? (await hubHash()) };
}

// A repository's record, only if its bytes are the BLAKE3 the signed root names.
export async function readRecord(index, repo) {
  const entry = index.root.records.find((r) => r.repo.toLowerCase() === repo.toLowerCase());
  if (!entry) throw new HubClientError(CODES.RecordUnknown, `the signed root does not name ${repo}`, { repo });
  if (!HEX64.test(entry.blake3 ?? "")) throw new HubClientError(CODES.RecordUnaddressed, `the signed root names ${entry.repo} without a BLAKE3`, { repo: entry.repo });
  const bytes = await bytesOf(index.fetch, new URL(`models/${entry.repo}.json`, index.base), CODES.RecordUnreachable, `the record for ${entry.repo}`);
  const found = await index.hash(bytes);
  if (found !== entry.blake3) {
    throw new HubClientError(CODES.RecordMismatch, `the record for ${entry.repo} is not the bytes the signed root names`, { repo: entry.repo, expected: entry.blake3, found });
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

// The BLAKE3 a record gives a file, 64 lowercase hex, or null.
export function fileBlake3(record, path) {
  const file = record.files?.find((f) => f.path === path);
  return HEX64.test(file?.blake3 ?? "") ? file.blake3 : null;
}

// Hashes `input` (bytes, a stream, or any async iterable of chunks) and
// returns { blake3, length } only if they are what the record names.
export async function checkFile(index, record, path, input) {
  const file = record.files?.find((f) => f.path === path);
  const expected = fileBlake3(record, path);
  if (!expected) throw new HubClientError(CODES.FileUnaddressed, `the record for ${record.repo} gives ${path} no BLAKE3`, { repo: record.repo, path });
  const seen = { length: 0 };
  const found = await index.hash(counted(input, seen));
  if (found !== expected || seen.length !== file.size) {
    throw new HubClientError(CODES.FileMismatch, `${record.repo}/${path} is not the bytes its record names`, { repo: record.repo, path, expected, found, size: file.size, length: seen.length });
  }
  return { blake3: found, length: seen.length };
}
