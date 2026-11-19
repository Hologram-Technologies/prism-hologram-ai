// Which shard of the UOR address index a model's labels belong to.
//
// The index is one row per addressed file, and it grows with every model the
// hub addresses, so the page reads one shard and never the set. That only
// finds a model if the shard the builder wrote it to is the shard the page
// looks in, which makes this rule the whole of the agreement between them. It
// lives here, imported by both, so the agreement cannot drift:
// `tools/uor-index.mjs` files a row by it and `web/hub.mjs` resolves a model
// by it. `web/swarmkey.mjs` is the same arrangement for the swarm catalog, and
// `HK-21` is the claim that keeps it honest.
//
// The bucket is the leading bits of the SHA-256 of the repository name, which
// is `tools/roll.mjs`'s rule for the roll's 2,048 shards. The same rule, with
// the width left to the data: a few hundred models want sixteen shards, not
// two thousand mostly empty ones, and `shardBits` is published in the root so
// widening it later is a rebuild rather than a second rule.
//
// Hashing is asynchronous because WebCrypto is, and WebCrypto is what both
// sides have: Node carries `crypto.subtle` as a global and so does every
// browser. One implementation, no second spelling.

// A κ-label: an algorithm, a colon, and sixty-four lowercase hex characters.
// Two axes are in use here and both are 71 bytes long, because both prefixes
// are seven characters. That coincidence is why the axis is read from the
// prefix and never from the length: a label whose axis was guessed from its
// size would be checked with the wrong hash and fail for the wrong reason.
//
// `blake3` is a tensor's address and a model's label, the axis this family
// addresses on. `sha256` is uor-addr's own canonical-form leaf, published
// beside it. Thirty-two bytes either way, which is what the kernel's accept
// rule takes.
export const UOR_LABEL = /^(blake3|sha256):[0-9a-f]{64}$/;

export const isUorLabel = (value) => typeof value === "string" && UOR_LABEL.test(value);

// The axis a label is on, or null when it is not a label at all.
export const uorAxis = (value) => (isUorLabel(value) ? value.slice(0, value.indexOf(":")) : null);

export const isBlake3Label = (value) => uorAxis(value) === "blake3";

// The sixty-four hex characters after the axis, which is what a digest check
// is given. A label that does not match the grammar has no digest rather than
// some, so a caller cannot half-check one.
export function uorDigestHex(label) {
  return isUorLabel(label) ? label.slice(label.indexOf(":") + 1) : null;
}

// The thirty-two bytes behind a κ-label, which is what the kernel's accept
// rule is given.
export function uorLabelBytes(label) {
  const hex = uorDigestHex(label);
  if (!hex) return null;
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

const hex = (buffer) => Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");

// The SHA-256 of a string, as lowercase hex.
export async function sha256Hex(text) {
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

// The shard a repository's rows are filed under, as a number.
export async function uorBucket(repo, shardBits) {
  if (!Number.isInteger(shardBits) || shardBits < 1 || shardBits > 16) {
    throw new Error(`shardBits must be 1 to 16, not ${shardBits}`);
  }
  const digest = await sha256Hex(repo.toLowerCase());
  return parseInt(digest.slice(0, 4), 16) >>> (16 - shardBits);
}

// Where that shard lives, relative to the index root. Fixed width, so the
// names sort the way the numbers do.
export const uorShardPath = (n) => `shards/${n.toString(16).padStart(3, "0")}.json`;

// The one call both sides make: the path of the only shard that can hold this
// repository's rows.
export async function uorShardOf(repo, shardBits) {
  return uorShardPath(await uorBucket(repo, shardBits));
}
