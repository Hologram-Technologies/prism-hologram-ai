// GP-27, without a browser: the hub's own client, vendored at
// site/app/engine/hub/, checks an index root by its Sigstore signature against
// a policy, a record by the BLAKE3 the root names, and a file by the BLAKE3
// and size its record names, and refuses each failure by its code. The chain
// is minted by hub-sign.mjs. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/hub-check.mjs <repository root>

import { mintTrust, signRoot } from "./hub-sign.mjs";
import { CODES, HubClientError, checkFile, openIndex, readRecord } from "../../../../site/app/engine/hub/hub-client.mjs";
import { blake3Hex } from "../../../../try/vendor/prism-code/src/blake3.mjs";

const te = new TextEncoder();
const hash = async (input) => {
  if (input instanceof Uint8Array) return blake3Hex(input);
  const parts = [];
  for await (const chunk of input) parts.push(chunk);
  return blake3Hex(Buffer.concat(parts));
};

// A hub of one model: its file, its record, and a root naming the record.
const repo = "fixture-owner/fixture-model";
const file = te.encode("the bytes of one model file");
const record = te.encode(JSON.stringify({
  format: "hologram.hub.release/v4",
  repo,
  revision: "0".repeat(40),
  files: [{ path: "model.onnx", size: file.length, blake3: blake3Hex(file) }, { path: "config.json", size: 2 }],
}));
const rootOf = (entries) => te.encode(JSON.stringify({ format: "hologram.hub.index/v2", records: entries }));
const root = rootOf([{ repo, revision: "0".repeat(40), blake3: blake3Hex(record) }]);

// A fetch over a map of paths; anything else is a 404.
const served = (files) => async (url) => {
  const bytes = files[new URL(url).pathname];
  return bytes ? new Response(bytes) : new Response(null, { status: 404 });
};
const base = "https://hub.test/data/index/";
const hubOf = ({ rootBytes = root, bundle, recordBytes = record }) => served({
  "/data/index/root.json": rootBytes,
  "/data/index/index.sigstore.json": bundle,
  [`/data/index/models/${repo}.json`]: recordBytes,
});

// What a call ends in: "ok", or the code it is refused by and the facts it names.
async function outcome(call) {
  try {
    await call();
    return { code: "ok" };
  } catch (error) {
    if (!(error instanceof HubClientError)) return { code: "thrown", message: String(error?.message ?? error) };
    return { code: error.code, failed: error.failed ?? null };
  }
}

const trust = await mintTrust();
const foreign = await mintTrust();
const bundle = await signRoot(trust, root);
const open = (hub, policy = trust.policy) => openIndex(base, policy, { fetch: hub, hash });
const index = await open(hubOf({ bundle }));
const read = await readRecord(index, repo);

const results = {
  good: { code: "ok", repo: read.repo, signedAt: index.facts.signedAt, checked: await checkFile(index, read, "model.onnx", file) },
  // The root signed by an intermediate and a log key the policy does not name.
  foreignKey: await outcome(async () => open(hubOf({ bundle: await signRoot(foreign, root) }))),
  // The policy's intermediate, another log key.
  foreignLog: await outcome(async () => open(hubOf({ bundle: await signRoot({ ...trust, log: foreign.log }, root) }))),
  // The policy's chain, another workflow's identity.
  otherIdentity: await outcome(async () => open(hubOf({ bundle: await signRoot(trust, root, { identity: "https://github.com/someone/else/.github/workflows/x.yml@refs/heads/main" }) }))),
  // A root changed after it was signed.
  changedRoot: await outcome(async () => open(hubOf({ rootBytes: rootOf([{ repo, revision: "1".repeat(40), blake3: blake3Hex(record) }]), bundle }))),
  // The hub's own pinned policy, which names none of this chain.
  pinnedPolicy: await outcome(async () => open(hubOf({ bundle }), { ...(await import("./hub-sign.mjs")).pinnedPolicy() })),
  noBundle: await outcome(async () => openIndex(base, trust.policy, { fetch: served({ "/data/index/root.json": root }), hash })),
  unknownRepo: await outcome(async () => readRecord(index, "nobody/nothing")),
  changedRecord: await outcome(async () => readRecord({ ...index, fetch: hubOf({ bundle, recordBytes: te.encode(new TextDecoder().decode(record).replace("v4", "v5")) }) }, repo)),
  unaddressedRecord: await outcome(async () => readRecord({ ...index, root: { records: [{ repo }] } }, repo)),
  unreachableRecord: await outcome(async () => readRecord({ ...index, fetch: served({}) }, repo)),
  changedFile: await outcome(async () => checkFile(index, read, "model.onnx", te.encode("the bytes of one model filf"))),
  shortFile: await outcome(async () => checkFile(index, read, "model.onnx", file.subarray(1))),
  unaddressedFile: await outcome(async () => checkFile(index, read, "config.json", te.encode("{}"))),
  codes: Object.keys(CODES),
};
console.log(JSON.stringify(results));
