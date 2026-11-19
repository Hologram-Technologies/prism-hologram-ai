// GP-27's signer: a Sigstore-shaped chain minted for the test, so the page's
// vendored verifier (site/app/engine/hub/sigstore.mjs) checks a real
// signature over the fixture's index root, against a policy that names this
// chain's keys, with no flag in the page that skips the check.
//
// Why minted rather than recorded: the fixture's root names generated models
// the live hub has never seen, and a recorded bundle signs only the live root.
// The verifier reads its trust anchors from the policy it is given, so a
// policy naming a test intermediate and a test log key exercises the same code
// the page runs against the hub's own anchors.
//
// What is minted: a P-384 intermediate certificate (the policy's
// fulcioIntermediate), a P-256 leaf certificate it signs, carrying the
// policy's identity as a URI SAN and its issuer as Fulcio's extension
// 1.3.6.1.4.1.57264.1.8; a P-256 log key (the policy's rekorKey and
// rekorLogId) signing the entry timestamp; and a DSSE envelope whose in-toto
// statement names the root's SHA-256 under the index predicate type.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const subtle = globalThis.crypto.subtle;
const te = new TextEncoder();
const b64 = (bytes) => Buffer.from(bytes).toString("base64");
const sha256Hex = (bytes) => createHash("sha256").update(bytes).digest("hex");

// ---------------------------------------------------------------- DER
const concat = (parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};
const length = (n) => (n < 0x80 ? [n] : n < 0x100 ? [0x81, n] : [0x82, n >> 8, n & 0xff]);
const tlv = (tag, ...parts) => {
  const body = concat(parts);
  return concat([Uint8Array.from([tag, ...length(body.length)]), body]);
};
const seq = (...parts) => tlv(0x30, ...parts);
const set = (...parts) => tlv(0x31, ...parts);
const integer = (bytes) => {
  let at = 0;
  while (at < bytes.length - 1 && bytes[at] === 0) at++;
  const trimmed = bytes.subarray(at);
  return tlv(0x02, trimmed[0] & 0x80 ? concat([Uint8Array.of(0), trimmed]) : trimmed);
};
const oid = (dotted) => {
  const parts = dotted.split(".").map(Number);
  const out = [40 * parts[0] + parts[1]];
  for (const value of parts.slice(2)) {
    const digits = [value & 0x7f];
    for (let v = Math.floor(value / 128); v > 0; v = Math.floor(v / 128)) digits.unshift((v & 0x7f) | 0x80);
    out.push(...digits);
  }
  return tlv(0x06, Uint8Array.from(out));
};
const utcTime = (seconds) => {
  const d = new Date(seconds * 1000);
  const two = (n) => String(n).padStart(2, "0");
  const text = `${two(d.getUTCFullYear() % 100)}${two(d.getUTCMonth() + 1)}${two(d.getUTCDate())}${two(d.getUTCHours())}${two(d.getUTCMinutes())}${two(d.getUTCSeconds())}Z`;
  return tlv(0x17, te.encode(text));
};
const commonName = (name) => seq(set(seq(oid("2.5.4.3"), tlv(0x0c, te.encode(name)))));
const extension = (id, value) => seq(oid(id), tlv(0x04, value));

const ECDSA_SHA256 = "1.2.840.10045.4.3.2";
const ECDSA_SHA384 = "1.2.840.10045.4.3.3";

// WebCrypto signs ECDSA as r || s; a certificate and Sigstore carry it as DER.
const derSignature = (p1363) => {
  const half = p1363.length / 2;
  return seq(integer(p1363.subarray(0, half)), integer(p1363.subarray(half)));
};
async function sign(privateKey, hash, message) {
  return derSignature(new Uint8Array(await subtle.sign({ name: "ECDSA", hash }, privateKey, message)));
}
const keyPair = (namedCurve) => subtle.generateKey({ name: "ECDSA", namedCurve }, true, ["sign", "verify"]);
const spkiOf = async (key) => new Uint8Array(await subtle.exportKey("spki", key.publicKey));

async function certificate({ serial, issuer, subject, issuerKey, hash, algorithm, publicKey, from, until, extensions = [] }) {
  const tbs = seq(
    tlv(0xa0, integer(Uint8Array.of(2))),
    integer(Uint8Array.of(serial)),
    seq(oid(algorithm)),
    commonName(issuer),
    seq(utcTime(from), utcTime(until)),
    commonName(subject),
    publicKey,
    ...(extensions.length ? [tlv(0xa3, seq(...extensions))] : []),
  );
  const signature = await sign(issuerKey, hash, tbs);
  return seq(tbs, seq(oid(algorithm)), tlv(0x03, Uint8Array.of(0), signature));
}

// The hub's own trust policy, as this repository pins it.
export const pinnedPolicy = () =>
  JSON.parse(readFileSync(new URL("../../../../site/app/engine/hub/policy.json", import.meta.url), "utf8"));

// A chain of its own: an intermediate, a log key, and the policy that names
// them, keeping the pinned policy's identity, issuer and predicate types.
export async function mintTrust() {
  const intermediate = await keyPair("P-384");
  const log = await keyPair("P-256");
  const now = Math.floor(Date.now() / 1000);
  const intermediateDer = await certificate({
    serial: 1, issuer: "test-intermediate", subject: "test-intermediate", issuerKey: intermediate.privateKey,
    hash: "SHA-384", algorithm: ECDSA_SHA384, publicKey: await spkiOf(intermediate), from: now - 86400, until: now + 86400,
  });
  const logSpki = await spkiOf(log);
  const policy = {
    ...pinnedPolicy(),
    fulcioIntermediate: b64(intermediateDer),
    rekorKey: b64(logSpki),
    rekorLogId: sha256Hex(logSpki),
  };
  return { intermediate, log, policy };
}

// A Sigstore bundle over `bytes` (the index root), signed through `trust`'s
// chain. `identity` overrides the leaf's URI SAN.
export async function signRoot(trust, bytes, { identity = trust.policy.identity } = {}) {
  const leaf = await keyPair("P-256");
  const integratedTime = Math.floor(Date.now() / 1000);
  const leafDer = await certificate({
    serial: 2, issuer: "test-intermediate", subject: "", issuerKey: trust.intermediate.privateKey,
    hash: "SHA-384", algorithm: ECDSA_SHA384, publicKey: await spkiOf(leaf), from: integratedTime - 60, until: integratedTime + 600,
    extensions: [
      extension("2.5.29.17", seq(tlv(0x86, te.encode(identity)))),
      extension("1.3.6.1.4.1.57264.1.8", tlv(0x0c, te.encode(trust.policy.issuer))),
    ],
  });
  const statement = {
    _type: "https://in-toto.io/Statement/v1",
    subject: [{ name: "root.json", digest: { sha256: sha256Hex(bytes) } }],
    predicateType: trust.policy.indexPredicateType,
    predicate: {},
  };
  const payload = te.encode(JSON.stringify(statement));
  const payloadType = "application/vnd.in-toto+json";
  const pae = te.encode(`DSSEv1 ${te.encode(payloadType).length} ${payloadType} ${payload.length} `);
  const sig = b64(await sign(leaf.privateKey, "SHA-256", concat([pae, payload])));
  const body = b64(te.encode(JSON.stringify({
    apiVersion: "0.0.1",
    kind: "dsse",
    spec: { payloadHash: { algorithm: "sha256", value: sha256Hex(payload) }, signatures: [{ signature: sig }] },
  })));
  const logId = sha256Hex(new Uint8Array(await subtle.exportKey("spki", trust.log.publicKey)));
  const logIndex = 1;
  const timestamp = JSON.stringify({ body, integratedTime, logID: logId, logIndex });
  const entry = {
    logIndex: String(logIndex),
    logId: { keyId: b64(Buffer.from(logId, "hex")) },
    kindVersion: { kind: "dsse", version: "0.0.1" },
    integratedTime: String(integratedTime),
    inclusionPromise: { signedEntryTimestamp: b64(await sign(trust.log.privateKey, "SHA-256", te.encode(timestamp))) },
    canonicalizedBody: body,
  };
  return Buffer.from(JSON.stringify({
    mediaType: "application/vnd.dev.sigstore.bundle.v0.3+json",
    verificationMaterial: { certificate: { rawBytes: b64(leafDer) }, tlogEntries: [entry] },
    dsseEnvelope: { payload: b64(payload), payloadType, signatures: [{ sig }] },
  }));
}
