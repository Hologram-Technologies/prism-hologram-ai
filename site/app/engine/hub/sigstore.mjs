// Checks a Sigstore bundle (v0.3, DSSE, keyless) with the browser's own
// WebCrypto, against trust roots the kernel's model pins. It measures; it does
// not decide. Every check becomes one fact, and the kernel's trust rule turns
// the facts into the label a page shows.
//
// Checked: the certificate chains to the pinned Fulcio intermediate; it was
// valid when the transparency log recorded the entry; the log's signed entry
// timestamp verifies with the pinned Rekor key and binds this envelope; the
// envelope's signature verifies with the certificate's key; the certificate
// names the expected workflow and OIDC issuer; the statement's subject is the
// SHA-256 of the bytes the caller holds, under the expected predicate type.

const te = new TextEncoder();
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const hexOf = (bytes) => [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
const sha256 = async (bytes) => new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));

// ---------------------------------------------------------------- DER
function node(buf, at) {
  const tag = buf[at];
  let len = buf[at + 1];
  let head = 2;
  if (len & 0x80) {
    const n = len & 0x7f;
    len = 0;
    for (let i = 0; i < n; i++) len = len * 256 + buf[at + 2 + i];
    head = 2 + n;
  }
  return { tag, start: at, body: at + head, end: at + head + len };
}
function children(buf, parent) {
  const out = [];
  for (let at = parent.body; at < parent.end;) {
    const n = node(buf, at);
    out.push(n);
    at = n.end;
  }
  return out;
}
const raw = (buf, n) => buf.subarray(n.start, n.end);
const content = (buf, n) => buf.subarray(n.body, n.end);
function oid(bytes) {
  const parts = [Math.floor(bytes[0] / 40), bytes[0] % 40];
  let v = 0;
  for (const b of bytes.subarray(1)) {
    v = v * 128 + (b & 0x7f);
    if (!(b & 0x80)) { parts.push(v); v = 0; }
  }
  return parts.join(".");
}
function time(buf, n) {
  const s = new TextDecoder().decode(content(buf, n));
  const full = n.tag === 0x17 ? (+s.slice(0, 2) < 50 ? "20" : "19") + s : s;
  return Date.UTC(+full.slice(0, 4), +full.slice(4, 6) - 1, +full.slice(6, 8), +full.slice(8, 10), +full.slice(10, 12), +full.slice(12, 14)) / 1000;
}

function certificate(der) {
  const top = node(der, 0);
  const [tbs, alg, sig] = children(der, top);
  const fields = children(der, tbs);
  const i = fields[0].tag === 0xa0 ? 1 : 0;
  const validity = children(der, fields[i + 3]);
  const spki = fields[i + 5];
  const extensions = {};
  const holder = fields.find((f) => f.tag === 0xa3);
  if (holder) {
    for (const ext of children(der, children(der, holder)[0])) {
      const parts = children(der, ext);
      extensions[oid(content(der, parts[0]))] = content(der, parts[parts.length - 1]);
    }
  }
  return {
    tbs: raw(der, tbs),
    signatureAlgorithm: oid(content(der, children(der, alg)[0])),
    signature: content(der, sig).subarray(1),
    notBefore: time(der, validity[0]),
    notAfter: time(der, validity[1]),
    spki: raw(der, spki),
    curve: oid(content(der, children(der, children(der, spki)[0])[1])),
    extensions,
  };
}
function sanUris(ext) {
  if (!ext) return [];
  const seq = node(ext, 0);
  return children(ext, seq).filter((n) => n.tag === 0x86).map((n) => new TextDecoder().decode(content(ext, n)));
}
function utf8Extension(ext) {
  if (!ext) return "";
  const n = node(ext, 0);
  return new TextDecoder().decode(n.tag === 0x0c ? content(ext, n) : ext);
}

// ---------------------------------------------------------------- ECDSA
const CURVES = { "1.2.840.10045.3.1.7": ["P-256", 32], "1.3.132.0.34": ["P-384", 48] };
const HASHES = { "1.2.840.10045.4.3.2": "SHA-256", "1.2.840.10045.4.3.3": "SHA-384" };
function p1363(der, size) {
  const seq = node(der, 0);
  const [r, s] = children(der, seq).map((n) => content(der, n));
  const out = new Uint8Array(size * 2);
  const put = (v, at) => { const t = v.subarray(Math.max(0, v.length - size)); out.set(t, at + size - t.length); };
  put(r, 0);
  put(s, size);
  return out;
}
async function ecdsa(spki, curveOid, hash, signatureDer, message) {
  const [namedCurve, size] = CURVES[curveOid] ?? [];
  if (!namedCurve) return false;
  const key = await crypto.subtle.importKey("spki", spki, { name: "ECDSA", namedCurve }, false, ["verify"]);
  return crypto.subtle.verify({ name: "ECDSA", hash }, key, p1363(signatureDer, size), message);
}
const spkiCurve = (spki) => oid(content(spki, children(spki, children(spki, node(spki, 0))[0])[1]));

// ---------------------------------------------------------------- bundle
export async function verifyBundle(bundle, bytes, policy) {
  const facts = { chain: false, time: false, log: false, signature: false, identity: false, subject: false, predicate: false };
  try {
    const leafDer = b64(bundle.verificationMaterial.certificate.rawBytes);
    const leaf = certificate(leafDer);
    const intermediate = certificate(b64(policy.fulcioIntermediate));
    facts.chain = await ecdsa(intermediate.spki, intermediate.curve, HASHES[leaf.signatureAlgorithm], leaf.signature, leaf.tbs);

    const entry = bundle.verificationMaterial.tlogEntries[0];
    const integrated = Number(entry.integratedTime);
    facts.time = integrated >= leaf.notBefore && integrated <= leaf.notAfter;

    const envelope = bundle.dsseEnvelope;
    const payload = b64(envelope.payload);
    const signature = b64(envelope.signatures[0].sig);
    const body = JSON.parse(new TextDecoder().decode(b64(entry.canonicalizedBody)));
    const logId = hexOf(b64(entry.logId.keyId));
    const set = JSON.stringify({ body: entry.canonicalizedBody, integratedTime: integrated, logID: logId, logIndex: Number(entry.logIndex) });
    const rekorKey = b64(policy.rekorKey);
    const bound = body.spec?.payloadHash?.value === hexOf(await sha256(payload))
      && body.spec?.signatures?.some((s) => s.signature === envelope.signatures[0].sig);
    facts.log = logId === policy.rekorLogId && bound
      && await ecdsa(rekorKey, spkiCurve(rekorKey), "SHA-256", b64(entry.inclusionPromise.signedEntryTimestamp), te.encode(set));

    const type = envelope.payloadType;
    const pae = te.encode(`DSSEv1 ${te.encode(type).length} ${type} ${payload.length} `);
    const message = new Uint8Array(pae.length + payload.length);
    message.set(pae);
    message.set(payload, pae.length);
    facts.signature = await ecdsa(leaf.spki, leaf.curve, "SHA-256", signature, message);

    const issuer = utf8Extension(leaf.extensions["1.3.6.1.4.1.57264.1.8"]) || utf8Extension(leaf.extensions["1.3.6.1.4.1.57264.1.1"]);
    facts.identity = sanUris(leaf.extensions["2.5.29.17"]).includes(policy.identity) && issuer === policy.issuer;

    const statement = JSON.parse(new TextDecoder().decode(payload));
    const digest = hexOf(await sha256(bytes));
    facts.subject = statement.subject?.some((s) => s.digest?.sha256 === digest) ?? false;
    facts.predicate = statement.predicateType === policy.predicateType;
    facts.signedAt = integrated;
  } catch (error) {
    facts.error = String(error?.message ?? error);
  }
  return facts;
}
