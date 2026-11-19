# Vendored hub client

The page reads a model's hub record through the hub's own client, so the
rules for trusting a record are the hub's, reused by agreement rather than
rewritten here (Q_ENGINE_PLAN.md, decision 10). These files are byte-for-byte
copies; nothing here is rebuilt or edited.

- Repository: <https://github.com/Hologram-Technologies/hub>
- Commit: `ec26789fe4a7f936abd0f0c207b9e1a556d8c33c` (2026-11-18)
- `hub-client.mjs`, from `web/hub-client.mjs`: BLAKE3
  `cdcc05c1c331a8fc2f5b6748ae84891ec13d82a627a2844c094588c8da8ba49a`
- `sigstore.mjs`, from `web/sigstore.mjs`: BLAKE3
  `0f25ca52fed3967f36d291179b35b36f10ed39f60c46be948b94037a369f3afa`, the
  value `hub-client.mjs` pins as `SIGSTORE_BLAKE3`

`hub-client.mjs` loads the hub's own hasher (`piece-hash.mjs`,
`piece-hash.wasm`, `pieces.mjs`) only when its caller brings no hash. The page
always passes its own BLAKE3, the one vendored with prism-code, so those files
are not carried.

## The trust policy

`policy.json` is this repository's pinned copy of the hub model's trust
anchors, its screen 3 `trust.policy` in `app/tools/site_spec.py` at the same
commit (BLAKE3 of that file
`1af2787ae532e35a5534212427ac0cf6c5e930f29f98a7197ef251fb0f509ef3`): the
workflow identity, the OIDC issuer, the predicate types, the Sigstore
intermediate certificate, Rekor's public key and its log id. The values are
copied unchanged; `policy.json` itself has BLAKE3
`8c42c7ba1cddb97ce014e71eb573b7ea85a06313f919614b911773ca81fca349`.

The page carries the policy rather than fetching it: a policy served by the
hub would be vouched for by the thing it is meant to check. A change of the
hub's anchors reaches the page as a reviewed change to this file.
