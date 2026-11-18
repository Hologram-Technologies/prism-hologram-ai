# `just vv` is the normative acceptance gate. Every other recipe is either one
# of its parts or a writer the gate checks the output of.
#
# Each check in `vv` runs exactly once. The parts are ordered so the cheapest
# failure is reported first: formatting needs no compiler, the repository gates
# compile only `xtask`, and the SDK's checks run before the workspace build.

default: vv

# The whole gate.
vv: fmt-check model template-check lint test deny app engine page-journey
    @echo "vv: the acceptance gate passed"

fmt-check:
    cargo fmt --all -- --check

# R1, R4, R5 and the bootstrap trust root: `check-model`, `audit-limits` (the
# failure register), `audit-deferral` and `audit-bootstrap` (which includes
# `audit-dependency-updates`), in one `xtask` process.
model:
    cargo run -q --locked -p xtask -- validate

# The SDK's own checks of the contract and both locks. `template check` also
# regenerates `CONFORMANCE.md` in an isolated copy through `model-write`.
template-check:
    prismpm template check
    prismpm lock check

# Clippy at `-D warnings` over every target with every feature enabled. This is
# the one place an optional feature is compiled: a rename upstream of an
# optional dependency fails nowhere else until someone turns the flag on, and
# `--all-targets` because the tests behind a flag are code too. `test` compiles
# the default feature set.
lint:
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

# The workspace suite, which includes the `repo-conformance` meta-gate (R3,
# `CM-02`, `CM-03`).
test:
    cargo test --workspace --locked

# R6: nothing shipped depends on a dev-only crate, no wildcard version
# requirement, no advisory against anything in the tree. `cargo-deny` is
# supplied by the locked SDK.
deny:
    cargo deny --all-features check

# The Hologram AI kernel, this repository's Prism application. The model must
# be canonical and locked. The modules under `app/src/Kernel` hold only what the
# guest executes; the propositions and theorems about them are in
# `app/src/Proofs`, outside the application's module graph, because the SDK
# gives that graph's Lean build a fixed 600 seconds and the arithmetic and
# induction proofs exceed it there (`PP5007`). `lexlean verify` has the Lean
# kernel and leanchecker check every proof module, each with its exact axiom
# policy; `failglob` makes an empty match an error, not a vacuous pass. `prismpm verify` builds the holo, replaying its
# acceptance vectors in the compiled guest, then has the Lean kernel and
# leanchecker check every theorem (the `KN` rows) and the generated code against the
# model. It runs the build itself, so a separate `prismpm build` would repeat it.
#
# Build and verify the holo, every time.
#
# This had a shortcut: on a pull request, if `.prism-proof` held an attestation
# for byte-identical proof inputs, the recipe stopped after `lexlean verify`
# and left the holo unbuilt, on the grounds that main proves the merge commit
# in full. It saved about seven and a half minutes a run and cost more than
# that the day a later recipe wanted the holo that was never built: every pull
# request touching no proof failed, pointing at a missing file rather than at
# the shortcut. `bootstrap.yml` still sets `PRISM_PROOF_EVENT` and still caches
# `.prism-proof`; nothing reads the first any more, and the second is kept as
# evidence of what was proved rather than as a way to skip proving it.
app:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s failglob
    (cd app && lexlean fmt --all --check && lexlean lock --check)
    (cd app && lexlean verify src/Proofs/*.lex.tex)
    prismpm --project app verify | tee .prism-proof.log
    mkdir -p .prism-proof
    sed -n 's/^verified attestation: \([0-9a-f]\{64\}\)$/\1/p' .prism-proof.log > .prism-proof/attestation
    rm -f .prism-proof.log
    test -s .prism-proof/attestation
    holos=( app/.prism/build/*/*.holo )
    [[ ${#holos[@]} -eq 1 ]] || { echo "expected one holo, found ${#holos[@]}" >&2; exit 64; }
    node crates/conformance/tests/kernel/replay.mjs "${holos[0]}" crates/conformance/tests/kernel/claims.json

# The engine the page loads: this repository's crates as one WebAssembly
# module, built for the browser and put where the page asks for it. Release,
# because release is what ships; `engine-size` then fails if it is larger than
# a page should ask a visitor to fetch.
#
# Not committed: `site/app/engine/engine.wasm` is built here and by pages.yml,
# the way `site/app.holo` is, so what is published is what this workspace
# builds rather than a file someone remembered to update.
engine:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release --locked --target wasm32-unknown-unknown -p hologram-ai-wasm
    mkdir -p site/app/engine
    cp target/wasm32-unknown-unknown/release/hologram_ai_wasm.wasm site/app/engine/engine.wasm
    cargo run -q --locked -p xtask -- engine-size site/app/engine/engine.wasm
    # What a visitor actually fetches, which is the compressed size. Reported,
    # not gated: it depends on what the server does, and the ceiling is on the
    # bytes this repository produces.
    if command -v gzip >/dev/null; then
      printf 'engine: %s bytes gzipped\n' "$(gzip -9 -c site/app/engine/engine.wasm | wc -c)"
    fi

# The page as it is published, driven end to end: the holo this workspace
# built, the engine it built, the page assembled from them, and a real
# browser loading a container through the verified kernel.
#
# `vv` runs it after `app` and `engine`, because it needs what both make.
# `pages.yml` runs this same recipe between its build and its deploy, on
# the tree it is about to publish: one script, two callers, so the page
# that passes here is the page that goes out.
page-journey assembled="":
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s failglob
    site="{{assembled}}"
    if [[ -z $site ]]; then
      holos=(app/.prism/build/*/*.holo)
      [[ ${#holos[@]} -eq 1 ]] || { echo "expected one holo, found ${#holos[@]}" >&2; exit 64; }
      site=target/page-journey
      rm -rf "$site"
      mkdir -p "$site"
      cp -r site/. "$site/"
      cp "${holos[0]}" "$site/app.holo"
      cp -r try/vendor "$site/vendor"
      rm -rf "$site/vendor/prism-code/fixtures"
      node bootstrap/page.mjs "$site"
      node bootstrap/precache.mjs "$site"
    fi
    [[ -f $site/app.holo ]] || { echo "$site has no holo: run just app" >&2; exit 64; }
    [[ -f $site/app/engine/engine.wasm ]] || { echo "$site has no engine: run just engine" >&2; exit 64; }
    node crates/conformance/tests/gpu/journey.mjs "$site"

# Only the meta-gate, while iterating on a scenario. `test` already runs it.
bdd:
    cargo test --locked -p repo-conformance

fmt:
    cargo fmt --all

# Regenerate everything the model owns: CONFORMANCE.md.
model-write:
    cargo run -q --locked -p xtask -- check-model --write

# R3's first step in one command: a register row, its scenario, a failing test
# named for the ID, and the regenerated CONFORMANCE.md.
#
#     just new-capability CX-01 build parsing "A parsed document round-trips"
new-capability id level suite statement:
    cargo run -q --locked -p xtask -- new-capability {{quote(id)}} {{quote(level)}} {{quote(suite)}} {{quote(statement)}}

# R5's first step, through R3's: a row in model/failures.toml, its negative
# scenario, a failing test, and the regenerated CONFORMANCE.md.
#
#     just new-failure FL-01 kernel-client FetchError Timeout engine "The download stalls past its deadline"
new-failure id crate type variant suite condition:
    cargo run -q --locked -p xtask -- new-failure {{quote(id)}} {{quote(crate)}} {{quote(type)}} {{quote(variant)}} {{quote(suite)}} {{quote(condition)}}
