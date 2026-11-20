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
# policy. `xtask proofs` runs it on only the proof modules whose inputs changed:
# each module's earlier verification is kept in `.prism-proof/lean`, keyed by the
# BLAKE3 of its source, its import closure, the lock and the tools, and reused
# only after its attestation re-derives and binds this module's inputs now; a
# failed check verifies in full (xtask/src/proofs.rs). No proof module is an error.
# It reuses on a pull request or a merge queue run only; a main push, a tag and a
# local run verify every proof module, as `audit-bootstrap` requires. `prismpm verify` builds the holo, replaying its
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
# the shortcut. The holo is built and verified on every run. `bootstrap.yml`
# sets `PRISM_PROOF_EVENT` and caches `.prism-proof`: the recipe reads the first
# only to let `xtask proofs` reuse unchanged proof modules on a pull request or
# a merge queue run, and the second keeps each module's checked verification.
app:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s failglob
    (cd app && lexlean fmt --all --check && lexlean lock --check)
    (cd experience && lexlean fmt --check && lexlean lock --check && lexlean check)
    reuse=
    case "${PRISM_PROOF_EVENT:-}" in
      pull_request|merge_group)
        reuse=1
        ;;
    esac
    PRISM_PROOF_REUSE=$reuse cargo run -q --locked -p xtask -- proofs
    prismpm --project app verify | tee .prism-proof.log
    mkdir -p .prism-proof
    sed -n 's/^verified attestation: \([0-9a-f]\{64\}\)$/\1/p' .prism-proof.log > .prism-proof/attestation
    rm -f .prism-proof.log
    test -s .prism-proof/attestation
    holos=( app/.prism/build/*/*.holo )
    [[ ${#holos[@]} -eq 1 ]] || { echo "expected one holo, found ${#holos[@]}" >&2; exit 64; }
    node crates/conformance/tests/kernel/replay.mjs "${holos[0]}" crates/conformance/tests/kernel/claims
    # The page's decision table and card, which are generated, not committed: pages.yml
    # assembles the page from what this recipe leaves in site/.
    cargo run -q --locked -p xtask -- page --write

# The engine the page loads: this repository's crates as one WebAssembly
# module, built for the browser and put where the page asks for it. Release,
# because release is what ships; `engine-size` then fails if it is larger than
# a page should ask a visitor to fetch.
#
# Not committed: `site/app/engine/engine.wasm` is built here and by pages.yml,
# the way `site/app.holo` is, so what is published is what this workspace
# builds rather than a file someone remembered to update.
#
# EN-90: the module is the same bytes on every machine, and those bytes are the
# address the model seals (app/src/Kernel/Seal.lex.tex), which the page asks
# decision 80 about before it compiles them. A change to crates/, Cargo.lock or
# rust-toolchain.toml changes the address: `engine-seal` fails naming the new
# one, and the seal moves in the same change as the code it seals.
engine:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p site/app/engine
    just engine-build site/app/engine/engine.wasm
    cargo run -q --locked -p xtask -- engine-size site/app/engine/engine.wasm
    cargo run -q --locked -p xtask -- engine-seal site/app/engine/engine.wasm
    # What a visitor actually fetches, which is the compressed size. Reported,
    # not gated: it depends on what the server does, and the ceiling is on the
    # bytes this repository produces.
    if command -v gzip >/dev/null; then
      printf 'engine: %s bytes gzipped\n' "$(gzip -9 -c site/app/engine/engine.wasm | wc -c)"
    fi

# The page's module built so that nothing of the machine reaches its bytes.
# Found on 2026-11-19: CI's and a WSL build of one commit differed only in the
# cargo home's absolute path, which panic locations carry from every registry
# crate (/tmp/prismpm-cargo-1001 against /home/<user>/.cargo). So the registry
# and the workspace are remapped to fixed names; every other rustflags source
# and every profile override the environment could carry is cleared, since
# CARGO_ENCODED_RUSTFLAGS outranks the rest; and SOURCE_DATE_EPOCH is fixed so
# a dependency that starts reading the clock cannot make two builds differ.
# The toolchain and target are rust-toolchain.toml's; codegen-units is 1 in
# the release profile.
#
# One engine is sealed, and it is the one amd64 builds. Found on 2026-11-20:
# stable cargo mixes `rustc -vV`, which names the host, into every crate's
# metadata hash, which reaches the symbol names rustc orders items by, so an
# arm64 host builds the same tree to other bytes (x86 told by a wrapper that
# its host was aarch64 gave arm64's engine exactly). On any other host the
# build runs in the linux/amd64 variant of the SDK image prismpm.lock pins,
# under the host's emulation, with the workspace at the same path and a target
# directory of its own, so every host checks the one engine that is published.
engine-build out:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ $(uname -m) != x86_64 ]]; then
      image=$(sed -n 's/.*"sdk_image":"\([^"]*\)".*/\1/p' prismpm.lock)
      [[ -n $image ]] || { echo "engine-build: prismpm.lock names no SDK image to build the amd64 engine in" >&2; exit 64; }
      case "{{out}}" in /*) out="{{out}}" ;; *) out="$PWD/{{out}}" ;; esac
      [[ $out == "$PWD"/* ]] || { echo "engine-build: on $(uname -m) the engine is built in an amd64 container, which sees only this workspace; {{out}} is outside it" >&2; exit 64; }
      echo "engine-build: $(uname -m) host, building the engine in $image as linux/amd64"
      exec docker run --rm --platform linux/amd64 \
        --user "$(id -u):$(id -g)" \
        --tmpfs /tmp:rw,exec,nosuid,size=2g \
        --volume "$PWD:$PWD" --workdir "$PWD" \
        --env HOME=/tmp/prismpm-home --env PRISMPM_EPHEMERAL_HOME=1 \
        --env CARGO_TARGET_DIR="$PWD/target/engine-amd64" \
        "$image" just engine-build "$out"
    fi
    unset RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER CARGO_BUILD_RUSTC_WRAPPER CARGO_INCREMENTAL RUSTDOCFLAGS
    for name in $(compgen -e | grep '^CARGO_PROFILE_' || true); do unset "$name"; done
    meta=$(cargo metadata --format-version 1 --locked)
    registry=$(grep -o '"manifest_path":"[^"]*/registry/src/[^/"]*/' <<<"$meta" | sed -n '1{s/^"manifest_path":"//; s#/$##; p}')
    target=$(grep -o '"target_directory":"[^"]*"' <<<"$meta" | sed 's/^"target_directory":"//; s/"$//')
    [[ -n $registry && -n $target ]] || { echo "engine-build: cargo metadata names no registry source or target directory" >&2; exit 64; }
    # rustc applies the last prefix that matches, so the registry comes last:
    # a cargo home inside the workspace still maps to /registry.
    export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$PWD=/workspace"$'\x1f'"--remap-path-prefix=$registry=/registry"
    export SOURCE_DATE_EPOCH=0
    cargo build --release --locked --target wasm32-unknown-unknown -p hologram-ai-wasm
    cp "$target/wasm32-unknown-unknown/release/hologram_ai_wasm.wasm" "{{out}}"

# EN-90's gate: the module built twice, from two copies of the tree with two
# fresh cargo homes of different lengths and two target directories, the second
# with host flags in its environment that engine-build must clear; any byte
# that differs fails, and then the address must be the sealed one. engine.yml
# runs it on every change to the engine's inputs.
engine-twice out="":
    #!/usr/bin/env bash
    set -euo pipefail
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    for side in a bb; do
      mkdir -p "$work/$side/tree"
      git ls-files -z -co --exclude-standard | tar --null -T - -cf - | tar -xf - -C "$work/$side/tree"
    done
    (cd "$work/a/tree" && CARGO_HOME="$work/a/home" CARGO_TARGET_DIR="$work/a/target" just engine-build "$work/a.wasm")
    (cd "$work/bb/tree" && CARGO_HOME="$work/bb/cargo-home-two" CARGO_TARGET_DIR="$work/bb/target" \
      RUSTFLAGS="-C target-cpu=native" CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
      just engine-build "$work/b.wasm")
    if ! cmp "$work/a.wasm" "$work/b.wasm"; then
      echo "engine-twice: two builds of one tree differ ($(cmp -l "$work/a.wasm" "$work/b.wasm" | wc -l) bytes)" >&2
      exit 1
    fi
    echo "engine-twice: two builds are the same $(wc -c <"$work/a.wasm") bytes"
    cargo run -q --locked -p xtask -- engine-seal "$work/a.wasm"
    if [[ -n "{{out}}" ]]; then cp "$work/a.wasm" "{{out}}"; fi

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
      node bootstrap/listing-blake3.mjs "$site"
    fi
    [[ -f $site/app.holo ]] || { echo "$site has no holo: run just app" >&2; exit 64; }
    [[ -f $site/app/engine/engine.wasm ]] || { echo "$site has no engine: run just engine" >&2; exit 64; }
    node crates/conformance/tests/kernel/page-theorems.mjs "$site"
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
