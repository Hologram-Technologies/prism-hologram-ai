# Implementation plan: hologram-ai as a Prism app

Companion to [REBUILD_PROMPT.md](REBUILD_PROMPT.md). Where the two differ, this plan wins: it was written after checking the ground.

Status on 2026-11-12: Phase 0 is partly done (everything that could be read is read). Nothing is built. No repository exists yet.

## Test rule for every phase

No capability is implemented before its test exists and fails. Each capability gets, in this order: a registered id, a scenario, a failing test, the implementation, a planted defect that the test catches, and a note recording that defect. A gate that has never been seen to fail is not counted as a gate.

Seven kinds of test are used. Each phase below names which it adds.

| Kind | What it shows | Tool |
|---|---|---|
| Proof | A stated property holds for every input | Lean, through LexLean |
| Example | A request gives exactly this response | Acceptance vectors carried in the holo |
| Pinned | A theorem's statement has not been quietly changed | Rust test named for the capability id |
| Unit and property | One function is right, including on generated inputs | `cargo test`, `proptest` |
| Differential | Two paths that must agree do agree | Rust tests comparing outputs byte for byte |
| Oracle | We match an authority we did not write | ONNX Runtime, HuggingFace `tokenizers`, BLAKE3 vectors, ONNX node corpus |
| Journey and measurement | The whole thing works and is fast in a real browser | Playwright in headless Chromium; a benchmark harness |

## Phase 0. Verify the ground

### Confirmed by reading

| Question | Finding | Evidence |
|---|---|---|
| Which SDK does the template pin? | `prismpm-sdk-candidate@sha256:cd3b47…`, Lean 4.32.1, LexLean language 1.2. `prism-code` on npm is 0.7.2. | `template.lock`, `app/lexlean.toml` |
| Guest limits | Profile `prismpm/text-application/1`. The app declares its own request, response and allocation maximums; the template's are 384, 256 and 512 bytes. Requests must be small. | `app/src/Contract.lex.tex` line 158 |
| Does the page allow HuggingFace? | No. `connect-src` is its own origin plus four fixed ones; `worker-src 'self'`. The policy is a meta tag in `site/page.template.html`, which `frame.lock` pins. | template files |
| Does the page budget allow the engine? | No. 122,880 bytes and 28 requests in total at load, Lighthouse 100 on mobile. It measures the root page only. | `lighthouserc.json`, pinned by `frame.lock` |
| May the repository use a Git dependency? | No. `cargo deny` allows crates.io only and denies unknown Git sources; rule R6 says the same. | `deny.toml`, `AGENTS.md` |
| Are the hologram crates on crates.io? | No. None of the nine names checked is published. | crates.io API |
| What does hologram-ai actually pin? | A commit in a private, non-fork repository (`15d155b`, 2026-08-26, "Canonical fixed-order f32 dot and sum"), not the public v0.10.0 the comment names. That commit is not in the public repository. | `Cargo.toml`, GitHub API |
| Latest public runtime | Tag v0.13.0; head `b6d9a76` is 0.13.1, untagged. | GitHub API |
| ONNX corpus coverage today | 10 single-operator cases are pinned, not the corpus. | `onnx_spec_conformance.rs` |
| Do ONNX weights go in by κ? | No. The importer emits inline bytes or a native memory map, never an external κ parameter. | `hologram-ai-onnx` sources |
| Existing tests to carry over | 454 Rust test functions, 56 integration test files, 63 feature files, 6 TypeScript test files. | source tree |
| This machine | Rust nightly 1.98 with the wasm32 target, wasm-pack, Node 22, pnpm, Docker and WSL are present. `just`, `wasm-opt` and `wasmtime` are missing. | local check |

### What the findings change

1. **The engine can live in the Prism repository only once its runtime is on crates.io.** Today the runtime is a Git dependency, and the gate forbids those. See Decision 1.
2. **The product page sits beside the standard Prism page, not inside it.** A second document under `site/app/` carries its own content-security policy and its own service worker. The root page stays the unmodified Prism page: it shows the kernel, its fingerprint and its examples. No pinned file is touched, and the page budget is measured on the root page only.
3. **Kernel requests are compact.** Every decision takes a handful of numbers. The operator preflight takes a bitmap of operators present, not a list.
4. **ONNX coverage starts near zero as evidence.** Ten cases is a smoke test. The first full corpus run sets the real baseline.

### Decision 1, decided 2026-11-12: one repository

The owner chose one repository for kernel, engine and page.

Consequence: the template's rule R6 and its `cargo deny` policy allow crates.io dependencies only, so the hologram runtime crates (and `holospaces`, if still needed) must be published to crates.io before the engine can pass the gate. That is a public release that cannot be withdrawn, and what is published must include the privately pinned numeric change if the engine needs it (spike S5). Publishing is its own step, with its own explicit approval, before Phase 2 can close. Loosening `deny.toml` to admit a Git source instead would break R6 and is not an option.

Phase 1 does not depend on this and starts first.

### What LexLean specifies of the engine

The engine itself cannot be a Prism app. A holo's code is generated from LexLean, imports nothing, keeps no state, and has no floating-point type (the `prism-app` lexicon holds booleans, naturals, fixed-width integers, bytes and lists; a search for "float" returns nothing). It cannot link the hologram runtime, use threads or SIMD, or hold a model. Writing inference in it would give up the runtime and the latency goal together.

LexLean specifies the engine in two ways instead.

1. **Its decisions**, proved: the eleven in Phase 1. The engine obeys them and never decides for itself.
2. **Its integer-exact parts**, as executable reference specifications. This is Phase 1b and it is required, not optional. Each is a small proved function in the holo that the Rust engine is tested against, example for example:
   - protobuf varint decoding and field-length bounds (the ONNX walker's arithmetic);
   - the safetensors header length and tensor range arithmetic;
   - the resident cache's position-to-slot arithmetic;
   - the stage partition over layers;
   - argmax and partial top-k selection over integer scores;
   - the int8 dot product, as the reference the quantised kernel must equal.

Floating-point inference stays tested against ONNX Runtime, never proved.

What "formally verified engine" therefore means here, stated so nobody over-reads it: every decision is proved; every integer-exact algorithm has a proved reference that the engine is tested equal to; the Rust code itself is not proved. Proving the Rust would need a Rust verifier, which is outside LexLean.

### Engine quality bars

**Precise.**

- One answer on every device. Floating-point sums and dot products run in one fixed order on every target, so wasm and native agree bit for bit. This depends on the privately pinned numeric change reaching the published runtime (spike S5). Test: the same model and prompt give identical logits natively and in wasm.
- Integer paths are exact. The int8 kernels equal the proved LexLean reference bit for bit, on generated inputs.
- Quantisation loss is measured, never assumed. int8 against full precision is reported per model as cosine similarity of logits; below 0.99 the tier is not offered for that model.
- Float results match ONNX Runtime within a stated tolerance per operator.

**Lean.** Budgets are recorded at the first green build and may not grow by more than 5 percent without a written reason: engine wasm size, number of crates in the dependency tree, lines of Rust and TypeScript, bytes loaded before first paint on the app page.

**Low latency.** The gates in Phase 5.

### Spikes still to run (each is a test, kept afterwards)

| Spike | Question | Test that answers it |
|---|---|---|
| S1 | Does the threaded wasm build of runtime 0.13 run the decode kernels? | `wasm-pack test --node` over the source's `wasm-test` kernels, re-pinned |
| S2 | Does a document under `site/app/` with its own service worker become cross-origin isolated on GitHub Pages and reach HuggingFace? | Playwright probe asserting `crossOriginIsolated === true` and a ranged fetch of a public file |
| S3 | What does one kernel call cost in the browser? | Benchmark: fresh instance, 64-byte request, median of 1,000 |
| S4 | What share of the ONNX node corpus passes today? | Run every case at a pinned ONNX release; write the share to the ledger |
| S5 | Does the public runtime contain the privately pinned numeric change? | Diff the two; run the hermetic chat reference on both |

Gate: findings note committed; Decision 1 answered; S1 to S5 green or their failure explained.

## Phase 1. Kernel: specify and prove

Independent of Decision 1. Can run in parallel with Phase 2.

Steps:

1. Approve the repository name and public visibility, then create it from `UOR-Foundation/template`. Clone with `core.autocrlf=false`.
2. Fix the request format, following the template's own model: a fixed binary layout. Byte 0 names the decision; each field after it is an unsigned big-endian number of fixed width; each decision has one exact length, and any other length answers "invalid request". Responses are short UTF-8 text. Set the three maximums to fit the largest request.
3. Show the eleven decisions' properties as plain sentences for one approval.
4. Build the decisions in this order, simplest first, so the toolchain is learned on easy ground: hash admission (5), tensor range (6), decode bucket (8), shape binding (11), generation bound (9), context trimming (7), quantisation tier (4), resource projection (3), operator preflight (10), preflight verdict (2), journey state machine (1).

Tests added, per decision:

| Kind | Minimum |
|---|---|
| Scenario | One feature file, written first |
| Proof | Every property listed in the prompt, each a named theorem with its exact axiom set |
| Example | One per verdict branch, one at each numeric boundary, one malformed request; across the app, unique requests and one of exactly the allocation maximum |
| Pinned | One test named for the id, fixing the theorem's statement and axioms |
| Planted defect | One recorded mutation (a flipped comparison, an off-by-one) that the examples or the proof reject |

Gate: `build --check` clean; `verify --lean` passes every theorem; `axioms` exact; every example replays; `just vv` green on GitHub Actions (the SDK image is 5.5 GB, so build there, verify here).

## Phase 2. Engine core: chat on runtime 0.13

Steps:

1. Re-pin the runtime to one public revision; resolve S5 first. Pin `holospaces` to a revision too; the source tracks a branch.
2. Replace the byte decode loop with the addressed loop, in-place cache writes and scalar-mask attention.
3. Threads and two SIMD binaries; pool spawned during load.
4. Paged loading, verify-once at download, warm start from the stored cone.
5. Leases for parked conversations and the system-prompt prefill; partial top-k sampling.
6. Remove what is cut: the extension, the superseded decode paths.

Tests added:

| Kind | Test |
|---|---|
| Carried over | Every existing unit, feature and structural test for the crates kept; none deleted without a named reason |
| Differential | Addressed loop equals byte loop, byte for byte, over 64 steps |
| Differential | Staged execution equals monolithic execution |
| Differential | Pooled kernels equal serial kernels, bit for bit |
| Differential | Paged load equals resident load on outputs |
| Oracle | Logits match ONNX Runtime within tolerance on SmolLM2-135M; token ids match HuggingFace `tokenizers`; BLAKE3 known answers |
| Unit | Partial top-k returns the same token as the full sort, for fixed seeds |
| Unit | A leased cache is not mutated by a later step; an unleased one is moved in place |
| Kernel conformance | Wherever the engine mirrors a kernel decision (the per-token stop check), it gives the kernel's answer on every acceptance vector |
| Negative | Every typed failure has a test: missing κ, corrupt κ, late worker registration, heap exhausted |
| wasm | The threaded and SIMD binaries pass the kernel tests under Node |

Gate: all of the above green natively and in wasm; no test from the source dropped silently.

## Phase 3. ONNX import

Steps:

1. Streaming protobuf walker in Rust that skips weight bodies.
2. Importer emits external κ parameters; typed fields normalised before hashing.
3. Ranged per-tensor streaming for embedded and external-data weights.
4. Shape binding, bucketing and the compile cache keyed by structure κ and binding.
5. Generic runner ports; label carry for stateful decoder loops.
6. Full corpus run; choose and pin the real-model set; desugar the highest-value missing operators.

Tests added:

| Kind | Test |
|---|---|
| Unit | Walker returns the same tensors, offsets and lengths as the full protobuf decoder on every fixture |
| Property | Generated protobuf inputs: the walker never reads past the buffer, never panics, and rejects truncated input, over-long varints and lengths beyond the file |
| Unit | The walker reads no byte inside a weight body (a counting reader asserts it) |
| Differential | Streamed weightless import gives the same outputs as the source's whole-file import, on every fixture |
| Differential | Typed-field weights and raw-byte weights of the same tensor get the same κ |
| Oracle | Outputs match ONNX Runtime within tolerance on every fixture and every model in the pinned set |
| Oracle | Node corpus share at a pinned ONNX release, written to the ledger; must not fall below S4 |
| Unit | Second import of the same model performs no compile (a counter asserts it); a new shape inside a bucket performs none either |
| Differential | Stateful loop by label equals the same loop by bytes, and hashes nothing after step one |
| Negative | Unsupported operator, unbound dimension, unsupported dtype, malformed file: each rejected by name with zero weight bytes read |
| Kernel conformance | The engine's supported-operator table equals the table in the kernel's model |

Gate: all green; coverage numbers published.

## Phase 4. Page and journeys

Steps:

1. `site/app/`: plain TypeScript, one stylesheet, three workers, four views, one import box.
2. Every decision routed through the verified holo; the engine wasm admitted only when its hash passes the kernel's check.
3. Token streaming batched to animation frames.

Tests added:

| Kind | Test |
|---|---|
| Unit | Chat template rendering, stream assembly, format detection, error formatting |
| Journey | Chat, hermetic: download the fixture, three-message handshake, transcript equals the committed reference at temperature 0 |
| Journey | Warm turn: the second send materialises nothing |
| Journey | ONNX, hermetic: drop the fixture, verdict shown, run, output equals the committed reference |
| Journey | ONNX rejection: unsupported operator named, no weight request made (asserted on the network log) |
| Journey | Re-import: no compile on the second drop |
| Journey | Isolation canary: the page is cross-origin isolated and a cross-origin file without a resource-policy header still loads |
| Journey | Offline: after one visit, reload with the network off and chat with the cached model |
| Journey | Kernel in the loop: with the holo replaced by a refusing stub, no stage advances |
| Journey | Tampered engine: a wasm with one flipped byte is refused |
| Gate | Root page still passes the template's Lighthouse run unchanged |

Gate: all journeys green in headless Chromium on every push.

## Phase 5. Latency

Steps:

1. Build the harness first: one fixed protocol, one named device, results written to the ledger.
2. Record the baseline before any tuning.
3. Apply the levers in the prompt's order. Measure after each. Keep a lever only if its number moves.

Tests added:

| Kind | Test |
|---|---|
| Measurement | Cold and warm time to first token; tokens per second at context 128, 2,048 and 8,192; load time; peak heap; dispatched and skipped kernels |
| Measurement | ONNX: drop to verdict, drop to first result, repeat run with the same input, repeat run with a changed input, reopen |
| Gate | SmolLM2-135M int8, 4-core laptop, Chrome: warm first token under 300 ms; at least 40 tokens per second at context 2,048 |
| Gate | ONNX preflight verdict under one second for a local file, on every model in the pinned set |
| Regression | On the CI runner, relative floors: no metric more than 10 percent worse than the last recorded run |
| Differential | Every lever leaves outputs byte-identical, except quantisation tiers, which are reported as lossy |

Gate: targets met on the named device, or the measured number and its bound reported and the owner asked.

## Phase 6. Ship

Steps: push; template gate and build workflow green; Pages live; live-model journey on the scheduled matrix.

Tests added: the live journey against SmolLM2-135M and Qwen2.5-0.5B at pinned revisions; replay of the live holo's examples from this machine; a scheduled check that every pinned upstream still resolves.

Gate and final report: live URL, repository, holo fingerprint and examples replayed, theorems proved versus behaviour tested, measured performance on the named device, ONNX coverage.

## Order and parallelism

Phase 1 and Phase 2 share nothing and can run at the same time. Phase 3 needs Phase 2. Phase 4 needs 1, 2 and 3. Phase 5 needs 4. Phase 6 needs 5.

## Risks, most serious first

1. **The private runtime pin.** The source depends on a commit that is not public. If the public runtime lacks it, the hermetic reference may not reproduce. S5 settles this before anything else in Phase 2.
2. **Cross-origin isolation on Pages beside the frame's own service worker.** Two workers, nested scopes. S2 settles it; if it fails, the app page needs a different host.
3. **The latency targets are estimates.** No browser measurement of this engine at this runtime version exists.
4. **ONNX coverage.** The gap between ten cases and "any model" is unknown until S4 runs.
5. **Kernel request size.** The text profile is byte-bounded and stateless; a decision that needs more than a few hundred bytes must be split and folded by the host.
