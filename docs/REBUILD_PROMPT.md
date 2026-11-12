# Prompt: rebuild hologram-ai as a Prism app

Paste everything below the line into a fresh agent session with the `prism` skill, the `lexlean` and `prism` MCP servers, `gh`, Rust, and Node 22 available.

---

## Mission

Rebuild `Hologram-Technologies/hologram-ai` as a new Prism app in a new repository under `Hologram-Technologies`. It must be fully functional, lean, and very low latency. Its decision logic must be specified in LexLean and proved by Lean. Its inference must run on the `Hologram-Technologies/hologram` runtime, used the fast way.

It has two front doors of equal standing: HuggingFace safetensors models for chat, and ONNX models of any kind, imported and run in the browser with one gesture.

Work in phases. Each phase ends at a gate. Do not start a phase until the previous gate is green. Share progress at each gate in three lines or fewer.

## Why, how, what of the source

**Why.** Anyone should be able to run an open model privately in a browser tab, with no server, and be able to check every byte and every step. The tab's limits on memory, storage and network should not be the model's limits.

**How.** One idea: every piece of content is named by the BLAKE3 hash of its bytes (its κ-label). That single rule gives integrity (re-hash and compare), deduplication (same hash, same blob) and compute reuse (a result whose label is already resident is not recomputed). Weights stream tensor by tensor into a κ-store. The compiled archive holds structure and κ-bindings, no weights. Loading is resolving labels against the store. Development is strictly ordered: prose model, typed model, Gherkin feature, test, external oracle, with a gate that forbids asserting more than the evidence shows.

**What.** A Rust workspace of ten crates (about 71,000 lines) plus a React web app (about 5,500 lines of TypeScript) and a wasm binding. One user journey: Download from HuggingFace, Compile to a weightless `.holo`, Run, Chat. Families: Llama and Qwen2 style decoders, parametric over the model's own `config.json`. Canonical model: SmolLM2-135M-Instruct. Deployed on GitHub Pages.

Read these before writing anything: `docs/conceptual-model/00-source.md` to `03`, `docs/architecture/ARCHITECTURE.md`, `docs/notes/throughput-latency-analysis.md`, ADR-0018 (threaded wasm) and ADR-0019 (resident KV), `model/dictionary.toml`, `features/suites/`.

Note one stale claim in the source docs: they say "there is no KV-cache". ADR-0019 superseded that. Production decode uses a resident KV cache carried by label.

Note the state of ONNX in the source. A native importer exists (`crates/hologram-ai-onnx`, about 3,400 lines, roughly a hundred operator names mapped, verified against ONNX Runtime and a pinned subset of the official node-test corpus). The browser does not have it: `apps/web/src/download.worker.ts` refuses ONNX downloads because the file cannot be streamed into the κ-store, and the wasm `compile(onnx)` export takes the whole model as one in-memory buffer and copies it. Browser ONNX is therefore new work, described under "ONNX import" below.

## The boundary you must respect

A Prism holo is a pure, stateless function from bytes to bytes. Its guest is generated from LexLean, imports nothing, cannot be hand-written, keeps no state between calls, and runs each request on a fresh instance. It cannot fetch, store, stream, use threads or hold a model.

So the app has three planes. Keep them separate. The owner has approved this split; do not reopen it.

| Plane | What it is | How it is trusted |
|---|---|---|
| Kernel | The holo. Every decision the app makes, as LexLean models. | Proved by Lean. Examples replayed. |
| Engine | Rust compiled to wasm, linking the hologram crates. Download, hashing, compile, inference, tokenizer. | Tested against external oracles. Not proved. |
| Page | The UI and workers. Calls the kernel for decisions, the engine for work. | Tested by a browser journey. Not proved. |

Rule: the engine and page never decide policy. They ask the kernel, passing state in and taking state out. The kernel is never called per token; a fresh instance per call is too slow for the hot path. Call it per stage and per turn.

Say "proved" only for what Lean proved. Say "tested" for everything else. Inference is tested, never proved.

## Kernel: what to specify and prove

Use profile `prism-app`, LexLean language 1.2. The lexicon covers booleans, naturals, fixed-width arithmetic, bytes, maps, sets and state machines. No floats. One holo has one entry, so the first request field selects the decision.

Draft these as capabilities. For each, state the property in one plain sentence, then prove it. If a property cannot be proved as stated, say so and propose the nearest provable one. Do not weaken silently.

1. **Journey state machine.** Stages: idle, preflight, download, compile, materialize, ready, generating, failed. Prove: `ready` is reachable only through a passed preflight, a completed download and a materialize in which every label verified. Prove: no transition leaves `failed` except reset.
2. **Preflight verdict.** Input: which config keys are present, the family name, manifest shape flags. Output: supported, derived, or rejected with a reason. Prove: total, exactly one verdict per input. Prove: rejected implies zero shard bytes are authorised.
3. **Resource projection.** Input: parameter count, bytes per weight, local headroom, address space, context length. Output: cache coverage, window, stage plan. Prove: it always returns a plan (the journey is never refused for resources). Prove: the window never exceeds the address-space budget. Prove: the stage partition covers every layer exactly once.
4. **Quantisation tier.** int8 when it fits resident, int4 only when int8 does not fit and int4 does, otherwise int8 paged. Prove: if any tier fits resident, the chosen tier fits resident. Prove: more headroom never moves the choice down the order paged, int4 resident, int8 resident. (An earlier wording, "more headroom never yields a lower-quality tier", is false: a small budget gives int8 paged and a medium one gives int4 resident.)
5. **κ admission.** Input: expected label, observed label, declared byte length, observed byte length. Prove: a buffer is admitted only when both pairs are equal.
6. **Tensor range check.** Input: file length and a tensor's offset and length, plus the previous tensor's end. Prove: accepted ranges are in bounds and disjoint. The host folds this over the safetensors header; the full header may exceed the guest allocation limit.
7. **Context trimming.** Input: token counts per turn, context length, reserve. Output: how many oldest whole user and assistant pairs to drop. Prove: the result fits. Prove: the latest user turn is always kept. Prove: it drops the minimum number of pairs.
8. **Decode bucket.** Prove: the bucket is at least the needed length, at most the model's `max_position_embeddings`, and never shrinks within a session.
9. **Generation bound.** Prove: the token budget for a turn never exceeds the remaining context. Stop on end-of-sequence, budget, or cancel; the per-token check lives in the engine and is tested against this model's examples.
10. **ONNX operator preflight.** Input: the model's operator list as (domain, operator, opset version) codes, and the supported set. Output: runnable, or rejected naming the first unsupported operator. Prove: runnable exactly when every operator is in the supported set. Prove: rejected implies zero weight bytes are authorised.
11. **Shape binding.** Input: each symbolic dimension's lower bound, upper bound and requested value. Output: the bound value rounded up to its bucket, or rejected naming the dimension. Prove: a compile is authorised only when every dimension is bound within its bounds. Prove: the bucket is at least the requested value and never above the upper bound.

Every capability carries real examples: requests and their exact responses. Follow the template's order of work for each: `just new-capability`, model, examples, failing test, make it pass, plant the defect, confirm the gate rejects it.

## Engine: built for latency

Link the Rust crates directly into your own wasm module. Do not use the `@tryhologram/wasm` SDK driver: it exposes only the byte path, which re-hashes and copies the KV cache every token.

Pin `Hologram-Technologies/hologram` by explicit revision at 0.13.1 or later, never by tag. The source pins 0.10.0 through a personal fork; check whether that fork holds commits the org repository lacks before moving.

Port, do not rewrite, the parts of hologram-ai that are already correct: the parametric decoder builder, the ONNX importer's operator mapping, lowering, κ-map emission, the tokenizer, quantisation. Leave behind what the browser journey does not need: the browser extension, React, markdown and KaTeX rendering, and any path superseded by ADR-0019.

Latency levers, in the order to apply them. Each exists in the runtime today.

1. **Addressed decode.** `intern_input` once, then `execute_addressed` and `resolve`. Carry the KV cache as labels, never as bytes.
2. **In-place KV.** Use `OpKind::KvCacheWrite` and the six-input fused `OpKind::Attention`. Keep the live cache uniquely owned; a leased cache falls back to a copy.
3. **Scalar-mask decode attention** (κ121, new since the source's pin). Cost follows the realised context, not the bucket.
4. **Threads.** Feature `wasm-threads`, backend compiled `no_std` so idle workers sleep on `Atomics.wait` instead of spinning. Flags: `+simd128,+atomics,+bulk-memory,+mutable-globals`. Every worker calls `hologram_worker_run` before the first execute; late registration traps. Run execute in a worker, never on the main thread. Spawn the pool during model load, not on the first send.
5. **SIMD tiers.** The wasm tier is chosen at compile time. Ship a simd128 binary and a relaxed-simd binary and pick one in JS.
6. **Weightless archive with paged weights.** `load_paged`, not the fully resident `load`, which hashes each weight three times. Verify each tensor's label once, at download.
7. **Warm start.** Persist the compiled archive and its folded cone in OPFS (`fold_archive`, `persist_cone`, `warm_from_store`). A return visit should not recompile.
8. **Leases.** `retain_label` on a parked conversation and on the system-prompt prefill. Switching back should cost a hash.
9. **int8 by default.** int4 is a memory tier only; it measured slower than int8 everywhere.
10. **Sampling.** Partial top-k, no full-vocabulary sort.
11. **Session lifetime.** The generate worker and its session survive across turns. A warm turn pays decode only.
12. **Streaming.** One message per token to the page; write to the DOM once per animation frame; never re-render the whole reply per token.
13. **Build.** `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, then `wasm-opt`.

Do not use the WebGPU backend. It covers seven f32 ops, allocates per call and needs `std`. Speculative decode stays off by default; it measured 1.0 to 1.5 times and depends on the workload.

Known traps: wasm32 has one 4 GiB heap and no memory64. f32 results differ bit-for-bit between wasm and native, so native oracles compare within tolerance and the browser fixture compares exactly against its own committed reference. Threads need cross-origin isolation: `COOP: same-origin` and `COEP: credentialless` (not `require-corp`, which breaks HuggingFace), served on Pages through a service-worker shim.

## ONNX import: one gesture, no waiting on bytes that are not needed

The user drops a `.onnx` file, pastes a URL, or names a HuggingFace repository. Nothing else is asked. The model lands in the same pipeline as safetensors: weights in the κ-store, a weightless archive, paged loading, one session.

**What "any ONNX model" means here.** It cannot be asserted; the runtime's operator catalogue is closed and the source's importer rejects random-number, ONNX-ML, sequence, optional and string operators. Treat coverage as an `open` row: measure it, publish the number, raise it. An operator the catalogue lacks is desugared into canonical operators. One that cannot be desugared is an upstream request to the runtime, never a local kernel and never a silent fallback. A model outside coverage is rejected in preflight, naming the operator, before any weight byte moves.

Build it in this order.

1. **Structure-first scan.** Write a streaming protobuf walker in Rust, compiled into the engine wasm. It reads tags and lengths and skips over every initializer's `raw_data` body. For a local file use `File.slice`; for a URL use ranged requests. Result: the graph, the operator list, the opset imports, and every tensor's name, dtype, shape, offset and length, having read the structure only, not the weights.
2. **Preflight on the scan.** Send the operator list to kernel decision 10 and the symbolic dimensions to decision 11. Show the verdict, the resource projection and the input and output ports. This is the moment the user learns whether the model will run.
3. **Stream weights by tensor.** For each tensor, read its byte range, hash it incrementally, write it to the κ-store, drop it. Peak memory is one tensor. External-data files (`location`, `offset`, `length`) use the same ranged path. Typed fields (`float_data`, `int32_data` and the like) are normalised to little-endian bytes before hashing. Fetch ranges in parallel.
4. **Compile weightless.** Weights enter the graph as external κ parameters, exactly as in the safetensors path. Fold shape subgraphs and constants at compile time so the lowered graph is static.
5. **Bind shapes once.** The runtime has no runtime shape machinery. Bind each symbolic dimension at compile, round sequence-like dimensions up to a bucket so nearby sizes share one archive, and cache every compiled archive in OPFS keyed by the structure's κ and the binding. Re-importing a model, or re-running at a seen shape, costs a hash, not a compile.
6. **Generic runner.** Build the input form from the archive's ports: JSON or `.npy` tensors always; text when a `tokenizer.json` sits beside the model; an image when a port is shaped like one. Show outputs as tensors, with a top-k view when the output is a single vector.
7. **Fast repeat runs.** `intern_input` every input and call `execute_addressed`. Identical inputs hit the whole-graph memo; changed inputs recompute only the affected cone. Keep the session and worker alive between runs.
8. **Stateful ONNX loops.** For decoder exports with `past_key_values` inputs and `present` outputs, carry each `present` output's label straight into the next step's `past` input. No bytes cross the boundary, so nothing is re-hashed. Where a repository offers both safetensors and ONNX for a supported decoder family, prefer safetensors for chat: it reaches the fused decode path.

Coverage evidence, both reported on every run: the share of the official ONNX node-test corpus that passes at a pinned ONNX release, and a pinned set of real models covering at least one image classifier, one transformer encoder, one object detector, one speech encoder and one decoder export. Correctness authority is ONNX Runtime, natively, within tolerance.

## Page: lean

No framework. Plain TypeScript modules, one stylesheet, three workers (download, generate, pool). Four views: models, chat, run, logs. One import box on the models view accepts a dropped file, a URL or a repository id and detects the format itself. Load the engine wasm after first paint. Every failure shows the stage, the reason and the offending identity.

## Performance contract

Latency claims are measurements, never assertions, until measured on a named device.

Instrument and report on every run: cold time to first token, warm time to first token, tokens per second at context 128, 2,048 and 8,192, model load time, peak heap, and dispatched versus skipped kernels.

For ONNX, also report: time from drop to preflight verdict, time from drop to first result, repeat-run time with identical input, repeat-run time with changed input, and time to reopen a previously imported model. Target, confirmed by the owner as a gate: the preflight verdict for a local file arrives in under one second whatever the file's size, because only structure is read. Measure it on the same named device, on every model in the pinned real-model set, from drop to verdict on screen. The same rule applies as for the chat targets: if it is unmet after the structure-first scan is in place, do not lower it; report the measured number and the step that bounds it, and ask the owner. The source has no browser ONNX baseline for the other figures; the first measurement becomes their baseline.

Baselines from the source, to beat: about 1.6 tokens per second warm on SmolLM2-135M int8 in the deployed single-threaded build; 81 tokens per second and 685 ms to first token on Qwen2.5-0.5B under wasmtime with four participants on runtime 0.8.2.

Targets, confirmed by the owner: on a 4-core laptop in Chrome, SmolLM2-135M int8, warm time to first token under 300 ms and at least 40 tokens per second at context 2,048. Name the exact device and browser version you measured on. Once a target is met, a regression past it fails the gate. If a target is still unmet after every lever is applied, do not lower it: report the measured number and the step that bounds it, and ask the owner.

## Phases and gates

**Phase 0. Verify the ground.** Run `npx prism-code@latest setup`. Read the template's `AGENTS.md`, `README.md` and `app/src/Contract.lex.tex`. Then answer, with evidence, before designing:

- Which SDK does the template pin? `prism-code@latest` is 0.7.2 on npm; "SDK v0.4" is the PrismPM SDK candidate the template locks. Use what the lock pins.
- What is the guest allocation limit and text profile version in that SDK?
- Does the template page's content-security policy allow `huggingface.co`? It is reported to allow only its own origin and four fixed ones, in a file pinned by `frame.lock`.
- Does the template's Lighthouse budget (reported as 120 KiB total) survive a lazily loaded engine?
- Does the threaded wasm build of runtime 0.13.1 pass the source's `wasm-test` kernels?
- What share of the official ONNX node-test corpus does the source's importer pass today, at a pinned ONNX release? This is the starting coverage number.
- Do ONNX initializers lower correctly as external κ parameters? The source uses that form only on the safetensors path.

Gate: a one-page findings note. If the pinned frame blocks HuggingFace or the engine, stop and ask the owner to choose: diverge from the frame in this repository, or host the app page beside the template page and load the verified holo from it. Never edit a pinned file without that answer.

**Phase 1. Specify and prove the kernel.** Ask once for approval of the repository name and public visibility, then create it from `UOR-Foundation/template`. Show the eleven decisions' properties as plain sentences for approval before proving. Write the models through the `lexlean` MCP: check, format, verify. Gate: `build --check` clean, `verify --lean` passes every theorem, `axioms` exact, every example replays.

**Phase 2. Engine.** Port to runtime 0.13.1 with the addressed decode loop. Gate: native parity with ONNX Runtime logits on the canonical model within tolerance, tokenizer parity with HuggingFace `tokenizers`, BLAKE3 known-answer tests, threaded wasm kernels green; for ONNX, the streamed weightless import produces the same outputs as the source's whole-file import on every fixture, parity with ONNX Runtime within tolerance, and the node-corpus share is no lower than the Phase 0 number.

**Phase 3. Journey.** Wire the page so every decision goes through the kernel. Gate: two hermetic browser journeys in headless Chromium. Chat: download to three-message chat, matching the committed reference at temperature 0; the warm second turn materialises nothing. ONNX: drop a committed fixture, see the preflight verdict, run it, match the committed reference output; drop a fixture with an unsupported operator and see it rejected by name with no weight bytes read; re-import and see no recompile.

**Phase 4. Latency.** Measure, apply levers in order, measure again after each. Keep a lever only if the number moves. Gate: the performance contract reported on a named device.

**Phase 5. Ship.** Push; the template's gate and the build workflow must be green. Gate: live URL, holo fingerprint, examples replayed on the live holo.

## Rules

- Never hand-write a guest. Never edit files under `prism/` or files a lock pins.
- Keep sources LF. Clone with `core.autocrlf=false`.
- No stubs, flags, deferrals or silent fallbacks. Unfinished work is an `open` row with a measurement, never a `TODO`.
- No model name, dimension or limit hard-coded in engine code. Everything derives from the model's `config.json`.
- Hosted verification sends `src/` to a verifier that is unlisted, not private. Put no secrets there.
- Fix the cause of a red gate. Never weaken the gate.
- Ask before any outward-facing step: creating the repository, enabling Pages, diverging from the frame.

## Final report

Six lines: the live URL; the repository; the holo fingerprint and number of examples replayed; what is proved (list the theorems) versus what is tested; the measured performance contract with device and model; ONNX coverage as the node-corpus share and the list of real models that run.
