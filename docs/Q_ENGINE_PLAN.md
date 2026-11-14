# Plan: Holo Q's engine as this repository's engine plane

Companion to [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Where the two differ on the engine, this plan wins once approved.

Status on 2026-11-14: approved. On branch `q-engine`: KN-12 to KN-16 are proved by Lean, registered, routed and pinned by tests; KN-17's proofs are in progress. None has acceptance examples yet: the kernel's example list stops building above 108 entries, and that limit is being fixed in LexLean. Nothing in the engine is built.

## Why

Holo Q (`hologram-apps/apps/q`) already runs GGUF models in a browser tab on WebGPU, with no server, and verifies every weight block by its hash before use. This repository already has the kernel: eleven proved decisions about what may be loaded and run. It has no engine yet, and the engine it planned (Rust on the hologram runtime) is blocked until those crates are published to crates.io.

Q's engine depends on no unpublished crate. Bringing it here gives the kernel an engine now, and gives Q a proved kernel instead of rules spread through JavaScript. One kernel, proved once, is the reason for merging rather than building a second repository.

## Decisions taken on 2026-11-14

| # | Decision | Consequence |
| --- | --- | --- |
| 2 | Q merges into this repository; no separate repo. | Q's engine becomes the engine plane under `site/app/`. Phase 2 below replaces Phase 2 of the implementation plan for language-model chat. ONNX import (Phase 3 there) stays as written and moves after milestone 1 here, still waiting on Decision 1's crates.io publish. |
| 3 | "Formally verified" means: every decision is proved, every integer-exact algorithm has a proved reference the engine is tested equal to, and floating-point inference is tested, never proved. | Each claim in `model/ids.toml` keeps its honest level. Nothing on the GPU is called proved. |
| 4 | Two backends behind one engine interface: WebGPU, and a CPU fallback compiled to WebAssembly. | Every browser runs something. The kernel picks backend and model tier from what the device has. |
| 5 | Core inference first; the rest of Q follows in later milestones. | Milestone 1 is defined below. Parity is the end state, not the first release. |
| 6 | The first model is [HOLOGRAMTECH/q-ternary-bonsai-1.7b](https://huggingface.co/HOLOGRAMTECH/q-ternary-bonsai-1.7b), and milestone 1 also keeps Q's four formats. | Milestone 1 adds Bonsai's ternary format and YaRN rotary scaling to Q's q4, q3f, t2 and t2r. |
| 7 | `.holo` means only the Prism SDK package; weights stay outside it. | Bonsai's single file is read directly as a weight container. Code and docs call it a weight container, never `.holo`. |
| 8 | Lift the acceptance-example ceiling before adding examples. | The kernel's example list stops building near 115 entries. The new decisions need about 23. No existing example is cut. |
| 9 | Models are pulled from the decentralised hub as easily as from Hugging Face. | The loader takes a base URL plus repo, revision and path in Hugging Face's resolve form, never a fixed host. |
| 10 | The page does the hub runner's job itself; no server. | The hub's kernel holo runs in the page beside this one: the page checks the hub's signed index and the model's record with the hub's own code (reused by agreement, not rewritten), asks the hub kernel's resolve decision (104 bytes) which source to use, range-fetches from it, and asks the hub kernel's accept decision (81 bytes) on the whole file's SHA-256 and size. An empty source list falls back to the configured base URL. None of the hub's rules are copied into this kernel or into page code; per-block admission stays with KN-05. Placement: after the first Bonsai load from Hugging Face (to be confirmed). Prerequisites: a real browser `fetch()` with a Range header through a cross-origin redirect; and the hub's index, record and resolve-request code lifted into one importable module (`web/client.mjs` in the hub repo, proposed, not started). Only `web/sigstore.mjs` is importable today. The Prism browser runtime comes from prism-code. |

## The format boundary

The canonical `.holo` (HOLO version 4: eight sections, four content blobs, a BLAKE3 footer) holds the kernel guest, its view, its capability set and its model document. Its blob set is fixed, so it cannot carry model weights, and nothing here tries to make it.

Models stay outside it. Two layouts are read:

1. **Weight container** (Bonsai's file: magic `HOLK1`, format `holo-kappa/1`). A JSON header lists every tensor's offset, stored length and BLAKE3 name, then the embedded tokenizer, then gzip'd tensor blocks. Checked on 2026-11-14 against the published file: `l0.wq`'s stored bytes hash to the BLAKE3 name the header gives, and decompress to 1,179,648 bytes, the size the ternary layout implies for 2048 by 2048. The engine range-reads one tensor at a time; KN-06 checks each byte range against the file, KN-05 admits each block by its name and length, KN-13 checks the decompressed length.
2. **Block set** (Q's layout: `manifest.json` plus `b/<label>.gz`), relabelled from SHA-256 to BLAKE3 so KN-05 applies directly.

In both, the manifest is admitted only when its label equals the one pinned in the app's model catalog. This keeps the property Q already has: a host can serve a model but cannot change it. Where a source also publishes a whole-file SHA-256 (the hub's index does), it is checked as well, incrementally as the bytes stream, without holding the whole file.

Tokenizer: Bonsai embeds a GGUF header (5.9 MB), not `tokenizer.json`. `crates/tokenizer` reads `tokenizer.json` and is checked against HuggingFace `tokenizers`; a GGUF-vocabulary reader feeding the same tokenizer is needed, owned by whichever session the coordinator assigns. Q's own tokenizer wasm is not carried over. Q's archive format (`forge/holo-archive.mjs`, "holo/2") is not carried over either.

First model, pinned (from the hub's signed index record, whose SHA-256 matches `data/index/root.json`; the root's Sigstore signature not yet checked): `HOLOGRAMTECH/q-ternary-bonsai-1.7b` at revision `c6d6a12384ecc52ae7e006bef263bfedf5e7496c`, file `q-ternary-bonsai-1.7b.v1.holo`, 424,013,749 bytes, SHA-256 `acf0b5ad96b2a9b5b6a72b47cdddf5ec7ae4d15d5c50d0b5dea1ab245288ba99`. Its `sources[]` is empty, so bytes come from the configured base URL (Hugging Face). Measured in Chromium on 2026-11-14 by the coordinator session: a cross-origin `fetch()` with a Range header to Hugging Face's resolve route answers 206 through its CDN redirect, about 0.5 s per range. Safari, Firefox and parallel or large ranges are not yet measured.

Where a model comes from: the hub's release record (format `hologram.hub.release/v4`) carries `origin` with `kind` and `resolve`, frozen by the hub's own tests. A file's URL is `origin.resolve + path`, with no separator added: `resolve` ends in `/` and already pins the revision, so it is good for that record's revision only. The engine reads the bytes from `resolve` and does not assume a `kind`. A record without `origin` (older than v4) falls back to the configured base URL. Also frozen: `repo`, `revision`, `files[]` with `path`, `size` and `gitSha1`, and `sources[]`; `files[].sha256` exists only where the source published one, so its absence is not an error.

Sources: a model source is a base URL, a repo, a revision and a path, fetched by Hugging Face's `/{repo}/resolve/{revision}/{path}` route, so huggingface.co and the decentralised hub differ only in the base URL. Fetches must survive a cross-origin redirect while carrying a Range header. The hub's own kernel has an "accept" decision over SHA-256 and size; for the same facts it must give the same verdict as KN-05.

## What is proved, what is tested

The kernel is a guest call, so it decides at load and turn boundaries, never once per token. Where the engine must make a kernel-shaped choice per token (the stop check, the greedy pick), it mirrors the kernel and is tested to give the kernel's answer on every acceptance vector, the pattern the implementation plan already uses.

New kernel decisions, each proved in LexLean with its scenario, pinned test and planted defect:

| ID | Decision | Property to prove |
| --- | --- | --- |
| KN-12 | Backend | WebGPU is chosen exactly when the adapter is present and the largest buffer fits its binding limit, otherwise the CPU path; a wider limit never takes WebGPU away. Whether the model fits memory stays with KN-03 and KN-04. Proved: `gpu_usable_decides`, `limit_monotone`. |
| KN-13 | Block length | The kernel answers the length a weight block must have, from its format and shape: q4 20 bytes per 32 weights, q3f 16 per 32, t2 4 per 16, t2r 68 per 256, Bonsai's ternary 36 per 128 (sign plane, nonzero plane, f32 scale), f32 vectors such as the norms 64 per 16; a shape whose columns do not divide into groups is refused. The caller compares it with the decompressed length. Proved: `shaped_decides`, `group_bounds`; the length formula is tested by examples and was checked against a real Bonsai tensor. |
| KN-14 | Greedy pick | The engine claims an index and score; the claim is accepted exactly when every earlier score is lower, the score at the index is the claimed one, and no later score is higher: the first maximum. Scores are f32 logits mapped to order-preserving 32-bit keys. Proved: `standing_decides`, `holds_decides` (induction). |
| KN-15 | Penalty window | A token is kept unpenalised exactly when no position inside the last `window` positions holds it; the window size is part of the request (Q uses 64). Proved: `clear_decides`, `clear_from_decides` (induction). |
| KN-16 | Speculative commit | A proposed commit count is accepted exactly when it lies in 1 to k+1, every draft before the last committed position equals its verified pick, and the draft at that position, if any, differs. Proved: `agrees_decides`, `agrees_from_decides` (induction). |
| KN-17 | Quant codes | Unpacking any place of a packed group gives back the code packed there, for q4, q3f (including its two codes in the spare bits), t2/t2r and Bonsai's sign and nonzero planes; the answer is the code minus the format offset. Proved: one round-trip theorem per format. |

Existing decisions Q starts using: KN-01 drives load states, KN-05 admits manifest and blocks, KN-07 trims context instead of Q's "context full" error, KN-08 grows the KV bucket, KN-09 bounds each turn's tokens, KN-04 picks the quantisation tier.

Tested, not proved: every WGSL kernel, the CPU kernels, prefill and decode, attention, RoPE, normalisation, int4 KV, and the receipt's re-derivation.

## Checkpoint 1: first light

Milestone 1 is reached in checkpoints. The first is the only one on the critical path. It is done when Ilya can do this himself, and each step is also checked by a test, not only by eye.

| Step | What he does | The test that checks it |
| --- | --- | --- |
| 1 | Opens the app's page in Chrome on a laptop. Nothing to install, no server of ours. | Journey in Chromium: the page loads from a static server; the network log shows requests only to the page's own origin, Hugging Face's resolve route and its CDN, and the hub's Pages site. Any other host fails the test. |
| 2 | Picks Bonsai and watches it download by Range request from Hugging Face, with each check shown: revision, size and whole-file SHA-256 from the hub's signed record (checked against `root.json`), each block's BLAKE3 against the container's header, and the kernel's answers. | Hermetic journey: a small weight container built for the test, in the same layout, is served locally; the network log shows Range requests only; the page's check list equals the expected list. Negative journeys: one flipped byte in a block, a wrong whole-file hash, and a record that does not match the root each stop the load before any weight reaches the GPU. Kernel conformance: every kernel request the page sends and the answer it acts on are compared with the kernel's own examples for that decision. One live journey against the pinned Hugging Face revision runs on a schedule, not on every push. |
| 3 | Sends a prompt and gets an answer by greedy decoding. | Reference test: for three fixed prompts, the first 64 generated token ids equal a committed reference transcript. The reference comes from a second implementation of the same model, not from this engine: llama.cpp on the source GGUF if its build reads the format, otherwise this repository's float CPU reference. Which one is not decided; the model card reports four of five outputs byte-identical to llama.cpp's WebGPU build, so equality is expected on most prompts, and any prompt where the two disagree is recorded with how many tokens agreed, never dropped. Unit test: the YaRN rotary factors equal the formula for every pair index. Differential test: batched prefill equals one-token prefill, token for token. |
| 4 | Compares speed with the model card's baseline on the same class of machine: 85 to 114 tokens per second, 0.31 to 0.45 s to first token with the system prompt's state cached. | Measurement harness: a fixed protocol (prompt length, warm-up, number of runs, median) writes time to first token, tokens per second, load time and GPU memory to `model/ledger.toml` for one named laptop. The card's numbers are a target, not a gate, until they are re-measured on that laptop; after that, a regression of more than 10 percent fails. |

In checkpoint 1: the ternary-per-128 GPU kernel, YaRN, batched prefill, the tokenizer read from the container's GGUF header through `crates/tokenizer` built for wasm, the kernel asked for every decision this path exercises (KN-01, 05, 06, 09, 12, 13, 14, and KN-07 or KN-08 when the conversation outgrows the context), and a minimal chat surface under `site/app/`.

Later checkpoints, each with its own gate: Q's four other formats (q4, q3f, t2, t2r); speculative decoding; the CPU fallback; offline use; Safari, Firefox and phones; inference receipts; the in-page hub runner that chooses a source; the 25 examples in the kernel list.

### What stands between today and checkpoint 1, in order

| # | Item | Needs | Estimate |
| --- | --- | --- | --- |
| 1 | PR #18 green: decisions 12 to 17 on main | Hologram AI: names for 12 to 17 in `site/decision-names.json`. CI: the holo build, which has not yet run with these modules | hours, once the names exist |
| 2 | Read the weight container's header (`HOLK1`) and directory in the page | Hologram AI: where the reader lives (a Rust crate built for wasm, beside the GGUF reader) | about a day once placed |
| 3 | Tokenizer in the page from the container's embedded GGUF header | Hologram AI: the wasm build of `crates/tokenizer` (its GGUF reader is on main since 8c2795c) | Hologram AI's to estimate |
| 4 | Streaming hashes in the page: BLAKE3 per block, SHA-256 over the whole file without holding it | Hologram AI: `crates/download` built for wasm, with an incremental whole-file SHA-256 added | Hologram AI's to estimate |
| 5 | The hub's signed record checked in the page | The hub session: an importable module for reading the index and record (`web/client.mjs`, proposed); only `web/sigstore.mjs` is importable today. Without it, checkpoint 1 would have to read the record itself, which duplicates the hub's code, so it waits | cannot estimate |
| 6 | The kernel holo running in the page and answering the engine | Reuse the page's existing guest worker (`site/guest-worker.mjs`) and the JS encoder generated from the kernel | about a day |
| 7 | WebGPU engine: Q's resident path ported for Qwen3, plus the ternary-per-128 kernel, YaRN and batched prefill | The patched Q engine the model card was measured with, if it can be found: it already has the ternary kernel and YaRN. Where it lives is not known to me | days; cannot estimate more closely until I know whether that patch exists |
| 8 | Reference transcripts for step 3 | A llama.cpp build that reads the source GGUF's ternary format, or this repository's float reference for Qwen3 | cannot estimate |
| 9 | Minimal chat surface under `site/app/`, with the check list | Items 2 to 7 | about a day |
| 10 | The tests above, and the measurement harness on one named laptop | Checked on 2026-11-14 in the SDK image, which has no GPU: headless Chromium gives a WebGPU adapter only with `--enable-unsafe-swiftshader --enable-unsafe-webgpu` (Google's software renderer), and with it a compute shader runs correctly and the storage-binding limit is 1 GiB. So CI can run the GPU code for real, slowly: a small hermetic model on every push, the Bonsai reference test on a schedule, speed only on the named laptop | about two days |

## Checkpoints 2 and 3

Agreed with Ilya on 2026-11-14. Speed parity with the original Q engine runs alongside as its own measured track.

**Checkpoint 2: Foundry runs the engine behind a flag for Bonsai 1.7B.** Done when the Foundry session can integrate it behind a flag and measure load time, time to first token, tokens per second and page weight.

- `site/app/engine/index.mjs` is the engine as an importable module, and the engine page is one caller of it; `window.__qengine` stays the page's own reporting. `models()`; `load(id, { kernel, onStatus, onProgress, signal })` takes a kernel the caller has verified and refuses any whose identity is not the one the module pins; `handle.generate(textOrIds, { maxNew, onToken, signal })` answers `{ text, tokenIds, stats }`; `handle.dispose()`.
- Generation stops at the model's end token (needs the tokenizer to read special tokens as single ids), at a hard token cap, and within one submission of an abort.
- Downloads are cached in the browser's private file system (OPFS) by block label; every read from the cache is admitted by the same kernel decisions as a download, and the whole-file root is recomputed.

**Checkpoint 3: constrained output and Bonsai 8B.**

- Output constrained to a JSON schema: a token mask before each pick. First as one position per submission while a schema is set, with the mask applied to the pick; a mask kept on the GPU is a later step if the round trip proves too slow.
- Bonsai 8B's one-bit format (a single sign plane with one scale per 128 weights): a kernel variant and a block-length format in decision 13.

## Speed: where a token's time goes, measured

Decision on 2026-11-14: reach parity with the original Q engine first (85 to 114 tokens per second on Bonsai 1.7B, measured on this machine on the same prompts), then set the next target from those measurements.

Measured on this machine's AMD RDNA 3 GPU in the browser pane (medians, pane hidden, so absolute numbers carry noise):

| Quantity | Value |
| --- | --- |
| Raw read bandwidth | 175 to 209 GB/s |
| Weight bytes per token (bit planes and scales, head included) | 0.48 GB (computed) |
| Bound for one token per weight pass | about 390 tokens per second |
| Current engine | about 43 tokens per second, 0.6 to 0.7 s to the first token |
| Body matrices per token | 12.4 ms; head 1.5 ms; everything else about 9 ms |
| Matrix kernels' bandwidth, by shape | 18 to 52 GB/s; Foundry's shipped kernel is within ten percent on every shape |

What this says: the kernel's inner loop is not what separates this engine from the original. A 2048 by 2048 matrix should stream in about 6 microseconds and takes about 46, so the cost of each of about 450 dispatches per token dominates. The original fuses its dispatches. The next step is fusion, from 16 dispatches per layer to about 6: norm with q, k and v; head norms with rotation and the cache write; attention; the output projection with its residual; norm with gate, up and SiLU; down with its residual. Every step keeps GP-01 and GP-04 (identical tokens to the reference and to one position per submission) and is kept only if the measured number moves. Past parity, the bound above can only be crossed by verifying several tokens per weight pass (speculative decoding, already a proved kernel decision, KN-16), whose gain depends on the text.

A row-and-lane kernel geometry ported from Foundry's engine was measured slower here (37.5 against 43 tokens per second) and was not kept.

## Phase 2. Engine core (milestone 1)

Scope: Bonsai 1.7B first (Qwen3, ternary per 128, YaRN rotary scaling, read from its weight container), then Llama, Qwen2 and Qwen3 dense models and BitNet, Falcon-E and TriLM ternary models; formats q4, q3f, t2, t2r and Bonsai's. Models load from Hugging Face or the decentralised hub by base URL. Verified load, prefill, greedy decode, speculative decode, int4 KV cache, inference receipts with re-derivation, the memo, the chat interface, offline use after one visit.

Steps:

1. Port `qvac-gpu.js` to `site/app/engine/gpu/`, keeping only the milestone-1 paths. Keep its op-for-op structure so Q's outputs are the reference. Add the ternary-per-128 kernel and YaRN rotary scaling Bonsai needs: pairs below 17 keep their frequency, pairs above 34 run at one quarter, a linear ramp between, and the rotated vector scaled by 0.1 ln 4 + 1.
2. Add batched prefill. Q runs the prompt one token at a time with a CPU round trip each, so time to first token grows with prompt length; this is Q's largest latency gap.
3. Detokenize incrementally and remove the fixed 25 ms sleep per decode call.
4. Tile the matrix-vector kernels and use subgroup operations where the adapter offers them, with the current kernel kept as fallback.
5. Add `crates/cpu-decode`: the same formats and forward pass in Rust, built to wasm with SIMD, and with threads when the page is cross-origin isolated. Crates.io dependencies only (R6).
6. Port `compile2bit.mjs` to emit BLAKE3-labelled block sets with `tokenizer.json`.
7. Wire every load and turn through the kernel.

Tests added:

| Kind | Test |
| --- | --- |
| Reference | Q, run unchanged at a pinned commit, gives the committed transcript for each model at greedy decode; the port gives the same token ids |
| Differential | Batched prefill equals one-token prefill, token for token, over 2,048 positions |
| Differential | WebGPU and CPU backends give the same token ids at greedy decode on every milestone-1 model, over 256 tokens |
| Differential | Speculative decode equals sequential decode, token for token |
| Kernel conformance | Engine argmax, penalty window, stop check and speculative commit give the kernel's answer on every acceptance vector |
| Kernel conformance | GPU and CPU unpack equal the KN-17 reference on generated blocks |
| Oracle | Tokens match HuggingFace `tokenizers`; BLAKE3 known answers |
| Negative | Wrong manifest label, wrong block label, short block, no WebGPU, adapter limit too small, model over memory: each refused by name, nothing uploaded |
| Journey | Chat, hermetic, in Chromium, WebKit and Firefox; tampered block refused; offline reload still chats; receipt re-derives |

Gate: all green in all three browser engines on every push; the root page still passes the template's Lighthouse run unchanged.

## Latency

Baseline first: run Q unchanged on one named desktop and one named phone and record time to first token at prompts of 32, 512 and 2,048 tokens, tokens per second, and load time, per model, in `model/ledger.toml`. The first target is to beat Q's own numbers on the same device; absolute targets are set from that baseline, not guessed. For Bonsai the baseline is its model card's measurement with a patched Q engine on one laptop (Radeon 8050S, Chromium): 85 to 114 tokens per second, 0.31 to 0.45 s to first token with the system prompt's state cached, 1,176 MB GPU memory at 3,000 tokens. Those numbers are re-measured here before they are used. Every lever is kept only if its number moves and outputs stay token-identical.

## Browser matrix

| Engine | Desktop | Phone |
| --- | --- | --- |
| Chromium | Chrome, Edge: WebGPU | Chrome on Android: WebGPU where the device has it, else CPU |
| WebKit | Safari 26: WebGPU | Safari on iOS 26: WebGPU; earlier: CPU |
| Gecko | Firefox: WebGPU where shipped, else CPU | Firefox on Android: CPU |

Playwright covers the three engines on every push. Each named phone is measured by hand per release and recorded in the ledger.

## Later milestones

| Milestone | Adds |
| --- | --- |
| 2 | Mixture of experts (OLMoE), diffusion (Dream), E8 codebook weights, layer streaming and disk-backed models for weights larger than memory |
| 3 | Agents, tools, MCP, skills, memory, code tools |
| 4 | Speech (Whisper, Moonshine), LoRA training |

Each milestone gets its kernel decisions first, in the same register-scenario-test order.

## Risks, most serious first

1. **BLAKE3 over gigabytes in a phone browser.** Q hashes with the browser's native SHA-256. BLAKE3 in wasm must be measured on the named phone before the label change is final; if it is too slow, hashing moves to a worker during download, not after it.
2. **Mobile GPU limits.** Storage-buffer binding limits as low as 128 MB force tiling of the largest matrices. Q already tiles the output head; every other large tensor must be checked against the adapter's limits by KN-12.
3. **The CPU path's speed.** A 2B ternary model on a phone CPU may be too slow to be useful. Measured in milestone 1; the kernel then offers only the models that meet a stated floor.
4. **Q's reference is a moving target.** Q is pinned to one commit for the reference transcripts; later changes to Q are not followed automatically.
