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

## Milestone 1: done when

Set by Ilya on 2026-11-14 to close the milestone. Each line has an owner; the milestone is published when all six are done.

| # | Done when | Owner |
| --- | --- | --- |
| 1 | A pull request that leaves the proofs untouched passes the gate: `just app` proves and builds the holo on every run, with no reuse path. | Hologram AI (PR #24) |
| 2 | The page compares the kernel it verified with the kernel it was built for before asking any question; on a mismatch it stops with "stale kernel", names both, and says to reload. The built-for identity comes from a source the service worker's cache cannot serve stale. | Q engine, with Hologram AI on the source |
| 3 | README and the page say what is proved, what is tested, what the holo carries, and the command that checks it. | Hologram AI |
| 4 | Typed text replaces the default prompt rather than joining it, and Enter sends (Shift+Enter adds a line). | Q engine |
| 5 | Qwen2.5-1.5B-Instruct runs in the browser on this engine (WebGPU for the math, the wasm module for reading, hashing and tokenizing) as its checkpoint below states, aiming at 100 tokens per second and under 300 ms to the first token. | Q engine |
| 6 | A final run on the published page: first visit, return visit, and a prompt of about 440 tokens. | Coordinator |
| 7 | A returning visitor whose browser holds an earlier deploy's page or engine files gets this deploy's: the page removes them and reloads once before the engine is imported (GP-07). Added on 2026-11-16, when the final run found the worker serving the engine of the deploy before; Ilya closes the milestone when this is published. | Q engine |

Speed is reported, not gated. The 300 ms aim is for a short prompt (about 30 tokens) with the model loaded and nothing cached. The cold 440-token prompt is reported beside it, and is a stretch aim: it is about 1.3 trillion operations, 150 to 300 ms at a good fraction of this GPU's compute. The 100 tokens per second aim is half of Qwen2.5's bandwidth bound on this machine.

Moved to milestone 2: Bonsai's own speed target (below), Qwen3-0.6B, Foundry's checkpoint 2, the proof carried by the package, the 25 examples, and the SDK upgrade. Milestone 2 was reset on 2026-11-16; its section says where each of these now sits.

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

Side by side on 2026-11-14, browser pane visible, one load each, the same five prompts of about 440 to 470 tokens (Foundry's system block and a decision-app request), 256 new tokens:

| | Original Q engine (Foundry's copy) | This engine |
| --- | --- | --- |
| Time to first token | 0.45 to 1.13 s | 11.3 to 12.1 s |
| Tokens per second | 43.5 to 57.0, median 52.5 | 18.8 to 22.4, median 21.5 |

That run streamed the model from Hugging Face during the measurement. Repeated with a local, hash-verified copy of the same file, pane visible, the same prompts:

| | Original Q engine (Foundry's copy) | This engine |
| --- | --- | --- |
| Time to first token | 0.30 to 0.47 s, system prompt cached | 12 s; 2.3 to 6.0 s after batched prefill |
| Tokens per second | median 92.2 | median 31.0 |

These are the numbers parity is measured against. Two gaps explain most of it, ahead of fusion: the prompt is run one position at a time, each a full pass over the weights (batched prefill reads each weight once for a block of positions); and attention walks the whole cache serially per head, so speed halves as the context grows (the original splits attention across the cache). The order is now: batched prefill, then split attention, then two batches in flight, then the remaining dispatch reductions.

A row-and-lane kernel geometry ported from Foundry's engine was measured slower here (37.5 against 43 tokens per second) and was not kept.

Split attention (each 64-position chunk of the cache its own workgroup, then a combine per head) is in, with GP-01 extended to 140 positions against the reference and GP-04 to a 100-token prompt; its speed is not yet measured with the pane visible.

### Target

Moved to milestone 2 on 2026-11-14. Set by Ilya earlier that day, replacing parity as the goal: Bonsai 1.7B ternary at 100 or more tokens per second and under 300 ms to the first token, cold: model loaded, nothing cached, a prompt of about 440 tokens, timed from send to first token. Cached time to first token is reported beside it, never instead of it. Tokens stay identical to the greedy reference at every step. The bounds allow it: about 390 tokens per second from bandwidth, and about 70 ms of weight reads for the prompt in blocks of 16; the remaining gap is this engine's kernels.

## Checkpoint: Qwen2.5-1.5B-Instruct in the browser

Asked by Ilya on 2026-11-14, part of milestone 1, ahead of further Bonsai speed work: the new engine runs Qwen2.5-1.5B-Instruct at 4 bits in the browser, and both engines are measured on it.

Source: [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) at revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, Apache-2.0. Its Q4_0 file (1,066,227,232 bytes) groups 32 weights under one f16 scale; widened to f32 that is q4's 20 bytes per 32, so its Q4_0 tensors are relaid into KN-17's order without requantising; its output head is a separate Q6_K tensor and is requantised to q4. Its Q4_K_M file (1,117,320,736 bytes, the "4-bit, 1.1 GB" line) mixes block formats and would need requantising throughout. Shape: Qwen2, 28 layers, width 1536, 12 heads, 2 key-value heads, head size 128, FFN 8960, vocabulary 151,936, output head separate from the embedding, q/k/v biases, no per-head q/k norm, rotary base 1e6 without YaRN. About 0.96 GB read per token, a bound near 200 tokens per second on this machine.

Done when Ilya can:

| Step | What he does | The test that checks it |
| --- | --- | --- |
| 1 | Opens the engine page with the model's address and loads it; every block passes the kernel's admission and length checks. | The journey of checkpoint 1, run on a small generated Qwen2-shaped q4 container: good, flipped byte and wrong root. |
| 2 | Chats in Qwen2.5's own template and gets greedy answers. | GP tests on a generated Qwen2-shaped q4 model: every logit within tolerance of the CPU reference past 64 positions, and batched equal to one position at a time. On the real model, the first 32 tokens of three prompts equal the CPU reference run over the same weights. |
| 3 | Sees both engines' speed on the same prompts. | Side by side, pane visible: time to first token cold and tokens per second, the original's number beside this engine's, in this file. |

Gaps, in order, with estimates: q4 matrix, batched-matrix and embedding kernels, and q4 in the CPU reference (a day); q/k/v biases and optional q/k norm (half a day); a converter from the GGUF to the weight container with BLAKE3-named blocks and the tokenizer header (a day); the chat frame chosen per model, not Qwen3's fixed one (half a day); the baseline on the original engine and the side by side (half a day). About three and a half days. Publishing the packed file under HOLOGRAMTECH with a hub record is an outward step that waits for Ilya's yes; until then it is served locally with a local index, as Bonsai's benchmark is.

State on 2026-11-14: the container is packed locally (953,978,079 bytes, every block's BLAKE3 name verified) and both engines run it; this engine's first 64 tokens on a fixed prompt equal the original's text. Step 2 on the generated model is GP-05.

Step 3, measured on 2026-11-16 in the coordinator's session (its pane visible, this machine's RDNA-3 adapter, nothing else on the GPU), both engines on the packed container served locally, greedy, every run cold. Short is one prompt of 41 tokens (Qwen2.5's default system line and "What is the capital of France? Answer in one sentence."), five runs; long is five prompts of 423 to 439 tokens, three runs each, 256 tokens out; medians, the long ones a median of per-prompt medians.

| | This engine | Original |
| --- | --- | --- |
| First token, short | 156 ms | 545 ms |
| First token, long | 1,850 ms | 4,995 ms |
| Tokens per second, long | 70.7 | 73.6 |
| Load, served locally | 14.0 s | 14.7 to 15.9 s |
| On the GPU | 1,177 MB | 1,086 MB |

Both engines produced the same tokens on every prompt: all 256 on each long prompt and all 7 on the short one, the same on every repeat. Against the aims: the short first token is inside 300 ms; 70.7 tokens per second is short of 100; the long first token is short of the stretch aim. This engine's rate fell across repeats of one prompt (88, 74, 72 on the first); the original's opened high after a pause and then held, so carried state between turns is the first thing to rule out.

Order after it: milestone 2 below, as reset on 2026-11-16.

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

## Milestone 2: transformers.js equivalence for text models

Set by Ilya on 2026-11-16. The engine page becomes the equivalent of Hugging Face's transformers.js for text models: a user on a desktop or a phone names a supported open text model on the hub, and it is imported, checked and run in the browser. Browser first and serverless: no server of ours is in the path, and no model is repacked before it runs. Text generation first, then embeddings and classification; no vision or audio. ONNX exports first, read from the hub's own files; GGUF and safetensors import follow in milestone 3.

What this milestone gives a person, promise by promise, is specified in LexLean in `experience/src/HologramAiExperience.lex.tex`.

Studied on 2026-11-16: transformers.js 4.3.2, Hologram-Technologies/hologram-ai at `d5337ec`, Hologram-Technologies/hologram at `6888878`.

### How ONNX is run

Decided by Ilya on 2026-11-16: the engine recognises known model families in an ONNX file and runs them on its own kernels. It is not a general ONNX interpreter. transformers.js hands every graph to ONNX Runtime; this engine reads the graph to learn the family, the shape and where each weight lies, and refuses by name a graph it does not recognise. The consequence is stated wherever support is claimed: ONNX's operator set is far larger than what is listed here, and a model outside the list does not run.

| | Supported in this milestone | Refused by name |
| --- | --- | --- |
| Chat families | Qwen2 and Qwen2.5, Qwen3, Llama 3.x, SmolLM2: one shape (RMS norm, rotary positions, grouped-query attention, gated SiLU feed-forward, optional q/k/v biases, optional q/k norm, tied or separate head) | Gemma, mixture of experts, hybrid and GPT-2-style models; transformers.js lists 77 causal families |
| Export styles | Decomposed: Add, Cast, Concat, ConstantOfShape, Div, Equal, Expand, Gather, Greater, MatMul, Mul, Neg, Pow, Range, ReduceMean, Reshape, ScatterND, Shape, Sigmoid, Slice, Softmax, Sqrt, Transpose, Trilu, Unsqueeze, Where. Fused: MatMulNBits, GroupQueryAttention, RotaryEmbedding, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization | Any operator outside the list of the recognised style |
| Precisions | q4 and q4f16 (MatMulNBits) first, then fp16 | q8, int8, uint8 (dynamic activation quantisation), bnb4 |
| Later in the milestone | BERT-style encoders for embeddings and classification; their operator list is fixed when that step starts | |

### Integer only

Set by Ilya on 2026-11-17: Hologram AI's inference uses integers only, with no floating-point operation between the bytes admitted from gethologram.ai and the token ids emitted. Three things follow, and they are why:

- **Deterministic.** The same model, prompt and settings give the same bytes at every step on every CPU, GPU, browser and phone, because integer arithmetic has one answer and floating point's depends on summation order and hardware.
- **Verifiable.** Anyone can recompute an answer exactly and confirm it; the kernel re-checks sampled blocks of a product (decision 34) and replays a receipt (decision 31).
- **Interpretable.** Any layer at any position opens to the exact integers that produced it, each weight named by its tensor's BLAKE3 address (decision 32).

How a model stored in floats becomes integers: every finite f32, f16 or bf16 value is exactly an integer times a power of two, so stored scales, dense tables and norm weights decode to integers with nothing rounded at load. Rounding happens only where the integer model narrows a value, at declared places, by one rule (round half to even, on integers). Activations are the one approximation: one int8 scale per 32-wide group is measured first, then int16, then outlier channels split out, and the width is chosen by greedy-token agreement with the float model on the real models of this milestone. Softmax, RMS and layer norms, rotary positions (with YaRN and Llama 3 scaling) and SiLU and GELU become integer exponent, square-root, sine and cosine and activation tables, each generated by an exact integer algorithm and named by its BLAKE3 like a tensor.

The engine is written in LexLean. Decided by Ilya on 2026-11-18, because Hologram AI is a Prism application: its behaviour is its LexLean model. Every integer stage has its definition in LexLean and the theorems Lean proves about it: reading a stored weight's bits as integers (decision 37), the exact group sum and its accumulator width (29), declared narrowing with its bound (38), the row's recombination, RMS and layer norm by integer square root, softmax and SiLU and GELU by addressed tables, rotary positions with YaRN and Llama 3 scaling, and the greedy pick (14). Each stage is registered, given its scenario and a failing test, defined and proved before any implementation of it is written (R3). What Lean proves is the stage: that it is exact where it says so, within its declared bound where it narrows, and that its blocked and tiled forms equal its plain form.

The engine runs in the guest, 100% LexLean. Decided by Ilya on 2026-11-18: milestone 2 includes moving the whole engine into the Prism guest, lean, very low latency, high throughput and formally verified, so that nothing that computes an answer lives outside the LexLean model. Today a guest answers one request in a fresh WebAssembly instance with no state, no imports (so no WebGPU and no network), request and response caps, a 16 MB holo and 64-bit numbers. Three toolchain capabilities remove those limits, and each is now in this milestone:

- **State across calls.** PrismPM's session world, `hologram:guest/core-wasm@2` (proposal 629, merged in PrismPM #638, not yet released): one instance serves a session's ordered steps, the model is a pure `step : State -> Request -> State × List Response`, the host never sees the state, and a session's responses are a function of its requests, so a history replays. Weights come in as inputs referenced by their BLAKE3 digest (proposal 630, the binary application profile), not as state.
- **Streams.** Input as a fold over chunks with one theorem, proved once, that the result does not depend on where the chunks are cut; output one response per token (LexLean design draft, question 3).
- **A GPU.** A closed integer kernel language inside LexLean with two printers from one syntax tree, WGSL and Rust over flat buffers, each proved correct once for every kernel; a guest's step answers a dispatch plan and prism-code runs it on WebGPU (LexLean design draft, question 4). Fixed-width integers, `Int128` accumulators and bounded loops come with it (question 6).

These are built in the UOR toolchain (LexLean, PrismPM, prism-code) by the session "Guest state, streams and GPU for LexLean engine", from the brief `hologram-ai-engine-in-lexlean-prompt.md`; its designs were approved on 2026-11-18. Hologram AI writes its side now, in LexLean, so it is ready when they ship: the engine session (`app/src/Engine/Session.lex.tex`, KN-45), a pure step over open, admit a page, prompt chunk, generate and close, whose proofs say a page not named by BLAKE3 fails the session, the session is ready only when every page of the manifest is admitted, prompt chunks compose, a run with a floating-point stage is refused, the position never passes the capacity, and an ended session stays ended. Until the session world is in the SDK image that `prismpm.lock` pins, the WebAssembly and WGSL fast paths run inference and are gated equal, bit for bit, to the LexLean definitions, and the guest re-checks sampled blocks, rows and layer positions; when it is, the session becomes the guest's root and those fast paths are removed.

The float engines in `site/app/engine` (`gpu.mjs`, `cpu.mjs`, `encoder.mjs` and the `wgsl*.mjs` kernels, about 2,700 lines) stop being what the page runs; they stay only as the yardstick the integer engine's quality is measured against, and are removed once that measurement is recorded. An audit refuses `f32`, `f64` and WGSL `f32` and `f16` in the shipped path, and a planted float proves it can.

Measured by the Integer AI session on 2026-11-18, on Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B: 16-bit activation codes with one scale per group of 32 come about 1,300 times closer to the float result than the 8-bit codes with one scale per vector first used, at the same scalar cost, with products still exact. That is the width the LexLean narrowing stage is written for.

The integer engine is built by the Integer AI session, each stage in LexLean first and then its two implementations, in the order of its brief (`docs/VERIFIABLE_INFERENCE_PLAN.md` and the session's prompt): exact products (EN-34, PR 52) and the block check (KN-34, PR 51) are written; the activation scheme, the integer non-matmul stages, a codec per stored format, integer WGSL, admission by address and receipts follow. This line of work moves the page onto that engine and keeps its checks.

### Kernel decisions

Each is registered, given a scenario and a failing test, and proved before its implementation (R3). Lean proves the decisions; GPU and CPU arithmetic is tested against the CPU reference; the hash functions, the signature primitive and the browser are trusted. Eleven were written on 2026-11-16 as new decisions beside the existing ones, so a page already deployed keeps the questions it asks; what each proves and what it only pins is in KERNEL-VERIFICATION.md.

| ID | Decision | What it answers |
| --- | --- | --- |
| KN-18 | Operator name | Whether an operator, by the bytes of its name, is on the closed list of its export style: 27 names for the decomposed style, 18 for the fused, both read from real exports. KN-10 keeps the opset window; the count it trusts is now the number of names KN-18 refused. |
| KN-19 | Family | Whether a configuration's type (qwen2, qwen3, llama), activation (silu) and shape are ones the engine runs. |
| KN-20 | MatMulNBits length | The bytes of a 4-bit weight's codes and of its f32 or f16 scales, from rows, columns and block size. Zero points are outside it; a file that carries them is refused. |
| KN-21 | Precision | The first of q4f16, q4, fp16 that is offered and fits the budget, or none. |
| KN-22 | Device | WebGPU, the CPU, or a refusal when there is no usable GPU and the model is above the CPU ceiling. KN-12 stays as it is. |
| KN-23 | File span | Whether a span lies inside a file the model declares, after the previous span. |
| KN-24 | Source | The name of each file of a task and precision, as the export convention has it; a part beyond the declared count is refused. |
| KN-25 | Stop | Stop at any id on the model's own list, or at the budget; the id is reported first. |
| KN-26 | Evict | Whether a cached block is dropped: only an unpinned one, only while space is short; does not fit when no candidate remains. |
| KN-27 | Reuse | Whether a computed prefix may be reused: its label equals the one asked for and it is shorter than the prompt. |
| KN-28 | Manifest | Whether a model is ready: the label of the addresses admitted equals the manifest's, and every listed tensor is admitted. |

Two of the decisions first listed were not written. The journey needs no change: a read from the browser's store is one more source in its download step, and every block read there is admitted as a downloaded one is, so a return visit takes the same proved path. The index root's signature waits on the hub: its signing scheme and key are the hub's to state, and no decision is invented ahead of them.

### Content addressing

Set by Ilya on 2026-11-16: every tensor of every open model has one UOR content address, its unique home. The authority for the address is UOR-Foundation/uor-addr (Ilya, 2026-11-16: the canonical address for every open tensor), and the hash is BLAKE3 (Ilya, the same day, over SHA-256): a tensor's address is the BLAKE3 of its raw data bytes, written as a uor-addr `blake3:` label. That is uor-addr's plain byte address on its BLAKE3 axis, hologram's canonical leaf, the hub's tensor address today, and the digest this repository's containers already name blocks by. It is not the leaf of uor-addr's canonical form for a GGUF or ONNX model: in that form (uor-addr 0.2.0) a tensor's leaf is the SHA-256 of its data region whatever the label hash, and a model's label is taken over a skeleton holding those leaves. So a tensor has two numbers: the BLAKE3 address, its identity for storing once and for agreeing across systems, and the SHA-256 leaf, which ties it to its model's label. Settled by Ilya on 2026-11-16: the hub publishes both, the SHA-256 leaf labelled as uor-addr's, and a model's label is taken on the BLAKE3 axis, so it too reads `blake3:`. Both kinds of label are 71 bytes, so which hash a label is on is read from its prefix and never from its length. The hub's `data/uor` index lands with SHA-256 first and moves to BLAKE3 in a second commit; nothing here is built against it before that second commit. SHA-256 is also the identity of a whole file, which is what Hugging Face publishes. Checking one byte range of a tensor by BLAKE3's tree is not built anywhere yet (the vendored hasher has no subtree access), so a tensor is checked by hashing all of it. A container's block names are BLAKE3 over repacked, compressed bytes, so they name a block of one file and not a weight. The kernel's admission compares 32-byte labels and lengths and is the same for either hash. Three things follow from one address per weight.

- **Self-verifying.** A tensor is checked against its own name, wherever it came from. A model is a manifest: its tensors' addresses in order, with each one's shape and format, and the manifest has an address of its own. Trusting a model is then trusting one address; everything under it is checked in the browser and admitted by the kernel.
- **Deduplication.** Tensors are stored in the browser's private file system by address. A tensor already held, under any model, precision or repository, is not downloaded again, and every read from storage is admitted as a download is.
- **Decentralisation.** Because the name checks the bytes, any source may serve a tensor: the model's own repository, a mirror, another static host. A manifest lists where each tensor can be read (file, offset, length); none of the sources is trusted and none is ours.

Hugging Face publishes one SHA-256 per file, so the manifest with per-tensor addresses comes from the hub index: static files (planned as `data/uor/root.json` and shards of rows, each row a model's label and every tensor's name, type, shape and address), computed once when a model is indexed. State of the hub on 2026-11-16: its records for ONNX repositories will give one address per file (Hugging Face's own SHA-256, not witnessed by the hub) and no tensor addresses, because its manifest tool reads safetensors headers only. Until the hub reads ONNX layouts, the browser derives each tensor's address itself as the file streams, checks the whole file against the record's SHA-256, and keeps the manifest it derived. Two consequences, stated and not hidden: a tensor of an ONNX file is checked against an address this browser computed, so what vouches for the bytes is the file's hash and not a published tensor address; and a tensor can be skipped before download only when its address is already known, which for ONNX means a file this browser has imported before. Deduplication across models that have never been imported here waits for published tensor addresses. What sharing buys is load time and storage, not tokens per second: a shared tensor is not fetched or hashed twice, and it runs at the same speed. How much open models share is a measurement, not an assumption: byte-identical tensors are certain for the same model in several repositories, for an adapter over an unchanged base, and for the unquantised tensors of two precisions; a full fine-tune changes nearly every tensor and shares little. The share is measured across the supported families before any claim is made.

Reuse of computed results (prompt prefixes first; lookup tables and anything else after) is kept only where a measurement shows a gain with tokens unchanged. Browsers expose no persistent store for compiled shaders, so naming shaders by address buys deduplication within a session and no more.

Measured on 2026-11-19 from the hub's UOR index, which now lists every tensor of the three families by BLAKE3 (ledger `OPEN-SHARE-M2-FAMILIES`, `OPEN-STORE-SECOND-PRECISION`): Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B share no tensor with one another at either precision. Within one model, the q4 and q4f16 exports share their 4-bit codes byte for byte, 39 to 54 percent of the q4f16 bytes, so a second precision downloads 24 to 54 percent less. So storing once pays across precisions, repositories and returning visits, not across families. The page now keeps tensors that way (GP-31), loads with the network off (GP-30), and reuses the chat frame's head with tokens unchanged (GP-32); the hub's UOR index is not signed, so an address in it says only where to look, and the record's whole-file BLAKE3 and decision 28 say whether the bytes are right.

### Reuse

| Source | Taken | Left |
| --- | --- | --- |
| This repository | `crates/onnx-preflight` (operator and opset reader), `crates/tokenizer`, `crates/model-reader`, `crates/download` | |
| transformers.js | The call shape (task, model, device, precision), the `onnx/` file naming and precision suffixes, the precision listing; as conventions | ONNX Runtime |
| hologram-ai | The download design: ranged header reads, per-tensor hashing into browser storage, refetch of an uncached tensor by address | Its runtime (CPU only), its ONNX importer (native only), its dependency on an unpublished fork (R6) |
| hologram | The ideas of derivation keys for reuse and of lookup tables; the address authority is uor-addr, which hologram also builds on | Its crates (not on crates.io), its store, its WebGPU backend (seven kernels) |

### Done when

| # | Done when |
| --- | --- |
| 1 | Every decision above is registered, proved, and covered by a scenario and a negative test. |
| 2 | The page loads Qwen2.5-1.5B-Instruct from its ONNX repository with no repack, runs it on the integer engine, and its tokens equal the integer CPU reference bit for bit on WebGPU and on the CPU; its greedy agreement with the float model is recorded. |
| 3 | Llama 3.2 1B and Qwen3-0.6B load and run the same way. |
| 4 | One call loads and runs: task, model, device, precision. Chat template and stop ids come from the model's own files. |
| 5 | A return visit loads with the network off; every cached block is admitted again. |
| 6 | Tensors are stored once by address: a second model or precision stores no tensor already held, and downloads none whose address is known beforehand, from any listed source. The bytes and load time saved are measured, and the share of byte-identical tensors across the supported families is recorded. |
| 7 | Prefix reuse is measured and token-identical. |
| 8 | A small model runs on the integer engine with WebGPU off, and one run on a named phone is recorded, with the same tokens and logits as this machine's. |
| 9 | One embedding model and one classifier run on the integer engine and equal its CPU reference bit for bit; their agreement with the float references is recorded. |
| 10 | An unsupported operator, family or precision is refused by name (R5). |
| 11 | On the published page, on this machine and the phone: time to first token, tokens per second, cold load and download size of the integer engine, beside transformers.js on the same model and precision. |
| 12 | No floating-point operation lies between the admitted bytes and the token ids: the audit refuses a float type in the shipped path, and a planted float is refused. |
| 13 | The engine is the guest: Hologram AI's application declares the session world, its root is the engine session (KN-45) built on the LexLean integer stages (KN-37 to KN-44), and Qwen3-0.6B loads from content-addressed pages and answers through it with no Rust, JavaScript or WGSL on the inference path. |
| 14 | A prompt and a model's files stream into the guest in chunks of any size, with the chunking theorem proved, and tokens leave one per step. |
| 15 | The guest runs its integer kernels on WebGPU from the same LexLean definitions it runs on the CPU, with the printers proved, and CPU and GPU give the same integers over 256 tokens of three prompts. |
| 16 | Measured on the published page, here and on the phone: warm time to first token under 300 ms on Qwen3-0.6B, at least 100 tokens per second on Qwen2.5-1.5B at 4 bits, guest call overhead under 50 µs warm, and the holo under 16 MB; a target missed is reported with the step that bounds it, not lowered. |

Aims, reported and not gated: under 300 ms to the first token for a short prompt with a warm cache; 100 tokens per second or more on Qwen2.5-1.5B at 4 bits. Both were set for the float engine and are measured again on the integer one; integer-only is the requirement and the aims move if it costs speed.

### What it displaces

Kept in milestone 2: Qwen3-0.6B (now row 3), the browser-storage cache and the importable module of checkpoint 2 (rows 4 and 5), the 100 tokens per second aim. Moved to milestone 3: GGUF and safetensors import without a repack, Bonsai's own speed target, mixture of experts, diffusion, E8 codebook weights, layer streaming, constrained output and Bonsai 8B. Hologram AI's items (the proof carried by the package, the 25 examples, the SDK upgrade) keep their owner and are not displaced.

### Answered by Ilya on 2026-11-18

1. A model with no manifest on the hub is refused by name (R5). No model is addressed in the browser on first use, and no local address is invented for safetensors while uor-addr has no realisation for it. Whether the browser re-derives a model's label itself or takes it from the hub's attested row and checks only the tensors is still open.
2. The output head runs as the file has it, at 16 or 32 bits, by default. A 4-bit head derived in the browser is a measured option under its own address, never a silent replacement.
3. Ilya will name the phone. Until he does, rows 8, 11 and 16 stay open. Qwen3-0.6B is the proposed model for the CPU path.
4. Chat templates are defined and proved in LexLean: the subset the supported families use, with any other template refused by name. No Jinja library is added. transformers.js and ONNX Runtime are still used for comparison only.
5. Download size stays informational: Qwen2.5-1.5B's ONNX q4 file is 1.79 GB and q4f16 1.22 GB, against 0.95 GB for the repacked container.

## Later milestones

| Milestone | Adds |
| --- | --- |
| 2 | transformers.js equivalence for text models, above |
| 3 | GGUF and safetensors import without a repack, Bonsai's speed target, mixture of experts (OLMoE), diffusion (Dream), E8 codebook weights, layer streaming and disk-backed models, constrained output, Bonsai 8B |
| 4 | Agents, tools, MCP, skills, memory, code tools |
| 5 | Speech (Whisper, Moonshine), LoRA training |

Each milestone gets its kernel decisions first, in the same register-scenario-test order.

## Risks, most serious first

1. **BLAKE3 over gigabytes in a phone browser.** Q hashes with the browser's native SHA-256. BLAKE3 in wasm must be measured on the named phone before the label change is final; if it is too slow, hashing moves to a worker during download, not after it.
2. **Mobile GPU limits.** Storage-buffer binding limits as low as 128 MB force tiling of the largest matrices. Q already tiles the output head; every other large tensor must be checked against the adapter's limits by KN-12.
3. **The CPU path's speed.** A 2B ternary model on a phone CPU may be too slow to be useful. Measured in milestone 1; the kernel then offers only the models that meet a stated floor.
4. **Q's reference is a moving target.** Q is pinned to one commit for the reference transcripts; later changes to Q are not followed automatically.
