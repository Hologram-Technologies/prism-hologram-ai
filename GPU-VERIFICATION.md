# GPU verification

What each gate over the browser engine's WebGPU computation (`site/app/engine`)
has been seen to reject. A gate is evidence only after a planted defect shows it
can fail ([AGENTS.md](AGENTS.md)); this file records the defect, the command,
the diagnostic and the restored result.

## What is tested, and what is not

| Claim | Evidence | Kind |
| --- | --- | --- |
| `GP-01`: the WebGPU forward pass of a ternary Qwen3 model gives the same greedy tokens as the CPU reference, with every logit within 0.002 | `gpu_gp_01`: a two-layer model with Bonsai's tensor names and layout, generated from a fixed seed, run for sixteen steps in headless Chromium on WebGPU's software renderer beside a double-precision reference that shares no code with the engine | tested |
| `GP-02`: the rotary frequencies are the YaRN formula, with its bounds computed from the model's parameters | `gpu_gp_02`: the engine's frequencies for Bonsai's parameters against the formula recomputed in Rust; bounds 17 and 34 as the model card states | tested |
| `GP-03`: the engine page loads a weight container through the verified kernel, answers as the reference does, and stops where a container is tampered with | `just page-journey` (run by `vv` after `app` and `engine`, and by `pages.yml` before it deploys) drives the assembled page in headless Chromium; `gpu_gp_03` reads that wiring and the journey's verdicts | tested |
| `GP-04`: many positions per submission, each pick written on the GPU and read by the next position, give the same tokens as one position per submission | `gpu_gp_04`: twelve tokens after a six-token prompt and after a forty-token prompt, which crosses a submission's thirty-two slots; every position runs exactly once | tested |
| `GP-07`: a page file held in the cache from an earlier deploy is removed and the page reloads once before the engine is imported, and a file still stale after that stops the page | `just page-journey` plants a cache named as the worker names its own, with one file of an earlier deploy and one of this deploy, and opens the page twice: once fresh, once with the reload already spent; `gpu_gp_07` reads that wiring and the worker's network-first fetch of the page, which the check rests on | tested |
| `GP-08`: a q4 Qwen2 model whose weights are given as ONNX MatMulNBits tensors, with f32 or f16 scales, runs from those bytes with no repack and gives the reference's tokens | `gpu_gp_08`: GP-05's model with every matrix written from the operator's definition (`packMatMulNBits` in the reference, which shares no code with the engine), once with f32 scales and once with f16; the f32 tensors are checked byte for byte equal to the engine's q4 block, and both run GP-05's checks. On 2026-11-16 the runner also passed on this machine's AMD RDNA-3 GPU (worst logit 6.6e-7) as well as SwiftShader (1.2e-4) | tested |
| `GP-09`: a dense f32 or f16 embedding with the head tied to it, as ONNX exports hold them, gives the reference's tokens, with the head split into tiles past one binding | `gpu_gp_09`: GP-08's models with the embedding as one dense matrix and no head tensor, as read on 2026-11-16 from onnx-community's q4 and q4f16 exports of Qwen2.5-1.5B (f32 and f16, 151936 by 1536), Qwen3-0.6B (f16, 151936 by 1024) and Llama 3.2 1B (f16, 128256 by 2048), each read by a Gather and, through a Transpose, by the head's MatMul; the binding limit is set to 15,000 bytes so the head runs in 12 and 6 tiles, the last one short. Also passed on this machine's AMD RDNA-3 GPU (worst logit 7.5e-7) as well as SwiftShader (1.4e-4) | tested |
| `GP-10`: a model shaped as Qwen2.5's q4f16 export (q4f16 weights, dense f16 embedding and tied head, f16 norm weights and biases) gives the reference's tokens | `gpu_gp_10`: GP-09's q4f16 model with every norm weight and bias in f16, as read on 2026-11-16 from the export; the engine widens them exactly on upload. Also passed on this machine's AMD RDNA-3 GPU (worst logit 6.1e-7) as well as SwiftShader (1.4e-4) | tested |
| `GP-11`: Llama 3's rotary scaling is transformers' llama3 formula, and a Llama-shaped model with it gives the reference's tokens | `gpu_gp_11`: the engine's frequencies for Llama 3.2 1B's parameters against the formula recomputed in Rust, 15 pairs kept, 3 blended, 14 slowed; then a model with no biases and the scaling over all three bands, beside a reference whose frequencies are written separately. AMD RDNA-3: worst logit 7.1e-7; SwiftShader 2.0e-4 | tested |
| onnx-community's Llama 3.2 1B export uses Llama 3 scaling although its config.json names none | read on 2026-11-16 from `model_q4f16.onnx_data` at revision 82ac693: its `cos_cache` and `sin_cache` rows at positions 1, 7, 1000 and 8191 are within 2.4e-4 (f16 rounding) of Llama 3 scaling at factor 32, low 1, high 4, original 8192, and up to 1.97 away from plain frequencies. So the importer takes the family's parameters, not the config's | imported, measured |
| The attention of the three exports | read on 2026-11-16: all three have separate q, k and v weights, half-split (not interleaved) rotary and the engine's shape. Qwen2.5's export spells attention out in plain operators with tensors named `onnx::MatMul_N`, so its weights are found by the name of the node that reads them; Qwen3's and Llama's use GroupQueryAttention and name weights by layer and role | measured; no kernel gap |
| `GP-12`: every weight of the three q4f16 exports is given its engine role by the node that reads it, shapes checked against the config, the unplaceable refused by name, and a model placed this way gives the reference's tokens | `gpu_gp_12`: `site/app/engine/onnx-roles.mjs` over the three graphs recorded on 2026-11-16 in `crates/conformance/tests/gpu/exports` (nodes, tensor types and shapes, config.json, at the hub's revisions; four rows of Llama's cos and sin cache): 338, 310 and 146 tensors placed, Llama 3 scaling confirmed from the cache (worst 3.6e-4); ten one-fault copies each refused by name; then GP-10's and GP-11's models written as a decomposed and a fused export, placed from their graphs alone and run. AMD RDNA-3: worst logit 7.1e-7; SwiftShader 2.0e-4 | tested |
| The recorded graphs are the exports' graphs | they were read by streaming each `model_q4f16.onnx` at the revision named and keeping only the nodes that read a weight of 16 or more numbers; the importer's own reading of the same files is not compared with them | imported, not cross-checked |
| A real ONNX export runs | not yet: the importer is not in the page | not tested |
| `GP-13`: a BERT encoder shaped as all-MiniLM-L6-v2's q4f16 export gives the reference's hidden states and its mean-pooled unit-length embedding | `gpu_gp_13`: a two-layer encoder in the export's formats, read on 2026-11-16 from Xenova/all-MiniLM-L6-v2 at 751bff3 (MatMulNBits q4f16 weights with f16 biases, LayerNorm with weight and bias, Erf GELU, word, position and token-type tables in f16), run by `site/app/engine/encoder.mjs` over sentences of 6 and 60 tokens beside a reference written from the papers. SwiftShader: states within 1.0e-6, embedding within 1.7e-8; AMD RDNA-3 the same order | tested |
| `GP-14`: a DistilBERT classifier shaped as distilbert-base-uncased-finetuned-sst-2-english's q4f16 export gives the reference's label scores and label | `gpu_gp_14`: the same encoder without token types and with the export's head (read at 0b6928e: Gemm with transB 1, f16 768 by 768 and 2 by 768, ReLU between), two sentences the reference labels differently. Scores within 1.3e-7 on both renderers | tested |
| The real all-MiniLM-L6-v2 and DistilBERT SST-2 run | not yet: neither is on the hub, the role map does not place encoder exports, and the tokenizer reads byte-pair encoding only, not BERT's WordPiece | not tested |
| A 467 MB head tiled on a real phone | the tiling is tested at a lowered limit; no phone has run it | not tested |
| The check reaches a visitor whose worker serves an earlier deploy | rests on the frame's worker fetching `index.html` from the network first (pinned by `gpu_gp_07`); the journey plants the cache directly and registers no worker, so the worker's own behaviour is read, not run | not tested |
| The engine gives the same tokens on a real GPU as on the software renderer | not checked here: CI has no GPU. The same test runs on a named laptop before checkpoint 1 is called done | not tested |
| Bonsai 1.7B itself, end to end | needs the weight container reader and the tokenizer in wasm; the reference for it is a transcript from a second implementation | not tested |

Nothing here is proved. The engine's numbers are floating point, which the
kernel's proof language does not have; the kernel's decisions that the engine
mirrors (the greedy pick among them) are proved there and tested against it.

## How the tests run

Both run in the SDK image, as the rest of the gate does. `gpu_gp_01` serves the
repository on a local port and opens it in the headless Chromium and Playwright
the image carries (`/opt/prismpm/oracles/node_modules/playwright`, the version
`package.json` pins), with `--enable-unsafe-swiftshader --enable-unsafe-webgpu`.
Without a WebGPU adapter the test fails; it never skips.

Measured on the first run: the largest logit difference 6.6e-4, on logits
spread over 6 to 8; the closest pair of best logits 0.075, so no step is a near
tie.

## Planted defects

| Defect | Command | Result |
| --- | --- | --- |
| The ternary matrix kernel reads the sign bit backwards | `cargo test -p repo-conformance --test gpu gpu_gp_01` | rejected: `step 0: a logit differs by 6.332613951387152 from the reference` |
| YaRN's attention scale left at 1 | the same | rejected: `step 1: a logit differs by 0.3351144151169181 from the reference` |
| YaRN's high bound rounded down instead of up | `cargo test -p repo-conformance --test gpu gpu_gp_02` | rejected: `high bound` (`left == right` failed) |
| The pick written to the current position's slot instead of the next | `cargo test -p repo-conformance --test gpu gpu_gp_04` | rejected: `a 6-token prompt: batched tokens differ from one at a time` |
| The prompt's rotation in a block ignores each position's offset within the block | `cargo test -p repo-conformance --test gpu gpu_gp_04` | rejected: `a 6-token prompt: batched tokens differ from one at a time` |
| The loader accepts any answer to a block's admission question | `just page-journey` (the journey) | rejected: `flip: expected to stop at the admission of l1.wq, stopped at stop: load stopped (failed)` |
| `vv` no longer runs `page-journey` | `cargo test -p repo-conformance --test gpu gpu_gp_03` | rejected: `vv does not run page-journey` |
| `pages.yml` runs the journey script directly instead of the recipe | the same | rejected: `pages.yml runs the page-journey recipe` |
| The page loads without comparing its kernel with the deploy's listing | `just page-journey` | rejected: `stale: expected to stop with "stale kernel" at the deploy comparison, stopped at undefined: undefined (ready: null)` and `good: the kernel was not compared with the deploy's listing (no check)` |
| the same | `cargo test -p repo-conformance --test gpu gpu_gp_06` | rejected: `load compares the kernel` |
| The prompt box starts with text in it | `just page-journey` | rejected: `good: the prompt box started with text, so typing would join it: "Tell me"` |
| Qwen2's q, k and v biases never added | `cargo test -p repo-conformance --test gpu gpu_gp_05` | rejected: `q4 Qwen2: step 0: a logit differs by 0.70114999954666 from the reference` |
| The q4 matrix kernel unpacks codes one too high (code - 7) | the same | rejected: `q4 Qwen2: step 0: a logit differs by 1.406732768207285 from the reference` |
| The batched q4 kernel swaps a word's low and high nibble halves | the same | rejected: `q4 Qwen2: a 6-token prompt: batched tokens differ from one at a time` |
| Attention's combine reads only the first 64-position chunk | `cargo test -p repo-conformance --test gpu gpu_gp_01` | rejected: `ternary Qwen3: past one attention chunk, a logit differs by 5.581626157661233 from the reference` |
| the same | `cargo test -p repo-conformance --test gpu gpu_gp_05` | rejected: `q4 Qwen2: past one attention chunk, a logit differs by 0.7724276952619027 from the reference` |
| The page never removes a cached file that differs from the deploy's listing | `just page-journey` | rejected: `earlier: the page loaded 1 times, expected one reload`, `earlier: the earlier deploy's engine file is still in the cache` and `again: expected "Stopped: stale page" and no Load button` |
| the same | `cargo test -p repo-conformance --test gpu gpu_gp_07` | rejected: ``site/app/index.html no longer has `if (held !== want) {` `` |
| With a reload already spent and a file still stale, the page imports the engine instead of stopping | `just page-journey` | rejected: `again: expected "Stopped: stale page" and no Load button, saw Stopped: stale page: 1 cached files are not this deploy's after a reload. Clear this site's data and open the page again. (disabled: false)` |
| The f16 scale read takes the other half of each word | `node crates/conformance/tests/gpu/run.mjs .` on SwiftShader, the runner `gpu_gp_08` calls (cargo was not run on that machine) | rejected: the q4f16 model's worst logit differs by 2.80 from the reference and its picks differ; q4 unchanged at 1.2e-4 |
| The engine as on main (no `parts`, no q4f16), GP-08 written first | the same | rejected: `TypeError: Cannot read properties of undefined (reading 'byteLength')` |
| The f16 embedding read takes the other half of each word | `node crates/conformance/tests/gpu/run.mjs .` on SwiftShader, the runner `gpu_gp_09` calls (cargo was not run on that machine) | rejected: the dense q4f16 model's worst logit differs by 3.93 and its picks differ; dense q4 unchanged at 1.4e-4 |
| The dense head writes each tile's rows from row 0, ignoring the tile's first row | the same | rejected: both dense models' worst logit differs by 3.12 and their picks differ |
| The engine as on main (no dense formats), GP-09 written first | the same | rejected: `embed is f32; this engine runs q1, q4 and q4f16 matrices` |
| The f16 vector widening reads each exponent one too high | `node crates/conformance/tests/gpu/run.mjs .` on SwiftShader, the runner `gpu_gp_10` and `gpu_gp_11` call (cargo was not run on that machine) | rejected: the Qwen2.5-shaped model's worst logit differs by 3.6 and the Llama-shaped one's by 4.6, picks differ |
| Llama 3's blend weights swapped | `node crates/conformance/tests/gpu/yarn.mjs . llama3 64 500000 32 1 4 8192`, as `gpu_gp_11` runs it | rejected: pairs 15, 16 and 17 differ from the formula by 30%, 140% and 800%; the Llama-shaped model's worst logit 0.31 |
| The engine ignores Llama 3 scaling and uses plain frequencies | the runner | rejected: the Llama-shaped model's worst logit 0.22 and its picks differ |
| The engine before this change, GP-10 and GP-11 written first | the runner and the rotary script | rejected: `l0.attn_norm is f16, not f32`; `llama3 is not a function` |
| The role map swaps gate and up | `node crates/conformance/tests/gpu/run.mjs .` on SwiftShader, as `gpu_gp_12` runs it (cargo was not run on that machine) | rejected: every count on the real graphs is unchanged (338, 310, 146), but the placed models' worst logits differ by 1.21 and 1.47 and their picks differ |
| The role map lets through an operator that reads a weight no role takes | `node crates/conformance/tests/gpu/roles.mjs .`, as `gpu_gp_12` runs it | rejected: the `unplaced` case is not refused (null) |
| No role map, as before this change, GP-12 written first | the same | rejected: `ERR_MODULE_NOT_FOUND` |
| The encoder's GELU is the tanh approximation instead of erf | `node crates/conformance/tests/gpu/encode.mjs .` on SwiftShader, the runner `gpu_gp_13` and `gpu_gp_14` call (cargo was not run on that machine) | rejected by GP-13: hidden states differ by 2.0e-4 and 2.1e-4, twice the tolerance. GP-14 alone would pass it: its label scores move by 2e-5 |
| Encoder attention made causal, each token seeing only those before it | the same | rejected: hidden states differ by 0.58 and 0.77, label scores by 0.14 and 0.17 |
| The token-type row never added | the same | rejected by GP-13: hidden states differ by 1.7 and 2.6; GP-14's model has no token types and is unchanged, as it should be |
| No encoder, as before this change, GP-13 and GP-14 written first | the same | rejected: `TypeError: Failed to fetch dynamically imported module` |
| none | all fourteen, and the journey | pass |
