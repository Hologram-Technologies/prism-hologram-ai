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
| none | all seven, and the journey | pass |
