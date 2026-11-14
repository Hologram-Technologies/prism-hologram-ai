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
| `GP-04`: many positions per submission, each pick written on the GPU and read by the next position, give the same tokens as one position per submission | `gpu_gp_04`: twelve tokens after a six-token prompt and after a forty-token prompt, which crosses a submission's thirty-two slots; every position runs exactly once | tested |
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
| none | all three | pass |
