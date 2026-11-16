Feature: gpu

  The browser engine's WebGPU computation, checked against references that share no code with it.

  @GP-01 @build
  Scenario: The WebGPU forward pass of a ternary Qwen3 model gives the same greedy tokens as the CPU reference, with every logit within a stated tolerance
    Given a two-layer ternary Qwen3 model with Bonsai's tensor names, generated from a fixed seed, and a six-token prompt
    When the engine in site/app/engine runs it in headless Chromium on WebGPU's software renderer, beside the double-precision reference in crates/conformance/tests/gpu/reference.mjs
    Then at each of sixteen steps the engine picks the reference's token
    And every logit is within 0.002 of the reference's
    And over a run of 140 positions, more than two of attention's 64-position chunks, the picks agree and every logit is within 0.005

  @GP-02 @build
  Scenario: The rotary frequencies are the YaRN formula, with the pairs that keep their frequency and the pairs that are scaled computed from the model base, original context and head size
    Given Bonsai's rotary parameters: head size 128, base 1000000, factor 4, original context 8192
    When the engine's yarn module computes the frequencies
    Then the pairs below 17 keep their trained frequency, the pairs above 34 turn at a quarter of it, and the pairs between follow the linear ramp
    And the rotated vectors are scaled by 1 + 0.1 ln 4

  @GP-04 @build
  Scenario: Running many positions in one submission, with each pick written on the GPU and read by the next position, gives the same tokens as running them one at a time
    Given the same generated model and two prompts, one of six tokens and one of a hundred, longer than a submission's thirty-two slots and than one 64-position attention chunk
    When one engine runs them one position per submission and another many positions per submission, each pick written to the ring on the GPU and read by the next position
    Then both give the same twelve tokens for each prompt
    And every position ran exactly once

  @GP-03 @build
  Scenario: The engine page loads a weight container through the verified kernel and answers as the reference does, and stops where a container is tampered with
    Given a weight container in Bonsai's layout built from the generated model, with its output head naming the embedding's bytes, and a hub record and index root that name it
    When `just page-journey` drives the published page, with the holo and engine this workspace built, in headless Chromium, as `vv` does after `app` and `engine` and `pages.yml` does before it deploys
    Then the page reaches ready with every check passed, confirms the output head is the embedding's bytes, and answers with the reference's first sixteen tokens
    And a container with one flipped byte stops at that block's admission, before anything is uploaded
    And a record naming a wrong whole-file SHA-256 stops at the root
    And the prompt box starts empty, so typed text is the whole prompt, and Enter in it sends

  @GP-06 @build
  Scenario: The engine page stops with stale kernel before asking the kernel anything when the kernel it verified is not the one its deploy lists
    Given the assembled site, with the precache.json the deploy writes, and the journey's container
    When the deploy's listing names a different SHA-256 for app.holo than the kernel the browser was handed, as when the service worker serves an earlier deploy's kernel from its cache
    Then the page stops at "kernel is the one this deploy lists" with an error beginning "stale kernel", naming both, before it reads the record or asks the kernel any question
    And with the listing matching, the comparison passes and the load goes on

  @GP-05 @build
  Scenario: The WebGPU forward pass of a q4 Qwen2 model, with attention biases and no per-head norm, gives the same greedy tokens as the CPU reference with every logit within tolerance past one attention chunk, and a prompt run in blocks gives the same tokens as one position at a time
    Given a two-layer q4 Qwen2 model generated from a fixed seed, with q, k and v biases, no per-head norm, plain rotary frequencies and an output head separate from the embedding
    When the engine runs it in headless Chromium on WebGPU's software renderer, beside the reference that unpacks q4 by KN-17
    Then at each of sixteen steps the engine picks the reference's token, and every logit is within 0.002 of the reference's
    And over a run of 140 positions the picks agree and every logit is within 0.005
    And prompts of six and a hundred tokens run in blocks give the same tokens as one position at a time

  @GP-07 @build
  Scenario: A page file held in the cache from an earlier deploy is removed and the page reloads once before the engine is imported, and a file still stale after that stops the page
    Given the assembled site, with the precache.json the deploy writes, and a cache named as the service worker names its own that holds one file of an earlier deploy and one of this deploy
    When the engine page is opened, which the worker always fetches from the network
    Then before importing the engine the page compares every cached file the listing names with the SHA-256 listed, removes the one that differs, keeps the one that matches, and reloads once
    And after the reload it says files from an earlier deploy were replaced and imports the engine
    And when a reload has already been spent and a cached file still differs, it stops with "Stopped: stale page", disables Load and does not import the engine

  @GP-08 @build
  Scenario: A q4 Qwen2 model whose weights are given as ONNX MatMulNBits tensors, with f32 or f16 scales, runs on WebGPU from those bytes with no repack and gives the CPU reference's greedy tokens with every logit within tolerance
    Given GP-05's generated q4 Qwen2 model, with each weight matrix written as MatMulNBits holds it at 4 bits, block size 32 and no zero points: B as [N][K / 32][16] bytes, low nibble first, and scales as [N][K / 32], once in f32 as a q4 export has them and once in f16 as a q4f16 export has them
    When the engine in site/app/engine runs each from those two tensors as given, in headless Chromium on WebGPU's software renderer, beside the double-precision reference in crates/conformance/tests/gpu/reference.mjs
    Then the f32 export's tensors are byte for byte the engine's q4 codes and scales
    And for both exports, at each of sixteen steps the engine picks the reference's token, every logit is within 0.002 of the reference's, and over 140 positions every logit is within 0.005
    And a prompt run in blocks gives the same tokens as one position at a time

  @GP-09 @build
  Scenario: A q4 Qwen2 model whose embedding is a dense f32 or f16 matrix with the output head tied to it, as ONNX exports hold them, gives the CPU reference's greedy tokens with every logit within tolerance, with the head split into tiles when it is larger than one binding
    Given GP-08's two models with the embedding written as onnx-community's exports of Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B hold it, one dense [vocab][width] matrix, f32 in a q4 export and f16 in a q4f16 one, read by a Gather and, through a Transpose, by the head's MatMul, with no head tensor of its own
    When the engine runs each with its binding limit set to 15,000 bytes, so the head is split into tiles of whole rows and the last tile is short
    Then the f32 head is in twelve tiles and the f16 head in six
    And for both, at each of sixteen steps the engine picks the reference's token, every logit is within 0.002 of the reference's, and over 140 positions every logit is within 0.005
    And a prompt run in blocks gives the same tokens as one position at a time

  @GP-10 @build
  Scenario: A q4 Qwen2 model shaped as Qwen2.5's q4f16 ONNX export, with q4f16 weights, a dense f16 embedding and tied head, and norm weights and biases in f16, gives the CPU reference's greedy tokens with every logit within tolerance
    Given GP-09's q4f16 model with its norm weights and its q, k and v biases written in f16, as onnx-community's q4f16 export of Qwen2.5-1.5B holds them
    When the engine runs it from those bytes, widening each f16 vector exactly
    Then the model's vectors reached the engine as f16 and it has its biases
    And at each of sixteen steps the engine picks the reference's token, every logit is within 0.002 of the reference's, and over 140 positions every logit is within 0.005
    And a prompt run in blocks gives the same tokens as one position at a time

  @GP-11 @build
  Scenario: The rotary frequencies of Llama 3's scaling are transformers' llama3 formula for Llama 3.2's parameters, and a Llama-shaped model with that scaling and no biases gives the CPU reference's greedy tokens with every logit within tolerance
    Given Llama 3.2 1B's parameters: head size 64, base 500000, factor 32, low frequency factor 1, high frequency factor 4, original context 8192, which onnx-community's export bakes into its cos and sin caches although its config.json names no scaling
    When the engine's rotary module computes the frequencies
    Then pairs 0 to 14 keep their frequency, pairs 18 to 31 turn 32 times slower, and pairs 15 to 17 blend, every one within one part in a million of the formula, with no attention scale
    And GP-10's model with no biases and Llama 3's scaling at factor 4 over an original context of 64, so its pairs fall in all three bands, gives the reference's tokens with every logit within 0.002, and within 0.005 over 140 positions
    And a prompt run in blocks gives the same tokens as one position at a time
