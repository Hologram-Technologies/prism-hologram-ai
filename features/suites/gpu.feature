Feature: gpu

  The browser engine's WebGPU computation, checked against references that share no code with it.

  @GP-01 @build
  Scenario: The WebGPU forward pass of a ternary Qwen3 model gives the same greedy tokens as the CPU reference, with every logit within a stated tolerance
    Given a two-layer ternary Qwen3 model with Bonsai's tensor names, generated from a fixed seed, and a six-token prompt
    When the engine in site/app/engine runs it in headless Chromium on WebGPU's software renderer, beside the double-precision reference in crates/conformance/tests/gpu/reference.mjs
    Then at each of sixteen steps the engine picks the reference's token
    And every logit is within 0.002 of the reference's

  @GP-02 @build
  Scenario: The rotary frequencies are the YaRN formula, with the pairs that keep their frequency and the pairs that are scaled computed from the model base, original context and head size
    Given Bonsai's rotary parameters: head size 128, base 1000000, factor 4, original context 8192
    When the engine's yarn module computes the frequencies
    Then the pairs below 17 keep their trained frequency, the pairs above 34 turn at a quarter of it, and the pairs between follow the linear ramp
    And the rotated vectors are scaled by 1 + 0.1 ln 4
