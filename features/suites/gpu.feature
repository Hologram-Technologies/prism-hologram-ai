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
    When the deploy's listing names a different BLAKE3 for app.holo than the kernel the browser was handed, as when the service worker serves an earlier deploy's kernel from its cache
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
    Then before importing the engine the page compares every cached file the listing names with the BLAKE3 listed, removes the one that differs, keeps the one that matches, and reloads once
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

  @GP-12 @build
  Scenario: Every weight of onnx-community's q4f16 exports of Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B is given its engine role by the node that reads it, with shapes checked against the config, what cannot be placed is refused by name, and a model placed this way gives the CPU reference's greedy tokens
    Given the graphs of the three q4f16 exports at their hub revisions, recorded in crates/conformance/tests/gpu/exports as their nodes and tensor shapes with their config.json, and four rows of Llama's cos and sin cache
    When site/app/engine/onnx-roles.mjs places each graph's weights by the names of the nodes that read them
    Then Qwen2.5 has 338 placed tensors, Qwen3 310 and Llama 146, each exactly the engine's roles for its config, with the head tied to the embedding
    And Llama's cache confirms Llama 3 scaling at factor 32, low 1, high 4, original 8192
    And a missing weight, a duplicate, an operator no role takes, zero points, a block size of 64, a head count the shapes contradict, an unknown family, too few layers, a cache no parameters match and interleaved rotary are each refused, naming what is wrong
    And GP-10's and GP-11's models, written as a decomposed and a fused export, placed from their graphs alone and run from the tensors the map names, give the reference's tokens with every logit within 0.002, and within 0.005 over 140 positions

  @GP-13 @build
  Scenario: A BERT encoder shaped as all-MiniLM-L6-v2's q4f16 ONNX export, with q4f16 weights, f16 biases, LayerNorms and embedding tables and token types, gives on WebGPU the CPU reference's hidden states and its mean-pooled unit-length sentence embedding within tolerance
    Given a two-layer encoder generated from a fixed seed in the formats Xenova's q4f16 export of all-MiniLM-L6-v2 holds: 4-bit weights as MatMulNBits' tensors with f16 scales, f16 biases and LayerNorm weights and biases, f16 word, position and token-type tables, LayerNorm epsilon 1e-12
    When site/app/engine/encoder.mjs runs a sentence of 6 tokens and one of 60 in headless Chromium on WebGPU's software renderer, beside the double-precision reference in crates/conformance/tests/gpu/encoder-reference.mjs
    Then every hidden state is within 1e-4 of the reference's
    And the mean of the token states scaled to unit length is within 1e-4 of the reference's, cosine above 0.99999

  @GP-14 @build
  Scenario: A DistilBERT classifier shaped as distilbert-base-uncased-finetuned-sst-2-english's q4f16 ONNX export, with no token types and an f16 dense head through ReLU, gives on WebGPU the CPU reference's label scores within tolerance and the same label
    Given a two-layer encoder generated from a fixed seed in the formats of Xenova's q4f16 export of distilbert-base-uncased-finetuned-sst-2-english: no token types, and a head of an f16 768-wide dense layer, ReLU and an f16 two-label dense layer, both with biases
    When the encoder classifies a sentence of 6 tokens and one of 60, which the reference labels differently
    Then each label score is within 1e-4 of the reference's and each probability within 1e-4
    And the engine picks the reference's label for both

  @GP-15 @build
  Scenario: Tensors are kept in the browser once each by their BLAKE3 address: a second model stores only the tensors the first lacks, a return visit with the network off reads every tensor back admitted again and runs to the reference's tokens, and a stored tensor whose bytes changed is refused and removed
    Given GP-10's model A, 40 blocks of tensor bytes, and a model B that is A with layer 1's feed-forward from another seed, five of its 40 blocks new
    When a first visit stores A, then B, then A again, through site/app/engine/store.mjs in the browser's private file system, each block under the blake3: label of its bytes
    Then A stores 40 blocks, B stores only its 5 new ones and holds the 35 it shares, 212,992 bytes not written again, and A again stores nothing
    And a second visit to the same origin, with the network off, reads all 40 of A's blocks back, each admitted again before use, and the engine run from them picks the reference's token at each of sixteen steps with every logit within 0.002
    And a stored block changed by one bit is not admitted, is removed and reads as missing, and a sha256: label is refused by its prefix

  @GP-16 @build
  Scenario: A prompt's computed prefix, kept on the GPU and restored into the same engine after its cache was overwritten or into a fresh engine of the same model, gives bit for bit the logits and tokens of computing the whole prompt
    Given GP-10's and GP-11's models, a 40-token prefix longer than one submission, and an 8-token suffix
    When one engine computes prefix and suffix and 8 greedy picks position by position, and another computes the prefix, keeps it with saveState, starts again from position 0 with 50 other tokens that overwrite every row the prefix held, restores the prefix and continues, and a third, fresh, is given the second's prefix and continues
    Then every logit at the 16 positions after the prefix is the same bits in all three
    And running the prefix in blocks, keeping it, resetting and running 50 other tokens, restoring and running the suffix picks the same 12 tokens as running prefix and suffix together
    And keeping more positions than were computed, or restoring a prefix saved from another shape of model, is refused

  @GP-17 @build
  Scenario: Every weight of Xenova's q4f16 exports of all-MiniLM-L6-v2 and distilbert-base-uncased-finetuned-sst-2-english is given its encoder role by the node that reads it, with shapes, activation and labels checked against the config, what cannot be placed is refused by name, and an encoder placed this way gives the CPU reference's embedding and labels
    Given the graphs of the two exports at 751bff3 and 0b6928e, recorded in crates/conformance/tests/gpu/exports as their weight-reading nodes and tensor shapes with their config.json
    When site/app/engine/onnx-roles.mjs places each with encoderRoles
    Then all-MiniLM-L6-v2 has 101 tensors placed, 36 of them q4f16, and DistilBERT 104 with its classifier head and its labels NEGATIVE and POSITIVE
    And an activation other than exact GELU, an unknown family, a missing bias, a head Gemm not transposed, a config with three labels, an operator no role takes and a feed-forward width the shapes contradict are each refused, naming what is wrong
    And GP-13's and GP-14's encoders, written as these exports name their nodes and placed from their graphs alone, give the reference's embedding within 1e-4 and its labels

  @GP-18 @build
  Scenario: With WebGPU off, the CPU engine runs q4 Qwen2, Qwen2.5's, Llama's and Qwen3's q4f16 export shapes from their own bytes and gives the CPU reference's greedy tokens with every logit within 1e-5, by steps, in a run and after restoring a prefix
    Given GP-05's q4 model, GP-10's Qwen2.5-shaped and GP-11's Llama-shaped models, and the same with Qwen3's shape: no biases and a norm on each query and key head
    When site/app/engine/cpu.mjs runs each in headless Chromium launched with its GPU disabled, where no WebGPU adapter is offered, beside the double-precision reference
    Then at each of sixteen steps the engine picks the reference's token and every logit is within 1e-5, and over 140 positions every logit is within 1e-5
    And a 40-token prompt run with run() picks the same 12 tokens as stepping it one position at a time
    And a prefix kept, overwritten by 50 other tokens and restored continues with the tokens of running the whole prompt

  @GP-19 @build
  Scenario: The engine page reads an ONNX export named on the hub from its own files: each small file checked against the record, the graph imported through the module with the kernel admitting every span, operator, file and 4-bit weight, every weight placed by the node that reads it, and it answers as the reference does on WebGPU and on the CPU, and stops where a file is not the record's
    Given a Qwen3-shaped model written as onnx-community's fused q4f16 export writes one, with its config.json, generation_config.json and a byte-level tokenizer.json, and a hub record and index root naming each file by its SHA-256
    When `just page-journey` opens the engine page with ?onnx= naming it, as `vv` does after `app` and `engine`, once with WebGPU and once with navigator.gpu taken away
    Then the page checks each small file against the record, imports the graph through the module asking the kernel, places every weight, reads the stop id from generation_config.json, reaches ready on WebGPU and on the CPU, and answers with the reference's first sixteen tokens
    And a graph with one byte flipped is refused by the kernel at the whole file, before any weight is placed
    And a record naming another SHA-256 for tokenizer.json stops the page at that file
    And a record that names config.json and generation_config.json by their Git blob SHA-1 alone, as the live hub's do, loads, each checked by that hash, and one naming another blob stops the page at that file

  @GP-20 @build
  Scenario: The chat templates of Qwen2.5-1.5B's, Qwen3-0.6B's and Llama 3.2 1B's exports render, from the model's own tokenizer_config.json, exactly as Python's jinja2 renders them configured as transformers configures it, including a template's own error, and a template using a tag, filter, test or method outside what the engine follows is refused by name
    Given the three exports' chat templates at their hub revisions and twenty conversations each, recorded in crates/conformance/tests/gpu/templates with what Python's jinja2 3.1.6 rendered for them in transformers' sandboxed environment (trim_blocks, lstrip_blocks, its tojson, raise_exception, strftime_now at a fixed date): one turn, a system turn, several turns, Unicode, Qwen3's thinking and thinking off, tool definitions, tool calls and results, a system turn after the first, a given date, tools outside the first user turn, and tools with no user turn
    And a fourth template written to reach what theirs do not, tags with no dash under lstrip_blocks and trim_blocks, a tag opting out with a plus, a trailing newline and nested loops, rendered the same way
    When site/app/engine/jinja.mjs compiles each template and renders each conversation
    Then every text is the one jinja2 rendered, byte for byte, and Llama's template raises its own error where jinja2 raised it
    And a macro, an include, the filter upper, the string method title, the test odd and an unclosed if are each refused, naming what is outside
    And the engine page frames an ONNX export's turn with the model's own template, read from tokenizer_config.json and checked against the record

  @GP-21 @build
  Scenario: The engine page keeps an ONNX export's files in the browser, each named by its BLAKE3 with a pointer from the SHA-256 the hub's record names, and a later visit reads them from there with the origin serving none, each file's BLAKE3 derived again as it is read and its bytes checked against the record and imported through the kernel exactly as a download's are; a kept file that is not its BLAKE3 is refused, and the load forgets every file it read, so the next visit downloads them again
    Given GP-19's Qwen3-shaped ONNX export, its files and a hub record naming each by its SHA-256, served to four visits in one browser profile
    When `just page-journey` opens the engine page with ?onnx= naming it: first with the origin serving the files, then with the origin serving the record and no file, then with the kept graph's bytes replaced by a graph with one byte flipped, then with the origin serving the files again
    Then the first visit downloads every file, keeps each named by its BLAKE3 and answers with the reference's tokens
    And the second reads every file from the browser, downloads none, derives each file's BLAKE3 again, checks each against the record, imports the graph through the kernel and answers with the reference's tokens
    And the third finds the kept graph is not its BLAKE3, stops, and forgets every file it read
    And the fourth downloads every file again and answers with the reference's tokens

  @GP-22 @build
  Scenario: The engine page asks the kernel which precision of an ONNX export to read (decision 21), what each of its files is called (decision 24) and which device runs it (decision 22), and reads, names and runs exactly what the kernel answers; a precision the hub's record does not offer is refused by the kernel before any file is read
    Given GP-19's Qwen3-shaped ONNX export, whose hub record offers its q4f16 files and no q4 files
    When `just page-journey` opens the engine page with ?onnx= naming it, with WebGPU and with navigator.gpu taken away, and once more asking for ?precision=q4
    Then each load asks the kernel for the precision and is answered q4f16, asks it for the graph's file and reads the file it names, and asks it for the device and runs on the one it names: WebGPU, then the CPU
    And the load asking for q4 stops at the precision, the kernel answering none fits, with no file read

  @GP-23 @build
  Scenario: The engine page keeps the computed state of its chat frame's fixed head, the tokens every turn opens with, under the BLAKE3 of those token ids, and on a later turn asks the kernel (decision 27) whether that label equals the label of the new prompt's own opening ids; on reuse it restores the head and computes only the rest, and the turn's tokens equal those of computing the whole prompt
    Given GP-19's Qwen3-shaped ONNX export, whose chat template opens every turn with the same system line
    When `just page-journey` loads it on WebGPU and on the CPU and sends a first turn, then a second whose text opens with another letter
    Then the first turn computes the whole prompt and asks nothing about reuse
    And the second asks the kernel with the label of its own opening ids, which the journey derives again from those ids, is answered reuse, computes only the positions after the head, and gives the reference's tokens for the whole prompt

  @GP-24 @build
  Scenario: When the browser has too little room to keep a downloaded file, the engine page asks the kernel (decision 26) about each file already kept, oldest first, with the files this load uses pinned, drops a file only on evict and stops at enough; a file there is no room for on does not fit is not kept, and the model still loads from the download
    Given a browser profile holding one 300,000-byte file kept for another model, named by its BLAKE3, and a browser reporting no free storage
    When `just page-journey` opens the engine page with ?onnx= naming GP-19's export
    Then the kernel answers evict for the other model's file and enough after it, and the first file downloaded is kept
    And for every later file the kernel answers keep for the pinned file and does not fit, and the file is not kept
    And the page answers with the reference's tokens
