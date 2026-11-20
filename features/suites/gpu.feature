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
  Scenario: The engine page keeps an ONNX export's files in the browser, each under the BLAKE3 the hub's record names for it, and a later visit reads them from there with the origin serving none, each file's BLAKE3 derived again as it is read and its bytes checked against the record and imported through the kernel exactly as a download's are; a kept file that is not its BLAKE3 is refused, and the load forgets every file it read, so the next visit downloads them again
    Given GP-19's Qwen3-shaped ONNX export, its files and a hub record naming each by its BLAKE3, served to four visits in one browser profile
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

  @GP-25 @build
  Scenario: One call, pipeline(task, model, { device, dtype }), loads an ONNX export named on the hub and answers a function that runs a turn, with every step the engine page takes put to the verified kernel; the page is one caller of it, and a task the kernel does not run is refused by the kernel's decision 24
    Given GP-19's Qwen3-shaped ONNX export and its hub record
    When `just page-journey` imports site/app/engine/pipeline.mjs on a page and calls pipeline("text-generation", the export, { index }) with no page code in between, then calls the function it answers with "hi"
    Then the kernel is verified, each file is named, the precision and device are chosen and every span admitted by the kernel, and the turn answers with the reference's tokens
    And pipeline("summarization", the export) stops at the kernel's answer unsupported task, with no file read
    And the engine page's own visits load and answer through the same call

  @GP-28 @build
  Scenario: The engine page runs an ONNX export on its CPU device on the integer engine: the pipeline hands the module every weight the import read and the config's head counts, rope theta, epsilon and scaling, and each token is the module's decode_step, so the CPU run's tokens are the integer forward pass's, the second turn carrying on from the run the module kept
    Given GP-19's Qwen3-shaped ONNX export on the hub, and the engine page with navigator.gpu taken away
    When `just page-journey` loads it at ?onnx= and asks two turns
    Then the page reports the integer engine on the CPU device, and its sixteen tokens are the reference's
    And the pipeline builds the CPU device's engine with createIntegerEngine from the import's tensors and the config, never the float CPU engine

  @GP-26 @build
  Scenario: The engine page checks every file of an ONNX export by the BLAKE3 the hub's record names for it, the small files in the module and the graph and data files through the importer, and refuses by name a record that names a file no BLAKE3; no file is checked by SHA-256 or SHA-1
    Given GP-19's Qwen3-shaped ONNX export and hub records that name each file by its BLAKE3, beside the SHA-256 Hugging Face publishes
    When `just page-journey` opens the engine page with ?onnx= naming it
    Then each small file is checked by its BLAKE3 in the module, and the graph is admitted by the kernel against its BLAKE3
    And a record that names config.json no BLAKE3 stops the page by name, before anything is imported
    And no file check names SHA-256 or SHA-1

  @GP-27 @build
  Scenario: The engine page reads a model's hub record through the hub's own client, vendored byte for byte from hub commit ec26789 and pinned by BLAKE3: the index root is checked by its Sigstore signature against the trust policy this repository pins, the record by the BLAKE3 the root names, and a root signed by a key the policy does not name, or for another identity, or changed after signing, is refused as RootUnverified naming the failed checks, as each other failure is by its code
    Given a Sigstore-shaped chain minted for the test, an intermediate, a log key and the policy naming them, and the fixture hubs' index roots signed through it
    When hub-check.mjs drives the vendored client in Node, and `just page-journey` opens the engine page on hubs signed through the chain and through a foreign one
    Then the root, the record and a file are checked, and the page shows the index root signature checked
    And a root signed through a foreign intermediate and log key fails chain and log, another log key fails log, another identity fails identity and a changed root fails subject, each refused as RootUnverified
    And the page stops at the index root signature on the foreign hub, before any record is read
    And a missing bundle, an unknown or changed or unaddressed or unreachable record, and a changed or unaddressed file are refused by their codes

  @GP-30 @build
  Scenario: A return visit to the engine page with the network off loads the page, its engine and its kernel from the page's own service worker, reads the hub's index root, its signature and the model's record from this browser and checks them again as a download is checked, reads every file and tensor of the model from this browser with decision 28 asked again on the importer's own addresses, and answers with the reference's tokens; a kept record changed by one byte is refused as RecordMismatch, and with no hub index kept the page stops at the index root as RootUnreachable
    Given GP-19's Qwen3-shaped ONNX export and its hub record, served to four visits in one browser profile
    When `just page-journey` opens the engine page with ?onnx= naming it once with the network on, then with the server answering nothing and the browser offline: once as it is, once with the kept record changed by one byte, and once with the kept hub index removed
    Then the first visit answers with the reference's tokens and leaves the page's worker active
    And the second loads the page from its worker, proves the kept index root by its signature and the kept record by its BLAKE3, says the network is off where it would compare the deploy's listing, reads every file and every tensor from the browser with no read of the network, is answered ready by decision 28, and answers with the reference's tokens
    And the third stops at the record as RecordMismatch and the fourth at the index root as RootUnreachable, neither reading a file

  @GP-31 @build
  Scenario: The engine page keeps an ONNX export's tensors in the browser once each by the BLAKE3 address the hub's UOR index lists: each span is asked of decision 23, a tensor already held under any model, precision or repository is read from the browser and not downloaded, only the bytes not held are range-read from the first of the record's sources that answers, and nothing new is kept until decision 28 answers ready on the importer's own addresses; the same export under a second repository downloads nothing, its q4 graph after its q4f16 graph downloads only the tensors the first lacks, a kept tensor changed by one byte is refused and forgotten, a model the UOR index does not list is refused by name before any file is read, and an index giving a tensor another address is refused by decision 28
    Given GP-19's Qwen3-shaped ONNX export with its q4 graph beside its q4f16 graph, whose 4-bit codes are the same bytes, a second repository holding the same files whose origin answers nothing and whose one listed source does, and the hub's UOR index for both computed from the files as the hub computes it
    When `just page-journey` loads the export, then the second repository, then the second repository's q4 graph, then the export after one byte of a kept tensor is changed, then the export again, in one browser profile; and in fresh profiles a hub whose UOR index does not list the export and one whose index gives a tensor another address
    Then the first load downloads every tensor in one read and keeps each under its address
    And the second repository reads every tensor from the browser and downloads nothing of the graph
    And its q4 graph reads from the browser the tensors the q4f16 graph shares and downloads from the listed source only the others, and answers with the q4 model's reference tokens
    And the changed tensor stops the load as a file not its BLAKE3, and is forgotten, and the next load downloads it again and answers
    And the unlisted export is refused by name before any file is read, and the lying index is refused by decision 28 with nothing kept

  @GP-32 @build
  Scenario: Prefix reuse is token-identical and timed: on WebGPU and on the CPU, a second turn of the same text is answered reuse by decision 27 for the chat frame's head, computes only the positions after it, and gives exactly the first turn's tokens, each turn reporting its time to first token
    Given GP-19's Qwen3-shaped ONNX export, whose chat template opens every turn with the same system line
    When `just page-journey` loads it on WebGPU and, with navigator.gpu taken away, on the CPU, and sends "hi" twice
    Then the first turn computes the whole prompt and asks nothing about reuse
    And the second is answered reuse for the head, computes only the positions after it, and its tokens are exactly the first turn's, which are the reference's
    And each turn reports its time to first token

  @GP-35 @build
  Scenario: pipeline("feature-extraction", model) loads a BERT encoder exported as Xenova's all-MiniLM-L6-v2 q4f16 export is, from its hub record and its own files, imports it with every operator asked of decision 18's encoder style and its file named by decision 24, places its weights by the node that reads them, int64 position ids left out, and answers each sentence's WordPiece ids and unit-length embedding from the module's integer encoder, bit for bit hologram-ai-cpu-encode's, on the CPU device decision 22 chose
    Given GP-13's encoder written as Xenova's all-MiniLM-L6-v2 export is, with its int64 position ids, a WordPiece tokenizer and a hub record and UOR index of its own
    When `just page-journey` calls pipeline("feature-extraction", the export, { index }) on a page and runs it on each sentence of ENCODER_TEXTS
    Then the device is the CPU, the precision q4f16, a check names the module's integer encoder, and every sentence's ids and embedding bits are encoder-expected.json's

  @GP-36 @build
  Scenario: pipeline("text-classification", model) loads a DistilBERT classifier exported as Xenova's SST-2 q4f16 export is and answers each sentence's label, chosen by the module by comparing the scores' bits as integers and named by the config's id2label, and the scores, bit for bit hologram-ai-cpu-encode's
    Given GP-14's classifier written as Xenova's SST-2 DistilBERT export is
    When `just page-journey` calls pipeline("text-classification", the export, { index }) and runs it on each sentence
    Then every sentence's ids, label, label name and score bits are encoder-expected.json's

  @GP-40 @build
  Scenario: Code written for transformers.js runs on the engine unchanged: pipeline(task, model, { device, dtype, progress_callback }) loads through decision 70 and the one call; a messages list answers [{ generated_text: [...messages, { role: assistant, content }] }] with the one call's text, a string answers [{ generated_text: prompt + completion }], a TextStreamer receives the answer as it is made, and an fp32 dtype, an unknown device or do_sample are refused by name by decision 70 before anything loads or runs
    Given GP-19's Qwen3-shaped ONNX export and its hub record, served to one page
    When `just page-journey` imports site/app/engine/transformers.mjs on the engine page and calls it as transformers.js is called
    Then pipeline("text-generation", repo, { device, dtype, progress_callback }) is answered by decision 70 before it loads, and progress_callback reports ready
    And a messages list answers [{ generated_text: [...messages, { role: "assistant", content }] }] with the one call's text, and a TextStreamer receives that text as it is made
    And a string answers [{ generated_text: prompt + completion }]
    And an fp32 dtype, an unknown device and do_sample are each refused by decision 70 by name, the dtype before any record or file is read

  @GP-41 @build
  Scenario: The integer engines run off the page's thread: the pipeline hands a worker a copy of the page's module and every weight the import placed, the worker computes every decode and encode step, the page's own thread calls none of them and answers throughout a turn, each token reaches the page as its own message, the tokens are the reference's and the embedding's bits GP-35's
    Given GP-19's Qwen3-shaped ONNX export and GP-35's BERT export, served to one page that counts every call its own thread makes into a module's forward pass or encoder
    When `just page-journey` calls pipeline on the CPU device, asks one turn while a heartbeat runs on the page's thread, and embeds GP-35's sentences
    Then the page's own thread made no decode or encode call, and the heartbeat was never unanswered for 500 ms
    And onToken was called once per token, and the sixteen tokens are the reference's
    And every embedding's bits are GP-35's

  @GP-50 @build
  Scenario: Before the page instantiates engine.wasm it computes the bytes BLAKE3 and asks decision 80: the sealed engine loads, and an engine with one byte flipped is refused by name, not the sealed engine, before it is compiled
    Given the assembled page, and the same page serving engine.wasm with one bit of one byte flipped
    When the page opens its engine through openModule, which every caller reaches
    Then the page records the engine check passing with the kernel's answer sealed, and loads the container
    And with the flipped engine it stops at the engine check with not the sealed engine, before the module is compiled and before any container byte is read

  @GP-51 @build
  Scenario: The integer engine computes no output head for a prompt id whose logits nothing reads: run sends every prompt id but the last through the module's decode_advance and the last through decode_step, and the page's tokens on the CPU device, in the worker and after a restored prefix, are the reference's, bit for bit as before
    Given the page's integer engine (site/app/engine/integer.mjs) and GP-19's Qwen3-shaped ONNX export
    When `just page-journey` runs a turn on the CPU device, in the worker, and a second turn on a restored prefix
    Then integer.mjs's run calls decode_advance for every prompt id but the last and decode_step for the last and each generated token
    And the sixteen tokens are the reference's, and the second turn's are the first's

  @GP-52 @build
  Scenario: On a cross-origin-isolated page the integer engine's worker hands every product to helper workers, each a fresh instance of the same engine.wasm holding one slice of each matrix's rows, the input and the rows passed through a SharedArrayBuffer; the engine page's service worker serves this deploy's files with Cross-Origin-Opener-Policy same-origin and Cross-Origin-Embedder-Policy credentialless, so a return visit is isolated; the tokens with helpers are the reference's, and a page that is not isolated runs the engine without helpers and gives the same tokens
    Given GP-19's Qwen3-shaped ONNX export and the engine page's service worker
    When `just page-journey` loads the page, reloads it under its worker, and runs a turn on the CPU device
    Then the reloaded page is cross-origin isolated and its engine ran with helpers, and the first page was not and ran without
    And both turns' sixteen tokens are the reference's
