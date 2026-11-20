# GPU verification

What each gate over the browser engine's WebGPU computation (`site/app/engine`)
has been seen to reject. A gate is evidence only after a planted defect shows it
can fail ([AGENTS.md](AGENTS.md)); this file records the defect, the command,
the diagnostic and the restored result.

## What is tested, and what is not

| Claim | Evidence | Kind |
| --- | --- | --- |
| `GP-03`: the engine page loads an ONNX export through the verified kernel, Qwen3-0.6B's by default, runs it on the integer engine, answers as the reference does, and stops where the deploy, the hub or the engine is not what it should be | `just page-journey` (run by `vv` after `app` and `engine`, and by `pages.yml` before it deploys) drives the assembled page in headless Chromium; `gpu_gp_03` reads that wiring and the journey's verdicts | tested |
| `GP-07`: a page file held in the cache from an earlier deploy is removed and the page reloads once before the engine is imported, and a file still stale after that stops the page | `just page-journey` plants a cache named as the worker names its own, with one file of an earlier deploy and one of this deploy, and opens the page twice: once fresh, once with the reload already spent; `gpu_gp_07` reads that wiring and the worker's network-first fetch of the page, which the check rests on | tested |
| The deploy's listing, the kernel check and the cached-file check name files by BLAKE3 | `bootstrap/listing-blake3.mjs` adds each served file's `blake3` (the vendored BLAKE3) to the listing the frame's `bootstrap/precache.mjs` writes, which keeps its own SHA-256 and version because the frame is the template's and pinned by `frame.lock` (editing it in place failed the deploy's frame check from 2026-11-18, `9836718`, until this was moved out on the same day); the page compares `app.holo` and `site/app/index.html` compares every cached file with it by BLAKE3. `just page-journey` on 2026-11-17 against a locally assembled site (59 files, 2,679,519 bytes): the kernel check passes as `blake3 5e3cc6bc…`, a listing naming another BLAKE3 for the kernel stops the page, and GP-07's earlier-deploy visits replace the stale file. `gpu_gp_06` and `gpu_gp_07` pin it. Planted: the page reading the listing's old `sha256` field fails the run, though not at a verdict: every load stops at the kernel check, nothing is kept, and the journey ends at its kept-file step (`NotFoundError`) before it prints verdicts Cost not measured: the boot check now hashes cached files in JavaScript (about 21 MB/s measured for this BLAKE3 in Node), where SHA-256 ran in WebCrypto | tested |
| onnx-community's Llama 3.2 1B export uses Llama 3 scaling although its config.json names none | read on 2026-11-16 from `model_q4f16.onnx_data` at revision 82ac693: its `cos_cache` and `sin_cache` rows at positions 1, 7, 1000 and 8191 are within 2.4e-4 (f16 rounding) of Llama 3 scaling at factor 32, low 1, high 4, original 8192, and up to 1.97 away from plain frequencies. So the importer takes the family's parameters, not the config's | imported, measured |
| The attention of the three exports | read on 2026-11-16: all three have separate q, k and v weights, half-split (not interleaved) rotary and the engine's shape. Qwen2.5's export spells attention out in plain operators with tensors named `onnx::MatMul_N`, so its weights are found by the name of the node that reads them; Qwen3's and Llama's use GroupQueryAttention and name weights by layer and role | measured; no kernel gap |
| `GP-12`: every weight of the three q4f16 exports is given its engine role by the node that reads it, shapes checked against the config, the unplaceable refused by name, and a model placed this way gives the reference's tokens | `gpu_gp_12`: `site/app/engine/onnx-roles.mjs` over the three graphs recorded on 2026-11-16 in `crates/conformance/tests/gpu/exports` (nodes, tensor types and shapes, config.json, at the hub's revisions; four rows of Llama's cos and sin cache): 338, 310 and 146 tensors placed, Llama 3 scaling confirmed from the cache (worst 3.6e-4); ten one-fault copies each refused by name; then GP-10's and GP-11's models written as a decomposed and a fused export, placed from their graphs alone and run. AMD RDNA-3: worst logit 7.1e-7; SwiftShader 2.0e-4. Qwen2.5's decomposed export gathers from its shapes by int64 constants (`/model/Gather_2`, read 2026-11-20 at d54957e), which the importer holds as tensors and the recorded graph, keeping only nodes that read weights, does not show: `roles.mjs` adds that node as the importer gives it, and the roles place Qwen2.5 with it, while the same node reading an f16 tensor is refused by name. The published page refused Qwen2.5 on 2026-11-20 with `unplaced: Gather /model/Gather_2 reads a weight no engine role takes` before this | tested |
| The recorded graphs are the exports' graphs | they were read by streaming each `model_q4f16.onnx` at the revision named and keeping only the nodes that read a weight of 16 or more numbers; the importer's own reading of the same files is not compared with them | imported, not cross-checked |
| `GP-19`: the engine page reads an ONNX export named on the hub from its own files, the kernel admitting every span, operator, file and 4-bit weight, and answers as the reference does on WebGPU and on the CPU | `just page-journey`: a Qwen3-shaped model written as an ONNX file field by field (`crates/conformance/tests/gpu/onnx-write.mjs`), its config, generation config and byte-level tokenizer, under a hub record and root; the page at `?onnx=` checks each small file against the record, imports the graph through the module asking the kernel (EN-33), places every weight (GP-12), reads the stop id and answers with the reference's sixteen tokens, six distinct, on WebGPU and with `navigator.gpu` taken away on the CPU engine; a flipped byte is refused by the kernel at the whole file (`FileRefused`), and a wrong record hash for `tokenizer.json` stops the page there. `gpu_gp_19` reads that wiring. Run on 2026-11-17 against a site assembled locally with the deployed kernel holo | tested |
| A real ONNX export runs in the page | run once by hand on 2026-11-17: the page of this branch, served locally with the deployed kernel holo, opened at `?onnx=onnx-community/Qwen3-0.6B-ONNX` against the live hub index. The record matched its root; config.json and generation_config.json matched the record's Git blob SHA-1 and tokenizer.json its SHA-256; `onnx/model_q4f16.onnx` (569,789,750 bytes) was streamed from its origin and imported with 919 questions to the kernel, every answer accepted, 508 tensors named; 310 weights placed for 28 layers; stop ids 151645 and 151643; ready in 342 s with 987 MB on this machine's AMD RDNA-3 GPU. Asked "What is the capital of France? Answer in one sentence.", it answered "The capital of France is Paris." in 7 tokens and stopped on its own id; first token 0.08 s (a headless tab, so not a speed claim). Two gaps it found are fixed and in the journey: the live records name files Git holds itself by their Git blob SHA-1 alone, and the page's connect-src did not name gethologram.ai | run by hand |
| An ONNX tensor checked against a published address | not possible yet: the hub's tensor index (`data/uor`) addresses three GGUF models and reads no ONNX canonical form, so an ONNX tensor's BLAKE3 is computed in the browser and the bytes are vouched for by the whole file's SHA-256, which is Hugging Face's own (Git LFS), not witnessed by the hub. Pulling single tensors from any source waits on published ONNX tensor addresses | not established |
| `GP-20`: the three exports' chat templates render as transformers renders them, and a template outside the engine's subset is refused by name | `gpu_gp_20`: `site/app/engine/jinja.mjs` over the three templates at the hub's revisions and a fourth written for whitespace (tags with no dash, a plus, a trailing newline), twenty conversations each, against what Python's jinja2 3.1.6 rendered in transformers' sandboxed environment (recorded in `crates/conformance/tests/gpu/templates`): all 80 byte for byte, Llama's own error where jinja2 raised it; six constructs refused by name. The page frames an ONNX export's turn with the model's own template (checked by `gpu_gp_20` reading the page and in the journey's `template` check); the journey's fixture template frames "hi" exactly as the fixed frame did, so the journey alone would not tell the two apart | tested against jinja2 |
| `GP-22`: the engine page asks the kernel which precision to read (21), what each file is called (24) and which device runs it (22), and reads, names and runs exactly what it answers | `just page-journey`: on every ONNX load the page asks decision 24 for the q4f16 and q4 graphs' names, offers the kernel each precision whose graph the record lists, with its bytes (graph and data files), and reads the file of the precision the kernel chooses; after import it asks decision 22 with the adapter's binding limit, the largest buffer the GPU engine would bind (one row of a tiled dense table, or the larger of a 4-bit weight's codes and scales) and the model's bytes, and runs on the device answered. The fixture's record offers q4f16 only: each load is answered `onnx/model_q4f16.onnx`, `q4f16`, and `webgpu`, or `cpu` with `navigator.gpu` taken away; a load asking `?precision=q4` stops at the precision, the kernel answering `none fits`, with no file read. `gpu_gp_22` reads that wiring. Run on 2026-11-17 on Windows with Chromium on the SwiftShader adapter, against a site assembled locally with the deployed kernel holo. Planted: the page taking the first offered precision, naming the graph itself, or running wherever an adapter is, each fails the journey | tested |
| What the kernel's choices are weighed against | no browser reports a GPU's memory, so the precision budget on WebGPU is a stated 4 GiB; the CPU engine's ceiling is a stated 2 GiB of model files, which admits the q4f16 exports of Qwen3-0.6B (570 MB), Llama 3.2 1B (1.09 GB) and Qwen2.5-1.5B (1.22 GB), so rows 2 and 3 of milestone 2 run on the CPU; raised from 1 GiB by Ilya on 2026-11-20 (the plan's open question 3). Both are choices, not measurements | stated, not measured |
| `GP-21`: an ONNX export's files are kept in the browser, each named by its BLAKE3, and read back on a later visit with the BLAKE3 derived again and the bytes checked as a download's are | `just page-journey`: four visits in one browser profile to GP-19's fixture. The first downloads and keeps all five files, each named by its BLAKE3 as the module computes it (`config.json` named `blake3:1275de1a…21fa3`, the same as an independent BLAKE3 of the file); the second, with the origin serving the record and no file, reads all five from the browser, derives each BLAKE3 again, imports the graph through the kernel and answers with the reference's tokens; the third, with the kept graph's bytes replaced by a graph with one byte flipped, stops at the graph as not its BLAKE3 and leaves no file kept; the fourth downloads all five again and answers. The module's hash (`hash_start`, `hash_take`, `hash_finish`) is tested against `blake3::hash` with two files interleaved, and ran at 973 MB/s in Node on 2026-11-17 where the page's JavaScript BLAKE3 ran at 21. `gpu_gp_21` reads that wiring. Run on 2026-11-17 on Windows with Chromium on the SwiftShader adapter, against a site assembled locally with the deployed kernel holo. Planted: `forget()` removing nothing fails the third visit's check; a held file never read back fails the second's; a kept file's BLAKE3 not derived again leaves the third visit refused by the kernel instead (`FileRefused`), which the journey rejects | tested |
| What is kept, and when it is dropped | only files the record names a SHA-256 for (Git LFS files: the graph, its data files, `tokenizer.json`), each under its BLAKE3 with a pointer from that SHA-256; the small files Git stores itself are fetched each time. Neither the name nor the pointer is trusted: the BLAKE3 is derived again on reading and the bytes go through the record and kernel checks a download does. Which files to drop when space is short (the kernel's decision 26) is not asked, and the browser may clear its private file system under storage pressure | not checked |
| The store's admission is the kernel's | not yet: in the test, admission is the stand-in check that the bytes' BLAKE3 equals the address (the vendored hasher); the page is to ask the kernel's decision on the wasm module's label, as for a download, and that wiring is not built | not tested |
| `GP-24`: when the browser has too little room to keep a downloaded file, the page drops kept files only on the kernel's decision 26, oldest first with this load's files pinned, and a file there is no room for is not kept while the model still loads | `just page-journey`: a profile holding another model's 300,000-byte file named by its BLAKE3, and `navigator.storage.estimate` reporting no free storage; GP-19's export loads, the kernel answers evict for the other file and enough after it, `config.json` is kept, the four later files are answered does not fit and not kept, and the page answers with the reference's tokens. `gpu_gp_24` reads that wiring. Run on 2026-11-17 on Windows with Chromium on the SwiftShader adapter. Planted: dropping on any answer but enough, never asking, and offering this load's files as unpinned each fail the journey. Not covered: a real quota (the estimate is replaced), and which files the browser itself clears under pressure | tested |
| `GP-25`: one call, `pipeline(task, model, { device, dtype })` in `site/app/engine/pipeline.mjs`, loads an ONNX export named on the hub and answers a function that runs a turn, every step put to the verified kernel; the engine page is one caller of it, and a task the kernel does not run is refused by decision 24 | `just page-journey`: a page imports the module and calls `pipeline("text-generation", …)` with no page code in between, then the function it answers with "hi": the kernel holo is verified, each file named, q4f16 chosen on WebGPU, every span admitted, and the 16 tokens are the reference's; `pipeline("feature-extraction", …)` stops at the kernel's `unsupported task` with no file read; and every engine-page visit of GP-19 to GP-24 now loads and answers through the same call (the page lost 192 lines). `gpu_gp_25` reads that wiring, and the GP-19 to GP-23 pins now read the module. Run on 2026-11-17 on Windows with Chromium on the SwiftShader adapter. Planted: the module asking every task as text generation, and the call reporting the device asked for instead of the kernel's, each fail the journey. Not built: the call shape for embeddings and classification (row 9), which waits on the integer encoders | tested |
| `GP-26`: every file of an ONNX export is checked by the BLAKE3 the hub's record names for it, small files in the module and the graph and data files through the importer (EN-42), and a record that names a file no BLAKE3 is refused by name; nothing is checked by SHA-256 or SHA-1 | `just page-journey` on 2026-11-18, Windows, Chromium on SwiftShader, against records that give each file's BLAKE3 beside the SHA-256 Hugging Face publishes: `config.json` checked as `blake3 1275de1a…`; the flipped graph refused by the kernel against its BLAKE3 (`FileRefused`); a wrong BLAKE3 for `tokenizer.json` stops the page there; a record naming `config.json` no BLAKE3 stops with `the record names no BLAKE3 for config.json` before any import; the kept-file, CPU and one-call visits load and answer. Kept files are held under the record's BLAKE3 directly, with no SHA-256 pointer. `gpu_gp_26` and `engine_en_42` pin it. Planted: the page not comparing a small file's BLAKE3, the graph put to the importer with its SHA-256 (every load then refused, so the run ends at the kept-file step rather than at a verdict), a file with no BLAKE3 let through, and the importer hashing whole files by SHA-256 again (rejected by `engine_en_42` and `engine_fl_45`). Against the live hub on 2026-11-18, after the hub published each file's BLAKE3 (Hologram-Technologies/hub#65, `738ff67`): this page, served locally with Chromium on SwiftShader (the machine's GPU failed to start in the test browser, `dxil.dll`), opened `?onnx=onnx-community/Qwen3-0.6B-ONNX`; `config.json`, `generation_config.json`, `tokenizer.json` and `tokenizer_config.json` passed against the record's BLAKE3 (`5011bbcd…`, `a007ce7e…`, `77c16932…`, `59111f79…`), `onnx/model_q4f16.onnx` was admitted by the kernel against its BLAKE3 with 508 tensors named and 919 questions, all 5 files were kept under their BLAKE3, and the model answered "The capital of France is Paris." in 7 tokens. The live records give a BLAKE3 only for the files the engine reads at q4 and q4f16, so another precision is refused by name; the hub record and index root are still checked by SHA-256, and move when the hub names them by BLAKE3 | tested |
| `GP-23`: the page keeps its chat frame's head, the ids every turn opens with, under the BLAKE3 of those ids, and a later turn reuses it only on the kernel's decision 27, computing only what follows | `just page-journey`: on GP-19's export, on WebGPU and on the CPU and on the kept-file visits, a first turn ("hi", 150 ids) computes all of its prompt and asks nothing; a second ("yes", 151 ids) asks the kernel with the BLAKE3 of its own first 115 ids, which the journey derives again from those ids with the vendored BLAKE3, is answered `reuse`, computes the 36 positions after the head, and gives the tokens of a reference given the whole prompt. The head is the common opening of the template rendered around two different texts. `gpu_gp_23` reads that wiring. Run on 2026-11-17 on Windows with Chromium on the SwiftShader adapter. Planted: restoring the head and then computing the whole prompt gives other tokens; reusing without asking leaves no reuse check (both run with a second turn of "hello"); holding one id more than the head, which "yes" does not share with "hi", is answered `recompute` and the page computes all 151 positions with the right tokens, which the journey rejects as no reuse | tested |
| `GP-30`: a return visit with the network off loads the engine page from its own worker (`site/app/sw.js`), re-checks the hub index kept in the browser (root by signature, record by BLAKE3), reads every file and tensor from the browser with decision 28 asked again, and answers as the reference does; a changed kept record is `RecordMismatch`, no kept index is `RootUnreachable` | `just page-journey` on 2026-11-19, Windows, Chromium on SwiftShader: one profile, a first visit with the network on, then the server answering nothing (sockets closed) and the browser offline. The offline visit loaded the page, kernel and engine from the worker, read 3 hub files from the browser and proved them again, said "the network is off" where it compares the deploy's listing, read all 5 files and all 15 tensors from the browser with 0 network reads, was answered `ready` by decision 28 (38 of 38) and gave the reference's 16 tokens; with the kept record changed by one byte it stopped at the record as `RecordMismatch`, and with the kept index removed at the index root as `RootUnreachable`, neither reading a file. `gpu_gp_30` reads that wiring and that the worker names no model or hub file. Planted: the worker without its offline answer: the offline visits do not open (`net::ERR_FAILED`) and every GP-30 verdict fails. Not covered: a browser that clears the private file system or the caches under pressure; the published page offline on a phone | tested |
| `GP-31`: tensors kept once by the BLAKE3 address the hub's UOR index lists; only what is not held downloaded, from the first listed source that answers; decision 23 on each span, decision 28 before anything new is kept; a model the index does not list refused by name | `just page-journey` on 2026-11-19, Windows, Chromium on SwiftShader, against a UOR index computed from the fixture's files as the hub's `tools/onnx.mjs` computes it and filed by the hub's shard rule (`hub/uorkey.mjs`, vendored): the q4f16 graph downloads its 15 tensors in one read and keeps each under its address; the same files under a second repository download nothing; that repository's q4 graph, whose 14 code tensors are the q4f16 graph's bytes, reads those 14 from the browser and downloads only the 7 others, from the listed mirror after its own origin answers nothing, and gives the q4 model's reference tokens; one byte of a kept tensor changed stops the load as `FileRefused` and the tensor is forgotten, and the next load downloads it again; with no free storage the q4 load drops kept tensors it does not read and keeps pinned the 14 it does; an index that does not list the model stops before any file is read with `no tensor manifest: the hub's UOR index lists no tensors for fixture/tiny-onnx`; an index giving one tensor another address is refused by decision 28 (`refused`, 1 of 38 not the address listed) with nothing kept. `gpu_gp_31` reads the wiring and that `uorkey.mjs` is the hub's file. Planted: held tensors never read (the twin and the q4 graph download them again), decision 28 asked with the listed label replaced by the derived one, so it always agrees (the lying index then loads and its bytes are kept; the page also checks the verdict a second time, so ignoring it in the store alone is still refused), and the room asked before this load's tensors were pinned (the short-of-room visit fails). The hub's UOR index is not signed: an address in it only says where to look; the record's whole-file BLAKE3 and decision 28 say whether the bytes are right. Measured shares are the ledger's `OPEN-SHARE-M2-FAMILIES` and `OPEN-STORE-SECOND-PRECISION` | tested |
| `GP-32`: prefix reuse is token-identical and timed: a second turn of the same text reuses the chat frame's head by decision 27, computes only the positions after it, and gives exactly the first turn's tokens | `just page-journey` on 2026-11-19, Windows, Chromium on SwiftShader: "hi" twice on GP-19's export, 150 prompt ids; the second turn is answered `reuse` for the 115-id head, computes 35 positions and gives the first turn's 16 tokens, which are the reference's, on WebGPU and on the CPU. Time to first token on this fixture, first and second turn: 1.04 s and 0.20 s on WebGPU, 0.083 s and 0.017 s on the CPU; the fixture is tiny, so these say the head is skipped, not what a real model gains. `gpu_gp_32` reads the wiring. Planted: the head not restored before the rest is computed: the second turn's tokens differ from the first's and from the reference's, on WebGPU and on the CPU | tested |
| `GP-57`: an ONNX export whose graph file holds no tensor, every one in a data file beside it, loads: each tensor's address is taken from the UOR index under the file it lies in, a listed tensor without one is refused, and a model none of whose files lists a tensor is refused | `just page-journey`'s visit to GP-19's export written again with all 38 tensors in `onnx/model_q4f16.onnx_data` (the writer names location, offset and length as onnxruntime's exports do), its record and UOR index row naming both files: the manifest passes with 38 tensors in 2 files, the graph listing none, decision 28 answers ready on 38 of 38 addresses, the importer reads 1 data file and the page answers with the reference's tokens. Run on 2026-11-20 on Windows, Playwright's Chromium with no WebGPU adapter, against main's deployed site at 3215ca2 with this change's `loader.mjs`; the visits that expect WebGPU failed there for want of it, the same with and without this change. `gpu_gp_57` reads the wiring. Llama 3.2 1B Instruct's q4f16 export, whose graph lists no tensor, was refused on the published page before this change with the planted defect's message | tested |
| Prefix reuse saves time on a real model | not measured: the fixture's head is 115 of 151 ids, so a second turn computes a quarter of its prompt; what that saves in time to first token on Qwen2.5-1.5B or Qwen3-0.6B is to be measured in the page by hand. The head is kept for the session only; keeping it across visits is not built | not measured |
| `GP-17`: every weight of the two encoder exports is given its role by the node that reads it, with shapes, activation and labels checked, the unplaceable refused by name, and an encoder placed this way gives the reference's embedding and labels | `gpu_gp_17`: `encoderRoles` in `site/app/engine/onnx-roles.mjs` over the two graphs recorded on 2026-11-16 at 751bff3 and 0b6928e in `crates/conformance/tests/gpu/exports`: 101 and 104 tensors placed, DistilBERT's labels NEGATIVE and POSITIVE; seven one-fault copies each refused by name; then GP-13's and GP-14's encoders written as these exports name their nodes, placed from their graphs alone and run: embedding within 1.7e-8, labels equal | tested |
| The CPU path on a real model, and on a phone | Qwen3-0.6B q4f16 measured in Node on this machine on 2026-11-19, one thread: 1.9 to 3.3 s per token, each prompt id a full step; the breakdown by stage is in docs/Q_ENGINE_PLAN.md. Not run in a browser on the real model (the 570 MB download did not fit the disk that day), and on no phone | measured in Node; not in a browser or on a phone |
| `GP-35`: `pipeline("feature-extraction", model)` imports an encoder shaped as Xenova's all-MiniLM-L6-v2 q4f16 export, int64 position ids included, with every operator asked of decision 18's encoder style, and answers each sentence's WordPiece ids and embedding from the module's integer encoder, bit for bit `hologram-ai-cpu-encode`'s, on the CPU | `just page-journey`: GP-13's encoder written by `encoder-export.mjs` under its own hub record and UOR index, two sentences, every id and bit compared with `encoder-expected.json` (EN-64 recomputes it from the crate); `gpu_gp_35` holds the page and the journey to it | tested in CI; not run locally, since the page needs a holo carrying KN-52 and KN-53, which only the SDK builds |
| `GP-36`: `pipeline("text-classification", model)` answers the label the module chose and its name from id2label, and the scores, bit for bit the crate's | the same journey, GP-14's classifier; `gpu_gp_36` | tested in CI, as GP-35 |
| `GP-41`: the integer engines run in a worker; the page's own thread makes no decode or encode call and answers throughout a turn, each token arrives as its own message, and tokens and bits are unchanged | `just page-journey` on 2026-11-19, Windows, Chromium: counts every call the page's thread makes into a module's forward pass or encoder (none), a heartbeat on the page's thread (longest gap 6.1 ms), `onToken` once per token, sixteen tokens equal to the reference's, GP-35's two embeddings bit for bit; `gpu_gp_41` reads the pipeline, the worker and the journey. On Qwen3-0.6B q4f16 in Node 22 with a worker-thread shim, the 46-token reply's ids and text sha256 equal the determinism run's (c27da8e7..., 9ebc5906...) | tested |
| The real all-MiniLM-L6-v2 and DistilBERT SST-2 run in the page | not run: both are on the hub (records and UOR rows at 751bff3 and 0b6928e), and the page path is the crate's bits by EN-64; running them in the page needs their downloads, which wait on Ilya's word. Their agreement with the float reference is the crate's, recorded in `OPEN-ENCODERS-INTEGER-AGREEMENT` | not tested |
| The check reaches a visitor whose worker serves an earlier deploy | rests on the frame's worker fetching `index.html` from the network first (pinned by `gpu_gp_07`); the journey plants the cache directly and registers no worker, so the worker's own behaviour is read, not run | not tested |
| The engine gives the same tokens on a real GPU as on the software renderer | not checked here: CI has no GPU. The same test runs on a named laptop before checkpoint 1 is called done | not tested |

The page's numbers are integers. What Lean proves of the WebGPU kernels is in
KN-73 and KN-95 and the section on them below; the claims in this table are
tested, not proved.

## How the tests run

Both run in the SDK image, as the rest of the gate does. The browser tests
(`gpu_gp_29` and `just page-journey`) serve the repository or the assembled
page on a local port and open it in the headless Chromium and Playwright the
image carries (`/opt/prismpm/oracles/node_modules/playwright`, the version
`package.json` pins), with `--enable-unsafe-swiftshader --enable-unsafe-webgpu`.
Without a WebGPU adapter they fail; they never skip.

## Retired with the float engines

Decided by Ilya on 2026-11-20: the page runs ONNX exports only, on the integer
engine, Qwen3-0.6B's by default, and the float WebGPU and CPU engines
(`gpu.mjs`, `cpu.mjs`, `encoder.mjs`, `wgsl*.mjs`, `yarn.mjs`) and the
repacked Bonsai container path are removed; Bonsai returns in milestone 3 on
the integer engine. The claims that were about those engines are retired with
them, their tests and runners removed: GP-01, GP-02, GP-04, GP-05, GP-08 to
GP-11, GP-13 to GP-16 and GP-18. What they covered that the page still does is
claimed on the integer engine: the forward pass and its rotary frequencies
(EN-48 to EN-51, EN-96), products on WebGPU (GP-56, KN-73, KN-95), the encoder
(GP-35, GP-36), tensors kept by address (GP-31) and prefix reuse (GP-23,
GP-32). GP-12 and GP-17 keep their role maps, which the integer engine reads.
The journey's stale-kernel, foreign-signature and flipped-engine runs moved
from the container to the ONNX export.

## Planted defects

| Defect | Command | Result |
| --- | --- | --- |
| `vv` no longer runs `page-journey` | `cargo test -p repo-conformance --test gpu gpu_gp_03` | rejected: `vv does not run page-journey` |
| `pages.yml` runs the journey script directly instead of the recipe | the same | rejected: `pages.yml runs the page-journey recipe` |
| The page loads without comparing its kernel with the deploy's listing | `just page-journey` | rejected: `stale: expected to stop with "stale kernel" at the deploy comparison, stopped at undefined: undefined (ready: null)` and `good: the kernel was not compared with the deploy's listing (no check)` |
| the same | `cargo test -p repo-conformance --test gpu gpu_gp_06` | rejected: `load compares the kernel` |
| The prompt box starts with text in it | `just page-journey` | rejected: `good: the prompt box started with text, so typing would join it: "Tell me"` |
| The page never removes a cached file that differs from the deploy's listing | `just page-journey` | rejected: `earlier: the page loaded 1 times, expected one reload`, `earlier: the earlier deploy's engine file is still in the cache` and `again: expected "Stopped: stale page" and no Load button` |
| the same | `cargo test -p repo-conformance --test gpu gpu_gp_07` | rejected: ``site/app/index.html no longer has `if (held !== want) {` `` |
| With a reload already spent and a file still stale, the page imports the engine instead of stopping | `just page-journey` | rejected: `again: expected "Stopped: stale page" and no Load button, saw Stopped: stale page: 1 cached files are not this deploy's after a reload. Clear this site's data and open the page again. (disabled: false)` |
| The role map lets through an operator that reads a weight no role takes | `node crates/conformance/tests/gpu/roles.mjs .`, as `gpu_gp_12` runs it | rejected: the `unplaced` case is not refused (null) |
| No role map, as before this change, GP-12 written first | the same | rejected: `ERR_MODULE_NOT_FOUND` |
| The decoder roles refuse an int64 constant a Gather reads, as before 2026-11-20 | `node crates/conformance/tests/gpu/roles.mjs .`, the runner `gpu_gp_12` calls | rejected: `indexConstant` is `unplaced: Gather /model/Gather_2 reads a weight no engine role takes`, the published page's own refusal of Qwen2.5 |
| The encoder map does not check the activation | `node crates/conformance/tests/gpu/encoder-roles.mjs .` | rejected: a gelu_new config is not refused (null) |
| No encoder map, as before this change, GP-17 written first | the same | rejected: `TypeError: encoderRoles is not a function` |
| The ONNX loader does not compare small files with the record | `node crates/conformance/tests/gpu/journey.mjs <site>` (cargo was not run on that machine) | rejected: `onnx tokenizer: expected to stop at tokenizer.json, stopped at undefined (ready: null)` |
| The ONNX loader never names the kernel for the module | the same | rejected: the import stops at its first span, `Import(SpanRefused(TensorAt(0), Unrecognised))`, and both runs fail to reach ready |
| The ONNX loader requires a SHA-256 for every small file, as first written | the page against the live hub, by hand | stopped: `the record names no SHA-256 for config.json`; the live records give only a Git blob SHA-1 for files Git holds itself. Now checked by that, and the journey's `gitblob` and `gitwrong` cases hold it |
| The template engine keeps the newline after a block tag (no trim_blocks) | `node crates/conformance/tests/gpu/templates.mjs .`, as `gpu_gp_20` runs it | rejected: the whitespace template differs in all 20 conversations |
| It keeps the indentation before a block tag (no lstrip_blocks) | the same | rejected: the same 20; no export's template reaches this, which is why the fourth template was written |
| tojson written without Python's spaces | the same | rejected: Qwen2.5 and Qwen3 differ in their 4 tool conversations |
| A slice with a negative step read forwards | the same | rejected: Qwen3 differs in its 2 thinking conversations |
| `in` on a mapping always false | the same | rejected: Llama differs in its 2 tool-call conversations |
| A template's last newline kept, as first written | the same | found by the whitespace template: all 20 differed by that newline; fixed, as jinja2's keep_trailing_newline is off |
| none | all twenty, and the journey | pass |
| A path whose own tensor list is empty refused, as before this change (GP-57) | `node crates/conformance/tests/gpu/journey.mjs <site>`, the runner `just page-journey` calls, on the site above (cargo was not run for the journey) | rejected: `external: the page did not reach ready (failed: no tensor manifest: the hub's UOR index row for onnx/model_q4f16.onnx gives no BLAKE3 address to some tensor)`, nothing read from the data file |

## GP-28, the page's CPU device on the integer engine

An ONNX export on the page's CPU device now runs on the integer engine:
`site/app/engine/integer.mjs` hands the page's module every weight the import
read (`decode_tensor`), opens it with the config's head counts, rope theta,
epsilon as binary32 bits and the rotary scaling the loader confirmed
(`decode_open`), and answers each token with `decode_step`; a second turn
carries on from the run the module kept (`decode_save`, `decode_restore`,
EN-53). The page computes no number of its own on this path. The float CPU
engine (`cpu.mjs`) still runs weight containers and stays the yardstick
GP-18 measures; WebGPU is still the float engine until the integer WGSL
kernels land.

`just page-journey`'s `cpu` run must report `integer: true` beside its
sixteen tokens equal to the reference's and its second turn's reuse; `gpu_gp_28`
reads that the pipeline builds the CPU device's engine with
`createIntegerEngine` and never the float one. GP-19's pinned line moved with
it.

| Planted defect | Result |
| --- | --- |
| the pipeline's CPU device built with `createCpuEngine` again | rejected: `pipeline.mjs no longer has ``createIntegerEngine({ module, ...raw, capacity: CAPACITY })``` |

Since GP-41 the pipeline builds it with `startIntegerEngine`, which opens `createIntegerEngine` in the integer worker; `gpu_gp_28` reads that line.

## GP-41, the integer engines off the page's thread

The pipeline no longer opens the integer engines on the page's module. At build time it hands a worker (`site/app/engine/integer-worker.mjs`) the compiled module, a copy of the page module's memory as it stands after the import (`moduleOf`, `site/app/engine/loader.mjs`), and every placed weight's bytes, transferred rather than copied. The worker opens `createIntegerEngine` or `createIntegerEncoder` on them exactly as the page did; `startIntegerEngine` and `startIntegerEncoder` answer the same interface with every call awaited, and `run` hands the page each token as its own message. The page keeps its module for the tokeniser and the hashes. On Qwen3-0.6B the page's module now holds 84 MiB, where it held 974 MiB with the weights.

`just page-journey` counts, on the page's realm only, every call into `decode_tensor`, `decode_open`, `decode_step`, `decode_reset` and the four encoder calls. It does this by copying each instance's frozen exports with those calls wrapped. A worker is a realm of its own, so its calls are not counted.

| Planted defect | Result |
| --- | --- |
| the pipeline's CPU device built with `createIntegerEngine` on the page again (`gpu_gp_41`) | rejected: `pipeline.mjs no longer has ``startIntegerEngine({ module, ...raw, capacity: CAPACITY })``` |
| the same, in the assembled page (`just page-journey`) | rejected: `worker: the page's own thread computed: decode_tensor x38, decode_open x1, decode_reset x1, decode_step x165` and `worker: the page's thread went 713 ms unanswered in the turn (1 beats)`; restored: no call counted, longest gap 6.1 ms |

The first hook proxied the exports object. Exports are frozen, so the proxy broke its own invariant on the first counted call, and the plant stopped with a TypeError instead of a count. Copying the exports fixed that, and the plant above is the run after the fix.

## GP-27, the index root checked by its signature

The page reads a model's record with the hub's own client, vendored at
`site/app/engine/hub/` (VENDOR.md): `openIndex` checks the index root by its
Sigstore bundle against `policy.json`, the hub model's trust anchors pinned
here, and `readRecord` checks the record by the BLAKE3 the root names. Until
this, the page checked the record against the root by SHA-256 and showed the
root's signature as not checked.

The tests cannot ask the live hub to sign the fixtures' roots, so
`crates/conformance/tests/gpu/hub-sign.mjs` mints a chain of the same shape:
a P-384 intermediate, a P-256 leaf carrying the policy's identity and issuer,
a P-256 log key signing the entry timestamp, and a DSSE envelope naming the
root's SHA-256. A policy naming that chain's keys, otherwise the pinned one,
is what the journey serves the page; the verifier is the hub's, unchanged,
and the page has no flag that skips it.

`hub-check.mjs` drives the client in Node, and `just page-journey` drives the
page:

| Case | Refused as |
| --- | --- |
| root signed through a foreign intermediate and log key | `RootUnverified`, chain and log |
| the policy's intermediate, a foreign log key | `RootUnverified`, log |
| another workflow's identity | `RootUnverified`, identity |
| the root changed after signing | `RootUnverified`, subject |
| the test chain under the hub's own pinned policy | `RootUnverified`, chain and log |
| no bundle | `RootUnreachable` |
| a repository the root does not name; a changed, unaddressed or unreachable record | `RecordUnknown`, `RecordMismatch`, `RecordUnaddressed`, `RecordUnreachable` |
| a changed or short file; a file its record gives no BLAKE3 | `FileMismatch`, `FileUnaddressed` |
| the journey's `foreign` hub | the page stops at "index root signature", before any record is read |

### The gate: `cargo test -p repo-conformance --test gpu -- gpu_gp_27`

| Planted defect | Result |
| --- | --- |
| one byte of the vendored `sigstore.mjs` changed | rejected: `site/app/engine/hub/sigstore.mjs is not the hub's file` |
| the policy's `rekorLogId` changed | rejected: `assertion failed` on `policy["rekorLogId"]` |
| the verifier's chain fact forced true | rejected: `sigstore.mjs is not the hub's file`; with that pin and `SIGSTORE_BLAKE3`'s taken out too, rejected: `foreignKey failed other checks` |
| `hubRecord` reading the root without `openIndex` | rejected: the loader no longer has `index = await openIndex(base, policy, { hash });` |
| none | passed |

## The integer kernels on WebGPU, in LexLean

Row 15 of the plan runs the integer stages on WebGPU. The kernels are written
first in LexLean, in `app/src/Engine/Gpu.lex.tex`, outside the guest: a kernel
is a statement over 32-bit words (`Expr`, `Stmt`), run by `exec` one
invocation at a time (`runAll`), with a read past a buffer's end giving 0 and
a store past it dropped. The same file holds the WGSL each kernel becomes,
twice: as a term (`toWgsl`, with WGSL's own arithmetic in `wgslEval`) and as
text (`renderKernel`).

The guest cannot hold those trees. PrismPM's production exporter refuses recursion over a
type of the application's own (`Expr.brecOn`: an escaping local function, `PP5008`), and its
code generator refuses a recursive type outright. So the guest holds each kernel as a byte
code, prefix order, one byte per node and four for a literal, in `app/src/Engine/Code.lex.tex`,
and prints it with recursion on a fuel count, as every other guest module recurses. Lean
checks, kernel by kernel, that this printer's text is byte for byte `renderKernel`'s text of
the tree, and decision 48 prints only the kernels so checked.

| Claim | Evidence | Kind |
| --- | --- | --- |
| Every expression the kernels use computes the same word in WGSL's arithmetic as in the model's, so every program runs alike | `wgsl_preserves_eval` and `wgsl_runs_every_program_alike` in `app/src/Proofs/GpuWgsl.lex.tex`, for all machines, expressions, statements and fuel | proved |
| The q4 group-sum kernel gives the exact sums of the crate's two q4 fixtures, zero point 8 and packed zero points | `q4_group_sums_zero_eight` and `q4_group_sums_packed_zeros` in `app/src/Proofs/GpuQ4.lex.tex`: two outputs each, worked out by Lean | checked on fixtures |
| `exp2Neg` gives EN-43's `exp2_neg` on 19 inputs covering every branch: shift 0, 1 to 31, 32 and past it | `exp2_neg_matches_reference` in `app/src/Proofs/GpuExp2.lex.tex`; the expected words were worked out by a port of `exp2_neg`, not by the crate | checked on fixtures |
| The guest prints exactly `renderKernel`'s text, and only for kernels so checked | twelve theorems in `app/src/Proofs/GpuCode.lex.tex` (`q4_code_prints_tree_k128` to `_k8960`, `exp2_code_prints_tree`, `q4_quantise_code_prints_tree`, `q4_accumulate_code_prints_tree`, `q4_finish_code_prints_tree`): for q4 group sums at zero point 8 with rows of 128, 256, 1024, 1536, 2048, 3072, 8192 and 8960 codes, for `exp2Neg`, and for the q4 product's quantiser, accumulator and rounding kernels, the guest's byte-code printer gives the same bytes as `renderKernel` on the kernel's tree, worked out by Lean. The code is read from thirty-two pieces of 32 bytes (`Engine.Code::Pieces`), since Lean's cost of reading one byte of a long byte literal grows with its position. Decision 48 (`Kernel.Render`, KN-49) answers only those, and the `app` recipe replays 59 claims through the compiled guest: every piece of the 128-code q4 kernel, of `exp2Neg` and of the three q4 product kernels, the empty piece past each end, and the refused requests (64 codes, packed zero points, units that do not match the width, 16 bytes, kernel 6, a quantiser with units, an accumulator with a width, a rounding kernel with a last byte) | checked per kernel, gated in the guest |
| The text `renderKernel` writes is the WGSL that ran on WebGPU | Kernel 1's text for two units compiled in Chrome's WebGPU and ran on AMD RDNA 3.5 and SwiftShader in the PrismPM bridge (UOR-Foundation/prismpm#732), giving [452167, 158806] as the CPU does, and `exp2Neg`'s text ran there on all 19 inputs Lean checked, giving Lean's words (`test/fixtures/webgpu/exp2-neg.rendered.wgsl`, merged in #732). The guest's `exp2Neg` text is byte for byte that file, and its 128-code q4 text differs from the two-unit file only in the three width constants | run in PrismPM's headless test, not yet from this repository |
| The q4 product's three kernels give `hologram_ai_exact`'s integers | `q4_quantise_codes_match_exact`, `q4_quantise_power_matches_exact` and `q4_row_matches_exact` in `app/src/Proofs/GpuQ4Chain.lex.tex` (KN-73): Lean runs the quantiser, then the accumulator and the rounding kernel, on the first fixture of `crates/exact` (32 activations, two rows) and gets its codes, its power and its two binary32 rows | checked on a fixture |
| Each kernel runs on WebGPU as `crates/exact` computes it | for the q4 product only, and outside this repository: the guest's own text of the quantiser, accumulator and rounding kernels, with kernel 1 between them, ran in headless Chrome through prism-code's `runPlan` on AMD RDNA 3 and on SwiftShader over 62 random cases from `crates/exact` (256 rows: 58 infinity, 63 subnormal, 132 negative), every stage equal bit for bit (2026-11-19). A test in this repository that does the same is the next change | run, not yet tested here |
| A kernel equals its stage for every input in bounds | not yet: the stages EN-43 to EN-47 are defined in Rust, not LexLean, and the step from 32-bit words to the stage's integers needs nonlinear arithmetic the proof forms do not have | not proved |

The text follows the PrismPM WebGPU bridge's conventions: buffer i is
`array<u32>` at `@group(0) @binding(i)`, read_write exactly when the kernel
stores to it; entry `main`; `@workgroup_size(64)`; and a pipeline constant
`invocations`, past which an invocation returns. Closed subexpressions are
worked out while printing, since WGSL refuses constant overflow and constant
division by zero, and shift amounts are masked to five bits, as the model's
are. The GPU runs invocations at once and the model one after another; they
agree when each invocation writes only its own outputs and reads no buffer it
writes, which holds for these kernels and is not yet checked.

## GP-50, the page refuses an engine it was not built with

Before the page compiles `engine.wasm`, `openModule` (site/app/engine/loader.mjs)
hashes the bytes it fetched with the vendored BLAKE3 and asks decision 80
(KN-80) whether they are the sealed engine; only those same bytes are then
compiled. `openModule` takes the page's asker, so neither caller (`app.mjs`,
`pipeline.mjs`) can open the engine without the check, and the integer worker
(GP-41) is handed the module compiled there, never bytes of its own: `gpu_gp_50`
requires one `WebAssembly.compile` in loader.mjs and no fetch or compile in
integer-worker.mjs. A wrong engine (a stale
cache's, one altered on the way, one built from other inputs) stops the load
at the `engine` check with the kernel's `not the sealed engine`.

The journey serves the deploy's engine with one bit of its middle byte
flipped and requires: the load fails at the `engine` check with that answer;
no module of the engine's length is compiled; and the container is never
asked for. On the true engine it requires the `engine` check to pass with
`sealed`.

### The gates

`cargo test -p repo-conformance --test gpu gpu_gp_50`, planted on 2026-11-19:

| Planted | Result |
| --- | --- |
| decision 80 asked after `WebAssembly.instantiate` | rejected: `the engine is compiled before decision 80 is asked` |
| none | passed |

`just page-journey` (in the acceptance gate and `pages.yml`): passed in acceptance run 44702842589, the engine with one bit flipped stopped at the `engine` check with "not the sealed engine", nothing of its size compiled and the container not read. The run before it (44642776590) failed here because the flip was routed on the page and the page's service worker fetched the engine itself; the route is now the context's.

## GP-56, the integer engine's products on WebGPU

Row 15 runs the integer engine on WebGPU from the same definitions it runs on
the CPU. A token's time is its products (86 per cent in #106's profile), and
EN-92 already lets a product be answered outside the module that runs the
model with every bit unchanged. So on WebGPU the module still runs the model
and every stage between the products, and its q4 products are answered by the
four kernels the guest prints (decision 48): the quantiser, the group sums for
the product's width, the accumulator and the rounding. Every value is a `u32`
word, no kernel names `f32` or `f16` (KN-73), and GP-29 shows the four give
`hologram_ai_exact`'s rows on WebGPU, so the tokens are the CPU's.

How the page does it (`site/app/engine/integer-gpu.mjs`): the module asks for
a product through its `kernel.product` import, which must answer before it
returns, and WebGPU answers only on an event loop. The model's worker writes
the input into a SharedArrayBuffer, posts the product's number to a WebGPU
worker and waits on `Atomics.wait`; the WebGPU worker dispatches the four
kernels, reads the rows back, writes them where the model's worker reads them
and wakes it. The weights reach that worker once, as `decode_slices` writes
them. The dense f16 output head runs on the dense kernel (KN-95) when its weights fit
one binding. A product no printed kernel takes is answered on an instance of
the same `engine.wasm` holding only it: a head too large to bind or stored as
binary32, and any matrix with binary32 scales or zero points.

WebGPU is usable, and offered to decision 22 as a GPU, only where both hold:
the page is cross-origin isolated (shared memory; the engine page's service
worker makes a return visit so, GP-52) and the guest prints the group-sum
kernel for every width the model's products read (`Kernel.Render`'s
`printedWidth`: 128, 256, 1024, 1536, 2048, 3072, 8192, 8960, which covers
Qwen3-0.6B, Qwen2.5-1.5B, Llama 3.2 1B and GP-19's fixture). Elsewhere the
kernel is told there is no usable GPU, so it chooses the CPU or refuses a model
above the CPU ceiling; a page that asks for `device: "webgpu"` there is
refused with the reason. The float WebGPU engine no longer runs an ONNX export;
`gpu.mjs` and the `wgsl*.mjs` kernels still run the repacked container
(`app.mjs`'s Bonsai path) and the conformance tests that use them as a
yardstick, until those move too.

`audit-float` (EN-58) now reads `integer-gpu.mjs` with the other integer-path
scripts, and finds no float.

| Claim | Evidence | Kind |
| --- | --- | --- |
| The page's q4 products run on the guest's printed kernels and the tokens are the reference's and the CPU's, by name and by the kernel's choice, on an isolated page | `just page-journey`, GP-56's run: all 15 products on WebGPU, the head on the dense kernel, 16 tokens each | tested on CI (SwiftShader) |
| A first visit, not isolated, runs on the CPU, as the kernel's device | `just page-journey`, GP-19's runs, the device expected from whether the page was isolated | tested on CI |
| The pieces between are in place and no ONNX export reaches the float engine | `gpu_gp_56` | tested |
| Every kernel equals its stage for every input | not yet, as for GP-29: the stages are being defined in LexLean (row 13) | not proved |

### The gate: `cargo test -p repo-conformance --test gpu gpu_gp_56`

Planted on 2026-11-20:

| Planted | Result |
| --- | --- |
| `pipeline.mjs` importing `createEngine` from `./gpu.mjs` again | rejected: `pipeline.mjs still reaches the float engine` |
| `onGpu` taking a matrix with binary32 scales | rejected: `site/app/engine/integer-gpu.mjs no longer has` the rule |
| none | passed |

## GP-58 and GP-59, receipts named by BLAKE3 and checked on any device

Why: Hologram AI's inference is integers only, so the same model, input and
budget give the same bytes on every CPU and GPU. That is worth something only
if anyone can check it. A receipt names everything a run depends on by its
BLAKE3, and Check runs it again and lets the kernel compare.

How:
- **The model's label** is the BLAKE3 of one line per file whose bytes decide
  the answers, sorted by path: the graph and its data files (every weight),
  `config.json` (shapes and rotary parameters) and `generation_config.json`
  (the stop ids). Each line is the path and the BLAKE3 the hub's record names
  for that file, which the load checked. The tokenizer and the chat template
  only turn text into ids, and a receipt carries the ids, so they are left out.
- **The input's label** is the BLAKE3 of its token ids and the budget, as
  four little-endian bytes each.
- **The output's label** is the BLAKE3 of the ids it produced, the same way.
- **The class** is decision 30's answer for this engine: products exact, every
  other stage exact, no float summed in any order, so `everywhere`.

A receipt is plain JSON and fits in a link (`#receipt=`). Check resets the
engine and runs the receipt's own ids, then puts the claimed and replayed
output labels to decision 31, with whether this tab's model has the receipt's
model label (decision 28's admission already holds the bytes to the record).
`confirmed` means this device produced the same bytes.

| Claim | Evidence | Kind |
| --- | --- | --- |
| On the CPU and on WebGPU one model, input and output give the same three labels, and the output label is the BLAKE3 of the tokens' bytes | `just page-journey`, GP-56's run: both devices on one isolated page | tested on CI (SwiftShader) |
| Each device confirms the other's receipt; a changed output is a mismatch; another model is not comparable | the same run, through decision 31 | tested on CI |
| A link carrying a receipt opens, loads, checks and shows Confirmed | `just page-journey`, the linked visit | tested on CI |
| The same holds on a real model and a real GPU | not yet run: the measurement session records it with row 15 | open |
| Probes into a layer (decision 32) | GP-60, below | tested on CI |

### The gates

`cargo test -p repo-conformance --test gpu gpu_gp_58 gpu_gp_59`, planted on 2026-11-20:

| Planted | Result |
| --- | --- |
| the output label taken over the tokens and the budget | rejected by `gpu_gp_58`: ``pipeline.mjs no longer has `output: { tokens: tokens.length, label: blake3Of(m, idBytes(tokens)) },` `` |
| a receipt for any model counted as ready | rejected by `gpu_gp_59`: ``pipeline.mjs no longer has `const ready = receipt?.model?.label === session.identity.label ? 1 : 0;` `` |
| none | passed |

## GP-60, a past answer opened at any layer

Why: an answer is interpretable when any layer at any position opens to the
exact integers that produced it. Because the engine is deterministic, nothing
has to be recorded while generating: a probe runs the answer again to that
point and reads the value there, which is the value that produced it.

How: a probe first checks its receipt (decision 31) and recovers the output
ids. It then asks decision 32 whether the replay is exact, given the layer,
the position, the component and the run's class. The engine replays the run's
ids to that position with the module's probe set (EN-97) and reads one
component: the residual after the layer, attention's output or the
feed-forward's. Its values are binary32 bit patterns of the residual stream's
width, named by their BLAKE3, so a probe can be shared and compared like a
receipt.

| Claim | Evidence | Kind |
| --- | --- | --- |
| The same answer probed on the CPU and on WebGPU reads the same bits, one BLAKE3, of the residual stream's width | `just page-journey`, GP-56's run | tested on CI |
| Attention's output at that point is other bits; a layer past the model's is out of range | the same run | tested on CI |
| Probes are the run's own values and change nothing | EN-97 | tested |

### The gate: `cargo test -p repo-conformance --test gpu gpu_gp_60`

Planted on 2026-11-20:

| Planted | Result |
| --- | --- |
| a probe that skips checking its receipt | rejected: ``pipeline.mjs no longer has `if (checked.verdict !== "confirmed") return { verdict: "not checked", checked };` `` |
| none | passed |

