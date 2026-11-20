# Verified tensor streaming

A design for pulling a model's tensors so that work starts at once and keeps pace with the network, with no byte used before BLAKE3 has checked it. This note is the proposal and its evidence. Nothing here is registered or built in this repository yet, and nothing here is proved; Ilya decides what ships.

## Why

Today's page waits for the whole file, hashes all of it, puts the importer's addresses to decision 28 and only then builds the engine. So a model's first token waits for every byte plus every serial step after the last byte. With Qwen3-0.6B's q4f16 export (569,789,750 bytes) on this machine's link, the first byte that may be used arrives after the last byte does: 299 s (measured, below).

BLAKE3 is a Merkle tree, so a byte range can be checked against an address as it arrives. That lets the engine admit a layer the moment that layer is checked, and lets download, verification, upload and compute run as one pipeline.

## What exists already (UOR native first)

Everything verified streaming needs is already shipped in UOR and Hologram repositories. This design composes those pieces and adds no new format or library.

| Piece | Where | State |
| --- | --- | --- |
| `PieceTree`: BLAKE3's tree cut at 1 MiB, one 32-byte chaining value per piece, wire form `HPTREE01`, self-certifying against the blob's `blake3:` label | Hologram-Technologies/hologram `crates/hologram-space/src/piece.rs` (blake3 1.8 `hazmat`, wasm32 SIMD) | merged (#156) |
| A piece table for every tensor longer than 1 MiB, at `data/uor/pieces/{address hex}` | Hologram-Technologies/hub `tools/uor-index.mjs`, root field `pieceTables` | live on gethologram.ai: 808 tables |
| A browser piece fetcher: Range reads from any source, a worker pool hashing pieces with WASM SIMD, an OPFS piece store | hub `web/pieces.mjs`, `web/piece-hash.wasm` (36 KB), `web/piece-worker.mjs` | merged |
| The piece rule, `pieceDecision`: deliver only when the table folds to the address and the bytes match their entry; reuse a stored piece only when it matches | hub `app/src/Pieces.lex.tex`, HK-39 to HK-42 | proved in the hub's kernel |

What is missing is on this repository's side: the engine never asks for a range, the kernel admits a whole model and not a piece, and the stream is in file order rather than forward-pass order.

## Bao

Bao (oconnor663/bao) is the reference implementation of BLAKE3 verified streaming, which is section 6.4 of the BLAKE3 paper. Its root is the plain BLAKE3 hash, so a Bao root is a uor-addr `blake3:` label. `HPTREE01` is that same construction with two choices made differently:

| | Bao | `HPTREE01` |
| --- | --- | --- |
| Verified unit | one 1 KiB chunk | one 1 MiB piece (a subtree of 1,024 chunks) |
| Outboard size | every tree node: 62,472 bytes per 1,000,000 (about 6.2 %) | the 1 MiB level only: 32 bytes per MiB (0.003 %); interior nodes recomputed in microseconds |
| Combined encoding | tree interleaved with content | none: sources serve the original bytes |
| Range check | a slice carries its own proof path, chunks inline; an outboard slice does not exist | fetch the table once (9,528 bytes for a 311 MB tensor), then any piece from any source |
| Browser | Rust crate and CLI only; no JS or WASM package; its README calls it beta and not audited | 36 KB WASM in the hub, BLAKE3 1.8.7's own `hazmat` |

Verdict: use Bao's construction, which is already in use, and not its crate or encodings. Combined encoding would have the hub re-host every model, because Hugging Face serves only original bytes. That breaks "any source serves the bytes", and GitHub Pages has about 65 MB of headroom. An outboard at 1 KiB costs 6.2 % more bytes on every load. That buys latency only where a piece takes longer to arrive than the work it unblocks; at the 1 MiB pieces and links measured here it does not. If finer pieces are ever needed, the outboard can carry an extra level (for example 64 KiB, 0.05 %) under the same format, which already records its piece length. No outside dependency is added.

## Design

### Two tables, both rooted in an address the page already trusts

1. A file table: `HPTREE01` over the whole ONNX file (graph and data file alike). It folds to the record's whole-file BLAKE3, and that BLAKE3 is under the hub index's Sigstore signature. With it, every byte of the file, including the graph and the framing between tensors, is checked per piece against the signed record. The UOR index is not trusted for this. For Qwen3-0.6B q4f16 the file table is 544 pieces, 17,432 bytes. The prototype derived it while hashing the whole file once and checked that it folds to `cd406861…d674`. The hub should publish it beside the tensor tables, named by the file's address.
2. Tensor tables, as the hub publishes them today. These are for deduplication: a tensor held under its address (another precision, another repository) fills its span of the file without being fetched. The file piece that holds a tensor boundary is then checked from the held bytes plus the fetched bytes. A tensor address taken from the UOR index is only a hint until the index root is signed; the file table is what admits bytes.

Why both: the file table alone closes the trust gap, because every piece descends from the signed record. Tensor tables alone leave about 282 KB of graph and framing unchecked until the end, and they lean on the unsigned UOR index. In the prototype's tensor mode, 569,507,840 of 569,789,750 bytes were covered.

### Stream order: the forward pass, not the file

| Class | Bytes for Qwen3-0.6B q4f16 | Why first |
| --- | --- | --- |
| 0 graph | header 175,042 and tail 75,286 | the importer needs the operators |
| 1 prompt rows | the embedding pieces holding the prompt's rows: 6 pieces for a 14-token chat prompt | prefill's input |
| 2 rotary rows | piece 0 of `cos_cache` and `sin_cache` (positions below 8,192) | layer 0 needs them |
| 3 to 30 layers 0 to 27 | 8,851,968 bytes each | the forward pass, in order |
| 100 head | the rest of the embedding, 311 MB | Qwen3-0.6B ties its head to the embedding, so logits need every row |
| 101 rotary rest | positions from 8,192 up | needed only for long contexts |

Pieces are grouped in that order into Range reads of contiguous file bytes, up to 8 MiB each. Gaps of framing under 64 KiB are read inside the same request.

### Admission rule

A piece is used only when the kernel answers deliver or reuse to `pieceDecision`: its table folds to the address, and the piece's chaining value, taken here, equals the table's entry with the length equal. That is the hub's rule HK-39 to HK-42, which this repository would import as a decision rather than restate. A tensor is placed in the engine once every piece covering it has been delivered. A layer runs once all of its tensors are placed. Decision 28 stays and moves earlier: the importer's derived address of each tensor is checked against the listed one as that tensor completes, not once at the end. A piece that fails is asked of the next source, and with no source left the load fails by name (R5). No unverified byte reaches the importer, the GPU or storage.

### Sources

Any host that answers a Range request. Today that is Hugging Face's resolve URL, whose 302 to its CDN carries `Cache-Control: no-store`, so every range pays the redirect unless it is resolved once and the CDN URL reused (the prototype does that; the gain was within the link's noise). Mirrors and a static host are equal sources, because the name checks the bytes. A WebRTC peer swarm (uor-comms' WebRTC with ntfy.sh setup) fits the same rule: a peer is one more untrusted source, and its pieces pass the same check. Peers pay only when the origin is the bottleneck, and many visitors hold the same model.

### Storage and return visits

Verified pieces are written to the private file system as they are delivered, in the background (the hub's `opfsStore` or this repository's `files.mjs`, keyed by address). A return visit reads and re-hashes them through the same rule, so offline equals warm.

## Measurements

Machine: this Windows 11 machine, 24 logical cores. Browser: the Claude desktop app's built-in Chromium pane. Model: onnx-community/Qwen3-0.6B-ONNX at `79346d3`, `onnx/model_q4f16.onnx`, 569,789,750 bytes, 508 tensors. Source: Hugging Face's CDN. Prototype: one page and one module that use the hub's `piece-hash.wasm`, `pieces.mjs` and worker unchanged, with tables fetched live from gethologram.ai. Dates: 2026-11-20, 19:20 to 20:30 UTC. Each figure is one run unless stated. The link varied between 6 and 29 Mbit/s over the hour, so numbers from different runs carry that spread.

| | Today's shape: one GET, hashed as it arrives, checked at the end | Streaming, tensor tables: forward order, 8 requests of up to 8 MiB, 22 hash workers | Streaming, file table: forward order, 4 requests of up to 8 MiB, 22 hash workers, pieces kept in the private file system |
| --- | --- | --- | --- |
| Index and tables before the first request | 1.1 s | 1.1 s + 1.1 s (87 tables, 17,288 bytes) | 1.1 s (the table, 17,432 bytes, derived earlier and folded to the record's BLAKE3) |
| First byte | 0.29 s | 0.85 s | 0.49 s |
| First verified, usable byte | 299.2 s | 0.85 s | 0.49 s |
| Graph bytes verified | 299.2 s | not covered | 0.53 s |
| Layer 0 usable (graph, prompt rows, rotary rows, layer 0) | 299.2 s | 6.3 s (graph not covered) | 12.7 s (layer 0 itself at 6.1 s; the six prompt-row pieces came last) |
| All 28 layers usable | 299.2 s | 227 s (172.6 s without two straggling ranges) | 122.7 s |
| Every byte verified | 299.2 s | 385.9 s (569,507,840 bytes covered) | 275.4 s (all 569,789,750 bytes) |
| Throughput | 15.2 Mbit/s | 11.8 Mbit/s | 16.6 Mbit/s |
| Hashing time | 0.43 s on the main thread in total | 0.94 s of piece-hash time summed over workers | 0.76 s summed over workers |

The three runs were made one after another, so each saw a different moment of the link. Compare the times to first use, which are orders of magnitude apart, more than the totals, which track the link.

Hashing alone, with the bytes already in memory: one thread 1.40 GB/s; 22 workers 3.71 GB/s (256 MiB, wasm32 SIMD). At this link's 2 MB/s, hashing is about 0.14 % of one core. It is not on the critical path, and a GPU hasher would buy nothing.

Verified against raw throughput, the same Range plan with and without hashing, alternated in 32 and 48 MiB windows (15 runs):

| | Mbit/s, each run | median |
| --- | --- | --- |
| raw, 1 request | 29.1, 14.0, 9.1 | 14.0 |
| raw, 8 requests | 9.5, 9.7, 8.4 | 9.5 |
| raw, 16 requests | 11.8 | |
| verified, 1 request | 11.9, 12.8, 8.9 | 11.9 |
| verified, 8 requests | 9.8, 15.3, 6.1 | 9.8 |
| verified, 8 requests through the redirect each time | 10.8 | |

Verified throughput is the raw throughput within this link's noise. Parallel requests did not raise throughput in the browser here; curl over HTTP/1.1, outside the browser, got 18.1 Mbit/s on one connection and 23.7 Mbit/s summed over four. In the browser the ranges likely share one HTTP/2 connection, though resource timing does not expose the protocol cross-origin, so that is not confirmed. The one cost of eight requests that was measured is stragglers: in the full run, layer 5 finished at 212 s and layer 19 at 227 s, while the layers around them finished by 47 s and 120 s.

Return visit, network off. The pieces kept by the file-mode run were read back from the private file system and every piece was verified again against the file table. `fetch` was replaced by one that refuses, and it was called 0 times. The index rows were read before the network was cut; the page already keeps those on a return visit.

| Workers | Every byte re-verified | First verified byte | Layer 0 usable | All layers usable |
| --- | --- | --- | --- | --- |
| 22 (two runs) | 0.40 s, 0.44 s (1.29 to 1.41 GB/s) | 5 to 6 ms | 26 to 29 ms | 190 to 219 ms |
| 0 (main thread) | 0.70 s (0.81 GB/s) | 3.5 ms | 41 ms | |

Today's page takes 9.5 s to be ready warm (below). At most 0.44 s of that is verification done this way; the rest is reading and importing on one thread and building the engine.

### Time to first token

Qwen3-0.6B's head is its embedding (tied weights, no `lm_head` initializer in the export), so the first logit needs all 311 MB of it. The first token therefore cannot come before every byte has arrived. Streaming does not lower that floor. What it removes is everything today's path does after the last byte: hashing (0.43 s), the import, decision 28, building the engine and uploading to the GPU, and the prefill of all 28 layers. With streaming, prefill for layers 0 to 27 completes while the head downloads (layers were usable by 172.6 to 227 s, the head by 381 s). Logits can be computed piece by piece as head pieces are verified (512 rows per piece), so the work left after the last byte is one piece's logits.

Today's page, measured on the same machine and pane at 2026-11-20 20:00 to 20:15 UTC (the published page at hologram-technologies.github.io/prism-hologram-ai/app/ with `?onnx=onnx-community/Qwen3-0.6B-ONNX&precision=q4f16`; prompt "Hello, how are you?"; 32 tokens):

| | Cold | Warm (files kept in this browser) |
| --- | --- | --- |
| Ready, as the page reports it | 371.4 s | 9.5 s |
| After the last byte: import ended, decision 28, roles, device, engine built | about 5.2 s (from the page's check timestamps: import 373.8 s, ready verdict 374.1 s, built 379.0 s on the page clock) | |
| First token after Send | 15.21 s | 15.24 s |
| Tokens per second | 0.7 | 0.7 |
| First token from the click | about 386.6 s | about 24.7 s |

Two things follow. Cold, the download is about 95 % of the time to the first token on this link. Streaming can hide the roughly 5 s of serial work after the last byte and the 15 s prefill under the 150 s head download, a gain of about 20 s of 387 (an upper bound from the parts above, not measured end to end). Warm, bytes are not the bottleneck: 15.2 s of 24.7 s is the engine's prefill, and the 9.5 s load is reading, re-hashing and importing 570 MB on one thread. Re-hashing alone, through the piece pool, is measured below.

The first token was not measured end to end with streaming, because the engine does not accept tensors as they arrive. That is the first build item below.

## What changes where

| Repository | Change |
| --- | --- |
| hub | Publish a file table (`HPTREE01`) for every ONNX graph and data file, named by the file's BLAKE3, beside the tensor tables. Put the UOR root's BLAKE3 under the signed index root, so tensor addresses are trusted as the record is. Serve all of one model's tables as one file, which removes 87 requests for Qwen3-0.6B; measured at 1.1 s. |
| hologram | None needed. `PieceTree` is the format. Publishing hologram-space (now `publish = false`) would let crates depend on it rather than copy it. |
| prism-hologram-ai | Import the hub's piece rule (HK-39 to HK-42) as a kernel decision. Let `files.mjs` read pieces in forward-pass order, through a worker hash pool, and keep them by address. Let `onnx-load.mjs` take tensors as they complete, with decision 28 asked per tensor. Let the integer engine accept a layer's weights when that layer is admitted and run prefill layer by layer. Compute the head in pieces as they arrive. Each change starts as a registered ID, a scenario and a failing test (R3). |

## Ranked next steps

1. **Admit and place per layer, with prefill as layers arrive** (prism-hologram-ai). The first usable layer moves from 299 s to 6.1 to 6.3 s, and all 28 layers are usable 123 s into a 275 s download (measured, file mode). The first token does not move as far. All three milestone 2 models tie their output layer to their embedding (`tie_word_embeddings` is true in each config.json, and no export has an `lm_head` initializer), so the first token needs every byte of every one of them. What step 1 can hide is the work after the last byte: about 5 s of serial work and the 15 s prefill measured on today's page. That is at most about 20 s of 387 s on this link (5 %), a bound from measured parts. On a link fast enough that the download is shorter than the compute, the same 20 s is most of the time to the first token. Cost: the importer (`crates/model-reader`, inside the sealed engine module) has to expose the graph's nodes and each tensor as it completes; that means a reseal, a kernel decision for pieces, file tables from the hub, and an engine that places weights per layer.
2. **File tables from the hub** (hub). This closes the trust gap without signing the UOR index, and covers the 282 KB of graph bytes. Cost: 17 KB per 570 MB. The table was derived and checked here.
3. **Fewer bytes before the first token.** On this link the bytes are the whole cost: 570 MB at 15 Mbit/s is about 5 minutes, and hashing is 0.14 % of that. The tied embedding is 55 % of Qwen3-0.6B's tensor bytes, 38 % of Qwen2.5-1.5B's (466,747,392 of 1,220,818,578) and 48 % of Llama 3.2 1B's (525,336,576 of 1,089,605,632). A 4-bit head under its own address would cut the head by about three quarters. Ilya's 2026-11-18 answer allows a head derived in the browser as a measured option; a head published by the hub is a new question. Second-precision deduplication is already measured at 24 to 54 % fewer bytes.
4. **Stragglers.** Hedge the slowest range from a second source or connection once it falls behind the others. Measured cost without hedging: two layers arrived 165 s and 107 s after their neighbours.
5. **Peers and mirrors.** These help only when the origin, not the visitor's link, is the limit. Here the visitor's link is the limit, so the gain is not measurable on this machine.
6. **Warm load off the main thread.** Today's warm load is 9.5 s; re-verifying every kept byte through the piece pool takes 0.40 to 0.44 s (measured). The other 9 s or so is the one-thread read, import and engine build. That is the largest warm gain, and it is not about bytes.
7. **Prefetch.** Warm the next likely model in the background, by address, once the current one is kept.

## Questions only Ilya can answer

1. Is the file table, rooted in the signed record's whole-file BLAKE3, the admission authority? The alternative is to sign the UOR root and admit by tensor tables.
2. May the hub publish a 4-bit head for tied-embedding models under its own address, so the first token waits for about 350 MB rather than 570 MB on Qwen3-0.6B (77.8 MB of 4-bit codes and 9.7 MB of f16 scales in place of 311 MB)?
3. Should the engine accept a layer before the whole model is admitted? That changes decision 28 from one verdict per model to one per tensor.
4. Is a peer swarm in scope for milestone 2, or later?
