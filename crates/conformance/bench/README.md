# Milestone 2 harnesses: equality (rows 2, 3) and speed (rows 11, 16)

Why: the last rows of milestone 2 (`docs/Q_ENGINE_PLAN.md`, "Done when") are
measured on real models, which are too large for the gate. These harnesses
run them the same way every time, on this machine, a named laptop and a phone,
and write what they measure to `model/ledger.toml` as open claims. Nothing in
this folder ships; transformers.js and ONNX Runtime are used here for
comparison only.

## Equality: rows 2 and 3

The integer engine's token ids over 256 tokens of three fixed prompts
(`equality-run.mjs`), the same on the CPU and on WebGPU, for Qwen2.5-1.5B-Instruct,
Llama 3.2 1B and Qwen3-0.6B, each loaded from its ONNX repository at q4f16.
The run goes past the model's end of turn, so every one of the 256 ids is
compared; where the end of turn fell is recorded.

| Step | Command | Writes |
| --- | --- | --- |
| CPU reference | `node equality.mjs <site> <cache> <key> <out.json>` | the integer CPU engine's ids, run in Node on a deployed site's own files |
| Float agreement | `python3 -P float_agreement.py <out.json> <cache> <float.json>` | ONNX Runtime 1.26.0 on the same file: teacher-forced argmax and free greedy run against the reference |
| Ledger | `node ledger.mjs equality <out.json> <float.json>` | `OPEN-GREEDY-256-<MODEL>-INTEGER` |
| WebGPU check | `node equality-browser.mjs <site> <key> <out.json>` | the same run in Chrome on device webgpu, compared id by id with `references/<key>.json`; exit 0 only when all 768 ids are equal |

`<site>` is a deployed site: the Pages artifact of a run (`gh run download
<run> -n github-pages`, then untar), or `just page-journey`'s
`target/page-journey`. `<key>` is `qwen2.5-1.5b`, `llama-3.2-1b` or
`qwen3-0.6b`. The cache mirrors Hugging Face's resolve paths, so the float
side opens the very file the integer engine read.

`equality-browser.mjs` refuses to compare unless the page's engine for the
device is the integer one. Until row 15 lands the integer GPU engine, the
page's WebGPU engine is the float one and the check stops there by name; the
day it lands, the same command closes rows 2 and 3. `--device cpu` runs the
browser's integer CPU engine against the same references, which checks the
harness and the browser against Node.

## Speed: rows 11 and 16

`speed-run.mjs` is the protocol, `m2-speed/1`, run in the browser by the same
code for this engine and for transformers.js 4.3.2: only the module
`pipeline` is imported from differs. Both run on the published page, in a
browser profile that holds nothing when the load starts.

| Quantity | How |
| --- | --- |
| Load, cold | `pipeline()` call to answer, nothing cached: download, checks, upload |
| Download size | bytes of every response body through the page's `fetch` during the load, by host |
| First token, cold | the first turn after the load, call to first token |
| First token, warm | after one warm-up turn, the median of five turns on a short prompt (about 20 to 50 tokens framed) |
| Tokens per second | over the same turns: tokens after the first, over the time from the first token to the turn's end; tokens counted by the model's own tokenizer for both libraries |
| Long prompt | about 440 to 480 tokens framed, three turns, every one timed |
| Guest call overhead | the kernel guest answering its smallest non-empty acceptance example, warm, median of 20 batches of 200 calls |
| Holo size | the deployed `app.holo` in bytes |

Up to 128 new tokens a turn, greedy, Qwen3's thinking off for both libraries.
The CPU device (`wasm`, this engine's integer CPU engine) runs the short leg
only, loaded from the browser's copy after WebGPU's cold load, so each model
is downloaded once per library.

```sh
node speed.mjs --lib hologram     --model onnx-community/Qwen3-0.6B-ONNX --machine "<laptop>" --out hologram.json
node speed.mjs --lib transformers --model onnx-community/Qwen3-0.6B-ONNX --machine "<laptop>" --out transformers.json
node ledger.mjs speed Z13 hologram.json transformers.json --bound warm-first-token="<the step that bounds it>"
```

On a phone: enable USB debugging, `adb forward tcp:9222
localabstract:chrome_devtools_remote`, open Chrome on the phone, then add
`--cdp http://127.0.0.1:9222` and use `PHONE` as the machine id. The page's,
the CDN's and Hugging Face's stored data are cleared before the first load.

Row 16's targets (warm first token under 300 ms on Qwen3-0.6B, 100 tokens per
second on Qwen2.5-1.5B at 4 bits, guest call under 50 µs, holo under 16 MB)
are judged on this engine's integer engines only: WebGPU once the deploy's
WebGPU engine is integer, the CPU until then. `ledger.mjs` refuses to write a
missed target without the step that bounds it, and never writes a target
other than the plan's.
