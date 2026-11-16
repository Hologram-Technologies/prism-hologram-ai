# Plan: deterministic, verifiable and interpretable inference in any browser

Companion to [Q_ENGINE_PLAN.md](Q_ENGINE_PLAN.md). It is written as the brief
for the session that builds it: everything that session needs is here or
linked.

Status on 2026-11-16: proposed. The kernel decisions 29 to 33 below are
written in LexLean with their proofs, registered as `KN-29` to `KN-33`,
scenarios in `features/suites/kernel.feature` and pinned by tests in
`crates/conformance/tests/kernel.rs`. Nothing in the engine is built.

## The user experience

What a person can do once this is built, and could not do before:

1. **Check any answer.** Every response carries a receipt: the model's
   address, the prompt and settings, and the tokens that came out. Pressing
   *Check* re-runs it in their own tab, on their own device, and shows
   *confirmed* or *mismatch*. No server is trusted, because none is involved.
2. **Share a conversation that proves itself.** A link carries the receipt,
   not the text. The recipient's browser regenerates the conversation and
   confirms every token. An edited transcript fails.
3. **See inside an answer.** Any layer, position and component of a past
   answer can be opened and inspected. The value is recomputed on demand and
   is the value that produced the answer, so nothing has to be recorded while
   generating.
4. **The same answer on every device.** A phone without WebGPU, a laptop and a
   desktop give the same tokens. The page says which promise holds for the
   current run: *the same everywhere*, *the same on this hardware*, or *not
   reproducible*, and never claims more than the run earned.
5. **Fast first token for prompts others have run.** A prompt prefix
   computed once, by anyone, is reused by its address, so a returning visitor
   or a common system prompt skips that work. It is reused across devices
   only when the run is *the same everywhere*.
6. **Any browser, no install, no server.** Chromium, Safari and Firefox, on
   desktop and phone, from a static page.

## Why

Hosted AI asks to be trusted. This engine already runs open models in the
tab, admits every weight block by its BLAKE3 name, and decides policy in a
proved kernel. What it cannot yet do is show that an answer is the answer:
floating-point inference differs between GPUs, so a re-run on another
device proves nothing.

Exact integer arithmetic removes that. A product of an 8-bit activation and
a ternary or 4-bit weight, summed in a 32-bit integer, has one value on every
device. When every stage is exact, the tokens are a function of the model,
the prompt and the settings alone, and anyone can recompute them.

Prior art, studied on 2026-11-16: Gensyn's RepOps gives bit-identical
floating-point results across supported GPUs by fixing reduction order (about
60% overhead, datacenter CUDA); EigenAI gives bit-exact results on fixed GPU
architectures with signed receipts; TOPLOC checks inference approximately
with hashed activations; WebLLM and transformers.js run in the browser with no
verification. No system found does bit-identical inference across GPU vendors
in a browser tab, checkable by any visitor. That is the claim to earn, and it
stays `open` until measured.

## How

The kernel decides; the engine computes; the page asks and shows.

| Plane | Adds | Trusted by |
| --- | --- | --- |
| Kernel | Decisions 29 to 33 below | Proved by Lean |
| Engine | Integer kernels on WebGPU and the CPU, fixed-point softmax, normalisation and rotary encoding, receipts, replay, probes, shared prefix cache | Tested against `uor-matmul` and a float reference |
| Page | Check, share, inspect, and the promise shown per run | Browser journeys |

Arithmetic comes from [`uor-matmul`](https://github.com/UOR-Foundation/uor-matmul)
0.1.0 on crates.io (R6 admits it): `uor-matmul-codec` already has `Ternary`,
`Sign` and `Packed<Grid<16>, 2>`, which are Bonsai's ternary, Bonsai 8B's one
bit and this engine's q4. It is the exact CPU path and the reference every
WGSL kernel is tested equal to. Do not port verdant-aqm's TypeScript
re-creation of it, its hand-assembled WebAssembly or its asserted flags: they
break R1, R4 and R6.

## What: the kernel decisions

Each is proved, routed in `app/src/Kernel.lex.tex`, and answered as text.

| # | Decision | Request | Answers | Proved |
| --- | --- | --- | --- | --- |
| 29 | Exact accumulator | 5 bytes: activation bits, weight bits, group bits (the bits of the group length), accumulator bits | `exact`, `may overflow` | `exact_fits_decides`, `exact_fits_wider`, `exact_fits_shorter` |
| 30 | Reproducibility class | 4 bytes: matrix products exact, other stages exact, float reductions in a fixed order (each 0 or 1) | `everywhere`, `same hardware`, `none` | `class_everywhere_decides`, `class_same_hardware_decides`, `class_none_decides`, `comparable_decides` |
| 31 | Receipt replay | 68 bytes: claimed and replayed output labels; whether decision 28 found the receipt's model ready; the class; whether the adapter is the same | `confirmed`, `mismatch`, `not comparable` | `receipt_confirmed_decides`, `receipt_mismatch_decides`, `receipt_not_comparable_decides` |
| 32 | Probe | 21 bytes: layer and layers, position and positions, component and components, the class, same adapter | `exact replay`, `approximate replay`, `out of range` | `probe_exact_decides`, `probe_approximate_decides`, `probe_out_of_range_decides` |
| 33 | Shared prefix | 76 bytes: decision 27's request, then where the prefix was computed, the class, same adapter | `reuse`, `recompute` | `shareable_decides` |

A replay is comparable exactly when the class is `everywhere`, or the class is
`same hardware` and the adapter is the same. Decisions 31, 32 and 33 share
that one definition, so a receipt, a probe and a shared prefix can never
disagree about what a run promised.

What the kernel does not prove, and who owns it:

| Not proved | Owner |
| --- | --- |
| That the bit widths given to 29 are the true widths of the engine's numbers | Engine, tested against the formats' definitions |
| That `a + w + g <= b` bits suffice for a signed sum of `2^g` products | Argued in [KERNEL-VERIFICATION.md](../KERNEL-VERIFICATION.md), not proved |
| That the flags given to 30 describe the kernels actually run | Engine, tested by the differential tests below |
| That each label is the BLAKE3 of what it names | The caller's hash, as for every label |
| That the replay was run on the receipt's own input | The engine: it replays from the receipt, so the input is the receipt's by construction |

Why decision 31 compares only the output: the kernel takes at most 81 bytes a
request (`requestMaximum`), and raising that waits for the next SDK image
([ENGINE-VERIFICATION.md](../ENGINE-VERIFICATION.md)). The model is already
checked by decision 28 against its root, so 31 takes that answer as a flag and
compares the one label left.

## What: the engine work, in order

1. **Exact CPU path** in `crates/cpu-decode` on `uor-matmul-codec`, built for
   wasm with SIMD, threads when cross-origin isolated. Activations quantised to
   int8 per row with an integer scale. Done when it runs Bonsai 1.7B and
   Qwen2.5-1.5B q4 to completion.
2. **Fixed-point non-matmul stages**: softmax by integer exponent tables,
   RMSNorm by integer square root, rotary encoding by integer sine and cosine
   tables, all with declared widths that decision 29 accepts. Done when the CPU
   path is integer from token ids to logits.
3. **Integer WGSL kernels** mirroring step 1: int8 by ternary and by 4-bit,
   32-bit accumulators per group, `dot4I8Packed` where the adapter offers
   `packed_4x8_integer_dot_product`, a scalar integer kernel elsewhere.
   Cross-group sums in two 32-bit words, since WGSL has no 64-bit integers.
4. **Receipts**: the model's root, the input label (BLAKE3 of token ids and
   settings), the output label, the class from decision 30, the adapter. A
   receipt fits in a URL fragment.
5. **Check and replay** in the page through decision 31.
6. **Probes** through decision 32: recompute a layer's output at a position
   and show it beside the token it led to.
7. **Shared prefix cache**: prefix state stored in OPFS by label, offered to
   peers and the hub by label, admitted through decision 33.
8. **Speed**: fusion first (about 450 dispatches a token today), then batched
   prefill on the integer path, then split attention.

Tests added, each with its planted defect:

| Kind | Test |
| --- | --- |
| Reference | `uor-matmul` and the WGSL kernel give the same 32-bit sums on generated blocks of every format |
| Differential | The CPU path and the WebGPU path give the same logits, bit for bit, over 256 tokens of three prompts |
| Differential | Chromium, WebKit and Gecko give the same token ids on the same receipt |
| Differential | Batched prefill equals one position at a time, bit for bit |
| Quality | The integer path's greedy tokens against the float reference, agreement recorded per prompt in `model/ledger.toml`, never dropped |
| Kernel conformance | Every request the page sends for 29 to 33 and the answer it acts on equal the kernel's |
| Journey | A receipt checked on a second browser engine says confirmed; a receipt with one changed output token says mismatch; a receipt of class `none` says not comparable |

## Targets

Reported, not gated, until measured on the named laptop and phone; then a
regression of more than 10 percent fails.

| Measure | Target |
| --- | --- |
| Time to first token, short prompt, cold | under 300 ms |
| Time to first token, shared prefix hit | under 100 ms |
| Tokens per second, Bonsai 1.7B, desktop | 100 or more |
| Tokens per second, phone | measured first, then set |
| Cross-vendor agreement | identical token ids on NVIDIA, AMD, Intel, Apple and the CPU |
| Integer path quality | greedy agreement with the float reference, per model, recorded |

The claim *bit-identical across GPU vendors in a browser* is `open` in
`model/ids.toml` terms until the cross-vendor differential test passes on
real hardware.

## Done when

A visitor opens the page on a phone, asks a question, sends the link to a
second person on a desktop in another browser, and that person presses
*Check* and sees *confirmed*, then opens layer 12 at the last position and
sees the same values the phone computed.
