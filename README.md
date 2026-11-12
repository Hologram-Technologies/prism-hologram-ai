# Hologram AI, as a Prism app

Run open models privately in a browser tab, with no server, where every byte
and every decision can be checked.

This repository rebuilds
[hologram-ai](https://github.com/Hologram-Technologies/hologram-ai) in three
planes, each trusted in a different way:

| Plane | What it is | How it is trusted |
| --- | --- | --- |
| Kernel | A Prism holo: every decision the app makes about what it may load and run, written in LexLean. | Proved by Lean, in proof modules beside the app. Examples replayed in the compiled guest. |
| Engine | Rust compiled to WebAssembly on the [hologram](https://github.com/Hologram-Technologies/hologram) runtime: download, hashing, compile, inference, ONNX import. | Tested against external authorities. Not proved. |
| Page | The interface and its workers. | Tested by browser journeys. Not proved. |

The engine and the page never decide policy. They ask the kernel.

## Status

Only the kernel has begun. What exists is listed in
[CONFORMANCE.md](CONFORMANCE.md), which is generated from
[model/ids.toml](model/ids.toml); a capability that is not built has no row.

| Decision | Claim | State |
| --- | --- | --- |
| 2, preflight verdict | A model is supported or derived exactly when no required key is missing, no unsupported feature is flagged and the manifest fits, and only then may weights download. | Proved (`supported_decides`, `derived_decides`, `authorised_decides`) |
| 3, memory plan | A proposed stage plan is accepted exactly when its stage fits the memory budget, is the largest that does, and its stages cover every layer once. | Proved (`planned_decides`, `can_run_decides`) |
| 4, quantisation tier | The tier chosen fits in memory whenever any tier does, and more memory never lowers it. | Proved (`tier_fits`, `tier_monotone`, `half_up_bounds`) |
| 5, hash admission | A buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal. | Proved (`admitted_decides`) |
| 6, tensor range | A tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file. | Proved (`within_decides`, `accepted_decides`, `all_accepted_ordered`) |
| 7, context trimming | A proposed trim is accepted exactly when what remains fits the context and restoring the newest dropped pair would not. | Proved (`trimmed_decides`, `trim_fits_decides`) |
| 8, decode bucket | The decode bucket is at least what is needed, never above the maximum, and never below the current bucket. | Proved (`fits_decides`, `grow_bounds`) |
| 9, generation bound | A turn's token budget never exceeds the context that remains and never exceeds the request. | Proved (`has_room_decides`, `budget_within_context`, `budget_within_request`) |
| 11, shape binding | A requested dimension is bound exactly when it lies within its bounds, to a size at least the request and never above the upper bound. | Proved (`in_bounds_decides`, `bind_bounds`) |

The engine and the page are not in this repository yet. The plan and its
order are in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md); the
brief it follows is [docs/REBUILD_PROMPT.md](docs/REBUILD_PROMPT.md).

## The kernel's request format

A request is a fixed binary layout. Byte 0 names the decision. Each field after
it is an unsigned big-endian number of fixed width. Each decision accepts one
exact length; any other request answers `invalid request`. Answers are short
text.

Decision 5, 81 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `5` |
| 1 to 32 | expected label, the BLAKE3 hash the archive names |
| 33 to 64 | observed label, the BLAKE3 hash of the bytes received |
| 65 to 72 | declared length |
| 73 to 80 | observed length |

It answers `admit` or `refuse`.

Decision 6, 33 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `6` |
| 1 to 8 | file length |
| 9 to 16 | end of the previous tensor, or the start of the data for the first |
| 17 to 24 | offset of this tensor |
| 25 to 32 | length of this tensor |

It answers `accept` or `reject`. The caller folds it over a file's tensors in
order, passing each accepted tensor's end as the next previous end.

Decision 8, 25 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `8` |
| 1 to 8 | length needed |
| 9 to 16 | current bucket |
| 17 to 24 | the model's maximum context |

It answers the new bucket as decimal text, `context exceeded` when the need is
past the maximum, or `bucket exceeds maximum` when the current bucket already is.

Decision 9, 25 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `9` |
| 1 to 8 | context length |
| 9 to 16 | tokens used |
| 17 to 24 | tokens requested, or 0 for no cap |

It answers the turn's token budget as decimal text, or `context full`.

Decision 2, 5 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `2` |
| 1 | family recognised, 0 or 1 |
| 2 | required config keys missing, a count |
| 3 | manifest fits the recipe, 0 or 1 |
| 4 | unsupported features flagged, a count |

It answers `supported`, `derived`, or `rejected:` with its reason. Weights may
download only on the first two.

Decision 4, 17 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `4` |
| 1 to 8 | parameter count |
| 9 to 16 | memory budget in bytes |

It answers `int8`, `int4` or `int8 paged`.

Decision 11, 33 bytes:

| Bytes | Field |
| --- | --- |
| 0 | `11` |
| 1 to 8 | lower bound |
| 9 to 16 | upper bound |
| 17 to 24 | requested size |
| 25 to 32 | step to round up to, or 0 for exact |

It answers the bound size as decimal text, `below minimum` or `above maximum`.

Decision 7, 41 bytes. The engine proposes a trim and the kernel checks it:

| Bytes | Field |
| --- | --- |
| 0 | `7` |
| 1 to 8 | context length |
| 9 to 16 | the fixed part: system prompt, newest user message and reserve |
| 17 to 24 | total of every question-and-answer pair |
| 25 to 32 | total of the pairs dropped |
| 33 to 40 | the newest dropped pair |

It answers `trim`, or the condition that failed: `too long`, `does not fit`,
`drops too many` or `dropped exceeds total`.

Decision 3, 49 bytes. The engine proposes a stage plan and the kernel checks it:

| Bytes | Field |
| --- | --- |
| 0 | `3` |
| 1 to 8 | memory budget in bytes |
| 9 to 16 | context bytes |
| 17 to 24 | bytes for one layer, at most 2^40 |
| 25 to 32 | layers, at most 65536 |
| 33 to 40 | layers in a stage |
| 41 to 48 | stages |

It answers `plan`, or `cannot run`, `stage does not fit`, `stage too small` or
`stages do not cover`.

## Run the gate

Everything runs inside the SDK image that `prismpm.lock` pins; nothing else is
installed on the host.

```sh
just vv
```

`just vv` is the whole acceptance boundary: formatting, the claim register, the
template contract, lints, tests, dependency policy, `lexlean verify` over the
proof modules, which has the Lean kernel check every theorem, and
`prismpm verify`, which builds the holo and replays its examples in the
compiled guest.

To add a capability, follow [AGENTS.md](AGENTS.md): register it, write its
scenario and a failing test, build it, then plant the defect the test exists to
catch.

```sh
just new-capability KN-06 build kernel "<the statement>"
```

## Layout

| Path | Purpose |
| --- | --- |
| `app/src/Kernel.lex.tex` | The kernel's entry: routes a request to its decision, and carries the examples. |
| `app/src/Kernel/` | One module per decision, and the wire format they share. Only what the guest executes. |
| `app/src/Proofs/` | The propositions and theorems about those modules, kept outside the application's build. |
| `app/src/Foundation/` | The template's byte and text-application library. |
| `model/` | The claim, authority and ledger registers. |
| `features/suites/` | One scenario per registered claim. |
| `crates/conformance/tests/kernel.rs` | One test per claim, pinning what its theorem says. |
| `site/app.json` | What the app page says about the kernel. |
| `docs/` | The brief and the implementation plan. |

[KERNEL-VERIFICATION.md](KERNEL-VERIFICATION.md) records each gate's planted
defect. [TEMPLATE-CONTRACT.md](TEMPLATE-CONTRACT.md),
[VERIFICATION.md](VERIFICATION.md) and
[TEMPLATE-VERIFICATION.md](TEMPLATE-VERIFICATION.md) are inherited from
[UOR-Foundation/template](https://github.com/UOR-Foundation/template) and
define the policy this repository is held to.

## License

MIT or Apache-2.0, at your option.
