# Hologram AI, as a Prism app

Run open models privately in a browser tab, with no server, where every byte
and every decision can be checked.

This repository rebuilds
[hologram-ai](https://github.com/Hologram-Technologies/hologram-ai) in three
planes, each trusted in a different way:

| Plane | What it is | How it is trusted |
| --- | --- | --- |
| Kernel | A Prism holo: every decision the app makes about what it may load and run, written in LexLean. | Proved by Lean. Examples replayed in the compiled guest. |
| Engine | Rust compiled to WebAssembly on the [hologram](https://github.com/Hologram-Technologies/hologram) runtime: download, hashing, compile, inference, ONNX import. | Tested against external authorities. Not proved. |
| Page | The interface and its workers. | Tested by browser journeys. Not proved. |

The engine and the page never decide policy. They ask the kernel.

## Status

Only the kernel has begun. What exists is listed in
[CONFORMANCE.md](CONFORMANCE.md), which is generated from
[model/ids.toml](model/ids.toml); a capability that is not built has no row.

| Decision | Claim | State |
| --- | --- | --- |
| 5, hash admission | A buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal. | Proved (`admitted_decides`) |
| 6, tensor range | A tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file. | Proved (`within_decides`, `accepted_decides`, `all_accepted_ordered`) |

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

## Run the gate

Everything runs inside the SDK image that `prismpm.lock` pins; nothing else is
installed on the host.

```sh
just vv
```

`just vv` is the whole acceptance boundary: formatting, the claim register, the
template contract, lints, tests, dependency policy, and `prismpm verify`, which
builds the holo, replays its examples in the compiled guest and has the Lean
kernel check every theorem.

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
| `app/src/Kernel/` | One module per decision, and the wire format they share. |
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
