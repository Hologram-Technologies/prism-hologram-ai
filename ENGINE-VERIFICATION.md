# Engine verification

What each gate over the engine has been seen to reject. A gate is evidence only
after a planted defect shows it can fail ([AGENTS.md](AGENTS.md)); this file
records the defect, the command, the diagnostic and the restored result.

The engine is tested, never proved. What is proved is in
[KERNEL-VERIFICATION.md](KERNEL-VERIFICATION.md).

## What the engine is, so far

One crate, `crates/kernel-client`: it writes the kernel's requests and reads its
answers. It has no dependencies, uses no allocation and no unsafe code, and no
function in it can fail: a request is built from typed arguments into a
fixed-size array, and an answer the kernel is not known to give reads as
`Unrecognised`, which a caller must handle and cannot mistake for a verdict.

Nothing that downloads, compiles or runs a model is here yet. That code needs
the hologram runtime, and rule R6 admits only crates.io dependencies; the
runtime's crates are not published there.

## What is tested

| Claim | Evidence | Kind |
| --- | --- | --- |
| `EN-01`: the kernel client writes each request exactly as the kernel's examples have it and reads every answer the kernel gives | `engine_en_01` in `crates/conformance/tests/engine.rs`, over every example in `app/src/Kernel.lex.tex` | tested |
| A number in an answer is read only when it is written as the kernel writes one | `decimal_reads_canonical_numbers` and `decimal_refuses_what_the_kernel_never_writes` in `crates/kernel-client/src/lib.rs` | tested |
| The client accepts no answer text beyond what the kernel gives | not tested: the test reads the answers the examples give, and cannot enumerate text no example contains | not tested |

## EN-01, the kernel client

The test reads the kernel's examples from the model itself, so an example added
or changed there is tested the next time the suite runs. For each example it
does two things.

Writing. When the request is a well-formed request of its decision, the test
takes the fields out of the request's own bytes, hands them to the client, and
requires the bytes the client writes to equal the request. A field at the wrong
offset, in the wrong order, at the wrong width or in the wrong byte order cannot
survive that. Every decision must have at least five requests reproduced.

Reading. The answer must read as a verdict the client knows, and never as
`Unrecognised`. Two different answers may not read as the same verdict. A
numeric answer must read as the number whose decimal text it is. And every
verdict the client knows must be given by some example, so the client cannot
carry a verdict the kernel never gives.

Then the negative cases: wrong case, trailing bytes, a leading zero, a sign and
a number one past 64 bits all read as `Unrecognised` for every decision.

### The gate: `cargo test -p hologram-ai-kernel-client -p repo-conformance`

| Planted defect | Result |
| --- | --- |
| The trim request writes the dropped total and the newest dropped pair in each other's place | rejected: `the client writes decision 7's request differently from the kernel's example` |
| The bucket request names decision 9 | rejected: the same message, for decision 8 |
| The admission request writes the observed label before the expected one | rejected: the same message, for decision 5 |
| A number is written little-endian | rejected: the same message, for decision 3, the first decision the test reaches |
| The reset event is written with the fail event's code | rejected: the same message, for decision 1 |
| The preflight request drops the manifest flag | rejected: the same message, for decision 2 |
| `derived` is expected with a capital letter | rejected: `decision 2 answered derived, which the client reads as Unrecognised` |
| `reject` reads as accept | rejected: `decision 6: two different answers both read as Accept` |
| The `stage too small` answer is not known | rejected: `decision 3 answered stage too small, which the client reads as Unrecognised` |
| A number with a leading zero is read as a number | rejected: the unrecognised-text assertion and `decimal_refuses_what_the_kernel_never_writes` |
| A number past 64 bits wraps instead of being refused | rejected: the same two |
| A manifest that does not fit still authorises a download | rejected: `ManifestDoesNotFit and downloading` |
| none, before and after | passed |

Before the crate existed, the scaffolded test failed with `EN-01 is registered
and not built`, as rule R3 requires.

## How the engine may fail

Rule R5 says every public failure is typed, modelled, documented and covered by
a negative test. Here that means a shipped crate (any crate under `crates/`
without `publish = false`) fails only in the ways `model/failures.toml` lists.

The register is the single source. Each row is one failure: an `FL-` ID, the
owning crate, the error type, the variant, the suite of its negative scenario,
and the condition in one sentence. `CONFORMANCE.md` lists the rows, generated
by `just model-write`. The register is empty: the kernel client has no
function that can fail. A row is added in the commit that adds code that can
return it; `just new-failure` writes the row, its scenario and a failing test.

### What the gate checks

`audit-limits` (in `just model`), for every shipped crate:

- every `Result<` in shipped code names, as its error type, a type registered
  for that crate; `Result<T>` aliases such as `io::Result`, foreign types, and
  `Result` renamed on import are rejected;
- each registered type is declared once, as an `enum`, not `#[non_exhaustive]`,
  and its variants equal the crate's rows in both directions;
- no variant is a catch-all (`Other`, `Unknown`) or carries free text or an
  unnamed error (`String`, `str`, `dyn`, `anyhow`);
- every crate root has `#![forbid(clippy::unwrap_used, clippy::expect_used,
  clippy::panic)]`, and `just lint` still runs clippy over every crate and
  target. `forbid` cannot be lifted by an inner `#[allow]`. Clippy, in
  `just lint`, is what rejects the call; the gate checks the attribute is
  there. Unit tests inside a crate are exempt through `clippy.toml`.

The meta-gate (`repo-conformance`, in `just test`) requires each registered
failure to have a scenario tagged `@FL-NN @build` in its suite, and a test in
`crates/conformance/tests/<suite>.rs` whose name ends in the ID and whose body
names `Type::Variant` in code. A mention in a comment or a string does not
count, and an `#[ignore]`d or `#[cfg]`-gated test discharges nothing.

Both gates read source through one small lexer (`crates/conformance/src/source.rs`):
comments and literal contents are blanked, and a `#[cfg(test)]` item hides only
itself. The line-based reader it replaces had two holes: by its rule, after
`#[cfg(test)] use x;` it skipped everything up to the next closing brace, and a
`{` inside a test's string literal hid all the code after the test module.

### What the gate does not establish

- That a test provokes its failure. The gate checks that the test names the
  variant; the planted defect recorded for each row shows the test fails
  without the failure.
- That the engine never panics. Out-of-bounds indexing and slicing, `assert!`,
  `unreachable!` and arithmetic overflow in debug builds can still panic; none
  of them is denied, and no claim is made about them. The kernel client's
  `put` slices a fixed array at fixed offsets.
- Failure signalled any way other than `Result`: an `Option`, a sentinel value
  or a `fmt::Result` is not read as a failure.
- Code reached from outside `src/` (`#[path]`, `include!`) is rejected, not
  read; a target declared with `[lib]` or `[[bin]]` is rejected for the same
  reason.

### The kernel's verdicts

A kernel verdict is an answer, not an engine failure. The client reads every
answer into a typed verdict and never fails. Where the engine must stop
because the kernel refused, it returns one variant that carries the kernel's
verdict, such as `Refused(preflight::Verdict)`, so the policy stays in the
kernel and is never restated in Rust.

### The gate: `just model`, `just lint`, `just bdd`

Planted in a throwaway shipped crate, `crates/plant`, with one registered
failure, `FL-01` (`PlantError::Refused`), its scenario and its test. Each
defect was planted alone, the gate run, the files restored and the gate run
again. Every restore passed. The crate, its row, scenario and test were then
removed.

| Planted defect | Command | Diagnostic |
| --- | --- | --- |
| `pub fn g() -> Result<(), &'static str>` | `just model` | `` `Result<_, &'static str>`: `&'static str` is not an error type registered for plant `` |
| An extra variant `Late` in `PlantError` | `just model` | `` `PlantError::Late` has no row in model/failures.toml `` |
| A row `FL-02` for `PlantError::Stalled`, not in the enum | `just model-write && just model` | `` FL-02 registers `PlantError::Stalled`, which the enum at crates/plant/src/lib.rs:7 does not declare `` |
| `Refused(String)` | `just model` | `` `PlantError::Refused` carries `String` `` |
| A registered variant `Other` | `just model-write && just model` | `` `PlantError::Other` is a catch-all `` |
| `Some(verdict).unwrap()` in `plan` | `just lint` | `` used `unwrap()` on an `Option` value ``, pointing at the `forbid` |
| The `#![forbid(...)]` line removed | `just model` | `` does not `#![forbid(clippy::unwrap_used)]` ``, and the same for `expect_used` and `panic` |
| The `FL-01` scenario removed | `just bdd` | `FL-01 is a registered failure but has no scenario in features/suites/plant.feature` |
| The test renamed `plant_refusal` | `just bdd` | ``no test in crates/conformance/tests/plant.rs has a name ending in `fl_01` `` |
| The test marked `#[ignore]` | `just bdd` | the same |
| The test names `PlantError::Refused` only in a comment | `just bdd` | `` the test for FL-01 does not name `PlantError::Refused` in code `` |
| The `FL-01` condition edited, `CONFORMANCE.md` not regenerated | `just model` | `CONFORMANCE.md is stale: it disagrees with model/*.toml` |
| `#[cfg(test)] use core::fmt;` then an unregistered `Result` | `just model` | `` crates/plant/src/lib.rs:24: `Result<_, &'static str>` `` ... |
| none, before and after | all three | passed |

Before `FL-01`'s test was written, the scaffold left it red twice, as rule R3
requires: `plant_fl_01` panicked with `FL-01 is registered and not built`, and
the meta-gate reported ``the test for FL-01 does not name `PlantError::Refused`
in code``, because the scaffold names the variant only in its message.

The same rejections are kept as unit tests over scratch repositories:
`audit::tests` in `xtask/src/audit.rs`, `source::tests` in
`crates/conformance/src/source.rs`, and
`a_scaffolded_failure_is_red_until_its_test_names_the_variant` in
`xtask/src/scaffold.rs`.
