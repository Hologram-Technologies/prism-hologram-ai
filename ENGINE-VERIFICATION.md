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

## What the rules mean for the engine to come

Rule R5, as the gate enforces it, lets a shipped crate return `Result` only
over three error names the template fixes. The kernel client meets that by
having no function that can fail. Code that reads untrusted files and talks to
the network cannot be written that way, so how the engine's failures are
modelled has to be settled before that code is added: it is a decision about
the rule, and it is not made here.
