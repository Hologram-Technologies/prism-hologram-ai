# Kernel verification

What each gate over the kernel has been seen to reject. A gate is evidence only
after a planted defect shows it can fail ([AGENTS.md](AGENTS.md)); this file
records the defect, the command, the diagnostic and the restored result.

The inherited gates and their own planted defects are in
[TEMPLATE-VERIFICATION.md](TEMPLATE-VERIFICATION.md).

## What is proved, and what is tested

| Claim | Evidence | Kind |
| --- | --- | --- |
| `KN-05`: a buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal | `admitted_decides`, checked by the Lean kernel, axioms `[propext]` | proved |
| `KN-06`: a tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file | `within_decides` and `accepted_decides`, checked by the Lean kernel, axioms `[Classical.choice, Quot.sound, propext]` | proved |
| A list of ranges is all accepted exactly when each starts at or after the end of the one before it and ends within the file | `all_accepted_ordered`, by induction over the list, same axioms | proved |
| Any two ranges of an accepted list, not only neighbours, are disjoint | follows from the ordered form by transitivity; no theorem here states it, because language 1.2 has no proof form that uses an induction hypothesis that is an implication | not proved |
| The verdict for each decision is the one its proved function gives, read from the request offsets the wire format fixes | `kernel_kn_05` and `kernel_kn_06` in `crates/conformance/tests/kernel.rs` | tested (string-matching, not a theorem) |
| The compiled guest gives the stated answer for each example | the 29 examples in `app/src/Kernel.lex.tex`, replayed by `prismpm verify` | tested |

Each proof is about the function the guest executes, named by its module. That
the request bytes are parsed into that function's arguments correctly is not
proved; the examples exercise it, with a one-bit difference in each label word
and in the low and high bytes of each number.

## Where the proofs live, and why

`app/src/Kernel.lex.tex` and the modules under `app/src/Kernel` hold only what
the guest executes. The propositions and theorems are in `app/src/Proofs`. Each
proof module imports the executable definitions and states its theorems about
them by qualified name, so there is one definition of each function and the
theorem is about that one.

Two failures of the SDK's build led here. Both appeared on the first push of
`KN-06` and are recorded as found.

| Failure | Cause | What changed |
| --- | --- | --- |
| `PP5007: application-lean timed out`, on both runners | The SDK gives the Lean build of the application's module graph a fixed 600 seconds. With the arithmetic and induction proofs in that graph the build exceeded it. | The proofs moved out of the application's module graph. The same stage then finished in about four minutes. |
| `application-cargo-check exited 101`, `error[E0428]: the name answer is defined multiple times` | The generated guest puts every executable module's declarations in one Rust namespace. Two modules each declared `answer`. | They are `admissionAnswer` and `rangeAnswer`. Every executable declaration has a name no other executable module uses. |

The proofs are still checked on every gate run. The `app` recipe of the
`Justfile` runs `lexlean verify src/Proofs/*.lex.tex`, which has the Lean
kernel and leanchecker check every proof module with its exact axiom policy, and
sets `failglob` so a pattern that matches nothing is an error. `lexlean verify`
exits with an error when Lean rejects a proof; that was checked directly on a
planted defect.

## The layout guards: `cargo test -p repo-conformance --test kernel`

Both pinned tests call one guard before anything else. It discovers the
executable modules from the folder, so a new decision is covered without
editing a list.

| Planted defect | Result |
| --- | --- |
| A theorem placed in an executable module, where the proof gate does not look | rejected: `app/src/Kernel/Admission.lex.tex holds a theorem the proof gate does not read` |
| The `lexlean verify src/Proofs/*.lex.tex` line removed from the `app` recipe | rejected: `the app recipe must run lexlean verify over every proof module` |
| `shopt -s failglob` removed, so an empty match would pass | rejected: the same message |
| One name declared in two executable modules | rejected: `admissionAnswer is declared in both app/src/Kernel/Admission.lex.tex and app/src/Kernel/Range.lex.tex` |

## KN-05, hash admission

### The proof gate: `lexlean verify src/Proofs/Admission.lex.tex`

| Planted defect | Result |
| --- | --- |
| `sameLabel`, in the executable module, no longer compares label word 2 | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Admission (error): unsolved goals`, the goal showing `w0`, `w1` and `w3` on the left and all four words on the right |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The theorem's statement weakened from `<->` to `->` | rejected: `KN-05's statement changed: admitted must be true exactly when Admitted holds` |
| The observed label read from offset 32 instead of 33 | rejected: `the claim must be read from offsets 1, 33, 65 and 73` |
| The proposition `Admitted` drops the length comparison | rejected: `Admitted must require the declared and observed lengths to be equal` |
| The verdict decided by the labels alone, not by `admitted` | rejected: `the verdict must be decided by admitted, the function KN-05 proves` |
| Decision 5 routed to no answer | rejected: `decision 5 must be routed to the admission answer` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-05 is registered
and not built`, as rule R3 requires.

### The examples

Fifteen. Two admit: the empty buffer under the BLAKE3 hash of the empty input,
and the three bytes `abc` under theirs. Seven refuse: one bit flipped in each of
the four label words, the length off by one, the length differing only in its
top byte, and both label and length different. Six are invalid: one byte short,
one byte long, empty, decision 0, decision 255, and a request of exactly the
allocation maximum.

## KN-06, tensor range

### Why the decided function is not written as the statement

The statement is `previousEnd <= offset` and `offset + length <= fileLength`.
A natural number is unbounded in the model and 64 bits wide in the compiled
guest, where a result outside that width is an overflow failure, never a
wrapped value. A decided function that formed `offset + length` would therefore
fail, with no verdict, on a request its proof says it rejects. `within` compares
the length with `fileLength - offset` instead, after checking
`offset <= fileLength`, and `within_decides` proves that equal to the statement.
The subtraction is why the proof draws on `Classical.choice`: Lean's linear
arithmetic uses it for truncated subtraction. The first version, which formed
the sum, rested on `propext` alone.

### The proof gate: `lexlean verify src/Proofs/Range.lex.tex`

| Planted defect | Result |
| --- | --- |
| `within`, in the executable module, no longer checks that the offset is inside the file | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Range (error): omega could not prove the goal`, with the counterexample `offset - fileLength >= 1` |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| `within` forms `offset + length`, which overflows 64 bits | rejected: `within must compare the length with the room left, not form offset + length` |
| The arithmetic statement weakened from `<->` to `->` | rejected: `KN-06's arithmetic statement changed: within must be true exactly when Within holds` |
| `Within` drops the end-of-file bound | rejected: `Within must be the mathematical statement, with the sum written as a sum` |
| Offset and length read from each other's bytes | rejected: `the span must be read from offsets 1, 9, 17 and 25` |
| `accepted` hands `within` the offset as the previous end | rejected: `accepted must hand the four fields to within in its parameter order` |
| `Ordered` does not pass each range's end to the next | rejected: `Ordered must require each range within bounds and pass its end to the next` |
| The verdict decided by the length check alone, not by `accepted` | rejected: `the verdict must be decided by accepted, the function KN-06 proves` |
| Decision 6 routed to no answer | rejected: `decision 6 must be routed to the range answer` |
| The 64-bit overflow example expects `accept` | rejected: `no example rejects a range whose offset plus length exceeds 64 bits` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-06 is registered
and not built`.

### The examples

Fourteen. Five accept: the first tensor after a 128-byte header filling the
file, both bounds tight, an empty tensor at the previous end, an empty tensor at
the end of the file, and the largest range the fields can name. Five reject: a
start one byte inside the previous tensor, an end one byte past the file, a
start past the file, and two at the top of the 64-bit field where offset plus
length exceeds 64 bits. Four are invalid: one byte short, one byte long, and
each of decisions 5 and 6 sent with the other's length.

## The examples in the compiled guest

`prismpm --project app verify` builds the holo and replays every example in the
compiled guest. Independently, `npx prism-code vectors` on the built holo
replayed 29 of 29 with the holo's identities verified, including the three
examples at the top of the 64-bit field.

One gate here has not been shown to fail: no example has been planted with a
wrong expected answer to confirm the replay rejects it.
