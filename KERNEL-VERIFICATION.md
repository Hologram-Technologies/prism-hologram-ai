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
| The verdict for decision 5 is the one `admitted` gives, read from the request offsets the wire format fixes | `kernel_kn_05` in `crates/conformance/tests/kernel.rs` | tested (string-matching, not a theorem) |
| `KN-06`: a tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file | `within_decides` and `accepted_decides`, checked by the Lean kernel, axioms `[Classical.choice, Quot.sound, propext]` | proved |
| A list of ranges is all accepted exactly when each starts at or after the end of the one before it and ends within the file | `all_accepted_ordered`, by induction over the list, same axioms | proved |
| Any two ranges of an accepted list, not only neighbours, are disjoint | follows from the ordered form by transitivity; no theorem here states it | not proved |
| The verdict for decision 6 is the one `accepted` gives, read from the request offsets the wire format fixes | `kernel_kn_06` in `crates/conformance/tests/kernel.rs` | tested (string-matching, not a theorem) |
| The compiled guest gives the stated answer for each example | the examples in `app/src/Kernel.lex.tex`, replayed by `prismpm verify` | tested |

The proof is about the model's `admitted`. That the request bytes are parsed
into the claim correctly is not proved; the examples exercise it, with a
one-bit difference in each of the four label words and in the low and high
bytes of the length.

## KN-05, hash admission

All commands ran inside the SDK image `prismpm.lock` pins.

### The proof gate: `lexlean verify`

| Planted defect | Command | Result |
| --- | --- | --- |
| `sameLabel` no longer compares label word 2 | `lexlean verify src/Kernel/Admission.lex.tex` | rejected: `error[LLV7002]: Lean rejected HologramAi.Kernel.Admission (error): unsolved goals`, the goal showing `w0`, `w1` and `w3` on the left and all four words on the right |
| none | the same command | `verified 3 modules` |

### The pinned test: `cargo test -p repo-conformance --test kernel`

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

### The examples: `prismpm --project app verify`

Fifteen examples. Two admit: the empty buffer under the BLAKE3 hash of the
empty input, and the three bytes `abc` under theirs. Seven refuse: one bit
flipped in each of the four label words, the length off by one, the length
differing only in its top byte, and both label and length different. Six are
invalid: one byte short, one byte long, empty, decision 0, decision 255, and a
request of exactly the allocation maximum.

## KN-06, tensor range

### Why the decided function is not written as the statement

The statement is `previousEnd <= offset` and `offset + length <= fileLength`.
A natural number is unbounded in the model and 64 bits wide in the compiled
guest, where a result outside that width is an overflow failure, never a
wrapped value. A decided function that formed `offset + length` would therefore
fail, with no verdict, on a request its proof says it rejects. `within` compares
the length with
`fileLength - offset` instead, after checking `offset <= fileLength`, and
`within_decides` proves that equal to the statement. The subtraction is why the
proof draws on `Classical.choice`: Lean's linear arithmetic uses it for
truncated subtraction. The first version, which formed the sum, rested on
`propext` alone.

### The proof gate: `lexlean verify`

| Planted defect | Command | Result |
| --- | --- | --- |
| `within` no longer checks that the offset is inside the file | `lexlean verify src/Kernel/Range.lex.tex` | rejected: `error[LLV7002]: Lean rejected HologramAi.Kernel.Range (error): omega could not prove the goal`, with the counterexample `offset - fileLength >= 1` |
| none | the same command | `verified 3 modules` |

### The pinned test: `cargo test -p repo-conformance --test kernel`

| Planted defect | Result |
| --- | --- |
| `within` forms `offset + length`, which can overflow 64 bits | rejected: `within must compare the length with the room left, not form offset + length` |
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

Fourteen examples. Five accept: the first tensor after a 128-byte header
filling the file, both bounds tight, an empty tensor at the previous end, an
empty tensor at the end of the file, and the largest range the fields can
name. Five reject: a start one byte inside the previous tensor, an end one byte
past the file, a start past the file, and two at the top of the 64-bit field
where offset plus length exceeds 64 bits. Four are invalid: one byte short, one
byte long, and each of decisions 5 and 6 sent with the other's length.
