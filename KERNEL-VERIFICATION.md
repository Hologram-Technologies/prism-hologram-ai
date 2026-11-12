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
| `KN-08`: the decode bucket is at least what is needed, never above the maximum, and never below the current bucket | `grow_bounds`, under the precondition `fits_decides` decides, checked by the Lean kernel, axioms `[Classical.choice, Quot.sound, propext]` | proved |
| `KN-09`: a turn's token budget never exceeds the context that remains and never exceeds the request | `budget_within_context`, `budget_within_request` and `has_room_decides`, checked by the Lean kernel | proved |
| A number in an answer is the decimal text of the number computed | `decimal` in `app/src/Kernel/Wire.lex.tex`; examples at 0, 1, 5, 48, 64, 128, 256, 300, 1948, 2048, 8192 and 2^64 - 1 | tested, not proved |
| `KN-02`: a model is supported or derived exactly when no required key is missing, no unsupported feature is flagged and the manifest fits, and only then may weights download | `supported_decides`, `derived_decides` and `authorised_decides`, checked by the Lean kernel | proved |
| `KN-04`: the quantisation tier chosen fits in memory whenever any tier does, and more memory never lowers it | `tier_fits`, `tier_monotone` and `half_up_bounds`, checked by the Lean kernel | proved |
| `KN-11`: a requested dimension is bound exactly when it lies within its bounds, to a size at least the request and never above the upper bound | `in_bounds_decides` and `bind_bounds`, checked by the Lean kernel | proved |
| The size a dimension is bound to is a multiple of the step, unless capped at the upper bound | examples only; the proof language does not carry the divisibility argument | tested, not proved |
| int8 costs one byte for each parameter and int4 half a byte | an assumption of `KN-04`, stated in its model; tensors left unquantised are not counted | assumed |
| The verdict for each decision is the one its proved function gives, read from the request offsets the wire format fixes | the seven `kernel_kn_` tests in `crates/conformance/tests/kernel.rs` | tested (string-matching, not a theorem) |
| The compiled guest gives the stated answer for each example | the 100 examples in `app/src/Kernel.lex.tex`, replayed by `prismpm verify` | tested |

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
| `application-cargo-check exited 101`, `error[E0428]: the name answer is defined multiple times` | The generated guest puts every executable module's declarations in one Rust namespace. Two modules each declared `answer`. | They are `admissionAnswer` and `rangeAnswer`, and later decisions follow the pattern. Every executable declaration has a name no other executable module uses. |

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

## KN-08, decode bucket

`grow` returns the current bucket when it already covers the need; otherwise
twice the current bucket, capped at the maximum, when that covers the need;
otherwise the need itself. `doubled` forms `current + current` only after
checking that the current bucket is at most the room left below the maximum, so
the sum cannot pass 64 bits.

A need past the maximum answers `context exceeded`. A current bucket already
past the maximum answers `bucket exceeds maximum`. The bucket is computed only
when `fits` holds.

### The proof gate: `lexlean verify src/Proofs/Bucket.lex.tex`

| Planted defect | Result |
| --- | --- |
| `doubled`, in the executable module, returns `current + current` without regard to the maximum | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Bucket (error): omega could not prove the goal`, with a counterexample in which twice the current bucket passes the maximum |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The statement drops the never-shrinks bound | rejected: `KN-08's statement changed: the new bucket must cover the need, stay within the maximum and not shrink` |
| `Fits` no longer bounds the current bucket | rejected: `Fits must require both the need and the current bucket within the maximum` |
| `doubled` adds before checking the room, which overflows 64 bits | rejected: `doubled must check the room before it adds` |
| The bucket is computed without checking `fits` | rejected: `the bucket must be grow's, computed only when fits holds, from offsets 1, 9 and 17` |
| The need and the current bucket read from each other's bytes | rejected: the same message |
| Decision 8 routed to no answer | rejected: `decision 8 must be routed to the bucket answer` |
| The widest number expected with a digit missing | rejected: `no example prints the widest 64-bit number` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-08 is registered
and not built`.

### The examples

Fourteen. Ten answer a bucket: the first bucket from nothing, doubling that
covers the need, a bucket already enough, a smaller need that does not shrink
it, a need doubling does not reach, doubling capped at the maximum, a bucket at
the maximum, zero, and two at the top of the 64-bit field. One answers
`context exceeded`, one `bucket exceeds maximum`. Two are invalid: one byte
short and one byte long.

## KN-09, generation bound

`budget` is the room that remains, capped by the request when one was made; a
request of zero means no cap. The room is `context - used`, a subtraction that
cannot overflow. A context with no room left answers `context full`, and "room
left" is strict: `used < context`.

### The proof gate: `lexlean verify src/Proofs/Generation.lex.tex`

| Planted defect | Result |
| --- | --- |
| `budget`, in the executable module, returns the request without regard to the room | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Generation (error): omega could not prove the goal`, with a counterexample in which the request exceeds `context - used` |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The statement drops the at-least-one bound | rejected: `KN-09's statement changed: the budget must fit the context that remains and be at least one` |
| The request bound weakened to a statement that says nothing | rejected: `the budget must never exceed a request that was made` |
| `hasRoom` stated to admit a context that is exactly full | rejected: `hasRoom must be true exactly when used is below the context` |
| `budget` ignores the room when a request is made | rejected: `budget must be the room, capped by a request that was made` |
| The budget is computed without checking `hasRoom` | rejected: `the budget must be budget's, computed only when hasRoom holds, from offsets 1, 9 and 17` |
| Decision 9 routed to no answer | rejected: `decision 9 must be routed to the generation answer` |
| The exactly-full example expects a budget | rejected: `no example refuses a context that is exactly full` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-09 is registered
and not built`.

### The examples

Twelve. Eight answer a budget: all the room when no request is made, a request
that fits, a request cut to the room, one token of room, an empty context, the
smallest context, and two at the top of the 64-bit field. Two answer
`context full`: exactly full and past full. Two are invalid: one byte short and
one byte long.

## KN-02, preflight verdict

Four flags come from the engine: whether the family is recognised, how many
required config keys are missing, whether the manifest fits the recipe, and how
many unsupported features are flagged. `verdictCode` gives one of six codes and
`verdictText` its text. A model is runnable when no key is missing, no
unsupported feature is flagged and the manifest fits. A missing key is named
before an unsupported feature, and that before a manifest that does not fit. A
flag that is neither 0 nor 1 answers `invalid request`.

Weights may download only when `authorised` holds, and `authorised_decides`
proves that is exactly when the model is runnable.

### The proof gate: `lexlean verify src/Proofs/Preflight.lex.tex`

| Planted defect | Result |
| --- | --- |
| `verdictCode`, in the executable module, gives a verdict without checking that the manifest fits | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Preflight (error): omega could not prove the goal`, for each of the three theorems |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| `Runnable` drops the manifest condition | rejected: `Runnable must require no missing key, no unsupported feature and a manifest that fits` |
| The supported theorem weakened from `<->` to `->` | rejected: `KN-02's supported_decides changed: it must hold exactly when the model is runnable` |
| Weights authorised on the manifest rejection too | rejected: `authorised must be exactly the derived and supported verdicts` |
| The supported verdict reads `derived` | rejected: `verdict 4 must read supported` |
| The first two flags read from each other's bytes | rejected: `the verdict must be verdictCode's, from the four flag bytes at offsets 1 to 4` |
| Decision 2 routed to no answer | rejected: `decision 2 must be routed to the preflight answer` |
| The everything-wrong example names the feature, not the missing key | rejected: `no example names the missing key first when everything is wrong` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-02 is registered
and not built`.

### The examples

Fourteen. One supported, one derived. Three name a missing key, one an
unsupported feature, two a manifest that does not fit, and two more show the
order when several things are wrong. Four are invalid: each flag byte set to 2,
one byte short and one byte long.

## KN-04, quantisation tier

`tierCode` is 2 for int8 in memory when the parameters fit the budget at one
byte each, 1 for int4 in memory when half of them, rounded up, fit, and 0 for
int8 paged otherwise. `halfUp` is the parameters less half of them rounded
down, which cannot overflow, and `half_up_bounds` proves it is the rounded-up
half.

An earlier wording of this claim, "more memory never gives a lower-quality
tier", was false: a small budget gives int8 paged and a medium one gives int4.
The order the theorem is about is paged, int4 in memory, int8 in memory.

### The proof gate: `lexlean verify src/Proofs/Tier.lex.tex`

| Planted defect | Result |
| --- | --- |
| `tierCode`, in the executable module, answers int8 in memory whenever the rounded-up half fits | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Tier (error): omega could not prove the goal`, with a counterexample in which the parameters exceed the budget |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The theorem no longer requires a tier in memory when one fits | rejected: `KN-04's statement changed: the chosen tier must fit, and a tier in memory must be chosen whenever one fits` |
| The monotone theorem compares a tier with itself | rejected: `more memory must never lower the tier` |
| `tierCode` prefers int4 to int8 | rejected: `tierCode must prefer int8 in memory, then int4 in memory, then paged` |
| `halfUp` rounds down | rejected: `halfUp must be the parameters less half of them rounded down` |
| The int8 tier reads `int4` | rejected: `each tier code must read as its tier` |
| Parameters and budget read from each other's bytes | rejected: `the tier must be tierCode's, from the two numbers at offsets 1 and 9` |
| Decision 4 routed to no answer | rejected: `decision 4 must be routed to the tier answer` |
| The odd count just past int4 expects `int4` | rejected: `no example answers int8 paged for 0000000200000001 parameters in 4 GiB` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-04 is registered
and not built`.

### The examples

Fifteen. Five answer `int8`, four `int4`, four `int8 paged`, on each side of
both boundaries in 4 GiB and at the top of the 64-bit field. Two are invalid:
one byte short and one byte long.

## KN-11, shape binding

A request is bound when it lies between the lower and upper bounds. `gapTo` is
the distance to the next multiple of the step, or nothing for a step of 0 or 1,
and the remainder is never taken by zero. `bind` adds the gap only after
checking the room below the upper bound, so the sum cannot pass the bound or 64
bits; otherwise it answers the upper bound. A request below the lower bound
answers `below minimum`, one above the upper bound `above maximum`.

### The proof gate: `lexlean verify src/Proofs/Shape.lex.tex`

| Planted defect | Result |
| --- | --- |
| `bind`, in the executable module, adds the gap without regard to the upper bound | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Shape (error): omega could not prove the goal`, with a counterexample in which the request plus the gap passes the upper bound |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The in-bounds theorem drops the upper bound | rejected: `KN-11's statement changed: inBounds must be true exactly when the request lies within its bounds` |
| The bind theorem drops the upper bound | rejected: `the bound size must be at least the request and never above the upper bound` |
| `bind` adds without checking the room, which overflows 64 bits | rejected: `bind must check the room before it adds, and cap at the upper bound` |
| A step of 1 is no longer exact | rejected: `gapTo must be the distance to the next multiple of the step, or nothing for a step of 0 or 1` |
| The size is computed without checking `inBounds` | rejected: `the size must be bind's, computed only when inBounds holds, from offsets 1, 9, 17 and 25` |
| The two refusals swapped | rejected: `a request below the lower bound and one above the upper bound must each be named` |
| Decision 11 routed to no answer | rejected: `decision 11 must be routed to the shape answer` |
| The capped example expects the uncapped multiple | rejected: `no example binds 00000000000007d0 to 2040` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-11 is registered
and not built`.

### The examples

Sixteen. Eleven answer a size: rounded up to a multiple, already a multiple,
exact for a step of 0 and of 1, the next multiple equal to the upper bound, the
next multiple past it and so capped, and at the top of the 64-bit field. Three
are refused: two below the minimum, one of them under bounds that admit nothing,
and one above the maximum. Two are invalid: one byte short and one byte long.

## The examples in the compiled guest

`prismpm --project app verify` builds the holo and replays every example in the
compiled guest. Independently, `npx prism-code vectors` on the built holo
replayed 29 of 29 with the holo's identities verified, including the three
examples at the top of the 64-bit field.

One gate here has not been shown to fail: no example has been planted with a
wrong expected answer to confirm the replay rejects it.
