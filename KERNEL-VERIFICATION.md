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
| `KN-01`: each working state is entered from exactly one state by exactly one event, failed is left only by reset, and every other step is refused | the eight `enters_` theorems and `failed_only_resets`, checked by the Lean kernel | proved |
| Ready is reached only after preflight passed, the download finished, the compile finished and every label verified, in that order | follows from the `enters_` theorems by the argument in the `KN-01` section; no theorem states it, because it is a claim about whole sequences of events | argued, not proved |
| `KN-02`: a model is supported or derived exactly when no required key is missing, no unsupported feature is flagged and the manifest fits, and only then may weights download | `supported_decides`, `derived_decides` and `authorised_decides`, checked by the Lean kernel | proved |
| `KN-04`: the quantisation tier chosen fits in memory whenever any tier does, and more memory never lowers it | `tier_fits`, `tier_monotone` and `half_up_bounds`, checked by the Lean kernel | proved |
| `KN-11`: a requested dimension is bound exactly when it lies within its bounds, to a size at least the request and never above the upper bound | `in_bounds_decides` and `bind_bounds`, checked by the Lean kernel | proved |
| The size a dimension is bound to is a multiple of the step, unless capped at the upper bound | examples only; the proof language does not carry the divisibility argument | tested, not proved |
| int8 costs one byte for each parameter and int4 half a byte | an assumption of `KN-04`, stated in its model; tensors left unquantised are not counted | assumed |
| `KN-07`: a proposed trim is accepted exactly when what remains fits the context and restoring the newest dropped pair would not | `trimmed_decides` and `trim_fits_decides`, checked by the Lean kernel | proved |
| `KN-03`: a proposed stage plan is accepted exactly when its stage fits the memory budget, is the largest that does, and its stages cover every layer once | `planned_decides` and `can_run_decides`, checked by the Lean kernel | proved |
| `KN-10`: a graph is importable exactly when its opset lies within the implemented window and no operator is unimplemented | `importable_decides`, checked by the Lean kernel | proved |
| A wider window never makes an importable graph unimportable, and a window that runs backwards is no question | `window_monotone` and `reversed_window_is_invalid`, checked by the Lean kernel | proved |
| The opset window and the count of operators outside it are what the importer reports | the kernel is told numbers and answers; which operators an importer builds is the importer's, and no importer exists here yet | not proved |
| The engine's proposed trim drops whole pairs, oldest first, and the totals it reports are the totals of those pairs | the kernel sees totals, not the pairs; `EN-10` in `crates/conformance/tests/engine.rs` tests it of the engine | tested, not proved |
| A plan exists whenever one layer fits beside the context | the kernel checks a plan, it does not produce one; an engine test, not yet written | not proved |
| `KN-12`: WebGPU is chosen exactly when the adapter is present and the largest buffer fits its binding limit, and a wider limit never takes WebGPU away | `gpu_usable_decides` and `limit_monotone`, checked by the Lean kernel | proved |
| Whether a model fits memory at all | left to `KN-03` and `KN-04`; decision 12 answers only which backend | not decided here |
| `KN-13`: a weight block's shape is accepted exactly when rows and columns are at least one and the columns divide into the format's groups | `shaped_decides` and `group_bounds`, checked by the Lean kernel | proved |
| The block length is rows times groups times bytes per group, with q4 20 per 32, q3f 16 per 32, t2 4 per 16, t2r 68 per 256 the ternary-per-128 format 36 per 128, and f32 vectors (the norms) 64 per 16 | the pinned test fixes the two tables; checked once against a real tensor of the Bonsai 1.7B weight container (2048 by 2048, 1,179,648 bytes); no examples yet | tested, not proved |
| `KN-14`: a claimed greedy pick is accepted exactly when its index is the first position holding the maximum score and the claimed score is that maximum | `standing_decides`, and `holds_decides` by induction over the positions, checked by the Lean kernel | proved |
| The engine's f32 logits map to 32-bit keys whose order is the logits' order | an engine property; the engine must also send negative zero as positive zero, or a tie between them is broken differently | not proved |
| `KN-15`: a token is kept unpenalised exactly when no position inside the recent window holds it | `clear_decides`, and `clear_from_decides` by induction over the history, checked by the Lean kernel | proved |
| `KN-16`: a speculative commit count is accepted exactly when it lies in 1 to k+1, every draft before the last committed position equals its verified pick, and the draft at that position, if any, differs | `agrees_decides`, and `agrees_from_decides` by induction over the drafts, checked by the Lean kernel | proved |
| `KN-17`: unpacking a place of a packed group gives back the code packed there, for q4, q3f (its ten fields and the two codes in its spare bits), t2 and t2r, and the ternary sign and nonzero planes | `nibble_round_trip`, `triple_round_trip`, `spare_round_trip`, `pair_low_round_trip`, `pair_high_round_trip` and `bit_round_trip`, checked by the Lean kernel | proved |
| The answer to decision 17 is the code minus the format's offset, written with its sign | `signedText`; no examples yet | tested, not proved |
| `KN-18`: a name is a listed operator's name exactly when the four words of its 32 bytes equal that name's | `opname_is_decides`, checked by the Lean kernel, axioms `[propext]` | proved |
| The decomposed style lists 27 operators and the fused style 18, and a style other than those two is no question | `kernel_kn_18` holds the names and checks that the model's numbers are their bytes and that it lists no other; the lists were read from the graphs of onnx-community/Qwen2.5-1.5B-Instruct (decomposed) and of Llama-3.2-1B-Instruct-ONNX, Qwen3-0.6B-ONNX and SmolLM2-360M-Instruct (fused), each `model_q4f16.onnx`, on 2026-11-16. That a name is supported exactly when it is on its style's list is not a theorem: the proof language has no disjunction, and the forms it has did not carry a statement over 27 alternatives | tested, not proved |
| `KN-19`: a configuration is supported exactly when its type is listed, its activation is silu and its shape is one the engine runs | `family_supported_decides`, `family_unknown_decides`, `family_listed_decides` and `family_shaped_decides`, checked by the Lean kernel | proved |
| The listed types are qwen2, qwen3 and llama, and the activation silu | `kernel_kn_19` computes the eight bytes of each name and finds them in the model | tested |
| A recognised configuration is a model the engine computes correctly | the GP suite's models are Qwen2- and Qwen3-shaped; no Llama-shaped model has been run | not tested |
| `KN-20`: a MatMulNBits shape is accepted exactly when the part is codes, f32 scales or f16 scales, rows and columns are between one and 16,777,215 (so the length fits the twenty digits an answer has) and the block size is 16, 32, 64, 128 or 256; the blocks of a row are the fewest that cover it | `nbits_shaped_decides`, `nbits_block_bounds`, and `nbits_blocks_fewest_16` to `_256`, one per block size because the proof forms divide by constants only, checked by the Lean kernel | proved |
| The length is rows times blocks times the bytes of one block | pinned by `kernel_kn_20`; a product of two unknowns is outside the proof forms. It matches the four files read: codes `[N, ceil(K/32), 16]` bytes and f16 scales `N * ceil(K/32)` elements, with no zero points. A file that carries zero points is outside this decision | tested, not proved |
| `KN-21`: the precision chosen is the first offered one that fits, none exactly when none fits, and a wider budget never moves the choice later | `precision_fits_decides`, `precision_first`, `precision_second`, `precision_third`, `precision_none` and `precision_budget_monotone`, checked by the Lean kernel | proved |
| The sizes put to decision 21 are what each precision costs on this device | the caller's; a precision whose f16 tensors must be widened costs more than its file | not proved |
| `KN-22`: WebGPU exactly when `gpuUsable` holds, the CPU exactly when it does not and the model is within the ceiling, refused otherwise | `device_webgpu_decides`, `device_cpu_decides` and `device_refused_decides`, checked by the Lean kernel | proved |
| The CPU ceiling is a size a CPU path can run usefully | the ceiling is a number the caller states; no CPU path exists yet to measure it against | not proved |
| `KN-23`: a span is accepted exactly when its file is declared, it starts at or after the previous span's end and it ends within the file; two spans accepted in order do not overlap | `chunk_declared_decides`, `chunk_inside_decides` and `chunk_spans_disjoint`, checked by the Lean kernel | proved |
| `KN-24`: a file is named exactly when the task is text generation, feature extraction or text classification (0 to 2, KN-53), the precision is listed, at most 100 data files are declared and the part is at most that count | `source_named_decides`, `source_undeclared_decides` and `source_task_decides`, checked by the Lean kernel | proved |
| The names are the ones transformers.js 4.3.2 reads: `onnx/model`, the precision's suffix, `.onnx`, then `_data` and `_data_1` onward | `sourceGraph` and `sourceName`, pinned by `kernel_kn_24`; text is outside the proof forms | tested, not proved |
| `KN-25`: no stop id is met exactly when every listed id differs from the token; the budget is spent exactly when the tokens generated reach it | `stop_differs_decides`, `stop_none_from_decides` by induction over the list, and `stop_spent_decides`, checked by the Lean kernel | proved |
| `KN-26`: evict exactly when the block is not pinned, a candidate remains and space is short; does not fit exactly when none remains and space is short; enough exactly when it is not short | `evict_decides`, `evict_does_not_fit_decides`, `evict_enough_decides` and `evict_short_decides`, checked by the Lean kernel | proved |
| Candidates are put in the order least recently used first, and the amounts free and freed are true | the caller's; the kernel is asked about one block at a time and keeps no state | not proved |
| `KN-27`: a prefix is reused exactly when its label equals the one asked for, it holds at least one position and it is shorter than the prompt | `reusable_decides` and `reuse_shorter_decides`, with `sameLabel` as decision 5 proves it, checked by the Lean kernel | proved |
| The label asked for is derived from the model's address and the prompt's first tokens | the caller's hash, as every label is | not proved |
| `KN-28`: a manifest is complete exactly when it lists at least one tensor and as many are admitted as it lists | `manifest_complete_decides`, checked by the Lean kernel | proved |
| The label of the addresses admitted is computed over the tensors actually admitted, in the manifest's order | the caller's; the kernel compares two labels and two counts | not proved |
| `KN-29`: a sum is accepted as exact exactly when the activation and weight bits are at least one and activation, weight and group bits together are no wider than the accumulator; a wider accumulator or a shorter group never loses it | `exact_fits_decides`, `exact_fits_wider` and `exact_fits_shorter`, checked by the Lean kernel | proved |
| That `a + w + g <= b` bits hold a signed sum of `2^g` products of `a`-bit and `w`-bit signed numbers without overflow | argued: each product's magnitude is at most `2^(a+w-2)`, the sum's at most `2^(a+w+g-2)`, below a `b`-bit signed maximum of `2^(b-1) - 1` | not proved |
| The bit widths the engine gives decision 29 are the widths of the numbers it actually sums | the caller's | not proved |
| `KN-30`: a run is `everywhere` exactly when its products and its other stages are exact, `same hardware` exactly when it is not and its float reductions keep a fixed order, `none` otherwise; a run is comparable exactly when it is `everywhere`, or `same hardware` on the same adapter | `class_everywhere_decides`, `class_same_hardware_decides`, `class_none_decides` and `comparable_decides`, checked by the Lean kernel | proved |
| The three flags given to decision 30 describe the kernels the engine actually ran | the caller's; to be tested by the differential tests in [docs/VERIFIABLE_INFERENCE_PLAN.md](docs/VERIFIABLE_INFERENCE_PLAN.md) | not proved |
| `KN-31`: a receipt is `confirmed` exactly when its model is ready, the run is comparable and the output replays to the label claimed; `mismatch` exactly when only the output differs; `not comparable` otherwise | `receipt_confirmed_decides`, `receipt_mismatch_decides` and `receipt_not_comparable_decides`, with `sameLabel` as decision 5 and `comparable` as decision 30 prove them, checked by the Lean kernel | proved |
| The model-ready flag given to decision 31 is decision 28's answer for the receipt's own model root, and the replay ran on the receipt's input | the caller's | not proved |
| `KN-32`: a probe is an `exact replay` exactly when layer, position and component are each below their counts and the run is comparable, an `approximate replay` exactly when they are and it is not, and `out of range` otherwise | `probe_exact_decides`, `probe_approximate_decides` and `probe_out_of_range_decides`, checked by the Lean kernel | proved |
| That a value recomputed for an exact replay equals the value that produced the answer | follows from the class only if decision 30's flags are true of the run | not proved |
| `KN-33`: a prefix is shared exactly when decision 27 reuses it and it was computed on this device or in a comparable run | `shareable_decides`, checked by the Lean kernel | proved |
| `KN-34`: every product of a 32-wide block lies in exactly one of the positive and negative parts, by the signs of its weight code and activation; a block is `exact` exactly when the claim plus the negative part equals the positive part plus 32768; at most one claim is exact | `parts_cover_every_product` (by induction over the places), the four sign theorems `agrees_when_both_positive`, `agrees_when_both_negative`, `differs_when_weight_positive` and `differs_when_weight_negative`, `weight_size_about_eight`, `activation_size_about_zero`, `accepts_decides` and `accepts_one_claim`, checked by the Lean kernel | proved |
| `KN-35`: a content label is admitted as an address exactly when its first seven bytes are `blake3:`; `sha256:` is refused | `address_blake3_decides` and `address_sha256_refused`, checked by the Lean kernel | proved |
| `KN-36`: a run may start exactly when its products and every other stage are exact, which is decision 30's class `everywhere` whatever the order of reductions; a run with a floating-point stage is refused | `integer_run_decides`, `integer_run_is_everywhere` and `integer_run_refuses_fixed_order_floats`, checked by the Lean kernel | proved |
| `KN-45`: the engine session, a pure step over open, admit, prompt chunk, generate and close: a page not named by BLAKE3 fails it, it is ready only when the page admitted is the manifest's last, prompt chunks compose, a generate request with a floating-point stage fails it, the position stays within the capacity, and an ended session stays ended | `session_ends_stay_ended`, `session_refuses_other_hashes`, `session_ready_only_with_every_page`, `session_refuses_floats`, `session_prompt_chunks_compose` and `session_position_within_capacity`, checked by the hosted LexLean verifier; not yet by the pinned SDK image | proved (hosted) |
| The weight codes and activations decision 34 sums are the request's bytes 1 to 48 in the engine's q4 order (two codes to a byte, low half first, as `uor-matmul`'s `Packed` reads them; activations one signed byte each) | `byteFromEnd` and `weightCode`, and the offsets and bases in `blockCheckAnswer`, pinned by `kernel_kn_34`; a model of those definitions agreed with a byte-order reference on 40,000 random blocks on 2026-11-17, outside the gate | tested, not proved |
| That `weightSize` times `activationSize`, signed by `signsAgree`, is the product of the signed code minus 8 and the signed activation | argued from the four sign theorems and the two size theorems; the product itself is not stated, because the proof language's arithmetic is linear | not proved |
| The block the page samples is a block of the tensor the run used, and the claim is the group sum that run computed | the caller's | not proved |
| `KN-37`: one group's sum times its IEEE half is `exact` exactly when the claimed magnitude is the sum's magnitude times `halfUnits` and the claimed sign is 1 just when that product is nonzero and the signs differ; a zero product is never negative; a half with exponent field 31 is `not finite` | `scale_accepts_size`, `scale_accepts_sign`, `scale_accepts_the_claim`, `zero_is_not_negative`, `sum_size_about_offset`, and `halfOf` by `half_subnormal_is_mantissa`, `half_normal_is_scaled` and `two_to_doubles`; checked by the Lean kernel | proved |
| That `halfUnits` is the half's value in units of 2^-24 | argued from the two `halfOf` theorems and `two_to_doubles`: a subnormal is its fraction, a normal half of field `e` is `(1024 + fraction) * 2^(e-1)`; the value itself is not stated, because the proof language's arithmetic is linear | not proved |
| Decision 37 answers the crate's claims in the compiled guest | `replay.mjs` in the `app` recipe, over the decision-37 claims in `crates/conformance/tests/kernel/claims.json` | tested on CI |
| `KN-42`: a claimed binary32 is `exact` exactly when it is `roundResult` of the sign, magnitude and exponent: zero for a zero magnitude, infinity when the leading bit is past 2^127, otherwise the kept bits plus one exactly when the guard bit is set and the sticky bit or the last kept bit is | `round_accepts_decides`, `round_zero_is_zero`, `round_past_largest_is_infinite`, `round_up_at_most_one`, `round_no_guard_no_change`, `round_shift_past_word`; checked by the Lean kernel | proved |
| That `roundResult` is the binary32 nearest the value | argued from the definition's steps, which follow IEEE 754's; nearness itself is not stated, because the proof language's arithmetic is linear | not proved |
| Decision 42 answers the crate's claims in the compiled guest | `replay.mjs` in the `app` recipe, over the decision-42 claims in `crates/conformance/tests/kernel/claims.json` | tested on CI |
| Decisions 29 to 34 answer their examples in the compiled guest | none are replayed, for the reason given for decisions 18 to 28 | not tested |
| `KN-38`: a stored q4 byte, half, bfloat16 or binary32 is read into integer fields that encode back to exactly the stored bits; every field lies within its width; fields within their widths read back unchanged | `codec_round_trip`, `codec_fields_fit` and `codec_reads_back_q4`, `_half`, `_bfloat16`, `_binary32`, checked by the Lean kernel | proved |
| The fields of a half, bfloat16 or binary32 stand for the IEEE 754 value `(-1)^sign * fraction-with-leading-bit * 2^(exponent - bias)` | the reading in `crates/exact` (`Dyadic`), tested against Rust's own conversions in `EN-34` on the cases drawn | tested, not proved |
| `KN-39`: a narrowing is accepted exactly when its step is known, its input is 1 to 16 bits (a table of at most 65,536 entries), its output 1 to 32 bits, and its certified error within its declared bound | `narrow_accepted_decides`, checked by the Lean kernel | proved |
| The certified error of a narrowing's table is its true worst error | the table generator's, to be checked when each table is built; no table exists yet | not proved |
| `KN-40`: one link of a row's exact recombination is `exact` exactly when its positive parts plus the claim's negative part equal its negative parts plus the claim's positive part; a part counts on one side only, by its sign; two exact links compose into one, so a chain confirms the plain sum of every term in it; at most one claim is exact | `rows_accepts_decides`, `rows_part_splits`, `rows_part_one_side`, `rows_links_compose` and `rows_one_claim`, checked by the Lean kernel | proved |
| A chain of any length confirms the plain sum | follows from `rows_links_compose` by induction on the number of links; no theorem states the induction, because the request holds one link | argued, not proved |
| `KN-40`: the next partial is a function, `rowsNextSign` and `rowsNextSize` of the positive and negative totals; the check accepts it, and any claim it accepts has that size, and that sign unless the size is zero | `rows_next_accepted`, `rows_accepts_only_next_positive` and `rows_accepts_only_next_negative`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| A term that fits has its shift at most 58, and its sum and mantissa below `2^24` | `rows_term_fits_bounds`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| No number decision 40 computes passes 64 bits: partials below `2^62`, and each term's `sum * mantissa * 2^shift` below `2^58` | the guard in `rowsAnswer`, pinned by `kernel_kn_40` and exercised at its edges in the compiled guest; the product bound is not stated as a theorem, because it is not linear | tested, not proved |
| The compiled guest answers decision 40 as its definition does | 52,920 links from 3,640 real Qwen3-0.6B rows (half planted one too high) and 12 crafted edge requests, each answered as expected by the guest of a holo built from this branch; a measurement across this branch and `crates/exact`, not yet a test in the suite | measured |
| `KN-41`: an input divided down to its code is within half a step of it (`2v <= 2p + s` for the lower code, `s + 2p <= 2v` for the upper), a tie gives the even code, and one input with one quotient witness has at most one code; an input multiplied up is its code exactly, within 32767 | `rq_down_witness`, `rq_down_half_step`, `rq_down_tie_even`, `rq_down_one_code` and `rq_up_exact`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| `KN-41`: a group's power is accepted only when its largest input is within 32767 at it and past it at the next power up | `rq_largest_down_decides` and `rq_largest_up_decides`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| The quotient witness is the input's quotient by the step | the guest checks `q * step <= value < q * step + step`, so a wrong witness is refused; that this pins `q` uniquely is the division algorithm, not stated as a theorem, because the product is not linear | argued, not proved |
| `rqDownCode` (the code as a function of input and step) gives the code the check accepts | not stated: it needs quotient and remainder facts for a variable divisor, outside `linear_arithmetic`; EN-39 (branch `exact-codecs`) checks, for every input it writes, that the code is `rqDownCode`'s, passes the transcribed `rqElementOk`, and is `quantise_with`'s | tested, not proved |
| No number decision 41 computes passes 64 bits | the guards in `rqElementOk` and `rqLargest` (mantissa below `2^24`, quotient at most 32767, a down shift at most 39 and an up shift at most 15, each inside an `if` so no product is formed before them), pinned by `kernel_kn_41` | tested, not proved |
| `KN-43`: a claimed root is `exact` exactly when its square is at most the value and the value is below the square of the root plus one, which defines the floor square root | `root_ok_decides`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| At most one root is accepted for a value | squares grow with the root, so two accepted roots are equal; this needs products of two variables, outside `linear_arithmetic`, and `simplify` may not unfold the runtime's product, so it is argued, not stated as a theorem | argued, not proved |
| No number decision 43 computes passes 64 bits | the guard in `rootAnswer`: a value below `2^62` and a root below `2^31`, checked before any square is formed, so `(root + 1)^2` is at most `2^62` | pinned by `kernel_kn_43` |
| `KN-44`: a code is `exact` for a value and a divisor exactly when KN-41's `rqDownOk` accepts it with the divisor as the step and `(2 half + odd) * divisor` as the product, so KN-41's four rounding theorems hold for it | `divide_is_requant_rule`, checked by Lean on the hosted verifier | proved (hosted verifier) |
| The witness `2 half + odd` is the floor quotient | `rqDownOk` checks `q * d <= value < q * d + d`; that this pins `q` is the division algorithm, as for KN-41, not stated because the product is of two variables | argued, not proved |
| No number decision 44 computes passes 64 bits | the guard in `divideAnswer`: a value below `2^62`, a divisor from 1 to below `2^31`, half below `2^30` and odd at most 1, checked before the product is formed, so `q * d` is below `2^62` | pinned by `kernel_kn_44` |
| Decision 41 answers as its definition does in the compiled guest | not run: building a holo on this machine has moved to `prism-code build --remote github`, which pushes | not tested |
| Decisions 38 and 39 answer their examples in the compiled guest | none are replayed, for the reason given for decisions 18 to 28 | not tested |
| Decisions 18 to 28 answer their examples in the compiled guest | none are replayed: the application's example list is at the nesting depth the proof language reads (108 examples), as it was for decisions 12 to 17. The answers were worked by a second implementation outside the repository and are not gate evidence | not tested |
| The verdict for each decision is the one its proved function gives, read from the request offsets the wire format fixes | the `kernel_kn_` tests in `crates/conformance/tests/kernel.rs` | tested (string-matching, not a theorem) |
| The compiled guest gives the stated answer for each example | the 108 examples in `app/src/Kernel.lex.tex`, replayed by `prismpm verify` | tested |

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

## KN-01, journey

Eight states: idle, preflight, download, compile, materialize, ready,
generating and failed. Ten events: start, preflight passed, download done,
compile done, labels verified, send, turn done, cancel turn, fail and reset. A
request gives the current state and an event; the answer is the next state, or
`refused`, which changes nothing. A model the preflight rejects is a fail;
choosing another model is a reset.

What is proved is one step at a time. Each of preflight, download, compile,
materialize and generating is entered from exactly one state by exactly one
event. Ready is entered only from materialize when the labels are verified, or
back from generating when a turn ends or is cancelled. Failed is entered only
by a fail while work is under way. Idle is entered only by reset. From failed,
every event resets or is refused.

### The claim about whole journeys, and why it is not a theorem

The claim that matters to a user is about a sequence: the application is never
ready unless preflight passed, the download finished, the compile finished and
every label verified, in that order.

It follows from the theorems. Take any run of accepted steps from idle that
ends in ready, and look at the last step that entered ready from a state other
than generating. By `enters_ready` it came from materialize by labels verified.
The step that entered materialize came, by `enters_materialize`, from compile
by compile done; that one, by `enters_compile`, from download by download
done; that one, by `enters_download`, from preflight by preflight passed; and
that one, by `enters_preflight`, from idle by start. Between them the state did
not change, since a refused step changes nothing. So the five events occurred,
in that order.

That argument is an induction over the run, with a hypothesis that is an
implication, and language 1.2 has no proof form that can use one. A check of a
proposal, as in `KN-07` and `KN-03`, does not help here: the kernel is given
one step, not the history. So the argument stands on paper, above, and the
theorems it rests on are checked.

### The proof gate: `lexlean verify src/Proofs/Journey.lex.tex`

| Planted defect | Result |
| --- | --- |
| `nextState`, in the executable module, lets materialize be entered from download, skipping compile | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Journey (error): omega could not prove the goal`, at `enters_materialize` |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| The download theorem lets download be entered from idle | rejected: `KN-01's enters_download changed: state 2 must be entered exactly from state 1 by event 2` |
| The materialize theorem weakened from `<->` to `->` | rejected: `KN-01's enters_materialize changed: state 4 must be entered exactly from state 3 by event 4` |
| `readyBy` admits ready straight from compile | rejected: `readyBy must be: from materialize by labels verified, or from generating by turn done or cancel` |
| The ready theorem is about `readyBy` being false | rejected: `ready must be entered exactly when readyBy holds` |
| The failed theorem no longer requires a working state | rejected: `failed must be entered exactly by a fail event from a working state` |
| The idle theorem lets idle be entered by start | rejected: `idle must be entered exactly by reset` |
| From failed, any next state at all is allowed | rejected: `from failed every event must reset or be refused` |
| Ready counts as a state that can fail | rejected: `working must be preflight to materialize, and generating` |
| The ready state reads `generating` | rejected: `state 5 must read ready` |
| A state code that does not exist reaches the decision | rejected: `the next state must be nextState's, from the state and event bytes at offsets 1 and 2` |
| Decision 1 routed to no answer | rejected: `decision 1 must be routed to the journey answer` |
| The shortcut example lets labels verified from idle reach ready | rejected: `no example answers refused for event 5 in state 0` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-01 is registered
and not built`.

### The examples

Eighteen. Eight walk the chain: idle to preflight, download, compile,
materialize, ready, generating, and back to ready by turn done and by cancel.
Five are refused: labels verified from idle and from the middle of a download,
send before anything is loaded, a fail in ready, and start from failed. One
fails while downloading, and one leaves failed by reset. Three are invalid: a
state that does not exist, an event that does not exist, and one byte short.

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

Nine. Two admit: the empty buffer under the BLAKE3 hash of the empty input,
and the three bytes `abc` under theirs. Six refuse: one bit flipped in each of
the four label words, the length off by one, and the length differing only in
its top byte. One is invalid: one byte short.

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

Nine. Four accept: the first tensor after a 128-byte header filling the file,
both bounds tight, an empty tensor at the end of the file, and the largest
range the fields can name. Three reject: a start one byte inside the previous
tensor, an end one byte past the file, and an offset plus length that exceeds
64 bits. Two are invalid: one byte short, and decision 6 sent with decision
5's length.

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

Ten. Seven answer a bucket: the first bucket from nothing, doubling that
covers the need, a smaller need that does not shrink it, doubling capped at
the maximum, zero, and two at the top of the 64-bit field. One answers
`context exceeded`, one `bucket exceeds maximum`. One is invalid: one byte
short.

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

Seven. Five answer a budget: all the room when no request is made, a request
that fits, a request cut to the room, one token of room, and the widest room
the field can name. One answers `context full`, at exactly full. One is
invalid: one byte short.

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

Ten. One supported, one derived. One names a missing key, one an unsupported
feature, one a manifest that does not fit, and two more show the order when
several things are wrong. Three are invalid: each flag byte set to 2, and one
byte short.

## KN-10, operator preflight

Four numbers come from the importer: the opset the graph declares, the lowest
and highest opsets the importer implements, and how many operators in the graph
it does not build. `operatorCode` gives one of five codes and `operatorText` its
text. A graph is importable when its opset lies within the window and no
operator is unimplemented; the opset is answered before the operators, so a
graph the importer could not read at all is not reported as a missing operator.

The window is the importer's own, so a window whose lowest opset is above its
highest is not a graph that was refused: it is a question that was never asked,
and it answers `invalid request`. `reversed_window_is_invalid` proves that is
exactly when the window runs backwards.

`window_monotone` is about the importer rather than the graph: widening the
window, by implementing an older or a newer opset, never makes an importable
graph unimportable. It is stated over `Importable`, not over the code, because
the code's verdict is already proved to be exactly that proposition.

Which operators an importer builds is not the kernel's to know, and no importer
exists in this repository yet. The kernel is told how many lie outside, and
answers.

### The proof gate: `lexlean verify src/Proofs/Operators.lex.tex`

| Planted defect | Result |
| --- | --- |
| `operatorCode`, in the executable module, calls a graph with one unimplemented operator importable | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Operators (error): omega could not prove the goal` |
| `operatorCode` excludes the highest opset in the window | rejected: the same |
| none | `verified`, with every proof module's axiom policy exact |

A fourth proof was written and withdrawn. `window_monotone` was first stated
over `operatorCode` rather than over `Importable`, and `linear_arithmetic` could
not prove it: the hypothesis that the code is the importable one has to be taken
apart case by case before the arithmetic begins, and the solver reported the
constraints it had left, among them an unimplemented operator it could not rule
out. Stated over the proposition, with the code's verdict proved equal to it, it
is arithmetic and it is proved.

### The pinned test

| Planted defect | Result |
| --- | --- |
| `Importable` drops the condition that every operator is implemented | rejected: `Importable must require the opset to be within the window and no unimplemented operator` |
| `Importable` excludes the lowest opset in the window | rejected: the same |
| `window_monotone` states narrowing the window, not widening it | rejected: `KN-10's window_monotone changed: widening the window must keep an importable graph importable` |
| A window that merely touches is called backwards | rejected: `KN-10's reversed_window_is_invalid changed: a window that runs backwards must be the invalid code` |
| A theorem quantifies over two of the four numbers | rejected: `importable_decides quantifies over every number and rests on Lean's three standard axioms` |
| The two opset refusals read each other's text | rejected: `verdict 1 must read opset above maximum` |
| The declared opset is read one byte late | rejected: `the verdict must be operatorCode's, from the four numbers at offsets 1, 9, 17 and 25` |
| The request is accepted one byte short | rejected: the same |
| Decision 10 routed to no answer | rejected: `decision 10 must be routed to the operator answer` |
| The lowest opset in the window is not an example | rejected: `opset 7 is an edge of the window in the examples and must answer importable` |
| No example asks about a window that runs backwards | rejected: `no example asks about a window that runs backwards` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-10 is registered
and not built`.

### The examples

Eight. One importable in the middle of the window and one at each edge, one
opset below the window, one above it, one with an operator the importer does not
build, one window that runs backwards, and one byte short.

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

Nine. Three answer `int8`, three `int4`, two `int8 paged`, on each side of
both boundaries in 4 GiB and at the top of the 64-bit field. One is invalid:
one byte short.

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

Nine. Six answer a size: rounded up to a multiple, exact for a step of 0, the
next multiple equal to the upper bound, the next multiple past it and so
capped, and two at the top of the 64-bit field. One answers `below minimum`,
one `above maximum`. One is invalid: one byte short.

## KN-07, context trimming

The kernel does not compute the trim. The engine proposes one, as five totals:
the context, the fixed part that is never dropped (the system prompt, the newest
user message and the reserve), the total of every question-and-answer pair, the
total of the pairs it dropped, and the size of the newest pair among those. The
kernel accepts the proposal exactly when it is right: nothing dropped that was
not there, what remains fits, and, when anything was dropped, putting the newest
dropped pair back would pass the context. That last condition is what "no fewer
pairs would have been enough" means when whole pairs are dropped oldest first.

It is a check and not a computation because the statement over a list needs an
induction whose hypothesis is an implication, and language 1.2 has no proof form
that can use one. The checked form is proved completely.

The decided functions never add: the room is `context - fixed`, so nothing can
pass 64 bits. A fixed part larger than the context answers `too long`.

### The proof gate: `lexlean verify src/Proofs/Trim.lex.tex`

| Planted defect | Result |
| --- | --- |
| `trimFits`, in the executable module, no longer checks that the fixed part fits the context | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Trim (error): omega could not prove the goal`, for both theorems, with the counterexample `fixed - context >= 1` |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| `Trimmed` counts an exact fit as too big to restore | rejected: `Trimmed must require that what remains fits and that restoring the newest dropped pair would not` |
| The trim theorem weakened from `<->` to `->` | rejected: `KN-07's statement changed: trimmed must be true exactly when Trimmed holds` |
| `trimFits` forms a sum, which overflows 64 bits | rejected: `trimFits must compare what remains with the room left, without a sum` |
| `trimMinimal` no longer requires the newest dropped pair to be one of those dropped | rejected: `trimMinimal must hold when nothing was dropped or the newest dropped pair exceeds the room left` |
| `trimmed` drops the fits condition | rejected: `trimmed must be all three conditions` |
| Two refusals swapped | rejected: `each refusal must name the condition that failed` |
| Decision 7 routed to no answer | rejected: `decision 7 must be routed to the trim answer` |
| The exact-fit example expects `trim` | rejected: `no example answers drops too many for 000000000000044c dropped with a newest dropped pair of 0000000000000030` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-07 is registered
and not built`.

### The examples

Twelve. Seven are accepted: nothing dropped, a dropped pair far too big to
restore, one a single token too big, what remains filling the context exactly,
the fixed part alone filling it, every pair dropped, and the top of the 64-bit
field. Four are refused, each by name: a dropped pair that would exactly fit
(`drops too many`), what remains one token over (`does not fit`), a fixed part
one token over (`too long`), and more dropped than there was
(`dropped exceeds total`). One is invalid: one byte short.

## KN-03, memory plan

The kernel does not compute the plan. The engine proposes one: how many layers
in a stage and how many stages. The kernel accepts it exactly when the context
and one stage fit the budget, one more layer would not fit unless the stage is
already the whole model, and the stages cover every layer with none empty.

It is a check and not a computation because layers for each stage is a division
by a variable, which the arithmetic prover does not handle. In the checked form
each product appears the same way in the decided function and in the statement,
and the proof is complete.

The layer count and both stage numbers are bounded by 65536 and a layer by 2^40
bytes before anything is multiplied, so no product can pass 64 bits. A model
whose single layer does not fit beside the context answers `cannot run`.

### The proof gate: `lexlean verify src/Proofs/Plan.lex.tex`

| Planted defect | Result |
| --- | --- |
| `stagesCover`, in the executable module, no longer requires at least one stage and none empty | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Plan (error): omega could not prove the goal`, with a counterexample in which the stages reach every layer but the last stage is empty |
| none | `verified`, with every proof module's axiom policy exact |

### The pinned test

| Planted defect | Result |
| --- | --- |
| `Planned` drops the largest-stage condition | rejected: `Planned must require one more layer not fitting, unless the stage is the whole model` |
| The plan theorem weakened from `<->` to `->` | rejected: `KN-03's statement changed: planned must be true exactly when Planned holds` |
| `planBounded` lets a layer of any size through, so a product can overflow | rejected: `planBounded must bound every factor before a product is formed` |
| `stageLargest` tests the stage itself, not one more layer | rejected: `stageLargest must hold when the stage is the whole model or one more layer would not fit` |
| `planned` drops the coverage condition | rejected: `planned must be all four conditions` |
| The plan is reached without checking that one layer fits | rejected: `the verdict must be planned's, reached only within the bounds and when one layer fits` |
| Decision 3 routed to no answer | rejected: `decision 3 must be routed to the plan answer` |
| The too-small example expects `plan` | rejected: `no example answers stage too small for a stage of 0000000000000010 layers in 0000000000000002 stages` |
| none, before and after | passed |

Before the model existed, the scaffolded test failed with `KN-03 is registered
and not built`.

### The examples

Twelve. Four are accepted: the whole model in one stage, two stages of 17
layers for a model of 30, one layer at a time, and the largest numbers the
bounds admit. Five are refused, each by name: a stage one layer too small, one
layer too big, one stage that leaves layers out, a third stage that would be
empty, and a model whose layer does not fit beside the context. Three are
invalid: more layers than the bound, a layer larger than the bound, and one byte
short.

## KN-12, backend

Three numbers from the page: whether a WebGPU adapter is present, the adapter's
largest storage-buffer binding, and the model's largest buffer as it will be
bound. `gpuUsable` is true exactly when the adapter is present and the buffer
fits. Whether the model fits memory at all is the question decisions 3 and 4
already answer, so this decision does not ask it again.

### The proof gate: `lexlean verify src/Proofs/Backend.lex.tex`

| Planted defect | Result |
| --- | --- |
| `gpuUsable` requires the largest buffer to be strictly below the limit | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Backend (error): omega could not prove the goal` |
| none | `verified`, with every proof module's axiom policy exact |

## KN-13, block length

The format and the shape give the length a decompressed weight block must have.
The kernel answers that length; the caller compares it with what it holds, so
the comparison has one home. Hash admission of the stored, compressed bytes is
decision 5's.

### The pinned test: `kernel_kn_13`

| Planted defect | Result |
| --- | --- |
| the ternary-per-128 format is 32 bytes per group instead of 36 | rejected: `kernel_kn_13` panics at `bytes per group changed` |
| none | passes |

The two tables are pinned by the test rather than proved, because the length
formula multiplies two unknowns and the proof forms here decide linear
arithmetic only.

## KN-14, greedy pick

The engine picks a token on the GPU every step and cannot ask the kernel each
time, so the kernel is the reference it is tested against. The engine claims an
index and a score; `standing` says, of one position, that before the index the
score is lower, at it the score is the claimed one, and after it none is higher.
`holdsFrom` checks every position, and `holds_decides` proves by induction that
it is true exactly when that holds at each one.

### The proof gate: `lexlean verify src/Proofs/Pick.lex.tex`

| Planted defect | Result |
| --- | --- |
| an earlier score equal to the claimed one is allowed, so a later tie would pass as the first maximum | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Pick (error): omega could not prove the goal` |
| none | `verified` |

## KN-15, penalty window

The window size is part of the request, so the boundary can be tested with a
short history; the engine uses 64. The window starts that many positions before
the end, or at the start of a shorter history.

### The proof gate: `lexlean verify src/Proofs/Penalty.lex.tex`

| Planted defect | Result |
| --- | --- |
| the position at the window's start is treated as outside it | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Penalty (error): omega could not prove the goal` |
| none | `verified` |

## KN-16, speculative commit

k drafts and k+1 verified picks. The proposed count is checked against every
draft position; the last verified pick is always committed and needs no check.

### The proof gate: `lexlean verify src/Proofs/Speculate.lex.tex`

| Planted defect | Result |
| --- | --- |
| the draft at the last committed position must equal its pick, rather than differ | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Speculate (error): omega could not prove the goal` |
| none | `verified` |

## KN-17, quant codes

Each format's unpacking is a division and a remainder by a power of its base,
with the power read from a table of literals, so the solver sees only division
by constants. A t2 word's sixteen codes are proved as two halves of eight, each
with the other half arbitrary: as one theorem of sixteen the case split did not
complete. The q3f round trip includes the spare bits above the ten fields, and
`spare_round_trip` shows the two codes kept there come back from all three words.

### The proof gate: `lexlean verify src/Proofs/Quant.lex.tex`

| Planted defect | Result |
| --- | --- |
| `nibbleAt` takes the remainder by 15 instead of 16 | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Quant (error): omega could not prove the goal` |
| none | `verified` |

## Decisions 12 to 17 carry no examples yet

The examples the holo replays sit in one list that stops building above 108
entries (see the next section). Decisions 12 to 17 add none until that limit is
lifted, so their answers are tested here by the pinned tests and proved by the
theorems, and not yet replayed in the compiled guest. Each will gain one example
per answer and one malformed request when the list can hold them.

## The limit on examples

A holo can carry about 120 examples. The SDK requires them to be one closed
list (`PP2001: application acceptanceVectors is not a closed list` refuses any
other shape), and LexLean reads a semantic module with a 128-level nesting
limit, which a closed list of 133 exceeded (`LLT4001: recursion limit
exceeded`). A balanced join of shorter lists, which the LexLean specification
uses for long lists in its own elaborations, is what the SDK refuses.

The kernel therefore carries 108: for each decision every verdict and refusal,
each boundary from both sides, the top of the 64-bit field, and one malformed
length; and three that name no decision. Examples removed to fit were second
instances of a verdict already shown and the one-byte-long malformed lengths.

Decision 10 arrived into a full list. At 116 the holo did not build: the SDK's
snapshot reads the application record as canonical JSON, whose reader stops at
128 levels of nesting, and a closed list of 116 examples carries the answer past
it (`recursion limit exceeded`, in `snapshot.rs`, after Lean had already checked
every proof). The ceiling is therefore the nesting of the list, not any limit on
examples as such, and it sits between 108, which builds, and 116, which does
not.

Eight examples made room for decision 10's eight, chosen as second instances of
an answer their decision already gives:

| Removed | Why it was a second instance |
| --- | --- |
| journey, state 0 event 0 | two other requests already answer `invalid request`, one of them malformed |
| journey, generating and cancel turn | `enters_ready` is the proof; turn done already shows ready being entered |
| preflight, a manifest flag of 2 | another flag out of range already answers `invalid request` |
| preflight, two unsupported features | one unsupported feature already answers the same |
| plan, a second accepted plan | the geometries that remain cover the same verdict |
| plan, a second malformed plan | the bounds that remain each have their own example |
| trim, an accepted trim away from any boundary | the boundary pairs already show the same verdict |
| trim, an accepted trim that drops every pair | the same |

Two attempts at this were rejected by the suite before the third passed: the
plan the `KN-03` test reads for its stage split, and the journey refusal the
`KN-01` test reads for a label verified during a download. Which examples no
suite reads is not left to guesswork; `app/src/Kernel.lex.tex` was searched one
example at a time for the ones whose removal keeps every suite green, and the
removals were chosen from those. Examples removed to fit were second
instances of a verdict already shown and the one-byte-long malformed lengths.

## KN-18 to KN-28, importing a model from its own files

Milestone 2's decisions: what an ONNX export may contain (18), which model
families run (19), how long a 4-bit weight's parts are (20), which precision
and device (21, 22), which bytes of which file may be read (23), what a
repository's files are called (24), when generation stops (25), what leaves
the browser's store (26), when a computed prefix may be reused (27) and when a
model's manifest is satisfied (28). None of them names a hash: a label is 32
bytes, whichever function made it.

### The pinned tests: `kernel_kn_18` to `kernel_kn_28`

| Planted defect | Result |
| --- | --- |
| `Add`'s first word is one too high in the decomposed list | rejected: `kernel_kn_18` panics at `opnameDecomposed changed` |
| decision 18 is routed to another decision's answer | rejected: `kernel_kn_18` panics at `decision 18 must be routed to Kernel.Opnames::opnameAnswer` |
| a head size of one is accepted | rejected: `kernel_kn_19` panics at `familyShaped changed` |
| a block of 4-bit codes is counted at one byte for each weight | rejected: `kernel_kn_20` panics at `nbitsPerBlock changed` |
| a precision exactly the budget's size does not fit | rejected: `kernel_kn_21` panics at `precisionFits changed` |
| a model exactly at the CPU ceiling is refused | rejected: `kernel_kn_22` panics at `deviceCode changed` |
| the last declared data file is undeclared | rejected: `kernel_kn_23` panics at `chunkDeclared changed` |
| data file names count from the part, not the part less one | rejected: `kernel_kn_24` panics at `sourceName changed` |
| generation runs one token past the budget | rejected: `kernel_kn_25` panics at `stopSpent changed` |
| a pinned block is evicted | rejected: `kernel_kn_26` panics at `evictCode changed` |
| `evict_decides` no longer says the block is not pinned | rejected: `kernel_kn_26` panics at `evict_decides's statement changed` |
| a prefix as long as the prompt is reused, leaving no position to run | rejected: `kernel_kn_27` panics at `reuseShorter changed` |
| a model is ready with tensors missing | rejected: `kernel_kn_28` panics at `manifestComplete changed` |
| a confirmed receipt and a mismatch swap verdicts | rejected: `kernel_kn_31` panics at `receiptCode changed` |
| a prefix computed on another device is shared whatever the run's class | rejected: `kernel_kn_33` panics at `shareable changed` |
| none | all 33 `kernel_kn_` tests pass |

### The proof gate: `lexlean verify src/Proofs/*.lex.tex`

| Planted defect | Result |
| --- | --- |
| a pinned block is evicted | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Evict (error): omega could not prove the goal` |
| a precision exactly the budget's size does not fit | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Precision (error): omega could not prove the goal` |
| none | `verified 58 modules`, with every proof module's axiom policy exact |
| an exact sum is accepted with one bit more than the accumulator holds (decision 29) | Lean rejects it: `error[LLV7002]: Lean rejected HologramAi.Proofs.Exact (error): omega could not prove the goal`. **`lexlean verify` still exits 0**, so the recipe's `set -e` does not stop on that step; the gate still stops, because `prismpm verify` must then print an attestation for `test -s` to pass. Reported as UOR-Foundation/lexlean#562 and UOR-Foundation/prismpm#714 |
| none, with decisions 29 to 33 | `verified 68 modules`, zero errors |

The other nine modules' proofs were each checked by the Lean kernel as they
were written; a defect was planted through Lean in two of the eleven, and
through the pinned test in all of them.

## KN-34, block check

Decision 34 recomputes one 32-wide group sum of signed 4-bit weight codes times
int8 activations, the unit the engine's q4 product sums exactly, and compares
it with a claimed sum. The request is 53 bytes: the tag, sixteen bytes of codes,
thirty-two activations and the claim at byte 49, offset by 32768 so it is never
negative. The sum lies between -32512 and 32768, so the claim fits four bytes.
The kernel reads the codes and activations as six 8-byte fields, so the page's
form names every byte of the request (`EN-14`), and takes each code and
activation from those fields by `byteFromEnd`. Each activation word is summed
on its own, with its weight word and the place its codes start at, because the
compiled guest holds a natural number in 64 bits: a first form that joined two
activation words into one 128-bit number verified in Lean and then failed the
guest's build with `literal out of range for u64`. The sum is split into a positive
and a negative part so the kernel works in natural numbers only.

The answer is `exact`, `differs` or `invalid request`. Its name in the guest is
`blockCheckAnswer` and its sign rule is `signsAgree`: `blockAnswer` and `agrees`
are decision 13's and decision 16's, and the guest holds every module's names in
one namespace, which `kernel_kn_34` caught.

### The pinned test: `kernel_kn_34`

| Planted defect | Result |
| --- | --- |
| the claim read from byte 48 | rejected: `blockCheckAnswer changed` |
| weight codes read high half first | rejected: `weightCode changed` |
| the claim offset by 32767 | rejected: `accepts changed` |
| the positive part of the last eight activations sums seven products | rejected: `blockCheckAnswer changed` |
| a byte taken modulo 255 | rejected: `byteFromEnd changed` |
| a 52-byte request answered | rejected: `blockCheckAnswer changed` |
| `accepts_decides` weakened from `<->` to `->` | rejected: `accepts_decides's statement changed` |
| decision 34 routed to decision 33's answer | rejected: `decision 34 must be routed to Kernel.Block::blockCheckAnswer` |
| `accepts_decides` declares `Classical.choice` | rejected: `accepts_decides must rest on [propext]` |
| none, before and after | passed |

### The proof gate: `lexlean verify src/Proofs/Block.lex.tex`

| Planted defect | Result |
| --- | --- |
| a positive weight code agrees with a negative activation (`signsAgree`) | rejected: `error[LLV7002]: Lean rejected HologramAi.Proofs.Block (error): omega could not prove the goal` |
| the negative part also counts agreeing products | rejected: the same, in `parts_cover_every_product` |
| the claim offset by 32767 | rejected: the same, in `accepts_decides` |
| a byte taken modulo 255 | not rejected: `verified 4 modules`. No theorem states how bytes become codes; `kernel_kn_34` rejects it |
| none | `verified 4 modules` |

Run on 2026-11-17 in the SDK image the gate pins
(`prismpm-sdk-candidate@sha256:cd3b47d7…`). In these runs `lexlean verify`
exited 1 on each rejection. Over every proof module, `lexlean verify
src/Proofs/*.lex.tex` gave `verified 70 modules`.

### The holo: `prismpm --project app verify`

On the machine these changes were made on (12 threads, 13 GB, WSL), this step
did not finish, for main as well as for this change: the leanchecker replay of
`HologramAi.Kernel` was stopped at the SDK's 300-second child limit (`LLV7003`,
exit 143), at main `dc1ba4f` and here, alone on the machine and at
`LEXLEAN_VERIFY_WIDTH=1`. An earlier form of this change failed sooner, in the
guest's build, which is how the 64-bit limit above was found. That the guest
builds and replays with decision 34 is left to the gate on CI.

## KN-35 and KN-36, BLAKE3 addresses and integer runs

Set by Ilya on 2026-11-17: BLAKE3 is the only hash Hologram AI names content
by, and its inference uses integers only. Decision 35 makes the first a verdict
the page asks before it admits any address: a label's first seven bytes,
`blake3:` or not. Decision 36 makes the second a verdict asked before a run
starts: the run's products and its other stages each exact or not, read
through decision 30's own class, so integer-only is decision 30's
`everywhere` and nothing is restated. Decision 30 still classifies a float
run with fixed-order reductions as `same hardware`; decision 36 refuses it.

Which formats have an exact integer codec is not decided here: those codecs
are being built, and a list written before them would claim what does not
run.

### The pinned tests: `kernel_kn_35` and `kernel_kn_36`

Each pins its theorems' statements and axioms, every definition the answer
reads, the answer's length and offsets, and the route; `kernel_kn_35` also
checks that its numbers are `blake3:` and `sha256:` read as fields of 4, 2
and 1 bytes: the page's form has fields of 1, 2, 4, 8 and 32 bytes only, and
a first form reading the seven bytes as one field failed `engine_en_14`.
Planted on 2026-11-17, each test run alone with `cargo test -p
repo-conformance --test kernel`:

| Planted | Result |
| --- | --- |
| decision 35 answering `refused` where it answers `not blake3` | rejected: `kernel_kn_35`, `addressAnswer changed` |
| tags 35 and 36 routed to each other's answer | rejected: `kernel_kn_36`, `decision 36 must be routed to Kernel.Integer::integerAnswer` |
| none | 7 passed |

### The proof gate: `lexlean verify src/Proofs/Address.lex.tex src/Proofs/Integer.lex.tex`

Run in the SDK image `prismpm.lock` pins
(`prismpm-sdk-candidate@sha256:cd3b47d7…`), WSL, 2026-11-17:

| Planted | Result |
| --- | --- |
| `addressBlake3` testing its first four bytes for `sha2` | rejected: Lean, `HologramAi.Proofs.Address`, omega could not prove the goal |
| `integerRun` reading decision 30's class with fixed-order reductions and admitting any class, `same hardware` among them (`classCode … 1 <=? 2`) | rejected: Lean, `HologramAi.Proofs.Integer`, omega could not prove the goal |
| none | `verified 7 modules`, attestation `f4c4c772a30775b09600d0de8efe94879b64058348c6acc0f98b188e03157993` |

`lexlean fmt --all --check` (76 modules canonical) and `lexlean lock
--check` passed. The holo was not built here: on this machine the
leanchecker replay of `HologramAi.Kernel` stops at the SDK's 300-second
limit, as recorded for KN-34, so that the guest builds and replays with
decisions 35 and 36 is left to the gate on CI. `site/og.png` was not redrawn
here (the pinned Pillow wheel does not match this machine's Python); the
deploy draws it.

## KN-37, scale check

Decision 37 checks one group's sum times its scale, the step after decision
34 in a q4f16 product. A half is an integer times a power of two, so the
product is exact in units of 2^-24, the smallest positive half: a subnormal
half is its 10-bit fraction, and a normal half of exponent field `e` is
`(1024 + fraction) * 2^(e-1)`, at most `2047 * 2^29`. The request is 16
bytes: the tag, the group sum plus 32768 in four bytes, the half in two, the
claim's sign in one and its magnitude in eight. The sum's magnitude is at most
32768, so the product is below 2^56 and fits the guest's 64-bit naturals.
The kernel works in natural numbers: the sum's magnitude and sign, the half's
magnitude and sign, and a product that is negative only when it is nonzero
and the two signs differ, so zero has one form.

The answer is `exact`, `differs`, `not finite` for a half whose exponent field
is 31, or `invalid request`.

The first form of `halfOf` was a 31-way chain of literal powers. Lean rejected
its theorems: `omega` treats the chain as one opaque number, and `simplify`
left `Nat.beq 0 0` unreduced. A second form, `if exponent = 0 then mantissa
else (1024 + mantissa) * twoTo (exponent - 1)`, failed one theorem on
`below + 1 - 1`. The third matches on the exponent and passed.

The axioms each theorem declares are the ones Lean observed: the hosted
prover refuses any other set (`LLV7005: violates its exact axiom policy`),
one theorem per run, so each was learned from its own run.

### The pinned test: `kernel_kn_35`

Run on 2026-11-18 in WSL with the pinned toolchain. Each defect was planted
alone and the file restored.

| Planted defect | Result |
| --- | --- |
| `halfOf` adds 1023 | rejected: `halfOf changed` |
| a zero product may be negative (`0 <=?`) | rejected: `scaledNegative changed` |
| a theorem declares `Classical.choice` it does not use | rejected: `must rest on` |
| the `not finite` claim dropped | rejected: no decision-37 claim is answered `not finite` |
| none | passed |

### The proof gate

Checked on the hosted LexLean prover (`prism-app` profile) on 2026-11-18,
with `Kernel.Scale`, `Kernel.Wire` and the parts of `Foundation.Bytes` it
reads. That is a diagnostic, not the locked SDK's verdict; `lexlean verify
src/Proofs/*.lex.tex` in the `app` recipe on CI is.
## KN-38 and KN-39, lossless reading and declared narrowing

The requirement: no floating point from the stored bytes to the token ids, and
lossless where lossless exists. Lossless is held to three parts:

- (a) Reading a stored weight or scale loses nothing: its integer fields encode
  back to exactly the stored bits. `KN-38` proves this over natural numbers for
  q4 bytes, IEEE halves, bfloat16 and binary32. It is also where a format with
  no exact codec will be refused by name as codecs are added.
- (b) Every matrix sum is exact and cannot overflow: `KN-29` (widths) and
  `KN-34` (one block re-checked).
- (c) A step that narrows a value is named, integer and bounded, and is never
  called lossless. Exp in softmax, square root in normalisation, sine and cosine
  in rotary encoding, SiLU, GELU and requantising activations have no exact
  result in finite integers, and float engines round them too. `KN-39` admits
  such a step only with a finite table domain, a declared output width, and a
  certified error within its declared bound. Each step's bound, and each
  model's quality against its float reference, are to be reported per model;
  none is yet, because no table exists.

Whether a run is integer at all is `KN-36`, and whether a label is a BLAKE3
address is `KN-35`, recorded above.

### The pinned tests: `kernel_kn_37` and `kernel_kn_38`

| Planted defect | Result |
| --- | --- |
| a narrowing accepted when its error exceeds its bound | rejected: `narrowAccepted changed` |
| a 17-bit table domain accepted | rejected: `narrowAccepted changed` |
| a half's exponent field taken as 31 wide | rejected: `codecMiddleSize changed` |
| a binary32 fraction taken as 22 bits | rejected: `codecLowSize changed` |
| decision 39 routed to decision 38's answer | rejected: `decision 39 must be routed to Kernel.Narrow::narrowAnswer` |
| a value one wider than its format answered | rejected: `codecAnswer changed` |
| `narrow_accepted_decides` declares fewer axioms | rejected: `narrow_accepted_decides must rest on [Classical.choice, Quot.sound, propext]` |
| none, before and after | passed |

### The proof gate: `lexlean verify src/Proofs/Narrow.lex.tex src/Proofs/Codec.lex.tex`

| Planted defect | Result |
| --- | --- |
| a narrowing accepted when its error exceeds its bound | rejected: `Lean rejected HologramAi.Proofs.Narrow (error): omega could not prove the goal` |
| a 17-bit table domain accepted | rejected: the same |
| a half's exponent field taken as 31 wide | rejected: `Lean rejected HologramAi.Proofs.Codec` |
| a binary32 fraction taken as 22 bits | rejected: the same |
| a value one wider than its format answered | not rejected: `verified 6 modules`. No theorem states which requests are valid; `kernel_kn_37` rejects it |
| none | `verified 6 modules` |

Run on 2026-11-17 in the pinned SDK image; `lexlean verify` exited 1 on each
rejection.

## KN-40, row recombination

A row of a product is the sum of its units' sums times their scales, each at its
own power of two. The guest checks one row as a chain of links, because a request
holds at most 81 bytes and a row has 32 to 280 units. A link is 79 bytes: the
exact partial so far (a sign byte and eight bytes), the claimed partial after it,
and up to six terms of a sign, a unit sum (four bytes), a scale mantissa (four
bytes) and a power of two (one byte) above the chain's lowest power. The page
carries the partial from one link to the next. A link whose partial reaches
`2^62`, or whose term's `sum * mantissa` reaches `2^(58 - shift)`, is `too wide`:
the guest's numbers are 64 bits, and the Lean model's are unbounded.

### The pinned test: `kernel_kn_40`

| Planted defect | Result |
| --- | --- |
| a link accepted with the claim on the wrong side | rejected: `rowsAccepts changed` |
| a part counted whatever its sign | rejected: `rowsPart changed` |
| a shift of 59 admitted | rejected: `rowsTermFits changed` |
| decision 40 routed to decision 39's answer | rejected: `decision 40 must be routed to Kernel.Rows::rowsAnswer` |
| the third term's sum read from its mantissa | rejected: `rowsAnswer changed` |
| none | passed |

### The proof gate: `lexlean verify src/Proofs/Rows.lex.tex`

| Planted defect | Result |
| --- | --- |
| a link accepted with the claim on the wrong side | rejected: `Lean rejected HologramAi.Proofs.Rows (error): omega could not prove the goal` |
| a part counted whatever its sign | rejected: the same |
| a shift of 59 admitted | not rejected in this first run: the bound was outside every theorem. Since `rows_term_fits_bounds`, rejected (below) |
| none | `verified 4 modules` |

### Added after the first run: the function and the bounds, on the hosted verifier

On 2026-11-18 this machine stopped running the SDK image, and Ilya chose the
hosted LexLean verifier (`mcp.uor.foundation`, profile `prism-app`) for proving.
Its results are the hosted verifier's diagnostic, not the locked SDK's verdict;
the locked SDK checks the same modules on CI.

- `Kernel.Rows` gained `rowsNextSign` and `rowsNextSize`: the partial a link
  computes, as a function a stateful guest can call.
- `Proofs.Rows` gained `rows_next_accepted`, `rows_accepts_only_next_positive`,
  `rows_accepts_only_next_negative` and `rows_term_fits_bounds`. The first
  attempt at the claim theorem took a signed claim through `rowsPart`, whose
  `if` stayed in a hypothesis `linear_arithmetic` does not split; it is stated
  once per sign instead.
- The whole of `Kernel.Rows` with `rowsAnswer`, and all nine theorems, with
  `Kernel.Wire` and `Foundation.Bytes`: `verified` on the hosted verifier.

| Planted defect | Hosted verifier |
| --- | --- |
| the next size computed the wrong way round | rejected: `Lean rejected ... Proofs.Rows (error): omega could not prove the goal`, in `rows_accepts_only_next_positive` |
| a shift of 59 admitted | rejected: the same, in `rows_term_fits_bounds` |

A chain of any length is still argued from `rows_links_compose`, not stated:
the request holds one link, and a theorem over a list of links has no request
it would describe.

### The compiled guest against the fast path

A holo built from this branch with `prismpm --project app build` (2026-11-18,
96 s, published `23a1c162…`), its guest run by the vendored `prism-code`
(`loadArchive`, then `dispatch`) under Node:

| Requests | Expected | Answered as expected |
| --- | --- | --- |
| 26,460 links of 3,640 rows of Qwen3-0.6B's 196 projections (every 97th row, sixteen-bit inputs per group of 32, 0.2% outliers), built by `crates/exact`'s `row_links` from the terms `q4_product_with` places | `exact` | 26,460 |
| the same links with the claimed partial one too high | `differs` | 26,460 |
| the largest sums the bounds admit, all-negative terms, cancellation, crossing zero, a zero term at shift 58 | `exact` (5), `differs` (1) | 6 |
| a term of exactly `2^58`, a sum of `2^24`, a partial of `2^62`, a shift of 59 | `too wide` | 4 |
| a sign of 2; a 78-byte request | `invalid request` | 2 |

No row of the 3,640 was too wide. The chain's last partial, rounded once,
equals the fast path's output for every row (asserted while the links were
built). This is a measurement across two branches; it becomes a test in the
suite where `crates/exact` and decision 40 are in one tree.

Run on 2026-11-18 in the pinned SDK image; `lexlean verify` exited 1 on each
rejection.

## KN-41, requantising inputs to sixteen bits

EN-37 measured that sixteen-bit codes with one power of two per group of 32
inputs bring products about 1,300 times closer to the float result than eight
bits per vector. Decision 41 checks that narrowing for up to five inputs of one
group in a 62-byte request: the group's largest input (mantissa, and whether its
power multiplies up or divides down, and by how many), then each input's
mantissa, direction and power, a quotient witness (half of it, two bytes, and
whether it is odd, one byte) and its code (two bytes). The answer is `exact`,
`differs`, `not the largest power`, or `invalid request`.

Nothing a proof reads holds an `if` or a remainder: the first form did, and two
of its theorems could not be proved, because `linear_arithmetic` splits an `if`
only in the goal and `prim remainder` brings one with it. The quotient comes
as `2 * half + odd`, and rounding is three exclusive linear cases on `2r`
against the step: above it the code is `q + 1`, below it `q`, on a tie
`q + odd`, which is even.

Proved on 2026-11-18 on the hosted LexLean verifier (`mcp.uor.foundation`,
profile `prism-app`), the hosted verifier's diagnostic and not the locked SDK's
verdict: the whole of `Kernel.Requant` with `rqAnswer`, and all seven theorems,
`verified`.

An all-zero group has no power: its largest input is 0, and no power makes 0
reach 32767. EN-39 found that `rqLargest` answered such a group `not the largest
power`. It now accepts a largest input of 0 first; every input of that group
is then 0, and `rqElementOk` accepts only code 0 for it. Re-verified the same
day on the hosted verifier, the whole of `Kernel.Requant` and `Proofs.Requant`,
`verified`.

### The pinned test: `kernel_kn_41`

| Planted defect | Result |
| --- | --- |
| a tie rounded up | rejected: `rqDownOk changed` |
| a down shift of 40 computed, past the 64-bit guard | rejected: `rqElementOk changed` |
| a quotient of 32768 admitted | rejected: `rqElementOk changed` |
| a power one too small accepted as the largest | rejected: `rqLargestUp changed` |
| decision 41 routed to decision 40's answer | rejected: `decision 41 must be routed to Kernel.Requant::rqAnswer` |
| an all-zero group refused again | rejected: `rqLargest changed` |
| none | passed |

### The proof gate, on the hosted verifier

| Planted defect | Result |
| --- | --- |
| an input just past half a step counted down to `q + 1` and the up case widened to `<=` | rejected: `Lean rejected ... Proofs.Requant (error): omega could not prove the goal`, in `rq_down_half_step` and `rq_down_one_code` |
| the largest-power check made `<=` | rejected: the same, in `rq_largest_down_decides` |

## KN-42, round check

Decision 42 checks the one rounding in a product's row: an exact value
`magnitude * 2^exponent`, negative or not, to the nearest binary32, ties to
even. It follows `hologram-ai-exact`'s `round_to_binary32` step by step: the
leading bit fixes the top power; past 2^127 the answer is infinity; the last
kept bit stands for 2^(top - 23), or 2^-149 below the normal range; the guard
bit below it and whether any lower bit is set decide one increment; and a
field at or past infinity's is infinity, so a carry past the largest finite
value is too. A zero magnitude is +0 whatever its sign, and a negative value
that rounds to zero keeps its sign bit, as the crate does. The request is 16
bytes: the tag, the sign, the magnitude in eight, the exponent plus 32768 in
two, and the claimed bits in four.

The compiled guest holds naturals in 64 bits, so no step forms a power of two
at or past 2^64: a shift by 64 or more is its own case (`roundShift`,
`roundBit` and `roundAny`), and the largest power formed is 2^63. A magnitude
is at most 2^64 - 1, so a row whose exact value needs more bits than that is
outside this decision; the crate's register is wider.

### The pinned test: `kernel_kn_38`

Run on 2026-11-18 in WSL with the pinned toolchain. Each defect was planted
alone and the file restored.

| Planted defect | Result |
| --- | --- |
| ties rounded away from zero (`roundUp` on the guard bit alone) | rejected: `roundUp changed` |
| the subnormal floor at 2^-148 | rejected: `roundUnit changed` |
| a theorem declares `Classical.choice` it does not use | rejected: `must rest on` |
| none | passed |

## KN-43, the integer square root

RMS and layer norms divide by a square root. On integers the floor square root
is exact, so decision 43 checks it as exact rather than admitting it as a
narrowing. A 13-byte request carries a value (8 bytes, below `2^62`) and a
claimed root (4 bytes, below `2^31`). The answer is `exact` when
`root^2 <= value < (root + 1)^2`, `differs` otherwise, and `invalid request`
for any other length or a number past its bound.

The bound is written as the square of `root + 1`, not as
`root^2 + 2 root + 1`. The second form needs `(r + 1)^2` expanded, which
`simplify` cannot do because it may not unfold the runtime's product, so a
theorem tying the two was not provable here. With the next root's square
written directly, `root_ok_decides` is linear in the two squares.

### The pinned test: `kernel_kn_43`

| Planted defect | Result |
| --- | --- |
| a value equal to the next square accepted (`<=?` for `<?`) | rejected: `rootOk changed` |
| a root up to `2^32` admitted, past the 64-bit guard | rejected: `rootAnswer changed` |
| decision 43 routed to decision 41's answer | rejected: `decision 43 must be routed to Kernel.Root::rootAnswer` |
| none | passed |

## KN-44, the rounded quotient

RMS norm divides each input by a root known only at run time, and softmax
divides each exponent by their sum. Each rounds the quotient once, half to
even. KN-41 already proves that rule for any step: a code within half a step,
the even code on a tie, and one code per witness. Decision 44 runs it with the
divisor as the step, so it adds no new rounding to trust.

A 20-byte request carries the value (8 bytes, below `2^62`), the divisor
(4 bytes, 1 to below `2^31`), half the floor quotient (4 bytes, below `2^30`),
whether the floor quotient is odd (1 byte) and the claimed code (2 bytes, at
most 32767, the bound `rqDownOk` keeps). The answer is `exact`, `differs` or
`invalid request`.

The module is `Kernel.Divide`, not `Kernel.Quotient`: LexLean refuses a module
segment spelled like a Lean namespace, and Lean has `Quotient`.

### The pinned test: `kernel_kn_44`

| Planted defect | Result |
| --- | --- |
| the odd bit dropped from the product, so the witness is taken as `2 half` | rejected: `divideProduct changed` |
| a divisor up to `2^32` admitted, past the 64-bit guard | rejected: `divideAnswer changed` |
| a divisor of 0 admitted | rejected: `divideAnswer changed` |
| decision 44 routed to decision 43's answer | rejected: `decision 44 must be routed to Kernel.Divide::divideAnswer` |
| none | passed |

## KN-45, the engine session

Decided by Ilya on 2026-11-18: milestone 2 runs the whole engine in the
guest, written in LexLean. PrismPM's session world (`core-wasm@2`, proposal
629, merged in PrismPM #638, not yet released) lets one guest instance serve a
session's ordered steps, a session being the fold of a pure step over its
requests. `app/src/Engine/Session.lex.tex` is Hologram AI's side of that: the
step's phase, admitted pages, page count, capacity and position, as functions
of the state and one request (open, admit a page, prompt chunk, generate,
close), built on decisions 35 and 36. It is outside `app/src/Kernel` and
nothing routes to it: it becomes the guest's root when the session world is in
the SDK image `prismpm.lock` pins. The forward pass it runs is the integer
stages (KN-37 to KN-44), joined in as they are proved.

### The proof gate

The local SDK image no longer runs on this machine (2026-11-18), so the proofs
were checked by the hosted LexLean verifier on a draft holding the session,
its proofs and the five modules they import. That is a diagnostic, not the
pinned SDK's verdict; CI gives that once the branch is pushed.

- `lexlean fmt --check`: canonical (all nine modules).
- `lexlean verify src/Proofs/Session.lex.tex`: verified on 2026-11-18 (hosted task-49, draft prism-app/a7ddf49d…), after three corrections the earlier runs found: two theorems restated so their `if` sits in the goal, and two axiom lists narrowed to what Lean observed (`Quot.sound, propext`).

Found on the way, kept for the next module: `first` is a keyword; a proof
module must import every module its statements name; parameter and
definition lists past 100 characters are written one item per line; and
`linear_arithmetic` splits an `if` in the goal but not in a hypothesis, so
a theorem whose premise would hold an `if` is stated with the `if` in the
goal (`session_ready_only_with_every_page` and `session_refuses_floats` are
written that way).

### The pinned test: `kernel_kn_45`

It pins the six statements and their axioms, that the five state functions are
declared, and that the session imports decisions 35 and 36. 37 kernel tests
pass.

## KN-50, a config number read from its decimal text

Hologram AI's inference is integers only, from the bytes admitted to the token
ids. A model's `config.json` states its RMS norm epsilon and its rotary theta
as decimal text, and the page used to turn them into numbers with JavaScript
doubles and a `Float32Array`. Decision 49 reads them as the integers they are:
a binary32 for the epsilon, a whole number for the theta and the head counts.

A request is fifty bytes: the tag 49, a mode byte (0 for the nearest binary32,
1 for a whole number), the claim in eight bytes and the text in forty, padded
with zero bytes, which end it. The kernel reads the text as five eight-byte
words, as decision 18 reads an operator's name, so the page draws them as
fields. The
kernel reads the digits, an optional fraction and an optional exponent, so the
value is `mantissa * 10^up / 10^down` with the mantissa below 2^63. The
binary32 is reached on 64-bit words alone: the mantissa's bits, then zeros,
pass through four long divisions by at most 10^18 each, chained bit by bit, so
the four together divide by 10^down exactly (a floor of a floor is the floor of
the product). They stop when 53 quotient bits stand; any remainder left, or a
mantissa bit not yet read, is the bit below them, and `Kernel.Round` rounds the
whole once, half to even, subnormals included. The whole number is the
mantissa times a power of ten, or divided by one that divides it exactly.

The answer is `exact`, `differs`, `not a decimal`, `out of range` (past 2^63,
or more than 72 places below the point), `not whole`, or `invalid request`.

The module is `Kernel.Config`, not `Kernel.Decimal`: LexLean refuses a module
segment spelled like a Lean namespace, and Lean has `Decimal`.

| Claim | Evidence | Level |
| --- | --- | --- |
| One step of the long division keeps its remainder below the divisor, and the remainder and the bit it gives reconstruct twice the old remainder plus the bit in | `decimal_left_below_divisor`, `decimal_step_takes_divisor`, `decimal_step_keeps_below`, `decimal_out_is_one_exactly_when_taken` | proved (hosted verifier, task-71) |
| A stage whose divisor is 1 passes its bit on and keeps nothing, so a value with fewer than 55 places uses fewer stages unchanged | `decimal_unit_stage_passes_bit`, `decimal_unit_stage_keeps_nothing` | proved (hosted verifier, task-71) |
| A claim is `exact` exactly when it is the result | `decimal_accepts_decides` | proved (hosted verifier, task-71) |
| The four stages together are the floor of the mantissa over 10^down, and the result is the nearest binary32 | not stated: the composition of floors is a statement about products of variables, outside `linear_arithmetic`. Tested instead: EN-58 compares the crate's bits with Rust's correctly rounded parse on 35 texts here and on 200,000 random decimals once, and the 62 claims the crate makes were run through a line-by-line Python port of this module, which gave the kernel's answer for each | tested |
| No number decision 49 forms reaches 2^64 | the mantissa stops below 2^63 as it is read, a power of ten is formed only up to 10^18, a remainder stays below its divisor, the quotient stops below 2^53, and the product with a power of ten is formed only when it fits | pinned by `kernel_kn_50` |
| Decision 49 answers as its definition does in the compiled guest | the 62 claims in `crates/conformance/tests/kernel/claims.json`, replayed by `replay.mjs` in the `app` recipe | replayed on CI |

### The pinned test: `kernel_kn_50`

| Planted defect | Result |
| --- | --- |
| the long division stopped at 21 quotient bits | rejected: `decimalRun stops at other than 53 quotient bits` |
| decision 49 routed to decision 44's answer | rejected: `decision 49 must be routed to Kernel.Config::decimalAnswer` |
| none | passed |

## KN-51, a rotary cache entry

A 12-byte request carries an f16 from an export's cos or sin cache (2 bytes), a
reference's sign (1 byte) and magnitude in units of `2^-30` (4 bytes, at most
`2^30`) and a tolerance in the same units (4 bytes, at most `2^30`). The kernel
reads the f16 exactly through `Kernel.Scale` (units of `2^-24`, times 64) and
answers `agrees` when the two lie within the tolerance, `disagrees` when they
do not, and `not finite` for an infinity or a NaN. No number it forms passes
`2^47`.

| Claim | Evidence | Level |
| --- | --- | --- |
| A claim is `agrees` exactly when the distance is within the tolerance; the distance is the difference for one sign and the sum for opposite signs; an entry equal to the reference agrees at any tolerance | `rotary_agrees_decides`, `rotary_same_sign_is_difference`, `rotary_opposite_signs_add`, `rotary_exact_entry_agrees` | proved (hosted verifier, task-83) |
| Decision 50 answers as its definition does in the compiled guest | 19 claims in `claims.json`, replayed by `replay.mjs`; a Python port of the module gave each | replayed on CI |

### The pinned test: `kernel_kn_51`

| Planted defect | Result |
| --- | --- |
| the f16 read at half its units (times 32) | rejected: `rotaryCache changed` |
| decision 50 routed to decision 49's answer | rejected: `decision 50 must be routed to Kernel.Rotary::rotaryAnswer` |
| none | passed |

## KN-52 and KN-53, an encoder's operators and files

Row 9 runs an embedding model and a classifier from the page, so their graphs
must pass decision 18 and their files be named by decision 24.

Decision 18 gains a third export style, 2, the encoder's: 25 operator names,
the union of every node of Xenova's `onnx/model_q4f16.onnx` and
`onnx/model_q4.onnx` for all-MiniLM-L6-v2 (751bff3) and
distilbert-base-uncased-finetuned-sst-2-english (0b6928e), read on 2026-11-19
from the graphs' node sections alone by Range request (the initializers begin
inside the first 512 KB of all four; the bytes were deleted). Style 3 or more
is still no question. The importer asks style 2 only when its caller says the
graph is an encoder's (`onnx_start_encoder`), since an encoder's 4-bit weights
name the fused domain as a decoder's do.

Decision 24 names the files of tasks 1 (feature extraction) and 2 (text
classification) as it names text generation's: Xenova's exports use the same
`onnx/model<suffix>.onnx`. A task of 3 or more is answered `unsupported task`.

| Claim | Evidence | Level |
| --- | --- | --- |
| Style 2 asks exactly the encoder's list, and no style past 2 lists any name | the body of `opnameListed`, pinned by `kernel_kn_18`; the name comparison itself is `opname_is_decides`, proved. Two theorems were tried on the hosted verifier (tasks 102 to 137): each needs `if false = true` reduced, which `simplify` here will not do with the rewrites it admits, and `linear_arithmetic` does not split on a Boolean result. As for KN-18's two lists, the dispatch is not a theorem | tested, not proved |
| The 25 names are the four graphs' | `kernel_kn_52` holds the names and checks the model's numbers are their bytes and that it lists no other | tested, not proved |
| A file is named exactly for tasks 0 to 2; a task of 3 or more is refused | `source_named_decides`, `source_undeclared_decides`, `source_task_decides`, restated | proved (hosted verifier, task-135) |
| Decisions 18 and 24 answer the new requests in the compiled guest | 9 claims in `claims.json`: `Erf` and `Gemm` supported in an encoder graph, `Erf` refused in a decomposed one and `GroupQueryAttention` in an encoder's, style 3 invalid; the two tasks' graph names, an undeclared part, task 3 refused | replayed on CI |

### The pinned tests: `kernel_kn_52`, `kernel_kn_53`

| Planted defect | Result |
| --- | --- |
| `Erf` taken out of `opnameEncoder` | rejected: `Erf is not in opnameEncoder` |
| none | passed, with `kernel_kn_18` and `kernel_kn_24` pinning the widened bodies |

## The examples in the compiled guest

`prismpm --project app verify` builds the holo and replays every example in the
compiled guest. Independently, `npx prism-code vectors` on the built holo
replayed 29 of 29 with the holo's identities verified, including the three
examples at the top of the 64-bit field.

One gate here has not been shown to fail: no example has been planted with a
wrong expected answer to confirm the replay rejects it.
