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
| Decision 37 answers the crate's claims in the compiled guest | `replay.mjs` in the `app` recipe, over the decision-37 claims in `crates/conformance/tests/kernel/claims/` | tested on CI |
| `KN-42`: a claimed binary32 is `exact` exactly when it is `roundResult` of the sign, magnitude and exponent: zero for a zero magnitude, infinity when the leading bit is past 2^127, otherwise the kept bits plus one exactly when the guard bit is set and the sticky bit or the last kept bit is | `round_accepts_decides`, `round_zero_is_zero`, `round_past_largest_is_infinite`, `round_up_at_most_one`, `round_no_guard_no_change`, `round_shift_past_word`; checked by the Lean kernel | proved |
| That `roundResult` is the binary32 nearest the value | argued from the definition's steps, which follow IEEE 754's; nearness itself is not stated, because the proof language's arithmetic is linear | not proved |
| Decision 42 answers the crate's claims in the compiled guest | `replay.mjs` in the `app` recipe, over the decision-42 claims in `crates/conformance/tests/kernel/claims/` | tested on CI |
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
| Every routed decision, 1 to 44, refuses at least one request in the compiled guest, and decisions 12 to 44 also accept one there | 1 to 11 by the holo's examples; 12 to 44 by `crates/conformance/tests/kernel/claims/`, replayed through the built holo by `kernel/replay.mjs` in the `app` recipe and required decision by decision by `assert_claimed` in each `kernel_kn_` test (see Milestone 2, row 1) | tested |

Each proof is about the function the guest executes, named by its module. That
the request bytes are parsed into that function's arguments correctly is not
proved; the examples exercise it, with a one-bit difference in each label word
and in the low and high bytes of each number.

## One file per decision

Each decision's record (its proof gate, its pinned test, its examples and the
planted defects each was seen to reject) is its own file,
[`kernel-verification/KN-NN.md`](kernel-verification/), named by the decision's
first `KN` ID; a section covering several IDs (`KN-18` to `KN-28`) is filed
under the first. A new decision adds its own file and edits nothing here, so
two pull requests that add decisions never touch the same lines.

The table above is closed: it records the decisions filed before the split. A
new decision's claims, evidence and kinds go in a table of the same columns at
the head of its own file.

### Adding a decision

Every pull request that added a decision used to edit the same five places, so
each merge sent every other open one back for another full acceptance run. Now a
decision adds its own files and edits one line that is its own:

| What | Where | Checked by |
| --- | --- | --- |
| Its record | `kernel-verification/KN-NN.md` | read by people |
| The claims its crate makes | `crates/conformance/tests/kernel/claims/NNN.json` (decision NNN) | the Rust suites (`kernel/claims.rs`) and `replay.mjs` refuse a claim whose first byte is not its file's decision, a file naming another decision, and an empty file; `replay.mjs` runs every claim in the compiled guest |
| Its names on the page, and its phrase in the page's description | `page/decision-names/NNN.json` | `xtask page` refuses a routed decision with no file, no name, an undescribed field or no phrase; `EN-14` checks the generated description names every routed tag |
| Its route | the body of `slotN` in `app/src/Kernel.lex.tex`, decisions 81 to 112, and its `\importmodule` line in sorted order | `assert_slot_table` in the kernel suite: the route falls back to `slots`, `slots` reaches each slot by a fixed `<=?` tree that names no tag, a slot either refuses with no axioms or answers exactly its own tag, and the imports are sorted. While every slot is empty, `slots` rests on no axioms (`axioms: []`, as Lean observed); the first filled slot raises it to its answer's axioms, `[Quot.sound, propext]` for every answer so far, an identical one-line edit that two pull requests merge cleanly |
| Its theorems | `app/src/Proofs/*.lex.tex` | `xtask proofs` verifies the module and its dependants; `page-journey` checks the assembled page names every theorem |

Generated, never committed: `site/decisions.json` and `site/card.html` (from
`page/card.template.html`), written by `cargo xtask page --write`, which
`xtask validate` and the `app` recipe run. The theorem list app.json used to
carry is gone: the page reads the theorems from `app/src`.

Still shared, and outside this split: `model/ids.toml` (and `CONFORMANCE.md`
generated from it), `features/suites/kernel.feature`,
`crates/conformance/tests/kernel.rs` and the examples in `application`. A
decision appends to each, so two pull requests can still meet there.

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

## Milestone 2, row 1: every decision refused somewhere real

Row 1 of the milestone asks that every decision be registered, proved, and
covered by a scenario and a negative test. Audited on 2026-11-19 at main
622d9e0: all 45 KN IDs are in `model/ids.toml`, each has a proved row above, a
`@KN-` scenario in `features/suites/kernel.feature` and a `kernel_kn_` test.
The negative test was the gap. A pinned test reads the model's text, and a
theorem says what the definition decides, but neither runs the compiled guest
on a request it should refuse. Decisions 1 to 11 had that in the holo's
examples, and 18, 19, 21, 34, 37 and 42 in `kernel/claims/`; the other 27 had none.

Each of the 27 now has, in `kernel/claims/`, one request it accepts, one for each
refusal it can give, and one a byte short, answered `invalid request`. Where
the engine's crates build the request (decisions 13, 20 and 22 to 28) the test
builds it with `hologram-ai-kernel-client`, so the claim is the crate's own
request; elsewhere no crate sends the decision yet and the bytes follow the
model's wire format. Decision 38 has no refusal but `invalid request`; it is
shown for a value too wide for its format and for a short request. Decision 24's
unsupported task and precision use code 255, away from the boundary.

The answers were taken from the compiled guest of the holo published from main
622d9e0 (`app.holo` on the project's Pages site), and agree in all 97 cases with
the answer read from each decision's definition.

KN-45, the engine session, is not a routed decision: it is not in the guest's
`route`, and its refusals (a page not named by BLAKE3, a floating-point stage,
a request after close) are the proved theorems `session_refuses_*`.

Decisions 49 and 50 (KN-50, KN-51) reached main after the audit with their
refusals already in `kernel/claims/`: `not a decimal`, `not whole` and `out of
range` for decision 49, `not finite` for decision 50. So did decision 45
(KN-46, #85), with `refused: unknown template`; KN-55 (#95) is a stage, not a
routed decision. KN-52 and KN-53, decisions 18 and 24 widened, are covered in
their section. Decision 70 (KN-70, #97) came with its refusals: unsupported
task, device and dtype, sampling, beam search and repetition penalty. Decisions
registered in open pull requests are their authors' to cover: KN-47 (#86), and
KN-49 and decision 48 (#96).

Planted defect: the claim that decision 12 answers `cpu` for a largest buffer
over the binding limit was changed to `webgpu`.

```
$ node crates/conformance/tests/kernel/replay.mjs app.holo crates/conformance/tests/kernel/claims
kernel replay: 1 of 195 claims answered otherwise
0c0100000000000001f400000000000003e8: the guest answers `cpu`, the claim lists `webgpu`
```

Exit 1. Restored, the same command prints `kernel replay: the guest answers all
195 claims as listed`.

## The examples in the compiled guest

`prismpm --project app verify` builds the holo and replays every example in the
compiled guest. Independently, `npx prism-code vectors` on the built holo
replayed 29 of 29 with the holo's identities verified, including the three
examples at the top of the 64-bit field.

One gate here has not been shown to fail: no example has been planted with a
wrong expected answer to confirm the replay rejects it.
