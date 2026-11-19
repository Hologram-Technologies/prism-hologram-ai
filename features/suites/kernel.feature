Feature: kernel

  The Hologram AI kernel, in app/: every decision the application makes about
  what it may load and run. A request names one decision in its first byte and
  carries that decision's numbers; the answer is the verdict. The engine and the
  page obey the verdict and never decide for themselves.

  @KN-05 @build
  Scenario: A buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal
    Given the admission model in app/src/Kernel/Admission.lex.tex and its proofs in app/src/Proofs/Admission.lex.tex
    When the Lean kernel checks admitted_decides during lexlean verify in the gate
    Then admitted is true exactly when all four label words and the two lengths are equal
    And the verdict the application returns for decision 5 is decided by that same admitted

  @KN-06 @build
  Scenario: A tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file
    Given the range model in app/src/Kernel/Range.lex.tex and its proofs in app/src/Proofs/Range.lex.tex
    When the Lean kernel checks within_decides, accepted_decides and all_accepted_ordered during lexlean verify in the gate
    Then accepted is true exactly when the previous end is at most the offset and the offset plus the length is at most the file length
    And a list of ranges is all accepted exactly when each starts at or after the end of the one before it and ends within the file
    And the verdict the application returns for decision 6 is decided by that same accepted

  @KN-08 @build
  Scenario: The decode bucket is at least what is needed, never above the maximum, and never below the current bucket
    Given the bucket model in app/src/Kernel/Bucket.lex.tex and its proofs in app/src/Proofs/Bucket.lex.tex
    When the Lean kernel checks fits_decides and grow_bounds during lexlean verify in the gate
    Then for every need and current bucket within the maximum the new bucket covers the need, stays within the maximum and is not below the current bucket
    And a need or a current bucket past the maximum is refused by name
    And the bucket the application returns for decision 8 is the one grow gives

  @KN-09 @build
  Scenario: A turn token budget never exceeds the context that remains and never exceeds the request
    Given the generation model in app/src/Kernel/Generation.lex.tex and its proofs in app/src/Proofs/Generation.lex.tex
    When the Lean kernel checks has_room_decides, budget_within_context and budget_within_request during lexlean verify in the gate
    Then with room left the budget fits the context that remains and is at least one token
    And the budget never exceeds a request that was made
    And a context with no room left is refused by name
    And the budget the application returns for decision 9 is the one budget gives

  @KN-02 @build
  Scenario: A model is supported or derived exactly when no required key is missing, no unsupported feature is flagged and the manifest fits, and only then may weights download
    Given the preflight model in app/src/Kernel/Preflight.lex.tex and its proofs in app/src/Proofs/Preflight.lex.tex
    When the Lean kernel checks supported_decides, derived_decides and authorised_decides during lexlean verify in the gate
    Then the verdict is supported exactly when the model is runnable and its family is recognised
    And the verdict is derived exactly when the model is runnable and its family is not recognised
    And weights are authorised exactly when the model is runnable
    And each rejection names its reason, a missing key first

  @KN-04 @build
  Scenario: The quantisation tier chosen fits in memory whenever any tier does, and more memory never lowers it
    Given the tier model in app/src/Kernel/Tier.lex.tex and its proofs in app/src/Proofs/Tier.lex.tex
    When the Lean kernel checks half_up_bounds, tier_fits and tier_monotone during lexlean verify in the gate
    Then int8 in memory is chosen exactly when the parameters fit the budget at one byte each
    And int4 in memory is chosen only when half the parameters, rounded up, fit
    And a tier in memory is chosen whenever one fits
    And a larger budget never gives a lower tier in the order paged, int4 in memory, int8 in memory

  @KN-11 @build
  Scenario: A requested dimension is bound exactly when it lies within its bounds, to a size at least the request and never above the upper bound
    Given the shape model in app/src/Kernel/Shape.lex.tex and its proofs in app/src/Proofs/Shape.lex.tex
    When the Lean kernel checks in_bounds_decides and bind_bounds during lexlean verify in the gate
    Then a request is bound exactly when it lies between the lower and upper bounds
    And the size it is bound to is at least the request and at most the upper bound, for every step
    And a request outside the bounds is refused by name

  @KN-07 @build
  Scenario: A proposed trim is accepted exactly when what remains fits the context and restoring the newest dropped pair would not
    Given the trim model in app/src/Kernel/Trim.lex.tex and its proofs in app/src/Proofs/Trim.lex.tex
    When the Lean kernel checks trimmed_decides and trim_fits_decides during lexlean verify in the gate
    Then a proposed trim is accepted exactly when what remains fits the context
    And, when anything was dropped, restoring the newest dropped pair would pass the context
    And each refusal names the condition that failed
    And the verdict the application returns for decision 7 is decided by that same trimmed

  @KN-03 @build
  Scenario: A proposed stage plan is accepted exactly when its stage fits the memory budget, is the largest that does, and its stages cover every layer once
    Given the plan model in app/src/Kernel/Plan.lex.tex and its proofs in app/src/Proofs/Plan.lex.tex
    When the Lean kernel checks planned_decides and can_run_decides during lexlean verify in the gate
    Then a proposed plan is accepted exactly when the context and one stage fit the budget
    And one more layer would not fit, unless the stage is the whole model
    And the stages cover every layer with none empty
    And a model whose single layer does not fit beside the context is refused by name

  @KN-01 @build
  Scenario: Each working state is entered from exactly one state by exactly one event, failed is left only by reset, and every other step is refused
    Given the journey model in app/src/Kernel/Journey.lex.tex and its proofs in app/src/Proofs/Journey.lex.tex
    When the Lean kernel checks the eight enters theorems and failed_only_resets during lexlean verify in the gate
    Then preflight, download, compile, materialize and generating are each entered from exactly one state by exactly one event
    And ready is entered only from materialize when the labels are verified, or back from generating
    And failed is entered only by a failure while work is under way, and left only by reset
    And every other step is refused

  @KN-10 @build
  Scenario: A graph is importable exactly when its opset lies within the implemented window and no operator is unimplemented, and a wider window never makes an importable graph unimportable
    Given the fixture KN-10 needs
    When the suite exercises KN-10
    Then A graph is importable exactly when its opset lies within the implemented window and no operator is unimplemented, and a wider window never makes an importable graph unimportable

  @KN-12 @build
  Scenario: WebGPU is chosen exactly when the adapter is present and the model largest buffer fits its binding limit, and a wider limit never takes WebGPU away
    Given the backend model in app/src/Kernel/Backend.lex.tex and its proofs in app/src/Proofs/Backend.lex.tex
    When the Lean kernel checks gpu_usable_decides and limit_monotone during lexlean verify in the gate
    Then gpuUsable is true exactly when the adapter is present and the largest buffer is at most the binding limit
    And raising the binding limit never turns WebGPU into the CPU path
    And the answer to decision 12 is webgpu or cpu by that same gpuUsable

  @KN-13 @build
  Scenario: A weight block length is the one its format and shape imply, and a shape whose columns do not divide into the format groups is refused
    Given the block model in app/src/Kernel/Blocks.lex.tex and its proofs in app/src/Proofs/Blocks.lex.tex
    When the Lean kernel checks shaped_decides and group_bounds during lexlean verify in the gate
    Then a shape is accepted exactly when rows and columns are at least one and the columns divide into the format groups
    And the answer to decision 13 is the block length rows times groups times bytes per group, or bad shape

  @KN-14 @build
  Scenario: A claimed greedy pick is accepted exactly when its index is the first position holding the maximum score and the claimed score is that maximum
    Given the pick model in app/src/Kernel/Pick.lex.tex and its proofs in app/src/Proofs/Pick.lex.tex
    When the Lean kernel checks standing_decides and holds_decides, the second by induction over the positions, during lexlean verify in the gate
    Then a claim is accepted exactly when every earlier score is below the claimed score, the score at the index equals it, and no later score is above it
    And the answer to decision 14 checks all n scores from position 0

  @KN-15 @build
  Scenario: A token is kept unpenalised exactly when no position inside the recent window holds it
    Given the penalty model in app/src/Kernel/Penalty.lex.tex and its proofs in app/src/Proofs/Penalty.lex.tex
    When the Lean kernel checks clear_decides and clear_from_decides, the second by induction over the history, during lexlean verify in the gate
    Then the token is kept exactly when no position at or after the window start holds it
    And the window starts window tokens before the end of the history, or at its start when the history is shorter

  @KN-16 @build
  Scenario: A speculative commit count is accepted exactly when every draft before the last committed position equals its verified pick and the draft at that position, if any, differs
    Given the speculation model in app/src/Kernel/Speculate.lex.tex and its proofs in app/src/Proofs/Speculate.lex.tex
    When the Lean kernel checks agrees_decides and agrees_from_decides, the second by induction over the drafts, during lexlean verify in the gate
    Then a count is accepted exactly when it lies in 1 to k plus 1, every draft before the last committed position equals its verified pick, and the draft at that position, if any, differs

  @KN-17 @build
  Scenario: Unpacking a packed weight group returns the code packed at each place, for each format, and each code maps to its stated level
    Given the packing model in app/src/Kernel/Quant.lex.tex and its proofs in app/src/Proofs/Quant.lex.tex
    When the Lean kernel checks the round trip theorems for nibbles, pairs, triples and bits during lexlean verify in the gate
    Then unpacking any place of a packed group gives back the code packed at that place, for every code below its base
    And the answer to decision 17 is that code minus the format offset, written with its sign

  @KN-18 @build
  Scenario: An operator is supported exactly when its name is on the closed list of the export style asked about, and no name is supported under an unknown style
    Given the operator name model in app/src/Kernel/Opnames.lex.tex and its proofs in app/src/Proofs/Opnames.lex.tex
    When the Lean kernel checks opname_is_decides during lexlean verify in the gate
    Then a name is a listed name exactly when the four words of its 32 bytes equal that name's four words
    And the decomposed style lists the 27 names read from Qwen2.5-1.5B's 4-bit export and the fused style the 18 read from Llama 3.2 1B, Qwen3-0.6B and SmolLM2-360M, each written in the model as the four words of its bytes
    And the answer to decision 18 is supported or unsupported operator by that same opnameListed, and a style other than 0 or 1 is no question
    And the guest the app recipe builds answers unsupported operator for LayerNormalization, which is on neither list, and for Add asked of a fused graph, and supported for MatMulNBits, as the claims it replays list

  @KN-19 @build
  Scenario: A model configuration is recognised exactly when its type is a listed family and its shape is one the engine runs
    Given the family model in app/src/Kernel/Family.lex.tex and its proofs in app/src/Proofs/Family.lex.tex
    When the Lean kernel checks family_listed_decides, family_shaped_decides, family_supported_decides and family_unknown_decides during lexlean verify in the gate
    Then a type is listed exactly when it is qwen2, qwen3 or llama, read as the eight bytes of its name
    And a shape is one the engine runs exactly when the key-value heads are at least one and divide the heads, the head size is even and at least two, and layers, width and vocabulary are at least one
    And the answer to decision 19 is supported exactly when the type is listed, the activation is silu and the shape is one the engine runs
    And the guest the app recipe builds answers rejected: unknown family for gemma2 and for mixtral, a mixture of experts, and rejected: unsupported shape for llama with gelu, as the claims it replays list
    And a type or activation longer than eight bytes, such as qwen3_moe or gelu_pytorch_tanh, has no word to ask with, so it is never taken for a listed one

  @KN-20 @build
  Scenario: A MatMulNBits weight and scale length is the one its rows, columns and block size imply, and its blocks are the fewest that cover a row
    Given the MatMulNBits length model in app/src/Kernel/Nbits.lex.tex and its proofs in app/src/Proofs/Nbits.lex.tex
    When the Lean kernel checks nbits_shaped_decides, nbits_block_bounds, nbits_blocks_fewest_16, nbits_blocks_fewest_32, nbits_blocks_fewest_64, nbits_blocks_fewest_128 and nbits_blocks_fewest_256 during lexlean verify in the gate
    Then a shape is accepted exactly when rows and columns are between one and 16,777,215 and the block size is 16, 32, 64, 128 or 256, and the blocks of a row are the fewest that cover its columns
    And the answer to decision 20 is rows times blocks times the bytes of one block, which are half the block size for the 4-bit codes, 4 for f32 scales and 2 for f16 scales, or bad shape

  @KN-21 @build
  Scenario: The precision chosen is the first offered one in the stated order that fits the budget, and none is chosen exactly when no offered one fits
    Given the precision model in app/src/Kernel/Precision.lex.tex and its proofs in app/src/Proofs/Precision.lex.tex
    When the Lean kernel checks precision_fits_decides, precision_first, precision_second, precision_third, precision_none and precision_budget_monotone during lexlean verify in the gate
    Then a precision fits exactly when it is offered and its size is at most the budget
    And q4f16 is chosen exactly when it fits, q4 exactly when q4f16 does not and q4 does, fp16 exactly when neither does and fp16 does, and none exactly when none fits
    And a wider budget never moves the choice to a later precision
    And the answer to decision 21 is the precision of that same precisionCode
    And the guest the app recipe builds answers none fits for a record that offers none of q4f16, q4 and fp16, as one with only q8, int8 or bnb4 files does, as the claims it replays list

  @KN-22 @build
  Scenario: The device is WebGPU exactly when the adapter can hold the model, the CPU exactly when it cannot and the model is within the CPU ceiling, and otherwise the model is refused
    Given the device model in app/src/Kernel/Device.lex.tex and its proofs in app/src/Proofs/Device.lex.tex
    When the Lean kernel checks device_webgpu_decides, device_cpu_decides and device_refused_decides during lexlean verify in the gate
    Then WebGPU is answered exactly when gpuUsable holds
    And the CPU is answered exactly when gpuUsable does not hold and the model is at most the CPU ceiling
    And the model is refused exactly when gpuUsable does not hold and the model is above the CPU ceiling
    And the answer to decision 22 is webgpu, cpu or refused: too large for cpu by that same deviceCode

  @KN-23 @build
  Scenario: A span in a model stored as several files is accepted exactly when its file is one the model declares, it follows the previous span, and it ends inside that file
    Given the file span model in app/src/Kernel/Chunks.lex.tex and its proofs in app/src/Proofs/Chunks.lex.tex
    When the Lean kernel checks chunk_declared_decides, chunk_inside_decides and chunk_spans_disjoint during lexlean verify in the gate
    Then a file is declared exactly when the model declares at most 100 data files and the file's index is at most that count
    And a span is inside exactly when it starts at or after the previous span's end and ends within the file, and two spans accepted in order do not overlap
    And the answer to decision 23 is undeclared file, or accept or reject by that same chunkInside

  @KN-24 @build
  Scenario: The file named for a task, precision and part is the one the export convention gives, and a part beyond the declared count is refused
    Given the source model in app/src/Kernel/Source.lex.tex and its proofs in app/src/Proofs/Source.lex.tex
    When the Lean kernel checks source_named_decides, source_undeclared_decides and source_task_decides during lexlean verify in the gate
    Then a file is named exactly when the task is text generation, the precision is one of the four listed, at most 100 data files are declared and the part is at most that count
    And a part beyond the declared count is answered undeclared file
    And the answer to decision 24 is the graph file onnx/model, the precision's suffix and .onnx for part 0, that name with _data for part 1, and with _data_ and the part less one after it

  @KN-25 @build
  Scenario: Generation stops exactly when the token is one of the stop ids of the model or the budget is spent, and a stop id is reported before the budget
    Given the stop model in app/src/Kernel/Stop.lex.tex and its proofs in app/src/Proofs/Stop.lex.tex
    When the Lean kernel checks stop_differs_decides, stop_none_from_decides and stop_spent_decides during lexlean verify in the gate
    Then no stop id is met exactly when every listed id differs from the token, by induction over the list
    And the budget is spent exactly when the tokens generated are at least the budget
    And the answer to decision 25 is stop: end token when a listed id is the token, otherwise stop: budget when the budget is spent, otherwise continue

  @KN-26 @build
  Scenario: A cached block is evicted exactly when it is not pinned and space is still short, and does not fit is answered exactly when no candidate remains and space is still short
    Given the eviction model in app/src/Kernel/Evict.lex.tex and its proofs in app/src/Proofs/Evict.lex.tex
    When the Lean kernel checks evict_short_decides, evict_decides, evict_does_not_fit_decides and evict_enough_decides during lexlean verify in the gate
    Then evict is answered exactly when the block is not pinned, a candidate remains and the space free and freed is short of what is needed
    And does not fit is answered exactly when no candidate remains and space is still short, and enough exactly when it is not short
    And the answer to decision 26 is enough, evict, keep or does not fit by that same evictCode

  @KN-27 @build
  Scenario: A computed prefix is reused exactly when its address equals the one asked for and it is shorter than the prompt
    Given the reuse model in app/src/Kernel/Reuse.lex.tex and its proofs in app/src/Proofs/Reuse.lex.tex
    When the Lean kernel checks reuse_shorter_decides and reusable_decides during lexlean verify in the gate
    Then a held prefix is reusable exactly when its label equals the label asked for, it holds at least one position and it is shorter than the prompt
    And the answer to decision 27 is reuse or recompute by that same reusable, over the labels at bytes 1 and 33

  @KN-28 @build
  Scenario: A model is ready exactly when the addresses admitted are the ones its manifest lists, every listed tensor is admitted, and it lists at least one
    Given the manifest model in app/src/Kernel/Manifest.lex.tex and its proofs in app/src/Proofs/Manifest.lex.tex
    When the Lean kernel checks manifest_complete_decides during lexlean verify in the gate
    Then a manifest is complete exactly when it lists at least one tensor and the tensors admitted are as many as it lists
    And the answer to decision 28 is refused when the label of the addresses admitted differs from the one the manifest gives, by the sameLabel of decision 5, and otherwise ready or incomplete by that same manifestComplete

  @KN-29 @build
  Scenario: An exact sum is accepted exactly when the activation, weight and group bits together are no wider than the accumulator, and a wider accumulator or a shorter group never loses it
    Given the exact model in app/src/Kernel/Exact.lex.tex and its proofs in app/src/Proofs/Exact.lex.tex
    When the Lean kernel checks exact_fits_decides, exact_fits_wider and exact_fits_shorter during lexlean verify in the gate
    Then an accumulator is exact exactly when the activation and weight bits are at least one and the activation, weight and group bits sum to no more than the accumulator bits
    And the answer to decision 29 is exact or may overflow by that same exactFits, over the bit widths at bytes 1 to 4

  @KN-30 @build
  Scenario: A run is reproducible everywhere exactly when its products and every other stage are exact, on the same hardware exactly when it is not and its float reductions keep a fixed order, and otherwise not at all
    Given the determinism model in app/src/Kernel/Determinism.lex.tex and its proofs in app/src/Proofs/Determinism.lex.tex
    When the Lean kernel checks class_everywhere_decides, class_same_hardware_decides, class_none_decides and comparable_decides during lexlean verify in the gate
    Then the class is everywhere exactly when products and every other stage are exact, same hardware exactly when it is not and the float reductions keep a fixed order, and none otherwise
    And the answer to decision 30 is everywhere, same hardware or none by that same classCode, and a run is comparable exactly when its class is everywhere or it is same hardware on the same adapter

  @KN-31 @build
  Scenario: A receipt is confirmed exactly when its model is ready, the run is comparable and the replayed output has the label it claims, a mismatch exactly when only the output differs, and otherwise not comparable
    Given the receipt model in app/src/Kernel/Receipt.lex.tex and its proofs in app/src/Proofs/Receipt.lex.tex
    When the Lean kernel checks receipt_confirmed_decides, receipt_mismatch_decides and receipt_not_comparable_decides during lexlean verify in the gate
    Then a receipt is confirmed exactly when decision 28 found its model ready, the run is comparable by decision 30 and the output label equals its replay by the sameLabel of decision 5
    And the answer to decision 31 is confirmed, mismatch or not comparable by that same receiptCode, over the labels at bytes 1 and 33

  @KN-32 @build
  Scenario: A probe of a past answer replays exactly when its layer, position and component lie within the run and the run is comparable, approximately when they lie within it and the run is not, and is otherwise out of range
    Given the probe model in app/src/Kernel/Probe.lex.tex and its proofs in app/src/Proofs/Probe.lex.tex
    When the Lean kernel checks probe_exact_decides, probe_approximate_decides and probe_out_of_range_decides during lexlean verify in the gate
    Then a probe replays exactly when its layer, position and component are each below their counts and the run is comparable by decision 30
    And the answer to decision 32 is exact replay, approximate replay or out of range by that same probeCode

  @KN-33 @build
  Scenario: A computed prefix is shared exactly when decision 27 reuses it and it was computed on this device or in a run that is comparable
    Given the share model in app/src/Kernel/Share.lex.tex and its proofs in app/src/Proofs/Share.lex.tex
    When the Lean kernel checks shareable_decides during lexlean verify in the gate
    Then a held prefix is shared exactly when decision 27 finds it reusable and it was computed on this device or the run is comparable by decision 30
    And the answer to decision 33 is reuse or recompute by that same shareable, over the labels at bytes 1 and 33

  @KN-34 @build
  Scenario: One 32-wide block of signed 4-bit weight codes times int8 activations is confirmed exact exactly when the claimed sum, offset by 32768, equals the sum the kernel recomputes from the block, and differs otherwise
    Given the block check model in app/src/Kernel/Block.lex.tex and its proofs in app/src/Proofs/Block.lex.tex
    When the Lean kernel checks accepts_decides, parts_cover_every_product and the four sign theorems during lexlean verify in the gate
    Then every product of the block falls in exactly one of the positive and negative parts, by the signs of its weight code and its activation
    And the answer to decision 34 is exact when the claim at byte 49 plus the negative part equals the positive part plus 32768, differs otherwise, and invalid request for any length but 53

  @KN-35 @build
  Scenario: A content label is admitted as an address exactly when its first seven bytes are blake3:, and any other axis, sha256: among them, is refused as not blake3
    Given the address model in app/src/Kernel/Address.lex.tex and its proofs in app/src/Proofs/Address.lex.tex
    When the Lean kernel checks address_blake3_decides and address_sha256_refused during lexlean verify in the gate
    Then a label's seven-byte prefix is admitted exactly when it is blake3:
    And the answer to decision 35 is blake3 for that prefix, not blake3 for sha256: or any other, and invalid request for any length but 8

  @KN-36 @build
  Scenario: A run may start exactly when its products and every other stage are exact, which is decision 30's class everywhere, and a run with a floating-point stage is refused whatever the order of its reductions
    Given the integer-run model in app/src/Kernel/Integer.lex.tex, decision 30's class in app/src/Kernel/Determinism.lex.tex, and the proofs in app/src/Proofs/Integer.lex.tex
    When the Lean kernel checks integer_run_decides, integer_run_is_everywhere and integer_run_refuses_fixed_order_floats during lexlean verify in the gate
    Then a run is admitted exactly when its products and every other stage are exact, which is decision 30's class everywhere for any order of reductions
    And the answer to decision 36 is integer for such a run, refused: not integer for a run with a floating-point stage, and invalid request for any length but 3 or a field above 1

  @KN-37 @build
  Scenario: One group's sum times its IEEE half scale is confirmed exact exactly when the claimed sign and magnitude, in units of 2^-24, equal the product the kernel recomputes from the half's exponent and fraction, a zero product is never negative, and a scale that is an infinity or a NaN is answered not finite
    Given a request of the tag 37, a group sum plus 32768 in four bytes, an IEEE half in two, a sign byte and a magnitude in eight
    When the kernel reads the half as its fraction when its exponent field is zero and as 1024 plus its fraction times 2 to the field less one otherwise, and multiplies the sum's magnitude by it
    Then the answer is exact exactly when the magnitude is that product and the sign is 1 just when the product is nonzero and the two signs differ
    And a zero product is never negative, a half whose exponent field is 31 is answered not finite, and a request of another length is answered invalid request
    And the theorems of Proofs.Scale state each of these and are pinned with their axioms
  @KN-38 @build
  Scenario: A stored q4 byte, IEEE half, bfloat16 or binary32 is read into integer fields, its two nibbles or its sign, exponent and fraction, that encode back to exactly the stored bits, and fields within their widths read back unchanged
    Given the codec model in app/src/Kernel/Codec.lex.tex and its proofs in app/src/Proofs/Codec.lex.tex
    When the Lean kernel checks codec_round_trip, codec_fields_fit and the four codec_reads_back theorems during lexlean verify in the gate
    Then every stored value narrower than its format encodes back from its fields to exactly the stored bits
    And every field lies within its width, and fields within their widths read back unchanged from their encoding
    And the answer to decision 38 is the fields in decimal, and invalid request for an unknown format or a value wider than it

  @KN-39 @build
  Scenario: A step that narrows a value, exp, reciprocal square root, sine and cosine, SiLU, GELU or requantising, is accepted exactly when its input is at most 16 bits so its table is finite, its output is 1 to 32 bits, and its certified error is within its declared bound
    Given the narrowing model in app/src/Kernel/Narrow.lex.tex and its proofs in app/src/Proofs/Narrow.lex.tex
    When the Lean kernel checks narrow_accepted_decides during lexlean verify in the gate
    Then a narrowing is accepted exactly when its step is known, its input is 1 to 16 bits, its output 1 to 32 bits, and its certified error at most its declared bound
    And the answer to decision 39 is accept or refuse, and invalid request for an unknown step or any length but 12

  @KN-40 @build
  Scenario: One link of a row's exact recombination is confirmed exactly when the partial before it plus its up to six terms, each a unit sum times a scale mantissa times a power of two, signed, equals the claimed partial after it; a link whose numbers could pass 64 bits is too wide, and two confirmed links compose, so a chain of them confirms the plain sum
    Given the row model in app/src/Kernel/Rows.lex.tex and its proofs in app/src/Proofs/Rows.lex.tex
    When the Lean kernel checks rows_accepts_decides, rows_part_splits, rows_part_one_side, rows_links_compose and rows_one_claim during lexlean verify in the gate
    Then a link is exact exactly when its positive parts plus the claim's negative part equal its negative parts plus the claim's positive part
    And two exact links compose into one, so a chain of links confirms the plain sum of every term in it
    And the answer to decision 40 is exact, differs, too wide when a partial reaches 2^62 or a term could reach 2^58, and invalid request for a sign above 1 or any length but 79

  @KN-41 @build
  Scenario: Up to five inputs of one group are confirmed put on sixteen-bit codes exactly when the group's power of two is the largest that keeps its largest input within 32767, and each code is its input at that power rounded to the nearest, ties to even, so within half a step of it; an input far below the step is code zero
    Given the requantising model in app/src/Kernel/Requant.lex.tex and its proofs in app/src/Proofs/Requant.lex.tex
    When the Lean kernel checks rq_down_witness, rq_down_half_step, rq_down_tie_even, rq_down_one_code, rq_up_exact, rq_largest_down_decides and rq_largest_up_decides
    Then a code below the step is within half a step of its input, a tie gives an even code, and at most one code is accepted for an input and its quotient
    And a power is accepted for the group only when its largest input is within 32767 at it and past 32767 at the next power
    And the answer to decision 41 is exact, differs, not the largest power, and invalid request for a direction or odd bit above 1 or any length but 62

  @KN-42 @build
  Scenario: A claimed binary32 is confirmed exact exactly when it is the bits the kernel rounds magnitude times 2 to the exponent to: the nearest, ties to even, from the leading bit, 24 bits or fewer below the normal range, infinity past the largest finite value, and zero for a zero magnitude
    Given a request of the tag 42, a sign byte, a magnitude in eight bytes, an exponent plus 32768 in two and claimed binary32 bits in four
    When the kernel finds the magnitude's leading bit, keeps 24 bits from it or fewer below 2^-126, and reads the guard bit and whether any bit below it is set
    Then the answer is exact exactly when the claim is those bits rounded to the nearest, ties to even, infinity past the largest finite value, and zero for a zero magnitude
    And no power of two the kernel forms reaches 2^64, so a shift of 64 or more is its own case
    And the theorems of Proofs.Round state these and are pinned with their axioms

  @KN-43 @build
  Scenario: A claimed integer square root of a value below 2^62 is confirmed exactly when its square is at most the value and the next square is above it, so a norm divides by the value's one floor root
    Given the root model in app/src/Kernel/Root.lex.tex and its proofs in app/src/Proofs/Root.lex.tex
    When the Lean kernel checks root_ok_decides during lexlean verify in the gate
    Then a root is accepted exactly when its square is at most the value and the value is below the square of the root plus one, so the accepted root is the floor square root
    And the answer to decision 43 is exact or differs, and invalid request for any length but 13, a value of 2^62 or more or a root of 2^31 or more, so no number it computes passes 64 bits

  @KN-44 @build
  Scenario: A claimed quotient of a value below 2^62 by a divisor below 2^31, rounded half to even to a code of at most 32767, is confirmed exactly when it passes KN-41's rounding rule with the divisor as the step, so a norm and a softmax divide by one rule
    Given the division model in app/src/Kernel/Divide.lex.tex and its proofs in app/src/Proofs/Divide.lex.tex
    When the Lean kernel checks divide_is_requant_rule during lexlean verify in the gate
    Then a quotient is accepted exactly when KN-41's rqDownOk accepts it with the divisor as the step and the quotient witness times the divisor as the product
    And so its code is within half a divisor of the value, a tie gives the even code, and one witness has at most one code
    And the answer to decision 44 is exact or differs, and invalid request for any length but 20, a value of 2^62 or more, a divisor of 0 or 2^31 or more, a half witness of 2^30 or more or an odd bit above 1, so no number it computes passes 64 bits

  @KN-45 @build
  Scenario: The engine session is a pure step over open, admit a page, prompt chunk, generate and close, ready to be the guest's root in the session world: a page not named by BLAKE3 fails the session, the session is ready only when the page admitted is the manifest's last, prompt chunks compose, a generate request with a floating-point stage fails the session, the position never passes the capacity, and an ended session stays ended
    Given the engine session in app/src/Engine/Session.lex.tex, built on decisions 35 and 36, and its proofs in app/src/Proofs/Session.lex.tex
    When the Lean kernel checks the six session theorems during lexlean verify in the gate
    Then admitting a page whose address is not blake3: fails the session, and admitting any page but the manifest's last leaves it not ready
    And two prompt chunks move the position as one chunk of their combined length, and the position stays within the capacity
    And a generate request whose products or other stages are not exact fails the session, and an ended or failed session does not change

  @KN-46 @build
  Scenario: A model's chat template is one the engine defines exactly when the BLAKE3 of its text is Qwen2.5-1.5B-Instruct's, Qwen3-0.6B's or Llama 3.2 1B Instruct's, as their exports carry them at the hub's pinned revisions; decision 45 answers qwen2.5, qwen3 or llama3.2 for those three and refuses any other template by name, before anything is rendered with it
    Given the template model in app/src/Kernel/Template.lex.tex and its proofs in app/src/Proofs/Template.lex.tex
    When the Lean kernel checks template_qwen25_decides, template_qwen3_decides, template_llama32_decides and template_code_bounded during lexlean verify in the gate
    Then a template is named a family exactly when the four 8-byte words of its BLAKE3 are that family's, and no other code is given
    And the app recipe replays decision 45 in the guest for the three templates, a template of the gate's own and a request of the wrong length
    And the engine page asks decision 45 for the template its model's tokenizer_config.json carries, and stops by name on any other

  @KN-47 @build
  Scenario: The kernel renders a conversation of up to eight system, user and assistant turns as Qwen2.5's, Qwen3's or Llama 3.2's chat template renders it, by its own definition (decision 46): Qwen2.5's default system line, Qwen3's empty thinking block when thinking is off and its trailing assistant turn, Llama 3.2's date line and Python's whitespace trim; a Qwen3 assistant turn carrying </think> reasoning, a role past assistant and a request of the wrong length are refused by name
    Given the chat model in app/src/Kernel/Chat.lex.tex and its proofs in app/src/Proofs/Chat.lex.tex
    When the Lean kernel checks the whitespace, role and family theorems during lexlean verify in the gate
    Then the trim strips exactly Python's whitespace below 33, only the three roles and the three families are rendered
    And the app recipe replays decision 46 in the guest for every recorded case that fits one request, each answered with the text transformers renders

  @KN-50 @build
  Scenario: A claimed binary32 or whole number for a config number written in decimal is confirmed exactly when it is the one the text names: the nearest binary32, ties to even, subnormals included, reached by long division on 64-bit words and Kernel.Round, or the whole number exactly; text that is not a decimal, past the range or not whole is refused by name
    Given a request of the tag 49, a mode byte (0 for binary32, 1 for a whole number), a claim in eight bytes and a decimal's text in at most forty
    When the kernel reads the digits, the fraction and the exponent, feeds the mantissa's bits through four long divisions by at most 10^18 each, chained bit by bit, until 53 quotient bits stand, and rounds them with any remainder as the bit below by Kernel.Round
    Then the answer is exact exactly when the claim is the nearest binary32, ties to even, subnormals included, or the whole number the text names
    And text that is not a decimal is answered not a decimal, a value past 2^63 or more than 72 places below the point out of range, and a fraction asked for as a whole number not whole
    And no number the kernel forms reaches 2^64, and the theorems of Proofs.Config state the long division's steps and are pinned with their axioms

  @KN-51 @build
  Scenario: A cache entry, an f16, is confirmed to agree with a reference rotary value in units of 2^-30 exactly when the f16 read exactly as an integer times a power of two lies within the tolerance of it, compared in integers; an infinity or a NaN is answered not finite
    Given a request of the tag 50, an f16 in two bytes, a reference's sign and magnitude in units of 2^-30, and a tolerance, each at most 2^30
    When the kernel reads the f16 exactly through Kernel.Scale and compares it with the reference in integers
    Then the answer is agrees exactly when the two lie within the tolerance, and disagrees otherwise
    And an infinity or a NaN is answered not finite, and the theorems of Proofs.Rotary state the distance and are pinned with their axioms
