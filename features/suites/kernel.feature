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
