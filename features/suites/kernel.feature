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
