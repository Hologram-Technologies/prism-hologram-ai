Feature: kernel

  The Hologram AI kernel, in app/: every decision the application makes about
  what it may load and run. A request names one decision in its first byte and
  carries that decision's numbers; the answer is the verdict. The engine and the
  page obey the verdict and never decide for themselves.

  @KN-05 @build
  Scenario: A buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal
    Given the admission model in app/src/Kernel/Admission.lex.tex
    When the Lean kernel checks admitted_decides during prismpm verify
    Then admitted is true exactly when all four label words and the two lengths are equal
    And the verdict the application returns for decision 5 is decided by that same admitted

  @KN-06 @build
  Scenario: A tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file
    Given the range model in app/src/Kernel/Range.lex.tex
    When the Lean kernel checks accepted_decides and all_accepted_ordered during prismpm verify
    Then accepted is true exactly when the previous end is at most the offset and the offset plus the length is at most the file length
    And a list of ranges is all accepted exactly when each starts at or after the end of the one before it and ends within the file
    And the verdict the application returns for decision 6 is decided by that same accepted
