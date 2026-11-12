Feature: engine

  The engine, in crates/: the code that does the work the kernel decides about.
  It is tested against the kernel's own examples and against authorities outside
  this repository. Nothing in it is proved.

  @EN-01 @build
  Scenario: The kernel client writes each request exactly as the kernel examples have it and reads every answer the kernel gives
    Given every example the kernel carries in app/src/Kernel.lex.tex
    When the kernel client writes each well-formed request again from the fields it carries, and reads each answer
    Then every request is reproduced byte for byte
    And every answer reads as a verdict the client knows, with no two answers reading as the same verdict
    And every verdict the client knows is given by some example
    And anything else reads as unrecognised
