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

  @EN-02 @build
  Scenario: A safetensors header is read into its tensors in file order, with every tensor byte range accepted by the kernel
    Given the fixture EN-02 needs
    When the suite exercises EN-02
    Then A safetensors header is read into its tensors in file order, with every tensor byte range accepted by the kernel

  @EN-03 @build
  Scenario: A model config is read into the preflight question: a named family is recognised, every missing required key is counted, and an unsupported activation is flagged
    Given the fixture EN-03 needs
    When the suite exercises EN-03
    Then A model config is read into the preflight question: a named family is recognised, every missing required key is counted, and an unsupported activation is flagged

  @FL-01 @build
  Scenario: The file has fewer than the eight bytes that carry the header length
    Given the input that makes model-reader return HeaderError::TooShortForLength
    When the suite exercises FL-01
    Then The file has fewer than the eight bytes that carry the header length

  @FL-02 @build
  Scenario: The header length the file declares does not fit inside the file
    Given the input that makes model-reader return HeaderError::HeaderExceedsFile
    When the suite exercises FL-02
    Then The header length the file declares does not fit inside the file

  @FL-03 @build
  Scenario: The header bytes are not a JSON object
    Given the input that makes model-reader return HeaderError::NotJson
    When the suite exercises FL-03
    Then The header bytes are not a JSON object

  @FL-04 @build
  Scenario: A header entry is not an object carrying a dtype, a shape of whole numbers and a pair of data offsets
    Given the input that makes model-reader return HeaderError::TensorNotDescribed
    When the suite exercises FL-04
    Then A header entry is not an object carrying a dtype, a shape of whole numbers and a pair of data offsets

  @FL-05 @build
  Scenario: A tensor ends before it starts
    Given the input that makes model-reader return HeaderError::OffsetsDoNotRunForward
    When the suite exercises FL-05
    Then A tensor ends before it starts

  @FL-06 @build
  Scenario: The kernel did not accept a tensor byte range
    Given the input that makes model-reader return HeaderError::RangeRefused
    When the suite exercises FL-06
    Then The kernel did not accept a tensor byte range

  @FL-07 @build
  Scenario: The config bytes are not a JSON object
    Given the input that makes model-reader return ConfigError::NotJson
    When the suite exercises FL-07
    Then The config bytes are not a JSON object
