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

  @EN-04 @build
  Scenario: An ONNX model's default-domain opset, its nodes' operators in file order and the tensors its graph carries are read from the file, and fields the preflight does not need are stepped over
    Given the fixture EN-04 needs
    When the suite exercises EN-04
    Then An ONNX model's default-domain opset, its nodes' operators in file order and the tensors its graph carries are read from the file, and fields the preflight does not need are stepped over

  @EN-05 @build
  Scenario: The operators an ONNX model names that a builder does not implement are counted into the kernel's import question, and reported each once in the order they first appear
    Given the fixture EN-05 needs
    When the suite exercises EN-05
    Then The operators an ONNX model names that a builder does not implement are counted into the kernel's import question, and reported each once in the order they first appear

  @FL-08 @build
  Scenario: A field runs past the end of the file
    Given the input that makes onnx-preflight return OnnxError::Truncated
    When the suite exercises FL-08
    Then A field runs past the end of the file

  @FL-09 @build
  Scenario: A variable-length number does not end within the ten bytes that carry sixty-four bits
    Given the input that makes onnx-preflight return OnnxError::NumberTooLong
    When the suite exercises FL-09
    Then A variable-length number does not end within the ten bytes that carry sixty-four bits

  @FL-10 @build
  Scenario: A field says it is encoded a way the format does not define, so its end cannot be found
    Given the input that makes onnx-preflight return OnnxError::UnknownEncoding
    When the suite exercises FL-10
    Then A field says it is encoded a way the format does not define, so its end cannot be found

  @FL-11 @build
  Scenario: A name the preflight reads is not text
    Given the input that makes onnx-preflight return OnnxError::NameNotText
    When the suite exercises FL-11
    Then A name the preflight reads is not text

  @FL-12 @build
  Scenario: The model carries no graph
    Given the input that makes onnx-preflight return OnnxError::NoGraph
    When the suite exercises FL-12
    Then The model carries no graph

  @FL-13 @build
  Scenario: The model declares no opset for the default domain
    Given the input that makes onnx-preflight return OnnxError::NoDefaultOpset
    When the suite exercises FL-13
    Then The model declares no opset for the default domain

  @EN-06 @some-true
  Scenario: A buffer's label is the BLAKE3 hash of its bytes, as the published test vectors give it
    Given the fixture EN-06 needs
    When the suite exercises EN-06
    Then A buffer's label is the BLAKE3 hash of its bytes, as the published test vectors give it

  @EN-07 @build
  Scenario: A download's label and length do not depend on how its bytes were split into chunks
    Given the fixture EN-07 needs
    When the suite exercises EN-07
    Then A download's label and length do not depend on how its bytes were split into chunks

  @EN-08 @build
  Scenario: A download is admitted exactly when the kernel admits the label and length it arrived with against the ones the archive names, and a refusal carries the kernel's verdict
    Given the fixture EN-08 needs
    When the suite exercises EN-08
    Then A download is admitted exactly when the kernel admits the label and length it arrived with against the ones the archive names, and a refusal carries the kernel's verdict

  @FL-14 @build
  Scenario: A label an archive names is not sixty-four lower-case hexadecimal digits
    Given the input that makes download return DownloadError::LabelNotHex
    When the suite exercises FL-14
    Then A label an archive names is not sixty-four lower-case hexadecimal digits

  @FL-15 @build
  Scenario: The kernel did not admit the buffer that arrived
    Given the input that makes download return DownloadError::Refused
    When the suite exercises FL-15
    Then The kernel did not admit the buffer that arrived

  @EN-09 @build
  Scenario: A conversation takes the step the kernel gives and stays exactly where it was when the kernel refuses
    Given the fixture EN-09 needs
    When the suite exercises EN-09
    Then A conversation takes the step the kernel gives and stays exactly where it was when the kernel refuses

  @EN-10 @build
  Scenario: The trim a conversation proposes drops whole pairs from the oldest, the fewest that leave the rest fitting, and the totals it reports are the totals of those pairs
    Given the fixture EN-10 needs
    When the suite exercises EN-10
    Then The trim a conversation proposes drops whole pairs from the oldest, the fewest that leave the rest fitting, and the totals it reports are the totals of those pairs

  @EN-11 @build
  Scenario: A conversation forgets the pairs it proposed only when the kernel accepts the trim, and reads the kernel's bucket and budget as the numbers they are
    Given the fixture EN-11 needs
    When the suite exercises EN-11
    Then A conversation forgets the pairs it proposed only when the kernel accepts the trim, and reads the kernel's bucket and budget as the numbers they are

  @FL-16 @build
  Scenario: The kernel did not give a state for the step that was asked
    Given the input that makes chat return ChatError::StepRefused
    When the suite exercises FL-16
    Then The kernel did not give a state for the step that was asked

  @FL-17 @build
  Scenario: The kernel did not accept the trim the conversation proposed
    Given the input that makes chat return ChatError::TrimRefused
    When the suite exercises FL-17
    Then The kernel did not accept the trim the conversation proposed

  @FL-18 @build
  Scenario: The kernel gave no decode bucket for the sequence
    Given the input that makes chat return ChatError::NoBucket
    When the suite exercises FL-18
    Then The kernel gave no decode bucket for the sequence

  @FL-19 @build
  Scenario: The kernel gave no token budget for the turn
    Given the input that makes chat return ChatError::NoBudget
    When the suite exercises FL-19
    Then The kernel gave no token budget for the turn

  @EN-12 @build
  Scenario: Reading an ONNX model costs time in proportion to its nodes: twice the nodes take less than three times as long, and fifty thousand are read in under a second
    Given the fixture EN-12 needs
    When the suite exercises EN-12
    Then Reading an ONNX model costs time in proportion to its nodes: twice the nodes take less than three times as long, and fifty thousand are read in under a second

  @EN-13 @build
  Scenario: Sixteen megabytes of download are labelled at no less than two hundred megabytes a second
    Given the fixture EN-13 needs
    When the suite exercises EN-13
    Then Sixteen megabytes of download are labelled at no less than two hundred megabytes a second

  @EN-14 @build
  Scenario: The page asks the kernel what its form says: every field lies where the kernel's examples have it, every byte of a request is a field, and every value the examples use can be chosen
    Given the fixture EN-14 needs
    When the suite exercises EN-14
    Then The page asks the kernel what its form says: every field lies where the kernel's examples have it, every byte of a request is a field, and every value the examples use can be chosen

  @EN-15 @build
  Scenario: Every reader of a file someone else wrote answers, for any bytes at all: it never panics, it fails only as the register lists, and it answers the same bytes the same way
    Given the fixture EN-15 needs
    When the suite exercises EN-15
    Then Every reader of a file someone else wrote answers, for any bytes at all: it never panics, it fails only as the register lists, and it answers the same bytes the same way

  @EN-16 @build
  Scenario: A model's tokenizer.json is read only when it is byte-level byte-pair encoding, and a file that describes anything else is refused by what it describes
    Given the fixture EN-16 needs
    When the suite exercises EN-16
    Then A model's tokenizer.json is read only when it is byte-level byte-pair encoding, and a file that describes anything else is refused by what it describes

  @EN-17 @build
  Scenario: Text is encoded by splitting it with the model's own pattern, writing each piece in the byte alphabet, and applying the vocabulary's merges in rank order until none applies
    Given the fixture EN-17 needs
    When the suite exercises EN-17
    Then Text is encoded by splitting it with the model's own pattern, writing each piece in the byte alphabet, and applying the vocabulary's merges in rank order until none applies

  @EN-18 @build
  Scenario: Decoding what was encoded gives back the text, for any text at all, because every byte has a character and every character of the alphabet has a byte
    Given the fixture EN-18 needs
    When the suite exercises EN-18
    Then Decoding what was encoded gives back the text, for any text at all, because every byte has a character and every character of the alphabet has a byte

  @FL-20 @build
  Scenario: The tokeniser file is not a JSON object
    Given the input that makes tokenizer return TokeniserError::NotJson
    When the suite exercises FL-20
    Then The tokeniser file is not a JSON object

  @FL-21 @build
  Scenario: The file describes a tokeniser this crate does not follow
    Given the input that makes tokenizer return TokeniserError::Unfollowable
    When the suite exercises FL-21
    Then The file describes a tokeniser this crate does not follow

  @FL-22 @build
  Scenario: An id to decode is not in the vocabulary
    Given the input that makes tokenizer return TokeniserError::UnknownId
    When the suite exercises FL-22
    Then An id to decode is not in the vocabulary

  @FL-23 @build
  Scenario: A token holds a character the byte alphabet does not have
    Given the input that makes tokenizer return TokeniserError::TokenNotByteLevel
    When the suite exercises FL-23
    Then A token holds a character the byte alphabet does not have

  @EN-19 @build
  Scenario: A conversation counts a turn with the model own tokeniser, so what the kernel is asked about is the model tokens and not the caller word
    Given the fixture EN-19 needs
    When the suite exercises EN-19
    Then A conversation counts a turn with the model own tokeniser, so what the kernel is asked about is the model tokens and not the caller word
