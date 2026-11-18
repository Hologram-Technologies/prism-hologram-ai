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
  Scenario: The page asks the kernel what its form says: every field lies where the kernel reads it, every byte of a request is a field of the form it is in, a list is drawn as the groups the kernel indexes into, and every value the examples use can be chosen
    Given the fixture EN-14 needs
    When the suite exercises EN-14
    Then The page asks the kernel what its form says: every field lies where the kernel reads it, every byte of a request is a field of the form it is in, a list is drawn as the groups the kernel indexes into, and every value the examples use can be chosen

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

  @EN-20 @build
  Scenario: The ids this repository gives are the ids the authority's own tokeniser gives, for the same file and the same text
    Given the fixture EN-20 needs
    When the suite exercises EN-20
    Then The ids this repository gives are the ids the authority's own tokeniser gives, for the same file and the same text

  @EN-21 @build
  Scenario: A model's own GGUF header is read into a tokeniser, with nothing past the metadata read and a pre-tokeniser name this repository has no cited pattern for refused
    Given the fixture EN-21 needs
    When the suite exercises EN-21
    Then A model's own GGUF header is read into a tokeniser, with nothing past the metadata read and a pre-tokeniser name this repository has no cited pattern for refused

  @EN-22 @build
  Scenario: A vocabulary read from a GGUF header gives the same ids as the same vocabulary written as a tokenizer.json, which the authority's own tokeniser checks
    Given the fixture EN-22 needs
    When the suite exercises EN-22
    Then A vocabulary read from a GGUF header gives the same ids as the same vocabulary written as a tokenizer.json, which the authority's own tokeniser checks

  @FL-24 @build
  Scenario: The file does not begin with the four bytes that announce the format
    Given the input that makes tokenizer return TokeniserError::NotGguf
    When the suite exercises FL-24
    Then The file does not begin with the four bytes that announce the format

  @FL-25 @build
  Scenario: A field of the header runs past the end of what was given
    Given the input that makes tokenizer return TokeniserError::Truncated
    When the suite exercises FL-25
    Then A field of the header runs past the end of what was given

  @EN-23 @build
  Scenario: The engine runs in a browser: for the same bytes, the wasm module answers what the crates answer
    Given the fixture EN-23 needs
    When the suite exercises EN-23
    Then The engine runs in a browser: for the same bytes, the wasm module answers what the crates answer

  @FL-26 @build
  Scenario: The page names a buffer this module did not allocate
    Given the input that makes wasm return CallError::NoBuffer
    When the suite exercises FL-26
    Then The page names a buffer this module did not allocate

  @FL-27 @build
  Scenario: The bytes handed in where text was asked for are not UTF-8
    Given the input that makes wasm return CallError::NotText
    When the suite exercises FL-27
    Then The bytes handed in where text was asked for are not UTF-8

  @FL-28 @build
  Scenario: A tokeniser or a download is used before one has been started
    Given the input that makes wasm return CallError::NothingRead
    When the suite exercises FL-28
    Then A tokeniser or a download is used before one has been started

  @FL-29 @build
  Scenario: A label handed in is not thirty-two bytes
    Given the input that makes wasm return CallError::WrongSize
    When the suite exercises FL-29
    Then A label handed in is not thirty-two bytes

  @EN-24 @build
  Scenario: A download's root is the sha256 of its bytes as they arrive, and it reproduces the standard's own examples
    Given the fixture EN-24 needs
    When the suite exercises EN-24
    Then A download's root is the sha256 of its bytes as they arrive, and it reproduces the standard's own examples

  @FL-30 @build
  Scenario: The root a container names is not sixty-four lower-case hexadecimal digits
    Given the input that makes download return DownloadError::RootNotHex
    When the suite exercises FL-30
    Then The root a container names is not sixty-four lower-case hexadecimal digits

  @EN-25 @build
  Scenario: A weight container's header is read as the published file is written, and a container this engine cannot follow is refused by name
    Given the fixture EN-25 needs
    When the suite exercises EN-25
    Then A weight container's header is read as the published file is written, and a container this engine cannot follow is refused by name

  @FL-31 @build
  Scenario: A container does not begin with the magic this reader follows
    Given the input that makes model-reader return HolkError::NotHolk
    When the suite exercises FL-31
    Then A container does not begin with the magic this reader follows

  @FL-32 @build
  Scenario: A container's header runs past the end of the bytes given
    Given the input that makes model-reader return HolkError::Truncated
    When the suite exercises FL-32
    Then A container's header runs past the end of the bytes given

  @FL-33 @build
  Scenario: A container's header is not JSON
    Given the input that makes model-reader return HolkError::NotJson
    When the suite exercises FL-33
    Then A container's header is not JSON

  @FL-34 @build
  Scenario: A container's header is padded with something other than spaces
    Given the input that makes model-reader return HolkError::PaddedWithSomethingElse
    When the suite exercises FL-34
    Then A container's header is padded with something other than spaces

  @FL-35 @build
  Scenario: A container describes something this reader does not follow
    Given the input that makes model-reader return HolkError::Unfollowable
    When the suite exercises FL-35
    Then A container describes something this reader does not follow

  @EN-26 @build
  Scenario: The page is published with the engine it loads, built from this workspace and under the size it is allowed
    Given the fixture EN-26 needs
    When the suite exercises EN-26
    Then The page is published with the engine it loads, built from this workspace and under the size it is allowed

  @EN-27 @build
  Scenario: A GGUF file's tensor table is read as the file writes it: every tensor's type, shape and where its bytes lie
    Given the fixture EN-27 needs
    When the suite exercises EN-27
    Then A GGUF file's tensor table is read as the file writes it: every tensor's type, shape and where its bytes lie

  @FL-36 @build
  Scenario: A file handed to the tensor reader does not begin with the GGUF magic
    Given the input that makes model-reader return GgufError::NotGguf
    When the suite exercises FL-36
    Then A file handed to the tensor reader does not begin with the GGUF magic

  @FL-37 @build
  Scenario: A GGUF file is of a version this reader does not follow
    Given the input that makes model-reader return GgufError::VersionNotKnown
    When the suite exercises FL-37
    Then A GGUF file is of a version this reader does not follow

  @FL-38 @build
  Scenario: A GGUF file's header runs past the bytes given
    Given the input that makes model-reader return GgufError::Truncated
    When the suite exercises FL-38
    Then A GGUF file's header runs past the bytes given

  @FL-39 @build
  Scenario: A tensor is stored in a type this reader does not follow
    Given the input that makes model-reader return GgufError::TypeNotFollowed
    When the suite exercises FL-39
    Then A tensor is stored in a type this reader does not follow

  @FL-40 @build
  Scenario: A tensor's shape is one this reader cannot measure
    Given the input that makes model-reader return GgufError::ShapeNotFollowed
    When the suite exercises FL-40
    Then A tensor's shape is one this reader cannot measure

  @EN-28 @build
  Scenario: The id a model stops on is the one the model's own file names, not a list written beside it
    Given the fixture EN-28 needs
    When the suite exercises EN-28
    Then The id a model stops on is the one the model's own file names, not a list written beside it

  @EN-29 @build
  Scenario: An ONNX file read as its bytes arrive gives its opsets, its nodes and its tensor table, the same for every way the bytes are split, and never holds a tensor's data
    Given an ONNX file shaped as the exports are, the graph before the opsets and nodes before tensors, with a megabyte of weights inside the file, a tensor kept in another file, one written as numbers, a negative attribute and fields no reader needs
    When it is fed to the streaming reader whole, and in pieces of 1, 2, 3, 7, 61, 64 and 4096 bytes asking each time for the byte the reader names
    Then every way gives the same opsets, nodes with their inputs, outputs and attributes, tensors with their names, shapes, types and where their data lies, and the span of each tensor's data inside the file as the data begins
    And the megabyte of weights is never fed, and the reader never holds more than the largest node and one piece
    And a caller that leaves the graph at its first tensor gets the nodes and the opsets
    And the whole-file reader reads the same operators and tensor count
    And a file that ends early is truncated wherever it ends, including inside data the reader stepped over

  @FL-41 @build
  Scenario: A node or an opset record is larger than the reader will hold
    Given a node, an opset record or a tensor's name one byte longer than the sixteen mebibytes the reader holds
    When its length is read
    Then the reader returns OnnxError::RecordTooLarge before any of it is fed
    And one exactly at the limit is waited for, not refused

  @FL-42 @build
  Scenario: A tensor kept outside the file names no location, or an offset or a length that is not a number
    Given a tensor marked as kept outside the file with no location, no entries, an offset that is a word, a negative length, or an offset past sixty-four bits
    When the reader reaches the end of its record
    Then it returns OnnxError::ExternalNotFollowed
    And a tensor that names a location alone is followed, its offset the start of that file

  @EN-30 @build
  Scenario: Each request the engine writes for decisions 18 to 28 puts every field at the offset and width the kernel reads it from, and every answer the kernel can give reads as a verdict
    Given the request table the page generator derives from the kernel's own answer functions, in site/decisions.json
    When the engine's client writes a request for each of decisions 18 to 28, every field a value that fills its width and differs from the others
    Then each request is the length the kernel reads, names its decision in its first byte, and carries each value at the offset and width the kernel reads it from
    And seventeen stop ids are written as a request the kernel refuses, not cut to sixteen
    And every answer text in each decision's kernel module reads as a verdict, and text the kernel never gives does not

  @EN-31 @build
  Scenario: An ONNX model's files are imported as they arrive: every tensor named by the BLAKE3 of its own bytes, every span, operator and 4-bit weight put to the kernel, and each whole file to the SHA-256 its record gives
    Given a fused-style model of one 4-bit projection and a multiply, its codes and scales inside the graph file and two tensors in a data file beside it, and each file's record
    When it is imported fed in pieces of 1, 7, 64 and 4096 bytes and whole, against a kernel answering in its own words
    Then every tensor whose data is bytes is listed with its file, offset and length and named by the BLAKE3 of exactly those bytes
    And every way of feeding imports the same model and asks the kernel the same questions in the same order
    And the kernel is asked each inline span as the graph passes it, the graph file whole, each operator in the fused style, the data file's spans before any of its bytes, that file whole, and each 4-bit weight's codes and scales against its shape
    And a graph that names no fused operator is asked in the decomposed style

  @FL-43 @build
  Scenario: The graph is not an ONNX model the streaming reader follows
    Given a graph file cut short inside a tensor's data, and one with a field that runs past its record
    When each is imported
    Then the first returns ImportError::Unreadable with OnnxError::Truncated when the file ends, and the second as the field is met

  @FL-44 @build
  Scenario: The kernel refuses a tensor's span in its file
    Given the model of EN-31, and a kernel that rejects the first inline span, or calls the first data-file span undeclared
    When it is imported
    Then it returns ImportError::SpanRefused with that tensor's name and the kernel's own verdict

  @FL-45 @build
  Scenario: The kernel refuses a whole file against the SHA-256 and length its record gives
    Given the graph of EN-31 under a record that names another SHA-256, and a kernel that refuses it
    When it is imported
    Then it returns ImportError::FileRefused with the file's name and the kernel's verdict
    And the question carried the record's SHA-256 as expected and the file's own as observed

  @FL-46 @build
  Scenario: The kernel refuses an operator the graph names
    Given the model of EN-31 and a kernel that answers unsupported operator for Mul
    When it is imported
    Then it returns ImportError::OperatorRefused for Mul with the kernel's verdict

  @FL-47 @build
  Scenario: A 4-bit weight's codes or scales are not the length the kernel gives for its shape
    Given the model of EN-31 and a kernel that gives 1,000 bytes for its codes, or bad shape for its scales
    When it is imported
    Then it returns ImportError::WeightRefused with the tensor, its length and the kernel's answer

  @FL-48 @build
  Scenario: A tensor's data lies in a file the import is not given, or past the end of the one it is
    Given the model of EN-31, imported from its graph alone, and again with its data file cut short and admitted anyway
    When the import finishes
    Then it returns ImportError::DataMissing for the first tensor in the data file, and for the span that never ended

  @EN-32 @build
  Scenario: A WordPiece tokenizer.json, BERT's normaliser and pre-tokeniser with one sentence between two special tokens, gives the same ids as the authority's own tokeniser, and a WordPiece file that asks for any other step is refused by the step it asks for
    Given five WordPiece files in BERT's shape written out in the test: all-MiniLM-L6-v2's settings with its truncation at 128 and padding to 128, DistilBERT's with neither, and three with the normaliser's cleaning, Chinese spacing, accent stripping and lowercasing each turned the other way, one truncating at 9
    And 27 written texts and 400 generated from controls, format characters, accents, combining marks, CJK, odd spaces, punctuation, letters whose lowercase is two characters, private use and the special tokens themselves
    When crates/tokenizer reads each file as WordPiece and encodes each text, beside the authority's own tokeniser with its padding set aside
    Then every text gives the authority's ids, the unknown token among them more than 100 times
    And a model other than WordPiece, a normaliser other than BERT's or with a setting not a flag, a pre-tokeniser other than BERT's, a template other than one sentence between two special tokens of type 0, truncation from the left or with a stride, an added token normalised or stripping space, an unknown token not in the vocabulary, and a file that is not a JSON object are each refused by what they ask for
    And the byte-pair reader still refuses a WordPiece file, as EN-16 states

  @EN-33 @build
  Scenario: An ONNX export is imported through the page's module: every span, operator, file and 4-bit weight is put to the kernel the page runs, through the module's one import, exactly as the importer asks natively, and the model comes back as the nodes, tensors and places the role map reads, each tensor named by the BLAKE3 of its bytes
    Given EN-31's model, a 4-bit projection and a multiply named as an export names them, its codes and scales in the graph and two tensors in a data file beside it, and a kernel that answers as the kernel does
    When the page's module imports it through onnx_start, onnx_take, onnx_graph_finish, onnx_data_start, onnx_data_take, onnx_data_finish and onnx_finish, in pieces of 7 bytes, 4096 and the whole graph, asking the kernel through its one import
    Then every call says yes, the graph names its data file, and the kernel is asked exactly the questions, in the order, that the importer asks natively
    And the model comes back as JSON with its nodes in file order, each with its name, its operator, its inputs that are tensors of the model (others null) and its attributes
    And every tensor's type and shape, and its file, offset and length, where exactly its bytes lie, named by the BLAKE3 of those bytes

  @FL-49 @build
  Scenario: The ONNX importer refused while the page's module was importing, and the page is told the importer's own failure
    Given EN-31's model and a kernel that answers every span "outside the file"
    When the page's module imports it
    Then onnx_take says no at the first inline tensor and leaves Import(SpanRefused(TensorAt(0), ...)), the importer's own failure, where the page reads answers
    And onnx_finish with nothing imported says no with NothingRead

  @EN-34 @build
  Scenario: A q4 or ternary matrix times a vector is computed in integers from its stored bytes: every scale and input is read as the exact integer times a power of two it stands for, each group sum equals the same sum written out in i64, each row is the exact sum of its groups rounded once to the nearest f32, the same whatever order its groups are in, and the page's module gives the same bytes as native code
    Given q4 matrices with f32 and with f16 scales and ternary matrices, of one to seven rows and one to eight scale groups, with weights decoded independently from each layout's definition
    And inputs that are ordinary, one spike among small values, near 1e-30, near 1e30, halfway between codes, at the edge of a step, and all zero
    And scales that are subnormal halves, far apart, cancelling, or past f32's range before the input's power of two
    When each is taken by hologram-ai-exact natively and by the exact_q4, exact_q4f16 and exact_ternary calls of the module the page loads
    Then every input code lies within 127 and is its input rounded to the nearest, ties to even, at the largest step the limit allows
    And every group sum equals the same sum of the codes times the weights written out in i64
    And every output's bits are the exact sum of its group sums times their scales, worked here from each scale's own bits, rounded once to the nearest f32, ties to even
    And a q4 row gives the same bits with its groups in reverse order
    And every output lies within the inputs' half steps through their weights and scales of the float product
    And the crate's source names no f64
    And the module answers every case with the crate's bytes, and refuses a misaligned row as GroupMisaligned

  @FL-50 @build
  Scenario: An input to an exact product is not a finite number
    Given an input of 32 values with one NaN, infinity or negative infinity among them
    When it is quantised, or a q4 product of it is asked for
    Then each returns ExactError::InputNotFinite

  @FL-51 @build
  Scenario: An exact product's row length is not a whole number of scale groups
    Given a q4 row of 48 weights and a ternary row of 64
    When their group sums are asked for
    Then each returns ExactError::GroupMisaligned

  @FL-52 @build
  Scenario: A buffer handed to an exact product is not the length its shape implies
    Given q4 codes one byte short, a missing scale, an input one value short, a ternary plane one byte short, and requests to the module too short for their counts or with counts that overflow
    When each product is asked for
    Then each returns ExactError::LengthMismatch

  @FL-53 @build
  Scenario: A scale or weight stored for an exact product is not a finite number
    Given a q4 matrix whose f32 scale is infinity or NaN, one whose half scale is infinity or NaN, and a ternary matrix whose scale is NaN
    When each product is asked for
    Then each returns ExactError::ScaleNotFinite

  @EN-35 @build
  Scenario: Every group sum hologram-ai-exact takes of a q4 block is confirmed by the kernel: the compiled guest answers decision 34 exact for the crate's own sum of each engine-stored block, and differs for that sum plus one
    Given q4 blocks as the engine stores them, at both ends of the sum's range, with every weight zero, and drawn at random, each with an activation of magnitude 127
    When hologram-ai-exact quantises each block's activations and takes its group sum, and the kernel client writes decision 34's request for that sum and for that sum plus one
    Then the request read as Kernel.Block reads it gives the crate's sum and the claim written
    And claims.json lists the crate's sum as exact and the sum plus one as differs
    And the app recipe runs every claim through the compiled guest of the holo it built, which answers each as listed

  @EN-36 @build
  Scenario: Every group sum hologram-ai-exact scales by an IEEE half is confirmed by the kernel: the compiled guest answers decision 37 exact for the crate's own product of each sum and half, and differs for that product plus one
    Given group sums at both ends of a block's range and zero, and halves that are one, the largest, the smallest subnormal, the smallest normal, negative and negative zero
    When hologram-ai-exact multiplies each sum by its half as an exact count of 2^-24, and the kernel client writes decision 37's request for that product and for that product plus one
    Then the request read as Kernel.Scale reads it gives the crate's product and the claim written
    And claims.json lists the crate's product as exact and the product plus one as differs, and the app recipe's replay of the guest answers each as listed
    And a half that is an infinity or a NaN has no product in the crate

  @FL-54 @build
  Scenario: A stored format has no exact codec, so it is refused by name instead of run approximately
    Given MatMulNBits at eight or two bits, at a block of 24 or 512, with bfloat16 scales or half zero points, and initializers of integer or double type
    When their codec is asked for, or a product at a block of 24
    Then each returns ExactError::NoExactCodec

  @EN-37 @build
  Scenario: An input is put on eight- or sixteen-bit integers per vector or per group of 32, each group at the largest power of two that keeps it within its limit and each value rounded to the nearest, ties to even, and every q4, ternary and dense product under each scheme is the exact sum at each group's own power, rounded once to binary32, with eight bits per vector giving EN-34's and EN-40's bits
    Given inputs of 1 to 256 values with one spike among small ones, and the four schemes: eight or sixteen bits, per vector or per group of 32
    When each is put on integers by quantise_with, and multiplied by MatMulNBits matrices at every block, dense half and binary32 weights, and a ternary matrix
    Then every code is its input rounded to the nearest at its group's step, the largest step within the width's limit, and a spike sets the step only for its own group
    And every output's bits are the exact sum of each code times its weight and scale at its group's power, worked in the test and rounded once there
    And eight bits per vector gives the same bits as EN-34's and EN-40's products

  @EN-38 @build
  Scenario: A row's exact terms are chained into decision 40's 79-byte requests, six terms a link, each link's partial the exact sum of the terms before it, and the chain's last partial is the row value the fast path places, rounded once to the same binary32 bits; a row too wide for the guest's 64-bit numbers is reported, not chained
    Given MatMulNBits matrices at every block, under each activation scheme, with inputs that have a spike
    When each row's terms are taken by q4_row_terms and chained by row_links
    Then every request is 79 bytes, starts with decision 40, and holds each link's partials and terms as decision 40 reads them
    And each link's next partial is its partial plus its terms, and the first partial is zero
    And the last partial, rounded once at the chain's power, gives the bits q4_product_with gives
    And a term whose value could reach 2^58 at the chain's power makes the row too wide

  @EN-39 @build
  Scenario: An input put on sixteen-bit codes per group of 32 is written as decision 41's 62-byte requests, five inputs a request with their group's largest input, and every request satisfies decision 41's definition as transcribed from Kernel.Requant: the power is the group's largest, each quotient witness is the input divided down, each code is the one rqDownCode computes, and the codes are quantise_with's
    Given inputs that are ordinary, exactly halfway between codes, below the normal range, zero, near the largest binary32, and spikes beside values too small to reach a code
    When each is put on sixteen-bit codes per group of 32 and written by requant_requests
    Then every request is 62 bytes and decision 41's, and names its group's largest input at the group's power
    And every input in it passes rqElementOk and the power passes rqLargest, as transcribed in the test from Kernel.Requant
    And every code is rqDownCode's for its input and step, and the codes are quantise_with's, in order

  @EN-40 @build
  Scenario: A four-bit MatMulNBits matrix at any block from 16 to 256, its last block padded, with half or binary32 scales and zero points absent or packed, and a dense binary32, half or bfloat16 MatMul weight, are each multiplied by a vector in integers from their stored bytes, every block sum equal to the sum written out from the operator's definition and every output the exact value rounded once to binary32
    Given MatMulNBits matrices at blocks of 16, 32, 64, 128 and 256, with a whole number of blocks and with the last block padded, half and binary32 scales of either sign, and zero points left out or packed four bits to a block
    And dense MatMul weights stored as binary32, half and bfloat16
    When each is multiplied by a vector by hologram-ai-exact
    Then every block sum equals the sum of each input code times its code less its block's zero point, written out from the operator's definition
    And every output's bits are the exact sum worked from each scale's or weight's own bits, rounded once to binary32, ties to even
    And zero points stored as eights give the same bits as zero points left out, and block 32 with none gives EN-34's q4 bits
