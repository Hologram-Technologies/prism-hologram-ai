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
  Scenario: An ONNX model's files are imported as they arrive: every tensor named by the BLAKE3 of its own bytes, every span, operator and 4-bit weight put to the kernel, and each whole file to the BLAKE3 its record gives
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
  Scenario: The kernel refuses a whole file against the BLAKE3 and length its record gives
    Given the graph of EN-31 under a record that names another BLAKE3, and a kernel that refuses it
    When it is imported
    Then it returns ImportError::FileRefused with the file's name and the kernel's verdict
    And the question carried the record's BLAKE3 as expected and the file's own as observed

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

  @EN-41 @build
  Scenario: Every binary32 hologram-ai-exact rounds a row to is confirmed by the kernel: the compiled guest answers decision 42 exact for the crate's own rounding of each value and differs for any other bits
    Given values that are exact, ties broken both ways, the largest finite value and the carry past it, overflow, subnormals, underflow to a signed zero, full 64-bit magnitudes, zero, and draws
    When hologram-ai-exact places each in a row register and rounds it with round_to_binary32, and the kernel client writes decision 42's request for those bits and for those bits with the last flipped
    Then the request read as Kernel.Round reads it gives the crate's bits
    And claims.json lists the crate's bits as exact and the flipped bits as differs, and the app recipe's replay of the guest answers each as listed

  @EN-43 @build
  Scenario: The softmax exponent is base two: 256 table entries, each the integer nearest 2^(30 - j/256), certified exactly by (2T - 1)^256 < 2^(7936 - j) < (2T + 1)^256; exp2_neg(d) for d in units of 2^-8 shifts the fractional entry by d's whole part, rounded half to even, and is within one unit of 2^(30 - d/256) for every d below 7936
    Given the 256 entries of hologram-ai-exact's EXP2_FRACTION
    When each entry is checked against its exact criterion with integers of any width
    Then every entry meets it, which is to say it is within half a unit of 2^(30 - j/256), so it is the nearest integer
    And exp2_neg(d) is entry d mod 256 over 2^(d div 256), rounded half to even, worked out in the test without the crate
    And exp2_neg(d) is within one unit of 2^(30 - d/256) for every d below 7936, checked as (e - 1)^256 * 2^d < 2^7680 < (e + 1)^256 * 2^d
    And exp2_neg(d) is 0 for every d from 7936, where 2^(30 - d/256) is below half a unit

  @EN-44 @build
  Scenario: Softmax is integer: for scores in units of 2^-8 of a base-two exponent, each output is exp2_neg of the largest score less its own, times 2^30, over their exact sum, rounded half to even, in units of 2^-30; the outputs sum to 2^30 within half a unit each, a common shift of the scores changes nothing, and a larger score never gets a smaller output
    Given rows of 1 to 4,096 scores drawn by xorshift, with ties, one spike, spreads past 2^40 and the extremes of i64
    When each row is put through hologram-ai-exact's softmax_exp2
    Then every output p meets 2 |e * 2^30 - p * S| <= S for its weight e and the row's sum S, and an exact tie gives the even p
    And the outputs of a row sum to 2^30 within half a unit each
    And a row shifted by any amount gives the same outputs, and a larger score never gets a smaller output

  @EN-45 @build
  Scenario: Rotary positions take sine and cosine from one quarter-wave table: 16,385 entries, entry r the integer nearest 2^30 sin(pi r / 32768), generated by an exact integer algorithm, certified entry by entry with integers of any width and named by its BLAKE3; sine_turn and cosine_turn of an angle in units of 2^-32 of a turn round it half to even to 2^-16 of a turn and read the table by quadrant, so sine is odd, cosine is sine a quarter turn on, and sin^2 + cos^2 is 2^60 within 2^31
    Given the 16,385 entries of hologram-ai-exact's sine_quarter table
    When each entry is checked against bounds on 2^256 sin(pi r / 32768) worked out in the test, with pi from Machin's formula, in integers of any width
    Then every entry is within half a unit of 2^30 sin(pi r / 32768), so it is the nearest integer, and the table's BLAKE3 is the one pinned
    And sine_turn at each step of 2^-16 of a turn is the entry its quadrant names, a tie between steps goes to the even step, and sine_turn of a negated angle is negated
    And cosine_turn of an angle is sine_turn a quarter turn on, and at every step sin^2 + cos^2 is within 2^31 of 2^60

  @EN-46 @build
  Scenario: SiLU is integer: for x in units of 2^-16 below 2^47 in size, the exponent |x| log2(e) is taken in units of 2^-8 with log2(e) as the integer nearest 2^32 log2(e), certified exactly, rounded half to even; w is exp2_neg of it, and silu(x) is x 2^30 / (2^30 + w) for x at or above 0 and x w / (2^30 + w) below, rounded half to even, so silu(x) - silu(-x) is x within one unit, and it is within |x| / 2048 plus one unit of x / (1 + e^-x)
    Given hologram-ai-exact's LOG2_E and every x from -2^20 to 2^20 in steps of 7, with x drawn by xorshift up to 2^47 in size
    When each x is put through hologram-ai-exact's silu
    Then LOG2_E meets (2L - 1) ln 2 < 2^33 < (2L + 1) ln 2, with ln 2 worked out in the test in integers of any width
    And every output p meets 2 |n - p D| <= D for the numerator n and the denominator D worked out in the test, and an exact tie gives the even p
    And silu(x) - silu(-x) is x within one unit, and silu is within |x| / 2048 plus one unit of x / (1 + e^-x)

  @EN-47 @build
  Scenario: RMSNorm is integer: for x and the weight g in units of 2^-16 below 2^31 in size, at most 2^16 of them, and epsilon in units of 2^-32, the sum of squares plus n epsilon is exact, the root r is the largest integer with n r^2 <= (that sum) 2^32, which is the integer square root of the floored mean times 2^32 as decision 43 decides it, and each output is x g 2^16 / r rounded half to even, in units of 2^-16, within |y| / 2^20 plus one unit of x g / sqrt(mean(x^2) + epsilon); a row whose root is 0 or whose sizes are past the bounds is refused
    Given rows of 1 to 8,192 values and weights drawn by xorshift up to 2^31 in size, one spike, all equal, and epsilon from 0 to 2^32 - 1
    When each row is put through hologram-ai-exact's rms_norm
    Then r meets n r^2 <= v 2^32 < n (r + 1)^2 for v the sum of x^2 plus n epsilon, worked out in the test by bisection
    And every output y meets 2 |x g 2^16 - y r| <= r, an exact tie giving the even y, and is within |y| / 2^20 plus one unit of x g / sqrt(mean(x^2) + epsilon)
    And a row of zeros with epsilon 0, an empty row, a row and weights of different lengths, and a value at 2^31 are refused

  @EN-54 @build
  Scenario: hologram-ai-chat renders a conversation in the subset the engine defines as transformers renders the model's own chat template, for every recorded case of Qwen2.5-1.5B-Instruct, Qwen3-0.6B and Llama 3.2 1B Instruct the subset holds, and asks decision 46 the same conversation, whose claims the guest replays
    Given the recorded templates and cases of the three exports in crates/conformance/tests/gpu/templates, each rendered by Python's jinja2 as transformers configures it
    When hologram-ai-chat's template::render renders every case whose turns are system, user and assistant
    Then each text is the recorded one, byte for byte, and each case that fits one request is a decision-46 claim answered with it

  @FL-60 @build
  Scenario: A Qwen3 assistant turn carries </think> reasoning, which the defined subset leaves out, so the conversation is refused by name
    Given Qwen3-0.6B's recorded thinking case, whose assistant turns carry </think>
    When hologram-ai-chat's template::render renders it
    Then it is refused as TemplateError::ReasoningInTurn

  @EN-48 @build
  Scenario: Rotary angles are integer: each pair i of a head of size d has a frequency in units of 2^-64 of a turn per position, generated in integers and within 2^6 units of 2^64 theta^(-2i/d) / (2 pi), certified exactly; Llama 3 scaling keeps it, divides it by its factor or blends the two as the config names, rounded half to even; a position's angle is position times frequency modulo a turn, rounded half to even to 2^-32 of a turn, and its cosine and sine on EN-45's table are within 10^-4 of cos and sin of position theta^(-2i/d)
    Given the head sizes and thetas of Qwen3-0.6B, Qwen2.5-1.5B and Llama 3.2 1B, and 64 at 10,000
    When hologram-ai-exact's rope_frequencies generates each pair's frequency, with Llama 3 scaling where the config names it
    Then each unscaled frequency F meets ((F - 2^6) 2 pi)^b theta^a < 2^(64b) < ((F + 2^6) 2 pi)^b theta^a for 2i/d = a/b, with pi from Machin's formula in the test
    And each Llama 3 frequency is the test's own keep, divide or blend of the unscaled one, rounded half to even, and within 10^-12 of the float formula's
    And rope_angle is position times frequency modulo 2^64 rounded half to even to 2^-32 of a turn, and cosine_turn and sine_turn of it are within 10^-4 of cos and sin of position theta^(-2i/d)
    And an odd or zero head size, a theta below 2, and a Llama 3 scaling with high at or below low are refused

  @EN-49 @build
  Scenario: The CPU path is integer from token ids to logits: hologram-ai-cpu-decode reads a Qwen3-shaped model from its fused q4f16 ONNX export by the nodes that read each weight, holds the residual stream as binary32 bit patterns touched only by integer code, and runs every stage on the exact products and the integer stages EN-43 to EN-48, each result rounded once, half to even; on GP-19's fixture its greedy tokens after the prompt are the float reference's sixteen, every logit within 2^-7 of the reference's, and a second run gives the same bits
    Given GP-19's Qwen3-shaped model as its fused q4f16 export, its config's rope_theta and rms_norm_eps, and the float reference's tokens and logits after its prompt
    When hologram-ai-cpu-decode reads the graph and steps through the prompt and sixteen greedy tokens
    Then the sixteen tokens are the reference's, and every logit is within 2^-7 of the reference's
    And a second run from a new session gives the same logits, bit for bit

  @FL-55 @build
  Scenario: The graph has no node for a role the forward pass reads, or the weight that node reads is missing or not the shape the role needs
    Given the fixture's graph with layer 0's query norm and without layer 1's
    When it is read
    Then the read is refused as DecodeError::Unrecognised(Role::QueryNorm)

  @FL-56 @build
  Scenario: An exact stage refuses its operands, as a 4-bit matrix whose block has no exact codec
    Given the fixture's graph with layer 0's query projection given a block size of 8
    When it is read
    Then the read is refused as DecodeError::Exact(ExactError::NoExactCodec)

  @FL-57 @build
  Scenario: A row is past RMSNorm's bounds, or its root is 0, so the norm is refused rather than run approximately
    Given the fixture model read with rms_norm_eps 2, past RMSNorm's bound on epsilon of 1
    When one token is stepped
    Then the step is refused as DecodeError::NormRefused

  @FL-58 @build
  Scenario: A token id at or past the vocabulary is refused before any weight is read
    Given the fixture model
    When a token id equal to its vocabulary is stepped
    Then the step is refused as DecodeError::TokenOutOfRange

  @EN-50 @build
  Scenario: The integer CPU path reads Qwen2.5's decomposed ONNX export, norms as Mul_1 and query, key and value biases as Add, the head counts from the config: on a Qwen2-shaped fixture written as that export, its sixteen greedy tokens after the prompt are the float reference's, every logit within 2^-7, and a second run gives the same bits
    Given a Qwen2-shaped model with biases written as Qwen2.5's decomposed q4f16 export, and the float reference's tokens and logits after its prompt
    When hologram-ai-cpu-decode reads the graph with the config's head counts and steps through the prompt and sixteen greedy tokens
    Then the sixteen tokens are the reference's, every logit is within 2^-7 of the reference's, and a second run gives the same bits

  @EN-51 @build
  Scenario: The integer CPU path runs Llama's fused export with Llama 3 rotary scaling from its config (EN-48): on a Llama-shaped fixture with factor 4, low 1, high 4 and 64 original positions, its sixteen greedy tokens after the prompt are the float reference's, every logit within 2^-7, and a second run gives the same bits
    Given a Llama-shaped model written as Llama's fused q4f16 export with Llama 3 rope scaling in its config, and the float reference's tokens and logits after its prompt
    When hologram-ai-cpu-decode reads the graph and steps through the prompt and sixteen greedy tokens
    Then the sixteen tokens are the reference's, every logit is within 2^-7 of the reference's, and a second run gives the same bits

  @EN-52 @build
  Scenario: The page's module runs the integer forward pass: after it imports a graph, it is handed each weight's bytes once, opened with the config's head counts, rope theta, epsilon and scaling, and stepped a token at a time, and each step answers the greedy token and the logits, binary32 bits, equal bit for bit to hologram-ai-cpu-decode's run of the same model; a refusal of the forward pass is answered by its DecodeError name
    Given GP-19's Qwen3-shaped export imported through the page's module, and each of its weights handed to decode_tensor
    When decode_open is given its config and decode_step is called for the prompt and sixteen greedy tokens
    Then every step's greedy token and logits are hologram-ai-cpu-decode's for the same model, bit for bit

  @FL-59 @build
  Scenario: The integer forward pass refuses a step or a model, and the module answers the refusal by the name DecodeError gives it
    Given GP-19's Qwen3-shaped export opened in the page's module
    When decode_step is called with a token id equal to the vocabulary
    Then the call answers no, and the module's answer is Decode(TokenOutOfRange)

  @EN-53 @build
  Scenario: The page's module keeps and restores a run of the integer forward pass: decode_save keeps it as it was after its first n tokens, decode_restore carries on from there, and decode_reset starts again, each step after a restore or a reset answering the same token and logits, bit for bit, as the run that first computed it
    Given GP-19's Qwen3-shaped export opened in the page's module and stepped through its prompt and eight greedy tokens
    When the run is kept at the prompt's length, stepped on, restored, and stepped again with the same tokens, and then reset and stepped through the prompt again
    Then every step after the restore and after the reset answers the token and logits the first run did, bit for bit

  @EN-42 @build
  Scenario: The ONNX importer hashes each whole file by BLAKE3 as its bytes arrive and asks the kernel to admit it against the BLAKE3 the record gives, so a file whose bytes are not the record's BLAKE3 is refused
    Given an ONNX graph and its data file, and records giving each file's BLAKE3
    When the importer takes their bytes in pieces of any size and asks the kernel about each whole file
    Then each file's question carries the record's BLAKE3 and the BLAKE3 of the bytes that arrived, and the kernel admits the file
    And a record giving another BLAKE3 for the graph has the graph refused as FileRefused

  @EN-58 @build
  Scenario: No float lies on the integer path: the page module reads config.json numbers from their decimal text through hologram-ai-exact, each confirmed by decision 49 in the compiled guest and equal to a correctly rounded parse, and audit-float refuses a float type, literal or conversion in the integer path, a planted one in Rust and in the page script included
    Given config.json's text handed to the page's module, and the files the integer path is computed in
    When the module reads num_attention_heads, num_key_value_heads, rope_theta and rms_norm_eps from their decimal text through hologram-ai-exact
    Then each is the whole number or the nearest binary32 the text names, equal to a correctly rounded parse, and the compiled guest answers decision 49 exact for each and differs for any other
    And integer.mjs converts no number, and audit-float finds no float type, literal or conversion in crates/exact, crates/cpu-decode, crates/wasm or integer.mjs
    And a float planted in Rust or in the page script is refused by audit-float

  @FL-69 @build
  Scenario: A config number that is not a decimal, past 2^63, more than 72 places below the point, or not the whole number asked for is refused, and the model is not opened
    Given GP-19's Qwen3-shaped export imported in the page's module, with a config whose rms_norm_eps is 1e1000
    When decode_open is called with that config's text
    Then the call answers no, the module's answer is Decode(Exact(DecimalRefused)), and no model is opened
    And a whole number asked of 1.5 is ExactError::DecimalRefused

  @EN-59 @build
  Scenario: The loader picks the rotary parameters an export's own cos and sin cache agree with in integers: the page module compares each cache entry's f16 bits with the engine's own angles from its integer frequencies and quarter-wave table, each entry within 2^-9 as decision 50 confirms it, and no float decides it
    Given Llama 3.2 1B's own cos and sin cache rows at positions 1, 7, 1000 and 8191, and a config that names no scaling
    When the page module compares each entry's f16 bits with cosine_turn and sine_turn of the engine's own angle for each candidate
    Then Llama 3 scaling at factor 32 is chosen, factor 8 and plain frequencies are not, and the compiled guest answers decision 50 as the crate does for each entry asked
    And onnx-load.mjs asks the module, onnx-roles.mjs holds no float check, and audit-float has no open finding left

  @EN-60 @build
  Scenario: LayerNorm is integer: for x, the weight g and the bias b in units of 2^-16 below 2^31 in size, at most 2^16 of them, and epsilon in units of 2^-32, the centred values n x - sum(x) and the variance's numerator V = n sum(x^2) - sum(x)^2 are exact, the root r is the largest integer with n^2 r^2 <= (V + n^2 epsilon) 2^32, and each output is (n x - sum(x)) g 2^16 / (n r) rounded half to even, plus b, in units of 2^-16, within |y - b| / 2^20 plus one unit of (x - mean(x)) g / sqrt(var(x) + epsilon) + b; a row whose root is 0 or whose sizes are past the bounds is refused
    Given EN-47's rows of 1 to 8,192 values and weights drawn by xorshift up to 2^31 in size, one spike, all equal, and epsilon from 0 to 2^32 - 1, each with biases drawn the same way
    When each row is put through hologram-ai-exact's layer_norm
    Then r meets n^2 r^2 <= (V + n^2 epsilon) 2^32 < n^2 (r + 1)^2, worked out in the test by bisection
    And every output y meets 2 |(n x - sum(x)) g 2^16 - (y - b) n r| <= n r, an exact tie giving the even y - b, and is within |y - b| / 2^20 plus one unit of the LayerNorm of the real numbers
    And a row whose values are all equal with epsilon 0, an empty row, weights or biases of another length, a value at 2^31 and an epsilon at 2^32 are refused

  @EN-61 @build
  Scenario: GELU is integer: for x in units of 2^-16, gelu(x) is x at or above 8 and 0 at or below -8; between, with X = |x| and Y = X^2, erf(X / sqrt 2) is c X A / B for the positive series A = sum of Y^k / (2k+1)!! and B = sum of (Y / 2)^k / k!, each term cut down to 2^-40, and c, standing for sqrt(2 / pi), the integer square root of 2^121 / floor(2^40 pi) with pi by Machin's formula; the output (x + X erf) / 2 is rounded half to even once, in units of 2^-16, within one unit of x Phi(x), and gelu(x) - gelu(-x) is x within one unit, exactly x when x is even
    Given 343 inputs: every eighth from -9 to 9, 160 drawn by a seeded generator between -9 and 9, 24 out to 2^20 units, the edges at 8 and -8, and inputs out to 2^47
    When each is put through hologram-ai-exact's gelu
    Then each output is within one unit of x Phi(x) computed by Python's math.erfc, recorded in the test
    And an input at or above 8 gives itself and one at or below -8 gives 0
    And gelu(x) - gelu(-x) is x within one unit, and exactly x for every even x

  @EN-62 @build
  Scenario: An encoder is integer from token ids to a sentence embedding: hologram-ai-cpu-encode reads a BERT- or DistilBERT-shaped encoder from its weights by the roles the page places them in, holds the hidden states as binary32 bit patterns touched only by integer code, runs every stage on the exact products, softmax (EN-43, EN-44), LayerNorm (EN-60) and GELU (EN-61), each result rounded once, half to even, with every token attending to every other, and pools the tokens' sum to unit length by the integer square root; on GP-13's fixture every component is within 2^-14 of the float reference's embedding, the length within 2^-20 of 1, and a second run gives the same bits
    Given GP-13's BERT-shaped encoder, all-MiniLM-L6-v2's shape with q4f16 matrices and f16 tables, biases and norms, its weights by role, and the float reference's embeddings of its two sentences of 6 and 60 tokens
    When hologram-ai-cpu-encode reads the weights and embeds each sentence twice
    Then every component is within 2^-14 of the reference's and the embedding's length within 2^-20 of 1
    And the second run gives the same bits

  @EN-63 @build
  Scenario: A classifier is integer from token ids to label scores: the first token's hidden state through the dense pre-classifier and its bias, ReLU, and the dense classifier and its bias, each product and each bias rounded once, and the label the first largest score; on GP-14's fixture the label is the float reference's, every score within 2^-10 of the reference's, and a second run gives the same bits
    Given GP-14's DistilBERT-shaped classifier, SST-2's shape with two labels and an f16 dense head, and the float reference's scores for its two sentences
    When hologram-ai-cpu-encode reads the weights and classifies each sentence twice
    Then the label is the reference's and every score within 2^-10 of the reference's
    And the second run gives the same bits

  @FL-70 @build
  Scenario: A role the encoder reads is missing, or its weight is not the shape or format the role needs
    Given GP-13's encoder with its first layer's query matrix taken away, and with its first layer's feed-forward matrix given the wrong shape
    When hologram-ai-cpu-encode reads the weights
    Then each read is refused as Unrecognised, naming the attention and the feed-forward

  @FL-71 @build
  Scenario: An exact stage of the encoder refuses its operands, as a stored scale that is not a finite number
    Given GP-13's encoder with one scale of its first layer's query matrix stored as an f16 infinity
    When a sentence is embedded
    Then it is refused as Exact, naming ScaleNotFinite

  @FL-72 @build
  Scenario: A row of the encoder is past LayerNorm's bounds or its root is 0, or a pooled sentence has length 0
    Given GP-13's encoder with its word and position tables and its token-type row all zero, so every embedding row is equal
    When a sentence is embedded with epsilon 0
    Then it is refused as NormRefused at the embedding's LayerNorm

  @FL-73 @build
  Scenario: A sentence is empty, holds a token id past the vocabulary, or has more tokens than the position table
    Given GP-13's encoder, whose vocabulary is 320 and position table 64
    When an empty sentence, a sentence holding id 320, and one of 65 tokens are embedded
    Then each is refused as TokenOutOfRange

  @EN-64 @build
  Scenario: The page's module runs an encoder through hologram-ai-cpu-encode: given each weight by the role the page placed it in and the config's text, whose head and layer counts are whole numbers and whose LayerNorm epsilon is read from its decimal text in units of 2^-32, it answers a sentence's embedding, or a classifier's label scores and the first largest label chosen by comparing their bits as integers, bit for bit as hologram-ai-cpu-encode answers the same ids; and the ids and bits the page's journey must show (encoder-expected.json) are those
    Given GP-13 and GP-14's encoders (seed 5) and the WordPiece tokenizer and config.json their page fixtures carry
    When the page's module is handed every weight by its role and the config's text, and each fixture sentence's ids
    Then encode_embed answers the embedding and encode_classify the label and the scores, bit for bit as hologram-ai-cpu-encode answers the same ids
    And crates/conformance/tests/gpu/encoder-expected.json holds exactly those ids and bits, and an epsilon that is not a number is refused

  @FL-74 @build
  Scenario: An encoder could not be read from the weights the page handed its module, or could not run, and the page is told the encoder's own failure
    Given GP-13's encoder with its first layer's query projection left out
    When the page's module is asked to open it
    Then encode_open says no with Encode(Unrecognised(Attention)), the encoder's own failure, where the page reads answers

  @EN-90 @build
  Scenario: engine.wasm is byte-identical on every build: the pinned toolchain and target, codegen-units 1, path prefixes remapped, SOURCE_DATE_EPOCH fixed and every host flag cleared, so two builds in different directories with different cargo homes give the same bytes, and the build fails unless their BLAKE3 is the address the model seals
    Given rust-toolchain.toml naming one exact toolchain with wasm32-unknown-unknown, and the release profile with codegen-units 1
    When the engine recipe builds the page's module
    Then it builds with every rustflags source but its own cleared, the cargo registry and the workspace remapped to fixed prefixes, and SOURCE_DATE_EPOCH fixed
    And it fails, naming both addresses, unless the module's BLAKE3 is the address app/src/Kernel/Seal.lex.tex seals
    And the engine-twice recipe builds it twice, from two copies of the tree with two cargo homes of different lengths and a host flag in the environment of the second, and fails on any byte that differs
    And engine.yml runs engine-twice on every change to the engine's inputs, keyed by every one of them, and re-derives the address of a kept module before it is trusted

  @EN-91 @build
  Scenario: hologram-ai-cpu-decode's advance takes one token through the model as step does but computes no output head, and leaves the session exactly as step leaves it: on GP-19's Qwen3, Qwen2 and Llama fixtures, a prompt whose ids but the last are advanced leaves every key and value step leaves and gives step's sixteen tokens and every one of their logits bit for bit, and an id past the vocabulary is refused as step refuses it
    Given GP-19's Qwen3-, Qwen2- and Llama-shaped fixtures read by hologram-ai-cpu-decode
    When the prompt's ids but the last are advanced and the last is stepped, beside a run that steps every id
    Then the two sessions hold the same keys and values, bit for bit
    And the sixteen greedy tokens and every logit of each are the same bits
    And an id past the vocabulary is refused by advance as by step
