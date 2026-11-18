# Engine verification

What each gate over the engine has been seen to reject. A gate is evidence only
after a planted defect shows it can fail ([AGENTS.md](AGENTS.md)); this file
records the defect, the command, the diagnostic and the restored result.

The engine is tested, never proved. What is proved is in
[KERNEL-VERIFICATION.md](KERNEL-VERIFICATION.md).

## What the engine is, so far

`crates/kernel-client` writes the kernel's requests and reads its answers. It
has no dependencies, uses no allocation and no unsafe code, and no function in
it can fail: a request is built from typed arguments into a fixed-size array,
and an answer the kernel is not known to give reads as `Unrecognised`, which a
caller must handle and cannot mistake for a verdict.

`crates/tokenizer` turns text into the tokens a model counts in, from the
model's own `tokenizer.json`. It invents nothing: the vocabulary, the merges
and the pattern that splits text into pieces all come from that file, and a
file it cannot follow is refused by what it describes rather than followed
approximately.

`crates/chat` keeps a conversation: where it is in the journey, how many
tokens its system prompt and newest message hold, and each question-and-answer
pair in turn. It counts them with the model's own tokeniser, so the numbers the
kernel's arithmetic is about are the model's tokens. It proposes a trim, asks for a decode bucket and a token budget,
and takes the step the kernel gives.

`crates/download` takes a download's bytes as they arrive, labels them with the
BLAKE3 hash the archive names them by, and asks the kernel whether the buffer
may be used. It fetches nothing: whoever holds the connection hands the chunks
in, which is how it works in a browser, where fetching belongs to the page.

`crates/onnx-preflight` reads an ONNX model's opset and operators out of the
file and puts the kernel's import question. An ONNX file is a protocol-buffer
message; this crate reads the handful of fields a preflight needs and steps over
everything else, so a file written against a newer schema still reads. It builds
no graph, so it generates no parser for the whole schema and has no dependency
beyond the kernel client.

`crates/model-reader` reads the two documents a model carries: the safetensors
header, which says where each tensor's bytes lie, and the config, which says
what kind of model it is. Both are untrusted bytes, so this is the first crate
here that can fail, and the first with a dependency: `serde_json` parses them.
It counts and it measures; it decides nothing. Whether a tensor's bytes lie
where they claim, and whether a model may run, are put to the kernel.

Nothing that downloads, compiles or runs a model is here yet. That code needs
the hologram runtime, and rule R6 admits only crates.io dependencies; the
runtime's crates are not published there.

## What is tested

| Claim | Evidence | Kind |
| --- | --- | --- |
| `EN-01`: the kernel client writes each request exactly as the kernel's examples have it and reads every answer the kernel gives | `engine_en_01` in `crates/conformance/tests/engine.rs`, over every example in `app/src/Kernel.lex.tex` | tested |
| A graph may be built on one verdict of decision 10 and on nothing else | `engine_en_01`, over every verdict the client knows, including the ones it reads as unrecognised | tested |
| A number in an answer is read only when it is written as the kernel writes one | `decimal_reads_canonical_numbers` and `decimal_refuses_what_the_kernel_never_writes` in `crates/kernel-client/src/lib.rs` | tested |
| The client accepts no answer text beyond what the kernel gives | not tested: the test reads the answers the examples give, and cannot enumerate text no example contains | not tested |
| `EN-02`: a safetensors header is read into its tensors in file order, with every tensor byte range accepted by the kernel | `engine_en_02`, over files built byte for byte in the test | tested |
| `EN-03`: a model config is read into the preflight question | `engine_en_03`, over a whole config and every way of spoiling it this crate counts | tested |
| `FL-01` to `FL-07`: the reader fails only as `model/failures.toml` lists | `engine_fl_01` to `engine_fl_07` | tested |
| A tensor's length agrees with its shape and its dtype | not tested: the reader records what the header says and does not multiply it out | not tested |
| `EN-04`: an ONNX model's opset, operators and tensor count are read from the file | `engine_en_04`, over files written out field by field in the test | tested |
| `EN-05`: the operators a builder does not implement are counted into the kernel's question | `engine_en_05` | tested |
| `FL-08` to `FL-13`: the preflight fails only as `model/failures.toml` lists | `engine_fl_08` to `engine_fl_13` | tested |
| `EN-15`: every reader of a file someone else wrote never panics, fails only as the register lists, and answers the same bytes the same way | `engine_en_15`, over 4,369 files from one seed | tested |
| An ONNX file from a hub reads as its publisher meant it | not tested: every file in this suite is written here | not tested |
| `EN-06`: a buffer's label is the BLAKE3 hash of its bytes | `engine_en_06`, over all 35 published test vectors, whose vendored file is checked against the checksum in `model/authorities.toml` first | reproduced from the authority, not established here |
| `EN-07`: a download's label and length do not depend on how its bytes were split | `engine_en_07`, over ten lengths either side of the hash's own block and chunk sizes, in six chunk sizes each | tested |
| `EN-08`: a download is admitted exactly when the kernel admits it | `engine_en_08` | tested |
| `FL-14`, `FL-15`: the download fails only as `model/failures.toml` lists | `engine_fl_14` and `engine_fl_15` | tested |
| `EN-09`: a conversation takes the step the kernel gives and stays where it was when refused | `engine_en_09`, driving the whole journey through the kernel's own examples | tested |
| `EN-10`: the trim proposed drops whole pairs from the oldest, the fewest that fit | `engine_en_10`, over twelve conversations including the empty one and the one too long to fit at all | tested |
| `EN-11`: pairs are forgotten only on the kernel's acceptance, and its numbers are read as numbers | `engine_en_11` | tested |
| `FL-16` to `FL-19`: the conversation fails only as `model/failures.toml` lists | `engine_fl_16` to `engine_fl_19` | tested |
| `EN-16`: a tokeniser file is read only when it is byte-level byte-pair encoding, and anything else is refused by what it describes | `engine_en_16` | tested |
| `EN-17`: text is split by the model's own pattern and merged in rank order | `engine_en_17` | tested |
| `EN-18`: decoding what was encoded gives back the text, for any text | `engine_en_18`, over thirteen texts and three hundred generated ones | tested |
| `FL-20` to `FL-23`: the tokeniser fails only as `model/failures.toml` lists | `engine_fl_20` to `engine_fl_23` | tested |
| `EN-20`: the ids are the ids HuggingFace's own `tokenizers` gives, for the same file and the same text | `engine_en_20`, over 1,526 texts and seven files | tested against the authority |
| `EN-21`: a model's own GGUF header is read into a tokeniser, and a pre-tokeniser name with no cited pattern is refused | `engine_en_21` | tested |
| `EN-22`: a vocabulary read from a GGUF header gives the same ids as the same vocabulary written as a `tokenizer.json` | `engine_en_22`, with the authority checking the JSON side | tested |
| `FL-24`, `FL-25`: the GGUF reader fails only as `model/failures.toml` lists | `engine_fl_24` and `engine_fl_25` | tested |
| `EN-32`: a WordPiece `tokenizer.json` in BERT's shape gives the ids HuggingFace's own `tokenizers` gives, and a WordPiece file asking for another step is refused by the step | `engine_en_32`, over 2,135 texts and five files, the unknown token reached 13,820 times; eleven refusals | tested against the authority |
| `EN-33`: an ONNX export is imported through the page's module, every question put to the kernel the page runs through one import exactly as the importer asks natively, and the model handed back as the nodes, tensors and places the role map reads | `engine_en_33`, EN-31's model in pieces of 7, 4096 and the whole graph; and once, locally, Qwen3-0.6B's real q4f16 export through the release module and the published kernel | tested; real export measured |
| `EN-35`: every group sum `hologram-ai-exact` takes of a q4 block is one the compiled guest answers `exact` for decision 34 | `engine_en_35`, eight blocks and 17 claims; `replay.mjs` in the `app` recipe | tested here; replayed in main's published guest; replayed in this holo's guest on CI |
| `EN-36`: every group sum `hologram-ai-exact` scales by an IEEE half is one the compiled guest answers `exact` for decision 37 | `engine_en_36`, nine groups and 20 claims; `replay.mjs` in the `app` recipe | tested here; replayed in this holo's guest on CI |
| `FL-49`: the importer's refusal reaches the page as the importer's own failure | `engine_fl_49` | tested |
| That a published model's own file is read | not tested: every file here is built in the suite; none comes from a hub |
| `EN-19`: a conversation counts a turn with the model's own tokeniser | `engine_en_19` | tested |
| That BLAKE3 is what its paper says it is | cited, never established here: the hash is the authority's own implementation, checked against the authority's own vectors | not proved |
| `EN-12`: reading an ONNX model costs time in proportion to its nodes, and fifty thousand are read in under a second | `engine_en_12`, timing 25,000 nodes and 50,000 and comparing | tested |
| `EN-13`: sixteen megabytes are labelled at no less than two hundred megabytes a second | `engine_en_13` | tested |
| How long any of this takes on the machine a reader has | not established: the numbers recorded here are this machine's, and the gates are floors far below them |
| A config names a model that will in fact run | not tested, and not this crate's to say: the counts go to the kernel, which answers | not tested |

## EN-01, the kernel client

The test reads the kernel's examples from the model itself, so an example added
or changed there is tested the next time the suite runs. For each example it
does two things.

Writing. When the request is a well-formed request of its decision, the test
takes the fields out of the request's own bytes, hands them to the client, and
requires the bytes the client writes to equal the request. A field at the wrong
offset, in the wrong order, at the wrong width or in the wrong byte order cannot
survive that. Every decision must have at least five requests reproduced.

Reading. The answer must read as a verdict the client knows, and never as
`Unrecognised`. Two different answers may not read as the same verdict. A
numeric answer must read as the number whose decimal text it is. And every
verdict the client knows must be given by some example, so the client cannot
carry a verdict the kernel never gives.

Then the negative cases: wrong case, trailing bytes, a leading zero, a sign and
a number one past 64 bits all read as `Unrecognised` for every decision.

### The gate: `cargo test -p hologram-ai-kernel-client -p repo-conformance`

| Planted defect | Result |
| --- | --- |
| The trim request writes the dropped total and the newest dropped pair in each other's place | rejected: `the client writes decision 7's request differently from the kernel's example` |
| The bucket request names decision 9 | rejected: the same message, for decision 8 |
| The admission request writes the observed label before the expected one | rejected: the same message, for decision 5 |
| A number is written little-endian | rejected: the same message, for decision 3, the first decision the test reaches |
| The reset event is written with the fail event's code | rejected: the same message, for decision 1 |
| The preflight request drops the manifest flag | rejected: the same message, for decision 2 |
| `derived` is expected with a capital letter | rejected: `decision 2 answered derived, which the client reads as Unrecognised` |
| `reject` reads as accept | rejected: `decision 6: two different answers both read as Accept` |
| The `stage too small` answer is not known | rejected: `decision 3 answered stage too small, which the client reads as Unrecognised` |
| A number with a leading zero is read as a number | rejected: the unrecognised-text assertion and `decimal_refuses_what_the_kernel_never_writes` |
| A number past 64 bits wraps instead of being refused | rejected: the same two |
| A manifest that does not fit still authorises a download | rejected: `ManifestDoesNotFit and downloading` |
| none, before and after | passed |

Before the crate existed, the scaffolded test failed with `EN-01 is registered
and not built`, as rule R3 requires.

## EN-02 and EN-03, the model reader

The tests build safetensors files byte for byte: eight bytes of header length,
a padded JSON header, then the data. What the reader recovers is compared with
where the test put it, so a header length read the wrong way round, a data start
that forgets the eight bytes, tensors left in the order the header happens to
list them, or a previous end that is not the previous tensor's end cannot
survive.

No answer about a byte range is written in the test. One kernel double answers
only what the kernel's own examples record, and the files built for it are the
ones whose range question is an example, so the guest has been seen to answer
exactly that. The other replays the kernel's own text for an accepted range and
keeps every request it was asked, which is how the fold over three tensors is
checked.

The config tests take a whole config and spoil it one way at a time: every name
in the registry and several outside it, each required key dropped in turn, a key
present as zero, a negative number, text, `null` and a fraction, and each gate
activation the recipe implements and several it does not.

### The gate: `cargo test -p repo-conformance --test engine`

Twenty-one defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| The header length is read big-endian | `engine_fl_03` |
| The data starts at the header, not after the eight bytes of length | `engine_en_02` |
| The tensors are left in the order the header lists them | `engine_en_02` |
| The tensors are ordered by name instead of by where their bytes lie | `engine_en_02` |
| Every tensor is asked about from the start of the data | `engine_en_02` |
| The previous end is the previous tensor's start | `engine_en_02` |
| The format's own `__metadata__` is read as a tensor | `engine_en_02` |
| A tensor's length is its end plus its start | `engine_fl_05` |
| A tensor that ends before it starts is read as an empty one | `engine_fl_05` |
| A header that is any JSON at all is read | `engine_fl_03` |
| A file shorter than its header length is read to its end | `engine_fl_02` |
| A file with no header length reads as an empty header | `engine_fl_01` |
| The kernel's answer about a range is not read | `engine_fl_06` |
| Anything the kernel answers but a rejection is an acceptance | `engine_fl_06` |
| A tensor's dtype is not required | `engine_fl_04` |
| A family name is recognised whatever its case | `engine_en_03` |
| A required key that is present but is not a positive number counts as present | `engine_en_03` |
| A required key of zero counts as present | `engine_en_03` |
| An activation that is not text is read as absent | `engine_en_03` |
| The config's own flag is passed to the kernel in place of what the weights say | `engine_en_03` |
| A config that is any JSON at all is read | `engine_fl_07` |
| none, before and after | passed |

Before the crate existed, the scaffolded tests failed with `EN-02 is registered
and not built` and `FL-01 is registered and not built`, as rules R3 and R5
require.

One defect was planted and missed, and the code changed rather than the record:
a header length added with wrapping arithmetic instead of checked. No caller can
see the difference, because a sum that wraps lands before the eight bytes of
length and a sum that saturates lands past the end of the file, and both are
refused as a header that does not fit. The checked addition claimed a gate that
could not fail, so the sum now saturates and says why.

### What EN-02 and EN-03 do not establish

- That the reader is right about a file it has not been given. The files are
  built here; no checkpoint from a hub is read in this suite.
- That a multi-tensor file's ranges are ones the compiled guest has answered.
  The kernel's recorded examples carry no chain of range questions, so the
  three-tensor fold is checked against a double that accepts, and only the
  single-tensor files put a question the guest has been seen to answer.
- That a tensor's bytes are what its name, dtype and shape say. Nothing here
  reads the data, and `EN-02` is about where the bytes lie, not what they are.
- Whether the tensor manifest fits the family's recipe. That is read from the
  weights, and it reaches the kernel's preflight question as the caller's input.

## EN-04 and EN-05, the ONNX preflight

The files are written out in the tests, field by field, from the numbers
`onnx.proto` gives: a tag, then a length, then the bytes. None of the reader is
used to build them, so a field number read from the wrong place, a length
counted wrongly, or a name taken from the wrong field cannot pass. Each file
also carries fields a preflight does not need, in each encoding it has to step
over, and their contents are bytes no tag could be: a reader that steps over the
wrong number of them reads one of them as a field.

What the tests fix: the opset of the default domain and no other, the operator
and domain of each node in the order the file has them, the count of tensors the
graph carries, a later opset declaration replacing an earlier one, a declaration
that names no domain being the default domain's, a node that names no operator
reading as one with no name, and a version written as a negative number reading
as the large number it stands for, which no builder implements.

Which operators a builder implements is not this crate's to say, and no graph is
built in this repository. The count of what lies outside that set is this
crate's, and `EN-05` fixes it: an operator of another domain is a different
operator that happens to share a name, and what is missing is reported once each
in the order it first appears.

### The gate: `cargo test -p repo-conformance --test engine`

Eighteen defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| The opset is read from the field that holds the domain | `engine_en_04` |
| A node's operator is read from the field that holds its name | `engine_en_04` |
| A node's domain is read from the field that holds its documentation | `engine_en_05` |
| The graph is read from the field that holds the opsets | `engine_en_04` |
| A graph's tensors are counted from the field that holds its nodes | `engine_en_04` |
| An earlier opset declaration wins over a later one | `engine_en_04` |
| Every domain's opset is read, not the default domain's | `engine_en_04` |
| An eight-byte field is stepped over as four | `engine_en_04` |
| A model with no graph reads as one with an empty graph | `engine_fl_09` |
| A model with no default opset reads as opset zero | `engine_fl_13` |
| An operator of another domain counts as the ONNX one of that name | `engine_en_05` |
| What is missing is reported once for every node that names it | `engine_en_05` |
| The count of what is missing is the count of what is implemented | `engine_en_05` |
| A number may run past the bytes that carry sixty-four bits | `engine_fl_09` |
| A length that reaches past the end is read to the end | `engine_fl_08` |
| A fixed-width field that runs off the end is accepted | `engine_fl_08` |
| An undefined encoding is read as a number | `engine_fl_10` |
| A name that is not text is read as the bytes allow | `engine_fl_11` |
| none, before and after | passed |

One of them was missed at first, and the test changed rather than the code: the
fields a preflight steps over were written as zero bytes, and zero bytes read as
empty fields, so stepping over four of eight left nothing that failed. Their
contents are now bytes no tag could be.

### What EN-04 and EN-05 do not establish

- That a file from a hub reads as its publisher meant it. Every file here is
  written in the test.
- That a model which passes preflight can be built. Nothing here builds a graph,
  and the set of operators a builder implements is the builder's to declare.
- How long a preflight takes. No file of a real model's size is read here, and
  no time is measured.
- That an operator's attributes, its inputs or its shapes are ones a builder
  could use. Only the operator's name and domain are read.

## EN-06 to EN-08, the download

A download arrives in chunks from somewhere else. The label is taken as they
arrive, so nothing is held twice, and the length is counted with it. What
arrived and what the archive promised both go to the kernel, which admits or
refuses; a refusal comes back carrying the kernel's own verdict.

`EN-06` is the one claim in this repository that is reproduced rather than
built. The hash is the authority's own implementation, cited in
`model/authorities.toml`, and the test runs all 35 of the authority's published
vectors. The vendored vector file is checked against the checksum in that
register before a single case runs, by a SHA-256 written out in the test, so a
vendored file that drifted from what was cited cannot quietly pass, and the
check does not go through the crate under test.

`EN-07` is about the stream, not the hash: ten lengths either side of the
hash's own block and chunk sizes, each split six ways, must give one label, and
a stream one byte short must not.

`EN-08` pins what the kernel is told: the promised label and length, then the
observed ones, in that order. A download that stops early is told as the length
that arrived, not the length that was promised, so the kernel sees the
difference rather than being handed the answer.

### The gate: `cargo test -p repo-conformance --test engine`

Fourteen defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| A label may be written in upper case | `engine_fl_14` |
| A label of any length is read | `engine_fl_14` |
| A label's two digits are read in the other order | `engine_en_06` |
| A label is written with its digits in the other order | `engine_en_06` |
| The letters of a label are read one place off | `engine_en_06` |
| What arrived is counted in chunks, not bytes | `engine_en_07` |
| Only the first chunk is taken in | `engine_en_07` |
| The label is of nothing, whatever arrived | `engine_en_07` |
| The kernel is told what arrived as what was promised | `engine_en_08` |
| The kernel is told what was promised as what arrived | `engine_en_08` |
| The promised and the observed labels are told in each other's place | `engine_en_08` |
| An answer the client does not know admits the buffer | `engine_en_08` |
| The kernel's answer is not read | `engine_fl_15` |
| One byte of the vendored vector file changed | `engine_en_06`, at the checksum |
| none, before and after | passed |

### What EN-06 to EN-08 do not establish

- That BLAKE3 is what its paper says it is. The vectors are the authority's,
  and so is the implementation.
- That a download arrives at all. Nothing here opens a connection, retries,
  resumes or times out; this crate is handed chunks.
- That a buffer the kernel admits is the buffer someone intended to publish.
  The archive's label is taken as given; where the archive itself comes from is
  another question, and not this crate's.
- How fast a label is taken. No file of a real model's size is hashed here, and
  no time is measured.

## EN-09 to EN-11, the conversation

Four of the kernel's decisions are about a conversation, and the tests put
three of them to a kernel that answers only what the kernel's own examples
record. The whole journey runs on those answers — idle to preflight, download,
compile, materialize, ready, generating and back to ready — so the engine's
state machine is driven by answers the compiled guest has replayed, not by
answers written here. So are the decode bucket and the token budget.

`EN-10` is the claim `KERNEL-VERIFICATION.md` records as the engine's to make:
that a proposed trim drops whole pairs, oldest first, and that the totals it
reports are the totals of those pairs. The kernel checks a proposal; it never
sees the pairs. The test checks the arithmetic against the pairs themselves,
including that the trim is the fewest pairs that fit, which is what the kernel
accepts, and that nothing is dropped from a conversation that already fits.

`EN-11` is about what changes and when. The pairs are forgotten only on the
kernel's acceptance: every refusal leaves the conversation whole, and the
verdict comes back as the kernel gave it.

### The gate: `cargo test -p repo-conformance --test engine`

Seventeen defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| A refused step is taken anyway | `engine_en_09` |
| The step is asked about the state the conversation is going to | `engine_en_09` |
| An answer the client does not know is taken as a step | `engine_en_09` |
| The part that never goes is left out of the question | `engine_en_10` |
| The question reports the pairs that remain, not the total | `engine_en_10` |
| The trim drops one pair more than it has to | `engine_en_11` |
| The trim drops from the newest pair | `engine_en_11` |
| What is reported dropped is the pair count, not the tokens | `engine_en_11` |
| The newest dropped is the oldest pair dropped | `engine_en_11` |
| The pairs are forgotten whatever the kernel answers | `engine_en_11` |
| An accepted trim forgets nothing | `engine_en_11` |
| An accepted trim forgets one pair too many | `engine_en_11` |
| The bucket is asked for the sequence as the bucket it already has | `engine_en_11` |
| The budget is asked about the tokens used as the context | `engine_en_11` |
| A bucket the kernel would not give is read as none | `engine_en_11` |
| The total counts every pair twice | `engine_en_11` |
| A turn with no room left is given the whole context | `engine_fl_19` |
| none, before and after | passed |

### EN-19, where the counts come from

`chat` took its counts from whoever called it, and the kernel's arithmetic was
only as true as those. It now counts a turn's question and answer with the
model's own tokeniser, and fixes the part that is never dropped from the system
prompt, the newest message and the reserve, the same way.

The test works one count out by hand — a tokeniser with a single merge, where
`th` is one token and `xy` is two — then puts two hundred generated turns
through and requires the conversation's count to be the tokeniser's, whatever
the text.

| Planted defect | Rejected by |
| --- | --- |
| A turn is counted as its question only | `engine_en_19` |
| A turn is counted in bytes, not in tokens | `engine_en_19` |
| The part never dropped leaves out the reserve | `engine_en_19` |
| The part never dropped leaves out the newest message | `engine_en_19` |
| What the conversation reports as never dropped is not what it uses | `engine_en_19` |
| none, before and after | passed |

### What EN-09 to EN-11 do not establish

- That the token counts are a tokeniser's. No tokeniser is here; the counts are
  the caller's, and a wrong count is a wrong question honestly asked.
- That a whole journey is what the eight steps add up to. The kernel proves
  each step; the claim about sequences is argued, as `KERNEL-VERIFICATION.md`
  records.
- That a conversation generates anything. Nothing here runs a model: the
  conversation keeps counts and asks what may be done with them.

## EN-12 and EN-13, how long it takes

Two gates, each shaped by what it can actually catch rather than by a number
that sounds good.

`EN-12` times a graph of 25,000 nodes and one of 50,000, best of three runs
each, and requires the larger to take less than three times the smaller. Two is
what reading each node once costs; four is what reading the graph again for
every node costs. A ratio is machine-independent: a busy machine moves both
measurements, not the shape of the relation between them. It also requires
50,000 nodes to be read in under a second, which is the ONNX preflight budget
this repository was asked for.

`EN-13` requires sixteen megabytes to be labelled at no less than twenty
megabytes a second. A floor, not a ratio, because the way to make hashing slow
is to feed it badly rather than to change its order.

That floor was first set at two hundred, from the machine this was written on,
and `main` went red on it: the arm64 runner labelled at 61 MB/s in the same
unoptimised build, sixty times slower than the same code on x86-64 here. The
floor now clears the slowest machine the gate runs on by three times, and the
defect that hands the hash one byte at a time ran at 15 MB/s, which is below
it. The margin on that side is thin, and that is the honest state of it: a
floor set from one machine is a measurement of that machine.

Measured here, in the gate's own debug build, on an AMD Ryzen AI Max 390:

| What | Measured | The gate |
| --- | --- | --- |
| 50,000 ONNX nodes, read and asked about | 69 ms, 31.8 MB/s | under 1 s |
| 25,000 against 50,000 nodes | ratio 1.84 | under 3 |
| 16 MB labelled, in 64 KB chunks | 4.6 ms, 3650 MB/s | at least 20 MB/s |
| the same, on the arm64 runner | 274 ms, 61 MB/s | the same floor |

### The gate: `cargo test -p repo-conformance --test engine`

| Planted defect | Result |
| --- | --- |
| Each node is put at the front of the list, moving every one already read | rejected by `engine_en_12`: ratio 4.19 |
| The hash is given one byte at a time | rejected by `engine_en_13`: 15.1 MB/s |
| The set of implemented operators is rebuilt for every node | **missed**: the cost is a constant factor, and the ratio is unchanged |
| Every chunk is copied before it is hashed, behind the whole download | **missed**: the copy allocates pages it never touches, so it costs almost nothing after the first run |
| none, before and after | passed |

### What EN-12 and EN-13 do not establish

- That the engine is fast on any machine but this one. The numbers above are
  this machine's; the gates are floors far below them, so a slower machine
  passes and a faster one is not rewarded.
- That a constant-factor slowdown is caught. Both missed defects are of that
  kind: one allocates per node, the other copies per chunk. A gate that caught
  them would be a gate that failed whenever the machine was busy.
- Anything about the browser. Nothing here is compiled to WebAssembly, where
  the hash loses the vector instructions that make it fast natively.

### A planted defect that was left behind

One run of the defect script was killed by a timeout between planting a defect
and restoring it, and the defect stayed in the working tree. Reading the diff
before committing found it. The rule it illustrates is the one that found it:
read what is about to be committed, not what was meant to be committed.

## EN-14, the page's own surface

The published page runs the verified guest in a browser. Until now it took a
request as hexadecimal, which is a way of saying that only someone who had read
the wire format could use it. `site/card.html`, `card.css` and `card.mjs` are
the app's own surface, which the frame leaves to the app: choose one of the
seventeen decisions, fill in its numbers by name, and the guest answers.

Every layout comes from `site/decisions.json`, one source read twice: the page
builds its form from it, and `EN-14` checks it against the kernel's own
examples. What the gate checks of the table: every decision the kernel routes
is offered and nothing else; every byte after the tag belongs to exactly one
field of the form it is in; each decision's length is the length its answered
examples have; a list's groups begin where the fields end and fill the length
the kernel accepts; every format the kernel reads has a form and every form
tiles its own request; every state and event the examples use can be chosen;
and every value the form starts with fits its field.

What the gate checks of the page itself: Node runs `card.mjs`'s own encoder
over every example the kernel carries, taking each field out of the example's
own bytes as the table has it, and requires the bytes back to be the example's.
The encoder is the thing being checked, not a copy of it.

The page's description is checked too: it names every decision the kernel
routes, and it knows every answer the examples give, so an answer the kernel
gives that the page cannot recognise fails the gate rather than appearing
unmarked.

### The gate: `cargo test -p repo-conformance --test engine`

Thirteen defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| A field is read one byte late | the two fields then share a byte |
| A field is one byte wide instead of eight | some byte of the request is no field |
| A field of the request is not on the form | the same |
| A decision the kernel routes is not offered | the offered set is not the routed set |
| A request is a byte shorter than the kernel accepts | a field runs past the request |
| A state the kernel knows cannot be chosen | the examples use a code the page does not offer |
| A form starts with a number its field cannot hold | 300 does not fit one byte |
| A label starts as the wrong number of digits | a label of 32 digits for 32 bytes |
| The page writes numbers least significant byte first | the encoder, over the kernel's examples |
| The page writes a label's digits in the other order | the same |
| The page writes no decision in the first byte | the same |
| The description leaves out a decision the kernel routes | the description does not mention decision 10 |
| The page does not know an answer the kernel gives | the page does not know `importable` |
| none, before and after | passed |

### Where the table comes from, and why the page stopped deploying

The first version of this card shipped an empty chooser and an empty field
area and built both from `site/decisions.json` after the page loaded. That is
a layout shift by construction, and the frame measures exactly that: its own
build script says everything the page says is in the HTML before any script
runs, so nothing shifts after first paint. The `pages` workflow went red the
moment the card landed, at `84f23b5`, and stayed red for four merges —
performance 0.99 against a required 1, cumulative layout shift 0.028 against
0, largest contentful paint 2119 ms against 1500 — and the site did not deploy
in that time. The acceptance jobs were green throughout, and the page's own
job is skipped on pull requests, which is why it went unnoticed.

The fix is also the de-duplication. `cargo xtask page --write` reads the
kernel's own modules — the tag each decision is routed by, the length its
answer accepts, and the offset and width of every field it reads — and writes
them into `site/decisions.json` and into `site/card.html`: the chooser's
options, the first decision's fields, and the table itself as inline JSON. The
page is whole before its script runs and fetches nothing. `site/decision-names.json`
holds what the model cannot: what a field is called and what the form starts
with.

`just vv` runs `cargo xtask page` without `--write`, so a layout changed in
the kernel and not regenerated here fails the gate rather than reaching the
page.

### The gate on the generator: `cargo xtask page`

Seven defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Result |
| --- | --- |
| A decision's request grows by a byte in the kernel and not on the page | `site/decisions.json` is stale |
| A field moves in the kernel and not on the page | the field at that byte has no name in the companion file |
| The page's written part is edited by hand | `site/card.html` is stale |
| The table is edited by hand | `site/decisions.json` is stale |
| A field is renamed only in the table the page reads | the same |
| The generator takes the length from the first number it sees | the table it writes disagrees with itself |
| The generator writes no inline table, so the page would fetch again | `site/card.html` is stale |
| none, before and after | passed |

### The three lengths a request can have

A decision's length is where the generator starts, and the kernel writes it
three ways. Most decisions check a number: `=? 25`. A decision with a list in
it checks a base plus a stride times a count, `=? (7 + (prim multiply 4
(Kernel.Wire::fieldAt value 6 1) : Nat))`, which says both how long the request
is and where the count lives. A decision whose shape depends on a format
checks a different length in each branch, beside the test that picks it, and
one branch may stand for two formats.

All three are read, and `xtask`'s own tests carry one example of each, quoted
from the modules that use them — including the six decisions a browser engine
adds, so the parser is known to read them before they arrive. A module whose
answer checks no length is refused rather than given one.

### A list in a request, and a form per format

The six decisions a browser engine asks about brought the other two shapes with
them, so the page now draws all three.

A list is drawn as the groups the kernel indexes into, and those are read from
the definitions that do the indexing. A body of the form `fieldAt value (7 +
(prim multiply 4 position : Nat)) 4` is a group of four-byte items from byte
seven; one of the form `fieldAt value ((3 + (prim multiply 4 window : Nat)) +
(prim multiply 4 position : Nat)) 4` is a group that begins after as many items
as another group holds. How many items a group holds is then the distance to
the next group, or to the end of the request, both of which the length its
answer accepts gives: a constant and a multiple of the count. For a speculative
commit that comes out as `k` drafted tokens from byte three and `k + 1`
verified ones after them, which is what `7 + 8k` leaves room for. Nothing about
the groups is written down a second time; `site/decision-names.json` gives each
one a name and a starting value, and nothing else.

A request whose own format chooses its shape is drawn as one form per format,
each the fields its branch reads — the test that picks the branch included,
since the format byte is a byte of the request too. The same byte is four bytes
of codes to one format of a quantisation code and one byte of signs to another,
so the forms do not agree on what a byte is, and only a form per format can say
what each one holds. A format the kernel reads nothing for is refused rather
than written as some other one.

What this costs: the form draws at most sixty-four items of one group, and says
so in the field that counts them. The kernel takes as many as a byte can count.

### The gate on the forms

Four more defects, planted one at a time and then restored. Every one was
rejected; none was missed. The first two are planted in the generator and the
table is rewritten from it, so that `cargo xtask page` has nothing to complain
about — the file on disk is exactly what the generator says — and `EN-14` is
what is left to notice.

| Planted defect | Rejected by |
| --- | --- |
| The generator reads only the branch a format takes, not the test that chooses it | decision 17, format [0]: a byte of the request is no field of this form |
| The generator gives every group the items the count gives it, and no more | decision 16: its groups do not fill the length its answer accepts, (3, 8) against (7, 8) |
| The page starts every group of a list at the byte the first one starts at | the encoder: decision 16, two fields write byte 3 |
| The page writes every format of a request at its longest | the encoder: decision 17 writes 15 bytes, the kernel accepts 7 |
| none, before and after | passed |

The first two also fail the generator's own tests, which is the point of having
both: the one that reads the model and the one that reads what it wrote.

### What EN-14 does not establish

- That the kernel answers what these forms write. The model carries no example
  of decisions 12 to 17 — the list of acceptance vectors is at its ceiling —
  so what is asked of their forms is what can be: that the request comes out at
  the length the module accepts, and that every byte of it is written by exactly
  one field. That those requests are answered the way the decision says is the
  kernel's own claim, proved in `app/src/Proofs` and pinned by the kernel suite,
  not replayed here.
- That the page works in a browser. The gate runs the encoder in Node and
  checks the table; it does not open a page. The real page was assembled with
  `python -I bootstrap/og.py && python -I bootstrap/icons.py && node
  bootstrap/page.mjs site` — the social image and the icons carry the app.json
  they were drawn for, and the committed ones are the `pages` workflow's, which
  draws them again before it deploys — and driven by hand with no guest behind
  it:
  every decision's form drew, the count redrew the list and moved the second
  group, each format of a quantisation code drew its own fields, and the bytes
  the page wrote were the bytes Node writes for the same values. The published
  page is what the pages workflow builds.
- That the verdict shown is the verdict the guest gave. The card passes the
  answer through, and nothing here replays a guest.
- That someone can tell what the numbers mean. The form names each field; it
  does not explain the decision.

## EN-15, what the readers do with anything at all

Two readers here are handed bytes someone else wrote: the safetensors header
and the ONNX model. Both parse untrusted input, which is where a stray index or
a subtraction that goes below zero stops being a question of style.

`EN-15` puts 4,369 files to them, from one seed, so a failure can be reproduced
exactly: the good file, the good file with one to five bytes changed, the good
file cut short at every length and lengthened with noise, bytes with no shape
at all, and — for the safetensors header — four hundred headers that are well
formed as JSON and adversarial as numbers.

Three things are required of every one: it does not panic, caught by running
each read inside `catch_unwind`; it fails only as `model/failures.toml` lists;
and the same bytes give the same answer, which a reader that depended on
anything but its input would not.

### The gate: `cargo test -p repo-conformance --test engine`

| Planted defect | Result |
| --- | --- |
| A length-delimited field is taken without asking whether it is there | rejected: the ONNX reader panicked on 196 bytes |
| A number is read past the end of the file | rejected: the same |
| The header length is read from a file that may not have one | rejected: the safetensors reader panicked on 0 bytes |
| A tensor's length is taken without checking it runs forward | rejected, after the corpus was strengthened |
| The reader answers the same bytes differently each time | rejected: the same bytes answered differently |
| none, before and after | passed |

The fourth was missed at first, and the generator changed rather than the
record. Changing bytes at random almost never reaches a tensor whose offsets
run backwards, because almost every change breaks the JSON before the numbers
are read. The four hundred built headers reach it, and a reader that subtracts
without checking now panics on one of them.

A sixth defect was planted and withdrawn: stepping over a fixed-width field
without checking that the position plus four or eight overflows. The check
could not fail — the position is inside a slice that exists and the width is
four or eight — so the code now adds plainly and says why. A branch no file can
take is a branch no test can reach.

### What EN-15 does not establish

- That the readers are right about a file. It establishes what they do not do.
  What they read is `EN-02`, `EN-04` and the rest.
- That no input panics. Four thousand files from one seed is a search, not a
  proof; a reader that panicked on one input in a million would pass this.
- Anything about the JSON parser underneath the config reader, which is a
  crates.io release, not this repository's.

## EN-16 to EN-18, the tokeniser

A conversation's arithmetic is in tokens, and the kernel proves that
arithmetic over whatever counts it is given. A count is the model's own only if
the text was split the model's way, so everything here comes from the model's
own file: the vocabulary, the merges and their order, and the pattern that
splits text into pieces. The pattern is run by `fancy-regex`, because the
published patterns use a look-ahead that the plain regular-expression crate
does not have.

What is refused, and by name: a model that is not byte-pair encoding, a
pre-tokeniser that is not byte level, a pattern this crate's regular
expressions cannot run, a merge that is not two pieces, and a missing
vocabulary. A tokeniser that was nearly right would give counts that were
nearly right, which is the one thing a proof about those counts cannot survive.

The byte alphabet is what makes `EN-18` possible: every one of the 256 bytes
has its own character and every character has its byte, so there is no text
that cannot be encoded and no unknown token to stand in for one.

### The gate: `cargo test -p repo-conformance --test engine`

Twelve defects, planted one at a time and then restored. Every one was
rejected; none was missed.

| Planted defect | Rejected by |
| --- | --- |
| The soft hyphen is left where it is, like its neighbours | `engine_en_18` |
| The moved bytes start one place lower | `engine_en_17` |
| The merges are applied from the left instead of by rank | `engine_en_17` |
| The merge with the highest rank goes first | `engine_en_17` |
| A later merge of the same pair wins over the earlier one | `engine_en_17` |
| The text is never split, whatever pattern the file carries | `engine_en_17` |
| What the pattern steps over is dropped instead of kept | `engine_en_17` |
| A model that is not byte-pair encoding is read anyway | `engine_en_16` |
| A pre-tokeniser that is not byte level is read anyway | `engine_en_16` |
| A file with no vocabulary is read as an empty one | `engine_en_16` |
| An id that is not in the vocabulary decodes as nothing | `engine_fl_22` |
| A token outside the alphabet decodes as the first byte | `engine_fl_23` |
| none, before and after | passed |

Three of those were missed at first, and each time the test changed rather than
the code, because each named a case the tests did not reach: two merges that
overlap, where applying them in the wrong order gives a different answer; a
pattern that matches only part of the text, so that something is stepped over;
and a pre-tokeniser that is a pattern and nothing else. All three are now in
`EN-16` and `EN-17`.

### EN-20, against the authority itself

`EN-16` to `EN-18` say that this crate does what it says it does. They cannot
say that what it does is what the model's own tokeniser does, and that is the
claim the counts actually rest on.

So the oracle is the authority's own implementation: HuggingFace's `tokenizers`
crate, a dev-dependency of the conformance crate, which does not ship, so
nothing the engine carries changed. Both are given the same file and the same
text, and the ids must be the same ids. 1,526 comparisons: seven files, from
merges only to the shape a published model has, over eighteen written texts and
two hundred generated ones.

It costs one allowance to run. The oracle's crate brings in `paste`, an
unmaintained proc-macro (`RUSTSEC-2024-0436`), and `deny.toml` now names that
advisory with its reason: it is reached only through a dev-dependency of a
crate that does not ship, `cargo tree -i paste --target all` shows the single
path, and every version of the oracle carries it. The allowance is Ilya's,
scoped to that one advisory, and it goes the day either side drops the other.

It earned its place immediately. The authority's reader refused the files this
suite was building, for want of `add_prefix_space`, and what that field means
turned out to be something this crate was not doing at all:

- `use_regex` on a byte-level pre-tokeniser means it splits by the authority's
  own pattern, which the file does not carry. This crate treated a byte-level
  pre-tokeniser with no pattern beside it as *no splitting*, which is what a
  published file asks for least often.
- `add_prefix_space` means a space goes in front of text that does not start
  with one, so the first word is the same token as that word anywhere else.
  This crate was not doing it.

Both are fixed, and both are now planted as defects.

| Planted defect | Rejected by |
| --- | --- |
| The pattern is the authority's with one alternative dropped | the ids differ on a run of spaces |
| The space before the first word is never put there | the ids differ for `the` |
| The space is put there whatever the file says | the same |
| The byte-level pre-tokeniser never splits, whatever the file says | the ids differ for ` the the the` |
| The merges are applied from the left instead of by rank | the ids differ for `abc abcabc` |
| The moved bytes start one place lower | the ids differ for a space |
| A flag the file leaves out is read as off, not as the authority's default | **missed** |
| none, before and after | passed |

The missed one cannot be caught this way, and it is worth saying why: the
authority's own reader refuses a file that leaves those flags out, so there is
no file on which the two can be compared. What this crate does with such a file
follows the authority's documented defaults, and that is a claim about a file
the authority will not read.

Two of the defects above are also the two mistakes this crate actually had. A
gate that only confirms what its author believed is worth less than one that
has already found something, and this one found two.

### What EN-16 to EN-18 do not establish

- **That a published model's own file is read.** Every tokeniser in this suite
  is built here, in the shapes a published file has. None comes from a hub, so
  nothing here establishes that a real `tokenizer.json` is read as its
  publisher meant it; what `EN-20` establishes is that for the files it does
  read, the ids are the authority's.
- That the conversation's counts are a published model's. `chat` now counts
  with this crate (`EN-19`), so the two agree with each other; whether either
  agrees with the model's own tokeniser is the open question above.
- That normalisers are applied. A file that normalises text before splitting it
  is read, and the normaliser is not run, so text that a normaliser would
  change is counted as it was written.

## EN-21 and EN-22, the tokeniser a model carries in its own file

A GGUF model holds its vocabulary in the first part of its own file: a magic
number, a version, two counts, then metadata entries until the tensors begin.
A hundred and fifty thousand tokens sit in the header of a file of gigabytes,
and the header is all a count needs, so nothing past the metadata is read.
`EN-21` fixes that: the same header with nothing after it reads the same way,
and so does the same header with a megabyte of tensors behind it.

Two things are cited rather than decided here, and both are in
`model/authorities.toml`. The format is the GGUF specification's. What a
pre-tokeniser's *name* stands for is `llama.cpp`'s, because the file carries
only the name — `tokenizer.ggml.pre` — and the pattern it means lives in the
reader. The two patterns this crate carries, for `gpt2` and for `qwen2`, are
quoted from that file at a named commit. **A name with no pattern quoted here
is refused rather than read as a near one**, which is the whole point: a
tokeniser that was nearly right would give counts that were nearly right, and
the kernel's arithmetic is proved over those counts.

The names are not interchangeable and the tests show it. The `qwen2` pattern
takes digits one at a time where `gpt2` takes a run of them, so a merge of two
digits applies under one name and not under the other.

`EN-22` is the cross-check: the same vocabulary and merges, written once as a
GGUF header and once as a `tokenizer.json`, must give the same ids — and the
JSON side is also put to HuggingFace's own tokeniser, so the agreement reaches
the authority through it.

### The gate: `cargo test -p repo-conformance --test engine`

Fifteen defects, planted one at a time and then restored.

| Planted defect | Rejected by |
| --- | --- |
| The magic is not checked | `engine_fl_24` |
| A version this crate does not read is read anyway | `engine_en_21` |
| The tensor count is read as the metadata count | `engine_en_21` |
| A tokeniser model that is not byte-pair encoding is read anyway | `engine_en_21` |
| A file with no vocabulary is read as an empty one | `engine_en_21` |
| A string of the header may run past the end of the file | `engine_fl_25` |
| An array of numbers is read as a vocabulary of text | `engine_en_21` |
| A value type the format does not define is stepped over as four bytes | `engine_en_21` |
| The merges are taken from the key that holds the tokens | `engine_en_22` |
| A pre-tokeniser name with no cited pattern is read as the default one | `engine_en_21` |
| The `qwen2` name is read as the `gpt2` pattern | `engine_en_21` |
| The ids of a GGUF vocabulary are not its positions | `engine_en_22` |
| The `gpt2` pattern is the published one, with a trailing run of spaces | **missed** |
| none, before and after | passed |

The missed one is worth stating plainly. `llama.cpp`'s `gpt2` pattern ends at
`\s+(?!\S)` where the published byte-level pattern adds `|\s+`. No text was
found that tells them apart here, and the reason is this crate's own doing:
what a pattern does not match is kept as its own piece rather than dropped, so
the alternative that would have matched it changes nothing. The difference may
still be real in a reader that drops what it does not match.

### What EN-21 and EN-22 do not establish

- **That a published model's GGUF file is read as its publisher meant it.**
  Every file in this suite is built here, field by field, from the format's own
  specification. The first test model's header was read once by hand to see
  what it says — `tokenizer.ggml.model` is `gpt2`, `tokenizer.ggml.pre` is
  `qwen2`, with 151,669 tokens — and that reading is not in the gate.
- That the `qwen2` pattern gives the ids `llama.cpp` gives. It is quoted from
  `llama.cpp`, and nothing here runs `llama.cpp` to check the quotation does
  what the original does.
- Anything about tensors, weights or quantisation. This reads a header.

## A token the file adds, and the turn that never ended

The tokeniser counted `<|im_end|>` as text. It is in the vocabulary, with an
id of its own, but nothing matched it as itself: the pre-tokeniser split it
into `<`, `|`, `im`, and so on, and byte-level merging turned those into
ordinary ids. A prompt written by a chat template arrived at the model
mangled, and — the part that matters — the model's own end-of-turn id was
never a token it had been shown, so it answered and kept going, writing
`</|im_end|>` as text forever.

That was found by running the thing, not by a gate here. The session that had
the GPU loaded the real model through this engine and watched a turn fail to
end. Three of this repository's own gates were green while it happened: the
oracle agreed with the authority on every text it was given, and none of those
texts contained a token the file adds.

What a file adds is now matched as itself, before anything is merged, longest
first so a token that begins another is not taken in its place. Both readers
say which: a `tokenizer.json` lists them under `added_tokens`, and a GGUF
header marks them in `tokenizer.ggml.token_type`, as control or user-defined,
which is the same statement in the other format.

The gate that was blind is the gate that now sees it. `EN-20` gives
HuggingFace's own tokeniser the exact turn a chat template writes, and eight
more texts around the awkward edges — a token at the start, at the end, two
together, a prefix that is nearly one, a close bracket too many — and requires
the same ids for every one. `EN-22` builds the same vocabulary both ways, the
control marks in the GGUF header against `added_tokens` in the file, and
requires the ids to agree, with the authority judging the file it can read.

Writing those fixtures turned up a second thing worth knowing: the authority
numbers added tokens itself, straight after the vocabulary, rather than taking
the id the file declares. A published file numbers them that way already, so
the two never disagree in practice; a fixture that numbered them otherwise did,
and the fixtures now number them as a published file does.

### The defects

Four, planted one at a time and then restored. Every one was rejected; none was
missed.

| Planted defect | Rejected by |
| --- | --- |
| The tokens a file adds are merged like any other text | EN-20: the ids differ for the turn a chat template writes |
| An added token is matched but its id is not the file's | the same |
| The tokens a file adds are tried shortest first | EN-20: the ids differ for `<\|im_start\|>system` |
| A header's control tokens are read as ordinary text | EN-22: the header and the file disagree |
| none, before and after | passed |

### A comparison that could not fail

The matcher first held a tie-break: at the same place in the text, the longer
token wins. Planted against — first match wins, whatever its length — nothing
failed, and nothing could: `added` is already in longest-first order, so the
longer token is tried first and the shorter one never displaces it. The
comparison was dead, and restating a guarantee the order already gives is a
line that can rot without anything noticing.

It was removed, and the comment says the order is what decides. The defect
that reads on the same behaviour is the order itself, and reversing it fails
EN-20 on `<|im_start|>system`, which is the text the fixture gained for this:
a token whose first characters are another token, which a model that gives its
system header an id of its own has.

Finding this cost a fixture that was added to catch a defect and a planting
that did not catch it. That is what the planting is for.

### What this does not establish

- That the whole chat template is right. What is checked is the ids of the
  text it produces. Whether the page wraps a turn in the tokens this model
  expects is the page's business and is not checked here.
- That every published model marks its added tokens the same way. Two formats
  are read, and a file that records them some third way would have none, which
  is the behaviour a model with no tokens of its own needs anyway.

## EN-23, the engine in a browser

The page runs the kernel already: the holo's guest, verified before it is
compiled. What it could not run was everything around the kernel. Reading the
tokeniser a model carries, counting the tokens of a turn, labelling the bytes
of a download as they arrive — those are this repository's crates, and a page
that did them again in JavaScript would be a second implementation of rules
that exist here, which is the thing this repository is organised to prevent.

`crates/wasm` is those crates as one WebAssembly module. It adds no dependency
and no binding generator: the interface is bytes in linear memory. `allocate`
reserves a buffer and says where it is, the page writes into it, a call reads
it and leaves its answer in one result buffer that `result_at` and `result_len`
describe, and `release` gives the buffer back. Every call answers 0 for yes and
1 for no, and a call that answers no leaves the failure's registered name where
the answer would have been.

There is no `unsafe` code in it. A buffer is a `Vec<u8>` the module keeps, and
a call that names an offset the module did not give is refused rather than
followed, so nothing reads memory this module does not hold. Exporting a symbol
is an unsafe attribute — a module that exports nothing cannot be called — and
each export says so; `audit-limits` now also permits a shipped crate to declare
`crate-type` and nothing else, because a browser instantiates a cdylib, while a
`[lib]` that moves the root where the gate would not read it is still refused.

### The gate: `cargo test -p repo-conformance --test engine engine_en_23`

The test builds the module for `wasm32-unknown-unknown` in release, which is
what ships, and instantiates it in Node exactly as a page does —
`WebAssembly.instantiate`, no glue. Then it asks the module what it asks the
crates, and requires the same bytes: the vocabulary's size; the ids, the count
and the decoded bytes of four turns, including an empty one and one the model
has never seen; the same turns again through a tokeniser read from a GGUF
header rather than a `tokenizer.json`, which must give the same ids; a header
this engine does not follow, which must be refused by name rather than
answered; and a download labelled block by block, three blocks, with the label
compared after each one and the kernel's admission request compared at the end.

Forty-odd answers, each the crate's own. The crates are the source; the module
is the thing being checked.

### The defects

Six, planted one at a time and then restored. Every one was rejected; none was
missed.

| Planted defect | Rejected by |
| --- | --- |
| The module writes a token's number the other way round | EN-23: the module's answer is not the crate's |
| The module counts a turn by its bytes rather than its tokens | the same |
| The module labels only the last block of a download, not all of it | the same |
| The module reads a GGUF header as if it were a `tokenizer.json` | the same |
| A call reads memory the module was never asked to allocate | FL-26: the failure is not `NoBuffer` |
| The crate moves its root where the gate would not read it | `audit-limits`: a shipped crate keeps its targets where Cargo finds them |
| none, before and after | passed |

### What EN-23 does not establish

- That a browser runs it. Node and a browser both instantiate the same module
  through the same interface, and this runs Node. The page's own loading of it
  is not written yet: the module is built and checked, and nothing on the page
  calls it.
- That the module is fast enough, or small enough. It is 1.5 MB in release and
  nothing here measures either.
- Anything about the kernel. The module asks the kernel's questions and carries
  its answers; it decides nothing, and nothing here replays a guest.

## EN-24, the root a container names

A weight container names each buffer it carries by a label, the BLAKE3 hash of
those bytes, and the whole file they arrive in by a root, its SHA-256. The
labels were already taken as the bytes arrived; the root is now taken the same
way, in the same pass, so a download is read once.

`sha2` is the authority's own implementation, locked from crates.io (R6) and
cited in `model/authorities.toml`. What is reproduced from the standard is
three digests — FIPS 180-4's Appendix B examples, `abc`, the fifty-six byte
message and one million `a` — written into the test beside the inputs they are
for, with the empty message's digest beside them. The test computes them and
requires them to be those values; it establishes nothing about SHA-256 itself.

Besides the published values, EN-24 requires the root to be of what arrived and
not of how it arrived: a hundred thousand bytes taken one at a time, two at a
time, and in chunks of 3, 64, 1021 and 65,536 all come out at the root of the
same bytes taken at once. Taken partway through a download it answers for what
has arrived so far, and the download goes on. A root and a label of the same
bytes are different hashes, which the test also requires, so the two cannot be
swapped unnoticed.

`FL-30` is a root read loosely: upper case, a digit short, a digit long, a
letter that is not hexadecimal, the `sha256:` written twice, another algorithm's
name, a leading space. A name read loosely is a name that can be made to match.

### The defects

Five, planted one at a time and then restored. Every one was rejected; none was
missed.

| Planted defect | Rejected by |
| --- | --- |
| The root is taken of the first chunk only | EN-24: a download taken one byte at a time has another root |
| The root is another hash under this name | EN-24: the root of the empty message is not the one the standard publishes |
| A root is read in upper case too | FL-30: an upper-case name is not a root and was not refused |
| The module roots each block instead of the file | EN-23: the module's answer is not the crate's |
| The module labels the file instead of the block | the same |
| none, before and after | passed |

### What EN-24 does not establish

- That SHA-256 is what the standard says it is. Three published digests are
  reproduced; the rest of the algorithm is the authority's.
- That a container's own root is the root of its file. Nothing here has read a
  container yet: that is the reader this engine does not have.

## EN-25, the weight container

A model's weights arrive in one file: a magic, the length of a header, that
header as JSON, and then the buffers it describes. The header names the whole
file by a root, names each buffer by a label and a span, says where the
tokeniser lies, and says what shape and quantisation each buffer is.

The layout is not a published specification, and this says so rather than
implying one. It is one file's, and that file's first 98,314 bytes — its magic,
its header length and its header — are vendored at
`model/authorities/holk-bonsai-header.bin` with their SHA-256 in the register,
fetched from the revision the register cites. The magic is the five bytes
`HOLK1` and a newline, then a four-byte little-endian header length, then the
JSON, then spaces to fill the room. Two facts of the file decide that reading
and the gate checks both: the byte at ten is `{`, and the tokeniser the header
names begins at 98,314, which is ten plus the length those four bytes carry. A
reader that took the newline for a separator, or the length for a version,
disagrees with the file at one of them.

What the gate now reads there, rather than being told: 311 buffers in two
quantisation formats — 198 `q1` and 113 `f32` — a vocabulary of 151,669, 28
layers, and a furthest end of 424,013,749, which is the file's own length as
measured from the file rather than read from its header. Three things that file
settled, each of which a reader written from a description would have got
wrong: its ternary buffers are named `q1` and never `ternary`; it carries
`f32` buffers, which the kernel had no code for until decision 13 gained one;
and its buffers do not tile it — two of them share one span, because the model
ties its embedding to its output layer, so 311 buffers lie in 310 spans. That
last one is why `range_question` takes `previous_end` from its caller and
infers nothing from the order.

What `crates/model-reader/src/holk.rs` reads is the header and nothing else.
The buffers may be hundreds of megabytes and arrive later, or never. Each entry
becomes the kernel's own questions and no answer of the kernel's is restated:
where the buffer lies (decision 6), whether its bytes may be used (decision 5),
and how long one block of its shape is (decision 13, whose client the
kernel-client crate gained for this). That the block question carries the shape
where the kernel reads it is checked against `site/decisions.json`, which is
written from the kernel's own modules, so the client and the kernel cannot
drift apart quietly.

The names a container uses and the numbers the kernel has line up both ways,
and the gate checks both. A code the kernel offers with no name here is a model
that arrives one day and cannot be passed on. A name standing for a code the
kernel does not offer is worse: it builds a request the kernel cannot answer,
which looks like a refusal of the model rather than a mistake in the reader.

Which `q1` a buffer is in is the one thing the name alone cannot say. The
published container writes every ternary matrix as `q1`, and the same family
writes a one-bit format under that same name with another layout entirely, so
the manifest's own `tern` decides: `q1` is the kernel's ternary format where
the model says its weights are ternary, and is refused where it does not say
so. The first version of this reader knew only the name `ternary`, which that
container never uses, so it refused all 311 of its entries; the session holding
the file said so, and this is the fix.

### What it refuses, and by what name

| Failure | What it is |
| --- | --- |
| `FL-31` `NotHolk` | another magic, a shorter one, or the magic without its newline |
| `FL-32` `Truncated` | a header that runs past the bytes given, at six lengths including nine and ten |
| `FL-33` `NotJson` | a header that does not begin a document |
| `FL-34` `PaddedWithSomethingElse` | a zero, a newline, a tab or `0xff` in the room after the document |
| `FL-35` `Unfollowable` | another format, a part missing, a number that is not one, a name that is not text, a label that is not `blake3:` and sixty-four digits, a root that is not `sha256:` and sixty-four digits, a quantisation name the kernel has no code for |

Where the document ends is the parser's to say, not something looked for
backwards: a header whose last byte is a zero would otherwise read as a
document with no padding at all, which is how the first version of this read it
until `FL-34` said so.

### The defects

Thirteen, planted one at a time and then restored. Twelve were rejected on
the first planting. The thirteenth was missed, and is recorded below.

| Planted defect | Rejected by |
| --- | --- |
| Every `q1` is the ternary one, whatever the manifest says | EN-25: a `q1` whose manifest says `tern` is false was read as the ternary format |
| A container's name stands for a format the kernel does not have | EN-25: `q1` stands for format 9, which the kernel does not offer |
| The manifest is read for a flag it does not carry | EN-25: the ternary `q1` was refused |
| The header's length is read the other way round | EN-25: the published container's header does not lie where it was measured to |
| The magic is read without the newline that belongs to it | the same |
| A quantisation name stands for the number beside the right one | EN-25: no name this reader knows is the kernel's format 4 |
| Whatever follows the document in the header is stepped over | FL-34: a header padded with `0x00` was read anyway |
| A buffer's label is taken from whatever its name says | FL-35: a `sha256:` name was read as a label |
| The block question writes the shape a byte early | EN-25: the rows are not where the kernel reads them |
| A format decision 13 knows has no name in the reader | EN-25: `f32` is a name this reader has: `Unfollowable(FormatNotKnown)` |
| The page offers a format decision 13 does not know | EN-25: the page offers formats decision 13 does not know |
| An unreadable quantisation falls back to the ternary one | EN-25: a `q1` whose manifest says `tern` is false was read as the ternary format |
| The vendored header is changed by one byte | EN-25: the vendored header is not the file the register cites |
| none, before and after | passed |

### The one that was missed

The defect "the page offers a format decision 13 does not know" was first
planted in `site/decision-names.json`, where the names are written, and EN-25
passed. The gate reads `site/decisions.json`, which is generated from that
file, and the generated file had not been rewritten, so the planting never
reached the gate it was aimed at. `cargo xtask page` would have caught it, as a
table that disagrees with its source, but that is a different gate answering a
different question.

Replanted in `site/decisions.json`, the file EN-25 actually reads, it was
rejected. The lesson is about planting rather than about the gate: a defect
planted in a source that something regenerates tests the regeneration, not the
reader.

### What EN-25 does not establish

- That the file beyond its header is anything in particular. 98,314 bytes of a
  424 MB file are here. Every buffer's span is checked against the file's length
  and against the end of the tokeniser; none of their bytes has been seen.
- What the manifest's own root covers. This was an open question here and is
  now half answered, by measurement rather than by argument: the whole file was
  streamed through this repository's own download module and came to
  `acf0b5ad…`, which is what the hub record and the model card both say, so
  that is the file's hash. The manifest names `8932967d…`, which is therefore
  not this file's hash but the identity its catalogue pinned, over the manifest
  the model was built from. What that covers is not recomputed anywhere here.
  The reader carries it as `Identity`, a type of its own, precisely so that no
  caller can compare it to the `Root` a download arrives at: they are two
  different numbers about the same model, and the names invited the mistake
  until this was measured.
- That another container reads. One file is one file. A second one written
  differently would be refused by name, which is the most this can promise.
- That the kernel admits what the reader asks about. The questions are built
  and compared to the client's own; the answers are the kernel's, and nothing
  here replays a guest.
- That a container's root is the root of its file. The reader carries the root
  the manifest names; comparing it with the bytes as they arrive is the page's
  loop, which is not written yet.

## EN-26, publishing the engine the page loads

A module nobody builds is a module no page can load. The engine is now built
where the holo is built — in the locked SDK image, by the workflow that
publishes the page, in the same step — and put at `site/app/engine/engine.wasm`
for the page to fetch when a visitor asks for a model.

It is built rather than committed, the way `site/app.holo` is. A committed
module would be whatever someone last remembered to rebuild; this one is what
the workspace builds at the commit being published, and `.gitignore` keeps the
other possibility out.

`just engine` is the recipe, so what CI does is what anyone can do locally,
and `cargo xtask engine-size` is the ceiling: two mebibytes uncompressed. The
module is 1,634,615 bytes today, 77 per cent of that, and 505,142 bytes
gzipped. The ceiling is on what this repository produces; the gzipped figure is
reported beside it because that is what a visitor fetches, and it is not gated
because it depends on what the server does.

Two mebibytes is a number someone chose, and what it protects is worth saying
exactly: the module is fetched only when a visitor asks for a model, never for
the first paint, so this is not about the page's speed. It is about the
visitor's connection — a module that doubled without anyone noticing would be a
slower first model for everyone, and nothing else here would say so.

### What the gate reads

Not the module alone: the wiring that publishes it. A gate that checked only
the file would pass while nothing published it. EN-26 reads the `engine`
recipe and requires it to build this workspace's crate, for the browser's
target, in release, to the path the page asks for, and to run the size check;
it reads `pages.yml` and requires it to build the engine beside the holo and
before the page is assembled; it reads `.gitignore` and requires the built
module to be ignored. Then it runs the size check itself, against the module
built from this commit — the same check the recipe runs, not a second reading
of the same rule.

### The defects

Five, planted one at a time and then restored. Every one was rejected; none was
missed.

| Planted defect | Rejected by |
| --- | --- |
| The workflow publishes a page without the engine it loads | EN-26: the pages workflow does not build the engine beside the holo |
| The recipe builds the engine for the machine it is built on | EN-26: the recipe does not mention `--target wasm32-unknown-unknown` |
| The recipe builds it but nothing checks its size | EN-26: the recipe does not mention `engine-size` |
| The built engine may be committed | EN-26: the built engine is not ignored |
| The ceiling is raised to whatever the module happens to be | EN-26: 1,634,615 bytes, over the 1,048,576 a page may ask for |
| none, before and after | passed |

### The kernel a page was built for

A visitor can be served a stale holo. A service worker that never reinstalls
will do it: the one this frame ships reads its cache manifest only when it
installs, and its own bytes do not change between deploys, so a returning
visitor keeps last week's kernel indefinitely. That is
UOR-Foundation/template#83, reported there because `site/sw.js` is byte-bound
by `frame.lock` and the bug belongs to every app published this way, not to
this one.

What it does to a page is worse than being out of date. An old kernel refuses
a question it has never heard of, honestly and in its own words, and a page
that cannot tell those bytes from the ones it was built against reports that
refusal as the kernel's verdict. Every signature and every vector check
passes, because they are all checks of the old artifact. That is how it was
found: a block-length question answered `invalid request` by a kernel from
before that format existed, while the server was serving one that answers it.

So the workflow writes the built holo's own archive fingerprint into
`site/app.json`, beside the attestation URL, before the page is assembled, and
EN-26 requires that order: the identity is read from the holo, written to the
page, and only then is the page built. Two planted defects — the build not
writing it, and the page assembled before it is written — were both rejected.

The page comparing that identity with the holo it is handed is the loader's
own, and is not here yet.

### What EN-26 does not establish

- That the published page loads it. Nothing under `site/app/` fetches the
  module yet: that page is being written in another session, and the gate that
  will load a real container through a real kernel in a real browser is
  `GP-03`, which is not here.
- That the module is fast enough. Its size is gated; nothing measures what it
  does with the time.
- That the ceiling is the right ceiling. It was agreed between two sessions and
  the module sits at 77 per cent of it. Moving it should be a decision someone
  makes out loud, which is why the failure says so.

## EN-27, the tensor table of a GGUF file

A model published as GGUF carries its weights after a header that names every
tensor: what it is called, what shape it has, how it is stored, and where its
bytes begin. A converter turning such a file into a weight container has to
read that table. `crates/model-reader/src/gguf.rs` reads it and stops.

It reports what the *file* says and nothing about what a container stores. The
distinction is not pedantry: GGUF's `Q4_0` is eighteen bytes for every
thirty-two numbers, an f16 scale and a nibble each, while this repository's
container stores the same numbers in twenty, with an f32 scale. A reader that
reported twenty because that is what the converter wants would be the converter
lying through its reader, and the first sign of it would be a table that
disagrees with the file by two bytes a block, several hundred million bytes in.
The test says so in as many words.

Two readers of this format now live here, and they stop in different places on
purpose: the tokeniser reads the metadata, where a vocabulary lives, and this
reads the table that follows it. They have to agree about where the metadata
ends or one of them is wrong, so EN-27 builds one file and reads it with both.

### What the published file says

Measured from Qwen2.5-1.5B-Instruct's Q4_0 GGUF at revision 91cad51, by range
request rather than by downloading a gigabyte:

| | |
| --- | --- |
| Metadata ends | 5,931,189 |
| Tensor table | 19,304 bytes, 339 entries |
| Tensor data begins | 5,950,496, the default alignment of 32 |
| Tensor types | 197 `Q4_0`, 141 `F32`, one `Q6_K` |
| Pre-tokeniser | `qwen2`, which this repository has a cited pattern for |
| Vocabulary | 151,936 tokens, 151,387 merges, `token_type` as an int32 array |
| The tokens a turn needs | `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645 |

The one `Q6_K` is `output.weight`, dims [1536, 151936] at tensor offset 0 —
the largest tensor in the model. The embedding is *not* the same tensor: the
file carries `token_embd.weight` separately, as `Q4_0`, at offset 191,439,360.
The model's config ties its embeddings and this file does not, which is the
kind of thing only reading the table tells you; an earlier draft of this note
said the two were one tensor, on the strength of the config, and the table says
otherwise.

Neither session had planned for a `Q6_K`. This reader refuses it by its number
— `TypeNotFollowed(14)` — rather than reading it as a type it does know,
because a type read as the wrong one puts every tensor after it in the wrong
place.

### The defects

Five, planted one at a time and then restored. Every one was rejected; none was
missed.

| Planted defect | Rejected by |
| --- | --- |
| A block is as long as our container stores, not as the file says | EN-27: the file's own block, not ours |
| The tensor data begins wherever the table ends | EN-27: the data does not begin where the format says |
| A tensor's offset is left as the file writes it | EN-27: an offset that is not from the start of the file |
| A type this reader does not follow is read as the nearest one | FL-39: a type this reader does not follow was read anyway |
| A tensor that does not fill its last block is rounded up | FL-40: a tensor of thirty-one numbers was measured anyway |
| none, before and after | passed |

### What EN-27 does not establish

- That the published file reads. The numbers above were measured here, by hand,
  and are not yet a fixture this gate re-reads: its header is 5.95 MB, because
  the vocabulary is nearly all of it, and whether a repository whose other
  fixtures are 32 KB and 96 KB should carry that is a decision rather than a
  detail.
- That a converter can be built on it. The table is read; nothing reads a
  tensor's bytes, and the `Q6_K` question — requantise, or teach the container
  a format — is open.
- That the types it follows are the only ones a published model uses. Four are
  followed because four have been seen. The rest are refused by number, which
  is the behaviour that makes adding one a decision rather than an accident.

## The limit, and what the page claims

Two gaps between what this repository proves and what its page says, found by
reading the published artifact rather than the repository.

### A proof nobody could reach

The kernel states the largest request it will take: `requestMaximum := 81`,
which was right when every decision was a fixed length and admission was the
longest at 81 bytes. Decisions 14, 15 and 16 are counted — their length
follows a number inside the request — and a greedy pick over 255 scores is
7 + 4 x 255 = 1,027 bytes. The guest refuses that before any decision is
reached. Those three decisions are proved, and at their full range they could
not be asked at all.

The page took its lengths from the kernel and knew nothing of the limit, so it
would compose such a request and watch it bounce. Now `cargo xtask page` reads
`requestMaximum` from the kernel the way it reads the offsets, writes it into
`site/decisions.json`, and the page refuses past it with the number in the
message. EN-14 requires the page and the kernel to agree about it, that every
fixed decision fits, that every counted decision's shortest request fits, and
— in the encoder check — that a request past the limit is refused while the
same request is still written when no limit is stated, so the refusal is the
limit's doing rather than the encoder failing anyway.

Raising the limit is the kernel's to do and wants a rebuild, so it rides the
next SDK image rather than landing ahead of it.

### Forty-nine theorems, fourteen named

`site/app.json` listed fourteen theorems, in prose of its own. The proofs
carry forty-nine, across seventeen modules. A list maintained by hand beside a
model is a second source by construction, and this one had drifted by
thirty-five.

The page now states each theorem in the words the module proves it in — the
statement from its `theorem{}` block, with the symbols drawn — and EN-14 reads
`app/src/Proofs/*.lex.tex` and requires the two to agree, name for name and
statement for statement. A theorem added to a module and not named on the page
fails the gate; so does a statement edited on the page.

### The defects

Four, planted one at a time and then restored. Every one was rejected; none
was missed.

| Planted defect | Rejected by |
| --- | --- |
| The page is told a limit the kernel does not state | EN-14: the page and the kernel disagree about the largest request |
| The page writes a request past the limit rather than refusing it | the encoder check: a request past the limit was not refused |
| A theorem the proofs carry is not named on the page | EN-14: the page and the proofs disagree about which theorems there are |
| The page states a theorem differently from its proof | EN-14: the page states `gpu_usable_decides` differently from the module that proves it |
| none, before and after | passed |

### What this does not establish

- That the limit is right. It says 81 and the page now honours it; whether the
  kernel should take more is a decision for whoever raises it.
- That the statements are readable. They are the proofs' own words with four
  symbols drawn, which makes them true rather than friendly.

## EN-28, the id a model stops on

A chat loop has to stop somewhere. The published page decided where in
JavaScript: a set built from two token names, with the loop stopping at the
first member. No kernel decision covered it and no test pinned it — and
earlier the same day, the tokeniser was not matching those names at all, so
the model received an id it had never been shown and answered forever.

Which id to stop at is the model's to say, and the published file says it:
`tokenizer.ggml.eos_token_id`, which for Qwen2.5 is 151,645, the id of
`<|im_end|>`. The GGUF reader now takes it, `Tokeniser::stops_at()` carries
it, and the module exports it so the page can stop on the model's own answer
rather than on a list written beside it. A file that names none reads as one
with none; nothing here invents one, because a stop id invented for a model
that did not name one is the same bug in the other direction.

EN-28 reads it from the published header — 151,645, and the token at that id
is `<|im_end|>` — and requires a file that names none to report none, through
both readers. EN-23 requires the module to report what the crate read.

### The defects

Three, planted one at a time and then restored. Every one was rejected.

| Planted defect | Rejected by |
| --- | --- |
| The id a model stops on is read from the wrong key | EN-28: 151,643 where the file says 151,645 |
| A file that names no stop id is given one anyway | EN-28: `Some(0)` where the file names none |
| The module reports a stop id the crate did not read | EN-23: the module's answer is not the crate's |
| none, before and after | passed |

### What is deliberately not here

The **chat frame** — the text a turn is wrapped in — is not in this
repository, and this is the reason rather than an omission. The published file
carries `tokenizer.chat_template`, a Jinja template, which is the model's own
statement of how a turn is written. Writing ChatML by hand in Rust would be
the same mistake as the stop set in JavaScript: a rule about a model, kept
beside the model, correct until a model disagrees. The honest options are to
execute the template the file carries, or to model the frame as a decision and
prove it. Both are larger than a day, and neither is started.

So today the page writes the frame and this repository reads the id that ends
it. That is half the rule in the right place, and the half that was silently
wrong this morning.

## EN-29, an ONNX file read as it arrives

Milestone 2 reads models from their own ONNX exports, and those files are
mostly weights: 1.2 GB for Qwen2.5-1.5B at q4f16, all of it in one file. The
preflight reader takes a whole file in memory, which a browser cannot give it
for a file that size. `onnx_preflight::stream::Stream` is fed the file in
pieces, as the network hands them over, and reports what it passes: each
opset, each node with its inputs, outputs and attributes, and each tensor's
name, shape, type and where its data lies, inside the file or in a file
beside it. It never holds a tensor's data. When it meets one it steps over it,
and `position()` names the next byte it needs, so a caller reading by ranges
skips the weights and one reading a stream can hash them as they go by.

What it holds is a field whose bytes have not all arrived. A node is read
whole, so it is held whole; that is bounded at sixteen mebibytes
(`RECORD_LIMIT`), and a record past it is refused (FL-41), not buffered
without end. A tensor kept in another file must name it, and its offset and
length must be numbers (FL-42).

`finish` takes the file's length. The reader is not fed what it steps over,
so a file that ends inside a tensor's data leaves nothing held to show it;
only the length does. The first version of the test found that: a file cut
one byte short finished cleanly.

EN-29 builds a file shaped as the exports are (the graph before the opsets,
nodes before tensors, a megabyte of weights inside the file, a tensor kept
outside it, one written as numbers, a negative attribute, fields nothing
reads) and feeds it whole and in pieces of 1, 2, 3, 7, 61, 64 and 4096 bytes,
each time from the byte the reader asks for. Every way gives the same events;
the megabyte is never fed; the reader never holds more than the largest node
and one piece. It also checks leaving the graph early, agreement with the
whole-file reader, truncation at four places, and a field that runs past its
record.

### The defects

Three, planted one at a time and then restored. Every one was rejected.

| Planted defect | Rejected by |
| --- | --- |
| A tensor's inline data is read whole instead of stepped over | EN-29: `pieces of 1: 1048968 bytes were fed of a file whose weights alone are 1048576` |
| `finish` ignores the file's length | EN-29: `cut at 1049033`, `Ok(())` where `Err(Truncated)` was wanted |
| An offset that is not a number reads as zero | FL-42: `an offset that is a word`, a tensor at offset 0 where `Err(ExternalNotFollowed)` was wanted |
| none, before and after | passed |

The second plant was first written against the line before `cargo fmt` split
it; it applied to nothing and the test passed. It was planted again against
the formatted line and rejected.

### Run over the real exports

On 2026-11-16 the reader was run over each `onnx/model_q4f16.onnx` the hub
indexes for milestone 2, at the revision its record pins, fed only the bytes
it asked for by HTTP range reads from Hugging Face. A local probe, not part of
the gate: the gate has no network.

| Export | File | Bytes fed | Held at most | Nodes | Tensors, and where their data is | Finish |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct-ONNX at `82ac693` | 149,790 | 149,724 | 273 | 221 | 260, all in `model_q4f16.onnx_data`, 1,089,605,632 bytes | `Ok` |
| Qwen3-0.6B-ONNX at `79346d3` | 569,789,750 | 26,224,112 | 117 | 611 | 508 inline, 569,507,840 bytes | `Ok` |
| Qwen2.5-1.5B-Instruct at `d54957e` | 1,221,878,940 | 23,088,828 | 223 | 3,207 | 2,629 inline, 1,220,818,578 bytes | `Ok` |

- Every operator count, in both domains, equals the census KN-18's lists were
  read from (a separate script over the same files): the decomposed export's
  27 operators and the fused exports' 18, node for node.
- The tensor counts equal the census's initialiser counts.
- Llama's 260 external tensors add up to 1,089,605,632 bytes, which is the
  size the hub record gives for `model_q4f16.onnx_data`: the offsets and
  lengths the reader reports tile that file exactly.
- Every MatMulNBits node is `bits=4, block_size=32`, the layout KN-20 measures.

The bytes fed are what the probe fetched, not what the reader needed: it asked
for 64 KiB at each position, so after every tensor it took the first part of
the next tensor's data before the reader stepped over it. The reader itself
held at most 273 bytes.

Nothing yet hashes the tensors the reader locates, asks the kernel about
them, or builds an engine from them.

## EN-30, asking decisions 18 to 28

The kernel has answered decisions 18 to 28 since they were proved, but the
engine could not ask them: `hologram_ai_kernel_client` wrote requests for
decisions 1 to 17 only. It now has a module for each of the eleven, in the
crate's existing shape: a `request` that cannot fail and an `answer` that is
total, text the kernel is not known to give reading as `Unrecognised`.

Two of them write more than numbers. Decision 18's operator name and decision
19's family and activation are the bytes of the name, which `opname::name` and
`family::word` write and refuse when too long, since a longer name is on no
list. Decision 25 carries a list of stop ids; more than sixteen are written
with a count the length does not match, which the kernel refuses as invalid,
rather than cut to sixteen, which it would answer as if the model had fewer.

None of the eleven has an example in the compiled guest (KERNEL-VERIFICATION.md
says why), so a test cannot replay requests against recorded answers as the
older decisions' tests do. EN-30 checks the requests against the layout the
kernel reads instead: `site/decisions.json`, which `cargo xtask page` derives
from the kernel's own answer functions and the gate keeps equal to them. Each
request is written with a value in every field that fills its width and
differs from every other field's, and each must land at the offset and width
the table gives, at the length the kernel reads. Every answer text in each
decision's kernel module must read as a verdict.

### The defects

Four, planted one at a time and then restored. Every one was rejected.

| Planted defect | Rejected by |
| --- | --- |
| Decision 19's width written one byte early | EN-30: `decision 19: the field at 23, 2 bytes, is not where the kernel reads it` |
| Decision 22's binding limit and largest buffer swapped | EN-30: `decision 22: the field at 2, 8 bytes, is not where the kernel reads it` |
| Seventeen stop ids cut to sixteen | EN-30: the count written is 16 where 17 was given |
| `stop: budget` not read by the client | EN-30: ``decision module Stop answers `stop: budget`, which the client does not read`` |
| none, before and after | passed |

### What this does not show

That the kernel answers these requests as its proofs say is shown by the
proofs and, in the browser, by the page journey; this test shows only that the
engine writes what the kernel reads and reads what it writes. Nothing calls
these modules yet.

## EN-31, importing an ONNX model as it arrives

`hologram_ai_model_reader::onnx` takes a model's graph file, and any data
files beside it, as their bytes arrive. Each byte is taken once. The whole
file is hashed for the SHA-256 its hub record gives; each tensor's data is
hashed on its own for its address, the BLAKE3 of its raw bytes. The graph is
read by the streaming reader of EN-29, which now also reports where a
tensor's data begins as it steps over it, so the importer can hash those bytes
as they pass rather than learn the range afterwards.

Nothing is decided here; the kernel is asked:

- **decision 23**, each tensor's span, after the previous one, as the graph
  passes it, and every span of a data file before any of that file's bytes are
  taken;
- **decision 5**, each whole file against its record's SHA-256 and length;
- **decision 18**, each operator the graph names, in the fused style when any
  node is in the `com.microsoft` domain and the decomposed style otherwise;
- **decision 20**, each MatMulNBits weight's codes and scales against the
  node's `N`, `K` and `block_size`.

A refusal stops the import with the kernel's own verdict (FL-44 to FL-47). A
graph the reader cannot follow is FL-43, and a tensor whose data never passes
is FL-48.

A failure names what failed by its place, not by text the file wrote: a tensor
by its position among the graph's tensor records, an operator by its position
among the distinct operators, a file by its index in the record. R5 refuses
free text in a failure, and a name read from a file is text someone else
wrote; the caller holds the graph and can say what each place is called.

What comes out is the model's tensor table: each tensor's name, shape, type,
file, offset, length and address. It is the row a store keyed by address files
the weights under, and the shape of the hub's `hologram.hub.uor/v2` rows, which
do not yet exist for ONNX models.

### The tests

None of decisions 18 to 28 has an example in the guest, so EN-31 runs against
a kernel that answers in its own words: each text is checked to be one the
kernel's module gives, and admission's answers are taken from its recorded
examples. The model is fed in pieces of 1, 7, 64 and 4096 bytes and whole;
every way imports the same tensors, each named by the BLAKE3 of exactly its
bytes, and asks the kernel the same ten questions in the same order, which the
test lists request by request.

### The defects

Five, planted one at a time and then restored. Every one was rejected.

| Planted defect | Rejected by |
| --- | --- |
| A span hashed from its second byte | EN-31: `pieces of 1: DataMissing(TensorAt(0))`; FL-47 and FL-48 also fail |
| Inline spans not put to the kernel | EN-31: `the import asked the kernel other questions, or in another order`; FL-44 |
| The record's SHA-256 sent as the observed one | FL-45: `the file's own SHA-256 is the observed one` |
| A weight's scales never checked | EN-31 (the questions asked); FL-47 |
| Every graph asked in the decomposed style | EN-31 (the questions asked) |
| none, before and after | passed |

Two of the plants were first written against lines `cargo fmt` had reflowed;
they applied to nothing, and were planted again against the formatted text.

### Run over a real export

On 2026-11-16 the importer was run over `onnx/model_q4f16.onnx` of
onnx-community/Qwen3-0.6B-ONNX at `79346d3`, the revision its hub record pins,
streamed whole from Hugging Face in 1 MiB pieces into a local probe. The probe
is not part of the gate, which has no network, and is not committed.

The probe's kernel is a stand-in that applies decisions 5, 18, 20 and 23 as
their specifications state them: a second statement of those decisions, for
this run only. The kernel itself answers them in the browser.

| | Result |
| --- | --- |
| Bytes taken | 569,789,750, the record's size |
| Whole file (decision 5) | admitted: the importer's SHA-256 of the file equals the record's |
| Tensors | 508, all inline, 569,507,840 bytes; 508 distinct addresses |
| Spans (decision 23) | 508 asked, all accepted; independently, every span lies inside the file and none overlaps another |
| Operators (decision 18) | 18 asked in the fused style, all on the fused list: the same 18 the KN-18 census read |
| 4-bit weights (decision 20) | 392 asked, codes and scales of 196 MatMulNBits nodes, all the length their shapes give |
| Refusals | none |
| Time in the importer | 359 s for the whole file, on a machine shared with other work |

Ten addresses were then checked independently: the smallest tensor, the
largest (the 311,164,928-byte embedding), and eight at random, each fetched
again by range read and hashed with Python's own BLAKE3. All ten equal the
importer's addresses.

### What is not here yet

Nothing yet stores the tensors by address, is in the wasm module, or is
called by the page.

## How the engine may fail

Rule R5 says every public failure is typed, modelled, documented and covered by
a negative test. Here that means a shipped crate (any crate under `crates/`
without `publish = false`) fails only in the ways `model/failures.toml` lists.

The register is the single source. Each row is one failure: an `FL-` ID, the
owning crate, the error type, the variant, the suite of its negative scenario,
and the condition in one sentence. `CONFORMANCE.md` lists the rows, generated
by `just model-write`. The register is empty: the kernel client has no
function that can fail. A row is added in the commit that adds code that can
return it; `just new-failure` writes the row, its scenario and a failing test.

### What the gate checks

`audit-limits` (in `just model`), for every shipped crate:

- every `Result<` in shipped code names, as its error type, a type registered
  for that crate; `Result<T>` aliases such as `io::Result`, foreign types, and
  `Result` renamed on import are rejected;
- each registered type is declared once, as an `enum`, not `#[non_exhaustive]`,
  and its variants equal the crate's rows in both directions;
- no variant is a catch-all (`Other`, `Unknown`) or carries free text or an
  unnamed error (`String`, `str`, `dyn`, `anyhow`);
- every crate root has `#![forbid(clippy::unwrap_used, clippy::expect_used,
  clippy::panic)]`, and `just lint` still runs clippy over every crate and
  target. `forbid` cannot be lifted by an inner `#[allow]`. Clippy, in
  `just lint`, is what rejects the call; the gate checks the attribute is
  there. Unit tests inside a crate are exempt through `clippy.toml`.

The meta-gate (`repo-conformance`, in `just test`) requires each registered
failure to have a scenario tagged `@FL-NN @build` in its suite, and a test in
`crates/conformance/tests/<suite>.rs` whose name ends in the ID and whose body
names `Type::Variant` in code. A mention in a comment or a string does not
count, and an `#[ignore]`d or `#[cfg]`-gated test discharges nothing.

Both gates read source through one small lexer (`crates/conformance/src/source.rs`):
comments and literal contents are blanked, and a `#[cfg(test)]` item hides only
itself. The line-based reader it replaces had two holes: by its rule, after
`#[cfg(test)] use x;` it skipped everything up to the next closing brace, and a
`{` inside a test's string literal hid all the code after the test module.

### What the gate does not establish

- That a test provokes its failure. The gate checks that the test names the
  variant; the planted defect recorded for each row shows the test fails
  without the failure.
- That the engine never panics. Out-of-bounds indexing and slicing, `assert!`,
  `unreachable!` and arithmetic overflow in debug builds can still panic; none
  of them is denied, and no claim is made about them. The kernel client's
  `put` slices a fixed array at fixed offsets.
- Failure signalled any way other than `Result`: an `Option`, a sentinel value
  or a `fmt::Result` is not read as a failure.
- Code reached from outside `src/` (`#[path]`, `include!`) is rejected, not
  read; a target declared with `[lib]` or `[[bin]]` is rejected for the same
  reason.

### The kernel's verdicts

A kernel verdict is an answer, not an engine failure. The client reads every
answer into a typed verdict and never fails. Where the engine must stop
because the kernel refused, it returns one variant that carries the kernel's
verdict, such as `Refused(preflight::Verdict)`, so the policy stays in the
kernel and is never restated in Rust.

### The gate: `just model`, `just lint`, `just bdd`

Planted in a throwaway shipped crate, `crates/plant`, with one registered
failure, `FL-01` (`PlantError::Refused`), its scenario and its test. Each
defect was planted alone, the gate run, the files restored and the gate run
again. Every restore passed. The crate, its row, scenario and test were then
removed.

| Planted defect | Command | Diagnostic |
| --- | --- | --- |
| `pub fn g() -> Result<(), &'static str>` | `just model` | `` `Result<_, &'static str>`: `&'static str` is not an error type registered for plant `` |
| An extra variant `Late` in `PlantError` | `just model` | `` `PlantError::Late` has no row in model/failures.toml `` |
| A row `FL-02` for `PlantError::Stalled`, not in the enum | `just model-write && just model` | `` FL-02 registers `PlantError::Stalled`, which the enum at crates/plant/src/lib.rs:7 does not declare `` |
| `Refused(String)` | `just model` | `` `PlantError::Refused` carries `String` `` |
| A registered variant `Other` | `just model-write && just model` | `` `PlantError::Other` is a catch-all `` |
| `Some(verdict).unwrap()` in `plan` | `just lint` | `` used `unwrap()` on an `Option` value ``, pointing at the `forbid` |
| The `#![forbid(...)]` line removed | `just model` | `` does not `#![forbid(clippy::unwrap_used)]` ``, and the same for `expect_used` and `panic` |
| The `FL-01` scenario removed | `just bdd` | `FL-01 is a registered failure but has no scenario in features/suites/plant.feature` |
| The test renamed `plant_refusal` | `just bdd` | ``no test in crates/conformance/tests/plant.rs has a name ending in `fl_01` `` |
| The test marked `#[ignore]` | `just bdd` | the same |
| The test names `PlantError::Refused` only in a comment | `just bdd` | `` the test for FL-01 does not name `PlantError::Refused` in code `` |
| The `FL-01` condition edited, `CONFORMANCE.md` not regenerated | `just model` | `CONFORMANCE.md is stale: it disagrees with model/*.toml` |
| `#[cfg(test)] use core::fmt;` then an unregistered `Result` | `just model` | `` crates/plant/src/lib.rs:24: `Result<_, &'static str>` `` ... |
| none, before and after | all three | passed |

Before `FL-01`'s test was written, the scaffold left it red twice, as rule R3
requires: `plant_fl_01` panicked with `FL-01 is registered and not built`, and
the meta-gate reported ``the test for FL-01 does not name `PlantError::Refused`
in code``, because the scaffold names the variant only in its message.

The same rejections are kept as unit tests over scratch repositories:
`audit::tests` in `xtask/src/audit.rs`, `source::tests` in
`crates/conformance/src/source.rs`, and
`a_scaffolded_failure_is_red_until_its_test_names_the_variant` in
`xtask/src/scaffold.rs`.

## EN-32, WordPiece

The encoders of milestone 2, all-MiniLM-L6-v2 and DistilBERT SST-2, read text
as WordPiece ids, which the byte-pair tokeniser refuses (EN-16). `WordPiece`
in `crates/tokenizer/src/wordpiece.rs` follows BERT's normaliser, BERT's
pre-tokeniser, the WordPiece model and a one-sentence template, each written
from the authority's source (HuggingFace `tokenizers` 0.21.4,
`normalizers/bert.rs`, `pre_tokenizers/bert.rs`, `models/wordpiece`). Which
characters are controls, punctuation and nonspacing marks, and how each
decomposes, are read from `unicode_categories` 0.1.1 and
`unicode-normalization-alignments` 0.1.12, the tables the authority reads;
both were already locked as the authority's dependencies and are now
production dependencies of the tokeniser, which `cargo deny` accepts.

Padding is not applied: it adds ids the encoder masks out, and an encoder run
on the sentence alone gives the same states for it. Truncation is applied as
the file states.

### The gate: `cargo test -p repo-conformance --test engine engine_en_32`

Run on 2026-11-16 in WSL with the locked toolchain. Each defect was planted
alone in `wordpiece.rs`, the test run, the file restored.

| Planted defect | Diagnostic |
| --- | --- |
| Accents never stripped | `the ids differ for ...` (engine.rs, the comparison with the authority) |
| Punctuation other than `.` not split off | the same |
| A word with no full covering gives the pieces found before the gap | the same |
| Added tokens not matched in the text as written | the same |
| Truncation ignored | the same |
| Chinese characters not spaced | the same |
| none, before and after | passed |

Before the test was written the scaffold left `engine_en_32` red, as R3
requires: it panicked with `EN-32 is registered and not built`.

The two real files, at the hub's pinned revisions, were also compared once
with the authority on ten texts each, a 300-word text among them
(all-MiniLM-L6-v2 truncates it to 128 ids, as its file states): every id
equal. That comparison is not in the suite: the files are 0.7 MB each and the
suite builds its own.

### What EN-32 does not establish

Decoding WordPiece ids back to text, a pair of sentences, and padding are
not built: the encoders read one sentence and give a vector or a label. The
tokeniser is not yet reachable from the wasm module or the page.

## EN-33, the ONNX import in the page's module

The importer of EN-31 asks the kernel synchronously. In the page the kernel is
the holo's guest, verified before it is compiled, and `runGuest` answers
synchronously too; so the module asks it through one import, `kernel.ask`,
and the importer runs in the page unchanged. The import is handed the
request and a buffer of 256 bytes the calling function holds; the page writes
the answer there. That call is the crate's one `unsafe` block, with its
safety stated beside it. `openModule` in `site/app/engine/loader.mjs`
supplies the import and `setKernel` names the kernel; until it is named,
every question is answered with nothing, which the importer refuses.

The stream reader now keeps each node's name (field 3 of NodeProto), and the
importer hands out its nodes: an export names a weight far more plainly by
the node that reads it (`/model/layers.0/attn/q_proj/MatMul_Q4`) than by its
own name (`onnx::MatMul_9333`). The module's `onnx_finish` answers the JSON
the role map of GP-12 reads, with an input that is not a tensor of the model
written as null, and each tensor's file, offset, length and BLAKE3 address.

The module grows from 1,636,248 to 1,788,247 bytes (505,740 to 561,390
gzipped), 85 per cent of the 2 MiB a page may ask for, chiefly `serde_json`,
already locked.

### The gate: `cargo test -p repo-conformance --test engine engine_en_33 engine_fl_49`

Run on 2026-11-17 in WSL with the locked toolchain.

| Planted defect | Diagnostic |
| --- | --- |
| The kernel's answer dropped before the importer reads it | `onnx_finish in pieces of 7: Import(...)` (a call said no) |
| Every input named, weights or not | the nodes differ from the ones expected |
| Every address written from the first tensor's | `w_scales: address` |
| A data file named by its index, not as the graph names it (a mistake made while writing this) | `onnx_finish in pieces of 7: Import(DataMissing(TensorAt(2)))` |
| none, before and after | passed |

The scaffold left both tests red first, as R3 requires.

### A real export, once, by hand

On 2026-11-17, `onnx-community/Qwen3-0.6B-ONNX` at 79346d3, `onnx/model_q4f16.onnx`
(569,789,750 bytes), streamed from its origin through the release module in
Node, the module asking the published kernel (`app.holo` from the deployed
site, verified, its guest run with `runGuest`): 919 questions, every answer
accepted; 508 tensors named; 611 nodes. The role map placed 310 tensors from
the importer's own table, every one the same as from the graph recorded for
GP-12 by a separate reader. 373 s on a shared machine, most of it download.
Not in the suite: it reads 0.57 GB from the network.

## EN-34, exact q4 and ternary products

`hologram-ai-exact` computes a q4 or ternary product in integers from the
stored bytes to one final rounding, and has no floating-point type: it takes
the input as binary32 bit patterns and returns binary32 bit patterns. Each
binary32 or half scale and each input is read from its bits as the exact
`mantissa * 2^exponent` it stands for. The input is scaled by one power of two
onto `i8`, rounded to the nearest, ties to even, by integer shifts. Each
group's sum is taken by `uor-matmul`'s `dot_ref`. A row's group sums times
their scales' mantissas are placed, at their exponents less the input's, in
`uor-matmul`'s complete register (`Complete<12, -320>`), which holds every bit
of every row any finite inputs and scales can give. That exact value is
rounded once to the nearest binary32, ties to even, from the register's leading
bits and its sticky bit. A scale that is infinity or NaN is refused (`FL-53`).

Two earlier forms are measured below: one that added the groups in `f64`, and
one that added them in `i128` and refused a row whose scales were too far apart
(a failure that the complete register removes). On both real models and both
inputs, all three give the same output bytes.

### What runs on uor-matmul, and what is ours

| Piece | Where | Why |
| --- | --- | --- |
| A group's sum of products, exactly | `uor-matmul` `dot_ref` | its accumulator cannot overflow |
| q4 and ternary codes to integers | `uor-matmul` `Grid` | the table codec |
| A row's exact sum over its groups | `uor-matmul` `Complete` (`add_scaled`) | a fixed-point register over the whole exponent range, with no rounding |
| Reading a binary32 or half from its bits | ours (`Dyadic`, `Format::read`) | `uor-matmul` 0.1.0 reads floats only through `FloatElement::decode(self)`, for `f32` and `f64` values, so using it would put a float-typed value in the crate; it has no half or bfloat16. To be proposed upstream as a reader over bit patterns |
| Rounding the register to binary32 bits | ours (`round_to_binary32`), over `Complete`'s `magnitude_high_bit`, `magnitude_bit` and `magnitude_any_below` | `uor-matmul`'s own `EncodeFrom<Complete>` returns an `f32` value. To be proposed upstream as an encoder to bit patterns |
| Scaling the input onto `i8` | ours (`quantise`) | an activation scheme, not arithmetic `uor-matmul` names; brief step 3 chooses it |

### The gate: `cargo test -p repo-conformance --test engine -- engine_en_34 engine_fl_50 engine_fl_51 engine_fl_52 engine_fl_53`

Run on 2026-11-17 in WSL with the workspace toolchain. Each defect was planted
alone, the five tests run, the files restored.

| Planted defect | Rejected by |
| --- | --- |
| an input tie rounded away from zero | `engine_en_34`: `900 is not rounded to the nearest code, ties to even` |
| the binary32 rounding ignores the sticky bit and evenness | `engine_en_34`: `-880.87787 is not the stated -880.8778` |
| the binary32 rounding truncates | `engine_en_34`: a ternary row one unit below the stated value |
| a float's exponent read one too low | `engine_en_34`: input codes not the nearest |
| the register's floor at 2^-60, dropping lower bits | `engine_en_34`: `0 is not the stated -0.000000000000000000000000000008365156` |
| the input's power applied the wrong way | `engine_en_34`: `-125719 is not the stated -122.77246` |
| an infinite or NaN scale read as a number | `engine_fl_53`, and `engine_fl_50` |
| the input exponent never steps down | `engine_en_34`: `the step is too small` |
| an `f32` constant in the crate | `engine_en_34`: `names a float type outside a comment` |
| the module reads half scales as binary32 | `engine_en_34`: the module's bytes differ from the crate's |
| none, before and after | passed |

What each output is checked against: the exact sum worked in the test from
each scale's `f64` bits, independently of the crate's reading of binary32 and
half bits, and rounded once by Rust's own `f64`-to-`f32` conversion; the test
asserts that sum has at most 53 bits, so that conversion is the one rounding.
Each q4 row is also taken with its groups, scales and input columns in reverse
order, and must give the same bits. The crate's source may not name `f32` or
`f64` outside a comment.

### The first form's gate: `cargo test -p repo-conformance --test engine -- engine_en_34 engine_fl_50 engine_fl_51 engine_fl_52`

Run on 2026-11-17 in the pinned SDK image. Each defect was planted alone, the
four tests run, the file restored.

| Planted defect | Rejected by |
| --- | --- |
| q4 nibbles read high first | `engine_en_34`: the group sums differ |
| q4 code 0 decodes to +8 | `engine_en_34`: the group sums differ |
| Ternary sign inverted | `engine_en_34`: the group sums differ |
| Groups added in reverse order | `engine_en_34`: `-20 is not the stated 0` |
| The total rounded to `f32` before the rescale | `engine_en_34`: `inf is not the stated 2.8e32` |
| Input ties rounded away from zero | `engine_en_34`: not rounded to the nearest code, ties to even |
| The exponent never steps down | `engine_en_34`: the step is too small |
| A non-finite input admitted | `engine_fl_50` |
| The q4 group check dropped | `engine_fl_51`, and `engine_en_34` through the module |
| The q4 scale length unchecked | `engine_fl_52` (the crate indexes past its scales) |
| The ternary group check dropped | `engine_fl_51` |
| The module reads f16 scales as f32 | `engine_en_34`: the module's bytes differ from the crate's |
| Subnormal half scales halved | `engine_en_34`: `-0.00134 is not the stated -0.00268` |
| none, before and after | passed |

A first round missed six of these, and the code or cases changed rather than
the record. Four needed cases the generator never drew: inputs exactly
halfway between codes, subnormal half scales, groups whose order decides the
answer, and a total beyond `f32` that the rescale brings back. Two were code
no input can reach: a loop that raised the exponent (the float's own exponent
already puts the largest input in [64, 128)) and request-level group checks
that repeat the group sums' own. Both were deleted.

### Run over two real models

A measurement, not part of the suite, over every matrix of two models, one
synthetic input per matrix. Bonsai 1.7B at
`HOLOGRAMTECH/q-ternary-bonsai-1.7b@c6d6a12` (sha256 `acf0b5ad…`): every
ternary matrix. Qwen2.5-1.5B-Instruct Q4_0 at
`Qwen/Qwen2.5-1.5B-Instruct-GGUF@91cad51` (sha256 `dcd819ff…`): every Q4_0
matrix, repacked to the engine's q4f16 layout (the same codes and scales, the
nibbles in another order); the Q6_K output head was not measured. Inputs
roughly normal, and the same with 0.2% of values at 40 times.

With the integer form, on 2026-11-17, natively in WSL (one thread):

| Model | Matrices | Weights | Group sums checked | Wrong | Rows past bound | Relative error (L2), normal / with outliers | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bonsai 1.7B | 198 | 2,030,522,368 | 15,863,456 | 0 | 0 | 0.78% / 9.9% | 466 to 479 M weights/s |
| Qwen2.5-1.5B | 197 | 1,543,569,408 | 48,236,544 | 0 | 0 | 0.77% / 8.9% | 512 to 519 M weights/s |

With the complete register, on 2026-11-17, natively in WSL (one thread), the
same counts and the same errors, and the same output bytes as the `i128` form
for both models and both inputs. Throughput: 412 M weights/s (Bonsai) and 313 M
weights/s (Qwen) on the ordinary input; on the first pass, with outliers, 211
and 189, likely sharing the machine. The register costs about a third of the
throughput against the `i128` form; a narrower register sized to the scales a
tensor actually has is the obvious next measurement.

The module the page loads, built from the same source, gave the same bytes as
native code for Bonsai's 151,669 by 2,048 head and a Qwen 1,536 by 8,960 matrix,
for both inputs. The errors are those of the first form to the digits shown:
they are the input's rounding onto eight bits, which this change does not touch.

The first form's run had reported one Qwen matrix whose module bytes differed
from native code. The cause was a module built during that form's
planted-defect run, with half scales read as `f32`: rebuilt from that form's
committed source (`d08bc2b`, the same tree as `cafac69`), it gave identical
bytes on the saved Qwen request, checked again here. `engine_en_34` builds the
module it compares, so the gate cannot run against a stale one.

### What EN-34 does not establish

Exact is not the same as accurate. Rounding the input onto eight bits with one
power of two for the whole vector costs under 1% on ordinary inputs, but an
outlier sets the step for every value, and the error rises to about 9%. A
finer input scale, per group, is the next measurement, and an engine using
these products has to be judged on its answers, not on these figures. Softmax, normalisation and rotary encoding remain floating point,
and nothing on the page calls the module yet.

The input still arrives as binary32 bit patterns and leaves as binary32 bit
patterns, because the stages around the product are floating point until they
are made integer (brief step 4). The crate holds no float value. The reference the test checks
the rounding against is Rust's own conversion; that the crate's reading of
half and `f32` bits matches IEEE 754 is tested on the cases drawn, not proved.

## EN-35, the crate's group sums confirmed by the kernel

The engine's rule is that an integer stage is defined and proved in LexLean
and the fast path is gated equal to it. For the q4 group sum the definition
is `Kernel.Block` (KN-34); this gate ties `hologram-ai-exact` to it. Eight
blocks, at both ends of the sum's range, with every weight zero, and four
drawn at random, are summed by the crate's own path and written by
`block_check::request`. `crates/conformance/tests/kernel/claims.json` lists 17
claims: each sum `exact`, each sum plus one `differs`, and one 52-byte request
`invalid request`. `engine_en_35` checks that every crate sum and sum plus one
is listed, and that the `app` recipe runs
`crates/conformance/tests/kernel/replay.mjs` on the holo it has just built.
That script runs every claim through the compiled guest, the one built from
the proved definitions, and fails unless each gets the answer listed.

The claims are not the holo's examples. A first form carried them as decision
34's examples, and CI refused the holo: `error[LLT4001]: phase link: invalid
semantic-module JSON: recursion limit exceeded`, at 125 examples, past the
limit "The limit on examples" in KERNEL-VERIFICATION.md describes. A claims
file has no such limit.

`engine_en_35` also reads each request as `Kernel.Block` reads it, by its
fields, `byteFromEnd` and the high half for an even index, so a wrong layout
fails in the Rust suite as well as in the guest.

### The guest, locally: `node crates/conformance/tests/kernel/replay.mjs app.holo claims.json`

The holo published from main at 2026-11-18 (`site/app.holo` on the project's
Pages, which carries decision 34) answered all 17 claims as listed. With the
first claim's answer changed to `differs` it failed:
`1 of 17 claims answered otherwise ... the guest answers exact`.

### The gate: `cargo test -p repo-conformance --test engine -- engine_en_35`

Run on 2026-11-18 in WSL with the pinned toolchain (cargo 1.97.1), not in the
SDK image. Each defect was planted alone and the file restored.

| Planted defect | Result |
| --- | --- |
| the crate reads q4 nibbles high first | the kernel reads another block |
| the crate decodes q4 code 0 as +8 | the kernel reads another block |
| the client writes the low half for an even index | the kernel reads another block |
| the client offsets the claim by 32767 | the kernel reads another block |
| the client pairs a weight with its neighbour's activation | the kernel reads another block |
| an `exact` claim listed as `differs` | the claims lack 1 the crate makes of decision 34 |
| the replay dropped from the `app` recipe | the app recipe replays the claims through the holo's guest (missing) |
| none | passed |

### What EN-35 does not establish

The replay on CI runs against this branch's holo; locally it ran against the
published one. The gate covers the group sum only: the crate's quantisation of
the input and its recombination of the sums with the scales are still
floating point with no LexLean definition, which the engine's rule does not
allow in a shipped path.

## EN-36, the crate's scaled sums confirmed by the kernel

`hologram-ai-exact` reads each IEEE half scale as an exact `Dyadic`, a
mantissa times a power of two no lower than 2^-24 (EN-34, #52), and a row
combines each group's sum times that mantissa at that exponent. `Kernel.Scale`
(KN-37) defines one such product as a count of 2^-24, which is the sum times
the mantissa shifted up by the exponent's distance from -24. Nine groups, the
sums at both ends of a block's range and zero, times halves that are one, the
largest, the smallest subnormal, the smallest normal, negative and negative
zero, give 18 claims from the crate's `Dyadic::from_binary16`, each product
`exact` and each product plus one `differs`, with one `not finite` and one
`invalid request`. They join `crates/conformance/tests/kernel/claims.json`,
which the `app` recipe replays through the guest it builds.

The first form of this change added its own `half_units` to the crate; #52
reached main first with `Dyadic`, so the gate holds `Dyadic` to the kernel
instead, and the 18 claims computed from `half_units` were unchanged when
computed from `Dyadic`.

### The gate: `cargo test -p repo-conformance --test engine -- engine_en_36`

Run on 2026-11-18 in WSL with the pinned toolchain. Each defect was planted
alone in `hologram-ai-exact` or the client and the file restored.

| Planted defect | Result |
| --- | --- |
| a normal half's exponent read one too high | the kernel reads another group |
| a half's sign dropped | the kernel reads another group |
| a subnormal half read with the hidden bit | the kernel reads another group |
| the half's exponent bias taken as 16 | the kernel reads another group |
| the client writes the sign of a positive claim | the kernel reads another group |
| the client writes the half little-endian | the kernel reads another group |
| an `exact` claim listed as `differs` | the claims lack 1 |
| none | passed |

### What EN-36 does not establish

The kernel checks one group's product at a time; summing a row's products in
the complete register and rounding the row to binary32 are not yet defined in
LexLean, and the input's quantisation is not either. Only the published
guest, which has no decision 37, could be run locally; this holo's guest runs
the decision-37 claims on CI.
