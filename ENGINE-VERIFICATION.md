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
