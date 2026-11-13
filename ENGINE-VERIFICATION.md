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

`crates/chat` keeps a conversation: where it is in the journey, how many
tokens its system prompt and newest message hold, and each question-and-answer
pair in turn. It proposes a trim, asks for a decode bucket and a token budget,
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
| An ONNX file from a hub reads as its publisher meant it | not tested: every file in this suite is written here | not tested |
| `EN-06`: a buffer's label is the BLAKE3 hash of its bytes | `engine_en_06`, over all 35 published test vectors, whose vendored file is checked against the checksum in `model/authorities.toml` first | reproduced from the authority, not established here |
| `EN-07`: a download's label and length do not depend on how its bytes were split | `engine_en_07`, over ten lengths either side of the hash's own block and chunk sizes, in six chunk sizes each | tested |
| `EN-08`: a download is admitted exactly when the kernel admits it | `engine_en_08` | tested |
| `FL-14`, `FL-15`: the download fails only as `model/failures.toml` lists | `engine_fl_14` and `engine_fl_15` | tested |
| `EN-09`: a conversation takes the step the kernel gives and stays where it was when refused | `engine_en_09`, driving the whole journey through the kernel's own examples | tested |
| `EN-10`: the trim proposed drops whole pairs from the oldest, the fewest that fit | `engine_en_10`, over twelve conversations including the empty one and the one too long to fit at all | tested |
| `EN-11`: pairs are forgotten only on the kernel's acceptance, and its numbers are read as numbers | `engine_en_11` | tested |
| `FL-16` to `FL-19`: the conversation fails only as `model/failures.toml` lists | `engine_fl_16` to `engine_fl_19` | tested |
| A conversation's token counts are the ones a tokeniser would give | not tested: no tokeniser is here, and the counts are the caller's |
| That BLAKE3 is what its paper says it is | cited, never established here: the hash is the authority's own implementation, checked against the authority's own vectors | not proved |
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

### What EN-09 to EN-11 do not establish

- That the token counts are a tokeniser's. No tokeniser is here; the counts are
  the caller's, and a wrong count is a wrong question honestly asked.
- That a whole journey is what the eight steps add up to. The kernel proves
  each step; the claim about sequences is argued, as `KERNEL-VERIFICATION.md`
  records.
- That a conversation generates anything. Nothing here runs a model: the
  conversation keeps counts and asks what may be done with them.

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
