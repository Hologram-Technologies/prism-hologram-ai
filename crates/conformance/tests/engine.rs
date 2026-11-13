//! The `engine` suite: one test per scenario in `features/suites/engine.feature`.
//!
//! The engine is tested against the kernel's own examples, the requests and
//! answers the holo carries and replays in its compiled guest. Nothing here
//! restates them: they are read from `app/src/Kernel.lex.tex`, so an example
//! added to or changed in the model is tested the next time this runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use hologram_ai_model_reader::{config, header, Kernel};
use hologram_ai_onnx_preflight as onnx;

use hologram_ai_kernel_client::{
    admission, bucket, generation, journey, operators, plan, preflight, range, shape, tier, trim,
    INVALID_REQUEST,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/conformance is two below the root")
        .to_path_buf()
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(
        text.len().is_multiple_of(2),
        "an odd number of hex digits: {text}"
    );
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("a hex byte"))
        .collect()
}

/// Every example the kernel carries, as (request, answer).
fn examples() -> Vec<(Vec<u8>, Vec<u8>)> {
    let model = std::fs::read_to_string(root().join("app/src/Kernel.lex.tex"))
        .expect("the kernel's entry module");
    let mut out = Vec::new();
    let mut rest = model.as_str();
    while let Some(at) = rest.find("request := 0x") {
        rest = &rest[at + "request := 0x".len()..];
        let (request, after) = rest
            .split_once(", response := 0x")
            .expect("an example's answer follows its request");
        let (response, after) = after.split_once('}').expect("an example is closed");
        out.push((unhex(request), unhex(response)));
        rest = after;
    }
    out
}

fn word(bytes: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

fn label(bytes: &[u8], at: usize) -> [u8; 32] {
    bytes[at..at + 32].try_into().expect("thirty-two bytes")
}

/// The request the client writes for the fields `request` itself carries, when
/// `request` is a well-formed request of its decision; `None` otherwise.
fn rewritten(request: &[u8]) -> Option<Vec<u8>> {
    let r = request;
    Some(match (*r.first()?, r.len()) {
        (1, 3) => journey::request(
            journey::State::from_code(r[1])?,
            journey::Event::from_code(r[2])?,
        )
        .to_vec(),
        (2, 5) if r[1] <= 1 && r[3] <= 1 => {
            preflight::request(r[1] == 1, r[2], r[3] == 1, r[4]).to_vec()
        }
        (3, 49) => plan::request(
            word(r, 1),
            word(r, 9),
            word(r, 17),
            word(r, 25),
            word(r, 33),
            word(r, 41),
        )
        .to_vec(),
        (4, 17) => tier::request(word(r, 1), word(r, 9)).to_vec(),
        (5, 81) => {
            admission::request(&label(r, 1), &label(r, 33), word(r, 65), word(r, 73)).to_vec()
        }
        (6, 33) => range::request(word(r, 1), word(r, 9), word(r, 17), word(r, 25)).to_vec(),
        (7, 41) => trim::request(
            word(r, 1),
            word(r, 9),
            word(r, 17),
            word(r, 25),
            word(r, 33),
        )
        .to_vec(),
        (8, 25) => bucket::request(word(r, 1), word(r, 9), word(r, 17)).to_vec(),
        (9, 25) => generation::request(word(r, 1), word(r, 9), word(r, 17)).to_vec(),
        (10, 33) => operators::request(word(r, 1), word(r, 9), word(r, 17), word(r, 25)).to_vec(),
        (11, 33) => shape::request(word(r, 1), word(r, 9), word(r, 17), word(r, 25)).to_vec(),
        _ => return None,
    })
}

/// What the client reads `answer` as, for the decision `tag` names: the
/// variant's name, and the number it carries when it carries one. `None` when
/// `tag` names no decision.
fn read(tag: u8, answer: &[u8]) -> Option<(String, Option<u64>)> {
    let plain = |name: String| Some((name, None));
    match tag {
        1 => plain(format!("{:?}", journey::answer(answer))),
        2 => plain(format!("{:?}", preflight::answer(answer))),
        3 => plain(format!("{:?}", plan::answer(answer))),
        4 => plain(format!("{:?}", tier::answer(answer))),
        5 => plain(format!("{:?}", admission::answer(answer))),
        6 => plain(format!("{:?}", range::answer(answer))),
        7 => plain(format!("{:?}", trim::answer(answer))),
        8 => Some(match bucket::answer(answer) {
            bucket::Answer::Bucket(n) => ("Bucket".into(), Some(n)),
            other => (format!("{other:?}"), None),
        }),
        9 => Some(match generation::answer(answer) {
            generation::Answer::Budget(n) => ("Budget".into(), Some(n)),
            other => (format!("{other:?}"), None),
        }),
        10 => plain(format!("{:?}", operators::answer(answer))),
        11 => Some(match shape::answer(answer) {
            shape::Answer::Size(n) => ("Size".into(), Some(n)),
            other => (format!("{other:?}"), None),
        }),
        _ => None,
    }
}

/// Every answer each decision can give, by the name the client reads it as.
/// `Unrecognised` is absent on purpose: no example may read as it.
fn vocabulary() -> BTreeMap<u8, BTreeSet<&'static str>> {
    BTreeMap::from([
        (
            1,
            BTreeSet::from([
                "To(Idle)",
                "To(Preflight)",
                "To(Download)",
                "To(Compile)",
                "To(Materialize)",
                "To(Ready)",
                "To(Generating)",
                "To(Failed)",
                "Refused",
                "Invalid",
            ]),
        ),
        (
            2,
            BTreeSet::from([
                "Supported",
                "Derived",
                "MissingConfigKey",
                "UnsupportedFeature",
                "ManifestDoesNotFit",
                "Invalid",
            ]),
        ),
        (
            3,
            BTreeSet::from([
                "Plan",
                "CannotRun",
                "StageDoesNotFit",
                "StageTooSmall",
                "StagesDoNotCover",
                "Invalid",
            ]),
        ),
        (4, BTreeSet::from(["Int8", "Int4", "Int8Paged", "Invalid"])),
        (5, BTreeSet::from(["Admit", "Refuse", "Invalid"])),
        (6, BTreeSet::from(["Accept", "Reject", "Invalid"])),
        (
            7,
            BTreeSet::from([
                "Trim",
                "TooLong",
                "DoesNotFit",
                "DropsTooMany",
                "DroppedExceedsTotal",
                "Invalid",
            ]),
        ),
        (
            8,
            BTreeSet::from([
                "Bucket",
                "ContextExceeded",
                "BucketExceedsMaximum",
                "Invalid",
            ]),
        ),
        (9, BTreeSet::from(["Budget", "ContextFull", "Invalid"])),
        (
            10,
            BTreeSet::from([
                "Importable",
                "OpsetBelowMinimum",
                "OpsetAboveMaximum",
                "UnsupportedOperator",
                "Invalid",
            ]),
        ),
        (
            11,
            BTreeSet::from(["Size", "BelowMinimum", "AboveMaximum", "Invalid"]),
        ),
    ])
}

/// `EN-01`: The kernel client writes each request exactly as the kernel examples have it and reads every answer the kernel gives
#[test]
fn engine_en_01() {
    let examples = examples();
    assert!(
        examples.len() >= 100,
        "the kernel carries about a hundred examples; found {}",
        examples.len()
    );
    let vocabulary = vocabulary();

    let mut written: BTreeMap<u8, usize> = BTreeMap::new();
    let mut seen: BTreeMap<u8, BTreeSet<String>> = BTreeMap::new();
    let mut texts: BTreeMap<(u8, String), Vec<u8>> = BTreeMap::new();

    for (request, answer) in &examples {
        // Writing: a well-formed request is reproduced, byte for byte, from
        // the fields it carries. A field written at the wrong offset, in the
        // wrong order or at the wrong width cannot survive this.
        if let Some(bytes) = rewritten(request) {
            assert_eq!(
                &bytes, request,
                "the client writes decision {}'s request differently from the kernel's example",
                request[0]
            );
            *written.entry(request[0]).or_default() += 1;
        }

        // Reading: every answer the kernel gives is one the client knows.
        let tag = request.first().copied().unwrap_or(0);
        let Some((name, number)) = read(tag, answer) else {
            assert_eq!(
                answer, INVALID_REQUEST,
                "a request that names no decision must be answered `invalid request`"
            );
            continue;
        };
        assert!(
            vocabulary[&tag].contains(name.as_str()),
            "decision {tag} answered `{}`, which the client reads as `{name}`",
            String::from_utf8_lossy(answer)
        );
        if let Some(number) = number {
            assert_eq!(
                number.to_string().as_bytes(),
                answer,
                "decision {tag}: the number read is not the number the kernel wrote"
            );
        } else if let Some(other) = texts.insert((tag, name.clone()), answer.clone()) {
            assert_eq!(
                &other, answer,
                "decision {tag}: two different answers both read as `{name}`"
            );
        }
        seen.entry(tag).or_default().insert(name);
    }

    // Every decision's requests were written, and every answer in its
    // vocabulary was given by some example: a variant no example produces
    // would be one the client invented.
    for (tag, names) in &vocabulary {
        assert!(
            written.get(tag).copied().unwrap_or(0) >= 5,
            "decision {tag}: fewer than five example requests were reproduced"
        );
        let given: BTreeSet<&str> = seen
            .get(tag)
            .map(|set| set.iter().map(String::as_str).collect())
            .unwrap_or_default();
        assert_eq!(
            &given, names,
            "decision {tag}: the answers the examples give are not the answers the client knows"
        );
    }

    // Anything else is unrecognised, never mistaken for a verdict: wrong
    // case, trailing bytes, a number written as the kernel never writes one.
    for garbage in [
        &b""[..],
        b"Admit",
        b"admit\n",
        b"ready ",
        b"007",
        b"-1",
        b"18446744073709551616",
    ] {
        assert_eq!(journey::answer(garbage), journey::Step::Unrecognised);
        assert_eq!(preflight::answer(garbage), preflight::Verdict::Unrecognised);
        assert_eq!(plan::answer(garbage), plan::Verdict::Unrecognised);
        assert_eq!(tier::answer(garbage), tier::Tier::Unrecognised);
        assert_eq!(admission::answer(garbage), admission::Verdict::Unrecognised);
        assert_eq!(range::answer(garbage), range::Verdict::Unrecognised);
        assert_eq!(trim::answer(garbage), trim::Verdict::Unrecognised);
        assert_eq!(bucket::answer(garbage), bucket::Answer::Unrecognised);
        assert_eq!(
            generation::answer(garbage),
            generation::Answer::Unrecognised
        );
        assert_eq!(shape::answer(garbage), shape::Answer::Unrecognised);
        assert_eq!(operators::answer(garbage), operators::Verdict::Unrecognised);
    }

    // A graph may be built on one verdict and on nothing else.
    for (verdict, allowed) in [
        (operators::Verdict::Importable, true),
        (operators::Verdict::OpsetBelowMinimum, false),
        (operators::Verdict::OpsetAboveMaximum, false),
        (operators::Verdict::UnsupportedOperator, false),
        (operators::Verdict::Invalid, false),
        (operators::Verdict::Unrecognised, false),
    ] {
        assert_eq!(
            verdict.authorises_import(),
            allowed,
            "{verdict:?} and importing"
        );
    }

    // Weights may download on the two runnable verdicts and on nothing else.
    for (verdict, allowed) in [
        (preflight::Verdict::Supported, true),
        (preflight::Verdict::Derived, true),
        (preflight::Verdict::MissingConfigKey, false),
        (preflight::Verdict::UnsupportedFeature, false),
        (preflight::Verdict::ManifestDoesNotFit, false),
        (preflight::Verdict::Invalid, false),
        (preflight::Verdict::Unrecognised, false),
    ] {
        assert_eq!(
            verdict.authorises_download(),
            allowed,
            "{verdict:?} and downloading"
        );
    }
}

// ─── The model reader ───────────────────────────────────────────────────────
//
// The reader turns a model's own two documents into the kernel's questions.
// The tests below put real files to it: a safetensors file is built here, byte
// for byte, so what the reader recovers can be compared with where the test put
// it. Policy is never restated: every answer about a byte range comes from the
// kernel's own recorded examples, or is the kernel's own text for an accepted
// range, replayed by a double that keeps what it was asked.

/// A safetensors file: eight bytes of header length, `header` padded with
/// spaces to `header_length`, then zero bytes to `total`.
fn safetensors(header: &str, header_length: usize, total: usize) -> Vec<u8> {
    assert!(
        header.len() <= header_length,
        "the header is longer than the length it declares"
    );
    let mut out = (header_length as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.resize(8 + header_length, b' ');
    assert!(
        total >= out.len(),
        "the file is shorter than its own header"
    );
    out.resize(total, 0);
    out
}

/// Every tensor-range example the kernel carries, as request to answer.
fn range_examples() -> BTreeMap<Vec<u8>, Vec<u8>> {
    examples()
        .into_iter()
        .filter(|(request, _)| request.first() == Some(&6))
        .collect()
}

/// A kernel that answers only what the kernel's own examples record. A test
/// using it asks nothing the compiled guest has not been seen to answer.
struct Recorded {
    answers: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl Kernel for Recorded {
    fn ask(&mut self, request: &[u8]) -> Vec<u8> {
        self.answers
            .get(request)
            .unwrap_or_else(|| {
                panic!(
                    "the kernel has no example for this request: {}",
                    request
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                )
            })
            .clone()
    }
}

/// A kernel that accepts every range, in the kernel's own words, and keeps
/// every request it was asked.
struct Accepting {
    accept: Vec<u8>,
    asked: Vec<Vec<u8>>,
}

impl Accepting {
    /// The accepting answer is not written here: it is the answer the kernel
    /// gives to an example it accepts.
    fn new() -> Self {
        let accept = range_examples()
            .into_values()
            .find(|answer| range::answer(answer) == range::Verdict::Accept)
            .expect("the kernel accepts some example range");
        Self {
            accept,
            asked: Vec::new(),
        }
    }
}

impl Kernel for Accepting {
    fn ask(&mut self, request: &[u8]) -> Vec<u8> {
        self.asked.push(request.to_vec());
        self.accept.clone()
    }
}

/// A whole config, as a checkpoint on a hub carries it.
const WHOLE_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 64,
  "intermediate_size": 128,
  "max_position_embeddings": 128,
  "model_type": "llama",
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-6,
  "vocab_size": 512
}"#;

/// `EN-02`: A safetensors header is read into its tensors in file order, with every tensor byte range accepted by the kernel
#[test]
fn engine_en_02() {
    // One tensor, in a file whose range question is one of the kernel's own
    // examples: the guest has been seen to accept exactly this range.
    let one = safetensors(
        r#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[4,1024],"data_offsets":[0,4096]}}"#,
        120,
        4224,
    );
    let tensors = header::read(
        &one,
        &mut Recorded {
            answers: range_examples(),
        },
    )
    .expect("the kernel accepts this file's one range");
    assert_eq!(
        tensors,
        vec![header::Tensor {
            name: "model.embed_tokens.weight".into(),
            dtype: "F32".into(),
            shape: vec![4, 1024],
            offset: 128,
            length: 4096,
        }],
        "the tensor the header describes is not the one that was read"
    );

    // Three tensors, listed in neither the order their bytes lie nor the order
    // their names sort, beside the metadata the format reserves.
    let many = safetensors(
        r#"{"zebra":{"dtype":"F32","shape":[2],"data_offsets":[8,12]},"__metadata__":{"format":"pt"},"alpha":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"mid":{"dtype":"F32","shape":[0],"data_offsets":[12,12]}}"#,
        232,
        252,
    );
    let mut kernel = Accepting::new();
    let tensors = header::read(&many, &mut kernel).expect("every range is accepted");
    assert_eq!(
        tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["alpha", "zebra", "mid"],
        "the tensors are not in the order their bytes lie"
    );
    assert_eq!(
        tensors
            .iter()
            .map(|t| (t.dtype.as_str(), t.shape.clone(), t.offset, t.length))
            .collect::<Vec<_>>(),
        [
            ("BF16", vec![2, 2], 240, 8),
            ("F32", vec![2], 248, 4),
            ("F32", vec![0], 252, 0),
        ],
        "a tensor was not placed where the header puts it"
    );

    // Each tensor was asked about once, in file order, with the previous
    // tensor's end as its previous end: the fold the kernel's decision is
    // about. The numbers come from where this test put the tensors.
    assert_eq!(
        kernel.asked,
        [
            range::request(252, 240, 240, 8).to_vec(),
            range::request(252, 248, 248, 4).to_vec(),
            range::request(252, 252, 252, 0).to_vec(),
        ],
        "the kernel was not asked about the file the test built"
    );
}

/// `EN-03`: A model config is read into the preflight question: a named family is recognised, every missing required key is counted, and an unsupported activation is flagged
#[test]
fn engine_en_03() {
    let whole = config::read(WHOLE_CONFIG.as_bytes()).expect("a whole config is read");
    assert_eq!(
        whole,
        config::Reading {
            family_recognised: true,
            missing_keys: 0,
            unsupported_features: 0,
        }
    );
    assert_eq!(
        whole.question(true).to_vec(),
        preflight::request(true, 0, true, 0).to_vec(),
        "the question put to the kernel is not the one the config counts"
    );
    // What the weights say is the caller's to pass, and it reaches the kernel
    // as given.
    assert_eq!(
        whole.question(false).to_vec(),
        preflight::request(true, 0, false, 0).to_vec()
    );

    // Every name the registry carries is recognised, and a name it does not
    // carry is not. An unnamed decoder is not rejected here: the kernel
    // answers `derived` for it, which `engine_en_01` pins.
    for family in config::NAMED_FAMILIES {
        let named = WHOLE_CONFIG.replace("LlamaForCausalLM", family);
        assert!(
            config::read(named.as_bytes())
                .expect("a whole config is read")
                .family_recognised,
            "{family} is in the registry and was not recognised"
        );
    }
    for other in ["GemmaForCausalLM", "llamaforcausallm", ""] {
        let named = WHOLE_CONFIG.replace("LlamaForCausalLM", other);
        assert!(
            !config::read(named.as_bytes())
                .expect("a whole config is read")
                .family_recognised,
            "{other} is not in the registry and was recognised"
        );
    }
    // The architectures key itself may be missing or empty.
    for lacking in [
        r#"{"architectures": []}"#,
        r#"{"architectures": "LlamaForCausalLM"}"#,
        "{}",
    ] {
        assert!(
            !config::read(lacking.as_bytes())
                .expect("an object is read")
                .family_recognised
        );
    }

    // Each required key is required: dropping one counts one.
    for key in config::REQUIRED_KEYS {
        let without: serde_json::Value = {
            let mut parsed: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(WHOLE_CONFIG).expect("the whole config parses");
            parsed.remove(key);
            parsed.into()
        };
        assert_eq!(
            config::read(without.to_string().as_bytes())
                .expect("a config is read")
                .missing_keys,
            1,
            "dropping {key} did not count as one missing key"
        );
    }
    assert_eq!(
        config::read(b"{}").expect("an object is read").missing_keys,
        config::REQUIRED_KEYS.len() as u8,
        "an empty config is missing every required key"
    );
    // A key that is present and is not a positive whole number is missing:
    // zero layers, a negative count or a number written as text build nothing.
    for value in ["0", "-1", "\"64\"", "null", "1.5"] {
        let spoiled =
            WHOLE_CONFIG.replace("\"hidden_size\": 64", &format!("\"hidden_size\": {value}"));
        assert_eq!(
            config::read(spoiled.as_bytes())
                .expect("a config is read")
                .missing_keys,
            1,
            "hidden_size {value} was counted as present"
        );
    }

    // The gate activation is the one feature this document can ask for that
    // the recipe does not implement.
    for activation in ["\"silu\"", "\"swish\"", "null"] {
        let gated = WHOLE_CONFIG.replace(
            "\"model_type\": \"llama\"",
            &format!("\"hidden_act\": {activation}"),
        );
        assert_eq!(
            config::read(gated.as_bytes())
                .expect("a config is read")
                .unsupported_features,
            0,
            "hidden_act {activation} was flagged"
        );
    }
    for activation in ["\"gelu\"", "\"gelu_pytorch_tanh\"", "7"] {
        let gated = WHOLE_CONFIG.replace(
            "\"model_type\": \"llama\"",
            &format!("\"hidden_act\": {activation}"),
        );
        let reading = config::read(gated.as_bytes()).expect("a config is read");
        assert_eq!(
            reading.unsupported_features, 1,
            "hidden_act {activation} was not flagged"
        );
        assert_eq!(
            reading.question(true).to_vec(),
            preflight::request(true, 0, true, 1).to_vec(),
            "the flag did not reach the kernel's question"
        );
    }
}

/// `FL-01`: The file has fewer than the eight bytes that carry the header length
#[test]
fn engine_fl_01() {
    let mut kernel = Accepting::new();
    for short in [&b""[..], b"\x01", b"\x00\x00\x00\x00\x00\x00\x00"] {
        assert_eq!(
            header::read(short, &mut kernel),
            Err(header::HeaderError::TooShortForLength)
        );
    }
    assert!(
        kernel.asked.is_empty(),
        "nothing is asked of the kernel about a file with no header"
    );
}

/// `FL-02`: The header length the file declares does not fit inside the file
#[test]
fn engine_fl_02() {
    let mut kernel = Accepting::new();
    let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    // One byte more of header than the file holds.
    let mut file = safetensors(header, 64, 72);
    file.truncate(71);
    assert_eq!(
        header::read(&file, &mut kernel),
        Err(header::HeaderError::HeaderExceedsFile)
    );
    // A length no file could hold, and one that overflows the count of bytes.
    for declared in [u64::MAX, u64::MAX - 7] {
        let mut file = declared.to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        assert_eq!(
            header::read(&file, &mut kernel),
            Err(header::HeaderError::HeaderExceedsFile)
        );
    }
}

/// `FL-03`: The header bytes are not a JSON object
#[test]
fn engine_fl_03() {
    let mut kernel = Accepting::new();
    for header in [
        "",
        "not json at all",
        "[1,2,3]",
        "\"a string\"",
        "null",
        r#"{"t":{"dtype":"F32","#,
    ] {
        let file = safetensors(header, 64, 128);
        assert_eq!(
            header::read(&file, &mut kernel),
            Err(header::HeaderError::NotJson),
            "a header of {header:?} was read as an object"
        );
    }
}

/// `FL-04`: A header entry is not an object carrying a dtype, a shape of whole numbers and a pair of data offsets
#[test]
fn engine_fl_04() {
    let mut kernel = Accepting::new();
    for entry in [
        "4",
        "null",
        r#"{"shape":[1],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[1]}"#,
        r#"{"dtype":7,"shape":[1],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":"1","data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":[0]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":["0","4"]}"#,
    ] {
        let file = safetensors(&format!(r#"{{"t":{entry}}}"#, entry = entry), 96, 160);
        assert_eq!(
            header::read(&file, &mut kernel),
            Err(header::HeaderError::TensorNotDescribed),
            "an entry of {entry} was read as a tensor"
        );
    }
}

/// `FL-05`: A tensor ends before it starts
#[test]
fn engine_fl_05() {
    let mut kernel = Accepting::new();
    let file = safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
        96,
        160,
    );
    assert_eq!(
        header::read(&file, &mut kernel),
        Err(header::HeaderError::OffsetsDoNotRunForward)
    );
    assert!(
        kernel.asked.is_empty(),
        "a tensor that ends before it starts is no question for the kernel"
    );
}

/// `FL-06`: The kernel did not accept a tensor byte range
#[test]
fn engine_fl_06() {
    // A tensor that claims one byte more than the file holds: the range
    // question is one of the kernel's own examples, and the kernel rejects it.
    let file = safetensors(
        r#"{"t":{"dtype":"F32","shape":[901],"data_offsets":[0,901]}}"#,
        92,
        1000,
    );
    assert_eq!(
        header::read(
            &file,
            &mut Recorded {
                answers: range_examples(),
            }
        ),
        Err(header::HeaderError::RangeRefused(range::Verdict::Reject)),
        "the kernel's refusal did not reach the caller"
    );

    // Whatever the kernel answers that is not an acceptance stops the read,
    // carried as the kernel gave it.
    struct Says(Vec<u8>);
    impl Kernel for Says {
        fn ask(&mut self, _request: &[u8]) -> Vec<u8> {
            self.0.clone()
        }
    }
    let whole = safetensors(
        r#"{"t":{"dtype":"F32","shape":[900],"data_offsets":[0,900]}}"#,
        92,
        1000,
    );
    for (answer, verdict) in [
        (&b"reject"[..], range::Verdict::Reject),
        (INVALID_REQUEST, range::Verdict::Invalid),
        (
            b"a word the kernel never says",
            range::Verdict::Unrecognised,
        ),
    ] {
        assert_eq!(
            header::read(&whole, &mut Says(answer.to_vec())),
            Err(header::HeaderError::RangeRefused(verdict))
        );
    }
}

/// `FL-07`: The config bytes are not a JSON object
#[test]
fn engine_fl_07() {
    for bytes in [
        &b""[..],
        b"not json at all",
        b"[1,2,3]",
        b"\"a string\"",
        b"null",
        b"{\"architectures\": ",
    ] {
        assert_eq!(config::read(bytes), Err(config::ConfigError::NotJson));
    }
}

// ─── The ONNX preflight ─────────────────────────────────────────────────────
//
// The files below are written out here, field by field, from the numbers
// `onnx.proto` gives: a tag, then a length, then the bytes. Nothing of the
// reader is used to build them, so a field number read from the wrong place, a
// length counted wrongly, or a name taken from the wrong field cannot pass.

/// A variable-length number, as the wire format writes one.
fn proto_number(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = u8::try_from(value & 0x7f).expect("seven bits");
        value >>= 7;
        out.push(if value == 0 { byte } else { byte | 0x80 });
        if value == 0 {
            return out;
        }
    }
}

/// A field holding a number.
fn proto_field_number(field: u64, value: u64) -> Vec<u8> {
    let mut out = proto_number(field << 3);
    out.extend(proto_number(value));
    out
}

/// A field holding a length and that many bytes.
fn proto_field_bytes(field: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = proto_number((field << 3) | 2);
    out.extend(proto_number(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

/// One opset declaration: a domain and a version.
fn opset(domain: &str, version: u64) -> Vec<u8> {
    let mut out = proto_field_bytes(1, domain.as_bytes());
    out.extend(proto_field_number(2, version));
    out
}

/// One node: its inputs and outputs, its name, its operator and its domain,
/// written in the order a real file has them.
fn node(op_type: &[u8], domain: &str) -> Vec<u8> {
    let mut out = proto_field_bytes(1, b"in");
    out.extend(proto_field_bytes(2, b"out"));
    out.extend(proto_field_bytes(3, b"a name"));
    out.extend(proto_field_bytes(4, op_type));
    out.extend(proto_field_bytes(6, b"documentation"));
    out.extend(proto_field_bytes(7, domain.as_bytes()));
    out
}

/// A graph of `nodes`, carrying `initializers` tensors.
fn graph(nodes: &[Vec<u8>], initializers: usize) -> Vec<u8> {
    let mut out = proto_field_bytes(2, b"the graph");
    for node in nodes {
        out.extend(proto_field_bytes(1, node));
    }
    for _ in 0..initializers {
        out.extend(proto_field_bytes(5, &proto_field_bytes(8, b"a tensor")));
    }
    out.extend(proto_field_bytes(11, b"an input"));
    out
}

/// A model, with the fields a real file carries that a preflight does not need.
fn onnx_file(opsets: &[Vec<u8>], graph: Option<&[u8]>) -> Vec<u8> {
    let mut out = proto_field_number(1, 9);
    out.extend(proto_field_bytes(2, b"a producer"));
    out.extend(proto_field_bytes(3, b"1.0"));
    if let Some(graph) = graph {
        out.extend(proto_field_bytes(7, graph));
    }
    for opset in opsets {
        out.extend(proto_field_bytes(8, opset));
    }
    // Two fields a preflight does not read, in encodings it still has to step
    // over: eight bytes, and four.
    // Their contents are bytes no tag could be, so a reader that steps over
    // the wrong number of them reads one of them as a field.
    out.extend(proto_number((20 << 3) | 1));
    out.extend([0xff; 8]);
    out.extend(proto_number((21 << 3) | 5));
    out.extend([0xff; 4]);
    out
}

/// What a builder of this kind implements. Not a claim about any builder here:
/// no graph is built in this repository.
const IMPLEMENTED: [&str; 4] = ["MatMul", "Add", "Softmax", "Gelu"];

/// `EN-04`: An ONNX model's default-domain opset, its nodes' operators in file order and the tensors its graph carries are read from the file, and fields the preflight does not need are stepped over
#[test]
fn engine_en_04() {
    let nodes = [
        node(b"MatMul", ""),
        node(b"Add", ""),
        node(b"Attention", "com.microsoft"),
        node(b"MatMul", ""),
    ];
    let file = onnx_file(
        &[opset("", 18), opset("com.microsoft", 1)],
        Some(&graph(&nodes, 3)),
    );
    assert_eq!(
        onnx::read(&file),
        Ok(onnx::Model {
            opset: 18,
            operators: vec![
                onnx::Operator {
                    domain: "".into(),
                    op_type: "MatMul".into()
                },
                onnx::Operator {
                    domain: "".into(),
                    op_type: "Add".into()
                },
                onnx::Operator {
                    domain: "com.microsoft".into(),
                    op_type: "Attention".into()
                },
                onnx::Operator {
                    domain: "".into(),
                    op_type: "MatMul".into()
                },
            ],
            initializers: 3,
        }),
        "the model read is not the model written"
    );

    // A declaration that names no domain is the default domain's.
    let bare = onnx_file(
        &[proto_field_number(2, 17)],
        Some(&graph(&[node(b"Add", "")], 0)),
    );
    assert_eq!(onnx::read(&bare).map(|m| m.opset), Ok(17));

    // A later declaration of a domain replaces an earlier one.
    let twice = onnx_file(
        &[opset("", 11), opset("", 21)],
        Some(&graph(&[node(b"Add", "")], 0)),
    );
    assert_eq!(onnx::read(&twice).map(|m| m.opset), Ok(21));

    // A graph with no nodes and no tensors is read, not refused: what may be
    // done with it is not this crate's to say.
    let empty = onnx_file(&[opset("", 18)], Some(&graph(&[], 0)));
    assert_eq!(
        onnx::read(&empty),
        Ok(onnx::Model {
            opset: 18,
            operators: vec![],
            initializers: 0
        })
    );

    // A node that names no operator is read as one with no name, which no
    // builder implements.
    let nameless = onnx_file(
        &[opset("", 18)],
        Some(&graph(&[proto_field_bytes(3, b"only a name")], 0)),
    );
    assert_eq!(
        onnx::read(&nameless).map(|m| m.operators),
        Ok(vec![onnx::Operator {
            domain: "".into(),
            op_type: "".into()
        }])
    );

    // A version written as a negative number is the large number it stands
    // for, which is no opset any builder implements.
    let mut negative = Vec::new();
    negative.extend(proto_number(2 << 3));
    negative.extend([0xff; 9]);
    negative.push(0x01);
    let signed = onnx_file(&[negative], Some(&graph(&[node(b"Add", "")], 0)));
    assert_eq!(onnx::read(&signed).map(|m| m.opset), Ok(u64::MAX));
}

/// `EN-05`: The operators an ONNX model names that a builder does not implement are counted into the kernel's import question, and reported each once in the order they first appear
#[test]
fn engine_en_05() {
    let nodes = [
        node(b"MatMul", ""),
        node(b"Resize", ""),
        node(b"Attention", "com.microsoft"),
        node(b"Resize", ""),
        node(b"Add", ""),
        node(b"MatMul", "com.microsoft"),
    ];
    let file = onnx_file(&[opset("", 18)], Some(&graph(&nodes, 0)));
    let model = onnx::read(&file).expect("the model is read");

    // Three nodes name something this builder does not build: `Resize` twice,
    // an extension operator, and `MatMul` in a domain that is not the ONNX
    // one, which is a different operator that happens to share a name.
    assert_eq!(
        model.question(&IMPLEMENTED, 7, 21).to_vec(),
        operators::request(18, 7, 21, 4).to_vec(),
        "the question put to the kernel is not the one the file counts"
    );
    assert_eq!(
        model.missing(&IMPLEMENTED),
        vec![
            onnx::Operator {
                domain: "".into(),
                op_type: "Resize".into()
            },
            onnx::Operator {
                domain: "com.microsoft".into(),
                op_type: "Attention".into()
            },
            onnx::Operator {
                domain: "com.microsoft".into(),
                op_type: "MatMul".into()
            },
        ],
        "what is missing is not reported once each in the order it appears"
    );

    // A model every one of whose operators is implemented asks a question with
    // nothing outstanding, and the opset it declares reaches the kernel as it
    // is.
    let whole = onnx_file(
        &[opset("", 7)],
        Some(&graph(&[node(b"MatMul", ""), node(b"Gelu", "")], 0)),
    );
    let whole = onnx::read(&whole).expect("the model is read");
    assert_eq!(
        whole.question(&IMPLEMENTED, 7, 21).to_vec(),
        operators::request(7, 7, 21, 0).to_vec()
    );
    assert!(whole.missing(&IMPLEMENTED).is_empty());

    // A builder that implements nothing counts every node, and one that is
    // told about an operator of another domain still does not match it: the
    // list is of ONNX operators.
    assert_eq!(
        whole.question(&[], 7, 21).to_vec(),
        operators::request(7, 7, 21, 2).to_vec()
    );
    let extension = onnx_file(
        &[opset("", 7)],
        Some(&graph(&[node(b"Attention", "com.microsoft")], 0)),
    );
    let extension = onnx::read(&extension).expect("the model is read");
    assert_eq!(
        extension.question(&["Attention"], 7, 21).to_vec(),
        operators::request(7, 7, 21, 1).to_vec()
    );
}

/// `FL-08`: A field runs past the end of the file
#[test]
fn engine_fl_08() {
    // A length longer than what follows it.
    let mut long = proto_number((7 << 3) | 2);
    long.extend(proto_number(64));
    long.extend_from_slice(b"short");
    assert_eq!(onnx::read(&long), Err(onnx::OnnxError::Truncated));

    // A tag with nothing after it, and a tag that does not end.
    assert_eq!(
        onnx::read(&proto_number(7 << 3)),
        Err(onnx::OnnxError::Truncated)
    );
    assert_eq!(onnx::read(&[0x80]), Err(onnx::OnnxError::Truncated));

    // Fixed-width fields that run off the end.
    for (encoding, width) in [(1u64, 7usize), (5, 3)] {
        let mut file = proto_number((20 << 3) | encoding);
        file.extend(std::iter::repeat_n(0u8, width));
        assert_eq!(onnx::read(&file), Err(onnx::OnnxError::Truncated));
    }

    // A truncation inside the graph, not at the top of the file.
    let mut node = proto_number((4 << 3) | 2);
    node.extend(proto_number(9));
    node.extend_from_slice(b"MatMul");
    let file = onnx_file(&[opset("", 18)], Some(&proto_field_bytes(1, &node)));
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::Truncated));
}

/// `FL-09`: A variable-length number does not end within the ten bytes that carry sixty-four bits
#[test]
fn engine_fl_09() {
    let mut forever = vec![0xff; 11];
    forever.push(0x00);
    assert_eq!(onnx::read(&forever), Err(onnx::OnnxError::NumberTooLong));

    // Ten bytes are a number; the eleventh is not.
    let mut ten = vec![0x80; 9];
    ten.push(0x01);
    let mut file = proto_number(1 << 3);
    file.extend(ten);
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::NoGraph));
}

/// `FL-10`: A field says it is encoded a way the format does not define, so its end cannot be found
#[test]
fn engine_fl_10() {
    for encoding in [3u64, 4, 6, 7] {
        let mut file = proto_number((20 << 3) | encoding);
        file.extend_from_slice(&[0; 8]);
        assert_eq!(
            onnx::read(&file),
            Err(onnx::OnnxError::UnknownEncoding),
            "a field encoded {encoding} was stepped over"
        );
    }

    // Inside a node, too, where the reader is one message deeper.
    let file = onnx_file(
        &[opset("", 18)],
        Some(&proto_field_bytes(1, &proto_number((9 << 3) | 6))),
    );
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::UnknownEncoding));
}

/// `FL-11`: A name the preflight reads is not text
#[test]
fn engine_fl_11() {
    let file = onnx_file(
        &[opset("", 18)],
        Some(&graph(&[node(&[0xff, 0xfe], "")], 0)),
    );
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::NameNotText));

    // A domain, and an opset's domain, are read the same way.
    let domain = onnx_file(
        &[opset("", 18)],
        Some(&graph(
            &[{
                let mut out = proto_field_bytes(4, b"Add");
                out.extend(proto_field_bytes(7, &[0xff]));
                out
            }],
            0,
        )),
    );
    assert_eq!(onnx::read(&domain), Err(onnx::OnnxError::NameNotText));

    let declared = onnx_file(
        &[proto_field_bytes(1, &[0x80, 0x01])],
        Some(&graph(&[node(b"Add", "")], 0)),
    );
    assert_eq!(onnx::read(&declared), Err(onnx::OnnxError::NameNotText));
}

/// `FL-12`: The model carries no graph
#[test]
fn engine_fl_12() {
    let file = onnx_file(&[opset("", 18)], None);
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::NoGraph));
    assert_eq!(onnx::read(&[]), Err(onnx::OnnxError::NoGraph));
}

/// `FL-13`: The model declares no opset for the default domain
#[test]
fn engine_fl_13() {
    let file = onnx_file(
        &[opset("com.microsoft", 1)],
        Some(&graph(&[node(b"Add", "")], 0)),
    );
    assert_eq!(onnx::read(&file), Err(onnx::OnnxError::NoDefaultOpset));

    // No declaration at all is the same: the opset is not something to assume.
    let none = onnx_file(&[], Some(&graph(&[node(b"Add", "")], 0)));
    assert_eq!(onnx::read(&none), Err(onnx::OnnxError::NoDefaultOpset));
}
