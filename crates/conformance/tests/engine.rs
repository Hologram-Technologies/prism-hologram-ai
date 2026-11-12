//! The `engine` suite: one test per scenario in `features/suites/engine.feature`.
//!
//! The engine is tested against the kernel's own examples, the requests and
//! answers the holo carries and replays in its compiled guest. Nothing here
//! restates them: they are read from `app/src/Kernel.lex.tex`, so an example
//! added to or changed in the model is tested the next time this runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use hologram_ai_kernel_client::{
    admission, bucket, generation, journey, plan, preflight, range, shape, tier, trim,
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
