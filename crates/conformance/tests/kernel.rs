//! The `kernel` suite: one test per scenario in `features/suites/kernel.feature`.
//!
//! The app's modules under `app/src/Kernel` hold only what the guest executes.
//! The propositions and theorems about them are in `app/src/Proofs`, which
//! import the executable definitions and are checked by the Lean kernel when
//! `just vv` runs `lexlean verify` on them. The examples are replayed in the
//! compiled guest by `prismpm verify` in the same recipe. What these tests hold fixed is the claim each
//! proof is about: the theorem's exact statement and axioms, the proposition it
//! decides, the request bytes the proposition is read from, and that the
//! verdict the application returns is decided by the function the theorem is
//! about. A weakened statement, a dropped comparison, an extra axiom, a field
//! read from the wrong offset, or a verdict decided some other way would still
//! verify, and would no longer be the claim the register makes.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/conformance is two below the root")
        .to_path_buf()
}

fn model(path: &str) -> String {
    std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The body of the top-level declaration named `name`, up to its closing brace.
fn declaration<'a>(model: &'a str, name: &str) -> &'a str {
    let start = model
        .find(&format!("\n  name: {name},\n"))
        .unwrap_or_else(|| panic!("{name} is not declared"));
    let end = model[start..]
        .find("\n}\n")
        .unwrap_or_else(|| panic!("{name} is not closed"));
    &model[start..start + end]
}

/// The proofs count only if the gate checks them. The `app` recipe must verify
/// every module under `app/src/Proofs`, by a glob that fails when it matches
/// nothing, and no theorem may sit where that command does not look.
fn assert_proofs_are_gated() {
    let justfile = model("Justfile");
    assert!(
        justfile.contains("shopt -s failglob")
            && justfile.contains("lexlean verify src/Proofs/*.lex.tex"),
        "the app recipe must run lexlean verify over every proof module"
    );
    // The executable modules are discovered, not listed: a list would go stale
    // the first time a decision was added, and a module nobody reads reports
    // nothing.
    let mut modules = vec!["app/src/Kernel.lex.tex".to_string()];
    let mut found: Vec<String> = std::fs::read_dir(root().join("app/src/Kernel"))
        .expect("app/src/Kernel")
        .map(|entry| entry.expect("a directory entry").file_name())
        .map(|name| format!("app/src/Kernel/{}", name.to_string_lossy()))
        .collect();
    found.sort();
    assert!(
        found.len() >= 3,
        "app/src/Kernel holds the wire format and one module per decision: {found:?}"
    );
    modules.extend(found);
    // The engine modules a decision imports are compiled into the same guest.
    let mut at = 0;
    while at < modules.len() {
        for line in model(&modules[at]).lines() {
            if let Some(engine) = line
                .strip_prefix("\\importmodule{Engine.")
                .and_then(|rest| rest.strip_suffix('}'))
            {
                let path = format!("app/src/Engine/{}.lex.tex", engine.replace('.', "/"));
                if !modules.contains(&path) {
                    modules.push(path);
                }
            }
        }
        at += 1;
    }

    // The generated guest puts every executable module's declarations in one
    // Rust namespace, so a name declared in two of them does not compile.
    let mut seen = std::collections::BTreeMap::new();
    for module in &modules {
        let text = model(module);
        assert!(
            !text.contains("\ntheorem{"),
            "{module} holds a theorem the proof gate does not read"
        );
        for line in text.lines() {
            let Some(name) = line
                .strip_prefix("  name: ")
                .or_else(|| line.strip_prefix("definition{name: "))
                .and_then(|rest| rest.split(',').next())
            else {
                continue;
            };
            if let Some(first) = seen.insert(name.to_string(), module.clone()) {
                panic!("`{name}` is declared in both {first} and {module}");
            }
        }
    }
}

/// A theorem's statement and axiom set, pinned: a quiet change to either fails here.
fn assert_pinned(proofs: &str, theorem: &str, statement: &str, axioms: &str) {
    let found = declaration(proofs, theorem);
    assert!(
        found.contains(&format!("statement: {statement},")),
        "{theorem}'s statement changed:\n{found}"
    );
    assert!(
        found.contains(&format!("axioms: [{axioms}]")),
        "{theorem} must rest on [{axioms}]:\n{found}"
    );
}

/// Decision `tag` is answered by `answer`, and nothing else answers for it.
fn assert_routed(tag: u8, answer: &str) {
    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains(&format!("(if (tag =? {tag}) then ({answer} value)")),
        "decision {tag} must be routed to {answer}:\n{route}"
    );
    assert_eq!(
        route.matches(&format!("(tag =? {tag})")).count(),
        1,
        "decision {tag} is routed once"
    );
}

/// Each `(request, answer)` is a claim in `crates/conformance/tests/kernel/claims.json`,
/// which the `app` recipe replays through the guest of the holo it has just built. So
/// a refusal that row 10 of the plan names is the compiled kernel's own answer, not
/// only a reading of the model's text.
fn assert_claimed(decision: u8, cases: &[(Vec<u8>, &str)]) {
    let claims =
        std::fs::read_to_string(root().join("crates/conformance/tests/kernel/claims.json"))
            .expect("the claims the crates make of the kernel");
    let claims: serde_json::Value = serde_json::from_str(&claims).expect("claims.json is JSON");
    let listed: Vec<(&str, &str)> = claims["claims"]
        .as_array()
        .expect("a list of claims")
        .iter()
        .filter_map(|one| Some((one["request"].as_str()?, one["answer"].as_str()?)))
        .collect();
    for (request, answer) in cases {
        assert_eq!(
            request.first(),
            Some(&decision),
            "a decision-{decision} request"
        );
        let request: String = request.iter().map(|b| format!("{b:02x}")).collect();
        assert!(
            listed.contains(&(request.as_str(), *answer)),
            "no claim answers decision {decision}'s {request} with `{answer}`"
        );
    }
}

/// The lowercase hex of `text`, as an example's response is written in the model.
fn hex(text: &str) -> String {
    text.bytes().map(|b| format!("{b:02x}")).collect()
}

/// The bytes of lowercase hex `text`, for a request no crate builds yet.
fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

/// `KN-05`: A buffer is admitted exactly when its expected and observed labels are equal and its declared and observed lengths are equal
#[test]
fn kernel_kn_05() {
    assert_proofs_are_gated();
    let admission = model("app/src/Kernel/Admission.lex.tex");
    let proofs = model("app/src/Proofs/Admission.lex.tex");

    // The theorem is about the function the guest executes, by its
    // qualified name, not about a copy of it.
    let decides = declaration(&proofs, "admitted_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Admission::admitted claim) = true) <-> (Admitted claim)),"
        ),
        "KN-05's statement changed: admitted must be true exactly when Admitted holds:\n{decides}"
    );
    assert!(
        decides.contains("parameters: [claim : Kernel.Admission::Claim],")
            && decides.contains("axioms: [propext]"),
        "KN-05 quantifies over every claim and rests on propext only:\n{decides}"
    );

    // A label is four 64-bit words, the widest a Nat is compiled to, and two
    // labels are the same only when every word is.
    let same = declaration(&proofs, "SameLabel");
    for word in ["w0", "w1", "w2", "w3"] {
        assert!(
            same.contains(&format!("((project left {word}) = (project right {word}))")),
            "SameLabel must compare label word {word}"
        );
    }

    // The proposition is both halves: the labels and the lengths.
    let admitted = declaration(&proofs, "Admitted");
    assert!(
        admitted.contains("(SameLabel (project claim expected) (project claim observed))"),
        "Admitted must require the expected and observed labels to be the same"
    );
    assert!(
        admitted.contains("((project claim declaredLength) = (project claim observedLength))"),
        "Admitted must require the declared and observed lengths to be equal"
    );

    // The claim is read from the bytes the wire format fixes: the decision in
    // byte 0, two 32-byte labels, two 8-byte lengths, 81 bytes in all.
    let claim_at = declaration(&admission, "claimAt");
    assert!(
        claim_at.contains(
            "{expected := (labelAt value 1), observed := (labelAt value 33), \
             declaredLength := (Kernel.Wire::fieldAt value 65 8), \
             observedLength := (Kernel.Wire::fieldAt value 73 8)}"
        ),
        "the claim must be read from offsets 1, 33, 65 and 73:\n{claim_at}"
    );
    let label_at = declaration(&admission, "labelAt");
    for (word, offset) in [
        ("w0", "offset"),
        ("w1", "(offset + 8)"),
        ("w2", "(offset + 16)"),
        ("w3", "(offset + 24)"),
    ] {
        assert!(
            label_at.contains(&format!(
                "{word} := (Kernel.Wire::fieldAt value {offset} 8)"
            )),
            "label word {word} must be the eight bytes at {offset}"
        );
    }

    let answer = declaration(&admission, "admissionAnswer");
    assert!(
        answer.contains("(if ((Foundation.Bytes::byteLength value) =? 81) then"),
        "a request that is not exactly 81 bytes must not reach the decision"
    );
    assert!(
        answer.contains(
            "(if (admitted (claimAt value)) then (prim utf8_encode \"admit\" : Bytes) \
             else (prim utf8_encode \"refuse\" : Bytes))"
        ),
        "the verdict must be decided by admitted, the function KN-05 proves"
    );

    // Decision 5 is this decision, and nothing else answers for it.
    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 5) then (Kernel.Admission::admissionAnswer value)"),
        "decision 5 must be routed to the admission answer:\n{route}"
    );

    // The examples the holo carries exercise both verdicts and the refusal.
    let application = declaration(&kernel, "application");
    for verdict in ["admit", "refuse", "invalid request"] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(verdict))),
            "no example answers `{verdict}`"
        );
    }
}

/// `KN-06`: A tensor byte range is accepted exactly when it starts at or after the previous tensor end and ends within the file
#[test]
fn kernel_kn_06() {
    assert_proofs_are_gated();
    let range = model("app/src/Kernel/Range.lex.tex");
    let proofs = model("app/src/Proofs/Range.lex.tex");

    // The arithmetic claim, over four numbers: the decided function is true
    // exactly when the range starts at or after the previous end and ends
    // within the file.
    let decides = declaration(&proofs, "within_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Range::within fileLength previousEnd offset length) = true) <-> \
             (Within fileLength previousEnd offset length)),"
        ),
        "KN-06's arithmetic statement changed: within must be true exactly when Within holds:\n{decides}"
    );
    assert!(
        decides.contains(
            "parameters: [fileLength : Nat, previousEnd : Nat, offset : Nat, length : Nat],"
        ) && decides.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "KN-06 quantifies over every four numbers and rests on Lean's three standard axioms:\n{decides}"
    );
    let proposition = declaration(&proofs, "Within");
    assert!(
        proposition
            .contains("body: ((previousEnd <= offset) /\\ ((offset + length) <= fileLength))"),
        "Within must be the mathematical statement, with the sum written as a sum:\n{proposition}"
    );

    // The decided function never adds the offset to the length. A natural
    // number is compiled to 64 bits, where that sum can overflow; the remaining
    // room is computed by a subtraction that cannot.
    let within = declaration(&range, "within");
    assert!(
        within.contains(
            "body: ((previousEnd <=? offset) && ((offset <=? fileLength) && \
             (length <=? (prim subtract fileLength offset : Nat))))"
        ),
        "within must compare the length with the room left, not form offset + length:\n{within}"
    );

    // The same claim over the request's four fields.
    let span = declaration(&proofs, "accepted_decides");
    assert!(
        span.contains("statement: (((Kernel.Range::accepted span) = true) <-> (Accepted span)),")
            && span.contains("parameters: [span : Kernel.Range::Span],"),
        "accepted must be true exactly when Accepted holds, for every span:\n{span}"
    );
    let accepted_prop = declaration(&proofs, "Accepted");
    assert!(
        accepted_prop.contains(
            "body: (((project span previousEnd) <= (project span offset)) /\\ \
             (((project span offset) + (project span length)) <= (project span fileLength)))"
        ),
        "Accepted must state both bounds over the span's own fields:\n{accepted_prop}"
    );
    let accepted = declaration(&range, "accepted");
    assert!(
        accepted.contains(
            "body: (within (project span fileLength) (project span previousEnd) \
             (project span offset) (project span length))"
        ),
        "accepted must hand the four fields to within in its parameter order:\n{accepted}"
    );

    // The claim over a whole file: every range accepted in turn, each one's
    // end becoming the next one's previous end, exactly when the list is
    // ordered and inside the file.
    let all = declaration(&proofs, "all_accepted_ordered");
    assert!(
        all.contains(
            "statement: (((allAccepted fileLength previousEnd spans) = true) <-> \
             (Ordered fileLength previousEnd spans)),"
        ) && all.contains("generalizing: [previousEnd],")
            && all.contains("scrutinee: spans"),
        "the list claim must hold for every file length, previous end and list:\n{all}"
    );
    let ordered = declaration(&proofs, "Ordered");
    assert!(
        ordered.contains(
            "| List.cons head rest => ((Within fileLength previousEnd (first head) (second head)) /\\ \
             (Ordered fileLength ((first head) + (second head)) rest))"
        ),
        "Ordered must require each range within bounds and pass its end to the next:\n{ordered}"
    );
    let all_accepted = declaration(&proofs, "allAccepted");
    assert!(
        all_accepted.contains(
            "| List.cons head rest => ((Kernel.Range::within fileLength previousEnd (first head) (second head)) && \
             (allAccepted fileLength ((first head) + (second head)) rest))"
        ),
        "allAccepted must fold the same within the single decision uses:\n{all_accepted}"
    );

    // The span is read from the bytes the wire format fixes: the decision in
    // byte 0, then four 8-byte numbers, 33 bytes in all.
    let span_at = declaration(&range, "spanAt");
    assert!(
        span_at.contains(
            "{fileLength := (Kernel.Wire::fieldAt value 1 8), \
             previousEnd := (Kernel.Wire::fieldAt value 9 8), \
             offset := (Kernel.Wire::fieldAt value 17 8), \
             length := (Kernel.Wire::fieldAt value 25 8)}"
        ),
        "the span must be read from offsets 1, 9, 17 and 25:\n{span_at}"
    );
    let answer = declaration(&range, "rangeAnswer");
    assert!(
        answer.contains("(if ((Foundation.Bytes::byteLength value) =? 33) then"),
        "a request that is not exactly 33 bytes must not reach the decision"
    );
    assert!(
        answer.contains(
            "(if (accepted (spanAt value)) then (prim utf8_encode \"accept\" : Bytes) \
             else (prim utf8_encode \"reject\" : Bytes))"
        ),
        "the verdict must be decided by accepted, the function KN-06 proves"
    );

    // Decision 6 is this decision, and nothing else answers for it.
    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 6) then (Kernel.Range::rangeAnswer value)"),
        "decision 6 must be routed to the range answer:\n{route}"
    );

    // The examples exercise both verdicts, and the top of the 64-bit field,
    // where a sum would overflow: offset 2^64 - 1 with length 1 must be
    // rejected by the compiled guest, not wrapped into an acceptance.
    let application = declaration(&kernel, "application");
    for verdict in ["accept", "reject"] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(verdict))),
            "no example answers `{verdict}`"
        );
    }
    let overflow = format!(
        "request := 0x06{file}{previous}{offset}{length}, response := 0x{reject}}}",
        file = "ffffffffffffffff",
        previous = "0000000000000000",
        offset = "ffffffffffffffff",
        length = "0000000000000001",
        reject = hex("reject")
    );
    assert!(
        application.contains(&overflow),
        "no example rejects a range whose offset plus length exceeds 64 bits"
    );
}

/// `KN-08`: The decode bucket is at least what is needed, never above the maximum, and never below the current bucket
#[test]
fn kernel_kn_08() {
    assert_proofs_are_gated();
    let bucket = model("app/src/Kernel/Bucket.lex.tex");
    let proofs = model("app/src/Proofs/Bucket.lex.tex");

    // The three bounds, stated about the function the guest executes, for
    // every need and current bucket within the maximum.
    let bounds = declaration(&proofs, "grow_bounds");
    assert!(
        bounds.contains(
            "statement: ((Fits needed current maximum) -> \
             (((needed <= (Kernel.Bucket::grow needed current maximum)) /\\ \
             ((Kernel.Bucket::grow needed current maximum) <= maximum)) /\\ \
             (current <= (Kernel.Bucket::grow needed current maximum)))),"
        ),
        "KN-08's statement changed: the new bucket must cover the need, stay within the maximum and not shrink:\n{bounds}"
    );
    assert!(
        bounds.contains("parameters: [needed : Nat, current : Nat, maximum : Nat],")
            && bounds.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "KN-08 quantifies over every three numbers and rests on Lean's three standard axioms:\n{bounds}"
    );

    // The precondition is exactly the two bounds, and it is decided.
    let fits_prop = declaration(&proofs, "Fits");
    assert!(
        fits_prop.contains("body: ((needed <= maximum) /\\ (current <= maximum))"),
        "Fits must require both the need and the current bucket within the maximum:\n{fits_prop}"
    );
    let decides = declaration(&proofs, "fits_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Bucket::fits needed current maximum) = true) <-> \
             (Fits needed current maximum)),"
        ),
        "fits must be true exactly when Fits holds:\n{decides}"
    );

    // Doubling never adds past the maximum: the sum is formed only when the
    // current bucket is at most the room left, so it cannot overflow 64 bits.
    let doubled = declaration(&bucket, "doubled");
    assert!(
        doubled.contains(
            "body: (if (current <=? (prim subtract maximum current : Nat)) then \
             (current + current) else maximum)"
        ),
        "doubled must check the room before it adds:\n{doubled}"
    );

    // The numbers are read from the bytes the wire format fixes, the bucket
    // is computed only when the request fits, and each refusal is named.
    let answer = declaration(&bucket, "bucketAnswer");
    assert!(
        answer.contains("(if ((Foundation.Bytes::byteLength value) =? 25) then"),
        "a request that is not exactly 25 bytes must not reach the decision"
    );
    let numbers = "(Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) \
                   (Kernel.Wire::fieldAt value 17 8)";
    assert!(
        answer.contains(&format!(
            "(if (fits {numbers}) then (Kernel.Wire::decimal (grow {numbers})) else"
        )),
        "the bucket must be grow's, computed only when fits holds, from offsets 1, 9 and 17:\n{answer}"
    );
    assert!(
        answer.contains(
            "(if ((Kernel.Wire::fieldAt value 17 8) <? (Kernel.Wire::fieldAt value 9 8)) then \
             (prim utf8_encode \"bucket exceeds maximum\" : Bytes) else \
             (prim utf8_encode \"context exceeded\" : Bytes))"
        ),
        "a current bucket past the maximum and a need past the maximum must each be named:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 8) then (Kernel.Bucket::bucketAnswer value)"),
        "decision 8 must be routed to the bucket answer:\n{route}"
    );

    // The examples exercise both refusals, and the widest number: a need of
    // 2^64 - 1 must come back as its twenty digits.
    let application = declaration(&kernel, "application");
    for refusal in ["context exceeded", "bucket exceeds maximum"] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(refusal))),
            "no example answers `{refusal}`"
        );
    }
    let widest = format!(
        "request := 0x08{needed}{current}{maximum}, response := 0x{digits}}}",
        needed = "ffffffffffffffff",
        current = "0000000000000000",
        maximum = "ffffffffffffffff",
        digits = hex("18446744073709551615")
    );
    assert!(
        application.contains(&widest),
        "no example prints the widest 64-bit number"
    );
}

/// `KN-09`: A turn token budget never exceeds the context that remains and never exceeds the request
#[test]
fn kernel_kn_09() {
    assert_proofs_are_gated();
    let generation = model("app/src/Kernel/Generation.lex.tex");
    let proofs = model("app/src/Proofs/Generation.lex.tex");

    // With room left, the budget fits the context and is at least one token.
    let context = declaration(&proofs, "budget_within_context");
    assert!(
        context.contains(
            "statement: ((used < context) -> \
             (((used + (Kernel.Generation::budget context used requested)) <= context) /\\ \
             (1 <= (Kernel.Generation::budget context used requested)))),"
        ),
        "KN-09's statement changed: the budget must fit the context that remains and be at least one:\n{context}"
    );
    assert!(
        context.contains("parameters: [context : Nat, used : Nat, requested : Nat],")
            && context.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "KN-09 quantifies over every three numbers and rests on Lean's three standard axioms:\n{context}"
    );

    // A request, when one is made, is never exceeded.
    let request = declaration(&proofs, "budget_within_request");
    assert!(
        request.contains(
            "statement: ((0 < requested) -> \
             ((Kernel.Generation::budget context used requested) <= requested)),"
        ),
        "the budget must never exceed a request that was made:\n{request}"
    );

    // "Room left" is decided, and is strict: a full context has none.
    let has_room = declaration(&proofs, "has_room_decides");
    assert!(
        has_room.contains(
            "statement: (((Kernel.Generation::hasRoom context used) = true) <-> (used < context)),"
        ),
        "hasRoom must be true exactly when used is below the context:\n{has_room}"
    );

    // The room is a subtraction, which cannot overflow, and a request of zero
    // means no cap.
    let room = declaration(&generation, "room");
    assert!(
        room.contains("body: (prim subtract context used : Nat)"),
        "room must be the context less what is used:\n{room}"
    );
    let budget = declaration(&generation, "budget");
    assert!(
        budget.contains(
            "body: (if (requested =? 0) then (room context used) else \
             (if (requested <=? (room context used)) then requested else (room context used)))"
        ),
        "budget must be the room, capped by a request that was made:\n{budget}"
    );

    let answer = declaration(&generation, "generationAnswer");
    assert!(
        answer.contains("(if ((Foundation.Bytes::byteLength value) =? 25) then"),
        "a request that is not exactly 25 bytes must not reach the decision"
    );
    assert!(
        answer.contains(
            "(if (hasRoom (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8)) then \
             (Kernel.Wire::decimal (budget (Kernel.Wire::fieldAt value 1 8) \
             (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 17 8))) else \
             (prim utf8_encode \"context full\" : Bytes))"
        ),
        "the budget must be budget's, computed only when hasRoom holds, from offsets 1, 9 and 17:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 9) then (Kernel.Generation::generationAnswer value)"),
        "decision 9 must be routed to the generation answer:\n{route}"
    );

    // The examples exercise the refusal at exactly full, and a request cut to
    // the room that remains: 256 asked with 48 left answers 48.
    let application = declaration(&kernel, "application");
    assert!(
        application.contains(&format!("response := 0x{}}}", hex("context full"))),
        "no example answers `context full`"
    );
    let full = format!(
        "request := 0x09{context}{used}{requested}, response := 0x{answer}}}",
        context = "0000000000000800",
        used = "0000000000000800",
        requested = "0000000000000000",
        answer = hex("context full")
    );
    assert!(
        application.contains(&full),
        "no example refuses a context that is exactly full"
    );
    let cut = format!(
        "request := 0x09{context}{used}{requested}, response := 0x{answer}}}",
        context = "0000000000000800",
        used = "00000000000007d0",
        requested = "0000000000000100",
        answer = hex("48")
    );
    assert!(
        application.contains(&cut),
        "no example cuts a request to the room that remains"
    );
}

/// `KN-02`: A model is supported or derived exactly when no required key is missing, no unsupported feature is flagged and the manifest fits, and only then may weights download
#[test]
fn kernel_kn_02() {
    assert_proofs_are_gated();
    let preflight = model("app/src/Kernel/Preflight.lex.tex");
    let proofs = model("app/src/Proofs/Preflight.lex.tex");

    // What it means for a model to be runnable: all three conditions.
    let runnable = declaration(&proofs, "Runnable");
    assert!(
        runnable.contains("body: ((missing = 0) /\\ ((unsupported = 0) /\\ (manifestFits = 1)))"),
        "Runnable must require no missing key, no unsupported feature and a manifest that fits:\n{runnable}"
    );

    // Each verdict is decided, about the function the guest executes.
    let numbers = "recognised missing manifestFits unsupported";
    for (theorem, left, right) in [
        (
            "supported_decides",
            format!("((Kernel.Preflight::verdictCode {numbers}) = 4)"),
            "((Runnable missing manifestFits unsupported) /\\ (recognised = 1))",
        ),
        (
            "derived_decides",
            format!("((Kernel.Preflight::verdictCode {numbers}) = 3)"),
            "((Runnable missing manifestFits unsupported) /\\ (recognised = 0))",
        ),
        (
            "authorised_decides",
            format!("((Kernel.Preflight::authorised {numbers}) = true)"),
            "((Runnable missing manifestFits unsupported) /\\ (recognised <= 1))",
        ),
    ] {
        let declared = declaration(&proofs, theorem);
        assert!(
            declared.contains(&format!("statement: ({left} <-> {right}),")),
            "KN-02's {theorem} changed: it must hold exactly when the model is runnable:\n{declared}"
        );
        assert!(
            declared.contains(
                "parameters: [recognised : Nat, missing : Nat, manifestFits : Nat, unsupported : Nat],"
            ) && declared.contains("axioms: [Classical.choice, Quot.sound, propext]"),
            "{theorem} quantifies over every four numbers and rests on Lean's three standard axioms:\n{declared}"
        );
    }

    // Weights may download only on the two verdicts the theorems are about.
    let authorised = declaration(&preflight, "authorised");
    assert!(
        authorised.contains(&format!(
            "body: (((verdictCode {numbers}) =? 3) || ((verdictCode {numbers}) =? 4))"
        )),
        "authorised must be exactly the derived and supported verdicts:\n{authorised}"
    );

    // Each code has one text, and the answer is the text of the proved code,
    // read from the four flag bytes the wire format fixes.
    let text = declaration(&preflight, "verdictText");
    for (code, verdict) in [
        (4, "supported"),
        (3, "derived"),
        (2, "rejected: manifest does not fit"),
        (1, "rejected: unsupported feature"),
        (0, "rejected: missing config key"),
    ] {
        assert!(
            text.contains(&format!(
                "(if (code =? {code}) then (prim utf8_encode \"{verdict}\" : Bytes) else"
            )),
            "verdict {code} must read `{verdict}`:\n{text}"
        );
    }
    let answer = declaration(&preflight, "preflightAnswer");
    assert!(
        answer.contains(
            "(if ((Foundation.Bytes::byteLength value) =? 5) then (verdictText (verdictCode \
             (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1) \
             (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 4 1))) else"
        ),
        "the verdict must be verdictCode's, from the four flag bytes at offsets 1 to 4:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 2) then (Kernel.Preflight::preflightAnswer value)"),
        "decision 2 must be routed to the preflight answer:\n{route}"
    );

    // Every verdict has an example, and a missing key is named before
    // anything else that is also wrong.
    let application = declaration(&kernel, "application");
    for verdict in [
        "supported",
        "derived",
        "rejected: missing config key",
        "rejected: unsupported feature",
        "rejected: manifest does not fit",
    ] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(verdict))),
            "no example answers `{verdict}`"
        );
    }
    assert!(
        application.contains(&format!(
            "request := 0x0201020005, response := 0x{}}}",
            hex("rejected: missing config key")
        )),
        "no example names the missing key first when everything is wrong"
    );
}

/// `KN-04`: The quantisation tier chosen fits in memory whenever any tier does, and more memory never lowers it
#[test]
fn kernel_kn_04() {
    assert_proofs_are_gated();
    let tier = model("app/src/Kernel/Tier.lex.tex");
    let proofs = model("app/src/Proofs/Tier.lex.tex");

    // The int4 cost is half the parameters, rounded up, and is proved to be so.
    let half = declaration(&proofs, "half_up_bounds");
    assert!(
        half.contains(
            "statement: (((Kernel.Tier::halfUp parameters) <= parameters) /\\ \
             (parameters <= ((Kernel.Tier::halfUp parameters) + (Kernel.Tier::halfUp parameters)))),"
        ),
        "halfUp must be at most the parameters and at least half of them:\n{half}"
    );

    // int8 in memory exactly when it fits; int4 in memory only when it fits;
    // and some tier in memory whenever int4 fits.
    let fits = declaration(&proofs, "tier_fits");
    assert!(
        fits.contains(
            "statement: (((((Kernel.Tier::tierCode parameters budget) = 2) <-> (parameters <= budget)) /\\ \
             (((Kernel.Tier::tierCode parameters budget) = 1) -> ((Kernel.Tier::halfUp parameters) <= budget))) /\\ \
             (((Kernel.Tier::halfUp parameters) <= budget) -> (1 <= (Kernel.Tier::tierCode parameters budget)))),"
        ),
        "KN-04's statement changed: the chosen tier must fit, and a tier in memory must be chosen whenever one fits:\n{fits}"
    );
    let monotone = declaration(&proofs, "tier_monotone");
    assert!(
        monotone.contains(
            "statement: ((budget <= larger) -> ((Kernel.Tier::tierCode parameters budget) <= \
             (Kernel.Tier::tierCode parameters larger))),"
        ),
        "more memory must never lower the tier:\n{monotone}"
    );
    assert!(
        fits.contains("axioms: [Classical.choice, Quot.sound, propext]")
            && monotone.contains("axioms: [Quot.sound, propext]"),
        "each tier theorem declares exactly the axioms Lean observes for it"
    );

    // The order the theorems are about: paged, int4 in memory, int8 in memory.
    let code = declaration(&tier, "tierCode");
    assert!(
        code.contains(
            "body: (if (parameters <=? budget) then 2 else (if ((halfUp parameters) <=? budget) then 1 else 0))"
        ),
        "tierCode must prefer int8 in memory, then int4 in memory, then paged:\n{code}"
    );
    let half_up = declaration(&tier, "halfUp");
    assert!(
        half_up.contains(
            "body: (prim subtract parameters (prim quotient parameters 2 0 : Nat) : Nat)"
        ),
        "halfUp must be the parameters less half of them rounded down:\n{half_up}"
    );
    let text = declaration(&tier, "tierText");
    assert!(
        text.contains(
            "body: (if (code =? 2) then (prim utf8_encode \"int8\" : Bytes) else \
             (if (code =? 1) then (prim utf8_encode \"int4\" : Bytes) else \
             (prim utf8_encode \"int8 paged\" : Bytes)))"
        ),
        "each tier code must read as its tier:\n{text}"
    );
    let answer = declaration(&tier, "tierAnswer");
    assert!(
        answer.contains(
            "(if ((Foundation.Bytes::byteLength value) =? 17) then (tierText (tierCode \
             (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8))) else"
        ),
        "the tier must be tierCode's, from the two numbers at offsets 1 and 9:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 4) then (Kernel.Tier::tierAnswer value)"),
        "decision 4 must be routed to the tier answer:\n{route}"
    );

    // The examples sit on each side of both boundaries in 4 GiB: one parameter
    // past what int8 holds answers int4, and one past what int4 holds, where
    // the half rounds up, answers paged.
    let application = declaration(&kernel, "application");
    for (parameters, verdict) in [
        ("0000000100000000", "int8"),
        ("0000000100000001", "int4"),
        ("0000000200000000", "int4"),
        ("0000000200000001", "int8 paged"),
    ] {
        assert!(
            application.contains(&format!(
                "request := 0x04{parameters}0000000100000000, response := 0x{}}}",
                hex(verdict)
            )),
            "no example answers `{verdict}` for {parameters} parameters in 4 GiB"
        );
    }
}

/// `KN-11`: A requested dimension is bound exactly when it lies within its bounds, to a size at least the request and never above the upper bound
#[test]
fn kernel_kn_11() {
    assert_proofs_are_gated();
    let shape = model("app/src/Kernel/Shape.lex.tex");
    let proofs = model("app/src/Proofs/Shape.lex.tex");

    let decides = declaration(&proofs, "in_bounds_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Shape::inBounds lower upper requested) = true) <-> \
             ((lower <= requested) /\\ (requested <= upper))),"
        ),
        "KN-11's statement changed: inBounds must be true exactly when the request lies within its bounds:\n{decides}"
    );
    let bounds = declaration(&proofs, "bind_bounds");
    assert!(
        bounds.contains(
            "statement: ((requested <= upper) -> \
             ((requested <= (Kernel.Shape::bind upper requested step)) /\\ \
             ((Kernel.Shape::bind upper requested step) <= upper))),"
        ),
        "the bound size must be at least the request and never above the upper bound:\n{bounds}"
    );
    assert!(
        bounds.contains("parameters: [upper : Nat, requested : Nat, step : Nat],")
            && bounds.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "bind_bounds quantifies over every three numbers, for every step:\n{bounds}"
    );

    // Rounding up adds only after checking the room below the upper bound,
    // so the sum cannot pass the bound or 64 bits.
    let bind = declaration(&shape, "bind");
    assert!(
        bind.contains(
            "body: (if ((gapTo requested step) <=? (prim subtract upper requested : Nat)) then \
             (requested + (gapTo requested step)) else upper)"
        ),
        "bind must check the room before it adds, and cap at the upper bound:\n{bind}"
    );
    // A step of 0 or 1 means exact, and the remainder is never taken by zero.
    let gap = declaration(&shape, "gapTo");
    assert!(
        gap.contains(
            "body: (if (step <=? 1) then 0 else (if ((prim remainder requested step 0 : Nat) =? 0) then 0 else \
             (prim subtract step (prim remainder requested step 0 : Nat) : Nat)))"
        ),
        "gapTo must be the distance to the next multiple of the step, or nothing for a step of 0 or 1:\n{gap}"
    );

    let answer = declaration(&shape, "shapeAnswer");
    assert!(
        answer.contains("(if ((Foundation.Bytes::byteLength value) =? 33) then"),
        "a request that is not exactly 33 bytes must not reach the decision"
    );
    assert!(
        answer.contains(
            "(if (inBounds (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) \
             (Kernel.Wire::fieldAt value 17 8)) then (Kernel.Wire::decimal (bind \
             (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 17 8) \
             (Kernel.Wire::fieldAt value 25 8))) else"
        ),
        "the size must be bind's, computed only when inBounds holds, from offsets 1, 9, 17 and 25:\n{answer}"
    );
    assert!(
        answer.contains(
            "(if ((Kernel.Wire::fieldAt value 17 8) <? (Kernel.Wire::fieldAt value 1 8)) then \
             (prim utf8_encode \"below minimum\" : Bytes) else \
             (prim utf8_encode \"above maximum\" : Bytes))"
        ),
        "a request below the lower bound and one above the upper bound must each be named:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 11) then (Kernel.Shape::shapeAnswer value)"),
        "decision 11 must be routed to the shape answer:\n{route}"
    );

    // The examples round up to a multiple, cap at the upper bound when the
    // next multiple is past it, and name both refusals.
    let application = declaration(&kernel, "application");
    for refusal in ["below minimum", "above maximum"] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(refusal))),
            "no example answers `{refusal}`"
        );
    }
    for (upper, requested, step, size) in [
        (
            "0000000000000800",
            "0000000000000064",
            "0000000000000040",
            "128",
        ),
        (
            "00000000000007f8",
            "00000000000007d0",
            "0000000000000200",
            "2040",
        ),
    ] {
        assert!(
            application.contains(&format!(
                "request := 0x0b0000000000000001{upper}{requested}{step}, response := 0x{}}}",
                hex(size)
            )),
            "no example binds {requested} to {size}"
        );
    }
}

/// `KN-07`: A proposed trim is accepted exactly when what remains fits the context and restoring the newest dropped pair would not
#[test]
fn kernel_kn_07() {
    assert_proofs_are_gated();
    let trim = model("app/src/Kernel/Trim.lex.tex");
    let proofs = model("app/src/Proofs/Trim.lex.tex");

    // What a right trim is: nothing dropped that was not there, what remains
    // fits, and, when anything was dropped, the newest dropped pair is one of
    // them and putting it back would pass the context.
    let trimmed_prop = declaration(&proofs, "Trimmed");
    assert!(
        trimmed_prop.contains(
            "body: ((dropped <= total) /\\ \
             (((fixed + (prim subtract total dropped : Nat)) <= context) /\\ \
             ((0 < dropped) -> ((last <= dropped) /\\ \
             (context < ((fixed + (prim subtract total dropped : Nat)) + last))))))"
        ),
        "Trimmed must require that what remains fits and that restoring the newest dropped pair would not:\n{trimmed_prop}"
    );
    let decides = declaration(&proofs, "trimmed_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Trim::trimmed context fixed total dropped last) = true) <-> \
             (Trimmed context fixed total dropped last)),"
        ),
        "KN-07's statement changed: trimmed must be true exactly when Trimmed holds:\n{decides}"
    );
    assert!(
        decides.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "trimmed_decides rests on Lean's three standard axioms:\n{decides}"
    );
    let fits = declaration(&proofs, "trim_fits_decides");
    assert!(
        fits.contains(
            "statement: (((Kernel.Trim::trimFits context fixed total dropped) = true) <-> \
             ((fixed + (prim subtract total dropped : Nat)) <= context)),"
        ),
        "trimFits must be true exactly when what remains fits the context:\n{fits}"
    );

    // The decided functions never add: the room is found by subtraction, so
    // nothing can pass 64 bits.
    let trim_fits = declaration(&trim, "trimFits");
    assert!(
        trim_fits.contains(
            "body: ((fixed <=? context) && ((prim subtract total dropped : Nat) <=? \
             (prim subtract context fixed : Nat)))"
        ),
        "trimFits must compare what remains with the room left, without a sum:\n{trim_fits}"
    );
    let minimal = declaration(&trim, "trimMinimal");
    assert!(
        minimal.contains(
            "body: ((dropped =? 0) || ((last <=? dropped) && \
             ((prim subtract (prim subtract context fixed : Nat) (prim subtract total dropped : Nat) : Nat) <? last)))"
        ),
        "trimMinimal must hold when nothing was dropped or the newest dropped pair exceeds the room left:\n{minimal}"
    );
    let trimmed = declaration(&trim, "trimmed");
    assert!(
        trimmed.contains(
            "body: ((dropped <=? total) && ((trimFits context fixed total dropped) && \
             (trimMinimal context fixed total dropped last)))"
        ),
        "trimmed must be all three conditions:\n{trimmed}"
    );

    // The verdict is trimmed's, read from five numbers, and each refusal says
    // which condition failed.
    let numbers = "(Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) \
                   (Kernel.Wire::fieldAt value 17 8) (Kernel.Wire::fieldAt value 25 8)";
    let answer = declaration(&trim, "trimAnswer");
    assert!(
        answer.contains(&format!(
            "(if ((Foundation.Bytes::byteLength value) =? 41) then (if (trimmed {numbers} \
             (Kernel.Wire::fieldAt value 33 8)) then (prim utf8_encode \"trim\" : Bytes) else"
        )),
        "the verdict must be trimmed's, from the five numbers at offsets 1 to 33:\n{answer}"
    );
    assert!(
        answer.contains(&format!(
            "(if (trimFits {numbers}) then (prim utf8_encode \"drops too many\" : Bytes) else \
             (prim utf8_encode \"does not fit\" : Bytes)) else \
             (prim utf8_encode \"too long\" : Bytes)) else \
             (prim utf8_encode \"dropped exceeds total\" : Bytes))"
        )),
        "each refusal must name the condition that failed:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 7) then (Kernel.Trim::trimAnswer value)"),
        "decision 7 must be routed to the trim answer:\n{route}"
    );

    // The examples sit on both sides of both boundaries: a dropped pair one
    // token too big to restore is a right trim, one that would exactly fit is
    // one drop too many, and what remains may fill the context but not pass it.
    let application = declaration(&kernel, "application");
    let head = "000000000000080000000000000000640000000000000bb8";
    for (dropped, last, verdict) in [
        ("000000000000044c", "0000000000000031", "trim"),
        ("000000000000044c", "0000000000000030", "drops too many"),
        ("000000000000041c", "000000000000012c", "trim"),
        ("000000000000041b", "000000000000012c", "does not fit"),
    ] {
        assert!(
            application.contains(&format!(
                "request := 0x07{head}{dropped}{last}, response := 0x{}}}",
                hex(verdict)
            )),
            "no example answers `{verdict}` for {dropped} dropped with a newest dropped pair of {last}"
        );
    }
    for refusal in ["too long", "dropped exceeds total"] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(refusal))),
            "no example answers `{refusal}`"
        );
    }
}

/// `KN-03`: A proposed stage plan is accepted exactly when its stage fits the memory budget, is the largest that does, and its stages cover every layer once
#[test]
fn kernel_kn_03() {
    assert_proofs_are_gated();
    let plan = model("app/src/Kernel/Plan.lex.tex");
    let proofs = model("app/src/Proofs/Plan.lex.tex");

    // What a right plan is, with every product written as a product.
    let planned_prop = declaration(&proofs, "Planned");
    for (part, meaning) in [
        (
            "((1 <= stageLayers) /\\ (stageLayers <= layers))",
            "a stage of at least one layer and at most all of them",
        ),
        (
            "((contextBytes + (prim multiply stageLayers layerBytes : Nat)) <= budget)",
            "the context and one stage within the budget",
        ),
        (
            "((stageLayers < layers) -> (budget < (contextBytes + (prim multiply (stageLayers + 1) layerBytes : Nat))))",
            "one more layer not fitting, unless the stage is the whole model",
        ),
        (
            "((1 <= stageCount) /\\ (((prim multiply (prim subtract stageCount 1 : Nat) stageLayers : Nat) < layers) /\\ \
             (layers <= (prim multiply stageCount stageLayers : Nat))))",
            "the stages covering every layer with none empty",
        ),
    ] {
        assert!(
            planned_prop.contains(part),
            "Planned must require {meaning}:\n{planned_prop}"
        );
    }
    let decides = declaration(&proofs, "planned_decides");
    assert!(
        decides.contains(
            "statement: (((Kernel.Plan::planned budget contextBytes layerBytes layers stageLayers stageCount) = true) <-> \
             (Planned budget contextBytes layerBytes layers stageLayers stageCount)),"
        ),
        "KN-03's statement changed: planned must be true exactly when Planned holds:\n{decides}"
    );
    assert!(
        decides.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "planned_decides rests on Lean's three standard axioms:\n{decides}"
    );
    let can_run = declaration(&proofs, "can_run_decides");
    assert!(
        can_run.contains(
            "statement: (((Kernel.Plan::canRun budget contextBytes layerBytes) = true) <-> \
             ((contextBytes + layerBytes) <= budget)),"
        ),
        "canRun must be true exactly when the context and one layer fit the budget:\n{can_run}"
    );

    // The products cannot pass 64 bits: the layer count and stage numbers are
    // bounded by 2^16 and a layer by 2^40 before anything is multiplied.
    let bounded = declaration(&plan, "planBounded");
    assert!(
        bounded.contains(
            "body: (((layers <=? 65536) && (layerBytes <=? 1099511627776)) && \
             ((stageLayers <=? 65536) && (stageCount <=? 65536)))"
        ),
        "planBounded must bound every factor before a product is formed:\n{bounded}"
    );
    let planned = declaration(&plan, "planned");
    assert!(
        planned.contains(
            "body: (((1 <=? stageLayers) && (stageLayers <=? layers)) && \
             ((stageFits budget contextBytes layerBytes stageLayers) && \
             ((stageLargest budget contextBytes layerBytes layers stageLayers) && \
             (stagesCover layers stageLayers stageCount))))"
        ),
        "planned must be all four conditions:\n{planned}"
    );
    let largest = declaration(&plan, "stageLargest");
    assert!(
        largest.contains(
            "body: ((stageLayers =? layers) || ((prim subtract budget contextBytes : Nat) <? \
             (prim multiply (stageLayers + 1) layerBytes : Nat)))"
        ),
        "stageLargest must hold when the stage is the whole model or one more layer would not fit:\n{largest}"
    );

    // Bounds first, then whether the model can run at all, then the plan.
    let numbers: Vec<String> = [1, 9, 17, 25, 33, 41]
        .iter()
        .map(|offset| format!("(Kernel.Wire::fieldAt value {offset} 8)"))
        .collect();
    let answer = declaration(&plan, "planAnswer");
    assert!(
        answer.contains(&format!(
            "(if ((Foundation.Bytes::byteLength value) =? 49) then (if (planBounded {} {} {} {}) then \
             (if (canRun {} {} {}) then (if (planned {}) then (prim utf8_encode \"plan\" : Bytes) else",
            numbers[2],
            numbers[3],
            numbers[4],
            numbers[5],
            numbers[0],
            numbers[1],
            numbers[2],
            numbers.join(" ")
        )),
        "the verdict must be planned's, reached only within the bounds and when one layer fits:\n{answer}"
    );
    for refusal in [
        "stages do not cover",
        "stage too small",
        "stage does not fit",
        "cannot run",
    ] {
        assert!(
            answer.contains(&format!("(prim utf8_encode \"{refusal}\" : Bytes)")),
            "the answer must be able to say `{refusal}`"
        );
    }

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 3) then (Kernel.Plan::planAnswer value)"),
        "decision 3 must be routed to the plan answer:\n{route}"
    );

    // The examples: with 3584 MiB of room and 200 MiB layers, a stage of 17
    // is right, 16 is too small and 18 does not fit; two stages of 17 cover 30
    // layers, one leaves layers out and a third would be empty.
    let application = declaration(&kernel, "application");
    let head = "00000001000000000000000020000000000000000c800000000000000000001e";
    for (stage, count, verdict) in [
        ("0000000000000011", "0000000000000002", "plan"),
        ("0000000000000010", "0000000000000002", "stage too small"),
        ("0000000000000012", "0000000000000002", "stage does not fit"),
        (
            "0000000000000011",
            "0000000000000001",
            "stages do not cover",
        ),
        (
            "0000000000000011",
            "0000000000000003",
            "stages do not cover",
        ),
    ] {
        assert!(
            application.contains(&format!(
                "request := 0x03{head}{stage}{count}, response := 0x{}}}",
                hex(verdict)
            )),
            "no example answers `{verdict}` for a stage of {stage} layers in {count} stages"
        );
    }
    assert!(
        application.contains(&format!("response := 0x{}}}", hex("cannot run"))),
        "no example answers `cannot run`"
    );
}

/// `KN-01`: Each working state is entered from exactly one state by exactly one event, failed is left only by reset, and every other step is refused
#[test]
fn kernel_kn_01() {
    assert_proofs_are_gated();
    let journey = model("app/src/Kernel/Journey.lex.tex");
    let proofs = model("app/src/Proofs/Journey.lex.tex");

    // One way in to each step of the chain: the state before it, and the
    // event that completes that state. States: 0 idle, 1 preflight,
    // 2 download, 3 compile, 4 materialize, 5 ready, 6 generating, 7 failed.
    for (theorem, entered, from, by) in [
        ("enters_preflight", 1, 0, 1),
        ("enters_download", 2, 1, 2),
        ("enters_compile", 3, 2, 3),
        ("enters_materialize", 4, 3, 4),
        ("enters_generating", 6, 5, 6),
    ] {
        let declared = declaration(&proofs, theorem);
        assert!(
            declared.contains(&format!(
                "statement: (((Kernel.Journey::nextState state event) = {entered}) <-> \
                 ((state = {from}) /\\ (event = {by}))),"
            )),
            "KN-01's {theorem} changed: state {entered} must be entered exactly from state {from} by event {by}:\n{declared}"
        );
        assert!(
            declared.contains("parameters: [state : Nat, event : Nat],"),
            "{theorem} quantifies over every state and event:\n{declared}"
        );
    }

    // Ready has two ways in and no others: from materialize when the labels
    // are verified, and back from generating when a turn ends or is cancelled.
    let ready = declaration(&proofs, "enters_ready");
    assert!(
        ready.contains(
            "statement: (((Kernel.Journey::nextState state event) = 5) <-> ((readyBy state event) = true)),"
        ),
        "ready must be entered exactly when readyBy holds:\n{ready}"
    );
    let ready_by = declaration(&proofs, "readyBy");
    assert!(
        ready_by.contains(
            "body: (((state =? 4) && (event =? 5)) || ((state =? 6) && ((event =? 7) || (event =? 8))))"
        ),
        "readyBy must be: from materialize by labels verified, or from generating by turn done or cancel:\n{ready_by}"
    );

    // Failed is entered only by a failure while work is under way, idle only
    // by reset, and failed goes nowhere but idle.
    let failed = declaration(&proofs, "enters_failed");
    assert!(
        failed.contains(
            "statement: (((Kernel.Journey::nextState state event) = 7) <-> \
             ((event = 9) /\\ ((Kernel.Journey::working state) = true))),"
        ),
        "failed must be entered exactly by a fail event from a working state:\n{failed}"
    );
    let idle = declaration(&proofs, "enters_idle");
    assert!(
        idle.contains(
            "statement: (((Kernel.Journey::nextState state event) = 0) <-> (event = 10)),"
        ),
        "idle must be entered exactly by reset:\n{idle}"
    );
    let stuck = declaration(&proofs, "failed_only_resets");
    assert!(
        stuck.contains(
            "statement: ((((Kernel.Journey::nextState 7 event) =? 0) || \
             ((Kernel.Journey::nextState 7 event) =? 8)) = true),"
        ),
        "from failed every event must reset or be refused:\n{stuck}"
    );

    // The states that can fail are the ones doing work.
    let working = declaration(&journey, "working");
    assert!(
        working.contains("body: (((1 <=? state) && (state <=? 4)) || (state =? 6))"),
        "working must be preflight to materialize, and generating:\n{working}"
    );

    // Each code reads as its state, and 8 as refused.
    let text = declaration(&journey, "stateText");
    for (code, name) in [
        (0, "idle"),
        (1, "preflight"),
        (2, "download"),
        (3, "compile"),
        (4, "materialize"),
        (5, "ready"),
        (6, "generating"),
        (7, "failed"),
    ] {
        assert!(
            text.contains(&format!(
                "(if (code =? {code}) then (prim utf8_encode \"{name}\" : Bytes) else"
            )),
            "state {code} must read `{name}`:\n{text}"
        );
    }
    assert!(
        text.contains("else (prim utf8_encode \"refused\" : Bytes)"),
        "any other code must read `refused`:\n{text}"
    );

    // The answer is nextState's, for a state and an event that exist.
    let answer = declaration(&journey, "journeyAnswer");
    assert!(
        answer.contains(
            "(if ((Foundation.Bytes::byteLength value) =? 3) then \
             (if (((Kernel.Wire::fieldAt value 1 1) <=? 7) && ((1 <=? (Kernel.Wire::fieldAt value 2 1)) && \
             ((Kernel.Wire::fieldAt value 2 1) <=? 10))) then \
             (stateText (nextState (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1))) else"
        ),
        "the next state must be nextState's, from the state and event bytes at offsets 1 and 2:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 1) then (Kernel.Journey::journeyAnswer value)"),
        "decision 1 must be routed to the journey answer:\n{route}"
    );

    // The examples walk the whole chain, and refuse the shortcuts: labels
    // verified from idle or from the middle of a download does not reach ready.
    let application = declaration(&kernel, "application");
    for (state, event, next) in [
        (0, 1, "preflight"),
        (1, 2, "download"),
        (2, 3, "compile"),
        (3, 4, "materialize"),
        (4, 5, "ready"),
        (5, 6, "generating"),
        (6, 7, "ready"),
        (0, 5, "refused"),
        (2, 5, "refused"),
        (7, 10, "idle"),
        (7, 1, "refused"),
    ] {
        assert!(
            application.contains(&format!(
                "request := 0x01{state:02x}{event:02x}, response := 0x{}}}",
                hex(next)
            )),
            "no example answers `{next}` for event {event} in state {state}"
        );
    }
}

/// `KN-10`: A graph is importable exactly when its opset lies within the implemented window and no operator is unimplemented, and a wider window never makes an importable graph unimportable
#[test]
fn kernel_kn_10() {
    assert_proofs_are_gated();
    let operators = model("app/src/Kernel/Operators.lex.tex");
    let proofs = model("app/src/Proofs/Operators.lex.tex");

    // What it means for a graph to be importable: inside the window, and every
    // operator implemented.
    let importable = declaration(&proofs, "Importable");
    assert!(
        importable.contains(
            "body: ((lowest <= declared) /\\ ((declared <= highest) /\\ (unimplemented = 0)))"
        ),
        "Importable must require the opset to be within the window and no unimplemented operator:\n{importable}"
    );

    // The verdict is decided, about the function the guest executes.
    let numbers = "declared lowest highest unimplemented";
    let decides = declaration(&proofs, "importable_decides");
    assert!(
        decides.contains(&format!(
            "statement: (((Kernel.Operators::operatorCode {numbers}) = 3) <-> (Importable {numbers})),"
        )),
        "KN-10's importable_decides changed: the importable verdict must hold exactly when the graph is importable:\n{decides}"
    );

    // A wider window never makes an importable graph unimportable.
    let monotone = declaration(&proofs, "window_monotone");
    assert!(
        monotone.contains(&format!(
            "statement: (((Importable {numbers}) /\\ ((wider <= lowest) /\\ (highest <= taller))) \
             -> (Importable declared wider taller unimplemented)),"
        )),
        "KN-10's window_monotone changed: widening the window must keep an importable graph importable:\n{monotone}"
    );

    // A window that runs backwards is not a refusal: it is no question.
    let reversed = declaration(&proofs, "reversed_window_is_invalid");
    assert!(
        reversed.contains(&format!(
            "statement: (((Kernel.Operators::operatorCode {numbers}) = 4) <-> ((highest + 1) <= lowest)),"
        )),
        "KN-10's reversed_window_is_invalid changed: a window that runs backwards must be the invalid code:\n{reversed}"
    );

    for theorem in [
        "importable_decides",
        "window_monotone",
        "reversed_window_is_invalid",
    ] {
        let declared = declaration(&proofs, theorem);
        assert!(
            declared.contains("declared : Nat")
                && declared.contains("lowest : Nat")
                && declared.contains("highest : Nat")
                && declared.contains("unimplemented : Nat")
                && declared.contains("axioms: [Classical.choice, Quot.sound, propext]"),
            "{theorem} quantifies over every number and rests on Lean's three standard axioms:\n{declared}"
        );
    }

    // Each code has one text, and the answer is the text of the proved code,
    // read from the four numbers the wire format fixes.
    let text = declaration(&operators, "operatorText");
    for (code, verdict) in [
        (3, "importable"),
        (2, "unsupported operator"),
        (1, "opset above maximum"),
        (0, "opset below minimum"),
    ] {
        assert!(
            text.contains(&format!(
                "(if (code =? {code}) then (prim utf8_encode \"{verdict}\" : Bytes) else"
            )),
            "verdict {code} must read `{verdict}`:\n{text}"
        );
    }
    let answer = declaration(&operators, "operatorAnswer");
    assert!(
        answer.contains(
            "(if ((Foundation.Bytes::byteLength value) =? 33) then (operatorText (operatorCode \
             (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) \
             (Kernel.Wire::fieldAt value 17 8) (Kernel.Wire::fieldAt value 25 8))) else"
        ),
        "the verdict must be operatorCode's, from the four numbers at offsets 1, 9, 17 and 25:\n{answer}"
    );

    let kernel = model("app/src/Kernel.lex.tex");
    let route = declaration(&kernel, "route");
    assert!(
        route.contains("(if (tag =? 10) then (Kernel.Operators::operatorAnswer value)"),
        "decision 10 must be routed to the operator answer:\n{route}"
    );

    // Every verdict has an example, and both edges of the window are examples:
    // a decision about a range that is never asked at its edges is a decision
    // whose edges nothing has replayed.
    let application = declaration(&kernel, "application");
    for verdict in [
        "importable",
        "opset below minimum",
        "opset above maximum",
        "unsupported operator",
    ] {
        assert!(
            application.contains(&format!("response := 0x{}}}", hex(verdict))),
            "no example answers `{verdict}`"
        );
    }
    let window = |declared: u64, unimplemented: u64| {
        format!(
            "request := 0x0a{declared:016x}{:016x}{:016x}{unimplemented:016x}, response := 0x",
            7, 21
        )
    };
    for (declared, verdict) in [(7, "importable"), (21, "importable")] {
        assert!(
            application.contains(&format!("{}{}}}", window(declared, 0), hex(verdict))),
            "opset {declared} is an edge of the window in the examples and must answer `{verdict}`"
        );
    }
    assert!(
        application.contains(&format!(
            "request := 0x0a{:016x}{:016x}{:016x}{:016x}, response := 0x{}}}",
            18,
            21,
            7,
            0,
            hex("invalid request")
        )),
        "no example asks about a window that runs backwards"
    );
}

/// `KN-12`: WebGPU is chosen exactly when the adapter is present and the model largest buffer fits its binding limit, and a wider limit never takes WebGPU away
#[test]
fn kernel_kn_12() {
    assert_proofs_are_gated();
    let backend = model("app/src/Kernel/Backend.lex.tex");
    let proofs = model("app/src/Proofs/Backend.lex.tex");
    assert_pinned(
        &proofs,
        "gpu_usable_decides",
        "(((Kernel.Backend::gpuUsable present limit largest) = true) <-> ((present = 1) /\\ (largest <= limit)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "limit_monotone",
        "((limit <= wider) -> (((Kernel.Backend::gpuUsable present limit largest) = true) -> ((Kernel.Backend::gpuUsable present wider largest) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    // Eighteen bytes: presence in byte 1, the binding limit at 2, the largest
    // buffer at 10; the verdict is the function the theorems are about.
    let answer = declaration(&backend, "backendAnswer");
    assert!(
        answer.contains("((Foundation.Bytes::byteLength value) =? 18) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)")
            && answer.contains("(if (gpuUsable (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 8)) then (prim utf8_encode \"webgpu\" : Bytes) else (prim utf8_encode \"cpu\" : Bytes))"),
        "decision 12 must read 18 bytes at offsets 1, 2 and 10 and answer by gpuUsable:\n{answer}"
    );
    // Decision 12 answers cpu when the largest buffer is over the binding limit, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            12,
            &[
                (unhex("0c0100000000000003e800000000000001f4"), "webgpu"),
                (unhex("0c0100000000000001f400000000000003e8"), "cpu"),
                (
                    unhex("0c0100000000000003e800000000000001"),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(12, "Kernel.Backend::backendAnswer");
}

/// `KN-13`: A weight block length is the one its format and shape imply, and a shape whose columns do not divide into the format groups is refused
#[test]
fn kernel_kn_13() {
    assert_proofs_are_gated();
    let blocks = model("app/src/Kernel/Blocks.lex.tex");
    let proofs = model("app/src/Proofs/Blocks.lex.tex");
    assert_pinned(
        &proofs,
        "shaped_decides",
        "(((Kernel.Blocks::shaped format rows columns) = true) <-> ((1 <= rows) /\\ ((1 <= columns) /\\ ((prim remainder columns (Kernel.Blocks::groupOf format) 0 : Nat) = 0))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "group_bounds",
        "((16 <= (Kernel.Blocks::groupOf format)) /\\ ((4 <= (Kernel.Blocks::bytesPerGroup format)) /\\ ((Kernel.Blocks::bytesPerGroup format) <= 68)))",
        "Classical.choice, Quot.sound, propext",
    );
    // The layouts, as bytes per group of weights: q4 20 per 32, q3f 16 per 32,
    // t2 4 per 16, t2r 68 per 256, ternary-per-128 36 per 128, and f32 vectors
    // 64 per 16.
    let group = declaration(&blocks, "groupOf");
    assert!(
        group.contains("(if (format =? 2) then 16 else (if (format =? 3) then 256 else (if (format =? 4) then 128 else (if (format =? 5) then 16 else 32))))"),
        "group sizes changed:\n{group}"
    );
    let bytes = declaration(&blocks, "bytesPerGroup");
    assert!(
        bytes.contains("(if (format =? 0) then 20 else (if (format =? 1) then 16 else (if (format =? 2) then 4 else (if (format =? 3) then 68 else (if (format =? 4) then 36 else 64)))))"),
        "bytes per group changed:\n{bytes}"
    );
    let answer = declaration(&blocks, "blockAnswer");
    assert!(
        answer.contains("((Foundation.Bytes::byteLength value) =? 18) && (known (Kernel.Wire::fieldAt value 1 1))")
            && answer.contains("(Kernel.Wire::decimal (blockLength (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 8)))")
            && answer.contains("(prim utf8_encode \"bad shape\" : Bytes)"),
        "decision 13 must read 18 bytes at offsets 1, 2 and 10 and answer the length or bad shape:\n{answer}"
    );
    // Decision 13 refuses columns that do not divide into groups with bad shape, and an unknown format or a short request as invalid.
    {
        use hologram_ai_kernel_client::blocks;
        assert_claimed(
            13,
            &[
                (blocks::request(0, 2, 64).to_vec(), "80"),
                (blocks::request(0, 2, 33).to_vec(), "bad shape"),
                (blocks::request(255, 2, 64).to_vec(), "invalid request"),
                (blocks::request(0, 2, 64)[..17].to_vec(), "invalid request"),
            ],
        );
    }
    assert_routed(13, "Kernel.Blocks::blockAnswer");
}

/// `KN-14`: A claimed greedy pick is accepted exactly when its index is the first position holding the maximum score and the claimed score is that maximum
#[test]
fn kernel_kn_14() {
    assert_proofs_are_gated();
    let pick = model("app/src/Kernel/Pick.lex.tex");
    let proofs = model("app/src/Proofs/Pick.lex.tex");
    // Before the claimed index every score is strictly below the claimed one,
    // at it the score is the claimed one, after it none is above: the first
    // maximum, and its value.
    let standing = declaration(&proofs, "Standing");
    assert!(
        standing.contains("(((position < index) -> (score < best)) /\\ (((position = index) -> (score = best)) /\\ ((index < position) -> (score <= best))))"),
        "Standing must say first maximum:\n{standing}"
    );
    assert_pinned(
        &proofs,
        "standing_decides",
        "(((Kernel.Pick::standing position index best score) = true) <-> (Standing position index best score))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "holds_decides",
        "(((Kernel.Pick::holdsFrom value position remaining index best) = true) <-> (HoldsFrom value position remaining index best))",
        "Classical.choice, Quot.sound, propext",
    );
    assert!(
        declaration(&proofs, "holds_decides").contains("scrutinee: remaining"),
        "holds_decides is by induction over every position"
    );
    let answer = declaration(&pick, "pickAnswer");
    assert!(
        answer.contains("((Foundation.Bytes::byteLength value) =? (7 + (prim multiply 4 (Kernel.Wire::fieldAt value 6 1) : Nat)))")
            && answer.contains("((Kernel.Wire::fieldAt value 1 1) <? (Kernel.Wire::fieldAt value 6 1))")
            && answer.contains("(if (holdsFrom value 0 (Kernel.Wire::fieldAt value 6 1) (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4)) then (prim utf8_encode \"first maximum\" : Bytes)"),
        "decision 14 must check every one of the n scores from position 0:\n{answer}"
    );
    assert!(
        declaration(&pick, "scoreAt")
            .contains("(Kernel.Wire::fieldAt value (7 + (prim multiply 4 position : Nat)) 4)"),
        "score i is the four bytes at 7 + 4i"
    );
    // Decision 14 answers not first maximum when an earlier score ties the best, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            14,
            &[
                (unhex("0e0000000005020000000500000003"), "first maximum"),
                (unhex("0e0100000005020000000500000005"), "not first maximum"),
                (unhex("0e00000000050200000005000000"), "invalid request"),
            ],
        );
    }
    assert_routed(14, "Kernel.Pick::pickAnswer");
}

/// `KN-15`: A token is kept unpenalised exactly when no position inside the recent window holds it
#[test]
fn kernel_kn_15() {
    assert_proofs_are_gated();
    let penalty = model("app/src/Kernel/Penalty.lex.tex");
    let proofs = model("app/src/Proofs/Penalty.lex.tex");
    let clear = declaration(&proofs, "Clear");
    assert!(
        clear.contains("((start <= position) -> ((seen = token) -> (1 = 0)))"),
        "Clear must say no position in the window holds the token:\n{clear}"
    );
    assert_pinned(
        &proofs,
        "clear_decides",
        "(((Kernel.Penalty::clear position start token seen) = true) <-> (Clear position start token seen))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "clear_from_decides",
        "(((Kernel.Penalty::clearFrom value position remaining start token) = true) <-> (ClearFrom value position remaining start token))",
        "Classical.choice, Quot.sound, propext",
    );
    // The window starts `window` tokens before the end of the history, or at
    // its start when the history is shorter.
    let answer = declaration(&penalty, "penaltyAnswer");
    assert!(
        answer.contains("(if (clearFrom value 0 (Kernel.Wire::fieldAt value 7 1) (prim subtract (Kernel.Wire::fieldAt value 7 1) (Kernel.Wire::fieldAt value 1 2) : Nat) (Kernel.Wire::fieldAt value 3 4)) then (prim utf8_encode \"keep\" : Bytes) else (prim utf8_encode \"penalise\" : Bytes))"),
        "decision 15 must check the whole history against the window:\n{answer}"
    );
    // Decision 15 answers penalise when the token is in the window, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            15,
            &[
                (unhex("0f000100000007020000000700000003"), "keep"),
                (unhex("0f000100000007020000000300000007"), "penalise"),
                (unhex("0f0001000000070200000007000000"), "invalid request"),
            ],
        );
    }
    assert_routed(15, "Kernel.Penalty::penaltyAnswer");
}

/// `KN-16`: A speculative commit count is accepted exactly when every draft before the last committed position equals its verified pick and the draft at that position, if any, differs
#[test]
fn kernel_kn_16() {
    assert_proofs_are_gated();
    let speculate = model("app/src/Kernel/Speculate.lex.tex");
    let proofs = model("app/src/Proofs/Speculate.lex.tex");
    let agrees = declaration(&proofs, "Agrees");
    assert!(
        agrees.contains("((((position + 1) < count) -> (draft = verified)) /\\ (((position + 1) = count) -> ((draft = verified) -> (1 = 0))))"),
        "Agrees must say: equal before the last committed position, different at it:\n{agrees}"
    );
    assert_pinned(
        &proofs,
        "agrees_decides",
        "(((Kernel.Speculate::agrees position count draft verified) = true) <-> (Agrees position count draft verified))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "agrees_from_decides",
        "(((Kernel.Speculate::agreesFrom value window position remaining count) = true) <-> (AgreesFrom value window position remaining count))",
        "Classical.choice, Quot.sound, propext",
    );
    let answer = declaration(&speculate, "speculateAnswer");
    assert!(
        answer.contains("((1 <=? (Kernel.Wire::fieldAt value 2 1)) && ((Kernel.Wire::fieldAt value 2 1) <=? ((Kernel.Wire::fieldAt value 1 1) + 1)))")
            && answer.contains("(if (agreesFrom value (Kernel.Wire::fieldAt value 1 1) 0 (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1)) then (prim utf8_encode \"commit\" : Bytes)"),
        "decision 16 must bound the count to 1..k+1 and check every draft:\n{answer}"
    );
    // Decision 16 answers wrong count when the count stops at a draft that agrees, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            16,
            &[
                (
                    unhex("1002030000000100000002000000010000000200000009"),
                    "commit",
                ),
                (
                    unhex("1002010000000100000002000000010000000200000009"),
                    "wrong count",
                ),
                (
                    unhex("10020300000001000000020000000100000002000000"),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(16, "Kernel.Speculate::speculateAnswer");
}

/// `KN-17`: Unpacking a packed weight group returns the code packed at each place, for each format, and each code maps to its stated level
#[test]
fn kernel_kn_17() {
    assert_proofs_are_gated();
    let quant = model("app/src/Kernel/Quant.lex.tex");
    let proofs = model("app/src/Proofs/Quant.lex.tex");
    // Each format's round trip: unpacking a place of a word packed from codes
    // below their base gives back the code at that place. The statement names
    // the function the guest executes, and the code chosen by place.
    for (theorem, unpack, chosen, axioms) in [
        (
            "nibble_round_trip",
            "Kernel.Quant::nibbleAt",
            "choose8 place",
            "Quot.sound, propext",
        ),
        (
            "bit_round_trip",
            "Kernel.Quant::bitAt",
            "choose8 place",
            "Quot.sound, propext",
        ),
        (
            "triple_round_trip",
            "Kernel.Quant::tripleAt",
            "choose10 place",
            "Quot.sound, propext",
        ),
        (
            "pair_low_round_trip",
            "Kernel.Quant::pairAt",
            "choose8 place",
            "Quot.sound, propext",
        ),
        (
            "pair_high_round_trip",
            "Kernel.Quant::pairAt",
            "choose8 place",
            "Quot.sound, propext",
        ),
    ] {
        let found = declaration(&proofs, theorem);
        let flat: String = found.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains(&format!("(({unpack} ")) && flat.contains(&format!("= ({chosen} ")),
            "{theorem} must say {unpack} gives back the code chosen by place:\n{found}"
        );
        assert!(
            flat.contains(&format!("axioms: [{axioms}]")),
            "{theorem} must rest on [{axioms}]:\n{found}"
        );
    }
    // The q3f word carries a 2-bit spare above its ten fields; the round trip
    // includes it, so the spare never disturbs a field.
    let triple: String = declaration(&proofs, "triple_round_trip")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        triple.contains("(prim multiply 1073741824 spare : Nat)") && triple.contains("(spare < 4)"),
        "the q3f round trip must include the spare bits"
    );
    // The two q3f codes kept in the spares come back from all three words.
    let spare: String = declaration(&proofs, "spare_round_trip")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        spare.contains("30) = q30)")
            && spare.contains("31) = q31)")
            && spare.contains("axioms: [Classical.choice, Quot.sound, propext]"),
        "spare_round_trip must give back places 30 and 31"
    );
    // The answer is the code minus the format's offset, with its sign: q4 8,
    // q3f 3, t2 and t2r 1 (code 3 is not a ternary code), the ternary planes 1.
    let answer = declaration(&quant, "quantAnswer");
    for wanted in [
        "(signedText (nibbleAt (Kernel.Wire::fieldAt value 3 4) (Kernel.Wire::fieldAt value 2 1)) 8)",
        "(signedText (fieldCode (Kernel.Wire::fieldAt value 3 4) (Kernel.Wire::fieldAt value 7 4) (Kernel.Wire::fieldAt value 11 4) (Kernel.Wire::fieldAt value 2 1)) 3)",
        "(prim utf8_encode \"invalid code\" : Bytes)",
        "(signedText (pairAt (Kernel.Wire::fieldAt value 3 4) (Kernel.Wire::fieldAt value 2 1)) 1)",
        "(signedText (ternaryCode (bitAt (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 2 1)) (bitAt (Kernel.Wire::fieldAt value 4 1) (Kernel.Wire::fieldAt value 2 1))) 1)",
    ] {
        assert!(answer.contains(wanted), "decision 17 must answer {wanted}:\n{answer}");
    }
    // Decision 17 answers invalid code for the unused 2-bit code, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            17,
            &[
                (unhex("1100000000000f"), "7"),
                (unhex("11020000000003"), "invalid code"),
                (unhex("11000000000f"), "invalid request"),
            ],
        );
    }
    assert_routed(17, "Kernel.Quant::quantAnswer");
}

/// `KN-18`: An operator is supported exactly when its name is on the closed list of the export style asked about, and no name is supported under an unknown style
#[test]
fn kernel_kn_18() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Opnames.lex.tex");
    let proofs = model("app/src/Proofs/Opnames.lex.tex");
    assert_pinned(
        &proofs,
        "opname_is_decides",
        "(((Kernel.Opnames::opnameIs wa wb wc wd a b c d) = true) <-> ((wa = a) /\\ ((wb = b) /\\ ((wc = c) /\\ (wd = d)))))",
        "propext",
    );
    let found = declaration(&kernel, "opnameIs");
    assert!(
        found.contains("body: ((wa =? a) && ((wb =? b) && ((wc =? c) && (wd =? d))))"),
        "opnameIs changed:\n{found}"
    );
    let found = declaration(&kernel, "opnameDecomposed");
    assert!(
        found.contains("body: ((opnameIs wa wb wc wd 4712001061299159040 0 0 0) || ((opnameIs wa wb wc wd 4855288815335505920 0 0 0) || ((opnameIs wa wb wc wd 4859223896071864320 0 0 0) || ((opnameIs wa wb wc wd 4859223965108891252 5721352084397253888 0 0) || ((opnameIs wa wb wc wd 4929601009505796096 0 0 0) || ((opnameIs wa wb wc wd 5003909722269548544 0 0 0) || ((opnameIs wa wb wc wd 5005874549588492288 0 0 0) || ((opnameIs wa wb wc wd 5143520241161207808 0 0 0) || ((opnameIs wa wb wc wd 5148288793276936704 0 0 0) || ((opnameIs wa wb wc wd 5575865689692700672 0 0 0) || ((opnameIs wa wb wc wd 5575865689692720706 7598824915118063616 0 0) || ((opnameIs wa wb wc wd 5581486060451397632 0 0 0) || ((opnameIs wa wb wc wd 5649034557303816192 0 0 0) || ((opnameIs wa wb wc wd 5795982087332823040 0 0 0) || ((opnameIs wa wb wc wd 5936147174206210048 0 0 0) || ((opnameIs wa wb wc wd 5937262139099401573 7020548869117181952 0 0) || ((opnameIs wa wb wc wd 5937278575906415872 0 0 0) || ((opnameIs wa wb wc wd 6008753480639279694 4899916394579099648 0 0) || ((opnameIs wa wb wc wd 6010160838084657152 0 0 0) || ((opnameIs wa wb wc wd 6010448897420911616 0 0 0) || ((opnameIs wa wb wc wd 6011295478249947136 0 0 0) || ((opnameIs wa wb wc wd 6012136677800245248 0 0 0) || ((opnameIs wa wb wc wd 6012712820058095616 0 0 0) || ((opnameIs wa wb wc wd 6085033173542006643 7277816997830721536 0 0) || ((opnameIs wa wb wc wd 6085041961071280128 0 0 0) || ((opnameIs wa wb wc wd 6155984671800124794 7277816997830721536 0 0) || ((opnameIs wa wb wc wd 6298395620872814592 0 0 0) || false)))))))))))))))))))))))))))"),
        "opnameDecomposed changed:\n{found}"
    );
    let found = declaration(&kernel, "opnameFused");
    assert!(
        found.contains("body: ((opnameIs wa wb wc wd 4855288815335505920 0 0 0) || ((opnameIs wa wb wc wd 4859223896071864320 0 0 0) || ((opnameIs wa wb wc wd 4859223965108891252 0 0 0) || ((opnameIs wa wb wc wd 5143520241161207808 0 0 0) || ((opnameIs wa wb wc wd 5148299874224141669 8248696160930590324 7597412042676371456 0) || ((opnameIs wa wb wc wd 5575865689692700672 0 0 0) || ((opnameIs wa wb wc wd 5575865689692720706 7598824915118063616 0 0) || ((opnameIs wa wb wc wd 5581486060451397632 0 0 0) || ((opnameIs wa wb wc wd 5937262139099403125 7854277750134145024 0 0) || ((opnameIs wa wb wc wd 5937278575906415872 0 0 0) || ((opnameIs wa wb wc wd 5940094395406173549 7090183570793064192 0 0) || ((opnameIs wa wb wc wd 6010160838084657152 0 0 0) || ((opnameIs wa wb wc wd 6010448897420911616 0 0 0) || ((opnameIs wa wb wc wd 6010455507325249129 7306048477034017358 8030601350753122913 8388358299191345152) || ((opnameIs wa wb wc wd 6011014058812730736 7811887631509507169 8747523532439842145 7811909587634319214) || ((opnameIs wa wb wc wd 6013820629562687488 0 0 0) || ((opnameIs wa wb wc wd 6085033173542006643 7277816997830721536 0 0) || ((opnameIs wa wb wc wd 6155984671800124794 7277816997830721536 0 0) || false))))))))))))))))))"),
        "opnameFused changed:\n{found}"
    );
    let found = declaration(&kernel, "opnameListed");
    assert!(
        found.contains("body: (if (style =? 0) then (opnameDecomposed wa wb wc wd) else (if (style =? 1) then (opnameFused wa wb wc wd) else (if (style =? 2) then (opnameEncoder wa wb wc wd) else false)))"),
        "opnameListed changed:\n{found}"
    );
    let found = declaration(&kernel, "opnameAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 34) && ((Kernel.Wire::fieldAt value 1 1) <=? 2)) then (if (opnameListed (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 8) (Kernel.Wire::fieldAt value 18 8) (Kernel.Wire::fieldAt value 26 8)) then (prim utf8_encode \"supported\" : Bytes) else (prim utf8_encode \"unsupported operator\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "opnameAnswer changed:\n{found}"
    );
    // Each listed operator is the four words of its name's bytes: the names are
    // here, and the numbers in the model must be theirs.
    let styles: [(&str, &[&str]); 2] = [
        (
            "opnameDecomposed",
            &[
                "Add",
                "Cast",
                "Concat",
                "ConstantOfShape",
                "Div",
                "Equal",
                "Expand",
                "Gather",
                "Greater",
                "MatMul",
                "MatMulNBits",
                "Mul",
                "Neg",
                "Pow",
                "Range",
                "ReduceMean",
                "Reshape",
                "ScatterND",
                "Shape",
                "Sigmoid",
                "Slice",
                "Softmax",
                "Sqrt",
                "Transpose",
                "Trilu",
                "Unsqueeze",
                "Where",
            ],
        ),
        (
            "opnameFused",
            &[
                "Cast",
                "Concat",
                "Constant",
                "Gather",
                "GroupQueryAttention",
                "MatMul",
                "MatMulNBits",
                "Mul",
                "ReduceSum",
                "Reshape",
                "RotaryEmbedding",
                "Shape",
                "Sigmoid",
                "SimplifiedLayerNormalization",
                "SkipSimplifiedLayerNormalization",
                "Sub",
                "Transpose",
                "Unsqueeze",
            ],
        ),
    ];
    for (definition, names) in styles {
        let listed = declaration(&kernel, definition);
        for name in names {
            let mut bytes = [0u8; 32];
            bytes[..name.len()].copy_from_slice(name.as_bytes());
            let words: Vec<String> = bytes
                .chunks(8)
                .map(|w| u64::from_be_bytes(w.try_into().expect("eight bytes")).to_string())
                .collect();
            let wanted = format!("(opnameIs wa wb wc wd {})", words.join(" "));
            assert!(
                listed.contains(&wanted),
                "{name} is not in {definition}:\n{listed}"
            );
        }
        assert_eq!(
            listed.matches("(opnameIs ").count(),
            names.len(),
            "{definition} lists a name the test does not"
        );
    }
    // Row 10: an operator off both lists, and one on the decomposed list asked of
    // a fused graph, are refused by name by the compiled kernel; a listed one is not.
    {
        use hologram_ai_kernel_client::opname::{self, Style};
        let name = |n: &str| opname::name(n).expect("a name of at most 32 bytes");
        assert_claimed(
            18,
            &[
                (
                    opname::request(Style::Decomposed, &name("LayerNormalization")).to_vec(),
                    "unsupported operator",
                ),
                (
                    opname::request(Style::Fused, &name("LayerNormalization")).to_vec(),
                    "unsupported operator",
                ),
                (
                    opname::request(Style::Fused, &name("Add")).to_vec(),
                    "unsupported operator",
                ),
                (
                    opname::request(Style::Decomposed, &name("MatMulNBits")).to_vec(),
                    "supported",
                ),
            ],
        );
    }
    assert_routed(18, "Kernel.Opnames::opnameAnswer");
}
/// `KN-19`: A model configuration is recognised exactly when its type is a listed family and its shape is one the engine runs
#[test]
fn kernel_kn_19() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Family.lex.tex");
    let proofs = model("app/src/Proofs/Family.lex.tex");
    assert_pinned(
        &proofs,
        "family_listed_decides",
        "(((Kernel.Family::familyListed kind) = false) <-> (((kind = 8176115172474093568) -> (1 = 0)) /\\ (((kind = 8176115172490870784) -> (1 = 0)) /\\ ((kind = 7812726575987687424) -> (1 = 0)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "family_shaped_decides",
        "(((Kernel.Family::familyShaped heads groups headSize layers width vocabulary) = true) <-> ((1 <= groups) /\\ ((groups <= heads) /\\ (((prim remainder heads groups 0 : Nat) = 0) /\\ ((2 <= headSize) /\\ (((prim remainder headSize 2 0 : Nat) = 0) /\\ ((1 <= layers) /\\ ((1 <= width) /\\ (1 <= vocabulary)))))))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "family_supported_decides",
        "(((Kernel.Family::familyCode kind activation heads groups headSize layers width vocabulary) = 2) <-> (((Kernel.Family::familyListed kind) = true) /\\ ((activation = 8316297436683304960) /\\ ((Kernel.Family::familyShaped heads groups headSize layers width vocabulary) = true))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "family_unknown_decides",
        "(((Kernel.Family::familyCode kind activation heads groups headSize layers width vocabulary) = 0) <-> ((Kernel.Family::familyListed kind) = false))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "familyListed");
    assert!(
        found.contains("body: ((kind =? 8176115172474093568) || ((kind =? 8176115172490870784) || (kind =? 7812726575987687424)))"),
        "familyListed changed:\n{found}"
    );
    let found = declaration(&kernel, "familyGated");
    assert!(
        found.contains("body: (activation =? 8316297436683304960)"),
        "familyGated changed:\n{found}"
    );
    let found = declaration(&kernel, "familyShaped");
    assert!(
        found.contains("body: ((1 <=? groups) && ((groups <=? heads) && (((prim remainder heads groups 0 : Nat) =? 0) && ((2 <=? headSize) && (((prim remainder headSize 2 0 : Nat) =? 0) && ((1 <=? layers) && ((1 <=? width) && (1 <=? vocabulary))))))))"),
        "familyShaped changed:\n{found}"
    );
    let found = declaration(&kernel, "familyCode");
    assert!(
        found.contains("body: (if (familyListed kind) then (if ((familyGated activation) && (familyShaped heads groups headSize layers width vocabulary)) then 2 else 1) else 0)"),
        "familyCode changed:\n{found}"
    );
    let found = declaration(&kernel, "familyAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 33) then (familyText (familyCode (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 17 2) (Kernel.Wire::fieldAt value 19 2) (Kernel.Wire::fieldAt value 21 2) (Kernel.Wire::fieldAt value 23 2) (Kernel.Wire::fieldAt value 25 4) (Kernel.Wire::fieldAt value 29 4))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "familyAnswer changed:\n{found}"
    );
    // The listed types and the activation are the bytes of their names, not numbers chosen here.
    let word = |name: &str| {
        let mut bytes = [0u8; 8];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        u64::from_be_bytes(bytes).to_string()
    };
    let listed = declaration(&kernel, "familyListed");
    for name in ["qwen2", "qwen3", "llama"] {
        assert!(
            listed.contains(&format!("(kind =? {})", word(name))),
            "{name} is not listed:\n{listed}"
        );
    }
    assert!(
        declaration(&kernel, "familyGated").contains(&format!("(activation =? {})", word("silu"))),
        "the gated activation is not silu"
    );
    // Row 10: Gemma and a mixture of experts (Mixtral) are refused by name as unknown
    // families, a listed family with GELU as an unsupported shape, by the compiled
    // kernel; a type or activation longer than eight bytes (qwen3_moe,
    // gelu_pytorch_tanh) has no word, so it is never put to the kernel at all.
    {
        use hologram_ai_kernel_client::family::{self, Shape};
        let word = |w: &str| family::word(w).expect("a word of at most eight bytes");
        let qwen = Shape {
            heads: 12,
            groups: 2,
            head_size: 128,
            layers: 28,
            width: 1536,
            vocabulary: 151_936,
        };
        let gemma = Shape {
            heads: 8,
            groups: 4,
            head_size: 256,
            layers: 26,
            width: 2304,
            vocabulary: 256_000,
        };
        let mixtral = Shape {
            heads: 32,
            groups: 8,
            head_size: 128,
            layers: 32,
            width: 4096,
            vocabulary: 32_000,
        };
        assert_claimed(
            19,
            &[
                (
                    family::request(&word("gemma2"), &word("gelu"), gemma).to_vec(),
                    "rejected: unknown family",
                ),
                (
                    family::request(&word("mixtral"), &word("silu"), mixtral).to_vec(),
                    "rejected: unknown family",
                ),
                (
                    family::request(&word("llama"), &word("gelu"), qwen).to_vec(),
                    "rejected: unsupported shape",
                ),
                (
                    family::request(&word("qwen2"), &word("silu"), qwen).to_vec(),
                    "supported",
                ),
            ],
        );
        for long in ["qwen3_moe", "qwen2_moe", "gelu_pytorch_tanh"] {
            assert_eq!(family::word(long), None, "{long} has no word");
        }
    }
    assert_routed(19, "Kernel.Family::familyAnswer");
}
/// `KN-20`: A MatMulNBits weight and scale length is the one its rows, columns and block size imply, and its blocks are the fewest that cover a row
#[test]
fn kernel_kn_20() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Nbits.lex.tex");
    let proofs = model("app/src/Proofs/Nbits.lex.tex");
    assert_pinned(
        &proofs,
        "nbits_shaped_decides",
        "(((Kernel.Nbits::nbitsShaped part rows columns size) = true) <-> ((part <= 2) /\\ ((1 <= rows) /\\ ((rows <= 16777215) /\\ ((1 <= columns) /\\ ((columns <= 16777215) /\\ ((Kernel.Nbits::nbitsBlock size) = true)))))))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "nbits_block_bounds",
        "(((Kernel.Nbits::nbitsBlock size) = true) -> ((16 <= size) /\\ ((size <= 256) /\\ ((prim remainder size 16 0 : Nat) = 0))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "nbits_blocks_fewest_16",
        "((columns <= (prim multiply (Kernel.Nbits::nbitsBlocks columns 16) 16 : Nat)) /\\ (((prim multiply (Kernel.Nbits::nbitsBlocks columns 16) 16 : Nat) + 1) <= (columns + 16)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "nbits_blocks_fewest_32",
        "((columns <= (prim multiply (Kernel.Nbits::nbitsBlocks columns 32) 32 : Nat)) /\\ (((prim multiply (Kernel.Nbits::nbitsBlocks columns 32) 32 : Nat) + 1) <= (columns + 32)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "nbits_blocks_fewest_64",
        "((columns <= (prim multiply (Kernel.Nbits::nbitsBlocks columns 64) 64 : Nat)) /\\ (((prim multiply (Kernel.Nbits::nbitsBlocks columns 64) 64 : Nat) + 1) <= (columns + 64)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "nbits_blocks_fewest_128",
        "((columns <= (prim multiply (Kernel.Nbits::nbitsBlocks columns 128) 128 : Nat)) /\\ (((prim multiply (Kernel.Nbits::nbitsBlocks columns 128) 128 : Nat) + 1) <= (columns + 128)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "nbits_blocks_fewest_256",
        "((columns <= (prim multiply (Kernel.Nbits::nbitsBlocks columns 256) 256 : Nat)) /\\ (((prim multiply (Kernel.Nbits::nbitsBlocks columns 256) 256 : Nat) + 1) <= (columns + 256)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "nbitsBlock");
    assert!(
        found.contains("body: ((size =? 16) || ((size =? 32) || ((size =? 64) || ((size =? 128) || (size =? 256)))))"),
        "nbitsBlock changed:\n{found}"
    );
    let found = declaration(&kernel, "nbitsBlocks");
    assert!(
        found.contains(
            "body: (prim quotient (columns + (prim subtract size 1 : Nat)) size 0 : Nat)"
        ),
        "nbitsBlocks changed:\n{found}"
    );
    let found = declaration(&kernel, "nbitsPerBlock");
    assert!(
        found.contains("body: (if (part =? 0) then (prim quotient size 2 0 : Nat) else (if (part =? 1) then 4 else 2))"),
        "nbitsPerBlock changed:\n{found}"
    );
    let found = declaration(&kernel, "nbitsLength");
    assert!(
        found.contains("body: (prim multiply (prim multiply rows (nbitsBlocks columns size) : Nat) (nbitsPerBlock part size) : Nat)"),
        "nbitsLength changed:\n{found}"
    );
    let found = declaration(&kernel, "nbitsAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 12) then (if (nbitsShaped (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4) (Kernel.Wire::fieldAt value 6 4) (Kernel.Wire::fieldAt value 10 2)) then (Kernel.Wire::decimal (nbitsLength (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4) (Kernel.Wire::fieldAt value 6 4) (Kernel.Wire::fieldAt value 10 2))) else (prim utf8_encode \"bad shape\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "nbitsAnswer changed:\n{found}"
    );
    // Decision 20 refuses a block size it does not take with bad shape, and a short request as invalid.
    {
        use hologram_ai_kernel_client::nbits::{self, Part};
        assert_claimed(
            20,
            &[
                (nbits::request(Part::Codes, 2, 64, 32).to_vec(), "64"),
                (nbits::request(Part::Codes, 2, 64, 48).to_vec(), "bad shape"),
                (
                    nbits::request(Part::Codes, 2, 64, 32)[..11].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(20, "Kernel.Nbits::nbitsAnswer");
}
/// `KN-21`: The precision chosen is the first offered one in the stated order that fits the budget, and none is chosen exactly when no offered one fits
#[test]
fn kernel_kn_21() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Precision.lex.tex");
    let proofs = model("app/src/Proofs/Precision.lex.tex");
    assert_pinned(
        &proofs,
        "precision_fits_decides",
        "(((Kernel.Precision::precisionFits offered size budget) = true) <-> ((offered = 1) /\\ (size <= budget)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "precision_first",
        "(((Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize budget) = 0) <-> ((Kernel.Precision::precisionFits lean leanSize budget) = true))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "precision_second",
        "(((Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize budget) = 1) <-> (((Kernel.Precision::precisionFits lean leanSize budget) = false) /\\ ((Kernel.Precision::precisionFits plain plainSize budget) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "precision_third",
        "(((Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize budget) = 2) <-> (((Kernel.Precision::precisionFits lean leanSize budget) = false) /\\ (((Kernel.Precision::precisionFits plain plainSize budget) = false) /\\ ((Kernel.Precision::precisionFits wide wideSize budget) = true))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "precision_none",
        "(((Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize budget) = 3) <-> (((Kernel.Precision::precisionFits lean leanSize budget) = false) /\\ (((Kernel.Precision::precisionFits plain plainSize budget) = false) /\\ ((Kernel.Precision::precisionFits wide wideSize budget) = false))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "precision_budget_monotone",
        "((budget <= wider) -> ((Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize wider) <= (Kernel.Precision::precisionCode lean leanSize plain plainSize wide wideSize budget)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "precisionFits");
    assert!(
        found.contains("body: ((offered =? 1) && (size <=? budget))"),
        "precisionFits changed:\n{found}"
    );
    let found = declaration(&kernel, "precisionCode");
    assert!(
        found.contains("body: (if (precisionFits lean leanSize budget) then 0 else (if (precisionFits plain plainSize budget) then 1 else (if (precisionFits wide wideSize budget) then 2 else 3)))"),
        "precisionCode changed:\n{found}"
    );
    let found = declaration(&kernel, "precisionText");
    assert!(
        found.contains("body: (if (code =? 0) then (prim utf8_encode \"q4f16\" : Bytes) else (if (code =? 1) then (prim utf8_encode \"q4\" : Bytes) else (if (code =? 2) then (prim utf8_encode \"fp16\" : Bytes) else (prim utf8_encode \"none fits\" : Bytes))))"),
        "precisionText changed:\n{found}"
    );
    let found = declaration(&kernel, "precisionAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 36) && (((Kernel.Wire::fieldAt value 1 1) <=? 1) && (((Kernel.Wire::fieldAt value 2 1) <=? 1) && ((Kernel.Wire::fieldAt value 3 1) <=? 1)))) then (precisionText (precisionCode (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 12 8) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 20 8) (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 28 8) (Kernel.Wire::fieldAt value 4 8))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "precisionAnswer changed:\n{found}"
    );
    // Row 10: a record offering none of q4f16, q4 and fp16 (only q8, int8 or bnb4
    // files) is answered `none fits` by the compiled kernel; one offering q4 gets q4.
    {
        use hologram_ai_kernel_client::precision;
        let budget = 4 << 30;
        assert_claimed(
            21,
            &[
                (
                    precision::request(budget, None, None, None).to_vec(),
                    "none fits",
                ),
                (
                    precision::request(budget, None, Some(1_100_000_000), None).to_vec(),
                    "q4",
                ),
            ],
        );
    }
    assert_routed(21, "Kernel.Precision::precisionAnswer");
}
/// `KN-22`: The device is WebGPU exactly when the adapter can hold the model, the CPU exactly when it cannot and the model is within the CPU ceiling, and otherwise the model is refused
#[test]
fn kernel_kn_22() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Device.lex.tex");
    let proofs = model("app/src/Proofs/Device.lex.tex");
    assert_pinned(
        &proofs,
        "device_webgpu_decides",
        "(((Kernel.Device::deviceCode present limit largest model ceiling) = 2) <-> ((Kernel.Backend::gpuUsable present limit largest) = true))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "device_cpu_decides",
        "(((Kernel.Device::deviceCode present limit largest model ceiling) = 1) <-> (((Kernel.Backend::gpuUsable present limit largest) = false) /\\ (model <= ceiling)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "device_refused_decides",
        "(((Kernel.Device::deviceCode present limit largest model ceiling) = 0) <-> (((Kernel.Backend::gpuUsable present limit largest) = false) /\\ ((ceiling + 1) <= model)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "deviceCode");
    assert!(
        found.contains("body: (if (Kernel.Backend::gpuUsable present limit largest) then 2 else (if (model <=? ceiling) then 1 else 0))"),
        "deviceCode changed:\n{found}"
    );
    let found = declaration(&kernel, "deviceText");
    assert!(
        found.contains("body: (if (code =? 2) then (prim utf8_encode \"webgpu\" : Bytes) else (if (code =? 1) then (prim utf8_encode \"cpu\" : Bytes) else (prim utf8_encode \"refused: too large for cpu\" : Bytes)))"),
        "deviceText changed:\n{found}"
    );
    let found = declaration(&kernel, "deviceAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 34) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)) then (deviceText (deviceCode (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 8) (Kernel.Wire::fieldAt value 18 8) (Kernel.Wire::fieldAt value 26 8))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "deviceAnswer changed:\n{found}"
    );
    // Decision 22 refuses a model with no adapter that is over the CPU ceiling, and a short request as invalid.
    {
        use hologram_ai_kernel_client::device;
        assert_claimed(
            22,
            &[
                (
                    device::request(true, 1000, 500, 2000, 100).to_vec(),
                    "webgpu",
                ),
                (device::request(false, 1000, 500, 50, 100).to_vec(), "cpu"),
                (
                    device::request(false, 1000, 500, 200, 100).to_vec(),
                    "refused: too large for cpu",
                ),
                (
                    device::request(false, 1000, 500, 200, 100)[..33].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(22, "Kernel.Device::deviceAnswer");
}
/// `KN-23`: A span in a model stored as several files is accepted exactly when its file is one the model declares, it follows the previous span, and it ends inside that file
#[test]
fn kernel_kn_23() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Chunks.lex.tex");
    let proofs = model("app/src/Proofs/Chunks.lex.tex");
    assert_pinned(
        &proofs,
        "chunk_declared_decides",
        "(((Kernel.Chunks::chunkDeclared declared index) = true) <-> ((declared <= 100) /\\ (index <= declared)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "chunk_inside_decides",
        "(((Kernel.Chunks::chunkInside fileLength offset length previousEnd) = true) <-> ((previousEnd <= offset) /\\ ((offset + length) <= fileLength)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "chunk_spans_disjoint",
        "((((Kernel.Chunks::chunkInside fileLength offset length previousEnd) = true) /\\ ((Kernel.Chunks::chunkInside fileLength nextOffset nextLength (offset + length)) = true)) -> (((offset + length) <= nextOffset) /\\ ((nextOffset + nextLength) <= fileLength)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "chunkDeclared");
    assert!(
        found.contains("body: ((declared <=? 100) && (index <=? declared))"),
        "chunkDeclared changed:\n{found}"
    );
    let found = declaration(&kernel, "chunkInside");
    assert!(
        found.contains("body: ((previousEnd <=? offset) && ((offset + length) <=? fileLength))"),
        "chunkInside changed:\n{found}"
    );
    let found = declaration(&kernel, "chunkAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 35) then (if (chunkDeclared (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1)) then (if (chunkInside (Kernel.Wire::fieldAt value 3 8) (Kernel.Wire::fieldAt value 11 8) (Kernel.Wire::fieldAt value 19 8) (Kernel.Wire::fieldAt value 27 8)) then (prim utf8_encode \"accept\" : Bytes) else (prim utf8_encode \"reject\" : Bytes)) else (prim utf8_encode \"undeclared file\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "chunkAnswer changed:\n{found}"
    );
    // Decision 23 rejects a span past the end of the file, refuses a file beyond those declared, and a short request as invalid.
    {
        use hologram_ai_kernel_client::chunk::{self, File};
        let file = |index| File {
            declared: 1,
            index,
            length: 100,
        };
        assert_claimed(
            23,
            &[
                (chunk::request(file(1), 10, 20, 0).to_vec(), "accept"),
                (chunk::request(file(1), 90, 20, 0).to_vec(), "reject"),
                (
                    chunk::request(file(2), 10, 20, 0).to_vec(),
                    "undeclared file",
                ),
                (
                    chunk::request(file(1), 10, 20, 0)[..34].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(23, "Kernel.Chunks::chunkAnswer");
}
/// `KN-24`: The file named for a task, precision and part is the one the export convention gives, and a part beyond the declared count is refused
#[test]
fn kernel_kn_24() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Source.lex.tex");
    let proofs = model("app/src/Proofs/Source.lex.tex");
    assert_pinned(
        &proofs,
        "source_named_decides",
        "(((Kernel.Source::sourceCode task precision declared part) = 4) <-> ((task <= 2) /\\ ((precision <= 3) /\\ ((declared <= 100) /\\ (part <= declared)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "source_undeclared_decides",
        "(((Kernel.Source::sourceCode task precision declared part) = 3) <-> ((task <= 2) /\\ ((precision <= 3) /\\ ((declared <= 100) /\\ ((declared + 1) <= part)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "source_task_decides",
        "(((Kernel.Source::sourceCode task precision declared part) = 0) <-> (3 <= task))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "sourceCode");
    assert!(
        found.contains("body: (if (task <=? 2) then (if (precision <=? 3) then (if (declared <=? 100) then (if (part <=? declared) then 4 else 3) else 2) else 1) else 0)"),
        "sourceCode changed:\n{found}"
    );
    let found = declaration(&kernel, "sourceGraph");
    assert!(
        found.contains("body: (if (precision =? 0) then (prim utf8_encode \"onnx/model_q4f16.onnx\" : Bytes) else (if (precision =? 1) then (prim utf8_encode \"onnx/model_q4.onnx\" : Bytes) else (if (precision =? 2) then (prim utf8_encode \"onnx/model_fp16.onnx\" : Bytes) else (prim utf8_encode \"onnx/model.onnx\" : Bytes))))"),
        "sourceGraph changed:\n{found}"
    );
    let found = declaration(&kernel, "sourceName");
    assert!(
        found.contains("body: (if (part =? 0) then (sourceGraph precision) else (if (part =? 1) then (prim append (sourceGraph precision) (prim utf8_encode \"_data\" : Bytes) : Bytes) else (prim append (prim append (sourceGraph precision) (prim utf8_encode \"_data_\" : Bytes) : Bytes) (Kernel.Wire::decimal (prim subtract part 1 : Nat)) : Bytes)))"),
        "sourceName changed:\n{found}"
    );
    let found = declaration(&kernel, "sourceText");
    assert!(
        found.contains("body: (if (code =? 4) then (sourceName precision part) else (if (code =? 3) then (prim utf8_encode \"undeclared file\" : Bytes) else (if (code =? 2) then (prim utf8_encode \"too many files\" : Bytes) else (if (code =? 1) then (prim utf8_encode \"unsupported precision\" : Bytes) else (prim utf8_encode \"unsupported task\" : Bytes)))))"),
        "sourceText changed:\n{found}"
    );
    let found = declaration(&kernel, "sourceAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 5) then (sourceText (sourceCode (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 4 1)) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 4 1)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "sourceAnswer changed:\n{found}"
    );
    // Decision 24 refuses an undeclared part, too many files, a precision and a task it has no files for, and a short request.
    {
        use hologram_ai_kernel_client::source::{self, Precision};
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            24,
            &[
                (
                    source::request(Precision::Q4, 1, 1).to_vec(),
                    "onnx/model_q4.onnx_data",
                ),
                (
                    source::request(Precision::Q4, 1, 2).to_vec(),
                    "undeclared file",
                ),
                (
                    source::request(Precision::Q4, 101, 1).to_vec(),
                    "too many files",
                ),
                (unhex("1800ff0101"), "unsupported precision"),
                (unhex("18ff010101"), "unsupported task"),
                (
                    source::request(Precision::Q4, 1, 1)[..4].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(24, "Kernel.Source::sourceAnswer");
}
/// `KN-25`: Generation stops exactly when the token is one of the stop ids of the model or the budget is spent, and a stop id is reported before the budget
#[test]
fn kernel_kn_25() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Stop.lex.tex");
    let proofs = model("app/src/Proofs/Stop.lex.tex");
    assert_pinned(
        &proofs,
        "stop_differs_decides",
        "(((Kernel.Stop::stopDiffers token listed) = true) <-> (Differs token listed))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "stop_none_from_decides",
        "(((Kernel.Stop::stopNoneFrom value position remaining token) = true) <-> (NoneFrom value position remaining token))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "stop_spent_decides",
        "(((Kernel.Stop::stopSpent generated budget) = true) <-> (budget <= generated))",
        "propext",
    );
    let found = declaration(&kernel, "stopIdAt");
    assert!(
        found.contains(
            "body: (Kernel.Wire::fieldAt value (14 + (prim multiply 4 position : Nat)) 4)"
        ),
        "stopIdAt changed:\n{found}"
    );
    let found = declaration(&kernel, "stopDiffers");
    assert!(
        found.contains("body: ((listed <? token) || (token <? listed))"),
        "stopDiffers changed:\n{found}"
    );
    let found = declaration(&kernel, "stopNoneFrom");
    assert!(
        found.contains("body: (match remaining
    | Nat.zero => true
    | Nat.succ rest => ((stopDiffers token (stopIdAt value position)) && (stopNoneFrom value (position + 1) rest token)))"),
        "stopNoneFrom changed:\n{found}"
    );
    let found = declaration(&kernel, "stopSpent");
    assert!(
        found.contains("body: (budget <=? generated)"),
        "stopSpent changed:\n{found}"
    );
    let found = declaration(&kernel, "stopAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? (14 + (prim multiply 4 (Kernel.Wire::fieldAt value 1 1) : Nat))) && ((Kernel.Wire::fieldAt value 1 1) <=? 16)) then (if (stopNoneFrom value 0 (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4)) then (if (stopSpent (Kernel.Wire::fieldAt value 6 4) (Kernel.Wire::fieldAt value 10 4)) then (prim utf8_encode \"stop: budget\" : Bytes) else (prim utf8_encode \"continue\" : Bytes)) else (prim utf8_encode \"stop: end token\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "stopAnswer changed:\n{found}"
    );
    // Decision 25 stops on an end token or a spent budget, and refuses more stop ids than a request carries as invalid.
    {
        use hologram_ai_kernel_client::stop;
        assert_claimed(
            25,
            &[
                (stop::request(5, 1, 10, &[2]).bytes().to_vec(), "continue"),
                (
                    stop::request(2, 1, 10, &[2]).bytes().to_vec(),
                    "stop: end token",
                ),
                (
                    stop::request(5, 10, 10, &[2]).bytes().to_vec(),
                    "stop: budget",
                ),
                (
                    stop::request(5, 1, 10, &[2; 17]).bytes().to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(25, "Kernel.Stop::stopAnswer");
}
/// `KN-26`: A cached block is evicted exactly when it is not pinned and space is still short, and does not fit is answered exactly when no candidate remains and space is still short
#[test]
fn kernel_kn_26() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Evict.lex.tex");
    let proofs = model("app/src/Proofs/Evict.lex.tex");
    assert_pinned(
        &proofs,
        "evict_short_decides",
        "(((Kernel.Evict::evictShort needed free freed) = true) <-> (((free + freed) + 1) <= needed))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "evict_decides",
        "(((Kernel.Evict::evictCode pinned remaining needed free freed) = 2) <-> ((pinned = 0) /\\ ((1 <= remaining) /\\ (((free + freed) + 1) <= needed))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "evict_does_not_fit_decides",
        "(((Kernel.Evict::evictCode pinned remaining needed free freed) = 0) <-> ((remaining = 0) /\\ (((free + freed) + 1) <= needed)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "evict_enough_decides",
        "(((Kernel.Evict::evictCode pinned remaining needed free freed) = 3) <-> (needed <= (free + freed)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "evictShort");
    assert!(
        found.contains("body: ((free + freed) <? needed)"),
        "evictShort changed:\n{found}"
    );
    let found = declaration(&kernel, "evictCode");
    assert!(
        found.contains("body: (if (evictShort needed free freed) then (if (remaining =? 0) then 0 else (if (pinned =? 0) then 2 else 1)) else 3)"),
        "evictCode changed:\n{found}"
    );
    let found = declaration(&kernel, "evictText");
    assert!(
        found.contains("body: (if (code =? 3) then (prim utf8_encode \"enough\" : Bytes) else (if (code =? 2) then (prim utf8_encode \"evict\" : Bytes) else (if (code =? 1) then (prim utf8_encode \"keep\" : Bytes) else (prim utf8_encode \"does not fit\" : Bytes))))"),
        "evictText changed:\n{found}"
    );
    let found = declaration(&kernel, "evictAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 30) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)) then (evictText (evictCode (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4) (Kernel.Wire::fieldAt value 6 8) (Kernel.Wire::fieldAt value 14 8) (Kernel.Wire::fieldAt value 22 8))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "evictAnswer changed:\n{found}"
    );
    // Decision 26 keeps a pinned block, answers does not fit when no candidate is left, and refuses a short request.
    {
        use hologram_ai_kernel_client::evict;
        assert_claimed(
            26,
            &[
                (evict::request(false, 3, 10, 10, 0).to_vec(), "enough"),
                (evict::request(false, 3, 100, 10, 0).to_vec(), "evict"),
                (evict::request(true, 3, 100, 10, 0).to_vec(), "keep"),
                (
                    evict::request(false, 0, 100, 10, 0).to_vec(),
                    "does not fit",
                ),
                (
                    evict::request(false, 3, 100, 10, 0)[..29].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(26, "Kernel.Evict::evictAnswer");
}
/// `KN-27`: A computed prefix is reused exactly when its address equals the one asked for and it is shorter than the prompt
#[test]
fn kernel_kn_27() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Reuse.lex.tex");
    let proofs = model("app/src/Proofs/Reuse.lex.tex");
    assert_pinned(
        &proofs,
        "reuse_shorter_decides",
        "(((Kernel.Reuse::reuseShorter heldLength promptLength) = true) <-> ((1 <= heldLength) /\\ ((heldLength + 1) <= promptLength)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "reusable_decides",
        "(((Kernel.Reuse::reusable asked held heldLength promptLength) = true) <-> (((Kernel.Admission::sameLabel asked held) = true) /\\ ((Kernel.Reuse::reuseShorter heldLength promptLength) = true)))",
        "propext",
    );
    let found = declaration(&kernel, "reuseShorter");
    assert!(
        found.contains("body: ((1 <=? heldLength) && (heldLength <? promptLength))"),
        "reuseShorter changed:\n{found}"
    );
    let found = declaration(&kernel, "reusable");
    assert!(
        found.contains("body: ((Kernel.Admission::sameLabel asked held) && (reuseShorter heldLength promptLength))"),
        "reusable changed:\n{found}"
    );
    let found = declaration(&kernel, "reuseAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 73) then (if (reusable (Kernel.Admission::labelAt value 1) (Kernel.Admission::labelAt value 33) (Kernel.Wire::fieldAt value 65 4) (Kernel.Wire::fieldAt value 69 4)) then (prim utf8_encode \"reuse\" : Bytes) else (prim utf8_encode \"recompute\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "reuseAnswer changed:\n{found}"
    );
    // Decision 27 recomputes under a different label or a prefix as long as the prompt, and refuses a short request.
    {
        use hologram_ai_kernel_client::reuse;
        assert_claimed(
            27,
            &[
                (reuse::request(&[1; 32], &[1; 32], 4, 10).to_vec(), "reuse"),
                (
                    reuse::request(&[1; 32], &[2; 32], 4, 10).to_vec(),
                    "recompute",
                ),
                (
                    reuse::request(&[1; 32], &[1; 32], 10, 10).to_vec(),
                    "recompute",
                ),
                (
                    reuse::request(&[1; 32], &[1; 32], 4, 10)[..72].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(27, "Kernel.Reuse::reuseAnswer");
}
/// `KN-28`: A model is ready exactly when the addresses admitted are the ones its manifest lists, every listed tensor is admitted, and it lists at least one
#[test]
fn kernel_kn_28() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Manifest.lex.tex");
    let proofs = model("app/src/Proofs/Manifest.lex.tex");
    assert_pinned(
        &proofs,
        "manifest_complete_decides",
        "(((Kernel.Manifest::manifestComplete listed admitted) = true) <-> ((1 <= listed) /\\ (admitted = listed)))",
        "propext",
    );
    let found = declaration(&kernel, "manifestComplete");
    assert!(
        found.contains("body: ((1 <=? listed) && (admitted =? listed))"),
        "manifestComplete changed:\n{found}"
    );
    let found = declaration(&kernel, "manifestAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 81) then (if (Kernel.Admission::sameLabel (Kernel.Admission::labelAt value 1) (Kernel.Admission::labelAt value 33)) then (if (manifestComplete (Kernel.Wire::fieldAt value 65 8) (Kernel.Wire::fieldAt value 73 8)) then (prim utf8_encode \"ready\" : Bytes) else (prim utf8_encode \"incomplete\" : Bytes)) else (prim utf8_encode \"refused\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "manifestAnswer changed:\n{found}"
    );
    // Decision 28 answers incomplete for a tensor not yet admitted, refused for a different label, and refuses a short request.
    {
        use hologram_ai_kernel_client::manifest;
        assert_claimed(
            28,
            &[
                (
                    manifest::request(&[1; 32], &[1; 32], 3, 3).to_vec(),
                    "ready",
                ),
                (
                    manifest::request(&[1; 32], &[1; 32], 3, 2).to_vec(),
                    "incomplete",
                ),
                (
                    manifest::request(&[1; 32], &[2; 32], 3, 3).to_vec(),
                    "refused",
                ),
                (
                    manifest::request(&[1; 32], &[1; 32], 3, 3)[..80].to_vec(),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(28, "Kernel.Manifest::manifestAnswer");
}
/// `KN-29`: An exact sum is accepted exactly when the activation, weight and group bits together are no wider than the accumulator, and a wider accumulator or a shorter group never loses it
#[test]
fn kernel_kn_29() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Exact.lex.tex");
    let proofs = model("app/src/Proofs/Exact.lex.tex");
    assert_pinned(
        &proofs,
        "exact_fits_decides",
        "(((Kernel.Exact::exactFits activationBits weightBits groupBits accumulatorBits) = true) <-> ((1 <= activationBits) /\\ ((1 <= weightBits) /\\ (((activationBits + weightBits) + groupBits) <= accumulatorBits))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "exact_fits_wider",
        "((accumulatorBits <= wider) -> (((Kernel.Exact::exactFits activationBits weightBits groupBits accumulatorBits) = true) -> ((Kernel.Exact::exactFits activationBits weightBits groupBits wider) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "exact_fits_shorter",
        "((shorter <= groupBits) -> (((Kernel.Exact::exactFits activationBits weightBits groupBits accumulatorBits) = true) -> ((Kernel.Exact::exactFits activationBits weightBits shorter accumulatorBits) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "exactFits");
    assert!(
        found.contains("body: (((1 <=? activationBits) && (1 <=? weightBits)) && (((activationBits + weightBits) + groupBits) <=? accumulatorBits))"),
        "exactFits changed:\n{found}"
    );
    let found = declaration(&kernel, "exactAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 5) then (if (exactFits (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 4 1)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"may overflow\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "exactAnswer changed:\n{found}"
    );
    // Decision 29 answers may overflow when the bits exceed the accumulator, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            29,
            &[
                (unhex("1d08040520"), "exact"),
                (unhex("1d08081120"), "may overflow"),
                (unhex("1d080405"), "invalid request"),
            ],
        );
    }
    assert_routed(29, "Kernel.Exact::exactAnswer");
}
/// `KN-30`: A run is reproducible everywhere exactly when its products and every other stage are exact, on the same hardware exactly when it is not and its float reductions keep a fixed order, and otherwise not at all
#[test]
fn kernel_kn_30() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Determinism.lex.tex");
    let proofs = model("app/src/Proofs/Determinism.lex.tex");
    assert_pinned(
        &proofs,
        "class_everywhere_decides",
        "(((Kernel.Determinism::classCode productsExact restExact fixedOrder) = 2) <-> ((productsExact = 1) /\\ (restExact = 1)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "class_same_hardware_decides",
        "(((Kernel.Determinism::classCode productsExact restExact fixedOrder) = 1) <-> ((((productsExact = 1) /\\ (restExact = 1)) -> (1 = 0)) /\\ (fixedOrder = 1)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "class_none_decides",
        "(((Kernel.Determinism::classCode productsExact restExact fixedOrder) = 0) <-> ((((productsExact = 1) /\\ (restExact = 1)) -> (1 = 0)) /\\ ((fixedOrder = 1) -> (1 = 0))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "comparable_decides",
        "(((Kernel.Determinism::comparable classOfRun sameAdapter) = true) <-> (((classOfRun = 2) -> (1 = 0)) -> ((classOfRun = 1) /\\ (sameAdapter = 1))))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "classCode");
    assert!(
        found.contains("body: (if ((productsExact =? 1) && (restExact =? 1)) then 2 else (if (fixedOrder =? 1) then 1 else 0))"),
        "classCode changed:\n{found}"
    );
    let found = declaration(&kernel, "comparable");
    assert!(
        found.contains("body: ((classOfRun =? 2) || ((classOfRun =? 1) && (sameAdapter =? 1)))"),
        "comparable changed:\n{found}"
    );
    let found = declaration(&kernel, "classAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 4) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)) && (((Kernel.Wire::fieldAt value 2 1) <=? 1) && ((Kernel.Wire::fieldAt value 3 1) <=? 1))) then (classText (classCode (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 3 1))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "classAnswer changed:\n{found}"
    );
    // Decision 30 answers none for a run neither exact nor fixed in order, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            30,
            &[
                (unhex("1e010100"), "everywhere"),
                (unhex("1e000101"), "same hardware"),
                (unhex("1e000000"), "none"),
                (unhex("1e0101"), "invalid request"),
            ],
        );
    }
    assert_routed(30, "Kernel.Determinism::classAnswer");
}
/// `KN-31`: A receipt is confirmed exactly when its model is ready, the run is comparable and the replayed output has the label it claims, a mismatch exactly when only the output differs, and otherwise not comparable
#[test]
fn kernel_kn_31() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Receipt.lex.tex");
    let proofs = model("app/src/Proofs/Receipt.lex.tex");
    assert_pinned(
        &proofs,
        "receipt_confirmed_decides",
        "(((Kernel.Receipt::receiptCode outputClaimed outputReplayed modelReady classOfRun sameAdapter) = 2) <-> ((modelReady = 1) /\\ (((Kernel.Determinism::comparable classOfRun sameAdapter) = true) /\\ ((Kernel.Admission::sameLabel outputClaimed outputReplayed) = true))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "receipt_mismatch_decides",
        "(((Kernel.Receipt::receiptCode outputClaimed outputReplayed modelReady classOfRun sameAdapter) = 1) <-> ((modelReady = 1) /\\ (((Kernel.Determinism::comparable classOfRun sameAdapter) = true) /\\ ((Kernel.Admission::sameLabel outputClaimed outputReplayed) = false))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "receipt_not_comparable_decides",
        "(((Kernel.Receipt::receiptCode outputClaimed outputReplayed modelReady classOfRun sameAdapter) = 0) <-> (((modelReady = 1) /\\ ((Kernel.Determinism::comparable classOfRun sameAdapter) = true)) -> (1 = 0)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "receiptCode");
    assert!(
        found.contains("body: (if ((modelReady =? 1) && (Kernel.Determinism::comparable classOfRun sameAdapter)) then (if (Kernel.Admission::sameLabel outputClaimed outputReplayed) then 2 else 1) else 0)"),
        "receiptCode changed:\n{found}"
    );
    let found = declaration(&kernel, "receiptAnswer");
    assert!(
        found.contains("body: (if (((((Foundation.Bytes::byteLength value) =? 68) && ((Kernel.Wire::fieldAt value 65 1) <=? 1)) && ((Kernel.Wire::fieldAt value 66 1) <=? 2)) && ((Kernel.Wire::fieldAt value 67 1) <=? 1)) then (receiptText (receiptCode (Kernel.Admission::labelAt value 1) (Kernel.Admission::labelAt value 33) (Kernel.Wire::fieldAt value 65 1) (Kernel.Wire::fieldAt value 66 1) (Kernel.Wire::fieldAt value 67 1))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "receiptAnswer changed:\n{found}"
    );
    // Decision 31 answers mismatch for another output, not comparable for a model not ready, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            31,
            &[
                (unhex("1f01010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010200"), "confirmed"),
                (unhex("1f01010101010101010101010101010101010101010101010101010101010101010202020202020202020202020202020202020202020202020202020202020202010200"), "mismatch"),
                (unhex("1f01010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000200"), "not comparable"),
                (unhex("1f010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010102"), "invalid request"),
            ],
        );
    }
    assert_routed(31, "Kernel.Receipt::receiptAnswer");
}
/// `KN-32`: A probe of a past answer replays exactly when its layer, position and component lie within the run and the run is comparable, approximately when they lie within it and the run is not, and is otherwise out of range
#[test]
fn kernel_kn_32() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Probe.lex.tex");
    let proofs = model("app/src/Proofs/Probe.lex.tex");
    assert_pinned(
        &proofs,
        "probe_exact_decides",
        "(((Kernel.Probe::probeCode layer layers position positions component components classOfRun sameAdapter) = 2) <-> (((layer + 1) <= layers) /\\ (((position + 1) <= positions) /\\ (((component + 1) <= components) /\\ ((Kernel.Determinism::comparable classOfRun sameAdapter) = true)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "probe_approximate_decides",
        "(((Kernel.Probe::probeCode layer layers position positions component components classOfRun sameAdapter) = 1) <-> (((layer + 1) <= layers) /\\ (((position + 1) <= positions) /\\ (((component + 1) <= components) /\\ ((Kernel.Determinism::comparable classOfRun sameAdapter) = false)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "probe_out_of_range_decides",
        "(((Kernel.Probe::probeCode layer layers position positions component components classOfRun sameAdapter) = 0) <-> ((((layer + 1) <= layers) /\\ (((position + 1) <= positions) /\\ ((component + 1) <= components))) -> (1 = 0)))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "probeInRange");
    assert!(
        found.contains(
            "body: (((layer <? layers) && (position <? positions)) && (component <? components))"
        ),
        "probeInRange changed:\n{found}"
    );
    let found = declaration(&kernel, "probeCode");
    assert!(
        found.contains("body: (if (probeInRange layer layers position positions component components) then (if (Kernel.Determinism::comparable classOfRun sameAdapter) then 2 else 1) else 0)"),
        "probeCode changed:\n{found}"
    );
    let found = declaration(&kernel, "probeAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 21) && ((Kernel.Wire::fieldAt value 19 1) <=? 2)) && ((Kernel.Wire::fieldAt value 20 1) <=? 1)) then (probeText (probeCode (Kernel.Wire::fieldAt value 1 4) (Kernel.Wire::fieldAt value 5 4) (Kernel.Wire::fieldAt value 9 4) (Kernel.Wire::fieldAt value 13 4) (Kernel.Wire::fieldAt value 17 1) (Kernel.Wire::fieldAt value 18 1) (Kernel.Wire::fieldAt value 19 1) (Kernel.Wire::fieldAt value 20 1))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "probeAnswer changed:\n{found}"
    );
    // Decision 32 answers approximate replay, out of range for a layer past the last, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            32,
            &[
                (
                    unhex("200000000000000002000000000000000400010200"),
                    "exact replay",
                ),
                (
                    unhex("200000000000000002000000000000000400010100"),
                    "approximate replay",
                ),
                (
                    unhex("200000000200000002000000000000000400010200"),
                    "out of range",
                ),
                (
                    unhex("2000000000000000020000000000000004000102"),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(32, "Kernel.Probe::probeAnswer");
}
/// `KN-33`: A computed prefix is shared exactly when decision 27 reuses it and it was computed on this device or in a run that is comparable
#[test]
fn kernel_kn_33() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Share.lex.tex");
    let proofs = model("app/src/Proofs/Share.lex.tex");
    assert_pinned(
        &proofs,
        "shareable_decides",
        "(((Kernel.Share::shareable asked held heldLength promptLength origin classOfRun sameAdapter) = true) <-> (((Kernel.Reuse::reusable asked held heldLength promptLength) = true) /\\ (((origin = 0) -> (1 = 0)) -> ((Kernel.Determinism::comparable classOfRun sameAdapter) = true))))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "shareable");
    assert!(
        found.contains("body: ((Kernel.Reuse::reusable asked held heldLength promptLength) && ((origin =? 0) || (Kernel.Determinism::comparable classOfRun sameAdapter)))"),
        "shareable changed:\n{found}"
    );
    let found = declaration(&kernel, "shareAnswer");
    assert!(
        found.contains("body: (if (((((Foundation.Bytes::byteLength value) =? 76) && ((Kernel.Wire::fieldAt value 73 1) <=? 1)) && ((Kernel.Wire::fieldAt value 74 1) <=? 2)) && ((Kernel.Wire::fieldAt value 75 1) <=? 1)) then (if (shareable (Kernel.Admission::labelAt value 1) (Kernel.Admission::labelAt value 33) (Kernel.Wire::fieldAt value 65 4) (Kernel.Wire::fieldAt value 69 4) (Kernel.Wire::fieldAt value 73 1) (Kernel.Wire::fieldAt value 74 1) (Kernel.Wire::fieldAt value 75 1)) then (prim utf8_encode \"reuse\" : Bytes) else (prim utf8_encode \"recompute\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "shareAnswer changed:\n{found}"
    );
    // Decision 33 recomputes a borrowed prefix from a run that is not comparable, and refuses a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            33,
            &[
                (unhex("2101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000000040000000a000000"), "reuse"),
                (unhex("2101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000000040000000a010000"), "recompute"),
                (unhex("2101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000000040000000a0000"), "invalid request"),
            ],
        );
    }
    assert_routed(33, "Kernel.Share::shareAnswer");
}
/// `KN-34`: One 32-wide block of signed 4-bit weight codes times int8 activations is confirmed exact exactly when the claimed sum, offset by 32768, equals the sum the kernel recomputes from the block, and differs otherwise
#[test]
fn kernel_kn_34() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Block.lex.tex");
    let proofs = model("app/src/Proofs/Block.lex.tex");
    assert_pinned(
        &proofs,
        "agrees_when_both_positive",
        "((8 <= code) -> ((byte < 128) -> ((Kernel.Block::signsAgree code byte) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "agrees_when_both_negative",
        "((code < 8) -> ((128 <= byte) -> ((Kernel.Block::signsAgree code byte) = true)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "differs_when_weight_positive",
        "((8 <= code) -> ((128 <= byte) -> ((Kernel.Block::signsAgree code byte) = false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "differs_when_weight_negative",
        "((code < 8) -> ((byte < 128) -> ((Kernel.Block::signsAgree code byte) = false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "weight_size_about_eight",
        "(((8 <= code) -> (((Kernel.Block::weightSize code) + 8) = code)) /\\ ((code < 8) -> (((Kernel.Block::weightSize code) + code) = 8)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "activation_size_about_zero",
        "(((byte < 128) -> ((Kernel.Block::activationSize byte) = byte)) /\\ (((128 <= byte) /\\ (byte <= 256)) -> (((Kernel.Block::activationSize byte) + byte) = 256)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "accepts_decides",
        "(((Kernel.Block::accepts claimed positive negative) = true) <-> ((claimed + negative) = (positive + 32768)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "accepts_one_claim",
        "(((Kernel.Block::accepts one positive negative) = true) -> (((Kernel.Block::accepts other positive negative) = true) -> (one = other)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "parts_cover_every_product",
        "(((Kernel.Block::positiveFrom weights activations base remaining) + (Kernel.Block::negativeFrom weights activations base remaining)) = (totalFrom weights activations base remaining))",
        "Quot.sound, propext",
    );
    let found = declaration(&kernel, "byteFromEnd");
    assert!(
        found.contains("body: (match index\n    | Nat.zero => (prim remainder word 256 0 : Nat)\n    | Nat.succ rest => (byteFromEnd (prim quotient word 256 0 : Nat) rest))"),
        "byteFromEnd changed:\n{found}"
    );
    let found = declaration(&kernel, "weightCode");
    assert!(
        found.contains("body: (if ((prim remainder index 2 0 : Nat) =? 1) then (prim remainder (byteFromEnd weights (prim quotient index 2 0 : Nat)) 16 0 : Nat) else (prim quotient (byteFromEnd weights (prim quotient index 2 0 : Nat)) 16 0 : Nat))"),
        "weightCode changed:\n{found}"
    );
    let found = declaration(&kernel, "weightSize");
    assert!(
        found.contains("body: (if (8 <=? code) then (prim subtract code 8 : Nat) else (prim subtract 8 code : Nat))"),
        "weightSize changed:\n{found}"
    );
    let found = declaration(&kernel, "activationSize");
    assert!(
        found.contains("body: (if (byte <? 128) then byte else (prim subtract 256 byte : Nat))"),
        "activationSize changed:\n{found}"
    );
    let found = declaration(&kernel, "signsAgree");
    assert!(
        found
            .contains("body: (((8 <=? code) && (byte <? 128)) || ((code <? 8) && (128 <=? byte)))"),
        "signsAgree changed:\n{found}"
    );
    let found = declaration(&kernel, "term");
    assert!(
        found.contains("body: (prim multiply (weightSize code) (activationSize byte) : Nat)"),
        "term changed:\n{found}"
    );
    let found = declaration(&kernel, "positiveFrom");
    assert!(
        found.contains("body: (match remaining\n    | Nat.zero => 0\n    | Nat.succ rest => ((if (signsAgree (weightCode weights (base + rest)) (byteFromEnd activations rest)) then (term (weightCode weights (base + rest)) (byteFromEnd activations rest)) else 0) + (positiveFrom weights activations base rest)))"),
        "positiveFrom changed:\n{found}"
    );
    let found = declaration(&kernel, "negativeFrom");
    assert!(
        found.contains("body: (match remaining\n    | Nat.zero => 0\n    | Nat.succ rest => ((if (signsAgree (weightCode weights (base + rest)) (byteFromEnd activations rest)) then 0 else (term (weightCode weights (base + rest)) (byteFromEnd activations rest))) + (negativeFrom weights activations base rest)))"),
        "negativeFrom changed:\n{found}"
    );
    let found = declaration(&kernel, "accepts");
    assert!(
        found.contains("body: ((claimed + negative) =? (positive + 32768))"),
        "accepts changed:\n{found}"
    );
    let found = declaration(&kernel, "blockCheckAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 53) then (if (accepts (Kernel.Wire::fieldAt value 49 4) ((((positiveFrom (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 25 8) 0 8) + (positiveFrom (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 17 8) 8 8)) + (positiveFrom (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 41 8) 0 8)) + (positiveFrom (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 33 8) 8 8)) ((((negativeFrom (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 25 8) 0 8) + (negativeFrom (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 17 8) 8 8)) + (negativeFrom (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 41 8) 0 8)) + (negativeFrom (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 33 8) 8 8))) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "blockCheckAnswer changed:\n{found}"
    );
    assert_routed(34, "Kernel.Block::blockCheckAnswer");
}

/// `KN-35`: A content label is admitted as an address exactly when its first seven bytes are blake3:, and any other axis, sha256: among them, is refused as not blake3
#[test]
fn kernel_kn_35() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Address.lex.tex");
    let proofs = model("app/src/Proofs/Address.lex.tex");
    assert_pinned(
        &proofs,
        "address_blake3_decides",
        "(((Kernel.Address::addressBlake3 head middle tail) = true) <-> ((head = 1651269995) /\\ ((middle = 25907) /\\ (tail = 58))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "address_sha256_refused",
        "((Kernel.Address::addressBlake3 1936220466 13622 58) = false)",
        "Quot.sound, propext",
    );
    // The numbers are "blake3:" and "sha256:" read as fields of 4, 2 and 1
    // bytes, the widths the wire format has.
    assert_eq!(u32::from_be_bytes(*b"blak"), 1651269995);
    assert_eq!(u16::from_be_bytes(*b"e3"), 25907);
    assert_eq!(u32::from_be_bytes(*b"sha2"), 1936220466);
    assert_eq!(u16::from_be_bytes(*b"56"), 13622);
    assert_eq!(b':', 58);
    let found = declaration(&kernel, "addressBlake3");
    assert!(
        found.contains("body: (((head =? 1651269995) && (middle =? 25907)) && (tail =? 58))"),
        "addressBlake3 changed:\n{found}"
    );
    let found = declaration(&kernel, "addressAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 8) then (if (addressBlake3 (Kernel.Wire::fieldAt value 1 4) (Kernel.Wire::fieldAt value 5 2) (Kernel.Wire::fieldAt value 7 1)) then (prim utf8_encode \"blake3\" : Bytes) else (prim utf8_encode \"not blake3\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "addressAnswer changed:\n{found}"
    );
    // Decision 35 answers not blake3 for another prefix, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            35,
            &[
                (unhex("23626c616b65333a"), "blake3"),
                (unhex("237368613235363a"), "not blake3"),
                (unhex("23626c616b6533"), "invalid request"),
            ],
        );
    }
    assert_routed(35, "Kernel.Address::addressAnswer");
}

/// `KN-36`: A run may start exactly when its products and every other stage are exact, which is decision 30's class everywhere, and a run with a floating-point stage is refused whatever the order of its reductions
#[test]
fn kernel_kn_36() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Integer.lex.tex");
    let proofs = model("app/src/Proofs/Integer.lex.tex");
    assert_pinned(
        &proofs,
        "integer_run_decides",
        "(((Kernel.Integer::integerRun productsExact restExact) = true) <-> ((productsExact = 1) /\\ (restExact = 1)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "integer_run_is_everywhere",
        "(((Kernel.Integer::integerRun productsExact restExact) = true) <-> ((Kernel.Determinism::classCode productsExact restExact fixedOrder) = 2))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "integer_run_refuses_fixed_order_floats",
        "((Kernel.Integer::integerRun productsExact 0) = false)",
        "Quot.sound, propext",
    );
    let found = declaration(&kernel, "integerRun");
    assert!(
        found.contains("body: ((Kernel.Determinism::classCode productsExact restExact 0) =? 2)"),
        "integerRun changed:\n{found}"
    );
    let found = declaration(&kernel, "integerAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 3) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)) && ((Kernel.Wire::fieldAt value 2 1) <=? 1)) then (if (integerRun (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1)) then (prim utf8_encode \"integer\" : Bytes) else (prim utf8_encode \"refused: not integer\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "integerAnswer changed:\n{found}"
    );
    // Decision 36 answers refused: not integer when the rest is not exact, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            36,
            &[
                (unhex("240101"), "integer"),
                (unhex("240100"), "refused: not integer"),
                (unhex("2401"), "invalid request"),
            ],
        );
    }
    assert_routed(36, "Kernel.Integer::integerAnswer");
}

/// `KN-37`: One group's sum times its IEEE half scale is confirmed exact exactly when the claimed sign and magnitude, in units of 2^-24, equal the product the kernel recomputes from the half's exponent and fraction, a zero product is never negative, and a scale that is an infinity or a NaN is answered not finite
#[test]
fn kernel_kn_37() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Scale.lex.tex");
    let proofs = model("app/src/Proofs/Scale.lex.tex");
    assert_pinned(
        &proofs,
        "two_to_doubles",
        "((Kernel.Scale::twoTo (power + 1)) = (prim multiply 2 (Kernel.Scale::twoTo power) : Nat))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "half_subnormal_is_mantissa",
        "((Kernel.Scale::halfOf 0 mantissa) = mantissa)",
        "propext",
    );
    assert_pinned(
        &proofs,
        "half_normal_is_scaled",
        "((Kernel.Scale::halfOf (below + 1) mantissa) = (prim multiply (1024 + mantissa) (Kernel.Scale::twoTo below) : Nat))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "sum_size_about_offset",
        "(((32768 <= offset) -> (((Kernel.Scale::sumSize offset) + 32768) = offset)) /\\ ((offset < 32768) -> (((Kernel.Scale::sumSize offset) + offset) = 32768)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "zero_is_not_negative",
        "(((Kernel.Scale::scaledSize offset bits) = 0) -> ((Kernel.Scale::scaledNegative offset bits) = false))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "scale_accepts_size",
        "(((Kernel.Scale::scaleAccepts offset bits sign size) = true) -> (size = (Kernel.Scale::scaledSize offset bits)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "scale_accepts_sign",
        "(((Kernel.Scale::scaleAccepts offset bits sign size) = true) -> (sign = (Kernel.Scale::signCode (Kernel.Scale::scaledNegative offset bits))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "scale_accepts_the_claim",
        "((Kernel.Scale::scaleAccepts offset bits (Kernel.Scale::signCode (Kernel.Scale::scaledNegative offset bits)) (Kernel.Scale::scaledSize offset bits)) = true)",
        "propext",
    );
    let found = declaration(&kernel, "halfExponent");
    assert!(
        found.contains("body: (prim remainder (prim quotient bits 1024 0 : Nat) 32 0 : Nat)"),
        "halfExponent changed:\n{found}"
    );
    let found = declaration(&kernel, "halfMantissa");
    assert!(
        found.contains("body: (prim remainder bits 1024 0 : Nat)"),
        "halfMantissa changed:\n{found}"
    );
    assert!(
        kernel.contains(
            "\ndefinition{name: halfNegative, parameters: [bits : Nat], result: Bool, \
             body: (32768 <=? bits)}\n"
        ),
        "halfNegative changed"
    );
    let found = declaration(&kernel, "halfUnits");
    assert!(
        found.contains("body: (halfOf (halfExponent bits) (halfMantissa bits))"),
        "halfUnits changed:\n{found}"
    );
    let found = declaration(&kernel, "sumSize");
    assert!(
        found.contains("body: (if (32768 <=? offset) then (prim subtract offset 32768 : Nat) else (prim subtract 32768 offset : Nat))"),
        "sumSize changed:\n{found}"
    );
    let found = declaration(&kernel, "scaledSize");
    assert!(
        found.contains("body: (prim multiply (sumSize offset) (halfUnits bits) : Nat)"),
        "scaledSize changed:\n{found}"
    );
    let found = declaration(&kernel, "scaledNegative");
    assert!(
        found.contains("body: ((0 <? (scaledSize offset bits)) && (if (halfNegative bits) then (32768 <=? offset) else (offset <? 32768)))"),
        "scaledNegative changed:\n{found}"
    );
    let found = declaration(&kernel, "signCode");
    assert!(
        found.contains("body: (if negative then 1 else 0)"),
        "signCode changed:\n{found}"
    );
    let found = declaration(&kernel, "scaleAccepts");
    assert!(
        found.contains("body: ((size =? (scaledSize offset bits)) && (sign =? (signCode (scaledNegative offset bits))))"),
        "scaleAccepts changed:\n{found}"
    );
    let found = declaration(&kernel, "scaleAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 16) && ((Kernel.Wire::fieldAt value 1 4) <=? 65536)) && ((Kernel.Wire::fieldAt value 7 1) <=? 1)) then (if ((halfExponent (Kernel.Wire::fieldAt value 5 2)) =? 31) then (prim utf8_encode \"not finite\" : Bytes) else (if (scaleAccepts (Kernel.Wire::fieldAt value 1 4) (Kernel.Wire::fieldAt value 5 2) (Kernel.Wire::fieldAt value 7 1) (Kernel.Wire::fieldAt value 8 8)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "scaleAnswer changed:\n{found}"
    );
    let found = declaration(&kernel, "twoTo");
    assert!(
        found.contains(
            "body: (match power | Nat.zero => 1 | Nat.succ rest => (prim multiply 2 (twoTo rest) : Nat))"
        ),
        "twoTo changed:\n{found}"
    );
    let found = declaration(&kernel, "halfOf");
    assert!(
        found.contains(
            "body: (match exponent\n    | Nat.zero => mantissa\n    \
             | Nat.succ below => (prim multiply (1024 + mantissa) (twoTo below) : Nat))"
        ),
        "halfOf changed:\n{found}"
    );
    assert_routed(37, "Kernel.Scale::scaleAnswer");
    // The claims the crates make of decision 37 cover every answer, a scale
    // that is not finite included; the app recipe replays them in the guest.
    let claims =
        std::fs::read_to_string(root().join("crates/conformance/tests/kernel/claims.json"))
            .expect("the claims the crates make of the kernel");
    let claims: serde_json::Value = serde_json::from_str(&claims).expect("claims.json is JSON");
    let answers: Vec<&str> = claims["claims"]
        .as_array()
        .expect("a list of claims")
        .iter()
        .filter(|one| one["request"].as_str().is_some_and(|r| r.starts_with("25")))
        .filter_map(|one| one["answer"].as_str())
        .collect();
    for answer in ["exact", "differs", "not finite", "invalid request"] {
        assert!(
            answers.contains(&answer),
            "no decision-37 claim is answered `{answer}`"
        );
    }
}
/// `KN-38`: A stored q4 byte, IEEE half, bfloat16 or binary32 is read into integer fields, its two nibbles or its sign, exponent and fraction, that encode back to exactly the stored bits, and fields within their widths read back unchanged
#[test]
fn kernel_kn_38() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Codec.lex.tex");
    let proofs = model("app/src/Proofs/Codec.lex.tex");
    assert_pinned(
        &proofs,
        "codec_round_trip",
        "((format <= 3) -> ((stored < (Kernel.Codec::codecWidth format)) -> ((Kernel.Codec::codecEncode format (Kernel.Codec::codecHigh format stored) (Kernel.Codec::codecMiddle format stored) (Kernel.Codec::codecLow format stored)) = stored)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "codec_fields_fit",
        "(((Kernel.Codec::codecLow format stored) < (Kernel.Codec::codecLowSize format)) /\\ ((Kernel.Codec::codecMiddle format stored) < (Kernel.Codec::codecMiddleSize format)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "codec_reads_back_q4",
        "((low < 16) -> ((middle < 16) -> (((Kernel.Codec::codecLow 0 (Kernel.Codec::codecEncode 0 high middle low)) = low) /\\ (((Kernel.Codec::codecMiddle 0 (Kernel.Codec::codecEncode 0 high middle low)) = middle) /\\ ((Kernel.Codec::codecHigh 0 (Kernel.Codec::codecEncode 0 high middle low)) = high)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "codec_reads_back_half",
        "((low < 1024) -> ((middle < 32) -> (((Kernel.Codec::codecLow 1 (Kernel.Codec::codecEncode 1 high middle low)) = low) /\\ (((Kernel.Codec::codecMiddle 1 (Kernel.Codec::codecEncode 1 high middle low)) = middle) /\\ ((Kernel.Codec::codecHigh 1 (Kernel.Codec::codecEncode 1 high middle low)) = high)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "codec_reads_back_bfloat16",
        "((low < 128) -> ((middle < 256) -> (((Kernel.Codec::codecLow 2 (Kernel.Codec::codecEncode 2 high middle low)) = low) /\\ (((Kernel.Codec::codecMiddle 2 (Kernel.Codec::codecEncode 2 high middle low)) = middle) /\\ ((Kernel.Codec::codecHigh 2 (Kernel.Codec::codecEncode 2 high middle low)) = high)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "codec_reads_back_binary32",
        "((low < 8388608) -> ((middle < 256) -> (((Kernel.Codec::codecLow 3 (Kernel.Codec::codecEncode 3 high middle low)) = low) /\\ (((Kernel.Codec::codecMiddle 3 (Kernel.Codec::codecEncode 3 high middle low)) = middle) /\\ ((Kernel.Codec::codecHigh 3 (Kernel.Codec::codecEncode 3 high middle low)) = high)))))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "codecWidth");
    assert!(
        found.contains(
            "body: (if (format =? 0) then 256 else (if (format =? 3) then 4294967296 else 65536))"
        ),
        "codecWidth changed:\n{found}"
    );
    let found = declaration(&kernel, "codecLowSize");
    assert!(
        found.contains("body: (if (format =? 0) then 16 else (if (format =? 1) then 1024 else (if (format =? 2) then 128 else 8388608)))"),
        "codecLowSize changed:\n{found}"
    );
    let found = declaration(&kernel, "codecMiddleSize");
    assert!(
        found.contains("body: (if (format =? 0) then 16 else (if (format =? 1) then 32 else 256))"),
        "codecMiddleSize changed:\n{found}"
    );
    let found = declaration(&kernel, "codecLow");
    assert!(
        found.contains("body: (prim remainder stored (codecLowSize format) 0 : Nat)"),
        "codecLow changed:\n{found}"
    );
    let found = declaration(&kernel, "codecMiddle");
    assert!(
        found.contains("body: (prim remainder (prim quotient stored (codecLowSize format) 0 : Nat) (codecMiddleSize format) 0 : Nat)"),
        "codecMiddle changed:\n{found}"
    );
    let found = declaration(&kernel, "codecHigh");
    assert!(
        found.contains("body: (prim quotient stored (prim multiply (codecLowSize format) (codecMiddleSize format) : Nat) 0 : Nat)"),
        "codecHigh changed:\n{found}"
    );
    let found = declaration(&kernel, "codecEncode");
    assert!(
        found.contains("body: (((prim multiply high (prim multiply (codecLowSize format) (codecMiddleSize format) : Nat) : Nat) + (prim multiply middle (codecLowSize format) : Nat)) + low)"),
        "codecEncode changed:\n{found}"
    );
    let found = declaration(&kernel, "codecText");
    assert!(
        found.contains("body: (if (format =? 0) then (prim append (prim append (Kernel.Wire::decimal (codecLow format stored)) (prim utf8_encode \" \" : Bytes) : Bytes) (Kernel.Wire::decimal (codecMiddle format stored)) : Bytes) else (prim append (prim append (prim append (prim append (Kernel.Wire::decimal (codecHigh format stored)) (prim utf8_encode \" \" : Bytes) : Bytes) (Kernel.Wire::decimal (codecMiddle format stored)) : Bytes) (prim utf8_encode \" \" : Bytes) : Bytes) (Kernel.Wire::decimal (codecLow format stored)) : Bytes))"),
        "codecText changed:\n{found}"
    );
    let found = declaration(&kernel, "codecAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 6) && ((Kernel.Wire::fieldAt value 1 1) <=? 3)) && ((Kernel.Wire::fieldAt value 2 4) <? (codecWidth (Kernel.Wire::fieldAt value 1 1)))) then (codecText (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 4)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "codecAnswer changed:\n{found}"
    );
    // Decision 38 refuses only as invalid request: a stored value too wide for its format, and a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            38,
            &[
                (unhex("260000000025"), "5 2"),
                (unhex("260000000100"), "invalid request"),
                (unhex("2600000025"), "invalid request"),
            ],
        );
    }
    assert_routed(38, "Kernel.Codec::codecAnswer");
}
/// `KN-39`: A step that narrows a value, exp, reciprocal square root, sine and cosine, SiLU, GELU or requantising, is accepted exactly when its input is at most 16 bits so its table is finite, its output is 1 to 32 bits, and its certified error is within its declared bound
#[test]
fn kernel_kn_39() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Narrow.lex.tex");
    let proofs = model("app/src/Proofs/Narrow.lex.tex");
    assert_pinned(
        &proofs,
        "narrow_accepted_decides",
        "(((Kernel.Narrow::narrowAccepted step inputBits outputBits bound certified) = true) <-> ((step <= 5) /\\ ((1 <= inputBits) /\\ ((inputBits <= 16) /\\ ((1 <= outputBits) /\\ ((outputBits <= 32) /\\ (certified <= bound)))))))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "narrowAccepted");
    assert!(
        found.contains("body: ((((step <=? 5) && ((1 <=? inputBits) && (inputBits <=? 16))) && ((1 <=? outputBits) && (outputBits <=? 32))) && (certified <=? bound))"),
        "narrowAccepted changed:\n{found}"
    );
    let found = declaration(&kernel, "narrowAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 12) && ((Kernel.Wire::fieldAt value 1 1) <=? 5)) then (if (narrowAccepted (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 1) (Kernel.Wire::fieldAt value 3 1) (Kernel.Wire::fieldAt value 4 4) (Kernel.Wire::fieldAt value 8 4)) then (prim utf8_encode \"accept\" : Bytes) else (prim utf8_encode \"refuse\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "narrowAnswer changed:\n{found}"
    );
    // Decision 39 answers refuse when the certified bound exceeds the bound, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            39,
            &[
                (unhex("270108100000006400000032"), "accept"),
                (unhex("2701081000000064000000c8"), "refuse"),
                (unhex("2701081000000064000032"), "invalid request"),
            ],
        );
    }
    assert_routed(39, "Kernel.Narrow::narrowAnswer");
}
/// `KN-40`: One link of a row's exact recombination is confirmed exactly when the partial before it plus its up to six terms, each a unit sum times a scale mantissa times a power of two, signed, equals the claimed partial after it; a link whose numbers could pass 64 bits is too wide, and two confirmed links compose, so a chain of them confirms the plain sum
#[test]
fn kernel_kn_40() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Rows.lex.tex");
    let proofs = model("app/src/Proofs/Rows.lex.tex");
    assert_pinned(
        &proofs,
        "rows_accepts_decides",
        "(((Kernel.Rows::rowsAccepts positive negative claimedPositive claimedNegative) = true) <-> ((positive + claimedNegative) = (negative + claimedPositive)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "rows_part_splits",
        "((sign <= 1) -> (((Kernel.Rows::rowsPart sign size 0) + (Kernel.Rows::rowsPart sign size 1)) = size))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_part_one_side",
        "(((sign = 0) -> ((Kernel.Rows::rowsPart sign size 1) = 0)) /\\ ((sign = 1) -> ((Kernel.Rows::rowsPart sign size 0) = 0)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_links_compose",
        "(((Kernel.Rows::rowsAccepts (startPositive + firstPositive) (startNegative + firstNegative) middlePositive middleNegative) = true) -> (((Kernel.Rows::rowsAccepts (middlePositive + secondPositive) (middleNegative + secondNegative) endPositive endNegative) = true) -> ((Kernel.Rows::rowsAccepts (startPositive + (firstPositive + secondPositive)) (startNegative + (firstNegative + secondNegative)) endPositive endNegative) = true)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_one_claim",
        "(((Kernel.Rows::rowsAccepts positive negative onePositive oneNegative) = true) -> (((Kernel.Rows::rowsAccepts positive negative otherPositive otherNegative) = true) -> ((onePositive + otherNegative) = (otherPositive + oneNegative))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_next_accepted",
        "((Kernel.Rows::rowsAccepts positive negative (Kernel.Rows::rowsPart (Kernel.Rows::rowsNextSign positive negative) (Kernel.Rows::rowsNextSize positive negative) 0) (Kernel.Rows::rowsPart (Kernel.Rows::rowsNextSign positive negative) (Kernel.Rows::rowsNextSize positive negative) 1)) = true)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_accepts_only_next_positive",
        "(((Kernel.Rows::rowsAccepts positive negative size 0) = true) -> ((size = (Kernel.Rows::rowsNextSize positive negative)) /\\ (((size = 0) -> (1 = 0)) -> ((Kernel.Rows::rowsNextSign positive negative) = 0))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_accepts_only_next_negative",
        "(((Kernel.Rows::rowsAccepts positive negative 0 size) = true) -> ((size = (Kernel.Rows::rowsNextSize positive negative)) /\\ (((size = 0) -> (1 = 0)) -> ((Kernel.Rows::rowsNextSign positive negative) = 1))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rows_term_fits_bounds",
        "(((Kernel.Rows::rowsTermFits sum mantissa shift) = true) -> ((shift <= 58) /\\ ((sum < 16777216) /\\ (mantissa < 16777216))))",
        "Classical.choice, Quot.sound, propext",
    );
    let found = declaration(&kernel, "rowsPow2");
    assert!(
        found.contains("body: (match power | Nat.zero => 1 | Nat.succ rest => (prim multiply 2 (rowsPow2 rest) : Nat)),\n  recursive_argument: power"),
        "rowsPow2 changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsTermSize");
    assert!(
        found.contains(
            "body: (prim multiply (prim multiply sum mantissa : Nat) (rowsPow2 shift) : Nat)"
        ),
        "rowsTermSize changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsTermFits");
    assert!(
        found.contains("body: (((sum <? 16777216) && (mantissa <? 16777216)) && ((shift <=? 58) && ((prim multiply sum mantissa : Nat) <? (rowsPow2 (prim subtract 58 shift : Nat)))))"),
        "rowsTermFits changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsPart");
    assert!(
        found.contains("body: (if (sign =? want) then size else 0)"),
        "rowsPart changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsTermPart");
    assert!(
        found.contains("body: (rowsPart sign (rowsTermSize sum mantissa shift) want)"),
        "rowsTermPart changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsAccepts");
    assert!(
        found.contains("body: ((positive + claimedNegative) =? (negative + claimedPositive))"),
        "rowsAccepts changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsNextSign");
    assert!(
        found.contains("body: (if (negative <=? positive) then 0 else 1)"),
        "rowsNextSign changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsNextSize");
    assert!(
        found.contains("body: (if (negative <=? positive) then (prim subtract positive negative : Nat) else (prim subtract negative positive : Nat))"),
        "rowsNextSize changed:\n{found}"
    );
    let found = declaration(&kernel, "rowsAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 79) && (((((((((Kernel.Wire::fieldAt value 1 1) <=? 1) && ((Kernel.Wire::fieldAt value 10 1) <=? 1)) && ((Kernel.Wire::fieldAt value 19 1) <=? 1)) && ((Kernel.Wire::fieldAt value 29 1) <=? 1)) && ((Kernel.Wire::fieldAt value 39 1) <=? 1)) && ((Kernel.Wire::fieldAt value 49 1) <=? 1)) && ((Kernel.Wire::fieldAt value 59 1) <=? 1)) && ((Kernel.Wire::fieldAt value 69 1) <=? 1))) then (if (((((((((Kernel.Wire::fieldAt value 2 8) <? 4611686018427387904) && ((Kernel.Wire::fieldAt value 11 8) <? 4611686018427387904)) && (rowsTermFits (Kernel.Wire::fieldAt value 20 4) (Kernel.Wire::fieldAt value 24 4) (Kernel.Wire::fieldAt value 28 1))) && (rowsTermFits (Kernel.Wire::fieldAt value 30 4) (Kernel.Wire::fieldAt value 34 4) (Kernel.Wire::fieldAt value 38 1))) && (rowsTermFits (Kernel.Wire::fieldAt value 40 4) (Kernel.Wire::fieldAt value 44 4) (Kernel.Wire::fieldAt value 48 1))) && (rowsTermFits (Kernel.Wire::fieldAt value 50 4) (Kernel.Wire::fieldAt value 54 4) (Kernel.Wire::fieldAt value 58 1))) && (rowsTermFits (Kernel.Wire::fieldAt value 60 4) (Kernel.Wire::fieldAt value 64 4) (Kernel.Wire::fieldAt value 68 1))) && (rowsTermFits (Kernel.Wire::fieldAt value 70 4) (Kernel.Wire::fieldAt value 74 4) (Kernel.Wire::fieldAt value 78 1))) then (if (rowsAccepts (((((((rowsPart (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) 0) + (rowsTermPart (Kernel.Wire::fieldAt value 19 1) (Kernel.Wire::fieldAt value 20 4) (Kernel.Wire::fieldAt value 24 4) (Kernel.Wire::fieldAt value 28 1) 0)) + (rowsTermPart (Kernel.Wire::fieldAt value 29 1) (Kernel.Wire::fieldAt value 30 4) (Kernel.Wire::fieldAt value 34 4) (Kernel.Wire::fieldAt value 38 1) 0)) + (rowsTermPart (Kernel.Wire::fieldAt value 39 1) (Kernel.Wire::fieldAt value 40 4) (Kernel.Wire::fieldAt value 44 4) (Kernel.Wire::fieldAt value 48 1) 0)) + (rowsTermPart (Kernel.Wire::fieldAt value 49 1) (Kernel.Wire::fieldAt value 50 4) (Kernel.Wire::fieldAt value 54 4) (Kernel.Wire::fieldAt value 58 1) 0)) + (rowsTermPart (Kernel.Wire::fieldAt value 59 1) (Kernel.Wire::fieldAt value 60 4) (Kernel.Wire::fieldAt value 64 4) (Kernel.Wire::fieldAt value 68 1) 0)) + (rowsTermPart (Kernel.Wire::fieldAt value 69 1) (Kernel.Wire::fieldAt value 70 4) (Kernel.Wire::fieldAt value 74 4) (Kernel.Wire::fieldAt value 78 1) 0)) (((((((rowsPart (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) 1) + (rowsTermPart (Kernel.Wire::fieldAt value 19 1) (Kernel.Wire::fieldAt value 20 4) (Kernel.Wire::fieldAt value 24 4) (Kernel.Wire::fieldAt value 28 1) 1)) + (rowsTermPart (Kernel.Wire::fieldAt value 29 1) (Kernel.Wire::fieldAt value 30 4) (Kernel.Wire::fieldAt value 34 4) (Kernel.Wire::fieldAt value 38 1) 1)) + (rowsTermPart (Kernel.Wire::fieldAt value 39 1) (Kernel.Wire::fieldAt value 40 4) (Kernel.Wire::fieldAt value 44 4) (Kernel.Wire::fieldAt value 48 1) 1)) + (rowsTermPart (Kernel.Wire::fieldAt value 49 1) (Kernel.Wire::fieldAt value 50 4) (Kernel.Wire::fieldAt value 54 4) (Kernel.Wire::fieldAt value 58 1) 1)) + (rowsTermPart (Kernel.Wire::fieldAt value 59 1) (Kernel.Wire::fieldAt value 60 4) (Kernel.Wire::fieldAt value 64 4) (Kernel.Wire::fieldAt value 68 1) 1)) + (rowsTermPart (Kernel.Wire::fieldAt value 69 1) (Kernel.Wire::fieldAt value 70 4) (Kernel.Wire::fieldAt value 74 4) (Kernel.Wire::fieldAt value 78 1) 1)) (rowsPart (Kernel.Wire::fieldAt value 10 1) (Kernel.Wire::fieldAt value 11 8) 0) (rowsPart (Kernel.Wire::fieldAt value 10 1) (Kernel.Wire::fieldAt value 11 8) 1)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"too wide\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "rowsAnswer changed:\n{found}"
    );
    // Decision 40 answers differs for a wrong sum, too wide for a value past its range, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            40,
            &[
                (unhex("28000000000000000005000000000000000011000000000200000003010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "exact"),
                (unhex("28000000000000000005000000000000000012000000000200000003010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "differs"),
                (unhex("28004000000000000000000000000000000011000000000200000003010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "too wide"),
                (unhex("280000000000000000050000000000000000110000000002000000030100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "invalid request"),
            ],
        );
    }
    assert_routed(40, "Kernel.Rows::rowsAnswer");
}
/// `KN-41`: Up to five inputs of one group are confirmed put on sixteen-bit codes exactly when the group's power of two is the largest that keeps its largest input within 32767, and each code is its input at that power rounded to the nearest, ties to even, so within half a step of it; an input far below the step is code zero
#[test]
fn kernel_kn_41() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Requant.lex.tex");
    let proofs = model("app/src/Proofs/Requant.lex.tex");
    assert_pinned(
        &proofs,
        "rq_down_witness",
        "(((Kernel.Requant::rqDownOk value step half odd product code) = true) -> ((product <= value) /\\ ((value < (product + step)) /\\ (code <= 32767))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rq_down_half_step",
        "(((Kernel.Requant::rqDownOk value step half odd product code) = true) -> (((code = ((half + half) + odd)) -> ((value + value) <= ((product + product) + step))) /\\ ((code = (((half + half) + odd) + 1)) -> ((step + (product + product)) <= (value + value)))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rq_down_tie_even",
        "((odd <= 1) -> (((Kernel.Requant::rqDownOk value step half odd product code) = true) -> (((value + value) = (step + (product + product))) -> (code = ((half + odd) + (half + odd))))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rq_down_one_code",
        "(((Kernel.Requant::rqDownOk value step half odd product one) = true) -> (((Kernel.Requant::rqDownOk value step half odd product other) = true) -> (one = other)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rq_up_exact",
        "(((Kernel.Requant::rqUpOk product code) = true) <-> ((code = product) /\\ (code <= 32767)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "rq_largest_down_decides",
        "(((Kernel.Requant::rqLargestDown largest step) = true) <-> ((largest <= (Kernel.Requant::rqLimitTimes step)) /\\ ((Kernel.Requant::rqLimitTimes step) < (largest + largest))))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "rq_largest_up_decides",
        "(((Kernel.Requant::rqLargestUp scaled) = true) <-> ((scaled <= 32767) /\\ (32767 < (scaled + scaled))))",
        "propext",
    );
    let found = declaration(&kernel, "rqPow2");
    assert!(
        found.contains("body: (match power | Nat.zero => 1 | Nat.succ rest => (prim multiply 2 (rqPow2 rest) : Nat)),\n  recursive_argument: power"),
        "rqPow2 changed:\n{found}"
    );
    let found = declaration(&kernel, "rqDownOk");
    assert!(
        found.contains("body: ((((product <=? value) && (value <? (product + step))) && (code <=? 32767)) && ((((step + (product + product)) <? (value + value)) && (code =? (((half + half) + odd) + 1))) || ((((value + value) <? (step + (product + product))) && (code =? ((half + half) + odd))) || (((value + value) =? (step + (product + product))) && (code =? (((half + half) + odd) + odd))))))"),
        "rqDownOk changed:\n{found}"
    );
    let found = declaration(&kernel, "rqUpOk");
    assert!(
        found.contains("body: ((code <=? 32767) && (code =? product))"),
        "rqUpOk changed:\n{found}"
    );
    let found = declaration(&kernel, "rqElementOk");
    assert!(
        found.contains("body: (if ((value <? 16777216) && (half <=? 16383)) then (if (direction =? 1) then (if (shift <=? 15) then (rqUpOk (prim multiply value (rqPow2 shift) : Nat) code) else false) else (if (shift <=? 39) then (rqDownOk value (rqPow2 shift) half odd (prim multiply ((half + half) + odd) (rqPow2 shift) : Nat) code) else (((code =? 0) && (half =? 0)) && (odd =? 0)))) else false)"),
        "rqElementOk changed:\n{found}"
    );
    let found = declaration(&kernel, "rqLimitTimes");
    assert!(
        found.contains("body: (prim multiply 32767 step : Nat)"),
        "rqLimitTimes changed:\n{found}"
    );
    let found = declaration(&kernel, "rqLargestDown");
    assert!(
        found.contains("body: ((largest <=? (rqLimitTimes step)) && ((rqLimitTimes step) <? (largest + largest)))"),
        "rqLargestDown changed:\n{found}"
    );
    let found = declaration(&kernel, "rqLargestUp");
    assert!(
        found.contains("body: ((scaled <=? 32767) && (32767 <? (scaled + scaled)))"),
        "rqLargestUp changed:\n{found}"
    );
    let found = declaration(&kernel, "rqLargest");
    assert!(
        found.contains("body: (if (largest =? 0) then true else (if (largest <? 16777216) then (if (direction =? 1) then (if (shift <=? 15) then (rqLargestUp (prim multiply largest (rqPow2 shift) : Nat)) else false) else (if (shift <=? 39) then (rqLargestDown largest (rqPow2 shift)) else false)) else false))"),
        "rqLargest changed:\n{found}"
    );
    let found = declaration(&kernel, "rqDownCode");
    assert!(
        found.contains("body: ((prim quotient value step 0 : Nat) + (if ((step <? (prim multiply 2 (prim remainder value step 0 : Nat) : Nat)) || (((prim multiply 2 (prim remainder value step 0 : Nat) : Nat) =? step) && ((prim remainder (prim quotient value step 0 : Nat) 2 0 : Nat) =? 1))) then 1 else 0))"),
        "rqDownCode changed:\n{found}"
    );
    let found = declaration(&kernel, "rqAnswer");
    assert!(
        found.contains("body: (if ((((((((Foundation.Bytes::byteLength value) =? 62) && ((Kernel.Wire::fieldAt value 5 1) <=? 1)) && (((Kernel.Wire::fieldAt value 11 1) <=? 1) && ((Kernel.Wire::fieldAt value 15 1) <=? 1))) && (((Kernel.Wire::fieldAt value 22 1) <=? 1) && ((Kernel.Wire::fieldAt value 26 1) <=? 1))) && (((Kernel.Wire::fieldAt value 33 1) <=? 1) && ((Kernel.Wire::fieldAt value 37 1) <=? 1))) && (((Kernel.Wire::fieldAt value 44 1) <=? 1) && ((Kernel.Wire::fieldAt value 48 1) <=? 1))) && (((Kernel.Wire::fieldAt value 55 1) <=? 1) && ((Kernel.Wire::fieldAt value 59 1) <=? 1))) then (if (rqLargest (Kernel.Wire::fieldAt value 1 4) (Kernel.Wire::fieldAt value 5 1) (Kernel.Wire::fieldAt value 6 1)) then (if (((((rqElementOk (Kernel.Wire::fieldAt value 7 4) (Kernel.Wire::fieldAt value 11 1) (Kernel.Wire::fieldAt value 12 1) (Kernel.Wire::fieldAt value 13 2) (Kernel.Wire::fieldAt value 15 1) (Kernel.Wire::fieldAt value 16 2)) && (rqElementOk (Kernel.Wire::fieldAt value 18 4) (Kernel.Wire::fieldAt value 22 1) (Kernel.Wire::fieldAt value 23 1) (Kernel.Wire::fieldAt value 24 2) (Kernel.Wire::fieldAt value 26 1) (Kernel.Wire::fieldAt value 27 2))) && (rqElementOk (Kernel.Wire::fieldAt value 29 4) (Kernel.Wire::fieldAt value 33 1) (Kernel.Wire::fieldAt value 34 1) (Kernel.Wire::fieldAt value 35 2) (Kernel.Wire::fieldAt value 37 1) (Kernel.Wire::fieldAt value 38 2))) && (rqElementOk (Kernel.Wire::fieldAt value 40 4) (Kernel.Wire::fieldAt value 44 1) (Kernel.Wire::fieldAt value 45 1) (Kernel.Wire::fieldAt value 46 2) (Kernel.Wire::fieldAt value 48 1) (Kernel.Wire::fieldAt value 49 2))) && (rqElementOk (Kernel.Wire::fieldAt value 51 4) (Kernel.Wire::fieldAt value 55 1) (Kernel.Wire::fieldAt value 56 1) (Kernel.Wire::fieldAt value 57 2) (Kernel.Wire::fieldAt value 59 1) (Kernel.Wire::fieldAt value 60 2))) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"not the largest power\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "rqAnswer changed:\n{found}"
    );
    // Decision 41 answers differs for a wrong code, not the largest power for a wrong scale, and invalid request for a short one.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            41,
            &[
                (unhex("2900000000000000000005010200000000140000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "exact"),
                (unhex("2900000000000000000005010200000000150000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "differs"),
                (unhex("2900000001010000000005010200000000140000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "not the largest power"),
                (unhex("29000000000000000000050102000000001400000000000000000000000000000000000000000000000000000000000000000000000000000000000000"), "invalid request"),
            ],
        );
    }
    assert_routed(41, "Kernel.Requant::rqAnswer");
}

/// `KN-42`: A claimed binary32 is confirmed exact exactly when it is the bits the kernel rounds magnitude times 2 to the exponent to: the nearest, ties to even, from the leading bit, 24 bits or fewer below the normal range, infinity past the largest finite value, and zero for a zero magnitude
#[test]
fn kernel_kn_42() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Round.lex.tex");
    let proofs = model("app/src/Proofs/Round.lex.tex");
    assert_pinned(
        &proofs,
        "round_zero_is_zero",
        "((Kernel.Round::roundResult sign 0 offset) = 0)",
        "propext",
    );
    assert_pinned(
        &proofs,
        "round_shift_past_word",
        "((64 <= count) -> ((Kernel.Round::roundShift value count) = 0))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "round_up_at_most_one",
        "((Kernel.Round::roundUp value offset) <= 1)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "round_no_guard_no_change",
        "(((Kernel.Round::roundGuard value offset) = 0) -> ((Kernel.Round::roundUp value offset) = 0))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "round_accepts_decides",
        "(((Kernel.Round::roundAccepts claimed sign value offset) = true) <-> (claimed = (Kernel.Round::roundResult sign value offset)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "round_past_largest_is_infinite",
        "((0 < value) -> ((32895 < (Kernel.Round::roundTop value offset)) -> ((Kernel.Round::roundResult sign value offset) = ((prim multiply sign 2147483648 : Nat) + 2139095040))))",
        "Quot.sound, propext",
    );
    let found = declaration(&kernel, "roundBits");
    assert!(
        found.contains("body: (match fuel\n    | Nat.zero => 0\n    | Nat.succ rest => (if (value =? 0) then 0 else (1 + (roundBits (prim quotient value 2 0 : Nat) rest))))"),
        "roundBits changed:\n{found}"
    );
    let found = declaration(&kernel, "roundShift");
    assert!(
        found.contains("body: (if (64 <=? count) then 0 else (prim quotient value (Kernel.Scale::twoTo count) 0 : Nat))"),
        "roundShift changed:\n{found}"
    );
    let found = declaration(&kernel, "roundBit");
    assert!(
        found.contains("body: (if (64 <=? count) then 0 else (prim remainder (prim quotient value (Kernel.Scale::twoTo count) 0 : Nat) 2 0 : Nat))"),
        "roundBit changed:\n{found}"
    );
    let found = declaration(&kernel, "roundAny");
    assert!(
        found.contains("body: (if (64 <=? count) then (0 <? value) else (0 <? (prim remainder value (Kernel.Scale::twoTo count) 0 : Nat)))"),
        "roundAny changed:\n{found}"
    );
    let found = declaration(&kernel, "roundTop");
    assert!(
        found.contains("body: (prim subtract (offset + (roundBits value 64)) 1 : Nat)"),
        "roundTop changed:\n{found}"
    );
    let found = declaration(&kernel, "roundUnit");
    assert!(
        found.contains("body: (if (32642 <=? (roundTop value offset)) then (prim subtract (roundTop value offset) 23 : Nat) else 32619)"),
        "roundUnit changed:\n{found}"
    );
    let found = declaration(&kernel, "roundDropped");
    assert!(
        found.contains("body: (prim subtract (roundUnit value offset) offset : Nat)"),
        "roundDropped changed:\n{found}"
    );
    let found = declaration(&kernel, "roundKept");
    assert!(
        found.contains("body: (if (offset <=? (roundUnit value offset)) then (roundShift value (roundDropped value offset)) else (prim multiply value (Kernel.Scale::twoTo (prim subtract offset (roundUnit value offset) : Nat)) : Nat))"),
        "roundKept changed:\n{found}"
    );
    let found = declaration(&kernel, "roundGuard");
    assert!(
        found.contains("body: (if ((offset <=? (roundUnit value offset)) && (1 <=? (roundDropped value offset))) then (roundBit value (prim subtract (roundDropped value offset) 1 : Nat)) else 0)"),
        "roundGuard changed:\n{found}"
    );
    let found = declaration(&kernel, "roundSticky");
    assert!(
        found.contains("body: (((offset <=? (roundUnit value offset)) && (2 <=? (roundDropped value offset))) && (roundAny value (prim subtract (roundDropped value offset) 1 : Nat)))"),
        "roundSticky changed:\n{found}"
    );
    let found = declaration(&kernel, "roundUp");
    assert!(
        found.contains("body: (if (((roundGuard value offset) =? 1) && ((roundSticky value offset) || ((prim remainder (roundKept value offset) 2 0 : Nat) =? 1))) then 1 else 0)"),
        "roundUp changed:\n{found}"
    );
    let found = declaration(&kernel, "roundField");
    assert!(
        found.contains("body: ((prim multiply (prim subtract (roundUnit value offset) 32619 : Nat) 8388608 : Nat) + ((roundKept value offset) + (roundUp value offset)))"),
        "roundField changed:\n{found}"
    );
    let found = declaration(&kernel, "roundResult");
    assert!(
        found.contains("body: (if (value =? 0) then 0 else (if ((32895 <? (roundTop value offset)) || (2139095040 <=? (roundField value offset))) then ((prim multiply sign 2147483648 : Nat) + 2139095040) else ((prim multiply sign 2147483648 : Nat) + (roundField value offset))))"),
        "roundResult changed:\n{found}"
    );
    let found = declaration(&kernel, "roundAccepts");
    assert!(
        found.contains("body: (claimed =? (roundResult sign value offset))"),
        "roundAccepts changed:\n{found}"
    );
    let found = declaration(&kernel, "roundAnswer");
    assert!(
        found.contains("body: (if (((Foundation.Bytes::byteLength value) =? 16) && ((Kernel.Wire::fieldAt value 1 1) <=? 1)) then (if (roundAccepts (Kernel.Wire::fieldAt value 12 4) (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 2)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "roundAnswer changed:\n{found}"
    );
    assert_routed(42, "Kernel.Round::roundAnswer");
    // The claims the crates make of decision 42 cover every answer; the app
    // recipe replays them in the guest.
    let claims =
        std::fs::read_to_string(root().join("crates/conformance/tests/kernel/claims.json"))
            .expect("the claims the crates make of the kernel");
    let claims: serde_json::Value = serde_json::from_str(&claims).expect("claims.json is JSON");
    let answers: Vec<&str> = claims["claims"]
        .as_array()
        .expect("a list of claims")
        .iter()
        .filter(|one| one["request"].as_str().is_some_and(|r| r.starts_with("2a")))
        .filter_map(|one| one["answer"].as_str())
        .collect();
    for answer in ["exact", "differs", "invalid request"] {
        assert!(
            answers.contains(&answer),
            "no decision-42 claim is answered `{answer}`"
        );
    }
}

/// `KN-43`: A claimed integer square root of a value below 2^62 is confirmed exactly when its square is at most the value and the next square is above it, so a norm divides by the value's one floor root
#[test]
fn kernel_kn_43() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Root.lex.tex");
    let proofs = model("app/src/Proofs/Root.lex.tex");
    assert_pinned(
        &proofs,
        "root_ok_decides",
        "(((Kernel.Root::rootOk value root) = true) <-> (((Kernel.Root::rootSquare root) <= value) /\\ (value < (Kernel.Root::rootSquare (root + 1)))))",
        "propext",
    );
    let found = declaration(&kernel, "rootSquare");
    assert!(
        found.contains("body: (prim multiply root root : Nat)"),
        "rootSquare changed:\n{found}"
    );
    let found = declaration(&kernel, "rootOk");
    assert!(
        found.contains(
            "body: (((rootSquare root) <=? value) && (value <? (rootSquare (root + 1))))"
        ),
        "rootOk changed:\n{found}"
    );
    let found = declaration(&kernel, "rootAnswer");
    assert!(
        found.contains("body: (if ((((Foundation.Bytes::byteLength value) =? 13) && ((Kernel.Wire::fieldAt value 1 8) <? 4611686018427387904)) && ((Kernel.Wire::fieldAt value 9 4) <? 2147483648)) then (if (rootOk (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 4)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "rootAnswer changed:\n{found}"
    );
    // Decision 43 answers differs for a wrong root, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            43,
            &[
                (unhex("2b000000000000000a00000003"), "exact"),
                (unhex("2b000000000000000a00000004"), "differs"),
                (unhex("2b000000000000000a000003"), "invalid request"),
            ],
        );
    }
    assert_routed(43, "Kernel.Root::rootAnswer");
}

/// `KN-44`: A claimed quotient of a value below 2^62 by a divisor below 2^31, rounded half to even to a code of at most 32767, is confirmed exactly when it passes KN-41's rounding rule with the divisor as the step, so a norm and a softmax divide by one rule
#[test]
fn kernel_kn_44() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Divide.lex.tex");
    let proofs = model("app/src/Proofs/Divide.lex.tex");
    assert_pinned(
        &proofs,
        "divide_is_requant_rule",
        "((Kernel.Divide::divideOk value divisor half odd code) = (Kernel.Requant::rqDownOk value divisor half odd (Kernel.Divide::divideProduct half odd divisor) code))",
        "propext",
    );
    let found = declaration(&kernel, "divideProduct");
    assert!(
        found.contains("body: (prim multiply ((half + half) + odd) divisor : Nat)"),
        "divideProduct changed:\n{found}"
    );
    let found = declaration(&kernel, "divideOk");
    assert!(
        found.contains("body: (Kernel.Requant::rqDownOk value divisor half odd (divideProduct half odd divisor) code)"),
        "divideOk changed:\n{found}"
    );
    let found = declaration(&kernel, "divideAnswer");
    assert!(
        found.contains("body: (if ((((((Foundation.Bytes::byteLength value) =? 20) && ((Kernel.Wire::fieldAt value 1 8) <? 4611686018427387904)) && ((0 <? (Kernel.Wire::fieldAt value 9 4)) && ((Kernel.Wire::fieldAt value 9 4) <? 2147483648))) && ((Kernel.Wire::fieldAt value 13 4) <? 1073741824)) && ((Kernel.Wire::fieldAt value 17 1) <=? 1)) then (if (divideOk (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 4) (Kernel.Wire::fieldAt value 13 4) (Kernel.Wire::fieldAt value 17 1) (Kernel.Wire::fieldAt value 18 2)) then (prim utf8_encode \"exact\" : Bytes) else (prim utf8_encode \"differs\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "divideAnswer changed:\n{found}"
    );
    // Decision 44 answers differs for a quotient not rounded half to even, and invalid request for a short request.
    {
        // No crate builds this request yet, so the bytes follow the model's wire format.
        assert_claimed(
            44,
            &[
                (unhex("2c00000000000000070000000200000001010004"), "exact"),
                (unhex("2c00000000000000070000000200000001010003"), "differs"),
                (
                    unhex("2c000000000000000700000002000000010100"),
                    "invalid request",
                ),
            ],
        );
    }
    assert_routed(44, "Kernel.Divide::divideAnswer");
}

/// `KN-45`: The engine session is a pure step over open, admit a page, prompt chunk, generate and close, ready to be the guest's root in the session world: a page not named by BLAKE3 fails the session, the session is ready only when the page admitted is the manifest's last, prompt chunks compose, a generate request with a floating-point stage fails the session, the position never passes the capacity, and an ended session stays ended
#[test]
fn kernel_kn_45() {
    assert_proofs_are_gated();
    let proofs = model("app/src/Proofs/Session.lex.tex");
    assert_pinned(
        &proofs,
        "session_ends_stay_ended",
        "((4 <= phase) -> ((phase <= 5) -> ((Engine.Session::sessionPhase phase admitted pages position capacity kind a b c d) = phase)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "session_refuses_other_hashes",
        "((phase <= 3) -> (((Kernel.Address::addressBlake3 a b c) = false) -> ((Engine.Session::sessionPhase phase admitted pages position capacity 2 a b c d) = 5)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "session_ready_only_with_every_page",
        "((((admitted + 1) = pages) -> (1 = 0)) -> (((Engine.Session::sessionPhase 1 admitted pages position capacity 2 a b c d) =? 2) = false))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "session_refuses_floats",
        "((phase <= 3) -> ((((a = 1) /\\ (b = 1)) -> (1 = 0)) -> ((Engine.Session::sessionPhase phase admitted pages position capacity 4 a b c d) = 5)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "session_prompt_chunks_compose",
        "(((Engine.Session::live phase) = true) -> (((position + (early + later)) <= capacity) -> ((Engine.Session::sessionPosition phase (Engine.Session::sessionPosition phase position capacity 3 early b) capacity 3 later b) = (Engine.Session::sessionPosition phase position capacity 3 (early + later) b))))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "session_position_within_capacity",
        "((position <= capacity) -> ((Engine.Session::sessionPosition phase position capacity kind a b) <= (Engine.Session::sessionCapacity phase capacity kind a b)))",
        "Quot.sound, propext",
    );
    // The session is the guest's root once the session world is in the
    // pinned SDK image; until then nothing routes to it, and it is proved
    // here so the root is ready.
    let session = model("app/src/Engine/Session.lex.tex");
    for name in [
        "sessionPhase",
        "sessionAdmitted",
        "sessionPages",
        "sessionCapacity",
        "sessionPosition",
    ] {
        declaration(&session, name);
    }
    assert!(
        session.contains("\\importmodule{Kernel.Address}")
            && session.contains("\\importmodule{Kernel.Integer}"),
        "the session no longer builds on decisions 35 and 36"
    );
}

/// `KN-46`: A model's chat template is one the engine defines exactly when the BLAKE3 of its text is Qwen2.5-1.5B-Instruct's, Qwen3-0.6B's or Llama 3.2 1B Instruct's, as their exports carry them at the hub's pinned revisions; decision 45 answers qwen2.5, qwen3 or llama3.2 for those three and refuses any other template by name, before anything is rendered with it
#[test]
fn kernel_kn_46() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Template.lex.tex");
    let proofs = model("app/src/Proofs/Template.lex.tex");
    for (theorem, code, words) in [
        (
            "template_qwen25_decides",
            1,
            [
                "8279920892041357147",
                "1773128108353419947",
                "11653179235714034099",
                "12784695345153184910",
            ],
        ),
        (
            "template_qwen3_decides",
            2,
            [
                "5043451083563981477",
                "8438563855072365186",
                "9276823912147443683",
                "9407911508815075403",
            ],
        ),
        (
            "template_llama32_decides",
            3,
            [
                "15723989467465989188",
                "4968833967170122917",
                "9281937141963328179",
                "6782964158007752886",
            ],
        ),
    ] {
        let [a, b, c, d] = words;
        assert_pinned(
            &proofs,
            theorem,
            &format!("(((Kernel.Template::templateCode a b c d) = {code}) <-> ((((a = {a}) /\\ (b = {b})) /\\ (c = {c})) /\\ (d = {d})))"),
            "Classical.choice, Quot.sound, propext",
        );
    }
    assert_pinned(
        &proofs,
        "template_code_bounded",
        "((Kernel.Template::templateCode a b c d) <= 3)",
        "Quot.sound, propext",
    );
    // The words are the BLAKE3 of each template's text, as the exports carry it.
    let templates = root().join("crates/conformance/tests/gpu/templates");
    for (file, words) in [
        (
            "Qwen2.5-1.5B-Instruct.json",
            "8279920892041357147 1773128108353419947 11653179235714034099 12784695345153184910",
        ),
        (
            "Qwen3-0.6B-ONNX.json",
            "5043451083563981477 8438563855072365186 9276823912147443683 9407911508815075403",
        ),
        (
            "Llama-3.2-1B-Instruct-ONNX.json",
            "15723989467465989188 4968833967170122917 9281937141963328179 6782964158007752886",
        ),
    ] {
        let recorded: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(templates.join(file)).expect("the recorded template"),
        )
        .expect("JSON");
        let text = recorded["template"].as_str().expect("a template");
        let mut arriving = hologram_ai_download::Arriving::new();
        arriving.take(text.as_bytes());
        let hash = arriving.label().0;
        let derived: Vec<String> = hash
            .chunks(8)
            .map(|w| u64::from_be_bytes(w.try_into().expect("eight bytes")).to_string())
            .collect();
        assert_eq!(
            derived.join(" "),
            words,
            "{file}: the kernel names another template"
        );
        assert!(
            kernel.contains(&derived.join(" ")),
            "{file}: Kernel.Template does not list it"
        );
    }
    let found = declaration(&kernel, "templateAnswer");
    assert!(
        found.contains("(if ((Foundation.Bytes::byteLength value) =? 33) then (templateText (templateCode (Kernel.Wire::fieldAt value 1 8) (Kernel.Wire::fieldAt value 9 8) (Kernel.Wire::fieldAt value 17 8) (Kernel.Wire::fieldAt value 25 8))) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "templateAnswer changed:\n{found}"
    );
    assert_routed(45, "Kernel.Template::templateAnswer");
    // The claims cover every answer; the app recipe replays them in the guest.
    use hologram_ai_kernel_client::template;
    let claims: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("crates/conformance/tests/kernel/claims.json"))
            .expect("claims"),
    )
    .expect("claims.json is JSON");
    let mine: Vec<(String, String)> = claims["claims"]
        .as_array()
        .expect("claims")
        .iter()
        .filter(|c| c["request"].as_str().is_some_and(|r| r.starts_with("2d")))
        .map(|c| {
            (
                c["request"].as_str().expect("r").to_string(),
                c["answer"].as_str().expect("a").to_string(),
            )
        })
        .collect();
    for answer in [
        "qwen2.5",
        "qwen3",
        "llama3.2",
        "refused: unknown template",
        "invalid request",
    ] {
        assert!(
            mine.iter().any(|(_, a)| a == answer),
            "no decision-45 claim is answered `{answer}`"
        );
    }
    for (request, answer) in &mine {
        let bytes: Vec<u8> = (0..request.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&request[i..i + 2], 16).expect("hex"))
            .collect();
        if let Ok(address) = <[u8; 32]>::try_from(&bytes[1..]) {
            assert_eq!(
                template::request(&address).to_vec(),
                bytes,
                "a decision-45 claim is not the client's request"
            );
        }
        assert_ne!(
            template::answer(answer.as_bytes()),
            template::Verdict::Unrecognised,
            "the client does not read `{answer}`"
        );
    }
}

/// `KN-47`: The kernel renders a conversation of up to eight system, user and assistant turns as Qwen2.5's, Qwen3's or Llama 3.2's chat template renders it, by its own definition (decision 46): Qwen2.5's default system line, Qwen3's empty thinking block when thinking is off and its trailing assistant turn, Llama 3.2's date line and Python's whitespace trim; a Qwen3 assistant turn carrying </think> reasoning, a role past assistant and a request of the wrong length are refused by name
#[test]
fn kernel_kn_47() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Chat.lex.tex");
    let proofs = model("app/src/Proofs/Chat.lex.tex");
    for (theorem, statement, axioms) in [
        ("chat_space_tab_to_return", "((9 <= byte) -> ((byte <= 13) -> ((Kernel.Chat::chatSpaceByte byte) = true)))", "Classical.choice, Quot.sound, propext"),
        ("chat_space_separators", "((28 <= byte) -> ((byte <= 32) -> ((Kernel.Chat::chatSpaceByte byte) = true)))", "Classical.choice, Quot.sound, propext"),
        ("chat_space_not_below", "((byte < 9) -> ((Kernel.Chat::chatSpaceByte byte) = false))", "Quot.sound, propext"),
        ("chat_space_not_between", "((13 < byte) -> ((byte < 28) -> ((Kernel.Chat::chatSpaceByte byte) = false)))", "Quot.sound, propext"),
        ("chat_space_not_above", "((32 < byte) -> ((Kernel.Chat::chatSpaceByte byte) = false))", "Quot.sound, propext"),
        ("chat_role_known_decides", "(((Kernel.Chat::chatRoleKnown role) = true) <-> (role <= 2))", "propext"),
        ("chat_family_known_decides", "(((Kernel.Chat::chatFamilyKnown family) = true) <-> ((1 <= family) /\\ (family <= 3)))", "propext"),
    ] {
        assert_pinned(&proofs, theorem, statement, axioms);
    }
    for name in [
        "chatTrim",
        "chatQwenTurns",
        "chatQwenHead",
        "chatQwenPrompt",
        "chatLlama",
        "chatRender",
        "chatAnswer",
    ] {
        declaration(&kernel, name);
    }
    assert_routed(46, "Kernel.Chat::chatAnswer");
    let claims: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("crates/conformance/tests/kernel/claims.json"))
            .expect("claims"),
    )
    .expect("claims.json is JSON");
    let answers: Vec<&str> = claims["claims"]
        .as_array()
        .expect("claims")
        .iter()
        .filter(|c| c["request"].as_str().is_some_and(|r| r.starts_with("2e")))
        .filter_map(|c| c["answer"].as_str())
        .collect();
    for wanted in [
        "refused: reasoning in a turn",
        "refused: unknown role",
        "invalid request",
    ] {
        assert!(
            answers.contains(&wanted),
            "no decision-46 claim is answered `{wanted}`"
        );
    }
    // Every rendered claim is hologram-ai-chat's render of its request: the
    // kernel's definition and the crate's agree on each one the guest replays.
    use hologram_ai_chat::template::{self, Family, Message, Options, Role};
    for one in claims["claims"].as_array().expect("claims") {
        let Some(request) = one["request"].as_str().filter(|r| r.starts_with("2e")) else {
            continue;
        };
        let answer = one["answer"].as_str().expect("an answer");
        if answer.starts_with("refused") || answer == "invalid request" {
            continue;
        }
        let bytes: Vec<u8> = (0..request.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&request[i..i + 2], 16).expect("hex"))
            .collect();
        let family = match bytes[1] {
            1 => Family::Qwen25,
            2 => Family::Qwen3,
            _ => Family::Llama32,
        };
        let turns: Vec<(Role, String)> = (0..usize::from(bytes[14]))
            .map(|i| {
                let role = [Role::System, Role::User, Role::Assistant][usize::from(bytes[15 + i])];
                let start = 23 + 14 * i;
                let text = bytes[start..start + usize::from(bytes[19 + i])].to_vec();
                (role, String::from_utf8(text).expect("text"))
            })
            .collect();
        let messages: Vec<Message<'_>> = turns
            .iter()
            .map(|(role, content)| Message {
                role: *role,
                content,
            })
            .collect();
        let options = Options {
            add_generation_prompt: bytes[2] & 1 == 1,
            thinking_off: bytes[2] & 2 == 2,
            date: bytes[3..14].try_into().expect("a date"),
        };
        assert_eq!(
            template::render(family, &messages, options).as_deref(),
            Ok(answer),
            "claim {request} is not hologram-ai-chat's render"
        );
        assert_eq!(
            template::request(family, &messages, options).as_deref(),
            Some(&bytes[..]),
            "claim {request} is not the crate's request"
        );
    }
    for family in [
        "<|im_start|>system\nYou are Qwen",
        "<|im_start|>user\nhi",
        "<|begin_of_text|>",
    ] {
        assert!(
            answers.iter().any(|a| a.starts_with(family)),
            "no decision-46 claim renders as `{family}`"
        );
    }
}

/// `KN-50`: A claimed binary32 or whole number for a config number written in decimal is confirmed exactly when it is the one the text names: the nearest binary32, ties to even, subnormals included, reached by long division on 64-bit words and Kernel.Round, or the whole number exactly; text that is not a decimal, past the range or not whole is refused by name
#[test]
fn kernel_kn_50() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Config.lex.tex");
    let proofs = model("app/src/Proofs/Config.lex.tex");
    for (name, statement, axioms) in [
        (
            "decimal_left_below_divisor",
            "((carried < divisor) -> ((bit <= 1) -> ((Kernel.Config::decimalLeft carried bit divisor) < divisor)))",
            "Quot.sound, propext",
        ),
        (
            "decimal_step_takes_divisor",
            "((divisor <= ((carried + carried) + bit)) -> (((Kernel.Config::decimalLeft carried bit divisor) + divisor) = ((carried + carried) + bit)))",
            "Quot.sound, propext",
        ),
        (
            "decimal_step_keeps_below",
            "((((carried + carried) + bit) < divisor) -> ((Kernel.Config::decimalLeft carried bit divisor) = ((carried + carried) + bit)))",
            "Quot.sound, propext",
        ),
        (
            "decimal_out_is_one_exactly_when_taken",
            "(((Kernel.Config::decimalOut carried bit divisor) = 1) <-> (divisor <= ((carried + carried) + bit)))",
            "Classical.choice, Quot.sound, propext",
        ),
        (
            "decimal_unit_stage_passes_bit",
            "((bit <= 1) -> ((Kernel.Config::decimalOut 0 bit 1) = bit))",
            "Quot.sound, propext",
        ),
        (
            "decimal_unit_stage_keeps_nothing",
            "((bit <= 1) -> ((Kernel.Config::decimalLeft 0 bit 1) = 0))",
            "Quot.sound, propext",
        ),
        (
            "decimal_accepts_decides",
            "(((Kernel.Config::decimalAccepts claimed mode mantissa up down) = true) <-> (claimed = (Kernel.Config::decimalResult mode mantissa up down)))",
            "propext",
        ),
    ] {
        assert_pinned(&proofs, name, statement, axioms);
    }
    for (name, body) in [
        (
            "decimalLeft",
            "body: (if (divisor <=? ((carried + carried) + bit)) then (prim subtract ((carried + carried) + bit) divisor : Nat) else ((carried + carried) + bit))",
        ),
        (
            "decimalStage",
            "body: (if ((prim subtract down skipped : Nat) <=? 18) then (decimalPow (prim subtract down skipped : Nat)) else (decimalPow 18))",
        ),
        (
            "decimalPacked",
            "body: (decimalRun (decimalFed mantissa up) (decimalStage down 0) (decimalStage down 18) (decimalStage down 36) (decimalStage down 54) 0 0 0 0 0 0 420)",
        ),
        (
            "decimalBinary",
            "body: (Kernel.Round::roundResult 0 (prim quotient (decimalPacked mantissa up down) 512 0 : Nat) (prim subtract 32831 (prim remainder (decimalPacked mantissa up down) 512 0 : Nat) : Nat))",
        ),
        (
            "decimalInRange",
            "body: (if (decimalFits mantissa up) then ((mode =? 1) || (down <=? 72)) else false)",
        ),
        (
            "decimalFits",
            "body: (if (mantissa =? 0) then true else (if (up <=? 18) then (mantissa <=? (prim quotient 9223372036854775807 (decimalPow up) 0 : Nat)) else false))",
        ),
    ] {
        let found = declaration(&kernel, name);
        assert!(found.contains(body), "{name} changed:\n{found}");
    }
    let run = declaration(&kernel, "decimalRun");
    assert!(
        run.contains(
            "(if (4503599627370496 <=? quotient) then (decimalPack fed r1 r2 r3 r4 quotient step)"
        ),
        "decimalRun stops at other than 53 quotient bits:\n{run}"
    );
    assert_routed(49, "Kernel.Config::decimalAnswer");
}

/// `KN-51`: A cache entry, an f16, is confirmed to agree with a reference rotary value in units of 2^-30 exactly when the f16 read exactly as an integer times a power of two lies within the tolerance of it, compared in integers; an infinity or a NaN is answered not finite
#[test]
fn kernel_kn_51() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Rotary.lex.tex");
    let proofs = model("app/src/Proofs/Rotary.lex.tex");
    for (name, statement, axioms) in [
        (
            "rotary_agrees_decides",
            "(((Kernel.Rotary::rotaryAgrees bits sign magnitude tolerance) = true) <-> ((Kernel.Rotary::rotaryDistance bits sign magnitude) <= tolerance))",
            "propext",
        ),
        (
            "rotary_same_sign_is_difference",
            "(((Kernel.Rotary::rotarySame bits sign) = 1) -> ((Kernel.Rotary::rotaryDistance bits sign magnitude) = ((prim subtract (Kernel.Rotary::rotaryCache bits) magnitude : Nat) + (prim subtract magnitude (Kernel.Rotary::rotaryCache bits) : Nat))))",
            "Quot.sound, propext",
        ),
        (
            "rotary_opposite_signs_add",
            "(((Kernel.Rotary::rotarySame bits sign) = 0) -> ((Kernel.Rotary::rotaryDistance bits sign magnitude) = ((Kernel.Rotary::rotaryCache bits) + magnitude)))",
            "Quot.sound, propext",
        ),
        (
            "rotary_exact_entry_agrees",
            "(((Kernel.Rotary::rotarySame bits sign) = 1) -> ((Kernel.Rotary::rotaryAgrees bits sign (Kernel.Rotary::rotaryCache bits) tolerance) = true))",
            "Quot.sound, propext",
        ),
    ] {
        assert_pinned(&proofs, name, statement, axioms);
    }
    for (name, body) in [
        (
            "rotaryCache",
            "body: (prim multiply (Kernel.Scale::halfUnits bits) 64 : Nat)",
        ),
        (
            "rotarySame",
            "body: (if (Kernel.Scale::halfNegative bits) then (if (sign =? 1) then 1 else 0) else (if (sign =? 0) then 1 else 0))",
        ),
        (
            "rotaryAgrees",
            "body: ((rotaryDistance bits sign magnitude) <=? tolerance)",
        ),
        (
            "rotaryAnswer",
            "((Kernel.Wire::fieldAt value 4 4) <=? 1073741824)) && ((Kernel.Wire::fieldAt value 8 4) <=? 1073741824))",
        ),
    ] {
        let found = declaration(&kernel, name);
        assert!(found.contains(body), "{name} changed:\n{found}");
    }
    assert_routed(50, "Kernel.Rotary::rotaryAnswer");
}
/// `KN-52`: An operator of an encoder graph is supported exactly when its name is on the encoder style's closed list, read from Xenova's q4f16 and q4 exports of all-MiniLM-L6-v2 and distilbert-base-uncased-finetuned-sst-2-english
#[test]
fn kernel_kn_52() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Opnames.lex.tex");
    // Which list a style asks is the body of `opnameListed`, pinned by
    // kernel_kn_18; it is not a theorem here, as KN-18's lists are not.
    // The 25 names every node of the four graphs uses, read on 2026-11-19 from
    // the node sections of onnx/model_q4f16.onnx and onnx/model_q4.onnx at the
    // revisions the hub records (751bff3 and 0b6928e).
    let names = [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
    ];
    let listed = declaration(&kernel, "opnameEncoder");
    for name in names {
        let mut bytes = [0u8; 32];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        let words: Vec<String> = bytes
            .chunks(8)
            .map(|w| u64::from_be_bytes(w.try_into().expect("eight bytes")).to_string())
            .collect();
        let wanted = format!("(opnameIs wa wb wc wd {})", words.join(" "));
        assert!(
            listed.contains(&wanted),
            "{name} is not in opnameEncoder:\n{listed}"
        );
    }
    assert_eq!(
        listed.matches("(opnameIs ").count(),
        names.len(),
        "opnameEncoder lists a name the test does not"
    );
    // The compiled kernel supports GELU's Erf in an encoder graph, refuses it in
    // a decoder's, refuses a decoder's fused attention in an encoder's, and
    // answers a fourth style as no question.
    {
        use hologram_ai_kernel_client::opname::{self, Style};
        let name = |n: &str| opname::name(n).expect("a short name");
        let mut fourth = opname::request(Style::Encoder, &name("Erf"));
        fourth[1] = 3;
        assert_claimed(
            18,
            &[
                (
                    opname::request(Style::Encoder, &name("Erf")).to_vec(),
                    "supported",
                ),
                (
                    opname::request(Style::Encoder, &name("Gemm")).to_vec(),
                    "supported",
                ),
                (
                    opname::request(Style::Decomposed, &name("Erf")).to_vec(),
                    "unsupported operator",
                ),
                (
                    opname::request(Style::Encoder, &name("GroupQueryAttention")).to_vec(),
                    "unsupported operator",
                ),
                (fourth.to_vec(), "invalid request"),
            ],
        );
    }
    assert_routed(18, "Kernel.Opnames::opnameAnswer");
}
/// `KN-53`: Decision 24 names the files of feature extraction and text classification as it names those of text generation, and refuses any other task
#[test]
fn kernel_kn_53() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Source.lex.tex");
    let found = declaration(&kernel, "sourceCode");
    assert!(
        found.contains("body: (if (task <=? 2) then"),
        "sourceCode must name the files of tasks 0, 1 and 2:\n{found}"
    );
    // The compiled kernel names the same graph for an embedding and a
    // classifier as for generation, and refuses a fourth task.
    {
        use hologram_ai_kernel_client::source::{self, Precision, Task};
        let mut fourth = source::request_for(Task::TextGeneration, Precision::Q4F16, 0, 0);
        fourth[1] = 3;
        assert_claimed(
            24,
            &[
                (
                    source::request_for(Task::FeatureExtraction, Precision::Q4F16, 0, 0).to_vec(),
                    "onnx/model_q4f16.onnx",
                ),
                (
                    source::request_for(Task::TextClassification, Precision::Q4, 0, 0).to_vec(),
                    "onnx/model_q4.onnx",
                ),
                (
                    source::request_for(Task::FeatureExtraction, Precision::Q4F16, 0, 1).to_vec(),
                    "undeclared file",
                ),
                (fourth.to_vec(), "unsupported task"),
            ],
        );
    }
    assert_routed(24, "Kernel.Source::sourceAnswer");
}

/// `KN-55`: The greedy token is defined in LexLean, not Rust or JavaScript: it is the index of the largest logit, each logit a binary32 bit pattern ordered by sign and magnitude so that both zeros tie, and Lean proves for every list of logits that its key is the largest key, that no logit's key is above it, and that it is an index of the list
#[test]
fn kernel_kn_55() {
    assert_proofs_are_gated();
    let stage = model("app/src/Engine/Stage/Greedy.lex.tex");
    let proofs = model("app/src/Proofs/StageGreedy.lex.tex");
    assert_pinned(
        &proofs,
        "greedy_key_is_max",
        "((Engine.Stage.Greedy::keyAt logits (Engine.Stage.Greedy::bestIn logits last)) = (Engine.Stage.Greedy::maxKey logits last))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "exceed_zero_decides",
        "(((Engine.Stage.Greedy::exceed logits last bound) = 0) <-> ((Engine.Stage.Greedy::maxKey logits last) <= bound))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "none_above_greedy",
        "((Engine.Stage.Greedy::exceed logits last (Engine.Stage.Greedy::keyAt logits (Engine.Stage.Greedy::bestIn logits last))) = 0)",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "greedy_in_range",
        "((Engine.Stage.Greedy::bestIn logits last) <= last)",
        "Quot.sound, propext",
    );
    // A negative logit's key is 2^32 less its bits, so -0 (2^31) and +0 meet at
    // 2^31 and a larger magnitude sorts lower, as the binary32 order has it.
    let found = declaration(&stage, "key");
    assert!(
        found.contains("body: (pick (below bits 2147483648) (2147483648 + bits) (prim subtract 4294967296 bits : Nat))"),
        "key changed:\n{found}"
    );
    // A strict comparison keeps the earlier index on a tie.
    let found = declaration(&stage, "bestIn");
    assert!(
        found.contains("(pick (below (keyAt logits (bestIn logits rest)) (keyAt logits (rest + 1))) (rest + 1) (bestIn logits rest))"),
        "bestIn changed:\n{found}"
    );
    // exceed sums each key's excess over the bound, so it is zero exactly when
    // no key is above the bound.
    let found = declaration(&stage, "exceed");
    assert!(
        found.contains(
            "((prim subtract (keyAt logits (rest + 1)) bound : Nat) + (exceed logits rest bound))"
        ),
        "exceed changed:\n{found}"
    );
    let found = declaration(&stage, "greedy");
    assert!(
        found.contains("body: (bestIn logits (prim subtract (size logits) 1 : Nat))"),
        "greedy changed:\n{found}"
    );
}

/// `KN-70`: A transformers.js pipeline call is answered the engine device it runs on, auto, webgpu or cpu, exactly when its task, device, dtype and generation options are ones the engine honours: webgpu and gpu name WebGPU, wasm and cpu name the integer CPU engine, q4f16, q4 and auto are the dtypes, decoding is greedy; any other task, device or dtype, sampling, beam search or a repetition penalty is refused by name
#[test]
fn kernel_kn_70() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Call.lex.tex");
    let proofs = model("app/src/Proofs/Call.lex.tex");
    let refused = |name: &str, guard: &str| {
        (
            name.to_string(),
            format!("({guard} -> (10 <= (Kernel.Call::callCode task device dtype sample beams penalty)))"),
        )
    };
    let mut pinned = vec![
        (
            "call_supported_runs".to_string(),
            "((task <= 2) -> ((device <= 4) -> ((dtype <= 2) -> ((beams <= 1) -> ((Kernel.Call::callCode task device dtype 0 beams 1) = (Kernel.Call::callDevice device))))))".to_string(),
        ),
        (
            "call_task_refused".to_string(),
            "((3 <= task) -> ((Kernel.Call::callCode task device dtype sample beams penalty) = 10))".to_string(),
        ),
        refused("call_device_refused", "(5 <= device)"),
        refused("call_dtype_refused", "(3 <= dtype)"),
        refused("call_beams_refused", "(2 <= beams)"),
        refused("call_penalty_refused", "((penalty =? 1) = false)"),
        refused("call_sampling_refused", "(1 <= sample)"),
    ];
    for (name, statement) in pinned.drain(..) {
        assert_pinned(&proofs, &name, &statement, "Quot.sound, propext");
    }
    for (name, statement) in [
        (
            "call_device_webgpu_decides",
            "(((Kernel.Call::callDevice device) = 1) <-> (((device =? 1) || (device =? 2)) = true))",
        ),
        (
            "call_device_cpu_decides",
            "(((Kernel.Call::callDevice device) = 2) <-> (((device =? 3) || (device =? 4)) = true))",
        ),
    ] {
        assert_pinned(&proofs, name, statement, "Classical.choice, Quot.sound, propext");
    }
    for (name, body) in [
        (
            "callDevice",
            "body: (if (device =? 0) then 0 else (if (device <=? 2) then 1 else (if (device <=? 4) then 2 else 3)))",
        ),
        (
            "callCode",
            "(if (1 <=? sample) then 13 else (if (2 <=? beams) then 14 else (if (penalty =? 1) then (callDevice device) else 15)))",
        ),
        ("callAnswer", "((Foundation.Bytes::byteLength value) =? 7)"),
    ] {
        let found = declaration(&kernel, name);
        assert!(found.contains(body), "{name} changed:
{found}");
    }
    assert_routed(70, "Kernel.Call::callAnswer");
}

/// `KN-49`: Decision 48 answers a GPU kernel's WGSL text in pieces of 256 bytes, printed in the guest from the kernel's byte code, and only for kernels whose text Lean checks equal to the model printer's: q4 group sums at zero point 8 for rows of 128, 256, 1024, 1536, 2048, 3072, 8192 or 8960 codes, and the exponent table kernel; any other kernel, width, zero point or request length answers invalid request
#[test]
fn kernel_kn_49() {
    assert_proofs_are_gated();
    let proofs = model("app/src/Proofs/GpuCode.lex.tex");
    assert_pinned(
        &proofs,
        "exp2_code_prints_tree",
        "((Engine.Code::renderKernelC 3 7 (Engine.Code::exp2NegCode) 0x) = (Engine.Gpu::renderKernel 3 7 (Engine.Gpu::exp2Neg 1)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k128",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000000400000080) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 4 128 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k256",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000000800000100) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 8 256 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k1024",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000002000000400) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 32 1024 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k1536",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000003000000600) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 48 1536 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k2048",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000004000000800) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 64 2048 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k3072",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000006000000c00) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 96 3072 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k8192",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000010000002000) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 256 8192 0 false)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "q4_code_prints_tree_k8960",
        "((Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode) 0x0000011800002300) = (Engine.Gpu::renderKernel 4 7 (Engine.Gpu::q4GroupSums 280 8960 0 false)))",
        "Quot.sound, propext",
    );
    let render = model("app/src/Kernel/Render.lex.tex");
    let found = declaration(&render, "printedWidth");
    assert!(
        found.contains("body: ((width =? 128) || ((width =? 256) || ((width =? 1024) || ((width =? 1536) || ((width =? 2048) || ((width =? 3072) || ((width =? 8192) || (width =? 8960))))))))"),
        "the widths decision 48 prints must be the widths the proofs check:
{found}"
    );
    let found = declaration(&render, "renderText");
    assert!(
        found.contains("(Engine.Code::renderKernelC 4 7 (Engine.Code::q4GroupSumsCode)")
            && found.contains("(Engine.Code::renderKernelC 3 7 (Engine.Code::exp2NegCode) 0x)"),
        "decision 48 must print the codes the proofs check:
{found}"
    );
    assert_routed(48, "Kernel.Render::renderAnswer");
    let q4 = |piece: u8, packed: u8, units: u8, codes: u16| {
        let mut request = vec![48, 1, 0, piece, 0, 0, 0, units, 0, 0];
        request.extend(codes.to_be_bytes());
        request.extend([0, 0, 0, 0, packed]);
        request
    };
    assert_claimed(
        48,
        &[
            (q4(5, 0, 4, 128), ""),
            (q4(0, 0, 2, 64), "invalid request"),
            (q4(0, 1, 4, 128), "invalid request"),
            (q4(0, 0, 5, 128), "invalid request"),
            (q4(0, 0, 4, 128)[..16].to_vec(), "invalid request"),
            (
                vec![48, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "invalid request",
            ),
        ],
    );
}

/// `KN-60`: A file streams into the guest in chunks of any size, from 0 to 6444 bytes (a request of 8 KiB), and the guest hashes it itself: decision 54 folds one chunk into a BLAKE3 state carried in the request as lowercase hex and answers the next state, or the file's BLAKE3 on the last chunk; the state after two chunks is the state after them joined, for every state and every two lists of bytes, and the guest reads its chunk as that fold; the digest is BLAKE3's on the official vectors; a request whose length is not its count's, whose flags pass 3 or whose carried state is not lowercase hex answers invalid request
#[test]
fn kernel_kn_60() {
    assert_proofs_are_gated();
    let proofs = model("app/src/Proofs/Stream.lex.tex");
    assert_pinned(
        &proofs,
        "stream_cut",
        "((Engine.Stream::feed (Engine.Stream::feed state early) later) = (Engine.Stream::feed state (Engine.Stream::join early later)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "stream_reads_its_chunk",
        "((Kernel.Stream::feedFrom fuel state value index) = (Engine.Stream::feed state (Kernel.Stream::chunkOf fuel value index)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "blake3_official_0",
        "((Engine.Blake3::hexDigest (Engine.Stream::digestOf (Engine.Stream::feed (Engine.Stream::initialState) (Engine.Stream::pattern 0 0)))) = 0x61663133343962396635663961316136613034303464656133366463633934393962636232356339616463313132623763633961393363616534316633323632)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "blake3_official_1",
        "((Engine.Blake3::hexDigest (Engine.Stream::digestOf (Engine.Stream::feed (Engine.Stream::initialState) (Engine.Stream::pattern 0 1)))) = 0x32643361646564666631316236316631346338383665333561666130333637333664636438376137346432376235633135313032323564306635393265323133)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "blake3_official_64",
        "((Engine.Blake3::hexDigest (Engine.Stream::digestOf (Engine.Stream::feed (Engine.Stream::initialState) (Engine.Stream::pattern 0 64)))) = 0x34656564373134316561346135636434623738383630366264323366343665323132616639636163656261636463376431663463366463376632353131623938)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "blake3_official_65",
        "((Engine.Blake3::hexDigest (Engine.Stream::digestOf (Engine.Stream::feed (Engine.Stream::initialState) (Engine.Stream::pattern 0 65)))) = 0x64653165356661306265373064663664326265386666666430653939636561613865623665386339336136336632643864316333306563623662323633646565)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "blake3_official_1025",
        "((Engine.Blake3::hexDigest (Engine.Stream::digestOf (Engine.Stream::feed (Engine.Stream::initialState) (Engine.Stream::pattern 0 1025)))) = 0x64303032373861653437656232376233346661656366363762346665323633663832643534313239313663316666643937633863623766623831346238343434)",
        "Quot.sound, propext",
    );
    let stream = model("app/src/Kernel/Stream.lex.tex");
    let found = declaration(&stream, "streamAnswer");
    assert!(
        found.contains("(Foundation.Bytes::byteLength value) =? (1748 + (prim multiply 1 (Kernel.Wire::fieldAt value 2 2) : Nat))")
            && found.contains("(hexValid 1744 value 0)")
            && found.contains("(feedFrom (Kernel.Wire::fieldAt value 2 2)"),
        "decision 54 must check its length and state, and fold exactly its chunk:\n{found}"
    );
    assert_routed(54, "Kernel.Stream::streamAnswer");
    let hex = |bytes: &[u8]| -> Vec<u8> { bytes.to_vec() };
    let request = |flags: u8, chunk: &[u8]| {
        let mut r = vec![54, flags];
        r.extend((chunk.len() as u16).to_be_bytes());
        r.extend(std::iter::repeat_n(b'0', 1744));
        r.extend(hex(chunk));
        r
    };
    let pattern = |n: usize| (0..n).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
    assert_claimed(
        54,
        &[
            (
                request(2, &[]),
                "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            ),
            (
                request(2, &pattern(1)),
                "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
            ),
            (
                request(2, &pattern(1025)),
                "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
            ),
            (request(4, &[]), "invalid request"),
        ],
    );
}

/// `KN-56`: The base-two exponent of softmax is defined in LexLean, not Rust: exp2Neg reads one of 256 entries, each certified as the integer nearest 2^(30 - j/256) by (2T - 1)^256 < 2^(7936 - j) < (2T + 1)^256, and shifts it right by d/256 rounding half to even; Lean checks the certificate for all 256 entries and, for every d below 8704, that the output is within half a step of the entry over 2^(d/256), a tie even
#[test]
fn kernel_kn_56() {
    assert_proofs_are_gated();
    let stage = model("app/src/Engine/Stage/Exp2.lex.tex");
    let low = model("app/src/Proofs/StageExp2.lex.tex");
    let mid = model("app/src/Proofs/StageExp2Mid.lex.tex");
    let high = model("app/src/Proofs/StageExp2High.lex.tex");
    assert_pinned(&low, "table_certified", "((certifiedBelow 256) = true)", "");
    // Every shift a d below 8704 can have, each over all 256 fractions.
    for shift in 0..34 {
        let (proofs, call) = match shift {
            0..=10 => (&low, "roundedBelow"),
            11..=21 => (&mid, "Proofs.StageExp2::roundedBelow"),
            _ => (&high, "Proofs.StageExp2::roundedBelow"),
        };
        assert_pinned(
            proofs,
            &format!("rounded_at_shift_{shift}"),
            &format!("(({call} {shift} 256) = true)"),
            "",
        );
    }
    // The certificate and the rounding check are what the theorems mean.
    for (name, body) in [
        (
            "certified",
            "body: (((powBy (prim subtract (prim multiply 2 (Engine.Stage.Exp2::entry fraction) : Nat) 1 : Nat) 256 9) <? (powBy 2 (prim subtract 7936 fraction : Nat) 13)) && ((powBy 2 (prim subtract 7936 fraction : Nat) 13) <? (powBy ((prim multiply 2 (Engine.Stage.Exp2::entry fraction) : Nat) + 1) 256 9)))",
        ),
        (
            "rounded",
            "body: (((distanceTwice value shift out) <=? (Engine.Stage.Exp2::pow2 shift)) && ((not ((distanceTwice value shift out) =? (Engine.Stage.Exp2::pow2 shift))) || ((prim remainder out 2 0 : Nat) =? 0)))",
        ),
    ] {
        let found = declaration(&low, name);
        assert!(found.contains(body), "{name} changed:\n{found}");
    }
    // The stage reads entry d mod 256 and shifts it by d / 256.
    for (name, body) in [
        (
            "exp2Neg",
            "body: (shifted (entry (prim remainder distance 256 0 : Nat)) (prim quotient distance 256 0 : Nat))",
        ),
        (
            "roundUp",
            "body: (Engine.Stage.Greedy::pick (Engine.Stage.Greedy::below half rest) 1 (Engine.Stage.Greedy::pick (Engine.Stage.Greedy::below rest half) 0 (prim remainder quotient 2 0 : Nat)))",
        ),
    ] {
        let found = declaration(&stage, name);
        assert!(found.contains(body), "{name} changed:\n{found}");
    }
    assert_eq!(
        stage.matches("\n  name: entryBlock").count(),
        16,
        "the table is sixteen blocks of sixteen entries"
    );
}

/// `KN-72`: Every product and limb sum an integer stage reasons about is a LexLean function with proved laws, since lexlean 0.3.0 can prove nothing about prim multiply of two variables: mulB, shift-and-add over a list of steps, splits into its low and high halves, is additive and scales in its multiplicand, and the 64-bit product of two words built from four 16-bit partial products equals mulB over 32 steps as two 32-bit limbs, each below 2^32, each partial at most 65535^2; add-with-carry over little-endian 32-bit limbs gives the sum of the two values and the carry in, the carry out at the weight of the limbs, and limbs below 2^32; that mulB equals the native multiply in exported code is trusted, not proved
#[test]
fn kernel_kn_72() {
    assert_proofs_are_gated();
    let proofs = model("app/src/Proofs/Word.lex.tex");
    assert_pinned(
        &proofs,
        "mul_splits",
        "((Engine.Word::mulB (Engine.Word::joined low high) a b) = ((Engine.Word::mulB low a b) + (Engine.Word::mulB high (Engine.Word::shiftUp low a) (Engine.Word::dropBits low b))))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "mul_left_additive",
        "((Engine.Word::mulB steps (x + y) b) = ((Engine.Word::mulB steps x b) + (Engine.Word::mulB steps y b)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "product_from_halves",
        "((Engine.Word::mulB (Engine.Word::thirtyTwo) a b) = (((Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim remainder b 65536 0 : Nat)) + (prim multiply 65536 (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient a 65536 0 : Nat) (prim remainder b 65536 0 : Nat)) : Nat)) + (prim multiply 65536 ((Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim quotient b 65536 0 : Nat)) + (prim multiply 65536 (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient a 65536 0 : Nat) (prim quotient b 65536 0 : Nat)) : Nat)) : Nat)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "product_as_limbs",
        "(((Engine.Word::productLow (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim remainder b 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim quotient b 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient a 65536 0 : Nat) (prim remainder b 65536 0 : Nat))) + (prim multiply 4294967296 (Engine.Word::productHigh (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim remainder b 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder a 65536 0 : Nat) (prim quotient b 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient a 65536 0 : Nat) (prim remainder b 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient a 65536 0 : Nat) (prim quotient b 65536 0 : Nat))) : Nat)) = (Engine.Word::mulB (Engine.Word::thirtyTwo) a b))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "low_limb_is_word",
        "((Engine.Word::productLow ll lh hl) < 4294967296)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "high_limb_is_word",
        "((Engine.Word::productHigh (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder (prim remainder a 4294967296 0 : Nat) 65536 0 : Nat) (prim remainder (prim remainder b 4294967296 0 : Nat) 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim remainder (prim remainder a 4294967296 0 : Nat) 65536 0 : Nat) (prim quotient (prim remainder b 4294967296 0 : Nat) 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient (prim remainder a 4294967296 0 : Nat) 65536 0 : Nat) (prim remainder (prim remainder b 4294967296 0 : Nat) 65536 0 : Nat)) (Engine.Word::mulB (Engine.Word::sixteen) (prim quotient (prim remainder a 4294967296 0 : Nat) 65536 0 : Nat) (prim quotient (prim remainder b 4294967296 0 : Nat) 65536 0 : Nat))) < 4294967296)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "low_half_product_bound",
        "((Engine.Word::mulB (Engine.Word::sixteen) (prim remainder x 65536 0 : Nat) y) <= 4294836225)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "limbs_add_with_carry",
        "((Engine.Word::sumOf left right carry) = (((Engine.Word::limbsValue left) + (Engine.Word::limbsValue right)) + carry))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "carry_weight_is_final_carry",
        "((Engine.Word::carryWeight left right carry) = (Engine.Word::shifted left (Engine.Word::finalCarry left right carry)))",
        "propext",
    );
    assert_pinned(
        &proofs,
        "added_limbs_are_words",
        "((Engine.Word::allWords (Engine.Word::addLimbs left right carry)) = true)",
        "Quot.sound, propext",
    );
    let word = model("app/src/Engine/Word.lex.tex");
    assert!(
        word.contains(
            "definition{name: thirtyTwo, parameters: [], result: (List Nat), body: (joined (sixteen) (sixteen))}"
        ),
        "the 32-step multiply must be two 16-step halves"
    );
    assert!(
        !word.contains("prim multiply a b") && !word.contains("prim multiply x y"),
        "a product of two variables must be mulB, not prim multiply"
    );
}

/// `KN-57`: The q4 group sum is defined in LexLean, not Rust: output k of Engine.Stage.Q4Sum::run is the 32-term sum of each signed 4-bit weight code less 8 times the signed 16-bit activation code, written as a two's complement word; Lean proves, for every input, that the word plus the negative terms equals the positive terms plus 2^32 when the sum is negative, that the word is below 2^31 exactly when the sum is not negative, and that each side is at most 2^24, over the shipped stage, and that its one product, the native multiply, equals Engine.Word::mulB for every multiplier up to 8
#[test]
fn kernel_kn_57() {
    assert_proofs_are_gated();
    let stage = model("app/src/Engine/Stage/Q4Sum.lex.tex");
    let proofs = model("app/src/Proofs/StageQ4Sum.lex.tex");
    assert_pinned(
        &proofs,
        "group_sum_value",
        "(((Engine.Stage.Q4Sum::groupSum acts weights units k) + (Engine.Stage.Q4Sum::negSum acts weights units k 32)) = ((Engine.Stage.Q4Sum::posSum acts weights units k 32) + (Engine.Stage.Greedy::pick (Engine.Stage.Greedy::below (Engine.Stage.Q4Sum::posSum acts weights units k 32) (Engine.Stage.Q4Sum::negSum acts weights units k 32)) 4294967296 0)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "group_sum_sign",
        "(((Engine.Stage.Q4Sum::groupSum acts weights units k) < 2147483648) <-> ((Engine.Stage.Q4Sum::negSum acts weights units k 32) <= (Engine.Stage.Q4Sum::posSum acts weights units k 32)))",
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "pos_sum_32",
        "((Engine.Stage.Q4Sum::posSum acts weights units k 32) <= 16777216)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "neg_sum_32",
        "((Engine.Stage.Q4Sum::negSum acts weights units k 32) <= 16777216)",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "times_bound",
        "((a <= 65536) -> ((w <= 8) -> ((Engine.Stage.Q4Sum::times a w) <= 524288)))",
        "Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "times_exact",
        "((w <= 8) -> ((Engine.Stage.Q4Sum::times a w) = (Engine.Word::mulB (four) a w)))",
        "Quot.sound, propext",
    );
    // The proofs are about the shipped stage, not a copy of it.
    assert!(
        proofs.contains("\\importmodule{Engine.Stage.Q4Sum}")
            && !proofs.contains("\n  name: groupSum,"),
        "the proofs must read the stage, not restate it"
    );
    assert!(
        declaration(&stage, "times").contains("body: (prim multiply a w : Nat)"),
        "the stage multiplies natively"
    );
}

/// `KN-80`: Decision 80 answers sealed exactly when a 32-byte BLAKE3 address, read as eight 4-byte words, is the engine address the model seals, not the sealed engine for any other address, one flipped byte among them, and invalid request for any length but 33
#[test]
fn kernel_kn_80() {
    assert_proofs_are_gated();
    let kernel = model("app/src/Kernel/Seal.lex.tex");
    let proofs = model("app/src/Proofs/Seal.lex.tex");
    let words = (0..8)
        .map(|i| format!("(Kernel.Seal::sealedWord {i})"))
        .collect::<Vec<_>>();
    assert_pinned(
        &proofs,
        "seal_decides",
        &format!(
            r"(((Kernel.Seal::engineSealed w0 w1 w2 w3 w4 w5 w6 w7) = true) <-> ((w0 = {}) /\ ((w1 = {}) /\ ((w2 = {}) /\ ((w3 = {}) /\ ((w4 = {}) /\ ((w5 = {}) /\ ((w6 = {}) /\ (w7 = {})))))))))",
            words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]
        ),
        "Classical.choice, Quot.sound, propext",
    );
    assert_pinned(
        &proofs,
        "seal_refuses_another_first_word",
        "(((w0 =? (Kernel.Seal::sealedWord 0)) = false) -> ((Kernel.Seal::engineSealed w0 w1 w2 w3 w4 w5 w6 w7) = false))",
        "Quot.sound, propext",
    );
    let found = declaration(&kernel, "engineSealed");
    assert!(
        found.contains("body: ((((((((w0 =? (sealedWord 0)) && (w1 =? (sealedWord 1))) && (w2 =? (sealedWord 2))) && (w3 =? (sealedWord 3))) && (w4 =? (sealedWord 4))) && (w5 =? (sealedWord 5))) && (w6 =? (sealedWord 6))) && (w7 =? (sealedWord 7)))"),
        "engineSealed changed:\n{found}"
    );
    let found = declaration(&kernel, "sealAnswer");
    assert!(
        found.contains("body: (if ((Foundation.Bytes::byteLength value) =? 33) then (if (engineSealed (Kernel.Wire::fieldAt value 1 4) (Kernel.Wire::fieldAt value 5 4) (Kernel.Wire::fieldAt value 9 4) (Kernel.Wire::fieldAt value 13 4) (Kernel.Wire::fieldAt value 17 4) (Kernel.Wire::fieldAt value 21 4) (Kernel.Wire::fieldAt value 25 4) (Kernel.Wire::fieldAt value 29 4)) then (prim utf8_encode \"sealed\" : Bytes) else (prim utf8_encode \"not the sealed engine\" : Bytes)) else (prim utf8_encode \"invalid request\" : Bytes))"),
        "sealAnswer changed:\n{found}"
    );
    // The sealed address is sealedWord's eight words, each `then` literal and
    // the last `else` one, big-endian, as `xtask engine-seal` reads them. The
    // claims the guest replays are derived from it here, so the seal has one
    // source: a moved seal fails until claims.json follows.
    let chain = declaration(&kernel, "sealedWord");
    let tokens: Vec<&str> = chain
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let sealed: Vec<u32> = tokens
        .windows(2)
        .filter(|pair| pair[0] == "then" || pair[0] == "else")
        .filter_map(|pair| pair[1].parse().ok())
        .collect();
    assert_eq!(sealed.len(), 8, "sealedWord names eight words:\n{chain}");
    let address: Vec<u8> = sealed.iter().flat_map(|w| w.to_be_bytes()).collect();
    let request = |bytes: &[u8]| [&[80u8][..], bytes].concat();
    let mut flipped = address.clone();
    flipped[0] ^= 0x01;
    assert_claimed(
        80,
        &[
            (request(&address), "sealed"),
            (request(&flipped), "not the sealed engine"),
            (request(&address[..31]), "invalid request"),
        ],
    );
    // The page's form for decision 80 starts at the sealed address, so it
    // answers sealed as shown; it follows the seal as the claims do.
    let names: serde_json::Value = serde_json::from_str(&model("site/decision-names.json"))
        .expect("decision-names.json is JSON");
    for (i, word) in sealed.iter().enumerate() {
        assert_eq!(
            names["decisions"]["80"]["fields"][(1 + 4 * i).to_string()]["value"],
            serde_json::json!(word),
            "site/decision-names.json's decision-80 field {} is not sealedWord {i}",
            1 + 4 * i
        );
    }
    assert_routed(80, "Kernel.Seal::sealAnswer");
}
