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
                .and_then(|rest| rest.strip_suffix(','))
            else {
                continue;
            };
            if let Some(first) = seen.insert(name.to_string(), module.clone()) {
                panic!("`{name}` is declared in both {first} and {module}");
            }
        }
    }
}

/// The lowercase hex of `text`, as an example's response is written in the model.
fn hex(text: &str) -> String {
    text.bytes().map(|b| format!("{b:02x}")).collect()
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
