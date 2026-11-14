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
    // t2 4 per 16, t2r 68 per 256, ternary-per-128 36 per 128.
    let group = declaration(&blocks, "groupOf");
    assert!(
        group.contains("(if (format =? 2) then 16 else (if (format =? 3) then 256 else (if (format =? 4) then 128 else 32)))"),
        "group sizes changed:\n{group}"
    );
    let bytes = declaration(&blocks, "bytesPerGroup");
    assert!(
        bytes.contains("(if (format =? 0) then 20 else (if (format =? 1) then 16 else (if (format =? 2) then 4 else (if (format =? 3) then 68 else 36))))"),
        "bytes per group changed:\n{bytes}"
    );
    let answer = declaration(&blocks, "blockAnswer");
    assert!(
        answer.contains("((Foundation.Bytes::byteLength value) =? 18) && (known (Kernel.Wire::fieldAt value 1 1))")
            && answer.contains("(Kernel.Wire::decimal (blockLength (Kernel.Wire::fieldAt value 1 1) (Kernel.Wire::fieldAt value 2 8) (Kernel.Wire::fieldAt value 10 8)))")
            && answer.contains("(prim utf8_encode \"bad shape\" : Bytes)"),
        "decision 13 must read 18 bytes at offsets 1, 2 and 10 and answer the length or bad shape:\n{answer}"
    );
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
    assert_routed(17, "Kernel.Quant::quantAnswer");
}
