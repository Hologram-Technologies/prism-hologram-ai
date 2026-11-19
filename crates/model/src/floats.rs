//! The integer path's float audit (EN-58).
//!
//! Why: Hologram AI's inference uses integers only, so no floating-point type
//! or operation may lie between the bytes admitted and the token ids. This
//! names the files the integer forward pass is computed in and finds every
//! float in them; `cargo xtask audit-float` refuses any, and the conformance
//! suite plants one in each language to show the refusal.

/// The Rust sources of the integer path: the exact stages, the forward pass
/// over them, and the page's module that runs it.
pub const RUST: [&str; 4] = [
    "crates/exact/src",
    "crates/cpu-decode/src",
    "crates/cpu-encode/src",
    "crates/wasm/src",
];

/// The page's side of the integer path: it hands the module bytes and text
/// and reads back the token and the logits' binary32 bits, or an encoder's
/// embedding and label scores (row 9).
pub const SCRIPT: [&str; 2] = [
    "site/app/engine/integer.mjs",
    "site/app/engine/integer-encoder.mjs",
];

/// Floats still on the way to the integer path, each reported on every run as
/// an open finding with the plan row that removes it, so none passes
/// silently: the file, the line its function starts with, and the row. The
/// function is read to its closing brace at the left margin.
pub const OPEN: [(&str, &str, &str); 0] = [];

/// The floats in the function of `text` that starts with `start`, as
/// [`float_uses`] reports them; `None` when no such function is there.
#[must_use]
pub fn open_uses(path: &str, text: &str, start: &str) -> Option<Vec<String>> {
    let lines: Vec<&str> = text.lines().collect();
    let first = lines.iter().position(|l| l.starts_with(start))?;
    let end = (first..lines.len()).find(|&i| lines[i] == "}")?;
    let body = lines[first..=end].join("\n");
    let found = float_uses(path, &body, false);
    // Lines counted from the file's start, not the function's.
    Some(
        found
            .into_iter()
            .map(|f| {
                let (head, rest) = f.split_once(": ").unwrap_or((&f, ""));
                let n: usize = head
                    .rsplit(':')
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                format!("{path}:{}: {rest}", n + first)
            })
            .collect(),
    )
}

/// What makes or works on a float in the page's script: arithmetic the
/// language does in doubles, and turning a number into binary32. A view of
/// bits the module wrote, `new Float32Array(bytes.buffer)`, computes nothing
/// and is not among them.
const SCRIPT_FLOATS: [&str; 7] = [
    "Math.",
    "parseFloat",
    "Float64Array",
    "Float32Array.from",
    "new Float32Array([",
    "toFixed",
    "toPrecision",
];

/// The code of a line: what is left after its comment and its string
/// literals, which name floats without computing with them.
fn code(line: &str) -> String {
    let mut out = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (at, c) in line.char_indices() {
        match quote {
            Some(_) if escaped => escaped = false,
            Some(_) if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            // A quote inside a character literal, `'"'`, opens nothing.
            None if (c == '"' || c == '`') && !line[..at].ends_with('\'') => quote = Some(c),
            None if line[at..].starts_with("//") => break,
            None => out.push(c),
        }
    }
    out
}

/// Whether `word` stands alone in `text`, not inside a longer name.
fn word(text: &str, word: &str) -> bool {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.match_indices(word).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(ident)
            && !text[at + word.len()..].chars().next().is_some_and(ident)
    })
}

/// A numeric literal with a fraction or an exponent, such as `1.5` or `2e-6`:
/// a run of digits that starts a token, then `.` and a digit, or `e` and a
/// digit or a minus.
fn real_literal(text: &str) -> bool {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit()
            && (i == 0
                || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'));
        if !starts {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        let next = b.get(j + 1).copied().unwrap_or(0);
        match b.get(j) {
            Some(b'.') if next.is_ascii_digit() => return true,
            Some(b'e' | b'E') if next.is_ascii_digit() || next == b'-' => return true,
            _ => {}
        }
        i = j.max(i + 1);
    }
    false
}

/// Every float in a file of the integer path, as `path:line: text`. `rust`
/// says which language the text is.
#[must_use]
pub fn float_uses(path: &str, text: &str, rust: bool) -> Vec<String> {
    let mut found = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let code = code(line);
        let float = if rust {
            word(&code, "f32") || word(&code, "f64") || real_literal(&code)
        } else {
            SCRIPT_FLOATS.iter().any(|f| code.contains(f)) || real_literal(&code)
        };
        if float {
            found.push(format!("{path}:{}: {}", i + 1, line.trim()));
        }
    }
    found
}
