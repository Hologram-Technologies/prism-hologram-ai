//! Reading Rust source the way the gates need it.
//!
//! The gates that read source (`audit-limits`, the meta-gate) look for tokens.
//! A token inside a comment or a string literal is a mention, not code, and a
//! token inside a `#[cfg(test)]` item is not shipped. Everything here turns
//! source into the code a gate should judge, keeping every newline so a line
//! number found in the result is a line number in the file.
//!
//! This is a small lexer, not a parser. What it does not understand it leaves
//! in place, so a misreading makes a gate see more code, never less.

use std::path::Path;

/// `text` with every comment and the contents of every string and character
/// literal replaced by spaces. Delimiters stay, so `"..."` reads as `"   "`.
pub fn code_only(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(c.len());
    let blank = |ch: char| if ch == '\n' { '\n' } else { ' ' };
    let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let mut i = 0;
    while i < c.len() {
        let at = |k: usize| c.get(k).copied();
        // A line comment, doc comments included.
        if c[i] == '/' && at(i + 1) == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // A block comment. Rust nests them.
        if c[i] == '/' && at(i + 1) == Some('*') {
            let mut depth = 0usize;
            while i < c.len() {
                if c[i] == '/' && at(i + 1) == Some('*') {
                    depth += 1;
                    out.extend([' ', ' ']);
                    i += 2;
                } else if c[i] == '*' && at(i + 1) == Some('/') {
                    depth -= 1;
                    out.extend([' ', ' ']);
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
            continue;
        }
        // A raw string: `r"`, `r#"`, and the `b`/`c` prefixed forms.
        let starts_token = i == 0
            || !ident(c[i - 1])
            || (matches!(c[i - 1], 'b' | 'c') && (i < 2 || !ident(c[i - 2])));
        if c[i] == 'r' && starts_token {
            let mut k = i + 1;
            while at(k) == Some('#') {
                k += 1;
            }
            if at(k) == Some('"') {
                let hashes = k - i - 1;
                out.extend(c[i..=k].iter().copied());
                i = k + 1;
                while i < c.len() {
                    if c[i] == '"' && (0..hashes).all(|h| at(i + 1 + h) == Some('#')) {
                        out.extend(c[i..=i + hashes].iter().copied());
                        i += hashes + 1;
                        break;
                    }
                    out.push(blank(c[i]));
                    i += 1;
                }
                continue;
            }
        }
        // A string, byte string or C string.
        if c[i] == '"' {
            out.push('"');
            i += 1;
            while i < c.len() {
                if c[i] == '\\' {
                    out.push(' ');
                    i += 1;
                    if i < c.len() {
                        out.push(blank(c[i]));
                        i += 1;
                    }
                } else if c[i] == '"' {
                    out.push('"');
                    i += 1;
                    break;
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
            continue;
        }
        // A character literal; otherwise a lifetime or a label, which is code.
        if c[i] == '\'' {
            if at(i + 1) == Some('\\') {
                out.push('\'');
                i += 1;
                while i < c.len() && c[i] != '\'' {
                    out.push(' ');
                    i += 1;
                    if c.get(i - 1) == Some(&'\\') && i < c.len() {
                        out.push(' ');
                        i += 1;
                    }
                }
                if i < c.len() {
                    out.push('\'');
                    i += 1;
                }
                continue;
            }
            if at(i + 2) == Some('\'') && at(i + 1) != Some('\n') {
                out.extend(['\'', ' ', '\'']);
                i += 3;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out.into_iter().collect()
}

/// `code` (already through [`code_only`]) with every item under
/// `#[cfg(test)]` blanked: the attribute, then the item up to its `;` or its
/// matching `}`.
///
/// An item, not a line range. `#[cfg(test)] use x;` covers that `use` and
/// nothing after it, and a brace inside a test's string literal was already
/// blanked, so it cannot move where the test module ends.
pub fn without_test_items(code: &str) -> String {
    const ATTR: &str = "#[cfg(test)]";
    let mut c: Vec<char> = code.chars().collect();
    let attr: Vec<char> = ATTR.chars().collect();
    let mut i = 0;
    while i + attr.len() <= c.len() {
        if c[i..i + attr.len()] != attr[..] {
            i += 1;
            continue;
        }
        let start = i;
        let mut k = i + attr.len();
        let mut nest = 0i32;
        let mut end = c.len();
        while k < c.len() {
            match c[k] {
                '(' | '[' => nest += 1,
                ')' | ']' => nest -= 1,
                ';' if nest == 0 => {
                    end = k + 1;
                    break;
                }
                '{' if nest == 0 => {
                    let mut depth = 0i32;
                    while k < c.len() {
                        match c[k] {
                            '{' => depth += 1,
                            '}' => depth -= 1,
                            _ => {}
                        }
                        k += 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    end = k;
                    break;
                }
                _ => {}
            }
            k += 1;
        }
        for ch in &mut c[start..end] {
            if *ch != '\n' {
                *ch = ' ';
            }
        }
        i = end;
    }
    c.into_iter().collect()
}

/// The shipped code in `text`: no comments, no literal contents, no
/// `#[cfg(test)]` items. Line numbers are preserved.
pub fn effective(text: &str) -> String {
    without_test_items(&code_only(text))
}

/// The 1-based line of byte offset `at` in `text`.
pub fn line_of(text: &str, at: usize) -> usize {
    text[..at.min(text.len())].matches('\n').count() + 1
}

/// The byte offset just past the `}` that closes the `{` at `open`, in code
/// already through [`code_only`].
pub fn block_end(code: &str, open: usize) -> usize {
    let mut depth = 0i32;
    for (k, ch) in code[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return open + k + 1;
                }
            }
            _ => {}
        }
    }
    code.len()
}

/// One `#[test]` function the workspace compiles and runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestFn {
    /// The function's name.
    pub name: String,
    /// Its file, relative to the repository root, with `/` separators.
    pub file: String,
    /// Its body through [`code_only`]: what it names in code, not in a comment
    /// or a message.
    pub body: String,
}

/// Every `#[test]` function under `crates/` and `xtask/`.
///
/// Read from the source rather than from `cargo test -- --list`, because this
/// runs inside `cargo test` and a nested invocation blocks on the target
/// directory lock. A test marked `#[ignore]` or put under its own `#[cfg(...)]`
/// is left out: it may never run, so it discharges nothing.
pub fn test_index(root: &Path) -> Vec<TestFn> {
    let mut out = Vec::new();
    let mut stack = vec![root.join("crates"), root.join("xtask")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let file = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                tests_in(&code_only(&text), &file, &mut out);
            }
        }
    }
    out.sort_by(|a, b| (&a.file, &a.name).cmp(&(&b.file, &b.name)));
    out
}

fn tests_in(code: &str, file: &str, out: &mut Vec<TestFn>) {
    let mut rest = 0usize;
    while let Some(found) = code[rest..].find("#[test]") {
        let mut at = rest + found + "#[test]".len();
        rest = at;
        // Further attributes, then the function.
        let mut runs = true;
        loop {
            let tail = code[at..].trim_start();
            at = code.len() - tail.len();
            if tail.starts_with("#[") {
                if tail.starts_with("#[ignore") || tail.starts_with("#[cfg(") {
                    runs = false;
                }
                let Some(close) = tail.find(']') else { return };
                at += close + 1;
            } else {
                break;
            }
        }
        let Some(sig) = code[at..].strip_prefix("fn ") else {
            continue;
        };
        let name: String = sig
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(open) = code[at..].find('{').map(|k| at + k) else {
            return;
        };
        let end = block_end(code, open);
        if runs && !name.is_empty() {
            out.push(TestFn {
                name,
                file: file.to_string(),
                body: code[open..end].to_string(),
            });
        }
        rest = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_and_literals_are_not_code() {
        let text = "let a = \"Result<T, String>\"; // Result<(), String>\n/* x /* y */ Result< */ let b = 'x'; let c: &'static str = r#\"a\"b\"#;\n";
        let code = code_only(text);
        assert!(!code.contains("Result"), "{code}");
        assert!(code.contains("let b = ' ';"), "{code}");
        assert!(code.contains("&'static str"), "{code}");
        assert_eq!(code.matches('\n').count(), text.matches('\n').count());
    }

    /// The line-based reader this replaced skipped everything after
    /// `#[cfg(test)] use x;` up to the next closing brace, and let a `{` in a
    /// test's string literal hide all the code after the test module.
    #[test]
    fn a_test_item_hides_only_itself() {
        let text = "#[cfg(test)]\nuse x;\nfn a() -> Result<(), String> { 1 }\n";
        assert!(effective(text).contains("Result<(), String>"));

        let text = "#[cfg(test)]\nmod t { fn f() { let s = \"{\"; } }\nfn b() -> Result<(), String> { 1 }\n";
        let out = effective(text);
        assert!(out.contains("Result<(), String>"), "{out}");
        assert!(!out.contains("fn f"), "{out}");
    }

    #[test]
    fn ignored_and_conditional_tests_discharge_nothing() {
        let code = code_only(
            "#[test]\nfn runs_fl_01() { E::A; }\n#[test]\n#[ignore]\nfn skipped_fl_02() {}\n#[test]\n#[cfg(feature = \"x\")]\nfn maybe_fl_03() {}\n// #[test]\n// fn commented_fl_04() {}\n",
        );
        let mut out = Vec::new();
        tests_in(&code, "f.rs", &mut out);
        let names: Vec<&str> = out.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["runs_fl_01"]);
        assert!(out[0].body.contains("E::A"));
    }
}
