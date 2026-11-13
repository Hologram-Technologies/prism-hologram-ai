//! The grep-shaped gates.
//!
//! These are crude on purpose. A gate that needs interpretation is a gate that
//! gets argued with; these ones read the source, find a token, and fail. Each
//! carries the rule it enforces in its failure message, because the point of a
//! red gate is to name the promise that was broken.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use repo_conformance::source;
use repo_model::{FailureRow, Model};

use crate::Fail;

/// The crates that ship: every crate under `crates/` that is not
/// `publish = false`. The rules below apply to those and not to the
/// dev-and-CI-only crates, which may use `std`, `alloc`, and floats freely.
///
/// Derived from the manifests rather than listed here. A list would be a second
/// place to remember, and the failure mode of a second place is that a crate
/// added to one is missing from the other --- silently, because a gate that
/// skips a crate cannot report on it. Reading `publish = false` asks the same
/// question `cargo publish` asks.
fn shipped(root: &Path) -> Result<Vec<String>, Fail> {
    let mut out = Vec::new();
    let dir = root.join("crates");
    if !dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        let manifest = path.join("Cargo.toml");
        let Ok(toml) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        if toml.lines().any(|l| l.trim() == "publish = false") {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            out.push(name.to_string());
        }
    }
    out.sort();
    Ok(out)
}

struct Source {
    rel: String,
    text: String,
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<Source>) -> Result<(), Fail> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, root, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path)?;
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(Source { rel, text });
        }
    }
    Ok(())
}

/// The lints every shipped crate root forbids. `forbid`, not `deny`: an inner
/// `#[allow]` cannot lift it, so the attribute at the root is the whole story.
const FORBIDDEN: [&str; 3] = [
    "clippy::unwrap_used",
    "clippy::expect_used",
    "clippy::panic",
];

/// Variant names that mean "anything else". A failure the model cannot name is
/// a failure it has not modelled.
const CATCH_ALL_NAMES: [&str; 2] = ["Other", "Unknown"];

/// Payload tokens that carry free text or an unnamed error instead of a
/// registered condition.
const FREE_TEXT: [&str; 4] = ["String", "str", "dyn", "anyhow"];

/// R5: every public failure is typed, modelled, documented and covered by a
/// negative test.
///
/// For every shipped crate: every `Result<` in its shipped code names, as its
/// error type, a type registered for that crate in `model/failures.toml`; each
/// registered type is one closed `enum` whose variants are exactly the crate's
/// rows for it, with no catch-all and no free text; and its roots forbid
/// `unwrap`, `expect` and `panic!`, which `just lint` then enforces. That each
/// row has a scenario and a test naming its variant is checked by the meta-gate
/// in `repo-conformance`.
pub fn audit_limits(root: &Path) -> Result<(), Fail> {
    let model = Model::load(&root.join("model"))?;
    model.check()?;
    let crates = shipped(root)?;

    let mut violations = Vec::new();
    for row in &model.failures.failure {
        if !crates.contains(&row.krate) {
            violations.push(format!(
                "model/failures.toml: {} names `{}`, which is not a shipped crate under crates/",
                row.id, row.krate
            ));
        }
    }
    violations.extend(lint_recipe(root)?);
    for name in &crates {
        let rows: Vec<&FailureRow> = model
            .failures
            .failure
            .iter()
            .filter(|r| &r.krate == name)
            .collect();
        audit_crate(root, name, &rows, &mut violations)?;
    }

    if !violations.is_empty() {
        return Err(format!(
            "R5: every public failure is typed, modelled, documented and covered by a \
             negative test. A shipped crate fails only as model/failures.toml lists, \
             through a closed enum whose variants are exactly its rows, and never by \
             `unwrap`, `expect` or `panic!`.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    println!(
        "audit-limits: {} shipped crates, {} registered failures; every Result names a \
         registered type, every registered type is a closed enum equal to its rows, and \
         every crate root forbids unwrap, expect and panic (R5)",
        crates.len(),
        model.failures.failure.len()
    );
    Ok(())
}

/// `just lint` is what turns the forbidden lints into a red gate, so the gate
/// checks it still runs clippy over every crate and every target, and that `vv`
/// still runs it.
fn lint_recipe(root: &Path) -> Result<Vec<String>, Fail> {
    let justfile = std::fs::read_to_string(root.join("Justfile"))?;
    let mut out = Vec::new();
    let vv_runs_lint = justfile.lines().any(|l| {
        l.strip_prefix("vv:")
            .is_some_and(|deps| deps.split_whitespace().any(|d| d == "lint"))
    });
    if !vv_runs_lint {
        out.push("Justfile: `vv` does not depend on `lint`".to_string());
    }
    let recipe: Vec<&str> = justfile
        .lines()
        .skip_while(|l| !l.starts_with("lint:"))
        .skip(1)
        .take_while(|l| l.starts_with(' ') || l.starts_with('\t'))
        .collect();
    let runs = recipe.iter().any(|l| {
        let l = l.trim();
        l.starts_with("cargo clippy")
            && l.contains("--workspace")
            && l.contains("--all-targets")
            && !l.contains("--exclude")
    });
    if !runs {
        out.push(
            "Justfile: `lint` does not run `cargo clippy --workspace --all-targets` over \
             every crate"
                .to_string(),
        );
    }
    Ok(out)
}

fn audit_crate(
    root: &Path,
    name: &str,
    rows: &[&FailureRow],
    violations: &mut Vec<String>,
) -> Result<(), Fail> {
    let dir = root.join("crates").join(name);
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml"))?;
    // A target declared elsewhere has a root this gate would not find.
    if manifest
        .lines()
        .any(|l| matches!(l.trim(), "[lib]" | "[[bin]]"))
    {
        violations.push(format!(
            "crates/{name}/Cargo.toml: declares its own `[lib]` or `[[bin]]`; a shipped \
             crate keeps its targets where Cargo finds them by convention, so the gate \
             reads every root"
        ));
    }
    let mut sources = Vec::new();
    if dir.join("src").exists() {
        collect(&dir.join("src"), root, &mut sources)?;
    }
    sources.sort_by(|a, b| a.rel.cmp(&b.rel));

    let roots: Vec<&Source> = sources.iter().filter(|s| is_root(&s.rel, name)).collect();
    if roots.is_empty() {
        violations.push(format!("crates/{name}: no crate root under src/"));
    }
    for src in &roots {
        let lints = forbidden_lints(&source::effective(&src.text));
        for lint in FORBIDDEN {
            if !lints.contains(lint) {
                violations.push(format!(
                    "{}: does not `#![forbid({lint})]`; a shipped crate does not fail by \
                     unwrap, expect or panic",
                    src.rel
                ));
            }
        }
    }

    let mut registered: BTreeMap<&str, BTreeMap<&str, &str>> = BTreeMap::new();
    for row in rows {
        registered
            .entry(row.error.as_str())
            .or_default()
            .insert(row.variant.as_str(), row.id.as_str());
    }

    let mut declared: BTreeMap<&str, Vec<Declaration>> = BTreeMap::new();
    for src in &sources {
        let code = source::effective(&src.text);
        let at = |offset: usize| format!("{}:{}", src.rel, source::line_of(&code, offset));

        for (offset, args) in result_uses(&code) {
            let error = match args.as_slice() {
                [_, error] => error.as_str(),
                _ => {
                    violations.push(format!(
                        "{}: `Result` with {} type argument(s) names no error type; write \
                         `Result<T, E>` with E registered for {name} in model/failures.toml",
                        at(offset),
                        args.len()
                    ));
                    continue;
                }
            };
            if !registered.contains_key(local_type(error)) {
                violations.push(format!(
                    "{}: `Result<_, {error}>`: `{error}` is not an error type registered for \
                     {name} in model/failures.toml",
                    at(offset)
                ));
            }
        }
        for offset in result_renames(&code) {
            violations.push(format!(
                "{}: `Result as`: renaming `Result` on import hides it from this gate",
                at(offset)
            ));
        }
        for (token, why) in [
            (
                "#[path",
                "a module from outside src/ is not read by this gate",
            ),
            (
                "include!(",
                "code included from outside src/ is not read by this gate",
            ),
        ] {
            for (offset, _) in code.match_indices(token) {
                violations.push(format!("{}: `{token}`: {why}", at(offset)));
            }
        }
        for ty in registered.keys() {
            for d in declarations(&code, ty) {
                declared.entry(ty).or_default().push(Declaration {
                    place: at(d.offset),
                    ..d
                });
            }
        }
    }

    for (ty, variants) in &registered {
        let found = declared.remove(ty).unwrap_or_default();
        let [decl] = found.as_slice() else {
            violations.push(format!(
                "crates/{name}: `{ty}` is registered and declared {} times; it is declared \
                 exactly once, as an enum",
                found.len()
            ));
            continue;
        };
        if decl.keyword != "enum" {
            violations.push(format!(
                "{}: `{ty}` is a {}; a registered error type is a closed enum",
                decl.place, decl.keyword
            ));
            continue;
        }
        if decl.non_exhaustive {
            violations.push(format!(
                "{}: `{ty}` is `#[non_exhaustive]`; a registered error type is closed",
                decl.place
            ));
        }
        let mut in_code = BTreeSet::new();
        for (variant, payload) in &decl.variants {
            in_code.insert(variant.as_str());
            if CATCH_ALL_NAMES.contains(&variant.as_str()) {
                violations.push(format!(
                    "{}: `{ty}::{variant}` is a catch-all; every failure is named",
                    decl.place
                ));
            }
            if let Some(token) = FREE_TEXT.iter().find(|t| has_word(payload, t)) {
                violations.push(format!(
                    "{}: `{ty}::{variant}` carries `{token}`; a failure carries typed data, \
                     not free text or an unnamed error",
                    decl.place
                ));
            }
            if !variants.contains_key(variant.as_str()) {
                violations.push(format!(
                    "{}: `{ty}::{variant}` has no row in model/failures.toml",
                    decl.place
                ));
            }
        }
        for (variant, id) in variants {
            if !in_code.contains(variant) {
                violations.push(format!(
                    "model/failures.toml: {id} registers `{ty}::{variant}`, which the enum \
                     at {} does not declare",
                    decl.place
                ));
            }
        }
    }
    Ok(())
}

fn is_root(rel: &str, name: &str) -> bool {
    let Some(path) = rel.strip_prefix(&format!("crates/{name}/src/")) else {
        return false;
    };
    match path.strip_prefix("bin/") {
        Some(bin) => {
            !bin.contains('/') || (bin.ends_with("/main.rs") && bin.matches('/').count() == 1)
        }
        None => path == "lib.rs" || path == "main.rs",
    }
}

/// Every lint named in a `#![forbid(...)]` in `code`.
fn forbidden_lints(code: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (at, open) in code.match_indices("#![forbid(") {
        let rest = &code[at + open.len()..];
        let Some(close) = rest.find(')') else {
            continue;
        };
        out.extend(
            rest[..close]
                .split(',')
                .map(|l| l.split_whitespace().collect::<String>())
                .filter(|l| !l.is_empty()),
        );
    }
    out
}

fn ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Does `word` occur in `code` as a whole identifier?
fn has_word(code: &str, word: &str) -> bool {
    code.match_indices(word).any(|(at, _)| {
        !code[..at].chars().next_back().is_some_and(ident)
            && !code[at + word.len()..].chars().next().is_some_and(ident)
    })
}

/// Every `Result<...>` (or `Result::<...>`) in `code`: its offset and its type
/// arguments split at the top level.
fn result_uses(code: &str) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    for (at, word) in code.match_indices("Result") {
        if code[..at].chars().next_back().is_some_and(ident) {
            continue;
        }
        let rest = &code[at + word.len()..];
        if rest.chars().next().is_some_and(ident) {
            continue;
        }
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("::").map_or(rest, str::trim_start);
        let Some(args) = rest.strip_prefix('<') else {
            continue;
        };
        out.push((at, type_arguments(args)));
    }
    out
}

/// Every `Result as ...` in `code`.
fn result_renames(code: &str) -> Vec<usize> {
    code.match_indices("Result")
        .filter(|(at, word)| {
            let rest = &code[at + word.len()..];
            !code[..*at].chars().next_back().is_some_and(ident)
                && rest.starts_with(char::is_whitespace)
                && rest
                    .trim_start()
                    .strip_prefix("as")
                    .is_some_and(|r| !r.starts_with(ident))
        })
        .map(|(at, _)| at)
        .collect()
}

/// The comma-separated arguments of a generic list whose `<` was just consumed.
fn type_arguments(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let (mut angle, mut nest) = (1i32, 0i32);
    let mut prev = ' ';
    for c in text.chars() {
        match c {
            '<' => angle += 1,
            '>' if prev != '-' => {
                angle -= 1;
                if angle == 0 {
                    break;
                }
            }
            '(' | '[' | '{' => nest += 1,
            ')' | ']' | '}' => nest -= 1,
            ',' if angle == 1 && nest == 0 => {
                args.push(std::mem::take(&mut current));
                prev = c;
                continue;
            }
            _ => {}
        }
        current.push(c);
        prev = c;
    }
    args.push(current);
    args.into_iter()
        .map(|a| a.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|a| !a.is_empty())
        .collect()
}

/// The crate-local type an error argument names: the identifier itself, or
/// one reached through `crate::`, `self::` or `super::`. Anything else ---
/// a foreign path, a reference, a tuple --- is not local and names nothing.
fn local_type(error: &str) -> &str {
    let mut rest = error.trim();
    for prefix in ["crate::", "self::"] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    while let Some(r) = rest.strip_prefix("super::") {
        rest = r;
    }
    let len = rest.find(|c: char| !ident(c)).unwrap_or(rest.len());
    let (name, tail) = rest.split_at(len);
    if name.is_empty() || !(tail.is_empty() || tail.trim_start().starts_with('<')) {
        return "";
    }
    name
}

/// One declaration of a registered type's name.
#[derive(Clone, Debug, Default)]
struct Declaration {
    offset: usize,
    place: String,
    keyword: &'static str,
    non_exhaustive: bool,
    /// (name, payload) per variant, for an enum.
    variants: Vec<(String, String)>,
}

/// Every `enum`, `struct`, `union` or `type` named `ty` in `code`.
fn declarations(code: &str, ty: &str) -> Vec<Declaration> {
    let mut out = Vec::new();
    for keyword in ["enum", "struct", "union", "type"] {
        for (at, _) in code.match_indices(keyword) {
            if code[..at].chars().next_back().is_some_and(ident) {
                continue;
            }
            let after = &code[at + keyword.len()..];
            if !after.starts_with(char::is_whitespace) {
                continue;
            }
            let after = after.trim_start();
            if !after.starts_with(ty) || after[ty.len()..].chars().next().is_some_and(ident) {
                continue;
            }
            // The attributes on this item: everything since the previous item
            // or block boundary.
            let start = code[..at].rfind([';', '{', '}']).map_or(0, |k| k + 1);
            let mut d = Declaration {
                offset: at,
                keyword,
                non_exhaustive: code[start..at].contains("non_exhaustive"),
                ..Declaration::default()
            };
            if keyword == "enum" {
                let body_from = at + keyword.len();
                if let Some(open) = code[body_from..].find('{').map(|k| body_from + k) {
                    let end = source::block_end(code, open);
                    d.variants = variants(&code[open + 1..end.saturating_sub(1)]);
                }
            }
            out.push(d);
        }
    }
    out
}

/// The variants of an enum body: (name, payload) for each, attributes removed.
fn variants(body: &str) -> Vec<(String, String)> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut nest = 0i32;
    let mut prev = ' ';
    for c in body.chars() {
        match c {
            '(' | '[' | '{' | '<' => nest += 1,
            '>' if prev == '-' => {}
            ')' | ']' | '}' | '>' => nest -= 1,
            ',' if nest == 0 => {
                parts.push(std::mem::take(&mut current));
                prev = c;
                continue;
            }
            _ => {}
        }
        current.push(c);
        prev = c;
    }
    parts.push(current);
    parts
        .into_iter()
        .filter_map(|p| {
            let mut p = p.trim();
            while let Some(attr) = p.strip_prefix("#[") {
                let close = attr.find(']')?;
                p = attr[close + 1..].trim_start();
            }
            let len = p.find(|c: char| !ident(c)).unwrap_or(p.len());
            let (name, payload) = p.split_at(len);
            (!name.is_empty()).then(|| (name.to_string(), payload.trim().to_string()))
        })
        .collect()
}

/// R4: nothing is deferred. No deferral marker, no stub, no placeholder
/// document section, no capability behind a flag that turns it off.
///
/// The markers are spelled in halves. This gate reads every crate *and*
/// `xtask`, which means it reads this file, and a list of forbidden tokens
/// written out in full is a list that matches itself --- the gate would fail on
/// its own definition, for ever. The alternative is exempting this file, and an
/// exemption is a hole: a real deferral parked in the gate would then be the one
/// place nothing looks. Split, the gate scans itself like anything else.
pub fn audit_deferral(root: &Path) -> Result<(), Fail> {
    let markers = [
        concat!("TO", "DO"),
        concat!("FIX", "ME"),
        concat!("XX", "X"),
        concat!("unimplemented", "!"),
        concat!("to", "do!"),
        concat!("for ", "now"),
        concat!("later ", "version"),
    ];
    let mut violations = Vec::new();

    // Every crate, not only the shipped ones, and `xtask` with them: R4 is a
    // promise about the repository, and a deferral parked in a gate is the same
    // deferral as one parked in shipped code.
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in [root.join("crates"), root.join("xtask")] {
        if dir.exists() {
            gather_all(&dir, &mut files)?;
        }
    }
    // Every Markdown file at the root, discovered rather than listed. A list
    // here would go stale the first time a document was added or renamed, and
    // the failure would be silent: a document nobody scans reports nothing.
    for entry in std::fs::read_dir(root)? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "md") {
            files.push(p);
        }
    }

    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        for (i, line) in text.lines().enumerate() {
            for marker in markers {
                if !line.contains(marker) {
                    continue;
                }
                // A backticked marker is a *mention*, not a use: the
                // documentation has to be able to name what the gate catches
                // without tripping it. Anything outside code spans is real.
                if outside_code_spans(line, marker) {
                    violations.push(format!("{rel}:{}: {}", i + 1, line.trim()));
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "R4: nothing is deferred. None of {} may appear, and no stub, no \
             placeholder section, and no capability behind a flag that turns it \
             off. Every capability ships in the one release.\n\n{}",
            markers.join(", "),
            violations.join("\n")
        )
        .into());
    }
    println!("audit-deferral: nothing is deferred (R4)");
    Ok(())
}

/// Does `marker` occur in `line` outside every backtick-delimited span?
fn outside_code_spans(line: &str, marker: &str) -> bool {
    let mut rest = line;
    let mut at = 0usize;
    while let Some(pos) = rest.find(marker) {
        let absolute = at + pos;
        // An odd number of backticks before this occurrence means it is inside
        // a span.
        if line[..absolute].matches('`').count().is_multiple_of(2) {
            return true;
        }
        at = absolute + marker.len();
        rest = &line[at..];
    }
    false
}

fn gather_all(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Fail> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            gather_all(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|e| e == "rs" || e == "md" || e == "toml")
        {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch repository with this one's model, a register of `failures`
    /// and one shipped crate `plant` whose `src/lib.rs` is `lib`.
    fn audit(name: &str, failures: &str, lib: &str) -> String {
        let root =
            std::env::temp_dir().join(format!("xtask-audit-limits-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let real = repo_model::repo_root();
        std::fs::create_dir_all(root.join("model")).unwrap();
        std::fs::create_dir_all(root.join("crates/plant/src")).unwrap();
        for file in ["ids.toml", "ledger.toml", "authorities.toml"] {
            std::fs::copy(real.join("model").join(file), root.join("model").join(file)).unwrap();
        }
        std::fs::copy(real.join("Justfile"), root.join("Justfile")).unwrap();
        std::fs::write(
            root.join("model/failures.toml"),
            format!("spec = \"template/1\"\n{failures}"),
        )
        .unwrap();
        std::fs::write(
            root.join("crates/plant/Cargo.toml"),
            "[package]\nname = \"plant\"\n",
        )
        .unwrap();
        std::fs::write(root.join("crates/plant/src/lib.rs"), lib).unwrap();
        let out = match audit_limits(&root) {
            Ok(()) => String::new(),
            Err(e) => e.to_string(),
        };
        std::fs::remove_dir_all(&root).unwrap();
        out
    }

    const LINTS: &str = "#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]\n";
    const ROW: &str = "[[failure]]\nid = \"FL-01\"\ncrate = \"plant\"\ntype = \"PlantError\"\nvariant = \"Refused\"\nsuite = \"plant\"\ncondition = \"The kernel refused.\"\n";

    fn lib(body: &str) -> String {
        format!("{LINTS}pub enum PlantError {{ Refused(u8) }}\n{body}")
    }

    #[test]
    fn a_registered_failure_passes() {
        let out = audit(
            "ok",
            ROW,
            &lib(
                "pub fn f() -> Result<(), crate::PlantError> { Ok(()) }\n\
                  /// Result<(), String>\n\
                  #[cfg(test)]\nmod t { fn g() -> Result<(), String> { let _ = \"{\"; Ok(()) } }\n",
            ),
        );
        assert_eq!(out, "");
    }

    #[test]
    fn an_unregistered_error_type_fails() {
        for (case, code) in [
            ("string", "pub fn f() -> Result<(), String> { Ok(()) }"),
            ("io", "pub fn f() -> std::io::Result<()> { Ok(()) }"),
            (
                "foreign",
                "pub fn f() -> Result<(), core::fmt::Error> { Ok(()) }",
            ),
            (
                "split",
                "pub fn f() -> Result<\n    (),\n    Box<dyn core::error::Error>,\n> { Ok(()) }",
            ),
            ("alias", "use core::result::Result as R;"),
        ] {
            let out = audit(case, ROW, &lib(code));
            assert!(out.contains("R5"), "{case}: {out}");
        }
        let out = audit(
            "after-test-use",
            ROW,
            &lib("#[cfg(test)]\nuse x;\nfn a() -> Result<(), String> { 1 }"),
        );
        assert!(out.contains("`Result<_, String>`"), "{out}");
    }

    #[test]
    fn the_enum_equals_its_rows() {
        let out = audit(
            "extra",
            ROW,
            &format!("{LINTS}pub enum PlantError {{ Refused(u8), Late }}\n"),
        );
        assert!(out.contains("`PlantError::Late` has no row"), "{out}");
        let out = audit(
            "missing",
            ROW,
            &format!("{LINTS}pub enum PlantError {{}}\n"),
        );
        assert!(
            out.contains("FL-01 registers `PlantError::Refused`"),
            "{out}"
        );
        let out = audit("struct", ROW, &format!("{LINTS}pub struct PlantError;\n"));
        assert!(out.contains("is a struct"), "{out}");
        let out = audit("absent", ROW, LINTS);
        assert!(out.contains("declared 0 times"), "{out}");
        let out = audit(
            "open",
            ROW,
            &format!("{LINTS}#[non_exhaustive]\npub enum PlantError {{ Refused(u8) }}\n"),
        );
        assert!(out.contains("non_exhaustive"), "{out}");
    }

    #[test]
    fn catch_alls_and_free_text_fail() {
        let other = format!(
            "{ROW}{}",
            ROW.replace("FL-01", "FL-02").replace("Refused", "Other")
        );
        let out = audit(
            "other",
            &other,
            &format!("{LINTS}pub enum PlantError {{ Refused(u8), Other }}\n"),
        );
        assert!(out.contains("`PlantError::Other` is a catch-all"), "{out}");
        for (case, payload) in [
            ("string", "(String)"),
            ("str", "(&'static str)"),
            ("dyn", "(Box<dyn core::error::Error>)"),
            ("anyhow", "(anyhow::Error)"),
            ("named", "{ why: String }"),
        ] {
            let out = audit(
                case,
                ROW,
                &format!("{LINTS}pub enum PlantError {{ Refused{payload} }}\n"),
            );
            assert!(out.contains("carries"), "{case}: {out}");
        }
    }

    #[test]
    fn the_lints_are_forbidden_at_the_root() {
        let out = audit(
            "deny",
            "failure = []\n",
            "#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]\n",
        );
        assert!(
            out.contains("does not `#![forbid(clippy::unwrap_used)]`"),
            "{out}"
        );
        let out = audit(
            "commented",
            "failure = []\n",
            "// #![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]\n",
        );
        assert!(
            out.contains("does not `#![forbid(clippy::panic)]`"),
            "{out}"
        );
        assert_eq!(audit("empty", "failure = []\n", LINTS), "");
    }

    #[test]
    fn a_row_names_a_shipped_crate() {
        let out = audit(
            "crate",
            &ROW.replace("\"plant\"\ntype", "\"nowhere\"\ntype"),
            &lib(""),
        );
        assert!(
            out.contains("`nowhere`, which is not a shipped crate"),
            "{out}"
        );
    }
}
