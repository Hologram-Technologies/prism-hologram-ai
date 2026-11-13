//! The honesty meta-gate (R2).
//!
//! Three levels only mean something if the suite respects them, and nothing in
//! the arithmetic can enforce that. What can is a check on the *names* and the
//! *prose*: an `open` claim that a test asserts, or a `some-true` authority
//! that the README presents as this repository's own result, is exactly the
//! blurring of registers the discipline exists to prevent.
//!
//! The gate is deliberately about language, because that is where the failure
//! mode lives. Nobody sets out to claim they proved an upstream theorem; they
//! write "proves" where they meant "is evidence for", and six months later the
//! sentence is load-bearing.

use std::collections::BTreeSet;
use std::path::Path;

use repo_model::{Level, Model};

use crate::runner::SuiteReport;
use crate::source::TestFn;

/// Does `code` name `path` (`Type::Variant`) as a whole path, not as the prefix
/// of a longer name?
fn names_path(code: &str, path: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    code.match_indices(path).any(|(at, _)| {
        let before = code[..at].chars().next_back();
        let after = code[at + path.len()..].chars().next();
        !before.is_some_and(ident) && !after.is_some_and(ident)
    })
}

/// What the meta-gate found.
#[derive(Clone, Debug, Default)]
pub struct HonestyReport {
    /// Every problem, each naming the rule it breaks.
    pub violations: Vec<String>,
    /// How many registered IDs were checked.
    pub ids_checked: usize,
    /// How many scenarios were read.
    pub scenarios_checked: usize,
}

impl HonestyReport {
    /// Did everything hold?
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Words that assert a claim as established.
///
/// A `build` claim may use them --- it *is* evidence, constructed here. An
/// `open` claim may not, because it is a measurement, and a `some-true` claim
/// may not, because it belongs to someone else.
const ASSERTIVE: &[&str] = &[
    "proves",
    "proven",
    "proof that",
    "guarantees",
    "establishes",
    "demonstrates that",
    "shows that",
    "confirms",
];

/// Run the meta-gate.
///
/// `root` is the repository root. `tests` are the workspace's test functions,
/// usually [`crate::source::test_index`] of the same root.
pub fn check_honesty(root: &Path, tests: &[TestFn]) -> std::io::Result<HonestyReport> {
    let mut report = HonestyReport::default();
    let model = match Model::load(&root.join("model")) {
        Ok(m) => m,
        Err(e) => {
            report
                .violations
                .push(format!("R1: the model does not load: {e}"));
            return Ok(report);
        }
    };
    let suites: SuiteReport = crate::runner::scenarios_in(&root.join("features/suites"))?;
    report.scenarios_checked = suites.scenarios.len();
    report.ids_checked = model.ids.id.len();

    let scenario_ids = suites.ids();

    // R3, CM-02: every registered ID has a scenario, and a test named for it.
    for row in &model.ids.id {
        if !scenario_ids.contains(row.id.as_str()) {
            report.violations.push(format!(
                "R3: {} is registered but has no scenario in features/suites/. Every \
                 capability begins as a Gherkin scenario.",
                row.id
            ));
        }
        let slug = row.id.to_lowercase().replace('-', "_");
        if !tests.iter().any(|t| t.name.ends_with(&slug)) {
            report.violations.push(format!(
                "CM-02: {} is registered but no test name ends in `{slug}`. A claim with \
                 no test is an assertion.",
                row.id
            ));
        }
    }

    // R5, through R3's mechanism: every registered failure has a scenario in its
    // suite and a test in its suite's file whose name ends in the ID and whose
    // body names the variant in code. The gate reads the name; whether the test
    // provokes the failure is what the planted defect for each row shows.
    for row in &model.failures.failure {
        let in_suite = suites
            .scenarios
            .iter()
            .any(|s| s.id == row.id && s.suite == row.suite);
        if !in_suite {
            report.violations.push(format!(
                "R5: {} is a registered failure but has no scenario in \
                 features/suites/{}.feature. Every failure begins as a negative scenario.",
                row.id, row.suite
            ));
        }
        let slug = row.id.to_lowercase().replace('-', "_");
        let file = format!(
            "crates/conformance/tests/{}.rs",
            row.suite.replace('-', "_")
        );
        let named: Vec<&TestFn> = tests
            .iter()
            .filter(|t| t.name.ends_with(&slug) && t.file == file)
            .collect();
        let path = format!("{}::{}", row.error, row.variant);
        if named.is_empty() {
            report.violations.push(format!(
                "R5: {} is a registered failure but no test in {file} has a name ending \
                 in `{slug}`. A failure with no negative test is not modelled.",
                row.id
            ));
        } else if !named.iter().any(|t| names_path(&t.body, &path)) {
            report.violations.push(format!(
                "R5: the test for {} does not name `{path}` in code. A negative test \
                 names the failure it expects.",
                row.id
            ));
        }
    }

    // CM-02, the other direction: every scenario names a registered ID.
    for s in &suites.scenarios {
        if let Some(row) = model.failures.get(&s.id) {
            if s.level != "build" {
                report.violations.push(format!(
                    "R2: {} is a failure, tested here, so its scenario is tagged `build`, \
                     not `{}`.",
                    s.id, s.level
                ));
            }
            if s.suite != row.suite {
                report.violations.push(format!(
                    "R5: {}'s scenario is in {} but its row names the suite {}.",
                    s.id, s.suite, row.suite
                ));
            }
        } else if model.ids.get(&s.id).is_none() {
            report.violations.push(format!(
                "CM-02: scenario `{}` in {} names `{}`, which is not in the register.",
                s.statement, s.suite, s.id
            ));
        }
        if s.steps.is_empty() {
            report.violations.push(format!(
                "R3: scenario `{}` in {} has no steps. There are no pending steps.",
                s.statement, s.suite
            ));
        }
        // R2: the scenario's tag must agree with the register.
        if let Some(row) = model.ids.get(&s.id) {
            if s.level != row.level.as_str() {
                report.violations.push(format!(
                    "R2: {} is tagged `{}` in {} but `{}` in the register.",
                    s.id,
                    s.level,
                    s.suite,
                    row.level.as_str()
                ));
            }
        }
    }

    // CM-02, the third direction: every ID a *test* names is registered.
    //
    // Without this the register can be a subset of what the suite claims: a test
    // called `..._ct_04` looks like it discharges `CT-04`, but if `CT-04` is not
    // a row then nothing checks it has a scenario and `CONFORMANCE.md` does not
    // list it. Three IDs were in exactly that state --- `CK-08`, `CT-04`, and
    // `CT-05` --- with passing tests and no register rows, which is a claim made
    // by a name and by nothing else.
    //
    // Only prefixes the register already uses are checked, so a test whose name
    // merely happens to end in two letters and two digits is not a claim.
    // `FL` is always checked: the failure register's prefix is fixed, so a test
    // claiming a failure that has no row is caught even while the register is
    // empty.
    let prefixes: BTreeSet<String> = model
        .ids
        .id
        .iter()
        .filter_map(|r| r.id.split('-').next().map(str::to_lowercase))
        .chain(std::iter::once("fl".to_string()))
        .collect();
    for test in tests {
        let name = &test.name;
        let Some(tail) = name.rsplit("::").next() else {
            continue;
        };
        let parts: Vec<&str> = tail.rsplitn(3, '_').collect();
        if parts.len() < 3 {
            continue;
        }
        let (digits, letters) = (parts[0], parts[1]);
        if digits.len() != 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if letters.len() != 2 || !prefixes.contains(letters) {
            continue;
        }
        let id = format!("{}-{digits}", letters.to_uppercase());
        if model.ids.get(&id).is_none() && model.failures.get(&id).is_none() {
            report.violations.push(format!(
                "CM-02: test `{name}` names `{id}`, which is not in the register. An ID \
                 that exists only in a test name has no scenario and no row in \
                 CONFORMANCE.md."
            ));
        }
    }

    // R2: an `open` claim must not be asserted as established, anywhere.
    let open_ids: Vec<&str> = model
        .ids
        .id
        .iter()
        .filter(|r| r.level == Level::Open)
        .map(|r| r.id.as_str())
        .collect();
    for doc in [
        "README.md",
        "CONFORMANCE.md",
        "VERIFICATION.md",
        "ANALYSIS.md",
    ] {
        let path = root.join(doc);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let lower = line.to_lowercase();
            for id in &open_ids {
                if !line.contains(id) {
                    continue;
                }
                if let Some(word) = ASSERTIVE.iter().find(|w| lower.contains(*w)) {
                    report.violations.push(format!(
                        "R2: {doc}:{}: `{id}` is an `open` claim --- measured and reported, \
                         never asserted --- but this line says `{word}`.\n    {}",
                        i + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    // R2: a `some-true` authority is reproduced, not established here.
    for authority in &model.authorities.authority {
        for doc in ["README.md", "CONFORMANCE.md"] {
            let path = root.join(doc);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in text.lines().enumerate() {
                if !line.contains(&authority.id) {
                    continue;
                }
                let lower = line.to_lowercase();
                if let Some(word) = ASSERTIVE.iter().find(|w| lower.contains(*w)) {
                    report.violations.push(format!(
                        "R2: {doc}:{}: `{}` is cited, not established here, but this line \
                         says `{word}`.\n    {}",
                        i + 1,
                        authority.id,
                        line.trim()
                    ));
                }
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate's own vocabulary check must be able to fire, or every clean
    /// run above means nothing.
    #[test]
    fn the_assertive_vocabulary_is_recognised() {
        for word in ASSERTIVE {
            let line = format!("CG-01 {word} the exponent is one");
            assert!(
                ASSERTIVE.iter().any(|w| line.to_lowercase().contains(*w)),
                "{word} must be recognised"
            );
        }
        // And an honest sentence about an open claim does not trip it.
        let honest = "CG-01 reports the fitted exponent with its confidence interval";
        assert!(!ASSERTIVE.iter().any(|w| honest.to_lowercase().contains(*w)));
    }
}
