//! The claims the engine's crates make of the kernel, one file per decision
//! under `crates/conformance/tests/kernel/claims/`, read as one document.
//!
//! Why one file per decision: every pull request that adds a decision adds
//! claims, and one shared list made every such pull request conflict with every
//! other. The aggregate is this function's, never a committed file.
//!
//! The filing is checked on every read: a file is `NNN.json` for decision NNN,
//! names that decision, holds at least one claim, and every request in it opens
//! with that decision's byte. A claim filed under another decision would be
//! replayed and still pass, but it would no longer be where a reader of that
//! decision looks, so it is refused.

use std::path::Path;

/// Where the claims are, from the repository root.
pub const DIR: &str = "crates/conformance/tests/kernel/claims";

/// Every claim, in decision order, as `{"claims": [{"request", "answer"}]}`.
pub fn document(root: &Path) -> serde_json::Value {
    let dir = root.join(DIR);
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "{DIR} holds no claims");
    let mut all = Vec::new();
    for name in files {
        let tag: u8 = name
            .strip_suffix(".json")
            .filter(|stem| stem.len() == 3)
            .and_then(|stem| stem.parse().ok())
            .unwrap_or_else(|| panic!("{DIR}/{name}: a claims file is NNN.json, NNN its decision"));
        let text = std::fs::read_to_string(dir.join(&name)).expect("a claims file");
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{DIR}/{name}: {e}"));
        assert_eq!(
            value["decision"].as_u64(),
            Some(u64::from(tag)),
            "{DIR}/{name} names decision {tag}"
        );
        let claims = value["claims"]
            .as_array()
            .unwrap_or_else(|| panic!("{DIR}/{name}: a list of claims"));
        assert!(!claims.is_empty(), "{DIR}/{name} holds no claim");
        let byte = format!("{tag:02x}");
        for claim in claims {
            let request = claim["request"].as_str().expect("a request");
            assert!(
                claim["answer"].is_string(),
                "{DIR}/{name}: {request} has no answer"
            );
            assert!(
                request.starts_with(&byte),
                "{DIR}/{name}: {request} is not a decision-{tag} request"
            );
            all.push(claim.clone());
        }
    }
    serde_json::json!({ "claims": all })
}
