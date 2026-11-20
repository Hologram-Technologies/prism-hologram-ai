//! `lexlean verify` over `app/src/Proofs`, reusing each module's earlier
//! verification by content address and verifying only what changed.
//!
//! Why: the proof modules took about six minutes of every acceptance run, and
//! a pull request changes a handful of them. Everything else was verified by
//! an earlier run against byte-identical inputs.
//!
//! How: each proof module is keyed by the BLAKE3 of everything its
//! verification reads. LexLean's own build manifest for that one module names
//! its source and every module in its import closure, the lock, the project
//! configuration and the lexicons, each by SHA-256, with the compiler version,
//! its semantics ID and the Lean toolchain. The key adds what the manifest does
//! not name: the version output and executable digest of `lexlean`, the
//! version output of `lean` and `lake`, and the SDK image `prismpm.lock` pins.
//!
//! The key only finds a candidate; nothing in the store is trusted. A
//! candidate is reused only when its attestation ID re-derives from the
//! attestation's bytes (LexLean §22.9), the build manifest it names is the one
//! kept beside it, it says `verified` with every declaration `ok`, it was made
//! by this `lexlean` executable and this `lean`, and its build manifest names
//! every input of this module's manifest now, byte for byte. A failed check is
//! a miss, and every miss is verified in full, together, in one `lexlean
//! verify`, whose attestation is kept for the next run.
//!
//! What: `.prism-proof/lean/keys/<key>` holds an attestation ID, and
//! `.prism-proof/lean/attestations/<id>/` holds that attestation and its build
//! manifest. bootstrap.yml caches `.prism-proof`.
//!
//! When: only with `PRISM_PROOF_REUSE=1`, which the `app` recipe sets on a pull
//! request or a merge queue run and on nothing else, as `audit-bootstrap`
//! requires. A main push, a tag and a local run verify every module.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Fail;

const STORE: &str = ".prism-proof/lean";
const PROOFS: &str = "src/Proofs";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// LexLean's framed SHA-256 (§21.1): the domain and a NUL, then each frame as
/// a big-endian u32 label length, the label, a big-endian u64 byte length and
/// the bytes.
fn framed_sha256(domain: &str, frames: &[(&str, &[u8])]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0u8]);
    for (label, bytes) in frames {
        hasher.update(u32::try_from(label.len()).unwrap_or(u32::MAX).to_be_bytes());
        hasher.update(label.as_bytes());
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The attestation ID the bytes of an attestation file re-derive to: LexLean
/// hashes the canonical body without its own `attestation_id` field, which in
/// canonical JSON is the first key, and the file ends in one LF.
fn attestation_id(bytes: &[u8]) -> Result<String, String> {
    const PREFIX: &[u8] = b"{\"attestation_id\":\"";
    let rest = bytes
        .strip_prefix(PREFIX)
        .ok_or("the attestation does not open with its ID")?;
    let after_id = rest
        .get(64..)
        .and_then(|tail| tail.strip_prefix(b"\","))
        .ok_or("the attestation's ID is not 64 characters followed by a field")?;
    let after_id = after_id
        .strip_suffix(b"\n")
        .ok_or("the attestation does not end in one LF")?;
    let mut body = Vec::with_capacity(after_id.len() + 1);
    body.push(b'{');
    body.extend_from_slice(after_id);
    Ok(framed_sha256(
        "lexlean-attestation-v1",
        &[("attestation-body", &body)],
    ))
}

/// The tools this run verifies with, as the key and the check name them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tools {
    pub lexlean_version: String,
    pub lexlean_sha256: String,
    pub lean_version: String,
    pub lake_version: String,
    pub sdk_image: String,
}

/// The BLAKE3 key of one proof module: every input its verification reads.
fn key(tools: &Tools, module: &str, manifest: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    let frames: [(&str, &[u8]); 7] = [
        ("lexlean-version", tools.lexlean_version.as_bytes()),
        ("lexlean-executable-sha256", tools.lexlean_sha256.as_bytes()),
        ("lean-version", tools.lean_version.as_bytes()),
        ("lake-version", tools.lake_version.as_bytes()),
        ("sdk-image", tools.sdk_image.as_bytes()),
        ("module", module.as_bytes()),
        ("build-manifest", manifest),
    ];
    hasher.update(b"prism-hologram-ai/proof-module/1\0");
    for (label, bytes) in frames {
        hasher.update(&(label.len() as u64).to_be_bytes());
        hasher.update(label.as_bytes());
        hasher.update(&(bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    hasher.finalize().to_hex().to_string()
}

fn text<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("no {pointer}"))
}

fn input_rows(manifest: &Value) -> Result<BTreeSet<(String, String, String)>, String> {
    manifest
        .get("inputs")
        .and_then(Value::as_array)
        .ok_or("the build manifest names no inputs")?
        .iter()
        .map(|row| {
            Ok((
                text(row, "/kind")?.to_owned(),
                text(row, "/path")?.to_owned(),
                text(row, "/sha256")?.to_owned(),
            ))
        })
        .collect()
}

fn module_rows(manifest: &Value) -> Result<BTreeSet<String>, String> {
    manifest
        .get("modules")
        .and_then(Value::as_array)
        .ok_or("the build manifest names no modules")?
        .iter()
        .map(|row| text(row, "/lean_module").map(str::to_owned))
        .collect()
}

/// The whole read-time check of one kept verification. Pure, so each planted
/// defect is a unit test. `Ok` means this module, with these inputs, under
/// these tools, is what the kept attestation verified.
pub fn binds(
    id: &str,
    attestation: &[u8],
    kept_manifest: &[u8],
    current_manifest: &[u8],
    tools: &Tools,
) -> Result<(), String> {
    if !is_id(id) {
        return Err("the kept attestation ID is not a SHA-256".into());
    }
    let derived = attestation_id(attestation)?;
    if derived != id {
        return Err(format!(
            "the attestation re-derives to {derived}, not the {id} it is kept as"
        ));
    }
    let att: Value = serde_json::from_slice(attestation).map_err(|e| e.to_string())?;
    if text(&att, "/status")? != "verified" {
        return Err("the attestation does not say verified".into());
    }
    let declarations = att
        .get("declarations")
        .and_then(Value::as_array)
        .ok_or("the attestation lists no declarations")?;
    if declarations.is_empty() {
        return Err("the attestation lists no declarations".into());
    }
    if let Some(bad) = declarations
        .iter()
        .find(|row| row.get("result").and_then(Value::as_str) != Some("ok"))
    {
        return Err(format!(
            "the attestation records {} as not ok",
            text(bad, "/name").unwrap_or("a declaration")
        ));
    }
    if text(&att, "/build_manifest/sha256")? != sha256_hex(kept_manifest) {
        return Err(
            "the build manifest kept beside the attestation is not the one it names".into(),
        );
    }
    if att
        .pointer("/build_manifest/byte_length")
        .and_then(Value::as_u64)
        != Some(kept_manifest.len() as u64)
    {
        return Err("the kept build manifest's length is not the one the attestation names".into());
    }
    if text(&att, "/lexlean/executable_sha256")? != tools.lexlean_sha256 {
        return Err("a different lexlean executable made the attestation".into());
    }
    if text(&att, "/toolchain/lean/version_output")?.trim() != tools.lean_version.trim() {
        return Err("a different Lean made the attestation".into());
    }
    if text(&att, "/toolchain/lake/version_output")?.trim() != tools.lake_version.trim() {
        return Err("a different Lake made the attestation".into());
    }

    let kept: Value = serde_json::from_slice(kept_manifest).map_err(|e| e.to_string())?;
    let now: Value = serde_json::from_slice(current_manifest).map_err(|e| e.to_string())?;
    if text(&att, "/build_id")? != text(&kept, "/build_id")? {
        return Err("the attestation and its build manifest name different builds".into());
    }
    for field in [
        "/compiler/version",
        "/compiler/semantics_id",
        "/language",
        "/lean_toolchain",
    ] {
        if text(&kept, field)? != text(&now, field)? {
            return Err(format!("{field} differs from what was verified"));
        }
    }
    let kept_inputs = input_rows(&kept)?;
    for row in input_rows(&now)? {
        if !kept_inputs.contains(&row) {
            return Err(format!(
                "{} {} is not the {} that was verified",
                row.0,
                row.1,
                &row.2[..12]
            ));
        }
    }
    let kept_modules = module_rows(&kept)?;
    for module in module_rows(&now)? {
        if !kept_modules.contains(&module) {
            return Err(format!("{module} was not among the modules verified"));
        }
    }
    Ok(())
}

fn run(dir: &Path, program: &str, args: &[&str]) -> Result<std::process::Output, Fail> {
    Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| format!("{program}: {e}").into())
}

fn stdout(dir: &Path, program: &str, args: &[&str]) -> Result<String, Fail> {
    let out = run(dir, program, args)?;
    if !out.status.success() {
        return Err(format!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn on_path(program: &str) -> Result<PathBuf, Fail> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(program))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("{program} is not on PATH").into())
}

fn tools(root: &Path, app: &Path) -> Result<Tools, Fail> {
    let lexlean = std::fs::canonicalize(on_path("lexlean")?)?;
    let lock = std::fs::read_to_string(root.join("prismpm.lock"))?;
    let sdk_image = lock
        .split("\"sdk_image\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .ok_or("prismpm.lock names no sdk_image")?
        .to_owned();
    Ok(Tools {
        lexlean_version: stdout(app, "lexlean", &["--version"])?,
        lexlean_sha256: sha256_hex(&std::fs::read(&lexlean)?),
        lean_version: stdout(app, "lean", &["--version"])?,
        lake_version: stdout(app, "lake", &["--version"])?,
        sdk_image,
    })
}

/// LexLean's build manifest for one module alone, which names its closure.
fn manifest_of(app: &Path, module: &str) -> Result<Vec<u8>, Fail> {
    let result: Value = serde_json::from_str(&stdout(
        app,
        "lexlean",
        &["build", module, "--diagnostic-format", "json"],
    )?)?;
    let build = text(&result, "/build_id")?;
    if !is_id(build) {
        return Err(format!("{module}: lexlean build reported no build ID").into());
    }
    Ok(std::fs::read(
        app.join(".lexlean/build").join(build).join("manifest.json"),
    )?)
}

fn read_kept(store: &Path, key: &str) -> Option<(String, Vec<u8>, Vec<u8>)> {
    let id = std::fs::read_to_string(store.join("keys").join(key)).ok()?;
    let id = id.trim().to_owned();
    if !is_id(&id) {
        return None;
    }
    let dir = store.join("attestations").join(&id);
    let attestation = std::fs::read(dir.join("attestation.json")).ok()?;
    let manifest = std::fs::read(dir.join("build-manifest.json")).ok()?;
    Some((id, attestation, manifest))
}

/// Reuse only when the `app` recipe says this run may: a pull request or a
/// merge queue run. Anything else, unset included, verifies in full.
fn reuse_allowed() -> bool {
    std::env::var("PRISM_PROOF_REUSE").is_ok_and(|value| value == "1")
}

/// `just app`'s proof step.
pub fn verify(root: &Path) -> Result<(), Fail> {
    let started = std::time::Instant::now();
    let app = root.join("app");
    let store = root.join(STORE);
    let mut modules: Vec<String> = std::fs::read_dir(app.join(PROOFS))?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".lex.tex"))
        .map(|name| format!("{PROOFS}/{name}"))
        .collect();
    modules.sort();
    if modules.is_empty() {
        return Err(format!("app/{PROOFS} holds no proof module").into());
    }
    let tools = tools(root, &app)?;
    let full = !reuse_allowed();

    let mut keyed = Vec::new();
    let mut misses = Vec::new();
    for module in &modules {
        let manifest = manifest_of(&app, module)?;
        let key = key(&tools, module, &manifest);
        let verdict = if full {
            Err(
                "this run verifies in full (only a pull request or a merge queue run reuses)"
                    .to_owned(),
            )
        } else {
            match read_kept(&store, &key) {
                None => Err("no earlier verification of these inputs".to_owned()),
                Some((id, attestation, kept)) => {
                    binds(&id, &attestation, &kept, &manifest, &tools).map(|()| id)
                }
            }
        };
        match verdict {
            Ok(id) => println!("proofs: {module} reused, attestation {id}"),
            Err(why) => {
                println!("proofs: {module} verifies: {why}");
                misses.push(module.clone());
            }
        }
        keyed.push((module.clone(), key, manifest));
    }

    if !misses.is_empty() {
        let mut args = vec!["verify"];
        args.extend(misses.iter().map(String::as_str));
        let out = run(&app, "lexlean", &args)?;
        let report = String::from_utf8_lossy(&out.stderr);
        eprint!("{report}");
        print!("{}", String::from_utf8_lossy(&out.stdout));
        if !out.status.success() {
            return Err("lexlean verify failed on the proof modules above".into());
        }
        let id = report
            .split("attestation ")
            .nth(1)
            .map(|rest| rest.chars().take(64).collect::<String>())
            .filter(|id| is_id(id))
            .ok_or("lexlean verify reported no attestation")?;
        let source = app.join(".lexlean/verified").join(&id);
        let kept = store.join("attestations").join(&id);
        std::fs::create_dir_all(&kept)?;
        for file in ["attestation.json", "build-manifest.json"] {
            std::fs::copy(source.join(file), kept.join(file))?;
        }
        std::fs::create_dir_all(store.join("keys"))?;
        let attestation = std::fs::read(kept.join("attestation.json"))?;
        let kept_manifest = std::fs::read(kept.join("build-manifest.json"))?;
        for (module, key, manifest) in &keyed {
            if !misses.contains(module) {
                continue;
            }
            // What was just verified must pass the very check a later run
            // makes; if it does not, the key or the check is wrong.
            binds(&id, &attestation, &kept_manifest, manifest, &tools).map_err(|why| {
                format!("{module}: the fresh attestation fails the read check: {why}")
            })?;
            std::fs::write(store.join("keys").join(key), format!("{id}\n"))?;
        }
    }

    // Keep only what this tree's modules name, so the store does not grow.
    let live: BTreeSet<&str> = keyed.iter().map(|(_, key, _)| key.as_str()).collect();
    let mut referenced = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(store.join("keys")) {
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if live.contains(name.as_str()) {
                if let Ok(id) = std::fs::read_to_string(entry.path()) {
                    referenced.insert(id.trim().to_owned());
                }
            } else {
                std::fs::remove_file(entry.path())?;
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(store.join("attestations")) {
        for entry in entries.filter_map(Result::ok) {
            if !referenced.contains(&*entry.file_name().to_string_lossy()) {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
    }
    println!(
        "proofs: {} modules, {} reused by content address, {} verified, in {}s",
        modules.len(),
        modules.len() - misses.len(),
        misses.len(),
        started.elapsed().as_secs()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Tools {
        Tools {
            lexlean_version: "lexlean 0.3.0\n".into(),
            lexlean_sha256: "aa".repeat(32),
            lean_version: "Lean (version 4.32.1)\n".into(),
            lake_version: "Lake version 5.0.0\n".into(),
            sdk_image: "ghcr.io/x@sha256:".to_owned() + &"bb".repeat(32),
        }
    }

    fn manifest(source_sha: &str) -> Vec<u8> {
        format!(
            "{{\"build_id\":\"{b}\",\"compiler\":{{\"semantics_id\":\"s\",\"version\":\"0.3.0\"}},\
             \"inputs\":[{{\"byte_length\":1,\"kind\":\"source\",\"path\":\"src/Proofs/A.lex.tex\",\"sha256\":\"{source_sha}\"}},\
             {{\"byte_length\":1,\"kind\":\"lock\",\"path\":\"lexlean.lock\",\"sha256\":\"{l}\"}}],\
             \"language\":\"1.2\",\"lean_toolchain\":\"leanprover/lean4:v4.32.1\",\
             \"modules\":[{{\"lean_module\":\"HologramAi.Proofs.A\"}}]}}",
            b = "cc".repeat(32),
            l = "dd".repeat(32),
        )
        .into_bytes()
    }

    /// An attestation as LexLean writes it: canonical JSON, its ID first,
    /// one final LF, the ID derived from the rest.
    fn attest(manifest: &[u8], status: &str, result: &str) -> (String, Vec<u8>) {
        let body = format!(
            "\"build_id\":\"{b}\",\"build_manifest\":{{\"byte_length\":{n},\"sha256\":\"{m}\"}},\
             \"declarations\":[{{\"name\":\"HologramAi.Proofs.A.t\",\"result\":\"{result}\"}}],\
             \"lexlean\":{{\"executable_sha256\":\"{x}\",\"version\":\"0.3.0\"}},\"status\":\"{status}\",\
             \"toolchain\":{{\"lake\":{{\"version_output\":\"Lake version 5.0.0\"}},\"lean\":{{\"version_output\":\"Lean (version 4.32.1)\"}}}}}}",
            b = "cc".repeat(32),
            n = manifest.len(),
            m = sha256_hex(manifest),
            x = "aa".repeat(32),
        );
        let id = framed_sha256(
            "lexlean-attestation-v1",
            &[("attestation-body", format!("{{{body}").as_bytes())],
        );
        (
            id.clone(),
            format!("{{\"attestation_id\":\"{id}\",{body}\n").into_bytes(),
        )
    }

    #[test]
    fn an_unchanged_module_is_reused() {
        let m = manifest(&"11".repeat(32));
        let (id, att) = attest(&m, "verified", "ok");
        assert_eq!(binds(&id, &att, &m, &m, &tools()), Ok(()));
    }

    #[test]
    fn planted_a_changed_source_is_a_miss() {
        let kept = manifest(&"11".repeat(32));
        let now = manifest(&"12".repeat(32));
        let (id, att) = attest(&kept, "verified", "ok");
        assert!(binds(&id, &att, &kept, &now, &tools()).is_err());
        // and the key differs, so the store is not even asked
        assert_ne!(key(&tools(), "A", &kept), key(&tools(), "A", &now));
    }

    #[test]
    fn planted_a_tampered_attestation_is_a_miss() {
        let m = manifest(&"11".repeat(32));
        let (id, att) = attest(&m, "verified", "ok");
        let forged = String::from_utf8(att).unwrap().replace("\"ok\"", "\"ok \"");
        assert!(binds(&id, forged.as_bytes(), &m, &m, &tools()).is_err());
    }

    #[test]
    fn planted_a_failed_or_unverified_attestation_is_a_miss() {
        let m = manifest(&"11".repeat(32));
        let (id, att) = attest(&m, "verified", "error");
        assert!(binds(&id, &att, &m, &m, &tools()).is_err());
        let (id, att) = attest(&m, "failed", "ok");
        assert!(binds(&id, &att, &m, &m, &tools()).is_err());
    }

    #[test]
    fn planted_a_swapped_build_manifest_is_a_miss() {
        let m = manifest(&"11".repeat(32));
        let other = manifest(&"13".repeat(32));
        let (id, att) = attest(&m, "verified", "ok");
        assert!(binds(&id, &att, &other, &other, &tools()).is_err());
    }

    #[test]
    fn planted_another_toolchain_is_a_miss() {
        let m = manifest(&"11".repeat(32));
        let (id, att) = attest(&m, "verified", "ok");
        let mut other = tools();
        other.lexlean_sha256 = "ee".repeat(32);
        assert!(binds(&id, &att, &m, &m, &other).is_err());
        let mut other = tools();
        other.lean_version = "Lean (version 4.33.0)\n".into();
        assert!(binds(&id, &att, &m, &m, &other).is_err());
        let mut other = tools();
        other.sdk_image.push('0');
        assert_ne!(key(&tools(), "A", &m), key(&other, "A", &m));
    }

    #[test]
    fn planted_a_kept_id_that_is_not_its_attestation_is_a_miss() {
        let m = manifest(&"11".repeat(32));
        let (_, att) = attest(&m, "verified", "ok");
        assert!(binds(&"00".repeat(32), &att, &m, &m, &tools()).is_err());
    }
}
