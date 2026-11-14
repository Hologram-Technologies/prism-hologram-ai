//! The `gpu` suite: one test per scenario in `features/suites/gpu.feature`.
//!
//! The engine is the page's JavaScript and WGSL, so these tests run it with
//! Node, and in a browser through the Playwright and Chromium the SDK image
//! carries. They run in the image, as the rest of the gate does.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/conformance is two below the root")
        .to_path_buf()
}

fn node(script: &str, args: &[&str]) -> serde_json::Value {
    let ran = Command::new("node")
        .arg(root().join(script))
        .arg(root())
        .args(args)
        .output()
        .expect("node runs");
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success(),
        "{script} failed:\n{stdout}{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let line = stdout.lines().last().unwrap_or_default();
    serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("{script} printed no JSON ({e}):\n{stdout}"))
}

/// The largest difference allowed between a logit the GPU computes in single
/// precision and the reference's double-precision logit. The first runs on
/// SwiftShader measured at most 6.6e-4 on logits spread over about 6 to 8.
const LOGIT_TOLERANCE: f64 = 2e-3;

/// `GP-01`: The WebGPU forward pass of a ternary Qwen3 model gives the same greedy tokens as the CPU reference, with every logit within a stated tolerance
#[test]
fn gpu_gp_01() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    // No adapter is a failure, not a skip: a test that cannot see a GPU has
    // checked nothing.
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let steps = result["steps"].as_array().expect("the steps");
    assert_eq!(steps.len(), 16, "six prompt tokens and ten generated");
    for (i, step) in steps.iter().enumerate() {
        let worst = step["worst"].as_f64().expect("worst");
        let margin = step["margin"].as_f64().expect("margin");
        assert!(
            worst <= LOGIT_TOLERANCE,
            "step {i}: a logit differs by {worst} from the reference"
        );
        // Equal picks are required only where the reference's best two
        // logits are further apart than twice the tolerance; closer than that,
        // rounding may pick either. The fixture keeps every step outside that.
        assert!(
            margin > 2.0 * LOGIT_TOLERANCE,
            "step {i}: the fixture has a near tie ({margin}); choose another seed"
        );
        assert_eq!(
            step["gpu"], step["reference"],
            "step {i}: the engine picked a different token from the reference"
        );
    }
}

/// `GP-02`: The rotary frequencies are the YaRN formula, with the pairs that keep their frequency and the pairs that are scaled computed from the model base, original context and head size
#[test]
fn gpu_gp_02() {
    // Bonsai 1.7B's own parameters, from its manifest: head size 128, base
    // 1e6, YaRN factor 4 over an original context of 8192.
    let (hd, base, factor, orig) = (128.0_f64, 1_000_000.0_f64, 4.0_f64, 8192.0_f64);
    let result = node(
        "crates/conformance/tests/gpu/yarn.mjs",
        &["128", "1000000", "4", "8192"],
    );

    // The pair at which a rotation makes `turns` turns over the original
    // context; YaRN keeps pairs below the 32-turn one and scales those above
    // the 1-turn one.
    let pair =
        |turns: f64| hd * (orig / (turns * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let low = pair(32.0).floor().max(0.0);
    let high = pair(1.0).ceil().min(hd - 1.0);
    assert_eq!((low, high), (17.0, 34.0), "the model card's bounds");
    assert_eq!(result["low"].as_f64(), Some(low), "low bound");
    assert_eq!(result["high"].as_f64(), Some(high), "high bound");

    let mscale = 1.0 + 0.1 * factor.ln();
    let got = result["mscale"].as_f64().expect("mscale");
    assert!((got - mscale).abs() < 1e-12, "mscale {got} is not {mscale}");

    let freqs = result["freqs"].as_array().expect("freqs");
    assert_eq!(freqs.len(), 64, "one frequency per pair");
    for (i, f) in freqs.iter().enumerate() {
        let i = i as f64;
        let original = base.powf(-2.0 * i / hd);
        let ramp = ((i - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
        let want = (original / factor) * ramp + original * (1.0 - ramp);
        let got = f.as_f64().expect("a frequency");
        assert!(
            ((got - want) / want).abs() < 1e-6,
            "pair {i}: frequency {got}, formula {want}"
        );
    }
    // Below the low bound a pair keeps its trained frequency; above the high
    // bound it turns at a quarter of it.
    let at = |i: usize| freqs[i].as_f64().expect("a frequency");
    assert!(
        ((at(16) - base.powf(-32.0 / hd)) / at(16)).abs() < 1e-6,
        "pair 16 keeps its frequency"
    );
    assert!(
        ((at(40) - base.powf(-80.0 / hd) / factor) / at(40)).abs() < 1e-6,
        "pair 40 is scaled by the factor"
    );
}
