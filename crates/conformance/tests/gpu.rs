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
// Over 140 positions f32 rounding in the cache accumulates: the single
// workgroup attention and the split one both measured 2.53e-3 from the
// reference. A combine that drops chunks measured 5.6, three orders above.
const LONG_LOGIT_TOLERANCE: f64 = 5e-3;

// The engine against the reference for one model's results: sixteen steps
// with every logit within tolerance and the same pick, and 140 positions,
// three attention chunks, within the long tolerance.
fn reference_agrees(model: &serde_json::Value, label: &str) {
    let steps = model["steps"].as_array().expect("the steps");
    assert_eq!(
        steps.len(),
        16,
        "{label}: six prompt tokens and ten generated"
    );
    for (i, step) in steps.iter().enumerate() {
        let worst = step["worst"].as_f64().expect("worst");
        let margin = step["margin"].as_f64().expect("margin");
        assert!(
            worst <= LOGIT_TOLERANCE,
            "{label}: step {i}: a logit differs by {worst} from the reference"
        );
        // Equal picks are required only where the reference's best two
        // logits are further apart than twice the tolerance; closer than that,
        // rounding may pick either. The fixture keeps every step outside that.
        assert!(
            margin > 2.0 * LOGIT_TOLERANCE,
            "{label}: step {i}: the fixture has a near tie ({margin}); choose another seed"
        );
        assert_eq!(
            step["gpu"], step["reference"],
            "{label}: step {i}: the engine picked a different token from the reference"
        );
    }
    // Attention splits the cache into 64-position chunks and combines them;
    // a context of one chunk would leave the combine unchecked.
    assert_eq!(
        model["longPositions"].as_u64(),
        Some(140),
        "{label}: the long run covers three attention chunks"
    );
    let long_worst = model["longWorst"].as_f64().expect("longWorst");
    assert!(
        long_worst <= LONG_LOGIT_TOLERANCE,
        "{label}: past one attention chunk, a logit differs by {long_worst} from the reference"
    );
}

// A prompt run in blocks against one position at a time, for one model.
fn blocks_agree(model: &serde_json::Value, label: &str) {
    let runs = model["batched"].as_array().expect("the batched runs");
    assert_eq!(
        runs.len(),
        2,
        "{label}: a short prompt and one longer than a submission"
    );
    for run in runs {
        let length = run["promptLength"].as_u64().expect("prompt length");
        let stepwise = run["stepwise"].as_array().expect("stepwise tokens");
        let together = run["together"].as_array().expect("batched tokens");
        assert_eq!(stepwise.len(), 12, "{label}: twelve tokens one at a time");
        assert_eq!(
            together, stepwise,
            "{label}: a {length}-token prompt: batched tokens differ from one at a time"
        );
        // Every prompt position and every generated one ran exactly once.
        assert_eq!(
            run["position"].as_u64(),
            Some(length + 12 - 1),
            "{label}: a {length}-token prompt: positions run"
        );
    }
    // The longer prompt crosses a submission's 32 slots and attention's
    // 64-position chunks, so both boundaries are exercised, not only the first.
    assert!(runs.iter().any(|r| r["promptLength"].as_u64() > Some(64)));
}

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
    reference_agrees(&result, "ternary Qwen3");
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

/// `GP-04`: Running many positions in one submission, with each pick written on the GPU and read by the next position, gives the same tokens as running them one at a time
#[test]
fn gpu_gp_04() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    blocks_agree(&result, "ternary Qwen3");
}

/// `GP-05`: The WebGPU forward pass of a q4 Qwen2 model, with attention biases and no per-head norm, gives the same greedy tokens as the CPU reference with every logit within tolerance past one attention chunk, and a prompt run in blocks gives the same tokens as one position at a time
#[test]
fn gpu_gp_05() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    // The generated model carries q4 weights, q/k/v biases and no per-head
    // norm, and an output head of its own; the reference decodes all of it
    // without the engine's code.
    let qwen2 = &result["qwen2"];
    assert!(qwen2.is_object(), "the run has no q4 Qwen2 model: {result}");
    reference_agrees(qwen2, "q4 Qwen2");
    blocks_agree(qwen2, "q4 Qwen2");
}

/// `GP-03`: The engine page loads a weight container through the verified kernel and answers as the reference does, and stops where a container is tampered with
#[test]
fn gpu_gp_03() {
    // The journey needs the holo and the engine this workspace built, so it
    // runs as `just page-journey`, after `app` and `engine` in the gate, and
    // the publishing workflow runs that same recipe before it deploys. This
    // test reads that wiring: a journey nobody calls proves nothing.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let justfile = read("Justfile");
    let vv = justfile
        .lines()
        .find(|l| l.starts_with("vv:"))
        .expect("the vv recipe");
    let steps: Vec<&str> = vv.trim_start_matches("vv:").split_whitespace().collect();
    let at = |name: &str| {
        steps
            .iter()
            .position(|s| *s == name)
            .unwrap_or_else(|| panic!("vv does not run {name}: {vv}"))
    };
    assert!(
        at("page-journey") > at("app") && at("page-journey") > at("engine"),
        "vv must run page-journey after app and engine: {vv}"
    );
    let recipe = &justfile[justfile
        .find("\npage-journey assembled=\"\":")
        .expect("the page-journey recipe")..];
    let recipe = &recipe[..recipe[1..].find("\n\n").map_or(recipe.len(), |i| i + 1)];
    assert!(
        recipe.contains("node crates/conformance/tests/gpu/journey.mjs \"$site\""),
        "page-journey must run the journey on the assembled site:\n{recipe}"
    );
    for refusal in [
        "has no holo: run just app",
        "has no engine: run just engine",
    ] {
        assert!(
            recipe.contains(refusal),
            "page-journey must refuse, not skip, when a part is missing: {refusal}"
        );
    }

    let pages = read(".github/workflows/pages.yml");
    let journey_step = pages
        .find("just page-journey site")
        .expect("pages.yml runs the page-journey recipe");
    assert!(
        !pages.contains("tests/gpu/journey.mjs"),
        "pages.yml must call the recipe, not a copy of what it runs"
    );
    assert!(
        journey_step
            > pages
                .find("name: Assemble the page")
                .expect("the assemble step")
            && journey_step
                < pages
                    .find("actions/deploy-pages")
                    .or_else(|| pages.find("deploy"))
                    .expect("a deploy step"),
        "pages.yml must run the journey after assembling the page and before deploying it"
    );

    // The journey's own verdicts: each says where the page must have stopped.
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "flipStop?.step !== \"admit\"",
        "flipStop?.subject !== `${victim.name} label and length`",
        "rootStop?.step !== \"root\"",
        "got.some((t, i) => t !== good.expected[i])",
        "the tied output head was not confirmed",
        "process.exit(1)",
        "No browser is a failure, never a skip",
        "the prompt box started with text",
        "await page.press(\"#prompt\", \"Enter\");",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks: {wanted}"
        );
    }
}

/// `GP-06`: The engine page stops with stale kernel before asking the kernel anything when the kernel it verified is not the one its deploy lists
#[test]
fn gpu_gp_06() {
    // The journey runs the stale case in a browser (`just page-journey`, as in
    // GP-03); this test reads that it does, and the premises it rests on.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };

    // The listing is fresh only because the service worker never answers it.
    // If the frame's worker starts caching precache.json, this pin can go
    // stale with the kernel, and this test must say so.
    let worker = read("site/sw.js");
    assert!(
        worker.contains("if (url.pathname.endsWith(\"/precache.json\")) return;"),
        "site/sw.js (the template's frame, not this repository's) no longer passes /precache.json          to the network, so the page's stale-kernel check (GP-06) rests on a listing that can now go          stale with the kernel; see UOR-Foundation/template#83, and choose a new source for the pin          in site/app/app.mjs before taking this frame"
    );

    // The page compares before it asks the kernel anything.
    let page = read("site/app/app.mjs");
    let at = |needle: &str| {
        page.find(needle)
            .unwrap_or_else(|| panic!("app.mjs no longer has: {needle}"))
    };
    assert!(page.contains("fetch(\"../precache.json\", { cache: \"no-store\" })"));
    let load = at("async function load()");
    let compare = load
        + page[load..]
            .find("await sameKernel(holo, listed);")
            .expect("load compares the kernel");
    let first_question = load
        + page[load..]
            .find("asker(kernel, checks)")
            .expect("load builds the asker");
    assert!(
        compare < first_question,
        "the page must compare the kernel with the deploy's listing before it can ask a question"
    );

    // The listing exists wherever the journey runs.
    let justfile = read("Justfile");
    assert!(justfile.contains(
        "node bootstrap/precache.mjs \"$site\"
    fi"
    ));
    let pages = read(".github/workflows/pages.yml");
    let listed = pages
        .find("node bootstrap/precache.mjs site")
        .expect("pages.yml writes precache.json");
    let journey_step = pages
        .find("just page-journey site")
        .expect("pages.yml runs the journey");
    assert!(
        listed < journey_step,
        "pages.yml must write precache.json before the journey"
    );

    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "stale: { container: fixture.container, sha256: fixture.sha256, staleKernel: true }",
        "f.sha256 = \"0\".repeat(64)",
        "staleStop?.subject !== \"kernel is the one this deploy lists\"",
        "!/^stale kernel/.test(stale.error ?? \"\")",
        "the page went on past the kernel",
        "the kernel was not compared with the deploy's listing",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks: {wanted}"
        );
    }
}

/// `GP-07`: A page file held in the cache from an earlier deploy is removed and the page reloads once before the engine is imported, and a file still stale after that stops the page
#[test]
fn gpu_gp_07() {
    // The journey runs both cases in a browser (`just page-journey`, as in
    // GP-03); this test reads that it does, and the premises it rests on.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };

    // The check reaches a visitor holding an earlier deploy only because the
    // worker fetches the page itself from the network first. If the frame's
    // worker starts serving index.html from its cache, the check goes stale
    // with everything else, and this test must say so.
    let worker = read("site/sw.js");
    assert!(
        worker.contains("const isPage = event.request.mode === \"navigate\" || url.pathname.endsWith(\"/\") || url.pathname.endsWith(\"/index.html\");")
            && worker.contains("const fresh = await fetch(event.request);
        if (fresh.ok && cache) await cache.put(\"./\", fresh.clone());"),
        "site/sw.js no longer fetches the page from the network first; the stale-file check in site/app/index.html rests on that (UOR-Foundation/template#83)"
    );
    assert!(
        worker.contains(".filter((name) => name.startsWith(\"prism-\"))"),
        "the worker's caches are no longer named prism-*; site/app/index.html looks for them by that name"
    );

    // The comparison is in the page, ahead of the engine: the engine is
    // imported only on the branch where nothing stale was found.
    let page = read("site/app/index.html");
    let check = page.find("fetch(\"../precache.json\", { cache: \"no-store\" })");
    let import = page.find("await import(\"./app.mjs\");");
    assert!(
        matches!((check, import), (Some(c), Some(i)) if c < i),
        "site/app/index.html must read the deploy's listing before it imports the engine"
    );
    assert!(
        !page.contains("src=\"./app.mjs\""),
        "the engine must not be loaded by a script tag, which runs whatever the cache holds"
    );
    for wanted in [
        "if (held !== want) {",
        "await cache.delete(request);",
        "location.reload();",
        "Stopped: stale page",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/index.html no longer has `{wanted}`"
        );
    }

    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "caches.open(\"prism-earlier\")",
        "earlier: the engine was not imported after the stale file was removed",
        "earlier: the earlier deploy's engine file is still in the cache",
        "earlier: a cached file that is this deploy's was removed",
        "expected one reload",
        "again: the engine was imported with a stale file still served after a reload",
        "again: expected \"Stopped: stale page\" and no Load button",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}
