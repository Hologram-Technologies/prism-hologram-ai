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

/// `GP-08`: A q4 Qwen2 model whose weights are given as ONNX MatMulNBits tensors, with f32 or f16 scales, runs on WebGPU from those bytes with no repack and gives the CPU reference's greedy tokens with every logit within tolerance
#[test]
fn gpu_gp_08() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let onnx = &result["onnx"];
    assert!(
        onnx.is_object(),
        "the run has no ONNX-layout models: {result}"
    );
    // The claim that a q4 export needs no repack rests on this: the reference
    // writes MatMulNBits' tensors from the operator's definition, and they are
    // the engine's own q4 bytes.
    assert_eq!(
        onnx["sameBytes"].as_bool(),
        Some(true),
        "MatMulNBits' q4 tensors differ from the engine's q4 block"
    );
    for (key, format) in [("q4", "q4"), ("q4f16", "q4f16")] {
        let model = &onnx[key];
        // Each weight reached the engine as two tensors in the format named,
        // so the f16 scales were read by the f16 kernels, not widened first.
        assert_eq!(
            model["formats"],
            serde_json::json!([format]),
            "{key}: the weights were not given as {format} parts"
        );
        reference_agrees(model, &format!("ONNX {key}"));
        blocks_agree(model, &format!("ONNX {key}"));
    }
}

/// `GP-09`: A q4 Qwen2 model whose embedding is a dense f32 or f16 matrix with the output head tied to it, as ONNX exports hold them, gives the CPU reference's greedy tokens with every logit within tolerance, with the head split into tiles when it is larger than one binding
#[test]
fn gpu_gp_09() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let dense = &result["dense"];
    assert!(
        dense.is_object(),
        "the run has no dense-embedding models: {result}"
    );
    // The limit is 15,000 bytes over 320 rows 128 wide: 29 rows a tile at four
    // bytes a number and 58 at two, so a head of one tile would mean the limit
    // was ignored, and neither count divides 320, so the last tile is short.
    assert_eq!(dense["tile"].as_u64(), Some(15_000), "the binding limit");
    for (key, tiles) in [("q4", 12), ("q4f16", 6)] {
        let model = &dense[key];
        assert_eq!(
            model["headTiles"].as_u64(),
            Some(tiles),
            "{key}: the head was not split into {tiles} tiles"
        );
        reference_agrees(model, &format!("dense {key}"));
        blocks_agree(model, &format!("dense {key}"));
    }
}

/// `GP-10`: A q4 Qwen2 model shaped as Qwen2.5's q4f16 ONNX export, with q4f16 weights, a dense f16 embedding and tied head, and norm weights and biases in f16, gives the CPU reference's greedy tokens with every logit within tolerance
#[test]
fn gpu_gp_10() {
    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let model = &result["exportShape"]["qwen2"];
    assert!(
        model.is_object(),
        "the run has no Qwen2.5-shaped model: {result}"
    );
    // Every norm weight and bias reached the engine in f16, and the biases
    // are there, so the f16 path is the one run.
    assert_eq!(model["vectorFormats"], serde_json::json!(["f16"]));
    assert_eq!(
        model["biases"].as_bool(),
        Some(true),
        "the model has biases"
    );
    reference_agrees(model, "Qwen2.5 q4f16 shape");
    blocks_agree(model, "Qwen2.5 q4f16 shape");
}

/// `GP-11`: The rotary frequencies of Llama 3's scaling are transformers' llama3 formula for Llama 3.2's parameters, and a Llama-shaped model with that scaling and no biases gives the CPU reference's greedy tokens with every logit within tolerance
#[test]
fn gpu_gp_11() {
    // Llama 3.2 1B: head size 64, base 5e5; rope type llama3 with factor 32,
    // low and high frequency factors 1 and 4, original context 8192.
    let (hd, base, factor, low, high, orig) = (64.0_f64, 500_000.0_f64, 32.0, 1.0, 4.0, 8192.0);
    let result = node(
        "crates/conformance/tests/gpu/yarn.mjs",
        &["llama3", "64", "500000", "32", "1", "4", "8192"],
    );
    assert_eq!(result["mscale"].as_f64(), Some(1.0), "no attention scale");
    let freqs = result["freqs"].as_array().expect("freqs");
    assert_eq!(freqs.len(), 32, "one frequency per pair");
    let mut bands = [0; 3];
    for (i, f) in freqs.iter().enumerate() {
        let theta = base.powf(-2.0 * i as f64 / hd);
        let wavelength = 2.0 * std::f64::consts::PI / theta;
        let want = if wavelength < orig / high {
            bands[0] += 1;
            theta
        } else if wavelength > orig / low {
            bands[2] += 1;
            theta / factor
        } else {
            bands[1] += 1;
            let smooth = (orig / wavelength - low) / (high - low);
            (1.0 - smooth) * theta / factor + smooth * theta
        };
        let got = f.as_f64().expect("a frequency");
        assert!(
            ((got - want) / want).abs() < 1e-6,
            "pair {i}: frequency {got}, formula {want}"
        );
    }
    assert_eq!(bands, [15, 3, 14], "kept, blended and slowed pairs");

    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let model = &result["exportShape"]["llama"];
    assert!(
        model.is_object(),
        "the run has no Llama-shaped model: {result}"
    );
    assert_eq!(
        model["biases"].as_bool(),
        Some(false),
        "Llama has no biases"
    );
    reference_agrees(model, "Llama 3 shape");
    blocks_agree(model, "Llama 3 shape");
}

/// `GP-12`: Every weight of onnx-community's q4f16 exports of Qwen2.5-1.5B, Qwen3-0.6B and Llama 3.2 1B is given its engine role by the node that reads it, with shapes checked against the config, what cannot be placed is refused by name, and a model placed this way gives the CPU reference's greedy tokens
#[test]
fn gpu_gp_12() {
    let result = node("crates/conformance/tests/gpu/roles.mjs", &[]);
    // (export, layers, width, heads, kv heads, head size, placed, f16 tensors)
    for (name, layers, d, heads, kv, hd, placed, f16) in [
        ("Qwen2.5-1.5B-Instruct", 28, 1536, 12, 2, 128, 338, 142),
        ("Qwen3-0.6B-ONNX", 28, 1024, 16, 8, 128, 310, 114),
        ("Llama-3.2-1B-Instruct-ONNX", 16, 2048, 32, 8, 64, 146, 34),
    ] {
        let x = &result["exports"][name];
        assert!(x.is_object(), "{name}: not placed: {result}");
        let dims = &x["dims"];
        for (key, want) in [
            ("n_layers", layers),
            ("d", d),
            ("n_heads", heads),
            ("n_kv_heads", kv),
            ("hd", hd),
        ] {
            assert_eq!(dims[key].as_u64(), Some(want), "{name}: {key}");
        }
        // Seven 4-bit matrices a layer, and the rest f16: two norms a layer,
        // three biases for Qwen2.5 or two head norms for Qwen3, the embedding
        // and the final norm.
        assert_eq!(x["placed"].as_u64(), Some(placed), "{name}: tensors placed");
        assert_eq!(
            x["formats"]["q4f16"].as_u64(),
            Some(7 * layers),
            "{name}: q4f16"
        );
        assert_eq!(x["formats"]["f16"].as_u64(), Some(f16), "{name}: f16");
        assert_eq!(x["head"].as_bool(), Some(false), "{name}: the head is tied");
        assert_eq!(x["embed"].as_str(), Some("model.embed_tokens.weight"));
    }
    // Qwen2.5 names its weights onnx::MatMul_N; the node placed it.
    assert_eq!(
        result["exports"]["Qwen2.5-1.5B-Instruct"]["q3"],
        serde_json::json!(["onnx::MatMul_9444_Q4", "onnx::MatMul_9444_scales"])
    );
    assert_eq!(
        result["exports"]["Llama-3.2-1B-Instruct-ONNX"]["rope"],
        serde_json::json!({"llama3": {"factor": 32, "low": 1, "high": 4, "orig": 8192}}),
        "Llama's cache confirms Llama 3 scaling"
    );
    for (case, start) in [
        ("missing", "missing: the graph has no l3.wq"),
        ("duplicate", "duplicate: l5.w_up"),
        (
            "unplaced",
            "unplaced: MatMulNBits /model/layers.2/mlp/extra_proj",
        ),
        (
            "zeroPoints",
            "precision: /model/layers.0/attn/k_proj/MatMul_Q4 carries zero points",
        ),
        (
            "block",
            "precision: /model/layers.1/attn/v_proj/MatMul_Q4 is 4 bits in blocks of 64",
        ),
        ("heads", "shape: /model/layers.0/attn/k_proj/MatMul_Q4 rows"),
        ("family", "family: gemma"),
        ("layers", "layer: /model/layers.27/"),
        ("rope", "rope: no rotary parameters agree"),
        (
            "interleaved",
            "rope: /model/layers.0/attn/q_rotary/RotaryEmbedding rotates interleaved",
        ),
    ] {
        let got = result["refusals"][case].as_str().unwrap_or_default();
        assert!(
            got.starts_with(start),
            "{case}: refused with {got:?}, not {start:?}"
        );
    }

    let result = node("crates/conformance/tests/gpu/run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    for key in ["qwen2", "llama"] {
        let model = &result["placed"][key];
        assert!(
            model.is_object(),
            "the run has no placed {key} model: {result}"
        );
        assert_eq!(
            model["sameAsModel"].as_bool(),
            Some(true),
            "{key}: the map placed other tensors than the model has"
        );
        reference_agrees(model, &format!("placed {key}"));
        blocks_agree(model, &format!("placed {key}"));
    }
}

/// The largest difference allowed between the encoder's single-precision
/// numbers and the reference's. SwiftShader measured 1.0e-6 on hidden states
/// and 9e-8 on label scores; a misplaced bias or norm moves them by tenths.
const ENCODER_TOLERANCE: f64 = 1e-4;

/// `GP-13`: A BERT encoder shaped as all-MiniLM-L6-v2's q4f16 ONNX export, with q4f16 weights, f16 biases, LayerNorms and embedding tables and token types, gives on WebGPU the CPU reference's hidden states and its mean-pooled unit-length sentence embedding within tolerance
#[test]
fn gpu_gp_13() {
    let result = node("crates/conformance/tests/gpu/encode.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the encoder did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    let runs = result["embeddings"].as_array().expect("the embeddings");
    let lengths: Vec<u64> = runs.iter().filter_map(|r| r["tokens"].as_u64()).collect();
    assert_eq!(lengths, [6, 60], "a short sentence and a long one");
    for run in runs {
        let n = &run["tokens"];
        let states = run["statesWorst"].as_f64().expect("statesWorst");
        assert!(
            states <= ENCODER_TOLERANCE,
            "{n} tokens: a hidden state differs by {states}"
        );
        let embedding = run["embeddingWorst"].as_f64().expect("embeddingWorst");
        assert!(
            embedding <= ENCODER_TOLERANCE,
            "{n} tokens: the embedding differs by {embedding}"
        );
        let cosine = run["cosine"].as_f64().expect("cosine");
        assert!(cosine > 0.99999, "{n} tokens: cosine {cosine}");
    }
}

/// `GP-14`: A DistilBERT classifier shaped as distilbert-base-uncased-finetuned-sst-2-english's q4f16 ONNX export, with no token types and an f16 dense head through ReLU, gives on WebGPU the CPU reference's label scores within tolerance and the same label
#[test]
fn gpu_gp_14() {
    let result = node("crates/conformance/tests/gpu/encode.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the encoder did not run: {}",
        result["error"]
    );
    let runs = result["classes"].as_array().expect("the classifications");
    assert_eq!(runs.len(), 2, "two sentences");
    // The fixture gives the two sentences different labels, so a head that
    // always answers one label fails.
    assert_ne!(
        runs[0]["reference"], runs[1]["reference"],
        "the fixture's sentences must differ in label; choose another seed"
    );
    for run in runs {
        let n = &run["tokens"];
        let margin = run["margin"].as_f64().expect("margin");
        assert!(
            margin > 2.0 * ENCODER_TOLERANCE,
            "{n} tokens: the fixture has a near tie ({margin})"
        );
        for key in ["logitsWorst", "probabilityWorst"] {
            let got = run[key].as_f64().expect(key);
            assert!(got <= ENCODER_TOLERANCE, "{n} tokens: {key} {got}");
        }
        assert_eq!(
            run["gpu"], run["reference"],
            "{n} tokens: the encoder picked another label"
        );
    }
}

/// `GP-15`: Tensors are kept in the browser once each by their BLAKE3 address: a second model stores only the tensors the first lacks, a return visit with the network off reads every tensor back admitted again and runs to the reference's tokens, and a stored tensor whose bytes changed is refused and removed
#[test]
fn gpu_gp_15() {
    let result = node("crates/conformance/tests/gpu/store-run.mjs", &[]);
    let visit = &result["visit"];
    let count = |at: &serde_json::Value, key: &str| at[key].as_u64().unwrap_or(u64::MAX);
    // Model A: 40 blocks, all distinct. B shares all but five.
    assert_eq!(count(&visit["a"], "blocks"), 40, "A's blocks");
    assert_eq!(
        count(&visit["a"], "distinct"),
        40,
        "A's blocks are distinct"
    );
    assert_eq!(count(&visit["b"], "newBlocks"), 5, "B's new blocks");
    let shared = count(&visit["b"], "sharedBytes");
    assert_eq!(
        count(&visit["afterA"], "stored"),
        40,
        "A stores every block"
    );
    assert_eq!(
        count(&visit["afterB"], "stored"),
        45,
        "B stores only its five new blocks"
    );
    assert_eq!(
        count(&visit["afterB"], "held"),
        35,
        "B holds the 35 it shares"
    );
    assert_eq!(
        count(&visit["afterB"], "heldBytes"),
        shared,
        "the bytes not written again are the shared ones"
    );
    assert_eq!(
        count(&visit["afterAgain"], "stored"),
        45,
        "A again stores nothing"
    );
    assert_eq!(count(visit, "held"), 45, "45 files held");

    let back = &result["back"];
    assert_eq!(
        back["offline"].as_bool(),
        Some(true),
        "the second visit has no network"
    );
    assert_eq!(count(back, "missing"), 0, "every block of A read back");
    assert_eq!(count(back, "admitted"), 40, "every block admitted again");
    assert_eq!(count(&back["counts"], "read"), 40, "40 reads");
    let steps = back["steps"].as_array().expect("the steps");
    assert_eq!(steps.len(), 16, "sixteen steps");
    for (i, step) in steps.iter().enumerate() {
        let worst = step["worst"].as_f64().expect("worst");
        assert!(
            worst <= LOGIT_TOLERANCE,
            "step {i}: a logit differs by {worst}"
        );
        let margin = step["margin"].as_f64().expect("margin");
        assert!(
            margin > 2.0 * LOGIT_TOLERANCE,
            "step {i}: a near tie ({margin})"
        );
        assert_eq!(step["gpu"], step["reference"], "step {i}: another token");
    }
    assert_eq!(
        back["tampered"].as_bool(),
        Some(true),
        "a changed block is not handed back"
    );
    assert!(back["afterTamper"].is_null(), "a changed block is removed");
    assert_eq!(count(&back["counts"], "refused"), 1, "one refusal");
    let prefix = back["wrongPrefix"].as_str().unwrap_or_default();
    assert!(
        prefix.contains("is not a blake3: label"),
        "a sha256: label: {prefix:?}"
    );
}

/// `GP-16`: A prompt's computed prefix, kept on the GPU and restored into the same engine after its cache was overwritten or into a fresh engine of the same model, gives bit for bit the logits and tokens of computing the whole prompt
#[test]
fn gpu_gp_16() {
    let result = node("crates/conformance/tests/gpu/prefix-run.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    for key in ["qwen2", "llama"] {
        let r = &result["results"][key];
        assert!(r.is_object(), "no {key} run: {result}");
        assert_eq!(
            r["restoredAt"].as_u64(),
            Some(40),
            "{key}: restored at the prefix's end"
        );
        assert_eq!(
            r["positions"].as_u64(),
            Some(16),
            "{key}: 16 positions compared"
        );
        // Bits, not a tolerance: the restored cache is the computed one.
        assert_eq!(
            r["differingB"].as_u64(),
            Some(0),
            "{key}: logits after restoring into the same engine differ"
        );
        assert_eq!(
            r["differingC"].as_u64(),
            Some(0),
            "{key}: logits after restoring into a fresh engine differ"
        );
        assert_eq!(
            r["resumed"].as_array().map(Vec::len),
            Some(12),
            "{key}: twelve tokens"
        );
        assert_eq!(
            r["resumed"], r["straight"],
            "{key}: the resumed run picked other tokens"
        );
        let too_long = r["tooLong"].as_str().unwrap_or_default();
        assert!(
            too_long.starts_with("prefix: 57 positions asked, 56 computed"),
            "{key}: {too_long:?}"
        );
        let other = r["otherShape"].as_str().unwrap_or_default();
        assert!(
            other.starts_with("prefix: saved from 3 layers"),
            "{key}: {other:?}"
        );
    }
}

/// `GP-17`: Every weight of Xenova's q4f16 exports of all-MiniLM-L6-v2 and distilbert-base-uncased-finetuned-sst-2-english is given its encoder role by the node that reads it, with shapes, activation and labels checked against the config, what cannot be placed is refused by name, and an encoder placed this way gives the CPU reference's embedding and labels
#[test]
fn gpu_gp_17() {
    let result = node("crates/conformance/tests/gpu/encoder-roles.mjs", &[]);
    for (name, placed, f16, width) in [
        ("all-MiniLM-L6-v2", 101, 65, 384),
        (
            "distilbert-base-uncased-finetuned-sst-2-english",
            104,
            68,
            768,
        ),
    ] {
        let x = &result["exports"][name];
        assert!(x.is_object(), "{name}: not placed: {result}");
        // Six layers of six 4-bit matrices, the rest f16.
        assert_eq!(x["placed"].as_u64(), Some(placed), "{name}: tensors placed");
        assert_eq!(x["formats"]["q4f16"].as_u64(), Some(36), "{name}: q4f16");
        assert_eq!(x["formats"]["f16"].as_u64(), Some(f16), "{name}: f16");
        assert_eq!(x["dims"]["d"].as_u64(), Some(width), "{name}: width");
        assert_eq!(x["dims"]["n_layers"].as_u64(), Some(6), "{name}: layers");
    }
    assert_eq!(
        result["exports"]["distilbert-base-uncased-finetuned-sst-2-english"]["labels"],
        serde_json::json!({"0": "NEGATIVE", "1": "POSITIVE"})
    );
    assert_eq!(
        result["exports"]["distilbert-base-uncased-finetuned-sst-2-english"]["head"].as_str(),
        Some("classifier.weight")
    );
    for (case, start) in [
        ("activation", "activation: gelu_new"),
        ("family", "family: roberta"),
        ("missing", "missing: the graph has no l4.b2"),
        ("transB", "shape: /classifier/Gemm is not y = x W^T + b"),
        (
            "labels",
            "shape: /classifier/Gemm rows is 2, the config gives 3",
        ),
        (
            "unplaced",
            "unplaced: MatMulNBits /encoder/layer.1/extra/dense",
        ),
        (
            "width",
            "shape: /encoder/layer.0/intermediate/dense/MatMul_Q4 rows",
        ),
    ] {
        let got = result["refusals"][case].as_str().unwrap_or_default();
        assert!(
            got.starts_with(start),
            "{case}: refused with {got:?}, not {start:?}"
        );
    }

    let result = node("crates/conformance/tests/gpu/encode.mjs", &[]);
    assert!(
        result.get("error").is_none(),
        "the encoder did not run: {}",
        result["error"]
    );
    for key in ["bert", "distilbert"] {
        let placed = &result["placed"][key];
        assert!(placed.is_object(), "no placed {key}: {result}");
        assert_eq!(
            placed["sameAsModel"].as_bool(),
            Some(true),
            "{key}: the map placed other tensors than the model has"
        );
        for run in placed["runs"].as_array().expect("runs") {
            let worst = run["worst"].as_f64().expect("worst");
            assert!(worst <= ENCODER_TOLERANCE, "{key}: differs by {worst}");
            assert_eq!(run["gpu"], run["reference"], "{key}: another label");
        }
    }
}

/// The largest difference allowed between the CPU engine's logits and the
/// reference's. Both sum in double precision; the engine keeps activations as
/// single precision between steps, as the GPU does. Measured at most 7.3e-7.
const CPU_TOLERANCE: f64 = 1e-5;

/// `GP-18`: With WebGPU off, the CPU engine runs q4 Qwen2, Qwen2.5's, Llama's and Qwen3's q4f16 export shapes from their own bytes and gives the CPU reference's greedy tokens with every logit within 1e-5, by steps, in a run and after restoring a prefix
#[test]
fn gpu_gp_18() {
    let result = node("crates/conformance/tests/gpu/cpu-run.mjs", &[]);
    // An adapter offered is a failure, not a skip: the claim is about the
    // engine with WebGPU off.
    assert!(
        result.get("error").is_none(),
        "the CPU engine did not run: {}",
        result["error"]
    );
    assert!(
        matches!(
            result["webgpu"].as_str(),
            Some("no adapter" | "no navigator.gpu")
        ),
        "WebGPU was not off: {}",
        result["webgpu"]
    );
    for key in ["q4", "qwen25", "llama", "qwen3"] {
        let model = &result[key];
        assert!(model.is_object(), "no {key} run: {result}");
        let steps = model["steps"].as_array().expect("the steps");
        assert_eq!(
            steps.len(),
            16,
            "{key}: six prompt tokens and ten generated"
        );
        for (i, step) in steps.iter().enumerate() {
            let worst = step["worst"].as_f64().expect("worst");
            assert!(
                worst <= CPU_TOLERANCE,
                "{key}: step {i}: a logit differs by {worst}"
            );
            let margin = step["margin"].as_f64().expect("margin");
            assert!(
                margin > 2.0 * CPU_TOLERANCE,
                "{key}: step {i}: a near tie ({margin})"
            );
            assert_eq!(
                step["gpu"], step["reference"],
                "{key}: step {i}: another token"
            );
        }
        assert_eq!(
            model["longPositions"].as_u64(),
            Some(140),
            "{key}: 140 positions"
        );
        let long = model["longWorst"].as_f64().expect("longWorst");
        assert!(
            long <= CPU_TOLERANCE,
            "{key}: over 140 positions a logit differs by {long}"
        );
        assert_eq!(
            model["together"].as_array().map(Vec::len),
            Some(12),
            "{key}: twelve tokens"
        );
        assert_eq!(
            model["together"], model["stepwise"],
            "{key}: run() picked other tokens"
        );
        assert_eq!(
            model["resumed"], model["straight"],
            "{key}: the restored prefix continued differently"
        );
    }
}

/// `GP-19`: The engine page reads an ONNX export named on the hub from its own files: each small file checked against the record, the graph imported through the module with the kernel admitting every span, operator, file and 4-bit weight, every weight placed by the node that reads it, and it answers as the reference does on WebGPU and on the CPU, and stops where a file is not the record's
#[test]
fn gpu_gp_19() {
    // The journey runs the ONNX cases in a browser (`just page-journey`, as
    // in GP-03 and GP-07); this test reads that it does, and that the page
    // takes the path it checks.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let page = read("site/app/app.mjs");
    for wanted in [
        "onnx: asked.get(\"onnx\")",
        "await loadOnnxModel({ kernel, ask, device, started });",
        "compileGuest(resolveArchive(kernel.archive).guest.content)",
        "createCpuEngine({ dims, tensors, capacity: CAPACITY })",
        "const stops = session.stops ??",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/app.mjs no longer has `{wanted}`"
        );
    }
    let loader = read("site/app/engine/onnx-load.mjs");
    for wanted in [
        "m.callWith(\"onnx_start\"",
        "m.callWith(\"onnx_take\", chunk)",
        "m.call(\"onnx_finish\")",
        "roles(table, config)",
        "is not the file the record names",
        "crypto.subtle.digest(\"SHA-1\", blob)",
    ] {
        assert!(
            loader.contains(wanted),
            "site/app/engine/onnx-load.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "buildOnnxFixture()",
        "Object.defineProperty(Navigator.prototype, \"gpu\"",
        "onnx ${name}: the page's tokens",
        "onnx ${name}: ran on ${run.device}, expected ${device}",
        "onnx flip: expected the kernel to refuse the graph file",
        "onnx tokenizer: expected to stop at tokenizer.json",
        "onnx gitblob: config.json was not checked by its Git blob SHA-1",
        "onnx gitwrong: expected to stop at generation_config.json",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}
