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

// Eight scenarios read the one engine run `run.mjs` makes. It takes no input
// but this tree, so it runs once and each scenario asserts on that output.
// Run eight times, four at once on the gate's four cores, it took 22 to 28
// of the gate's 45 minutes.
fn run() -> &'static serde_json::Value {
    static RUN: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    RUN.get_or_init(|| node("crates/conformance/tests/gpu/run.mjs", &[]))
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
    let result = run();
    // No adapter is a failure, not a skip: a test that cannot see a GPU has
    // checked nothing.
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    let errors = result["errors"].as_array().expect("the error list");
    assert!(errors.is_empty(), "WebGPU reported errors: {errors:?}");
    reference_agrees(result, "ternary Qwen3");
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
    let result = run();
    assert!(
        result.get("error").is_none(),
        "the engine did not run: {}",
        result["error"]
    );
    blocks_agree(result, "ternary Qwen3");
}

/// `GP-05`: The WebGPU forward pass of a q4 Qwen2 model, with attention biases and no per-head norm, gives the same greedy tokens as the CPU reference with every logit within tolerance past one attention chunk, and a prompt run in blocks gives the same tokens as one position at a time
#[test]
fn gpu_gp_05() {
    let result = run();
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
    assert!(
        page.contains("const held = blake3Hex(holo);") && page.contains("return entry.blake3;"),
        "the kernel is no longer compared with the listing by BLAKE3"
    );
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
      node bootstrap/listing-blake3.mjs \"$site\"
    fi"
    ));
    let pages = read(".github/workflows/pages.yml");
    let listed = pages
        .find(
            "node bootstrap/precache.mjs site
          node bootstrap/listing-blake3.mjs site",
        )
        .expect("pages.yml writes precache.json and names its files by BLAKE3");
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
        "f.blake3 = \"0\".repeat(64)",
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
    // Cached files and the listing are compared by BLAKE3, the one address
    // Hologram AI names its own files by.
    assert!(
        !page.contains("SHA-256"),
        "site/app/index.html compares cached files by SHA-256"
    );
    // The listing is the frame's (pinned by frame.lock); this repository adds
    // each file's BLAKE3 to it, and the page reads only that.
    let listing = read("bootstrap/listing-blake3.mjs");
    assert!(
        listing.contains("file.blake3 = blake3Hex(await readFile(join(site, file.path)));"),
        "bootstrap/listing-blake3.mjs no longer names each listed file by its BLAKE3"
    );
    for wanted in [
        "const held = blake3Hex(new Uint8Array(await (await cache.match(request)).arrayBuffer()));",
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
    let result = run();
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
    let result = run();
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
    let result = run();
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

    let result = run();
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
    // Which rotary parameters Llama's cache confirms is EN-59's, in integers.
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

    let result = run();
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
        ("floatIndex", "unplaced: Gather /embeddings/Gather"),
    ] {
        let got = result["refusals"][case].as_str().unwrap_or_default();
        assert!(
            got.starts_with(start),
            "{case}: refused with {got:?}, not {start:?}"
        );
    }
    // MiniLM's int64 position ids, read by /embeddings/Gather, are an index,
    // not a weight: the roles place the encoder with them in the graph.
    assert!(
        result["indexConstant"].is_null(),
        "an int64 constant was refused: {}",
        result["indexConstant"]
    );

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
        "const generate = await pipeline(\"text-generation\", MODEL.onnx, {",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/app.mjs no longer has `{wanted}`"
        );
    }
    let module = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "compileGuest(resolveArchive(kernel.archive).guest.content)",
        "startIntegerEngine({ module, ...raw, capacity: CAPACITY })",
        "const stops = session.stops ??",
    ] {
        assert!(
            module.contains(wanted),
            "site/app/engine/pipeline.mjs no longer has `{wanted}`"
        );
    }
    let loader = read("site/app/engine/onnx-load.mjs");
    for wanted in [
        "m.callWith(encoder ? \"onnx_start_encoder\" : \"onnx_start\", unhex(graph.blake3),",
        "m.callWith(\"onnx_take\", chunk)",
        "m.call(\"onnx_finish\")",
        "roles(table, config)",
        "is not the file the record names",
        "const digest = blake3(bytes);",
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
        "onnx good: config.json was not checked by its BLAKE3",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-20`: The chat templates of Qwen2.5-1.5B's, Qwen3-0.6B's and Llama 3.2 1B's exports render, from the model's own tokenizer_config.json, exactly as Python's jinja2 renders them configured as transformers configures it, including a template's own error, and a template using a tag, filter, test or method outside what the engine follows is refused by name
#[test]
fn gpu_gp_20() {
    let result = node("crates/conformance/tests/gpu/templates.mjs", &[]);
    for model in [
        "Qwen2.5-1.5B-Instruct",
        "Qwen3-0.6B-ONNX",
        "Llama-3.2-1B-Instruct-ONNX",
        // Written to reach what the exports' templates do not: tags with no
        // dash under lstrip_blocks and trim_blocks, a tag opting out with a
        // plus, a trailing newline, nested loops.
        "synthetic whitespace",
    ] {
        let r = &result["results"][model];
        assert_eq!(
            r["cases"].as_u64(),
            Some(20),
            "{model}: twenty conversations"
        );
        let differing = r["differing"].as_array().expect("the differing cases");
        assert!(
            differing.is_empty(),
            "{model}: rendered other text than jinja2: {differing:?}"
        );
    }
    for (case, start) in [
        ("macro", "template: {% macro %} is not one"),
        ("include", "template: {% include %} is not one"),
        ("filter", "template: the filter upper is not one"),
        ("method", "template: the string method title is not one"),
        ("test", "template: the test odd is not one"),
        ("unclosed", "template: a template that ends before"),
        ("raise", "no tools here"),
    ] {
        let got = result["refusals"][case].as_str().unwrap_or_default();
        assert!(got.starts_with(start), "{case}: {got:?}, not {start:?}");
    }
    // The page frames an ONNX export's turn with the model's own template.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let loader = read("site/app/engine/onnx-load.mjs");
    assert!(
        loader.contains("await small(\"tokenizer_config.json\", true)"),
        "the template is not read from the model's files"
    );
    assert!(
        loader.contains("render: compileTemplate(source)"),
        "the template is not compiled"
    );
    let page = read("site/app/engine/pipeline.mjs");
    assert!(page.contains("session.chat.render({ ...session.chat.specials, messages, add_generation_prompt, enable_thinking: false"), "the page does not frame a turn with the template");
    assert!(page.contains("const framing = (input) => render(Array.isArray(input) ? input : [{ role: \"user\", content: input }]);"), "the page does not frame a user's text as one user message");
}

/// `GP-21`: The engine page keeps an ONNX export's files in the browser, each under the BLAKE3 the hub's record names for it, and a later visit reads them from there with the origin serving none, each file's BLAKE3 derived again as it is read and its bytes checked against the record and imported through the kernel exactly as a download's are; a kept file that is not its BLAKE3 is refused, and the load forgets every file it read, so the next visit downloads them again
#[test]
fn gpu_gp_21() {
    // The journey runs the four visits in a browser (`just page-journey`, as
    // in GP-19); this test reads that it does, and that the page reads the
    // kept files through the same loader and checks a download takes.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let kept = read("site/app/engine/files.mjs");
    for wanted in [
        "m.callWith(\"hash_take\", chunk, id);",
        "if (digest() !== local.named) controller.error(",
        "return bytes.size === file.size ? { named, bytes } : null;",
        "else await handle.move(named);",
        "await dir.removeEntry(t.dir);",
    ] {
        assert!(
            kept.contains(wanted),
            "site/app/engine/files.mjs no longer has `{wanted}`"
        );
    }
    let page = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "files = await keepFiles({ record, open: fetched(base), module, kernel: ",
        "open: files ? files.open : fetched(base),",
        "if (files) await files.forget();",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/engine/pipeline.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "kept first: expected 0 read here and",
        "kept offline: expected",
        "kept tampered: expected the kept graph refused as not its BLAKE3",
        "kept first: expected every kept file named by its BLAKE3",
        "kept tampered: the files this load read were not forgotten",
        "kept again: expected every file downloaded again",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-22`: The engine page asks the kernel which precision of an ONNX export to read (decision 21), what each of its files is called (decision 24) and which device runs it (decision 22), and reads, names and runs exactly what the kernel answers; a precision the hub's record does not offer is refused by the kernel before any file is read
#[test]
fn gpu_gp_22() {
    // The journey runs the loads in a browser (`just page-journey`); this test
    // reads that the page asks the kernel's decisions 21, 24 and 22 and acts on
    // their answers, and that the journey checks each answer was the one used.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let page = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "new Uint8Array([24, code, PRECISIONS[name], 0, 0])",
        "new Uint8Array([24, code, PRECISIONS[chosen], sizes[chosen].data, part])",
        "join(new Uint8Array([21]), offer(\"q4f16\"), offer(\"q4\"),",
        "const path = sizes[chosen].path;",
        "join(new Uint8Array([22, gpu ? 1 : 0]),",
        "return on === \"webgpu\" ? createEngine(",
        "return Object.assign(generate, { task, model, dtype: chosen, device: on,",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/engine/pipeline.mjs no longer has `{wanted}`"
        );
    }
    for gone in [
        "ONNX_FILES",
        "decision 24 is not asked here yet",
        "(22) is not asked here yet",
    ] {
        assert!(
            !page.contains(gone),
            "site/app/engine/pipeline.mjs still names its own choice: `{gone}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "the precision was not the kernel's q4f16",
        "the graph's file was not the kernel's",
        "the device was not the kernel's",
        "onnx q4: expected the kernel to refuse a precision the record does not offer",
        "onnx q4: the load went on past the precision",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-23`: The engine page keeps the computed state of its chat frame's fixed head, the tokens every turn opens with, under the BLAKE3 of those token ids, and on a later turn asks the kernel (decision 27) whether that label equals the label of the new prompt's own opening ids; on reuse it restores the head and computes only the rest, and the turn's tokens equal those of computing the whole prompt
#[test]
fn gpu_gp_23() {
    // The journey runs two turns in a browser (`just page-journey`); this test
    // reads that the page asks the kernel's decision 27 with the label of the
    // prompt's own opening ids, and computes only what follows a reused head.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let page = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "const asked = label(prompt.slice(0, held.length));",
        "join(new Uint8Array([27]), asked, held.label, be(held.length, 4), be(prompt.length, 4))",
        "engine.restoreState(held.state);",
        "const tokens = await engine.run(prompt.slice(from), budget, {",
        "held = { length: head, label: label(prompt.slice(0, head)), state: await engine.saveState(head) };",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/engine/pipeline.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "the first turn asked to reuse a prefix",
        "the second turn did not ask the kernel to reuse the head",
        "the label asked is not the BLAKE3 of the prompt's first",
        "the second turn computed",
        "the second turn's tokens",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-24`: When the browser has too little room to keep a downloaded file, the engine page asks the kernel (decision 26) about each file already kept, oldest first, with the files this load uses pinned, drops a file only on evict and stops at enough; a file there is no room for on does not fit is not kept, and the model still loads from the download
#[test]
fn gpu_gp_24() {
    // The journey runs the short-storage visit in a browser (`just page-journey`);
    // this test reads that the store asks the kernel's decision 26 and acts on
    // each answer, and that the journey checks what was dropped and kept.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let kept = read("site/app/engine/files.mjs");
    for wanted in [
        "join(26, c ? pinned.has(c.name) : false, candidates.length - i, needed, free, freed)",
        "if (answer === \"evict\" && c) {",
        "if (answer === \"does not fit\") return false;",
        "if (!(await room(file.size))) {",
    ] {
        assert!(
            kept.contains(wanted),
            "site/app/engine/files.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "kept short: expected the other model's file dropped",
        "kept short: the other model's file was not dropped",
        "kept short: expected one file kept",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-25`: One call, pipeline(task, model, { device, dtype }), loads an ONNX export named on the hub and answers a function that runs a turn, with every step the engine page takes put to the verified kernel; the page is one caller of it, and a task the kernel does not run is refused by the kernel's decision 24
#[test]
fn gpu_gp_25() {
    // The journey calls the module on a page with no page code in between
    // (`just page-journey`); this test reads that the module asks the task
    // through decision 24, that the page loads through it, and that the
    // journey checks the call and the refusal.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let module = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "export async function pipeline(task, model, options = {}) {",
        "const code = TASKS[task] ?? 255;",
        "new Uint8Array([24, code, PRECISIONS[name], 0, 0])",
        "const generate = turner({ engine, module, ask, stops, chat });",
        "export function turner(session) {",
    ] {
        assert!(
            module.contains(wanted),
            "site/app/engine/pipeline.mjs no longer has `{wanted}`"
        );
    }
    let page = read("site/app/app.mjs");
    for wanted in [
        "import { CAPACITY, be, join, pipeline, turner } from \"./engine/pipeline.mjs\";",
        "session = { generate: turner({ engine, module, ask }) };",
        "const out = await session.generate(text, {",
    ] {
        assert!(
            page.contains(wanted),
            "site/app/app.mjs no longer has `{wanted}`"
        );
    }
    for gone in ["loadOnnx(", "keepFiles(", "engine.run("] {
        assert!(
            !page.contains(gone),
            "site/app/app.mjs runs `{gone}` itself instead of through the module"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "const { pipeline } = await import(\"./engine/pipeline.mjs\");",
        "pipeline: the call's tokens",
        "pipeline task: expected the kernel to refuse summarization",
        "pipeline task: the call went on past the task",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-28`: The engine page runs an ONNX export on its CPU device on the integer engine: the pipeline hands the module every weight the import read and the config's head counts, rope theta, epsilon and scaling, and each token is the module's decode_step, so the CPU run's tokens are the integer forward pass's, the second turn carrying on from the run the module kept
#[test]
fn gpu_gp_28() {
    // The journey runs the page (`just page-journey`); this test reads that
    // the CPU device's engine is the integer one and the journey checks it.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let pipeline = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "import { startIntegerEngine } from \"./integer.mjs\";",
        "startIntegerEngine({ module, ...raw, capacity: CAPACITY })",
    ] {
        assert!(
            pipeline.contains(wanted),
            "pipeline.mjs no longer has `{wanted}`"
        );
    }
    assert!(
        !pipeline.contains("createCpuEngine"),
        "the pipeline still builds the float CPU engine"
    );
    let integer = read("site/app/engine/integer.mjs");
    for call in [
        "\"decode_tensor\"",
        "\"decode_open\"",
        "\"decode_step\"",
        "\"decode_save\"",
        "\"decode_restore\"",
        "\"decode_reset\"",
    ] {
        assert!(integer.contains(call), "integer.mjs no longer calls {call}");
    }
    let load = read("site/app/engine/onnx-load.mjs");
    assert!(
        load.contains(
            "engineTensors(weights, read), { table, read, config: configText, rope: { ...rope } })"
        ),
        "onnx-load.mjs no longer hands the build the import's tensors and the config"
    );
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    assert!(
        journey.contains("the CPU device did not run the integer engine"),
        "the journey no longer checks the CPU device's engine"
    );
}

/// `GP-26`: The engine page checks every file of an ONNX export by the BLAKE3 the hub's record names for it, the small files in the module and the graph and data files through the importer, and refuses by name a record that names a file no BLAKE3; no file is checked by SHA-256 or SHA-1
#[test]
fn gpu_gp_26() {
    // The journey runs the loads in a browser (`just page-journey`); this test
    // reads that the loader and the store name files by the record's BLAKE3
    // and nothing by SHA-256 or SHA-1.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let loader = read("site/app/engine/onnx-load.mjs");
    for wanted in [
        "throw new Error(`the record names no BLAKE3 for ${p}`);",
        "const ok = digest === file.blake3 && bytes.length === file.size;",
        "m.callWith(encoder ? \"onnx_start_encoder\" : \"onnx_start\", unhex(graph.blake3),",
        "m.callWith(\"onnx_data_start\", unhex(file.blake3),",
    ] {
        assert!(
            loader.contains(wanted),
            "site/app/engine/onnx-load.mjs no longer has `{wanted}`"
        );
    }
    let kept = read("site/app/engine/files.mjs");
    assert!(
        kept.contains("return file && HEX.test(file.blake3 ?? \"\") ? file : null;"),
        "site/app/engine/files.mjs no longer keeps files by the record's BLAKE3"
    );
    for (path, text) in [
        ("site/app/engine/onnx-load.mjs", &loader),
        ("site/app/engine/files.mjs", &kept),
    ] {
        for gone in ["file.sha256", "gitSha1", "\"SHA-256\"", "\"SHA-1\""] {
            assert!(
                !text.contains(gone),
                "{path} still checks a file by `{gone}`"
            );
        }
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "onnx good: config.json was not checked by its BLAKE3",
        "onnx good: a file was checked by SHA-256 or SHA-1",
        "onnx noblake3: expected the record refused for naming config.json no BLAKE3",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-27`: The engine page reads a model's hub record through the hub's own client, vendored byte for byte from hub commit ec26789 and pinned by BLAKE3: the index root is checked by its Sigstore signature against the trust policy this repository pins, the record by the BLAKE3 the root names, and a root signed by a key the policy does not name, or for another identity, or changed after signing, is refused as RootUnverified naming the failed checks, as each other failure is by its code
#[test]
fn gpu_gp_27() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    // The vendored files are the hub's, byte for byte, at the BLAKE3 VENDOR.md names.
    let blake3 = |path: &str| {
        let bytes = std::fs::read(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut arriving = hologram_ai_download::Arriving::new();
        arriving.take(&bytes);
        arriving.label().to_hex()
    };
    let vendor = read("site/app/engine/hub/VENDOR.md");
    for (path, pinned) in [
        (
            "site/app/engine/hub/hub-client.mjs",
            "cdcc05c1c331a8fc2f5b6748ae84891ec13d82a627a2844c094588c8da8ba49a",
        ),
        (
            "site/app/engine/hub/sigstore.mjs",
            "0f25ca52fed3967f36d291179b35b36f10ed39f60c46be948b94037a369f3afa",
        ),
    ] {
        assert_eq!(blake3(path), pinned, "{path} is not the hub's file");
        assert!(
            vendor.contains(pinned),
            "VENDOR.md does not name {path}'s BLAKE3"
        );
    }
    let client = read("site/app/engine/hub/hub-client.mjs");
    assert!(
        client.contains(&format!(
            "export const SIGSTORE_BLAKE3 = \"{}\";",
            blake3("site/app/engine/hub/sigstore.mjs")
        )),
        "hub-client.mjs pins another sigstore.mjs"
    );
    // The policy is the hub model's screen 3 trust.policy, pinned here.
    let policy: serde_json::Value = serde_json::from_str(&read("site/app/engine/hub/policy.json"))
        .expect("policy.json is JSON");
    assert_eq!(
        policy["identity"],
        "https://github.com/Hologram-Technologies/hub/.github/workflows/pages.yml@refs/heads/main"
    );
    assert_eq!(
        policy["issuer"],
        "https://token.actions.githubusercontent.com"
    );
    assert_eq!(
        policy["indexPredicateType"],
        "https://hologram.foundation/hub/index/v1"
    );
    assert_eq!(
        policy["rekorLogId"],
        "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d"
    );
    for key in ["fulcioIntermediate", "rekorKey"] {
        assert!(
            policy[key].as_str().is_some_and(|v| v.len() > 64),
            "policy.json has no {key}"
        );
    }
    assert!(
        vendor.contains(&blake3("site/app/engine/hub/policy.json")),
        "VENDOR.md does not name policy.json's BLAKE3"
    );

    // The client, driven in Node through a chain minted for the test.
    let result = node("crates/conformance/tests/gpu/hub-check.mjs", &[]);
    assert_eq!(
        result["good"]["code"], "ok",
        "the minted chain was not proven: {result}"
    );
    assert_eq!(result["good"]["repo"], "fixture-owner/fixture-model");
    for (case, code, failed) in [
        ("foreignKey", "RootUnverified", Some(vec!["chain", "log"])),
        ("foreignLog", "RootUnverified", Some(vec!["log"])),
        ("otherIdentity", "RootUnverified", Some(vec!["identity"])),
        ("changedRoot", "RootUnverified", Some(vec!["subject"])),
        ("pinnedPolicy", "RootUnverified", Some(vec!["chain", "log"])),
        ("noBundle", "RootUnreachable", None),
        ("unknownRepo", "RecordUnknown", None),
        ("changedRecord", "RecordMismatch", None),
        ("unaddressedRecord", "RecordUnaddressed", None),
        ("unreachableRecord", "RecordUnreachable", None),
        ("changedFile", "FileMismatch", None),
        ("shortFile", "FileMismatch", None),
        ("unaddressedFile", "FileUnaddressed", None),
    ] {
        assert_eq!(result[case]["code"], code, "{case}: {result}");
        if let Some(failed) = failed {
            assert_eq!(
                result[case]["failed"],
                serde_json::json!(failed),
                "{case} failed other checks"
            );
        }
    }

    // The page asks the client, and nothing in it skips the signature.
    let loader = read("site/app/engine/loader.mjs");
    for wanted in [
        "from \"./hub/hub-client.mjs\";",
        "index = await openIndex(base, policy, { hash, fetch: hubFetch });",
        "const record = await readRecord(index, repo);",
        "new URL(\"./hub/policy.json\", import.meta.url)",
    ] {
        assert!(
            loader.contains(wanted),
            "site/app/engine/loader.mjs no longer has `{wanted}`"
        );
    }
    assert!(
        !loader.contains("waits on the hub's client module"),
        "the page still leaves the index root's signature unchecked"
    );
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "good: the index root's signature was not checked",
        "foreign: expected to stop at the index root signature",
        "if (path === \"/app/engine/hub/policy.json\")",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-30`: A return visit to the engine page with the network off loads the page, its engine and its kernel from the page's own service worker, reads the hub's index root, its signature and the model's record from this browser and checks them again as a download is checked, reads every file and tensor of the model from this browser with decision 28 asked again on the importer's own addresses, and answers with the reference's tokens; a kept record changed by one byte is refused as RecordMismatch, and with no hub index kept the page stops at the index root as RootUnreachable
#[test]
fn gpu_gp_30() {
    // The journey runs the visits in a browser with the server answering
    // nothing (`just page-journey`); this test reads that it does, and that
    // the page's worker, the kept hub index and the checks it says are there.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    for (path, wanted) in [
        ("site/app/sw.js", "await cache.addAll(listing.files.map((f) => new URL(f.path, ROOT).href));"),
        ("site/app/sw.js", "const fresh = await fetch(request);"),
        ("site/app/sw.js", "const kept = await cache.match(key);"),
        ("site/app/sw.js", "url.href.split(/[?#]/)[0] === LISTING) return;"),
        ("site/app/app.mjs", "navigator.serviceWorker.register(\"./sw.js\")"),
        ("site/app/app.mjs", "if (listed === OFFLINE) return checks.add(\"kernel\", subject, \"not checked\""),
        ("site/app/engine/loader.mjs", "const hit = store && (await store.match(url));"),
        ("site/app/engine/loader.mjs", "if (response.ok && store) await store.put(url, response.clone());"),
        ("site/app/engine/loader.mjs", "index = await openIndex(base, policy, { hash, fetch: hubFetch });"),
        ("site/app/engine/pipeline.mjs", "const record = await hubRecord({ indexBase: index, repo: model, checks, fetch: hubFetch });"),
        ("crates/conformance/tests/gpu/journey.mjs", "if (down) return request.socket.destroy();"),
        ("crates/conformance/tests/gpu/journey.mjs", "await context.setOffline(true);"),
        ("crates/conformance/tests/gpu/journey.mjs", "home offline: expected every tensor read here and no read of the network"),
        ("crates/conformance/tests/gpu/journey.mjs", "home record: expected a changed kept record refused as RecordMismatch"),
        ("crates/conformance/tests/gpu/journey.mjs", "home none: expected the page stopped at the index root, unreachable"),
    ] {
        assert!(read(path).contains(wanted), "{path} no longer has `{wanted}`");
    }
    // The worker answers no model byte and no hub file: those are checked
    // again on every read, which a worker answering for them would skip.
    let worker = read("site/app/sw.js");
    for absent in ["hologram-files", "/onnx/", "index/models", "uor/"] {
        assert!(!worker.contains(absent), "site/app/sw.js names `{absent}`");
    }
}

/// `GP-31`: The engine page keeps an ONNX export's tensors in the browser once each by the BLAKE3 address the hub's UOR index lists: each span is asked of decision 23, a tensor already held under any model, precision or repository is read from the browser and not downloaded, only the bytes not held are range-read from the first of the record's sources that answers, and nothing new is kept until decision 28 answers ready on the importer's own addresses; the same export under a second repository downloads nothing, its q4 graph after its q4f16 graph downloads only the tensors the first lacks, a kept tensor changed by one byte is refused and forgotten, a model the UOR index does not list is refused by name before any file is read, and an index giving a tensor another address is refused by decision 28
#[test]
fn gpu_gp_31() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    // The hub's shard rule is the hub's, byte for byte, at the BLAKE3 VENDOR.md names.
    let bytes = std::fs::read(root().join("site/app/engine/hub/uorkey.mjs")).expect("uorkey.mjs");
    let mut arriving = hologram_ai_download::Arriving::new();
    arriving.take(&bytes);
    assert!(
        read("site/app/engine/hub/VENDOR.md").contains(&arriving.label().to_hex()),
        "VENDOR.md does not name uorkey.mjs's BLAKE3"
    );
    for (path, wanted) in [
        ("site/app/engine/loader.mjs", "const shardPath = await uorShardOf(record.repo, root.shardBits);"),
        ("site/app/engine/loader.mjs", "if (blake3Hex(bytes) !== listed.blake3) refuse("),
        ("site/app/engine/loader.mjs", "refuse(`the hub's UOR index lists no tensors for ${record.repo} ${path} at ${record.revision}`);"),
        // Decision 23's wire: file length, offset, length, then the previous end.
        ("site/app/engine/files.mjs", "request.set(be(file.size, 8), 3);"),
        ("site/app/engine/files.mjs", "request.set(be(end, 8), 27);"),
        ("site/app/engine/files.mjs", "if (answer !== \"accept\") throw new Error("),
        ("site/app/engine/files.mjs", "segments.push({ from: t.offset, to: t.offset + t.length, named: t.address.slice(7), held: await heldAt(t.address.slice(7), t.length) });"),
        ("site/app/engine/files.mjs", "while (j < segments.length && !local(segments[j])) j++;"),
        ("site/app/engine/files.mjs", "response.status === 206 && span && Number(span[1]) === from && Number(span[2]) === to - 1"),
        // Decision 28 before anything new is named.
        ("site/app/engine/files.mjs", "request[0] = 28;"),
        ("site/app/engine/files.mjs", "if (verdict.answer !== \"ready\") {"),
        ("site/app/engine/files.mjs", "await p.handle.move(p.named);"),
        ("site/app/engine/onnx-load.mjs", "await admitted(table);"),
        ("site/app/engine/pipeline.mjs", "const manifest = await tensorManifest({ indexBase: index, record, paths: [path, ...dataPaths], checks, fetch: hubFetch });"),
        ("site/app/engine/pipeline.mjs", "files = await keepFiles({ record, open: fetched(base), module, kernel: decide, manifest });"),
        ("crates/conformance/tests/gpu/fixture.mjs", "export function uorIndexFiles(rows, { shardBits = 4, lie = null } = {}) {"),
        ("crates/conformance/tests/gpu/journey.mjs", "store twin: expected every tensor read here and none downloaded"),
        ("crates/conformance/tests/gpu/journey.mjs", "store q4: a tensor the q4f16 graph shares was downloaded again"),
        ("crates/conformance/tests/gpu/journey.mjs", "store q4: the fixture's two precisions share no tensor, so nothing is tested"),
        ("crates/conformance/tests/gpu/journey.mjs", "store tampered: the changed tensor is still kept"),
        ("crates/conformance/tests/gpu/journey.mjs", "store unlisted: expected the model refused by name for want of a manifest"),
        ("crates/conformance/tests/gpu/journey.mjs", "store lie: bytes were kept from a load decision 28 refused"),
        ("crates/conformance/tests/gpu/journey.mjs", "read by this load, was dropped for room"),
        ("site/app/engine/files.mjs", "// What this load reads from here is pinned before decision 26 is asked"),
    ] {
        assert!(read(path).contains(wanted), "{path} no longer has `{wanted}`");
    }
}

/// `GP-32`: Prefix reuse is token-identical and timed: on WebGPU and on the CPU, a second turn of the same text is answered reuse by decision 27 for the chat frame's head, computes only the positions after it, and gives exactly the first turn's tokens, each turn reporting its time to first token
#[test]
fn gpu_gp_32() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    for (path, wanted) in [
        ("site/app/engine/pipeline.mjs", "const verdict = await ask(\"reuse\", \"computed prefix to reuse\", join(new Uint8Array([27]), asked, held.label, be(held.length, 4), be(prompt.length, 4))"),
        ("site/app/engine/pipeline.mjs", "engine.restoreState(held.state);"),
        ("site/app/engine/pipeline.mjs", "computed: prompt.length - from,"),
        ("crates/conformance/tests/gpu/journey.mjs", "results.reuse[name] = await onnxVisit(context, { variant: \"again\", texts: [\"hi\", \"hi\"], before });"),
        ("crates/conformance/tests/gpu/journey.mjs", "reusing the head changed the tokens"),
        ("crates/conformance/tests/gpu/journey.mjs", "a turn reported no time to first token"),
    ] {
        assert!(read(path).contains(wanted), "{path} no longer has `{wanted}`");
    }
}

/// `GP-35`: pipeline("feature-extraction", model) loads a BERT encoder exported as Xenova's all-MiniLM-L6-v2 q4f16 export is, from its hub record and its own files, imports it with every operator asked of decision 18's encoder style and its file named by decision 24, places its weights by the node that reads them, int64 position ids left out, and answers each sentence's WordPiece ids and unit-length embedding from the module's integer encoder, bit for bit hologram-ai-cpu-encode's, on the CPU device decision 22 chose
#[test]
fn gpu_gp_35() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    // The journey runs the call in a browser (`just page-journey`); this test
    // holds the page and the journey to what they must do.
    let pipeline = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "const ENCODER_TASKS = new Set([\"feature-extraction\", \"text-classification\"]);",
        "encoder: encoding,",
        "const encoder = await startIntegerEncoder({ module, ...loaded });",
    ] {
        assert!(
            pipeline.contains(wanted),
            "pipeline.mjs no longer has `{wanted}`"
        );
    }
    let load = read("site/app/engine/onnx-load.mjs");
    for wanted in [
        "m.callWith(encoder ? \"wordpiece_read\" : \"tokeniser_read\"",
        "m.callWith(encoder ? \"onnx_start_encoder\" : \"onnx_start\"",
    ] {
        assert!(
            load.contains(wanted),
            "onnx-load.mjs no longer has `{wanted}`"
        );
    }
    let encoder = read("site/app/engine/integer-encoder.mjs");
    for wanted in [
        "m.callWith(\"encode_tensor\", given);",
        "m.callWith(\"encode_open\", named.encode(config));",
        "run(\"encode_embed\", tokens)",
    ] {
        assert!(
            encoder.contains(wanted),
            "integer-encoder.mjs no longer has `{wanted}`"
        );
    }
    for float in ["Math.", "parseFloat", " * ", " / "] {
        assert!(
            !encoder.contains(float),
            "integer-encoder.mjs computes a number of its own: `{float}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "[\"bert\", \"feature-extraction\"]",
        "the page's bits are not hologram-ai-cpu-encode's",
        "if (run.device !== \"cpu\" || run.dtype !== \"q4f16\")",
        "no check names the module's integer encoder",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
    let export = read("crates/conformance/tests/gpu/encoder-export.mjs");
    assert!(
        export.contains(
            "node(\"Gather\", \"/embeddings/Gather\", [hold(\"embeddings.position_ids\", \"i64\""
        ),
        "the fixture no longer carries MiniLM's int64 position ids"
    );
}

/// `GP-36`: pipeline("text-classification", model) loads a DistilBERT classifier exported as Xenova's SST-2 q4f16 export is and answers each sentence's label, chosen by the module by comparing the scores' bits as integers and named by the config's id2label, and the scores, bit for bit hologram-ai-cpu-encode's
#[test]
fn gpu_gp_36() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let encoder = read("site/app/engine/integer-encoder.mjs");
    for wanted in [
        "const said = run(\"encode_classify\", tokens);",
        "const index = said[0];",
        "label: labels?.[index] ?? String(index)",
    ] {
        assert!(
            encoder.contains(wanted),
            "integer-encoder.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "[\"distil\", \"text-classification\"]",
        "not label ${w.label}",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-40`: Code written for transformers.js runs on the engine unchanged: pipeline(task, model, { device, dtype, progress_callback }) loads through decision 70 and the one call; a messages list answers [{ generated_text: [...messages, { role: assistant, content }] }] with the one call's text, a string answers [{ generated_text: prompt + completion }], a TextStreamer receives the answer as it is made, and an fp32 dtype, an unknown device or do_sample are refused by name by decision 70 before anything loads or runs
#[test]
fn gpu_gp_40() {
    // The journey calls transformers.js's own surface on a page with no page
    // code in between (`just page-journey`); this test reads that the surface
    // puts every call to decision 70 before it loads or runs, answers
    // transformers.js's shapes, and that the journey checks each of them.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let surface = read("site/app/engine/transformers.mjs");
    for wanted in [
        "export async function pipeline(task = \"text-generation\", model, options = {}) {",
        "export class TextStreamer {",
        "const device = await ask(\"call\", \"a transformers.js pipeline call (decision 70)\", callRequest(task, options), RUNS);",
        "await ask(\"call\", \"a transformers.js generation call (decision 70)\", callRequest(task, options, generation), RUNS);",
        "return [{ generated_text: [...input, { role: \"assistant\", content: turn.text }] }];",
        "return [{ generated_text: return_full_text ? input + turn.text : turn.text }];",
    ] {
        assert!(
            surface.contains(wanted),
            "site/app/engine/transformers.mjs no longer has `{wanted}`"
        );
    }
    let first_ask = surface
        .find("await ask(\"call\"")
        .expect("no call to decision 70");
    let first_load = surface.find("await engine(").expect("no load");
    assert!(
        first_ask < first_load,
        "the surface loads before decision 70 answers"
    );
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "const { pipeline, TextStreamer } = await import(\"./engine/transformers.mjs\");",
        "transformers: a messages call must answer",
        "transformers: a string call must answer",
        "transformers: the streamer received",
        "[\"dtype\", \"refused: unsupported dtype\"]",
        "[\"device\", \"refused: unsupported device\"]",
        "[\"sampling\", \"refused: sampling\"]",
        "transformers dtype: the load went on past decision 70",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-41`: The integer engines run off the page's thread: the pipeline hands a worker a copy of the page's module and every weight the import placed, the worker computes every decode and encode step, the page's own thread calls none of them and answers throughout a turn, each token reaches the page as its own message, the tokens are the reference's and the embedding's bits GP-35's
#[test]
fn gpu_gp_41() {
    // The journey runs the page (`just page-journey`) and counts the page
    // thread's own calls into the module; this test reads that the pipeline
    // builds both integer engines in the worker, that the worker opens them
    // with the engines the page used before, and that the journey checks it.
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let pipeline = read("site/app/engine/pipeline.mjs");
    for wanted in [
        "startIntegerEngine({ module, ...raw, capacity: CAPACITY })",
        "const encoder = await startIntegerEncoder({ module, ...loaded });",
    ] {
        assert!(
            pipeline.contains(wanted),
            "pipeline.mjs no longer has `{wanted}`"
        );
    }
    for gone in ["createIntegerEngine(", "createIntegerEncoder("] {
        assert!(
            !pipeline.contains(gone),
            "pipeline.mjs builds an integer engine on the page's thread: `{gone}`"
        );
    }
    let worker = read("site/app/engine/integer-worker.mjs");
    for wanted in [
        "const module = moduleOf(data.compiled, data.memory);",
        "createIntegerEncoder(given) : createIntegerEngine({ ...given, rope: data.rope, capacity: data.capacity })",
        "onToken: (out) => say(\"token\", out)",
    ] {
        assert!(
            worker.contains(wanted),
            "integer-worker.mjs no longer has `{wanted}`"
        );
    }
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "Object.defineProperty(WebAssembly.Instance.prototype, \"exports\", {",
        "worker: the page's own thread computed:",
        "worker: tokens did not arrive one per step:",
        "worker: the encoder's bits are not GP-35's",
        "unanswered in the turn",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey no longer checks `{wanted}`"
        );
    }
}

/// `GP-50`: Before the page instantiates engine.wasm it computes the bytes BLAKE3 and asks decision 80: the sealed engine loads, and an engine with one byte flipped is refused by name, not the sealed engine, before it is compiled
#[test]
fn gpu_gp_50() {
    let read = |path: &str| {
        std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    // One way in: openModule takes the page's asker, hashes the bytes it
    // fetched, asks decision 80, and compiles those same bytes only after.
    let loader = read("site/app/engine/loader.mjs");
    let open = loader
        .split("export async function openModule(url, ask) {")
        .nth(1)
        .expect("openModule takes the page's asker");
    let fetched = open
        .find("const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());")
        .expect("openModule fetches the bytes once");
    let asked = open
        .find("await ask(\"engine\", \"engine.wasm is the engine the kernel seals (decision 80)\", join(new Uint8Array([80]), blake3(bytes)), \"sealed\");")
        .expect("openModule asks decision 80 about the bytes' BLAKE3");
    let compiled = open
        .find("return moduleOf(await WebAssembly.compile(bytes));")
        .expect("openModule compiles the bytes it checked");
    assert!(
        fetched < asked && asked < compiled,
        "the engine is compiled before decision 80 is asked"
    );
    // Bytes become a module in one place; everything else, the integer worker
    // included, is handed that module (GP-41) and never fetches or compiles one.
    assert_eq!(
        loader.matches("WebAssembly.compile(").count(),
        1,
        "the engine is compiled in one place"
    );
    assert!(!loader.contains("WebAssembly.instantiate("));
    let worker = read("site/app/engine/integer-worker.mjs");
    assert!(
        !worker.contains("fetch(") && !worker.contains("WebAssembly."),
        "the integer worker must take the module openModule checked, not bytes of its own"
    );
    for (caller, call) in [
        (
            "site/app/app.mjs",
            "openModule(\"./engine/engine.wasm\", ask)",
        ),
        (
            "site/app/engine/pipeline.mjs",
            "openModule(new URL(\"./engine.wasm\", import.meta.url), ask)",
        ),
    ] {
        assert!(
            read(caller).contains(call),
            "{caller} does not open the engine through decision 80"
        );
    }
    // The journey serves a flipped engine and requires the refusal by name,
    // at decision 80, with the engine never compiled and the container never read.
    let journey = read("crates/conformance/tests/gpu/journey.mjs");
    for wanted in [
        "await page.route(\"**/app/engine/engine.wasm\"",
        "body[body.length >> 1] ^= 0x01;",
        "unsealedStop?.step !== \"engine\"",
        "answered \"not the sealed engine\"",
        "unsealed.compiled.includes(unsealed.engineLength)",
        "unsealed.containerReads !== 0",
        "answered \"sealed\"",
    ] {
        assert!(
            journey.contains(wanted),
            "the journey does not check `{wanted}`"
        );
    }
}
