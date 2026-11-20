# The claims the engine's crates make of the kernel

One file per decision, `NNN.json` for decision NNN: `{"decision": NNN, "claims": [{"request", "answer"}]}`.
A decision adds its own file; nothing else changes, so two pull requests adding
claims for different decisions never touch the same file.

Each request's first byte is its file's decision, and a file holds at least one
claim; the Rust suites (`kernel/claims.rs`) and `replay.mjs` refuse anything
else. The Rust suites check that each request is the crate's own, and the `app`
recipe runs every claim through the compiled guest of the holo it has just built.
They are not the holo's examples: a closed list of examples longer than about
108 does not build (kernel-verification/README.md, The limit on examples).
