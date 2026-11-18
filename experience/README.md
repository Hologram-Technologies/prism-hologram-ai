# What Hologram AI does for the people who use it

`src/HologramAiExperience.lex.tex` is the specification of what milestone 2 gives a person who opens
Hologram AI: one named statement per promise, in plain words. It is a LexLean module of its own, outside
`app/`, so changing a statement does not change the kernel's proof inputs and does not rebuild or re-prove
the kernel.

These are statements, not theorems: Lean proves nothing about an experience. `lexlean check` holds the
module to the language. Each statement is held by what the milestone's done-when rows name in
`docs/Q_ENGINE_PLAN.md`: the kernel's decisions and their proofs, the scenarios and tests in `features/`
and `crates/conformance`, and the measurements in `GPU-VERIFICATION.md`.

    lexlean fmt --check && lexlean lock --check && lexlean check

`just app` runs that line beside the app's own checks.
