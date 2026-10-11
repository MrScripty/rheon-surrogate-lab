# Execution ledger

2026-10-11 — Inspected exact Rheon pin and Coding-Standards, read pressure/operator
contracts and tests. Separate reference checkout used; existing Rheon worktree
untouched. Creation lookup 404, createRepository denied by integration (PUB-1).
Independent read-only review confirmed sign, units, gauge and distinct predicted
versus actual f32 divergence gates.

Implemented fixed 8×7×6 reference data generation, strict schema/typed raw arrays,
completion and SHA-256 checks, full persisted requalification and deterministic
replay. Four nontrivial seed/family distributions and two check-only cases.
`cargo test --locked`: 16 passed. `cargo fmt --all --check` and
`cargo clippy --locked --all-targets -- -D warnings`: passed. Real CLI smoke is
pending a clean source commit. No upstream files changed.

2026-10-11 — Generator committed as cc2820646601f3db51fade707bfac5a521653257.
Real release CLI generated and independently reverified 18 cases, 144 arrays,
262,944 bytes. Max actual f32 divergence 1.909211277961731e-7 s^-1; max predicted
6.203842994878528e-10 s^-1; max 77 iterations. Receipt: reports/generator-smoke.json.
Push to exact requested remote failed: Repository not found. No alternative used.

Source review found missing predicted-divergence gate in raw prediction evaluator
and row/plane wrap in a test oracle. Both fixed with regressions before checkpoint;
16 Rust tests pass. Added optional NumPy 3D CNN with train-only normalization,
cell-zero gauge, Adam, and raw prediction evaluation using Rust/Rheon. Four Python
tests pass including finite-difference gradients and normalization leakage probe.

Independent CPU source review confirmed gradient/order/split handling and found
missing evaluator provenance. Added explicit evaluator build metadata and binary
hash to model receipts, with compatible schema/pin and clean-build checks. All
16 Rust and 4 Python tests, fmt and Clippy pass after that repair. CPU smoke run
will use a fresh release build of the resulting clean commit.
