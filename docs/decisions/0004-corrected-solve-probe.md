# ADR 0004: Bounded local corrected-solve probe

Author: Puma. Status: admitted after ADR 0003 signal passed, before probe timing.

Keep the root Rheon dependency at 34235d3. Apply the separately reviewed local
initial-pressure patch in sibling rheon-warmstart. The independent Cargo project
probes/warmstart links only that path and identifies its built commit/dirty state.
The patch supports sealed gauge-fixed domains only; free-surface warm starts are
explicitly unsupported. Both existing zero-start entry points remain intact.

Reuse final global-v1 weights with no optimizer steps or model selection. Verify
the old dataset first (including the documented old-test integrity exception),
then load only the 12 train/validation RHS and preprojection velocities. No test
predictions, final reserved seeds or pressure targets enter the native probe.
Confirm regenerated predictions match the frozen hashes before measuring.

Measure exactly 20 paired batches per preconditioner, alternate zero/warm order,
all 12 development cases per batch. Each zero path runs the native process with
original RHS/velocity file reads, hashes, RHS reconstruction, allocations, PCG,
f32 velocity correction and independent full-residual/actual-divergence checks.
Each warm path additionally performs NumPy batch inference, writes little-endian
predictions to a fresh temporary directory, and deletes them. Include the native
process launch/JSON transfer in Python wall time, not just Rust solve duration.
Report median/min/max batch wall times plus stage means and per-case counts.

Both paths use original-b relative 1e-10 / absolute 1e-12 linear thresholds,
predicted divergence 1e-9 and actual corrected f32 divergence 1e-5. Every case in
every batch must meet all gates. Iterations and timing remain separate evidence.

This is a warm-cache local pipeline measurement on 336 cells, not an integrated
simulator or acceleration claim. Python parent startup, package loading and argument parsing are excluded.
Git/hash provenance checks, basis/checkpoint/RHS preloading and complete reference
verification are outside the paired loop and separately recorded as preparation.
Native child startup, current RHS/velocity loading and prediction file exchange
are inside. No GPU transfers exist. No validation-driven change or timing-based
model selection is authorized. Report overhead and scope even if warm PCG takes
zero iterations. Deployment would need its own integrated benchmark and broader
frozen-test protocol, not extrapolation from this probe.
