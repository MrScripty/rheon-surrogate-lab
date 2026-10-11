# ADR 0001: A pinned reference and independently qualified data boundary

Status: accepted for this experiment. Author: Puma. Date: 2026-10-11.

The experiment needs pressure targets from actual preprojection velocities.
Rheon's accepted Simulation and Dense3dSequence fields have already been
projected, so they cannot supply the original RHS. Generate deterministic
boundary-valid MAC fields and call the public PressureOperator/PressureWorkspace
APIs from the exact Git dependency. Do not vendor a solver or change Rheon main.

`experiment.rs` owns families and numerical acceptance; `dataset.rs` owns the
persisted v1 contract, completion and corruption checks. Rheon remains the
numerical operator/solver authority. The model consumes verified exports, fits
normalization on training only, and writes predictions for Rheon to measure.
The manifest references source revisions; it does not grant source authenticity.

Use one fixed sealed 8×7×6 experiment first. A full residual is measured on every
cell with A positive semidefinite before gauge elimination and b=-V*div(u*)/dt.
Pressure[0] is exactly zero. Do not subtract the mean of b. Require both linear
and predicted-divergence gates, then separately measure actual corrected f32
velocity divergence. Full-family and seed separation prevents neighboring-frame
leakage; manufactured and zero cases are check-only.

Use raw typed little-endian arrays plus JSON to avoid a heavyweight tensor or
storage dependency. Fixed paths/shapes and bounded reads reject unsupported
metadata before allocation. The final complete manifest is renamed into place
only after qualified arrays are written; a failed generation is disposable,
visible as incomplete, and cannot be resumed or overwritten implicitly.

Later warm-start API requirements (upstream, separately reviewed):

- Accept only finite, exactly shaped initial pressure; canonicalize or reject
  gauge explicitly.
- Initialize the full residual as r=b-Ap, including the gauge row's reporting.
- Keep tolerance relative to the original b, never the initial residual.
- Reset direction, residual, product and preconditioner scratch correctly.
- Preserve cancellation/failure outcomes and zero-start behavior.
- Test exact, poor, shifted, nonfinite and wrong-length guesses, reused workspace
  behavior, both Jacobi/SGS, and actual f32 divergence.

Until that API exists, raw CNN predictions can be measured but warmstart+PCG
cannot be compared. A later comparison must match true residual AND actual
corrected divergence across raw predictions, warmstart+PCG and zero-PCG; count
inference, transfer, gauge conversion, solve, correction and validation costs.
A tiny overfit or lower pressure MSE is not evidence of acceleration.
