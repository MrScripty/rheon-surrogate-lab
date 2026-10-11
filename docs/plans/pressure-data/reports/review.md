# Independent numerical/source review

2026-10-11. Reviewer: independent read-only numerical_review agent.
Reference: Rheon 34235d33c886aa05b5bdf95d920b863d1fb2ad6f.

Confirmed from operator.rs:300-393,463-509 and pressure.rs:414-436:
A weights are V/(rho*h²), b=-V*div(u*)/dt, pressure is Pa, b and residual are
m³/s². Pressure cell zero is fixed; RHS must be compatible and not mean-shifted.
Full residual includes gauge row. Predicted divergence dt/V*||r||inf is distinct
from actual corrected f32 divergence. Upstream pressure_contract.rs:195-234 uses
1e-9 predicted and 1e-5 actual divergence limits.

Boundary constraints apply only to normal velocities; no-slip would overstate
the contract. MAC offsets and X-fastest order from geometry.rs:207-229 are
required in exports. CNN tensors must use [N,C,Z,Y,X]. solve_loaded resets pressure,
so warm-start measurement is unavailable. No AGENTS.md or .agents found in pin.

Reviewer recommended independent dyadic 3D manufacture, zero checks and isolated
negative tests. Implemented in tests/contracts.rs. Production source review found two medium issues: the prediction evaluator
omitted the predicted-divergence gate, and the 3D-variation test compared across
row/plane boundaries. Both were fixed with isolated regressions (near-threshold
prediction and extruded-field counterexamples). This report is evidence of source/API inspection, not an
independent proof of general numerical accuracy.

CPU model review independently reran all 16 Rust and 4 Python tests. Train-only
normalization, optimizer inputs, X-fastest patches, gauge differentiation and
Adam updates were confirmed. Reviewer found a missing evaluator identity in the
model receipt: a separately selected binary could contain stale checks. Fixed by
including CLI build commit/dirty flag/schema/Rheon pin, recording its binary
SHA-256 and rejecting incompatible/dirty or changed evaluators. This is traceable
provenance, not cryptographic source authentication. No other material findings
were reported. Timing remains explicitly a smoke measurement, not a benchmark.

Final follow-up: reviewer confirmed evaluator-provenance repair and inspected
both actual receipts: clean e45a4ceefc3ecc89531a2dcf859935c8459f652a and matching
binary hashes. No remaining material numerical/model review blocker. Explicitly
retain the negative held-out result: relative RMSE 7.664, max corrected divergence
17.458 s^-1, zero qualified predictions. Loss reduction is not acceleration.

Development diagnostic review: baseline definitions, training-only mean, centered
pressure errors, checkpoint replay and remote-RHS witness were independently
reviewed; no optimization occurs in diagnose_cpu.py. Reviewer flagged that full
dataset verification still reads old test targets before filtering. Disposition:
retain and explicitly document the integrity-verification exception (ADR 0002,
README); claim no fresh model test predictions/metrics, not unread old targets.
Reserved new seeds have not been materialized. No other material finding reported.

Final diagnostic evidence review found no material numerical overclaim or
transcription error in qualification-diagnosis.md versus its compact receipt.
Confirmed PCG iteration ranges, fitted-case gate ratios, zero-response and locality
measurements. Report distinguishes structural limitations from causal attribution,
iterations from runtime, and old-test integrity checks from new model evaluation.
The reserved test remains ungenerated and proposed model change is future work.

Global candidate preregistration/source review confirmed DCT-II ordering,
constant-mode omission, gauge-aware relative-loss gradient, exactly 500 updates
and final-only validation. No analytic inverse coefficients or validation-driven
updates. Reviewer requested prediction hashes and runtime metadata in the receipt;
added before running. Reviewer did not train or evaluate the candidate.

Isolated warm-start review: finite/shape/cell-zero validation, sealed-only scope,
full initial b-Ap, original-b tolerance, all four Krylov scratch resets and old
zero-start path were independently confirmed. Reviewer requested exact typed
cancellation/iteration failures instead of a generic error assertion; fixed.
Probe source review confirmed real patched-API calls, development-only case
selection, identical physical gates and frozen prediction hash checks. Paired
wall time encloses inference, file transfer/cleanup, native process and physical
qualification. Clarified that parent Python startup/imports are excluded, while
Git/hash/reference/basis/checkpoint/RHS preparation is measured separately.
Reviewer inspected actual candidate receipt: train8/8 and val1/4 raw qualification,
worst val linear ratio1.728; no numerical overclaim or protocol blocker found.

Final global/probe evidence review: compact and full receipts agree, full-receipt
SHA256 checks, all four 20-sample wall sequences and medians match, and all 960
repeated solves qualify across 12 development cases. Training warm iterations
are zero; validation totals5 Jacobi/3SGS, with actual divergence1.17657e-7 for
every path. Raw versus corrected acceptance, timing exclusions, old-test integrity
exception, ungenerated reservation and geometry-prior limits are accurately
reported. Two editorial corrections (spacing and floating-point scaling wording)
were applied. No remaining material blocker; reviewer ran no new experiments.
