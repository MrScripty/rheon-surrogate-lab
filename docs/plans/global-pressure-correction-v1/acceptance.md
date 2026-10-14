# Acceptance and claim boundaries

D criteria close the **design task**. E criteria close a future executed experiment.
All E rows are pending; no measured value is supplied by this PR. A failed candidate
can close an honestly completed experiment without satisfying a usefulness claim.

| ID | Observable design criterion | Kind / environment / mode | Status/evidence |
| --- | --- | --- | --- |
| D1 | Exact public lab/merge identities and actual CNN/model/native source inspected; earlier failures preserved | artifact / GitHub + clean Linux clone / read-only source review | satisfied: inspection.md |
| D2 | Primary-source comparison justifies one bounded target and admits analytic competitor | design / primary publications / sourced review | satisfied: research.md |
| D3 | Dataset/grouping, leakage, trained-weight restore, native quality, rollout and total-cost criteria specified | design / proposed CPU/full contracts / review | satisfied: data-and-model.md and this file |
| D4 | Generic Tuldok contracts, CLI/UI workflow and separate GPU path respect actual limits | contract / merged Tuldok source / review | satisfied: execution-and-handoff.md, local-gpu.md |
| D5 | Independent numerical/standards design review completed; material findings dispositioned | design / independent GPT-6.1 Sol High reviewer / read-only review | satisfied: [independent review](reports/design-review.md) |
| D6 | Only new design directory changed; commit/push and draft PR verified, no training/tags/outreach | artifact / local Git + GitHub / automated + diff review | pending: reports/verification.md |

## Common accepted-step physics

Freeze these criteria before test generation. They apply equally to baseline,
learned acceptance, warm-PCG and reset fallback, not just the solver status flag.

- All arrays finite, exactly shaped and boundary-valid; `p[0]=0` after gauge fix.
- Full unpreconditioned `r=b-Ap`, including cell 0; original RHS unchanged.
  `||r||2 <= max(1e-12, 1e-8*||original b||2)` in the operator's native units.
  Also report relative residual only for nonzero b, absolute residual and max norm.
- Predicted divergence `dt*||r||inf / cell_volume <= 1e-7 s^-1`, as qualified
  against native build_rhs/correct_velocity signs/scaling. Actual corrected MAC
  f32 divergence independently measured: maximum <=1e-5 s^-1; report L2/RMS too.
- Net boundary volume flux is exactly zero in stored f32 wall-normal faces.
  Recomputed `abs(sum(div_i)*V) <= max(1e-12 m³/s, 1e-8*sum(abs(div_i))*V)`;
  identify summation precision and physical unit. This checks cancellation/global
  flux separately from max-cell divergence, not engineering mass validation.
- Pressure/velocity correction is staged until all gates pass. Any native failure
  or nonfinite state prevents committing/advancing the frame.

The absolute residual value comes from the existing native trajectory contract;
operator scaling/units must be rechecked for each declared geometry. If a reference
cannot satisfy these unchanged gates, stop and replan before any model work.

## Residual-controlled path and bounded fallback

At a step, construct original b from the method's own preprojection velocity.
Use its own last accepted pressure as p_base (zero at t=0). Compute full r_base.
Predict a single correction. Checkpoint/source/split/normalization identity or data-contract corruption
fails before scientific execution; it is never silently rescued by fallback. For a
valid admitted model/data closure, an unsupported-domain inference request or
malformed/nonfinite numerical model output skips the proposal, records the exact
cause, and native PCG starts from p_base. No model call in an unsupported geometry, counts or dtype contract.

For valid proposals, independently compute full r_guess. Reject the proposal if
`||r_guess||2 > ||r_base||2` (when both zero, keep p_base), or gauge/finite checks
fail. If all common physics gates pass after f32 correction, accept directly.
Otherwise run the native warm-start API for at most **32 iterations** from the
eligible guess. If its candidate still fails any common gate or the solver errors,
reset all scratch and retry **once** from p_base with the original native limit
of **3,000 iterations**. Rejected proposals go directly to that full-budget path.
No recursion, retraining, relaxed tolerance or repeated inference. Both attempts
use original b for stopping, never the guess residual as a new relative scale.
A failed reset solve fails the Run/rollout; partial frames are diagnostic only.

Record proposal eligibility/rejection, direct acceptance, warm correction,
reset fallback and terminal failure as distinct outcomes. Since ordinary warm
correction may occur on most frames, report **any PCG use** as well as costly
reset-fallback frequency. Calling PCG corrections part of the model cost is required.

## Prospective experiment gates

| ID | Observable criterion | Kind / environment / mode | Status |
| --- | --- | --- | --- |
| E1 | Isolated native API admits guesses at fixed pin, retains zero-start/cancel/reuse semantics; hand-stencil/operator and analytic action agree | integration / real pinned native Linux / automated + review | pending |
| E2 | Deterministic reference data, complete qualified trajectories, persisted group separation and intended corruptions rejected | contract+system / fresh CPU processes / automated | pending |
| E3 | One train recipe performs exactly 200 actual Adam updates under smoke limits; records full history/input/source/runtime identity, only train-derived stats | system / actual Tuldok captured CPU Run / automated | pending |
| E4 | Fresh zero-update restore uses exact trained artifact bytes; same-runtime predictions bitwise equal; shape/hash/source/split/normalization drift refused | system / new process and managed artifact bindings / automated | pending |
| E5 | Own-state trajectories satisfy every common physics gate; loss reduction reported without treating it as solver quality | system / actual native CPU / automated | pending |
| E6 | Actual Pipeline/Runs displays train/restore/eval, diagnostic scope, verified downloads and post-reopen state | user-workflow / actual Tuldok + browser desktop/narrow / automated observation | pending |
| E7 | Independent source/physics/protocol review and committed freeze before test materialization; no numerical/test-driven edits after access | artifact / Git+independent reviewer / manual+automated | pending |
| E8 | All three new held-out suites evaluated as complete autonomous trajectories with stability/conservation/error/failure summaries | system / real native CPU, required local GPU only for full train / automated | pending |
| E9 | Matched-quality resident and cold total costs, fallback/correction/setup/offline cost and all failures reported against temporal/zero-PCG/analytic | performance / same recorded hardware / automated | pending |

## Rollout, conservation and fidelity

Compute per-frame pressure-relative L2 (after gauge), velocity-relative L2 and
absolute RMS, tracer-relative L2/RMS, kinetic energy and Courant number, against
independently evolved reference rollouts from the same initial state/forcing.
Floors are `max(reference norm,1e-12 in that field's SI L2 units)`; zero-reference
frames also report absolute error and are not hidden in relative averages.
No teacher pressures or state resets are injected in learned evaluation.

For every smoke test trajectory: 96 steps; full: 600. Require finite fields,
Courant <=0.5, no missing/restarted frames, max velocity-relative L2 <=1e-3 and
max tracer-relative L2 <=1e-2 versus qualified zero-Jacobi reference. These are
proposed v1 numerical-equivalence gates, not experimentally observed bounds.
Report first failing step and every retained failure; do not discard divergent paths.
Also report initial/final/time-series kinetic energy, tracer mass and extrema.

The passive tracer has no source after initialization and no post-step clamp.
Measure `M_t = V*sum(c_i)`, signed and absolute drift from M_0, and separate
reference transport drift from `(M_candidate-M_reference)/max(abs(M0),1e-12)`.
Require max candidate/reference mass discrepancy <=1e-3. Report absolute baseline
mass drift; native semi-Lagrangian advection need not conserve tracer exactly.
Do not claim exact tracer conservation or impose invented momentum/energy
conservation on externally forced, numerically dissipative rollouts. The geometry
is sealed and constant density; volume-flux conservation is the projection metric.

## Usefulness and acceleration decision

Smoke is a feasibility experiment. Loss/history, error, rejection/PCG/fallback
rates and iteration counts are all outputs; no mandatory loss reduction threshold
or CPU speedup is invented. Implementation/workflow success can coexist with an
unhelpful model. Unit-gain/untrained checkpoint and zero correction are diagnostic
ablations on train/validation only, not extra test competitors or a selection sweep.

Qualification must hold on **every** test trajectory, not only aggregate means.
For a useful warm-start signal, require median corrected PCG iterations below
previous-pressure PCG on each suite, while retaining all wall-cost information.
If iteration savings do not pay for inference, call it a corrected initial guess
with no acceleration advantage. This signal does not admit more training in v1.

For an acceleration claim, require all quality gates, resident **total rollout**
median speedup >1.10 versus the faster matched-quality previous-pressure Jacobi/SGS
baseline, and the paired trajectory bootstrap 95% lower speedup bound >1.0.
The 10% bar is a preregistered practical proposal to avoid weak marginal wins,
not a sourced expected gain. Bootstrap independent trajectories (seed 20261016,
10,000 resamples), never frames or timing repetitions as independent samples;
with four/eight trajectories per suite uncertainty remains substantial. Require
the criterion separately on each suite; report ratios against analytic regardless.
A broad best-solver acceleration claim additionally requires the same criterion
against analytic cosine. If analytic wins, do not claim a learned best-solver gain.
Cold-start results require their own passing comparison for any cold-start claim.

## Cost measurement contract

Seven matched-quality paths: zero-Jacobi, temporal-Jacobi, learned+Jacobi,
zero-SGS, temporal-SGS, learned+SGS, analytic cosine. Raw learned predictions are
quality diagnostics outside rollout ranking. All paths enforce identical physics
and autonomous state progression; both learned preconditioners use the same frozen
weights with zero updates. Analytic pays transform/setup/verification cost.

One untimed quality rollout per method/suite/trajectory, then **three resident
and three cold** balanced paired timing repetitions; use a seeded Latin rotation of seven methods, serial
execution, reversing alternate repetitions. Freeze the exact ordering manifest,
including starting offsets across trajectories and regime interleaving, before test access. Repeated
runs estimate timing variability; they do not increase independent test count.
Preserve every timing sample and min/median/max, failures and per-trajectory ratios.

Thus smoke executes 12 trajectories×96frames×7methods×(1quality+3resident+3cold)
= **56,448 method-frames**. Full executes 24×600×7×7 = **705,600 method-frames**.
Resident quality+timing alone is403,200 full method-frames; cold adds302,400.
Reference test generation is additional (smoke1,152frames/full14,400frames).
Before freeze/admission, benchmark only train fixtures for each method and extrapolate
all stages, both timing regimes, fixed repetitions, source verification/custody,
artifact writes and startup; include uncertainty headroom. Projections are feasibility
checks, not benchmark claims. If they exceed the preset's owned resource cap, stop
and review before expensive work or final-test generation. Never reduce repetitions
or crop trajectories after seeing test cost/outcomes.


Resident interval starts before native launch and ends after native teardown/JSON
parse, includes all stages, CPU preprocessing/inference, IPC, residual gating,
correction, 32-iteration work, reset fallback and verification. Report imports/model
setup and warmup separately, and amortized cost with those one-time costs included.
Cold interval starts before fresh Python process launch and ends after exit/parse;
includes imports, trained-weight load/validation, native setup and teardown. Do not
charge unused model imports to numerical baselines. Caches remain uncontrolled;
“cold” means process/model start, not cleared hardware/filesystem caches.

Record native/Torch/BLAS thread counts, versions, CPU/GPU model, cgroup quota,
RAM/VRAM peak and payload sizes. GPU inference timers synchronize and include
host↔device transfer; operational comparisons on CPU and optional GPU are separate.
Nested stage timers diagnose cost; enclosing wall time decides cost. Offline
costs include provisioning/build, reference generation/verification, training,
checkpoint/export/restore qualification, catalog capture and total harness time.
Report measured break-even calls only when positive savings exist; no hypothetical
amortization is evidence. No energy/$ claim without actual measurement.
