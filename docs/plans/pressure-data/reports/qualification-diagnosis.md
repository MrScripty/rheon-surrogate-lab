# Why the first CNN predictions fail qualification

Author: Puma. Date: 2026-10-11.
Diagnostic source: `cf0deca791add57277f98f5fe9692b8188ebc4bb`.
[Machine receipt](development-diagnostic.json). Rheon remains pinned to
`34235d33c886aa05b5bdf95d920b863d1fb2ad6f`.

## Scope and compute

One predeclared diagnostic comparison: 8 training and 4 validation cases, five
fixed nonlearned baselines, two unchanged CNN checkpoints, one deterministic
remote-face witness. Zero optimizer steps and zero new model candidates.
Measured diagnostic-section wall time: 0.339 s, including subprocesses, data I/O,
allocations and repeated dataset/reference verification. It excludes Python
startup/imports, initial Git/binary-hash checks and final receipt serialization. Baseline workflow: 0.214 s. The diagonal
uses 336 public `apply_full` basis probes; no copied solver implementation.
These are bounded workflow timings, not a matched-quality speed comparison.

No new predictions or model metrics were produced on the old test split. Full
integrity verification still reads/requalifies its existing reference arrays;
those values do not enter normalization, optimization or model diagnostics.
The 16 newly reserved final-test seeds have not been generated or inspected.
They remain a new-seed test of the previously observed held-out family, not a
new-family accuracy claim. See [ADR 0002](../../../decisions/0002-development-evaluation.md)
and [reservation](../../../final-test-reservation.json).

## Fixed validation comparison

Each row evaluates the same four validation cases using full residual and actual
f32 correction. Relative pressure error is mean(case RMSE / case target RMS);
relative residual is mean(case ||b-Ap||2 / ||b||2). Qualification also includes
the separately recorded predicted-divergence gate.

| Predictor | Relative pressure error | Relative residual | Max actual divergence, s^-1 | Qualified |
| --- | ---: | ---: | ---: | ---: |
| Zero pressure | 1.000 | 1.000 | 1.041 | 0/4 |
| Training-mean pressure | 4.879 | 5.643 | 2.074 | 0/4 |
| One diagonal inverse step, then gauge | 0.964 | 0.758 | 0.725 | 0/4 |
| Existing one-case CNN | 1.799 | 3.812 | 2.904 | 0/4 |
| Existing eight-case CNN | 4.384 | 8.623 | 4.464 | 0/4 |
| Rheon Jacobi PCG | 0 | 6.795e-11 | 1.177e-7 | 4/4 |
| Rheon SGS PCG | 1.997e-11 | 4.954e-11 | 1.177e-7 | 4/4 |

Both PCG implementations also qualified all eight training cases. Across the
12 development cases Jacobi used 76–77 iterations, SGS 29–30. Iteration counts
have different per-iteration costs; this is not a runtime or acceleration claim.
The simple approximations provide real comparison points and remain unqualified.
No warm-start+PCG comparison was attempted.

## Diagnostic findings

1. **Loss reduction was not numerical convergence.** The first one-case fit
   reduced normalized training MSE by roughly 10,000×, but still has 0.09656 Pa
   pressure RMSE (3.34% relative), 6.78% relative residual and 0.5211 s^-1 actual
   divergence on its fitted case. Its full residual exceeds the linear gate by
   6.78e8× and actual divergence exceeds its gate by 52,108×. Calling this a
   successful optimization smoke does not imply a sufficiently accurate target.
   Pressure MSE does not explicitly constrain A(p-target) or f32 divergence.

2. **A measured finite receptive-field limitation exists.** A single interior
   u-face at [7,5,4] produces a compatible two-cell RHS. Rheon's qualified
   pressure at [3,3,2] changes by -0.0066199434 Pa. Both CNNs change by exactly
   zero there because the observation's 3×3×3 patch and the gauge-anchor patch
   are identical to zero input. This is a structural counterexample, independent
   of checkpoint quality and train/validation/test distribution. It establishes
   a limitation, not the quantitative share of all observed error it causes.

3. **The learned map violates zero response and linear scaling.** For zero RHS,
   the eight-case CNN predicts pressure with RMS 4.245 Pa and maximum 9.959 Pa;
   the one-case CNN gives RMS 1.503 Pa. These are nonconstant, gauge-zero fields,
   so a constant gauge correction cannot remove them. On the first training
   field, the eight-case model's sign-symmetry and doubling defects are 4.886 Pa
   and 7.341 Pa RMS. The fixed-domain pressure inverse is linear; coordinate
   inputs, biases and tanh do not enforce that structure.

4. **The loss and distribution have material scale differences.** Polynomial
   targets have RMS 13.94 Pa versus 3.61 Pa for training white noise and 1.64 Pa
   for validation blobs. Polynomial cases contribute 93.7% of the zero-predictor
   training squared error despite equal case counts. This measures scale
   imbalance; it is not a causal ablation of optimizer gradients. Validation RHS
   RMS is 0.0912 of the aggregate training standard deviation. A nonzero learned
   prior is especially harmful for those small-amplitude validation fields.

5. **No normalization, ordering or gauge implementation bug was found.** Stored
   predictions replay exactly from both checkpoints. Normalization matches the
   original training-only receipt; tests prove changing validation/test/check
   values cannot affect it. X-fastest tensor ordering, gauge differentiation and
   gradients have focused checks. The fitted case's centered RMSE (0.0965589 Pa)
   nearly equals its ordinary RMSE (0.0965590 Pa), so its physics failure is not
   explained by a constant pressure offset. Strict whole-family splitting is
   intentional distribution shift; 8 training and 4 validation cases are tiny.

## Next useful bounded milestone

Do not add a warm-start API merely to conceal these diagnostic failures. First
admit one controlled global-context candidate with zero-input response enforced
by construction, and a pressure-plus-full-residual validation objective whose
units/scales are explicit. Use training and validation only; retain these fixed
baselines. Bound that first comparison to one architecture, one seed, at most
500 CPU optimizer steps and one validation evaluation, recording all compute.

Before running it, choose the actual global-context representation and derivative
path, add a gradient/zero-response check, and fix its stopping/selection rule.
A useful acceptance target is improvement over the diagonal baseline in both
validation relative residual and actual divergence without a pressure-error
regression; that still would not be direct reference qualification or acceleration.
Freeze model and procedure before materializing the reserved final-test seeds.

A subsequent optional upstream initial-guess API must implement the finite/shape/
gauge, original-b tolerance, r=b-Ap and Krylov-reset contract in ADR 0001 with both
preconditioners tested. No upstream patch was needed or prepared for this local
diagnosis, and no dependency revision or upstream source changed. Only after that
API exists can inference/transfer/correction overhead and matched-quality
warm-start versus zero-start solves be assessed.
