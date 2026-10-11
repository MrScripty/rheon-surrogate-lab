# ADR 0003: One bounded global spectral candidate and conditional warm start

Status: accepted for this bounded run. Author: Puma. Date: 2026-10-11.

Parent explicitly authorized one global-context candidate (maximum 500 CPU
optimizer steps), training/validation only, with zero response and scale balance.
Final reserved seeds remain ungenerated. Publication remains paused.

## Frozen candidate before the run

Use fixed orthonormal tensor-product cosine modes on the 8x7x6 grid, omit the
constant mode, learn one positive gain for each of 335 remaining modes:

    c = U^T b
    p_raw = U diag(exp(log_gain)) c
    p = p_raw - p_raw[0]

This is a global linear surrogate, not the previous local CNN. Zero response,
linearity and the gauge follow by construction. The basis is a strong
geometry-specific prior suitable for this uniform sealed box, not arbitrary
boundaries, coefficients, free surfaces or CFD accuracy. Do not inject analytic
Laplacian eigenvalues, exact inverse coefficients or a fitted validation response.

Initialize every gain to 1/mean(diag(A)), with A obtained from the pinned public
Rheon apply_full basis products. That export is used for the training loss and
its gradient, not as an inverse or an inference-time solver. Infer from b, U and
learned gains only. Training targets remain the existing qualified pressures.

Train all eight training cases in one batch, exactly 500 Adam steps, fixed lr
0.05, beta=(0.9,0.999), epsilon=1e-8, float64. No random initialization, sweep,
restarts, early stopping or checkpoint selection. Record training losses every
50 steps. Evaluate validation once at final step 500. No test predictions.

Equal case weighting addresses target-scale imbalance:

    L = mean_cases[0.5*||p-p_ref||2^2/||p_ref||2^2
                 +0.5*||A p-b||2^2/||b||2^2]

Denominators are from training cases only; require positive finite values.
This is a pressure-plus-residual training objective, not the stricter final
solver stopping contract. No validation value determines normalization or updates.

## Predeclared decision and evidence

Compare final validation pressure error, full relative residual and actual f32
corrected divergence with zero, diagonal-once, Jacobi-PCG and SGS-PCG baselines
already measured in development-diagnostic.json. A useful initial-guess signal
requires mean relative residual below diagonal's 0.7581104033832886, maximum
actual divergence below 0.7246515639126301 s^-1, and mean relative pressure error
no worse than diagonal's 0.9640346380051443. Record failure without another run
or a changed threshold. Raw reference-PCG qualification is NOT required for
useful initial guesses; ultimate acceptance is corrected solve quality/cost.

Unit checks: orthonormal basis, zero/sign/scaling/gauge construction, analytical
loss gradients against finite differences, and pinned operator export agreement.
Record training and inference scope, counts, source and artifact hashes.

## Conditional local upstream work

If that signal passes, prepare a minimal initial-guess API patch against pinned
Rheon in a separate local branch/checkout. Do not edit Rheon main, publish upstream,
or change this lab's Cargo dependency. Keep patch, base commit and tests explicit.
Preserve old zero-start entry points and behavior. New solve starts from finite,
exactly shaped, explicitly gauge-valid pressure; uses full r=b-Ap and the original
b in stopping tolerances, and resets Krylov scratch. Test both preconditioners,
zero/exact/poor/shifted/nonfinite/wrong-length guesses, cancellation and reuse.

A separate local probe may link the patched checkout and compare zero-start and
prediction+PCG under identical linear/predicted-divergence gates and actual f32
correction. Count inference, transfer, correction and verification overhead. Raw
loss, raw qualification or fewer iterations alone is not an acceleration claim.
