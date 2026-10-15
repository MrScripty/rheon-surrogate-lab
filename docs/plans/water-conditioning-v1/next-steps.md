# Prepared next model experiments

These are the next scientific controls, not additional runs already completed.
All optimization remains CUDA float32, with fresh CPU restore, the unchanged
1e-6 residual target and native physical gates. Keep source/configuration and
results versioned. Use qualified faster execution; keep its numerical parity
evidence separate from accuracy claims. Continue normal Tuldok Run custody and
project observation, with independent CPU work prepared during GPU optimization.

The bounded controls improved the fitted frame but did not solve it. Merely
increasing simulation count is therefore not the next deciding experiment.
Removing GroupNorm helps fitting, but does not establish normalization as the
only cause. The existing network already has several resolution levels and a
global pooled feature; do not describe it as having no global communication.

## 1. Physically conditioned residual representation

For the current uniform-density, cubic-cell, sealed pressure operator,
`A = (h / rho) L`, where `L` is the dimensionless grid Laplacian with the same
wall stencil. The correction equation is `L delta_p = (rho / h) residual`.
This follows from the face coefficient used by the checked native/model
operator, not a claim about arbitrary materials or boundaries.

Prototype one residual channel with scalar physical scaling outside the network.
Use the grid shape and one cubic spacing value; preserve pressure gauge and
reconstruction with the supplied base pressure. Density and timestep maps and
previous pressure need not be independent features for this restricted operator:
density sets its scale, timestep/forcing already affect the assembled RHS, and
base pressure is used to compute residual and reconstruct absolute pressure.
Do not remove these from the old deployment contract or reinterpret historical
cold-start failures. Changing the representation requires a new trained model
and adapter, plus explicit history/warm-up qualification.

Compare the current best fitted architecture against this new representation
on the same native frame before broadening data. Record exact preprocessing,
output scaling, pressure/gradient/equation errors and CPU numerical error. A
float32-rounded reference control remains necessary for each newly admitted grid.

## 2. Enforce the linear correction law

At fixed grid/operator, correction is linear in residual. Test positive/negative
scaling, zero input and superposition. Per-sample RMS plus output reconstruction
already enforces positive homogeneity up to arithmetic error; it does not by
itself enforce sign symmetry or superposition.

Compare a linear residual path (no activation normalization, no nonlinear
activations, no offset biases) with the nonlinear alternative. Keep the physical
representation and loss fixed for this comparison. Check the actual pressure
target, not just algebraic properties: a zero function can satisfy linearity.
Consider an optimizer comparison only with the representation fixed; keep
learning-rate/Adam-state effects distinguishable from architecture changes.

## 3. Measure coarse correction and repeated solves

Use the measured repeated-correction baseline and its minimum-residual step
control. Add a separately labelled coarse or spectral correction for the sealed
uniform operator to measure which error scales remain. An analytic correction
is algorithmic work and must not be counted as a pure AI success. Then compare
learned coarse correction or shared repeated learned updates, with residual
recomputation in the training loop. Require convergence measurements and total
CPU solve time including operator work, inference, transfers and fallback.

## 4. Resume varied-physics training after fitting is credible

Advance from one frame to a small multi-frame/multi-domain cohort, then the
existing larger training population. Match sample exposure as well as time when
comparing batch sizes. True GPU batches require equal grid dimensions; use
explicit shape buckets and report their sampling law, rather than padding the
domain and silently changing boundaries or normalization. The batch-of-four
throughput fixture repeats a frame and is not evidence of four independent
training simulations or better accuracy.

Report several seeds, error tails and independent trajectories after warm-up.
The prior reserved populations have been evaluated; reserve a new untouched
population before final candidate selection. Preserve existing results and do
not relax gates because a learned candidate fails.

## Execution follow-up

The current Tuldok supervisor is serial. Parallel managed GPU trials require an
owned scheduler/lifecycle change with per-device admission, resource accounting,
cancellation/reaping, restart and shutdown checks. Do not bypass that owner with
private training processes. Profile its verification/startup overhead separately
before increasing concurrency; setup dominates these short probes. Persistent
CPU workers improve pressure-state throughput, while three eight-thread workers
were not the fastest measured layout. Neither observation establishes full
simulation speedup; benchmark the actual native/AI trajectory workload next.
