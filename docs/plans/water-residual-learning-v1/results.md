# Residual learning results

Methods are versioned before managed source capture. No weights or datasets are
in Git; Tuldok retains Run/checkpoint custody and verified local downloads remain
under runs/water-residual-learning-v1. Each completed trial gets a small immutable
report with exact configuration, source/input hashes, sampling order, checkpoints,
training trace and CPU/native evaluation. No reserved final test has been used.

## Method and controls

One native 16x16x16 water state at frame 15, with actual preceding native pressure;
same model seed 20261015, sampling seed 2026102501, Adam and original 10,000-update
schedule. Initial interventions change only input representation, then activation
and offsets. Float32 CUDA parameters/gradients/moments; deterministic eager,
TF32/AMP disabled; physical assembly/reconstruction f64 as before. CPU restore is
fresh and optimizer-free. The fit is a capacity diagnostic. The 64-parent,
128-state validation is development evidence, not untouched qualification.

Physical residual input: q=(rho/h)*(b-A p_previous), one full-grid channel divided
by train-only q RMS. The output is float32, multiplied by training pressure RMS,
cell-zero gauge fixed and added to supplied base. Grid shape identifies the sealed
Neumann stencil. Density/time-step/coordinate maps are removed from the network;
one cubic cell spacing remains physical metadata. This contract is restricted to
uniform-density sealed full-fluid domains; no masks, obstacles or free surface.

Linear variant removes all convolution offsets and SiLU activations, retaining
linear pooling, resizing and skip/global-average operations. Test scaling, sign
and superposition separately from pressure accuracy. Recurrent variant applies
shared parameters with the actual residual recomputed between proposals.

## Baseline CPU diagnosis

The retained no-GroupNorm checkpoint on 128 validation states has median residual
0.0828159014 and median superposition discrepancy 0.463623591. Eight raw proposals
reach median 0.0755236577, so iteration alone does not establish convergence.
Analytic correction using the lowest 2/4/8 cosine modes per axis gives median
0.0756360062 / 0.0488392523 / 0.0128119390; the full analytic pseudoinverse gives
2.03706e-15. These are separately labelled f64 algorithmic controls, not learned
accuracy or full physical qualification. Rounded float32 ideal learned corrections
pass the equation-only gate on all 128 states. This motivates testing input and
linearity first, then training coarse/recurrent correction rather than assuming
that model size or data count alone is responsible.

CPU audit used 23 single-thread workers with distinct core affinity, 22 observed
cores for this small workload, no CUDA initialization, 7.59 seconds including pool
startup. The independent spectral regression recovers a known rectangular-grid
Neumann cosine eigenfunction and checks low-mode truncation.

## Managed experiments

The two-update physical-residual smoke has completed CUDA optimization; its fresh
CPU restore is currently being verified. Full nonlinear/linear trial source
captures are assembled but have not yet been executed. See execution ledger and
per-trial reports for terminal statuses as they become available.
