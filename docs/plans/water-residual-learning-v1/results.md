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

The two-update CUDA/CPU smoke passed and both Runs completed/drained, reopened
and redownloaded with whole-artifact parity. Full nonlinear training completed
10,000 updates at 91.44 updates/s in the optimizer interval; fresh CPU restore
preserved exact predictions. Fitted native residual is 0.0005193531068, with
0/1 physical qualification. Validation CPU diagnostic median is 0.04441808198,
maximum 0.2910017402; native validation confirms the same errors and 0/128
physical qualifications. All native child groups drain. Learned warm-start PCG
uses 15,963 iterations including fallback versus 11,867 for native temporal PCG;
this candidate does not accelerate the pressure solve. This improves
on the matched one-frame baseline but remains above the earlier large-cohort
median 0.0385793. Eight raw proposals reach median 0.01134085; all 128 rounded ideal
float32 corrections pass the equation-only control. All 23 worker cores were used.

The independent cube trajectory had 15 native warm-up steps and 26 actual model
predictions before the unchanged physical hard stop at step 41; 0/26 learned
proposals qualify, 41/41 native references qualify. Worst actual divergence reaches
92.39115. The raw failure and drained child are retained; this is accumulated
model error, not missing previous-pressure context. The native/native control
qualifies all 49 post-warm-up steps of a 64-step run. Two rectangular force scenarios
also have separate native controls and will evaluate learned checkpoints.

The matched linear 10,000-update trial completed fresh CPU restore/development
inference. Fitted residual is 0.0004192441490 and validation diagnostic median
0.04773792218. Superposition error falls to 2.23e-7, but eight raw proposals
increase median equation error to about 222.126. Linearity alone does not
establish a stable approximate inverse. Its native validation is being checked.
All planned CUDA fits have completed: recurrent 5,000 updates, physical-feedback
10,000 updates and matched single-correction 5,000 updates. CPU qualification
and independent trajectories remain in progress. A varied 16-simulation 32-state
cohort has been prepared without starting training; expansion remains gated on
credible known-case fitting. See execution ledger and per-trial reports for
terminal statuses as they become available.


Completed nonlinear trial: [full method/result record](reports/residual-nonlinear.json),
including all three independent development trajectories and their hard stops.


The separately qualified native paired controls pass all 64 reference frames and
49 post-warm-up projected steps for each cube, elongated [24,16,12] and compact
[12,8,10] scenario. Every domain retains one cubic spacing. These are development
scenarios with different force location, strength, swirl and fluctuation, not
an untouched final test. Their receipts and reusable JSON configurations are
retained. The next 16-simulation cohort contains 32 real native snapshots with
actual preceding pressure; it is prepared, not trained.
