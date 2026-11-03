# Initial model card drafts

These are retrospective descriptions grounded in retained source and receipts.
Historical and fresh values below are explicitly distinguished. The complete portfolio
still needs cards, comparable fresh measurements and executable reproduction
checks before native project migration is complete.

## C106 and C107: half-width learned coarse pressure proposals

**Method and purpose.** Both predict pressure corrections using a half-width
encoder/decoder: widths 4/8/12/16, bottleneck 24, decoder 16/12/8/4. The captured
chart records 93,758 network parameters. A learned coarse proposal precedes each
fine neural correction. Its 63 gains cover Neumann cosine modes 0–3 without DC;
an equation-residual gate accepts a coarse proposal only when it does not
increase that residual. The full teacher is not used at inference. The proposal
contains two correction stages.

**Experiment.** These are siblings, initialized by channel-slicing C101 into the
same half-width checkpoint. C106 adds fine-correction distillation with weight
1. C107 is the control with that weight set to 0. Both used 2,400 updates, the
same training configuration, exact input packet hashes, normalization and sample
order. The captured trainer uses recorded physical pressure requests and disables
inherited auxiliary banks. Its inputs include nine dataset packets; exact scene
membership remains a reproduction detail to resolve from those packets and
split manifests.

| Alias | Trained checkpoint SHA-256 | Historical worst equation residual, 128 validation requests | Historical median CPU pressure-call time |
| --- | --- | --- | --- |
| C106, `v4-student50-distill` | `066bfab33685fe5a9e604cb3c3ed7b92408c699a68cc37d581f176a73b76be5f` | 0.00678818 | 4.46213 ms |
| C107, `v4-student50-control` | `9d50b22dcad28fe27d6f7ce0bbb0583741ff63a5f3b8684d1398a18e58014365` | 0.00102944 | 4.20801 ms |

**Outcome and lesson.** The control had the lower retained worst equation error
in this pair. This supports keeping the no-distillation arm as a baseline; one
pair does not establish that distillation is generally harmful. Each historical
case was timed once, excluding setup, sample assembly, verification and fallback.
These numbers do not establish complete simulation throughput, pressure-field
error or native physical qualification. C107 should have C101 as its documented
initialization relationship and C106 as a comparison baseline, rather than an
invented C106-to-C107 training edge.

**Reproduction material.** The captured projects are
`runs/pressure-methods-v4/v4-student50-{distill,control}-project-01/`.
Their train and restore outputs are under the corresponding `-study-01/`
directories. Use their own `restore/weights.npz` and captured source/configuration;
the same bytes also appear as initialization checkpoints in later fits.
`experiments/full-ai-next-priorities-v1/study_support.py:load_checkpoint` resolves
both named entries. The captured CUDA implementation has now completed all 128
original pressure requests with unchanged parameters and architecture. Fresh
worst equation residual / mean-free pressure relative L2 error were
0.00678757 / 0.0155048 for C106 and 0.00102944 / 0.00815329 for C107.
These are pressure-only, consumed-case accuracy checks; no fresh latency or
cross-device bitwise equivalence is claimed. Clean source/data regeneration
remains pending. [Fresh paired CUDA result](reports/pressure-GPU-comparison-01.json).
[Paired source and result audit](reports/c106-c107-pair-audit-01.json).

## C128: a-tail8 learned spectral pressure correction

**Method and purpose.** A final learned spectral correction follows a frozen
`a-head-calibration-s2` routed parent over C107 and its calibrated small-grid
head. The parent has two correction stages. The tail uses eight cosine modes
per axis, excluding the constant mode: 511 positive learned gains, conditioned
by a 3-to-64 tanh-to-511 shape head. Eight denotes modes, not simulation outputs
or neural correction stages. There are 128,938 parameters in total, with 33,982
trainable tail parameters. The tail accepts its pressure proposal only when the
actual finite-volume equation residual does not increase; it does not evaluate
an inverse eigenvalue at inference.

**Experiment and applicability.** The parent stayed frozen for 10,000
calibration updates with batch size 256. The captured prediction interface
bypasses this tail whenever any grid axis exceeds 16 cells. All 128 current
common pressure requests fall within that limit. The result therefore does not
establish an improvement on 32-cubed or larger domains.

**Outcome and lesson.** With the original checkpoint unchanged, fresh GPU worst
mean-free pressure relative L2 was 7.936785622882134e-5 and worst equation
relative L2 was 2.255120612837712e-4. The pressure maximum is the lowest among
the 127 available named entries on this consumed population; C99 has the lowest
equation maximum. Keep those metrics and the tail's grid-dependent branch
separate. There is no fresh speed, independent-validation, full-state or drift
claim. The smaller-grid result needs an explicit larger-grid branch comparison.

**Reproduction material.** Source and configuration are in
`runs/full-ai-next-priorities-v1/a-tail8-project-01/`, including `model.py`,
`tail_physical.py` and `calibration.json`. The original train and restore receipts
are under `runs/full-ai-next-priorities-v1/a-tail8-pair-01/`. Selected checkpoint
SHA-256 is `726fdb52ac48eddb559146f7433332dbda6af20c1e32b717384aa2ebd9e44469`;
original training Run is `281e36aeb9f54b29b3aa1481c0657209`. The native project
now retains the captured source definition and retrospective card; original
Run history and checkpoint payload custody have not yet been migrated.
[Common measurements](reports/pressure-common-score-01.json).

## C37, C44 and C53: earlier full-width pressure baselines

**Method and purpose.** All three use the physical-residual pressure interface
and encoder widths 8/16/24/32, bottleneck 48 and decoder 32/24/16/8, with SiLU
inside the CNN and float64 pressure reconstruction. C37 uses one fine correction;
C44 and C53 use two. Equal widths do not mean equal proposal work or training.
The captured contracts support sealed, uniform-density Neumann pressure requests
with cubic cells and at most 65,536 cells.

**Experiment.** C37, `shape-one-longer`, trained for 10,000 updates with the
canonical pressure/equation objective. C44, `extra-late`, trained for 5,000 updates
with intermediate-contraction training and additional feedback sources. C53,
`current-late`, also trained for 5,000 updates with intermediate contraction,
adding the current/band feedback configuration. These names describe experimental
intent; they do not establish training ancestry between the checkpoints.

| Model | Fresh GPU worst equation residual | Fresh GPU worst mean-free pressure relative L2 |
| --- | --- | --- |
| C37 | 0.0489171 | 0.0538221 |
| C44 | 0.00398262 | 0.0134507 |
| C53 | 0.00390012 | 0.00602154 |

**Outcome and lesson.** These are unchanged captured CUDA proposals on the same
128 consumed native requests, with all inputs/reference hashes paired. C53's
worst pressure error is below C107's, while C107 has lower pressure error on 122
of the 128 cases and lower equation residual on 125. Keep the difficult cases
visible rather than declaring a single winner from a different metric or
aggregate. No fresh speed or complete-rollout comparison is established.

**Reproduction material.** The projects are
`runs/water-shape-repeatability-v1/shape-one-longer/project/`,
`runs/water-late-residual-v1/extra-late/project/`, and
`runs/pressure-current-policy-v1/current-late-project-02/`. The original
21-checkpoint material manifest binds their CPU restores and source witnesses;
the new adapter restores parameters on CPU without a neural forward, then uses
each captured implementation's CUDA path. Exact entries, checkpoint identities,
per-density/grid results and output archives are linked in the
[paired pressure comparison](reports/pressure-GPU-portfolio-01.json).

## Cell-coordinate full-state execution: quoted g025 result

**Method and purpose.** This is an execution composition over restored transport
and pressure components, not evidence of a new weight fit. It changes sampler
coordinates on the N8 branch while preserving the N1 branch. The complete native
entry accepts CPU fields and returns independently owned CPU pressure, staggered
velocity and tracer fields, including transfers and synchronization.

**Quoted result.** Development case `diversity-development-g025-cpm100`, start
192, K4, 16 by 24 by 40 cells, 100 cells/metre, 60 outputs/second: 0.223700 ms per
output and worst pressure-field error 9.25003e-17 across the four outputs.
That is a 15,360-cell domain, not 32 cubed. The recorded reference uses one
internal step per output and is not a temporal-convergence certificate.

**Outcome and lesson.** The case executes the unchanged N1 branch: four internal
learned advances for four outputs. Its accuracy and timing do not measure the
N8-only coordinate change or skipped intermediate advances. On nonzero-error
outputs the effective pressure denominator is about 0.083–0.085, above the frozen
9.47e-8 floor, so the tiny pressure error was not created by that floor. Keep this
as an exact-case regression anchor, alongside harder N8 and mixed-domain cases.

**Reproduction material.** Composition entry:
`experiments/full-ai-data-diversity-v1/cell_coordinate_pressure_graph_v1.py`.
The historical accuracy and timing entrypoints are
`evaluate_cell_coordinate_blocks_v1.py` and
`benchmark_cell_coordinate_entry_v1.py` in that directory. Their registered
protocols bind transport/shared pairs, CUDA builds, reference windows, all input
identities and qualification outputs. A fresh accuracy replay reproduced all
1,018 output hashes and accuracy rows exactly; timing remains pending.
[Raw anchor audit](reports/anchor-audit-01.json),
[branch audit](reports/cited-branch-audit-01.json).

## Temporal frozen pilot: quoted continuous-varying result

**Method and purpose.** The frozen temporal composition tests explicit output
cadence and grouping of a 240-update-per-second reference. K counts complete
output intervals. Eager and CUDA-graph paths are separate execution variants over
the same restored parameters.

**Quoted result.** Training case
`long-v1-continuous_varying-oscillating-r0-cpm50`, fine start 0, K4, 16 by 40 by
24 cells, 50 cells/metre, 120 outputs/second, graph `fine_reference` mode:
0.426637 ms per output and worst pressure-field error 9.15849e-16. Four outputs
span 4/120 seconds and use eight internal advances. This is a training case,
not independent validation.

**Outcome and lesson.** The fine-reference grouping is an accurate retained
baseline at this scope. Its timing is not evidence that the cheaper grouped or
single-pass modes attain the same error. Compare modes at equal physical output
interval and K, and report full block latency and actual advances separately.

**Reproduction material.** Historical qualification:
`experiments/full-ai-data-diversity-v1/qualify_temporal_frozen_backbone_v3.py`;
timing: `benchmark_temporal_frozen_v2.py`. Their protocols bind the long temporal
audit, restored pairs and control/kernel builds. The fresh replay exactly
reproduced all 540 output hashes and accuracy rows, including the reduced-pass
failures. Only the fine-reference schedule retained the quoted accuracy.
Timing remains pending. [Replay comparison](reports/anchor-accuracy-reproduction-01.json).
[Raw anchor audit](reports/anchor-audit-01.json).

## C01 and C02: early global CNN and GroupNorm follow-up

**Method and purpose.** C01 trained a global pressure-correction CNN for 200
updates. C02 added GroupNorm to address the early model's behavior. The historical
global model has encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8,
global mean broadcast, gauge subtraction and a float64 pressure output around a
float32 CNN.

**Outcome and lesson.** The C01 handoff records zero learned correction on all
32 evaluated requests and full fallback, with no demonstrated solver
acceleration. C02's historical chart has lower equation error, but its complete
native evaluation still reported 0/32 raw passes and 32/32 fallback. Preserve
these negative results separately from later successful components.

**Availability and reproduction.** Their declared checkpoints are not located
by the current filename-based scan. The historical handoff describes optional
executor-only artifacts under a different catalog ID; neither copying a later
checkpoint nor regenerating data with the new solver can replace that evidence.
Check extensionless and external archives, then exercise the documented old-source
generation path if necessary as an explicitly new reproduction lineage. Original
checkpoint reproduction remains unresolved.
[C01 handoff](../global-pressure-training-v1/handoff.md),
[C02 local handoff and source generation](../global-pressure-groupnorm-native-dev-run-v1/local-handoff.md).

## Legacy global-v1: 335 learned spectral gains

**Method and purpose.** The fixed 8 by 7 by 6 sealed box uses orthonormal
tensor-product cosine modes, with one positive learned gain for each of the 335
nonconstant modes. Prediction transforms the pressure RHS into modes, applies
the fitted gains, reconstructs pressure and fixes the gauge. This is a learned
global linear pressure response. It does not generalize its geometry contract
to arbitrary grids or boundaries.

**Training and outcome.** The preregistered fit used eight training requests,
500 float64 Adam updates and a pressure-plus-equation objective. The frozen
checkpoint is `f97ffa3e083e5fbf8f7c8b69b79b2c6176af1ff3b01737664752bb66747d20b9`.
Its historical final evaluation retained 16 requests: raw learned pressure passed
15, while learned warm starts followed by Jacobi/SGS-PCG passed all 16. These are
different execution compositions, not interchangeable model accuracies. All
final seeds are exposed; a rerun cannot become a new untouched test.

**Lesson and reproduction.** Keep raw pressure and corrected-solve results
separate. Preserve [ADR 0003](../../decisions/0003-global-candidate.md),
`scripts/global_candidate.py`, and the original
`experiments/final/{freeze,outcome,review}.json`. Original source/build and
generation commands are retained in `experiments/final/README.md`; a new
reproduction must have a separate freeze. Fresh GPU execution remains pending.

## Legacy trajectory-v2: supervised full-spectrum modal pressure

**Method and purpose.** `scripts/trajectory_model.py` defines a float64,
32-cubed cosine response with 32,767 fitted nonconstant gains. It uses supervised
least squares on pressure labels rather than gradient optimization. Zero
optimizer steps therefore does not mean untrained. Its checkpoint is
`25533a2fbaa5ab0f5fb574dc6307a2b8d6a686deb7db867d0d03338d4d70d5a9`.

**Training and outcome.** The freeze records twelve training trajectories
(seeds 100–111), consumed validation and a later final rollout comparison. The
historical result reports a 4.25-fold rollout speedup against Jacobi and 3.69-fold
against SGS, under its stated resident-model/native-process timing boundary.
Those are historical complete-rollout comparisons, not fresh pressure-call or
current full-AI generator timings. The learned component replaces pressure;
the surrounding native solver still advances the other simulation fields.

**Lesson and reproduction.** A geometry-specific spectral prior is a meaningful
portfolio baseline even without an optimizer training receipt. Retain the
frozen source, training data identity, freeze/result/verification records and
commands under `experiments/trajectory-v2/`. Reproduce its original reference
epoch before comparing newer solvers. Fresh CUDA pressure and own-history
qualification remain pending.

`runs/trajectory-v1/model.pt` is a distinct earlier modal checkpoint. Its retained
container identifies the modal schema and n=32, but no adjacent final freeze was
found. The corresponding `training.npz` remains available. Exact training-source
and data binding require further evidence; do not merge it with trajectory-v2
because their architectures resemble one another.
