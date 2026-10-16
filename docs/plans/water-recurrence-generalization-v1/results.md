# Recurrence and unseen-scene results

The shape-feedback checkpoint is the stronger generalization reference for the
next learning experiment. It passes all 16 previously untouched simulations and
all 1,808 AI steps; the preceding reference passes 11/16 simulations and 1,271/1,808
steps. Selection was frozen at 02:15:26 UTC before either final population ran.
The final population is now consumed and must not be reused as untouched evidence.

On these final simulations, mean network evaluations fall from
16.707 to 9.556
(42.8% fewer).
The reference hits the 32-evaluation cap 540 times; the candidate never does and
needs at most 12. Reference averages include its failed/capped cases; this is
work accounting, not an isolated timing or matched-success latency comparison.

Worst pressure-field relative L2 deviation versus native falls from
8.842312e-05 to 4.348436e-06
(20.33× lower).
Median final pressure deviation falls from 6.845026e-07
to 2.909686e-07. Pressure-field deviation is
distinct from the unchanged1e-6 equation-residual acceptance target.

## Complete comparisons

All rows pass128/128 familiar teacher states with adaptive inference. These are
CPU model evaluations, checked individually and capped at 32 per physical step.
Training recurrence does not specify the inference call count.

| Training recipe | Evaluations during training | Familiar mean / max evaluations | Fixed 8 passes | Fixed 8 median / worst residual |
|---|---:|---:|---:|---:|
| cohort-two-reference | 2 | 8.883 / 12 | 46/128 | 1.213e-06 / 4.533e-05 |
| cohort-four-control | 4 | 12.266 / 18 | 0/128 | 2.314e-05 / 3.010e-04 |
| cohort-one-control | 1 | 8.914 / 14 | 69/128 | 7.322e-07 / 1.631e-04 |
| cohort-two-seed2 | 2 | 9.047 / 14 | 50/128 | 1.149e-06 / 1.211e-04 |
| cohort-two-seed3 | 2 | 10.086 / 14 | 13/128 | 4.135e-06 / 5.014e-05 |
| cohort-two-shape-feedback | 2 | 9.414 / 12 | 26/128 | 2.586e-06 / 3.227e-05 |

The two extra seeds reproduce a familiar-population improvement over the older
single-frame reference 12.602 evaluations: the current two-evaluation recipe
ranges 8.883–10.086 across three seeds. They do not establish robust shape
generalization: one seed hard-stops on33×17×9 at step 32 despite15 native warm-up
steps. One-evaluation training nearly matches the control's familiar mean at
roughly half its forward work, but has an unstable fresh trajectory. Four-
evaluation training is worse at this schedule and is not selected.

| Population | Previous reference | Shape-feedback candidate |
|---|---:|---:|
| fresh development | 6/8 scenes; 542/648 AI steps | 8/8 scenes; 648/648 AI steps |
| long development | 3/3 scenes; 1491/1491 AI steps | 3/3 scenes; 1491/1491 AI steps |
| prior grids | 7/7 scenes; 56/56 AI steps | 7/7 scenes; 56/56 AI steps |
| final holdout | 11/16 scenes; 1271/1808 AI steps | 16/16 scenes; 1808/1808 AI steps |

The shape-feedback candidate also passes all 120 independent signed operator
probes. Its4,003 autonomous AI steps across the listed populations pass physical
gates. This is finite tested coverage, not proof for all shapes or arbitrarily
long simulations. Every native control qualifies. Failed learned trajectories
retain their actual terminal frames; no native cleanup contributes to success.

## Method and tradeoffs

Five new fits complete 25,000 CUDA updates and 15 real managed stages. Each fit
has 5,000 updates. Same-seed one/two/four controls and the shape-feedback trial
retain identical initial parameter inventories, real sampling order and train-
only normalization. Additional-seed trials intentionally change initialization
and sampling. There is no architecture, parameter-count or precision increase.

Training still uses 16 real sealed-water simulations and 32 captured states.
The coverage trial changes the analytical bank from five grids/640 examples to
eight grids/1,024 examples, adding 17×13×9,33×17×9 and 9×25×13 at 8 cells/metre.
These are manufactured Neumann pressure-operator examples, not additional CFD
simulations. Real training statistics determine amplitude; no final-test data
is used. Uniform grid exposure changes from 1,000 to 625 draws/grid. Cubic voxel
spacing remains scalar and physical extent follows cell count.

| New CUDA trial | Updates/s | Optimizer seconds | Forward cell evaluation proxy |
|---|---:|---:|---:|
| cohort-four-control | 12.46 | 401.34 | 500,278,176 |
| cohort-one-control | 42.49 | 117.67 | 125,069,544 |
| cohort-two-seed2 | 22.71 | 220.12 | 250,392,900 |
| cohort-two-seed3 | 18.77 | 266.39 | 250,395,820 |
| cohort-two-shape-feedback | 24.60 | 203.23 | 173,120,838 |

The five optimizer intervals total 1,208.75 seconds (20 minutes 8.75 seconds).
The selected shape-feedback fit takes 203.23 seconds (3 minutes 23.23 seconds)
for 5,000 updates. These durations are retained per trial in
[comparison.csv](reports/comparison.csv); their original start/finish timestamps
are retained in [summary.json](reports/summary.json)'s `optimizer_intervals`.
They exclude project preparation, queue/approval wait, CPU checkpoint restore,
validation and trajectory qualification. A complete controller wall-clock total
and an isolated latest-checkpoint inference latency comparison were not measured
for this sequence; neither can be inferred from optimizer time or call counts.
The next experiments must retain the separate timings specified in
[next steps](next-steps.md#timing-records).

Equal updates are not equal compute. These rates include the declared optimizer
interval under concurrent preparation/CPU work and are not isolated GPU
benchmarks. The shape trial's173,120,838 forward cell evaluation proxy is below
the control's250,139,088 because the added grids redistribute sampling.

There are material regressions: familiar mean work rises 8.883→9.414; fixed 8
passes fall 46→26, although the worst fixed 8 error improves. Prior-grid mean work
rises 8.911→10.482, including40³: 10→11.25. All prior physical gates still pass.
For the separate512-step lower-force population, worst pressure-field deviation
increases3.702e-6→6.140e-6 while all 1,491 AI steps pass. This is a generalization
gain, not universal accuracy or speed improvement. No new native-versus-AI
latency crossover is claimed.

## Execution and retained material

The one-evaluation CUDA fit succeeds, but its original controller times out on
CPU restore approval. That failed workflow remains retained. CPU-only recovery
uses the original trained checkpoint, normal fresh managed approvals and zero
extra optimizer updates. The explicit recovered custody index verifies all
three real Runs and repeated downloads. This is not a repeated training attempt.
Preparation/approval latency remains a separate Tuldok profiling concern.

The result index verifies disjoint optimizer intervals and 222 native trajectory
children exiting0/reaped. CPU diagnosis uses up to 23 independent single-thread
workers with a shared pool lease for later controllers; GPU training uses CUDA.
No model weights or simulation datasets are committed. Full artifacts remain in
the existing Tuldok catalog and ignored runs/. Historical records are preserved.

Start with [summary](reports/summary.json), [comparison CSV](reports/comparison.csv),
[frozen selection](reports/final-selection.json), [methods](reports/methods.md),
[checkpoint contract](model-contract.md) and [next steps](next-steps.md).
Rheon's deployed adapter and solver architecture are unchanged.
