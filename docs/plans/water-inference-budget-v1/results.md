# Inference budget and scaling results

The best new development policy uses the completed single-pass-trained
checkpoint with repeated, adaptive CPU corrections. It passes the fitted state,
all 128 development states and 854/854 AI trajectory steps, averaging 12.6015625
network evaluations on the development states. **One-evaluation accuracy remains
unqualified.** The policy changes both learned weights and stopping granularity.
No native pressure refinement contributes, and no deployed UI model changed.

## Model sizes

| Model | Trainable parameters | Actual checkpoint bytes |
| --- | ---: | ---: |
| Current one-channel learned corrector | 370,016 | 1,485,452 |
| Original successful 32-cubed modal model | 32,767 | 272,149 |
| Earlier ten-channel neural model | 372,377 | 1,500,186 |

Current versus modal: 11.29 times the parameters and 5.46 times the file size.
The modal checkpoint uses float64 gains plus a fixed cosine-basis buffer for
one fixed grid. Current neural parameters are float32 and shared between every
correction. Repeating inference does not duplicate weights. File sizes exclude
runtime/data/optimizer state; formats differ. See [inventory](reports/model-sizes.json).

## Training outcomes

Both completed fits use one actual captured CFD frame, 5,000 CUDA updates,
5,000 real-frame exposures and 5,000 sampled analytical examples. These examples
are pressure-operator patterns, not extra independent fluid simulations. Initial
parameters, sample order, bank and train-only normalization match their retained
four-pass controls. Terminal updates are fixed in advance. All evaluation remains
development evidence; no untouched final test was consumed.

| Single-pass training | Fitted residual | Development median | Worst | Physical passes with one evaluation |
| --- | ---: | ---: | ---: | ---: |
| Mode feedback | .00737933 | .0164437 | .106447 | 0/128 |
| Feedback + output scale, tenfold lower learning rates | .00781890 | .0258949 | .107453 | 0/128 |

Both raw policies finish the three short trajectories but qualify 0/147 AI steps.
The initial combined single-pass fit at learning rate .001 fails with non-finite
gradient norm after at least 2,400 updates (last progress loss 77,382.87). It is
drained, with no published model artifacts; its exact terminal update is unavailable.
The separate .0001 follow-up completes normally. Do not attribute its comparison
with the four-pass control to one factor: pass count and learning rate both change.

Two full fits complete 10,000 actual CUDA updates in six completed/drained managed
stages. The failed/drained Run is a seventh stage; its known update lower bound
makes all-attempt compute at least 12,400 updates. Complete CPU restores,
reopened Runs and whole-artifact redownload parity pass. Training interval rates
are 45.7144 and 47.3483 updates/s; runtime/preflight time is separate. The new
one-evaluation models perform less neural work per update than the controls.

## Iterative inference: counted work and accuracy

| Weights and inference policy | Development mean evaluations | Range | Physical development passes | Qualified trajectory steps |
| --- | ---: | ---: | ---: | ---: |
| Earlier mode-feedback weights, fixed | 32 | 32 | 128/128 | 854/854 |
| Earlier weights, adaptive in blocks of four | 14.96875 | 8–28 | 128/128 | 854/854 |
| New single-pass-trained weights, adaptive after each evaluation | 12.6015625 | 8–26 | 128/128 | 854/854 |

The last policy uses 60.62% fewer neural evaluations than fixed 32 and 15.81%
fewer than earlier adaptive inference. It changes both weights and stopping
frequency; a same-granularity ablation remains future work. The fitted state
passes with residual 5.11824e-7. Development median is 6.88415e-7, worst 9.91896e-7.
These adaptive candidates satisfy the unchanged 1e-6 profile and physical limits;
none satisfy the stricter original native 1e-8 equation profile at their stopping
points. The earlier fixed-32 reference passes 121/128 under that stricter profile.
Greater numerical oversolving by the fixed reference is not hidden as equal error.

For the new policy, extended cube/rectangular/compact runs average 9.24, 9.35
and 10.09 evaluations per AI step. Their final pressure-field relative L2
differences from native are 9.88e-8, 8.50e-7 and 1.77e-7 respectively. Those are
whole-field norms, not maximum per-voxel errors. Windows remain about four seconds.
Native controls and fixed timestep/force limits are preserved. No native cleanup
enters learned predictions; evaluator refinement methods are separate benchmarks.

Latest checkpoint: Tuldok training Run `406b606cd7a8431db744e86b2631f28e`, weight
SHA256 `393fac2cb82b5d259c3aedaa3b9ab7376d99d9896cbfb7535a52a0ac3a6432c7`.
Its raw one-evaluation failure remains in the immutable trial record. The new
iterative evidence is a separate [addendum](reports/single-pass-adaptive-evidence.json).

## Fresh paired CPU checkpoint timings

One CPU core, fresh Python processes per episode, three alternating repetitions,
15 native history steps, one excluded neural warm-up step, seven measured steps
per repetition. Every native control and all 144 AI steps qualify.

| Scene | Earlier adaptive | New iterative policy | Reference/candidate time | Native alongside candidate |
| --- | ---: | ---: | ---: | ---: |
| 16 x 16 x 16 | 81.18 ms | 70.01 ms | 1.16x | 14.39 ms |
| 24 x 16 x 12 | 82.75 ms | 67.09 ms | 1.23x | 18.51 ms |
| 12 x 8 x 10 | 41.14 ms | 26.54 ms | 1.55x | 2.79 ms |

The new combination is faster than earlier adaptive inference on these measured
scenes, but still takes 3.62–9.50 times native track time. These are whole
Engine.advance times including physics, IPC and the Python callback, not isolated
pressure-kernel timings or a deployment performance guarantee. Native uses its
existing 1e-8 equation tolerance, AI its existing 1e-6 profile. Full repeated values,
weights, commands and physical results are [retained](reports/checkpoint-comparison.json).

## Grid scaling: earlier checkpoint, separate study

Seven actual physical setups, three repetitions per policy: 42 paired candidate
episodes plus seven fresh native controls. All 336 AI benchmark steps qualify.
The study completes in 808.88 seconds. Physical volume and spacing are separate
sweeps; the common 16-cubed baseline is measured once. All voxel cells stay cubic.

| Grid, fixed 1-metre cube | Fixed-32 time / native | Adaptive time / native | Adaptive evaluations |
| --- | ---: | ---: | ---: |
| 16 cubed | 13.27x | 5.66x | 12 |
| 24 cubed | 18.97x | 7.57x | 12 |
| 32 cubed | 7.22x | 4.67x | 20 |
| 40 cubed | 9.95x | 9.36x | 32 |

At fixed 16 cells/metre, 24/32/40-cubed domains have side lengths 1.5/2/2.5
metres; their adaptive/native time ratios are 4.23/4.99/9.83, using 12/20/32
evaluations. Neither sweep demonstrates monotonic relative scaling or a speed
crossover. At 40 cubed, adaptive stopping saves no network work; small timing
differences at that point are within the observed episode variation. Above
65,536 cells remains unsupported by the current contract.

This scaling matrix belongs to the **earlier checkpoint**, not the newest
single-pass-trained weights. New larger-grid qualification/timing for the latter
is still future work. Larger-grid episodes here contain only eight AI steps
after warm-up, so they do not establish long-horizon generalization. Process
high-water RSS is about 849 MiB; child high-water includes the fork/exec lifecycle
and does not isolate native kernel memory. See [scaling JSON](reports/scaling.json),
[CSV](reports/scaling.csv) and [executed sources](reports/scaling-source-evidence.json).

## Retention and next work

Six focused budget/feedback checks, two real process-cleanup checks, three altered
real-receipt rejection checks, source/initialization parity, fresh CPU restore and
all independent native physical comparisons pass. Separate 72-mode mixture
probes diagnose residual one-pass fitting error without calling it CFD coverage.
All task-owned children are reaped; weights and datasets remain outside Git.

See [method](reports/methods.md), [derived index](reports/summary.json),
[comparison CSV](reports/comparison.csv), [ledger](execution-ledger.md) and
[prepared next steps](next-steps.md). Multi-grid analytical feedback, finer stopping
ablations, CPU profiling and broader CFD training are prepared follow-ups. Their
inference budget and capture/cache contracts must stay explicit. No architecture
replacement, broader training campaign or production solver deployment occurred.
