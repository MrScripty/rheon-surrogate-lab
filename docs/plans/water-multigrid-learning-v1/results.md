# Multi-grid pressure learning results

The 16-simulation cohort improves fixed-budget equation accuracy and reduces
adaptive inference work on the tested development population. All physical
acceptance targets stay unchanged. This is a laboratory development candidate;
application deployment and fresh final-population evaluation remain separate.

Three CUDA fits plus two pipeline smokes completed 15,020 optimizer updates.
All13 managed stages completed/drained, reopened and redownloaded identically.
The cohort contains16 distinct native water simulations and 32 real states;
analytical Neumann examples across five grids are additional operator training,
not additional fluid simulations. All candidates retain370,016 float32 parameters.

## Accuracy at equal work

On the same128 teacher states after a fixed eight-evaluation policy:

| Measure | Previous checkpoint | Cohort checkpoint |
| --- | ---: | ---: |
| Median relative equation residual | 6.92990e-6 | 1.21265e-6 |
| Worst relative equation residual | 1.67301e-3 | 4.53286e-5 |
| Physical passes at the unchanged target | 11/128 | 46/128 |

Median error improves5.71x and worst error36.91x. Eight evaluations remain
insufficient for every case. Raw one/two-evaluation outputs remain unqualified.
With adaptive stopping and the same maximum 32 network evaluations:

| Checkpoint training | Development passes | Mean evaluations | Worst evaluations |
| --- | ---: | ---: | ---: |
| Previous single-grid reference | 128/128 | 12.6016 | 26 |
| Multi-grid, one evaluation | 128/128 | 13.4688 | 23 |
| Multi-grid, two evaluations | 128/128 | 12.4141 | 19 |
| Multi-grid, two evaluations,16 simulations | 128/128 | 8.8828 | 12 |

The cohort reduces mean work29.51% and worst work53.85% versus the preceding
reference. Its32 fitted states all pass, at mean 9.125/max11 evaluations. Early
and final fitted-state replays reproduce identical numeric inventories.

The equal-stopping ablation also resolves prior attribution: earlier weights
need13.4375 evaluations versus12.6016 for the preceding latest checkpoint,
a6.22% training/weight improvement. The previous larger comparison also changed
stopping granularity. Multi-grid one-evaluation training regresses mean small-
state work while helping larger grids; it is retained as a measured tradeoff.

## Grid scaling and actual CPU latency

All seven short physical grid settings pass, with independently qualified native
controls. Fixed-volume cube means (23 steps,15 native history steps):

| Cube | Previous | Multi-grid one | Multi-grid two | Cohort two |
| --- | ---: | ---: | ---: | ---: |
| 16³ | 10 | 12.75 | 11 | 8 |
| 24³ | 10.875 | 13.25 | 11.875 | 8.375 |
| 32³ | 17.875 | 13.5 | 11.875 | 8.875 |
| 40³ | 22.875 | 13.625 | 12.125 | 10 |

Both fixed-volume and fixed-spacing sweeps preserve cubic cells. Cohort40³ work
falls56.28%. No extrapolation beyond the 65536-cell limit is admitted.

Final timings use five scenes, four checkpoints, three alternating repetitions,
fresh processes,15 native history steps and one excluded neural warm-up step.
There are 21 retained timing samples per checkpoint/scene. Task training and
numerical diagnostics finished before the serial studies. All960 AI steps and
all native controls pass; all 130 native trajectory children are reaped/exit0.

Median cohort whole-track time, including physical preparation/projection and
Python/JSON IPC; native receives the same CPU affinity/thread budget:

| Grid | Previous AI,1 thread | Cohort AI,1 thread | Previous AI,4 threads | Cohort AI,4 threads | Paired native,4-thread budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16³ | 71.09 ms | 57.50 ms | 44.01 ms | 41.54 ms | 14.36 ms |
| 24³ | 443.29 ms | 343.51 ms | 194.80 ms | 169.21 ms | 62.22 ms |
| 32³ | 685.30 ms | 463.62 ms | 461.64 ms | 262.30 ms | 181.21 ms |
| 40³ | 3221.29 ms | 1470.35 ms | 1442.14 ms | 701.78 ms | 419.11 ms |
| 24×16×12 | 67.28 ms | 67.08 ms | 47.38 ms | 45.49 ms | 18.67 ms |

At40³, new/previous AI speedup is 2.19x with one thread and 2.05x with four.
Four threads improve cohort latency versus one on all five sampled scenes.
Native remains faster: the cohort's four-thread tracks take1.45–2.89 times the
paired native time. Relative advantage is not monotonic with grid size, and the
rectangular one-thread case is essentially unchanged. This is paired whole-track
evidence with native 1e-8/AI1e-6 equation profiles, not identical-tolerance or
isolated pressure-kernel timing. Native's actual parallelism is implementation-
dependent; the comparison controls resource allocation.

## Finite trajectory drift and limits

The reference and all three new candidates each pass1,491 AI steps across three
512-step trajectories (8.5333s) after warm-up. Quarter-strength internal forcing
keeps native controls valid; this does not extend the original full-force claim.
Cohort maximum pressure-field relative L2 differences versus native are 3.95e-6
(cube),1.08e-6 (rectangular) and 2.74e-6 (compact). The reference values are
5.70e-7,1.24e-5 and 3.50e-6. Drift is therefore mixed, even with every equation/
physical check passing. The1e-6 equation target is not a1e-6 pressure-field bound.
Final/intermediate samples and complete receipt identities are retained in
[drift JSON](reports/pressure-drift.json) and [CSV](reports/pressure-drift.csv).

The multi-grid bank uses 640 signed entries,1,000 draws/grid during each full
fit. The two controlled one-frame fits preserve initialization, real sampling
and normalization exactly. Larger auxiliary grids increase the voxel-evaluation
proxy3.4125x; two-evaluation training doubles forward neural work. The cohort
also changes real population, train-only normalization and analytical amplitude,
so its improvement cannot be attributed to sample count alone. No final reserved
population is used in this tranche. No global contraction, long-term stability,
obstacle/free-surface or variable-density accuracy is claimed.

CPU profiles identify convolution/tensor-fill cost. Channels-last observations
are mixed, so no rollout layout migration was adopted. Thirty-three focused
source/scheduling/lifecycle tests pass, plus actual native/AI execution fixtures.
Historical results remain intact; weights/datasets stay outside Git.

See [summary](reports/summary.json), [comparison CSV](reports/comparison.csv),
[cohort record](reports/trial-cohort16-two-pass.json), [methods](reports/methods.md),
[model contract](model-contract.md), and [prepared next steps](next-steps.md).
