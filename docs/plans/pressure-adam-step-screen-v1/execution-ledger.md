# Completed CPU Adam step screen

No new CUDA training or model promotion. C53 remains deployed. The pressure
network was evaluated, not optimized. Ten disposable flat-parameter Adam steps
constructed hypothetical endpoints using captured CUDA gradients and moments;
no hypothetical weights were published or committed.

The pool completed in **15.357 seconds**, with three one-thread workers on CPUs
21/22/23 queueing five captured training updates. It evaluated **57 distinct
parameter vectors** across all eight fitting cases. There are 75 policy/fraction
entries (15 zero-step baseline aliases, 60 nonzero proposal aliases); duplicate
vectors share measured results. Of the 57 distinct vectors, five are baselines
and 52 are distinct nonzero proposals. Every state has the unchanged full loss
and 1/2/4/8-pass pressure measurements. No rollout, real-scene regression campaign
or untouched holdout was consumed.

| Captured training update | Worker seconds | Distinct states | Worker self-reported peak MiB | CPU |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.802 | 9 | 739.1 | 21 |
| 10 | 6.368 | 13 | 739.1 | 22 |
| 100 | 6.359 | 13 | 741.0 | 23 |
| 150 | 5.617 | 13 | 746.1 | 21 |
| 200 | 4.197 | 9 | 746.0 | 23 |

Worker times overlap; their sum is not a serial benchmark or measured speedup.
The controller also sampled /proc RSS (maximum 794.0 MiB), separately from each
worker's ru_maxrss measurement. Host available memory stayed above 47 GiB,
well above the predeclared 12 GiB reserve. Each snapshot had a monitored 2 GiB
RSS budget; vector workspace bound was 29,601,280 bytes (including transient conversion buffers). Qualification startup RSS is also monitored in the final controller. All six children
(one literal qualification plus five snapshot workers) exited 0 and were reaped.

**No nonzero proposal passed every predeclared guard.** Those guards are local
before-state retention checks: every case no worse at one/two passes, no formerly
qualifying four/eight-pass case lost, weaker case loss and four-pass residual
improved beyond its empirical CPU/GPU value gap, and lower worst four-pass error.
They are not C53 deployment qualification or a mathematical generalization test.
The 1e-6 physical target was never relaxed by the value-calibration envelopes.

| Training update | Best weak-case nonzero proposal | Weak four-pass residual before | After | Weak residual gain | Short-pass regressions | Previously fitted results lost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Preserved/reset, half step | 1.5224311e-4 | 1.2152181e-4 | 20.18% | 3 | 0 |
| 10 | Reset, half step | 1.8639813e-3 | 1.8543673e-3 | 0.52% | 4 | 0 |
| 100 | Reset, full step | 2.4695245e-3 | 1.9896495e-3 | 19.43% | 9 | 2 |
| 150 | Reset, full step | 2.0614057e-3 | 1.9847068e-3 | 3.72% | 6 | 0 |
| 200 | Reset, eighth step | 2.0261667e-3 | 2.0260280e-3 | 0.0068% | 9 | 0 |

This table selects the lowest weak four-pass residual among nonzero tested
entries, not a safe candidate. Update 200's tiny gain does **not** exceed the
empirical calibration envelope. At update 100 the full reset loses four-pass
qualification on cases 1 and 4. Early half steps can improve four-pass output
while worsening one/two-pass output, including the weak case's earlier passes.
Do not equate a single-case gain with an improved trained model.

For updates 10/100/150, all tested smaller steps along the observed/preserved
endpoint still worsened weak four-pass residual. Thus shrinking that direction
alone did not fix those snapshots. Resetting moments changes both direction and
step magnitude; these observations do not isolate first versus second moments
or establish that a reset policy would train better over time.

All five preserved-state endpoints reproduce observed CUDA parameters within the
predeclared 2e-7 maximum absolute tolerance (worst 7.45e-9). At the first update,
preserved and reset endpoints are bitwise identical. Literal vector interpolation,
independent scalar Adam recurrence, missing-history and undeclared-fraction
negative checks passed before admitting measurements. No CPU gradients were
computed: every endpoint uses the exact saved clipped CUDA gradient.

CPU/GPU loss and pressure value gaps were measured on the saved pre-state and
actual observed post-state, not inferred from the earlier derivative mismatch.
Worst loss gap across all cases/snapshots was 0.008828. Per-case/pass gaps and
the twice-gap empirical cautions are retained in results.json; they are not
rigorous rounding bounds for hypothetical states. The largest weak four-pass
pre/post gap was below 3.90e-7; tiny late gains need caution. Strict guard failures
remain failures even when their magnitude is below these gaps.

The final cancellation fixtures signaled the real queue controller both during
qualifier startup (SIGTERM, 0.154 seconds) and after exactly three snapshot
workers started (SIGINT, 1.974 seconds). Both recorded cancelled, stopped/reaped
all children and reaped the controller. No numerical worker failure, memory
breach, wall timeout or CUDA initialization occurred in the completed scientific
screen. Fixture timing is separate from the scientific screen.

The first report preview put the 1e-6 target on the absolute logarithmic axis,
which visually compressed 20% local changes. The final viewer defaults to change
from the pre-state (%), with an optional absolute logarithmic axis and unchanged
numeric tooltips/target. All numerical outcomes were checked identical across
this presentation change. Preview source/report and browser artifacts remain
under ignored runs/. The final browser fixture verified five updates × eight
cases × both measurements, percentage/absolute controls, strict guard displays,
keyboard labels, narrow layout and no console errors. Its own browser was reaped
and its screenshot visually inspected.

The initial 15.281-second screen used a nominal vector-workspace estimate that
omitted some transient float64 conversion copies. Before final acceptance, the
bound was raised conservatively to 29,601,280 bytes within the unchanged 64 MiB
configured vector budget. A fresh complete screen (16.463 seconds) produced
identical candidate hashes, metrics, calibration gaps and guard outcomes.
The first screen's raw reports/source remain under workspace-review-01; no
historical scientific result was overwritten or discarded. This rerun corrects
resource accounting, not model policy or observed accuracy.

A final controller review replaced throwing signal handlers with a cancellation
flag checked at controlled boundaries after child ownership registration. This
closes the launch/registration interruption window, makes qualification startup
cancellable, handles SIGINT explicitly and avoids rewriting launch markers on
every polling iteration. The complete final screen took 15.357 seconds, with
all parameter hashes, measured values, calibration gaps and guards unchanged.
The intermediate 16.463-second screen/source remain under lifecycle-review-02.
These repeated runs verify deterministic computation and controller changes;
they are not independent training seeds or additional generalization evidence.
