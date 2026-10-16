# Targeted feedback results

Targeted supplemental examples fix the two original slow-domain regressions and
reduce mean network work, while preserving familiar efficiency. The checkpoint
passes all 50 development and eight fresh final simulations. Its worst pressure
field difference remains higher than the retained learned control, so it is not
promoted. The independently seeded repeat stays prepared and unexecuted under
the prospective full-improvement rule. Application deployment is unchanged.

## Controlled comparison

Both new fits use intermediate feedback, 5,000 updates, 16 real parents/32 states,
original seeds/scales and two training corrections. They preserve all eight
primary analytical grids and their exact 625 exposures each. Each update adds
one weight-0.1 example: reused primary tensors for supplemental control, or six
cyclic grids around 16×23×25 and 10×7×25 for targeted. This matches extra example
count, not cell work. There is no neural architecture/precision/policy change.

| Familiar validation, 128 states | Retained learned control | Supplemental control | Targeted |
| --- | ---: | ---: | ---: |
| Adaptive qualified | 128 | 128 | 128 |
| Mean calls | 9.4141 | 6.3906 | 6.4453 |
| Maximum calls | 12 | 10 | 10 |
| Qualified with exactly eight calls | 26 | 121 | 119 |
| Forward cell evaluations during training | 173,120,838 | 334,694,588 | 227,855,938 |

The targeted fit retains a 31.5% familiar mean-call reduction against control.
Exactly eight calls still miss nine states. Adaptive qualification and fixed-call
accuracy are different results; no native fallback counts as learned success.

| Required whole development trajectories | Control | Supplemental | Targeted |
| --- | ---: | ---: | ---: |
| Qualified simulations | 50/50 | 50/50 | 50/50 |
| Step-weighted mean calls | 9.7586 | 9.6082 | 8.0234 |
| Maximum calls | 13 | 25 | 14 |
| Worst pressure relative L2 versus native | 6.1396e-6 | 8.3317e-5 | 2.5893e-5 |

Targeted uses 17.8% fewer development calls on average, but its maximum is one
higher and worst pressure difference is 4.22 times control's. The predeclared
maximum-call/pressure/mean ranking retains control and freezes targeted as the
best new challenger. Familiar efficiency alone would hide the remaining gap.

## Original regressions and diagnosis

One actual solve after 15 native warm-up steps reproduces old feedback's 28 calls
versus control's 11 on identical 16×23×25 input; two replays agree. The second
10×7×25 input needs old feedback 20 versus control 10. These take about two seconds.
The targeted checkpoint turns those diagnostic bounds green at 8 and 6 calls.
Full 128-step autonomous trajectories then pass all 113 AI steps, with maximum
8 and 7 calls respectively, versus old feedback's 28 and 22 maxima.

The bounded 66-probe diagnosis takes 16.330 s. Axis permutations dramatically
change old feedback's fixed-call residual, and an independent Z-mode probe fails
its 32-call bound. Zeroing the pressure base or changing amplitude barely changes
the curves. This supports shape/axis coverage and rejects reconstruction/precision
as the cause of that large slowdown. It does not uniquely prove a pooling defect
or a global convergence bound. No temporary production instrumentation remains.

All models pass three of four exploratory elongated cases. On 127×5×5, targeted
fails 17/49 AI steps, supplemental control 10/49, and retained control 13/49. These
failures remain outside selection eligibility and are explicitly retained. No
arbitrary-shape accuracy guarantee follows from the admitted input dimensions.

## Eight fresh final simulations

Selection freezes before execution; only control and targeted are evaluated.
Both pass 8/8 whole simulations and 904/904 AI steps after warm-up, with no cap
hits or hard stops.

| Final metric | Retained learned control | Targeted |
| --- | ---: | ---: |
| Mean network calls | 9.0586 | 7.2920 |
| Maximum calls | 11 | 10 |
| Worst pressure relative L2 versus native | 5.0565e-6 | 1.7215e-5 |
| Median final-frame pressure relative L2 | 1.8606e-7 | 2.6108e-7 |

The targeted model uses 19.5% fewer calls, with 3.40 times higher worst pressure
difference. Physical qualification succeeds; the stricter no-regression promotion
policy does not. Keep learned control checkpoint
17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8.
The new diagnostic checkpoint is
5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe.
No final-informed refit/reselection occurs. These eight final cases are consumed;
the separate prepared confirmation population remains untouched. No extra seed
is trained to search for a favorable result after the complete criterion fails.

The pressure gap is not evidence of runaway growth on required cases. On
31×11×7, relative difference is 1.1411e-5 at step 16, peaks at 2.5893e-5 at 88,
and ends at 1.0126e-6 at 96. On fresh 31×15×23 it peaks at 1.7215e-5 at 107 and
ends at 2.4523e-6 at 128. The 512-step 24×16×12 witness peaks at 1.1891e-5 at
282 and ends at 4.1817e-6. All those frames qualify physically. Error fluctuates;
these observations do not establish a linear or exponential accumulation law.
Source-bound first/peak/last witnesses are in reports/pressure-witnesses.json.

## Independent CPU preparation

Thirty-two late correction labels qualify against independently enumerated
neighbor actions: worst relative operator error 1.1062e-13. Removed incompatible
constant norm is at most 8.8883e-12 of the late residual. Float32 target-rounding
error relative to the original RHS is at most 4.6433e-12. The old captured teacher
has finite late-scale defect, retained separately. These labels are prepared for
future work and do not enter either current fit.

A one-solve native fixture qualifies first. Sixteen larger varied water parents
then complete 64 steps each (1,024 total), producing 784 post-warm-up pressure
states. All native paired physics checks pass, with zero neural calls. Labels
come from an independently checked CPU spectral teacher on actual native water
predictor/RHS fields; they are not exported native iterative pressure or AI
solutions. Water, cubic h and sealed free-slip boundaries remain fixed; force
locations, strengths, radii, swirl/fluctuations and domain proportions vary.

Each parent retains pressure/base/residual arrays, force configuration and native
telemetry. No velocity/tracer export is claimed. All 16 pressure/base timelines
are validated, imported as draft records and reopened with exact hashes in the
same Tuldok catalog. Each contains 49 evolving frames at a fixed domain shape.
Future packet/view integration and training on these parents remain separate.

## Time, evidence and lifecycle

| New fit | CUDA updates | Training interval | Full managed workflow | Standard CPU qualification |
| --- | ---: | ---: | ---: | ---: |
| Supplemental control | 5,000 | 373.062 s | 1057.517 s | 216.430 s |
| Targeted | 5,000 | 336.582 s | 1123.524 s | 219.472 s |

Combined training intervals total 709.644 s (11 min 49.6 s). The scientific
controller takes 3157.813 s (52 min 37.8 s); capture through final evaluation takes
3238.154 s (53 min 58.2 s). Final simulation execution takes 144.105 s. Qualified
CPU preparation takes 104.540 s internally (104.651 s delegated command), export
1.214 s and successful publication/reopening 45.228 s. Work overlaps; sums are
not makespan, pure GPU kernel time or isolated solver latency. Fewer calls do
not establish a measured speedup over Rheon's native algorithm.

The first CPU-preparation import failure lasts 4.219 s and produces no labels or
native child. A fresh attempt fixes its dependency path and succeeds. Publication
in the wrong interpreter fails before authentication/mutation in 0.064 s; the
existing controller environment then succeeds. Both failures, source/logs and
exact recovered timing receipts are retained. No training retries occur.

The audit passes: six unique managed Runs complete/drain and CPU weights restore
identically, all 240 study trajectory children exit/reap, 17 CPU-teacher children
exit/reap, 10 diagnostic populations/856 cases drain, and all 1,808 final native
frames match callbacks. Frozen selection/configuration precede final execution.
No owned controllers remain alive. Local retention is 456,248,576 bytes below
2 GiB. The task capture worktree is removed with its source head protected.
Historical results and unrelated cohort-progress files remain intact.

[Methods](reports/methods.md), [timings](reports/timings.json),
[decision](reports/decision.json), [audit](reports/summary.json),
[execution ledger](execution-ledger.md), [model contract](model-contract.md),
[future collection](reports/future-collection.json) and [next steps](next-steps.md)
provide the complete record. Training source capture is a1c5d6f; full identity,
managed Run IDs, checkpoint hashes and environments are in the sealed receipts.
No weights/raw simulation arrays are committed. Definitive promotion would
require independent-seed confirmation; its recorded trigger was not met.
