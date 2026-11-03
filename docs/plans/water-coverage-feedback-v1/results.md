# Coverage and intermediate-feedback results

Both new 5,000-update CUDA fits completed and passed all 42 required development
simulations. Neither replaces the retained control: coverage needs more network
calls, while feedback's familiar gains do not hold on the worst shapes. The
frozen final challenger was coverage. It passed all eight fresh final simulations
and had lower worst pressure error, but needed 68% more calls than the control.
Rheon's deployed model/adapter is unchanged.

## Familiar qualification

| Validation, 128 states | Retained control | Coverage-only | Feedback-only |
| --- | ---: | ---: | ---: |
| Adaptive qualified | 128 | 128 | 128 |
| Mean network calls | 9.4141 | 13.3984 | 6.5859 |
| Maximum network calls | 12 | 19 | 9 |
| Qualified with exactly eight calls | 26 | 0 | 126 |
| Fixed-eight median relative residual | 2.5862e-6 | 2.2062e-5 | 4.9738e-8 |
| Fixed-eight worst relative residual | 3.2271e-5 | 3.9508e-4 | 2.7128e-6 |

Feedback reduces familiar mean calls by 30.0%; its fixed-eight worst residual is
11.9 times smaller than control. Two states still fail at exactly eight calls.
All 32 fitting states qualify adaptively for each model. Adaptive success and
fixed-budget performance are separate results. Float32 neural precision, physical
gates and the 32-call cap remain fixed; native fallback never counts as AI success.

## Whole development simulations and frozen selection

| All 42 development trajectories | Control | Coverage | Feedback |
| --- | ---: | ---: | ---: |
| Qualified simulations | 42 | 42 | 42 |
| Step-weighted mean calls | 9.6776 | 15.2085 | 9.6199 |
| Maximum calls | 13 | 21 | 28 |
| Worst pressure relative L2 versus native | 6.1396e-6 | 1.0521e-5 | 5.8253e-5 |

Selection froze before final execution. The predeclared ranking prioritizes
maximum calls, worst pressure error, then mean calls. It retains control as the
overall choice and selects coverage as the best new challenger. Feedback averages
slightly fewer calls across development but its worst pressure difference is
9.49 times control's. On 16×23×25 it needs up to 28 calls, versus coverage's 19.
A 10×7×25 case needs 22 feedback calls. Familiar mean work alone hides these gaps.

The four elongated cases remain exploratory, outside eligibility because the
control already fails 127×5×5. Coverage fails 49/49 AI steps on 7×65×9 and 46/49
on 127×5×5; feedback fails 21/49 on 127×5×5, versus control's 13/49 there. All
other elongated cases qualify for feedback/control. These failures are retained,
not repaired with a native pressure solve or omitted from the record.

## Eight untouched final simulations

Only control and the frozen coverage challenger were evaluated. Feedback has
no result on this fresh final population. Both evaluated models pass 8/8 whole
simulations and 904/904 AI steps after 15 native warm-up steps per simulation.

| Final metric | Control | Coverage challenger |
| --- | ---: | ---: |
| Mean network calls | 10.1980 | 17.1250 |
| Maximum calls | 12 | 21 |
| Worst pressure relative L2 versus native | 1.9271e-6 | 5.6561e-7 |
| Median final-frame pressure relative L2 | 1.7468e-7 | 1.4877e-7 |

Coverage's worst pressure difference is 3.41 times smaller, while mean work is
67.9% higher. It was not the development winner and fails the no-work-regression
promotion rule. Keep checkpoint
17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8
as the laboratory control. No checkpoint reselection or refit used final data;
these eight simulations are now consumed development data for future work.

## Method, time and limits

| New run | Updates | Training interval | Full managed workflow | Standard CPU qualification |
| --- | ---: | ---: | ---: | ---: |
| Coverage | 5,000 | 209.590 s | 850.058 s | 276.261 s |
| Feedback | 5,000 | 224.214 s | 961.889 s | 214.208 s |

Combined training intervals total 433.804 s. The scientific controller takes
2720.512 s (45 min 20.5 s); source capture through final evaluation takes
2748.715 s (45 min 48.7 s). Final simulation execution takes 195.778 s. Independent
CPU late-residual preparation takes 8.401 s. CPU work overlaps managed execution;
these intervals must not be summed as makespan. Managed time includes preparation,
submission, queueing, restoration, verification and artifact transfer. Training
interval boundaries are retained; these are not pure GPU kernel timings or an
isolated inference speed benchmark.

Both fits retain 16 real parents/32 states, seeds, train-only scales, architecture
and two shared training corrections. Coverage's balanced 11-grid stream reduces
original-grid exposure from 625 to 454/455 draws and changes forward cell work
from 173,120,838 to 154,039,978. This compares bank/sampling at equal updates,
not equal old-grid exposure or cell work. Feedback retains 173,120,838 forward
cell evaluations and adds intermediate operator/reduction work; its training
interval is 10.3% longer than control's historical 203.232 s.

Across 18 standard trajectories, median per-scene successive residual ratios are
0.3038 (control), 0.5866 (coverage) and 0.2657 (feedback), with no residual-growth
steps there. These are observed diagnostics, not a global contraction bound.
CPU preparation of eight corrections on training states also found a captured
teacher defect reaching 1.6597% of a late residual; future late labels need their
own qualification. No new fit used those prepared states.

The final audit verifies six unique completed/drained managed Runs, identical
CPU-restored weights, 208 native trajectory children exited/reaped, 10 drained
native diagnostic populations/856 cases, matched 1,808 final callbacks/frames,
selection before final execution and no surviving owned controllers. Local
retention is 223,521,880 bytes, below 2 GiB. The existing Tuldok catalog/project
remains registered; historical results and unrelated user files remain intact.

[Methods](reports/methods.md), [timings](reports/timings.json),
[decision](reports/decision.json), [sealed audit](reports/summary.json),
[execution ledger](execution-ledger.md), [model contract](model-contract.md) and
[prepared follow-ups](next-steps.md) provide the detailed record. Training source
capture: 6b9ef674b25a738f87bb24e4f04b4d9f85e8c549. Both new checkpoint hashes and
managed Run IDs are in reports/trial-*.json. Weights/raw simulation arrays remain
local and are not committed.
