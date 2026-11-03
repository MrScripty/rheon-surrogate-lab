# Shape repeatability and correction-efficiency results

The previous two-correction reference remains the qualified laboratory control. The frozen one-correction10,000-update candidate improves familiar-case work but fails one untouched simulation. It is not promoted. All failures, timings and source identities are retained. No final-informed refitting or alternate-checkpoint selection occurred.

## Controlled CUDA fits

Four fresh fits complete 25,000 CUDA updates. All 12 managed stages complete/drain, reopen and download identically twice. CPU restoration preserves every weight bit, performs zero optimizer updates and initializes no CUDA context. Every model still has 370,016 float32 parameters and a 1,485,452-byte weights file.

| Model | Training corrections | Updates | Optimizer seconds | Familiar mean / max calls | Fixed 8 passes |
|---|---:|---:|---:|---:|---:|
| shape-two-reference | 2 | 5,000 | 203.23 | 9.414 / 12 | 26/128 |
| shape-two-seed2 | 2 | 5,000 | 206.82 | 8.812 / 11 | 48/128 |
| shape-two-seed3 | 2 | 5,000 | 205.91 | 9.148 / 12 | 39/128 |
| shape-one-control | 1 | 5,000 | 109.04 | 8.609 / 11 | 58/128 |
| shape-one-longer | 1 | 10,000 | 217.09 | 6.680 / 12 | 112/128 |

Fitting/familiar qualification uses adaptive CPU inference, checked after each actual network evaluation and capped at 32. Raw managed outputs remain0/12 fitting and 0/128 validation for all four models.

All four new models pass 32/32 fitting states,128/128 familiar validation states,120/120 signed analytical probes,8/8 fresh development scenes,7/7 prior-grid cases and 3/3 lower-force512-step trajectories. The analytical probes are not additional CFD simulations. Training still uses 16 real simulations/32 states and the same eight-grid/1,024-example operator bank.

The two additional seed pairs reproduce physical development qualification. Their familiar means8.813/9.148 compare with 9.414 for the original seed; this does not prove unseen-shape repeatability for every seed.

The same-seed one-correction5,000-update control preserves initial weights, real sampling order and normalization. Its training forward work is half the two-correction control. The longer one-correction fit starts from the same initialization, doubles updates and moves learning-rate reductions to 6,000/9,000. Its20,000 forward network evaluations match the two-correction control; cell work173,148,099 versus 173,120,838 differs by 0.016%. Optimizer/backward work is not identical. Rates are observations under concurrent CPU work, not isolated GPU benchmarks.

## Frozen final comparison

Selection freezes at 2026-10-16T04:44:53.675246+00:00 before either new final population executes. The candidate checkpoint is `8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f`. Both final batches now count as consumed development evidence for future work.

| Metric on 16 new simulations | Previous reference | Frozen candidate |
|---|---:|---:|
| Qualified simulations | 16/16 | 15/16 |
| Qualified AI steps | 1808/1808 | 1733/1808 |
| Mean / maximum network calls | 9.832 / 13 | 14.141 / 32 |
|32-call cap hits | 0 | 85 |
| Worst pressure-field relative L2 deviation | 5.40009e-06 | 0.000104297 |
| Median final pressure deviation | 3.93268e-07 | 5.59864e-07 |
| Hard stops | 0 | 0 |

The candidate misses75 of 113 AI steps in the 16×23×25 scene, seed2026101709, despite valid native controls and 15 warm-up steps. Worst relative equation residual is4.4277e-6 against1e-6. The other15 simulations qualify completely. All 3,616 final AI frames/callback counts independently match the retained index. No native refinement contributes to AI success.

Candidate mean work is43.8% higher on this final population; worst pressure deviation is19.3× larger. These work averages include failed/capped steps and are not matched-success latency measurements. Familiar fixed 8 median residual improves86.2×, but that gain does not generalize uniformly. Equation residual and pressure-field relative L2 are different metrics.

Post-test trace analysis shows slow monotone convergence on the failing grid: median successive residual ratio0.7574, with no residual-increasing correction steps. This is a failure to reach the target within budget, without a hard stop. Improving only familiar means is an inadequate selection rule for future experiments.

## Other trajectory tradeoffs

| Population | Reference mean calls / worst pressure deviation | Candidate mean calls / worst pressure deviation |
|---|---:|---:|
| Fresh development | 9.468 / 5.77329e-06 | 9.310 / 2.15811e-05 |
| Prior grids | 10.482 / 4.65375e-07 | 8.321 / 5.02265e-07 |
| 512-step development | 9.292 / 6.13957e-06 | 6.249 / 4.21578e-06 |

Every physical gate in these development populations passes. Fresh-development pressure drift nevertheless worsens; longer-window worst deviation improves while its median final deviation worsens. Retain these differences rather than claiming universal accuracy improvement.

Exploratory65×9×7,7×65×9,95×7×5 and 127×5×5 studies stay outside the declared selection gates. Both checkpoints pass 3/4 scenes and miss some127×5×5 steps, with all native controls valid and no hard stop. Reference/candidate qualify 183/196 and 180/196 AI steps respectively. On all 49 actual reference inputs for 127×5×5, an independently computed ideal correction rounded through float32 passes the equation/predicted-divergence target. Those oracle values never advance the simulation; all observed learned physics/error frames match the original trajectory exactly. This precision result applies to that reference track, not every unseen input.

## Time and preparation

New optimizer intervals total738.86 seconds (12m18.86s). The managed/CPU sequence takes 3613.45 seconds (60m13.45s). Capture-to-last-evaluation wall span is3889.34 seconds (64m49.34s), including final/exploratory evaluation, excluding earlier planning and subsequent review/documentation. Preparation capture takes 9.492 seconds.

| New fit | Managed train/restore/validation seconds | CPU qualification seconds |
|---|---:|---:|
| shape-two-seed2 | 794.40 | 216.01 |
| shape-two-seed3 | 883.50 | 217.23 |
| shape-one-control | 814.40 | 228.31 |
| shape-one-longer | 916.64 | 204.51 |

CPU qualification overlaps preparation/training of later fits. These durations nest and overlap; summing them is not elapsed workflow time. Per-command UTC/monotonic timings, queue/Run intervals and actual commands are retained. The first fit spends519 seconds outside its queued/executing Run intervals; preparation/approval/submission/download overhead needs separate profiling. No isolated latest-model inference speed comparison was performed.

CPU preparation also audits lossless32/64-simulation training views (64/128 states), all with distinct physical setups and shapes, from existing native captures. Neither view is used by current fits. A proposed11-grid bank is rejected by the enforced eight-grid trainer limit before optimization. The exact failed input, timing and reaping remain retained. A separate three-grid elongated auxiliary bank passes 384 label/operator checks with zero optimizer updates; combining banks is deferred to an explicit trainer-contract change.

## Retained evidence

[Decision](reports/qualification-decision.json), [comparison CSV](reports/comparison.csv), [native result index](reports/summary.json), [timings](reports/timings.json), [frozen selection](reports/final-selection.json), [flow correction traces](reports/flow-feedback.json), [methods](reports/methods.md), [model contract](model-contract.md) and [next experiments](next-steps.md). Full outputs remain in the existing Tuldok catalog and ignored runs/. Historical results are unchanged. No weights or simulation datasets are committed.
