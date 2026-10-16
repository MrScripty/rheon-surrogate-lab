# Two-correction longer control results

The prior two-correction control remains preferred. The new two-correction, 10,000-update model fixes the previous one-correction model’s qualification failure and improves familiar-case work, but regresses worst-case work and pressure error. All methods, timings and failures are retained; no final-informed refit or checkpoint reselection occurred.

## Controlled training comparison

One new CUDA fit completed 10,000 updates. Initialization and train-only normalization match the retained control. Real and analytical draw orders match the earlier one-correction, 10,000-update fit. Architecture, 370,016 float32 parameters, 19 arrays and 1,485,452-byte weights file are unchanged. All three managed stages completed/drained, reopened and redownloaded identically; fresh CPU restore initialized no CUDA context and performed no optimizer updates.

| Model | Training corrections | Updates | Training interval (s) | Familiar mean / max calls | Fixed-eight passes |
|---|---:|---:|---:|---:|---:|
| shape-two-reference | 2 | 5,000 | 203.23 | 9.414 / 12 | 26/128 |
| shape-one-longer | 1 | 10,000 | 217.09 | 6.680 / 12 | 112/128 |
| two-pass-10000 | 2 | 10,000 | 405.50 | 7.547 / 10 | 106/128 |

Training still uses 16 real CFD simulations/32 states and the same 1,024-example manufactured operator bank; the 120 analytical probes are not additional CFD simulations. All 32 fitting and 128 familiar validation states qualify under adaptive CPU inference; all 120 signed analytical probes pass. Familiar mean work improves 19.8% versus the retained control. The fixed-eight worst relative equation residual is 4.44122e-6, versus 3.22709e-5 for that control. These are network-call/error measurements, not isolated latency claims. Raw two-call outputs still fail their direct native checks; adaptive repeated correction remains necessary.

The new fit uses 40,000 training forward network evaluations, twice the 20,000 used by both historical rows. Equal updates versus the one-correction fit therefore cost more forward work; equal forward work versus the two-correction 5,000-update control is not claimed. The recorded training timer includes bank/cache preparation, update checks and checkpoints, rather than pure GPU kernel time.

## Development regressions and frozen final comparison

All 34 development trajectories qualify: eight standard fresh cases, seven prior grids, three lower-force 512-step trajectories, and the 16 consumed former final cases. The 16×23×25 scene now passes all 113 AI steps, versus 75 missed steps for the prior one-correction 10,000-update model. Its median observed successive residual ratio improves from 0.7574 to 0.5157; the retained control is stronger at 0.3049 on its own trajectory. These are trace diagnostics, not a global contraction theorem.

Development selection froze at 2026-10-16T05:44:27.235506+00:00, before the eight new final simulations. Both checkpoints were eligible; the retained control won the predeclared worst-case ranking: maximum calls 13 versus 17, and worst pressure-field relative L2 6.13957e-6 versus 2.49767e-5. Step-weighted development means were 9.581 versus 8.639. Familiar mean alone would miss those regressions.

| Metric on eight new final simulations | Retained control | New two-correction model |
|---|---:|---:|
| Qualified scenes / AI steps | 8/8; 904/904 | 8/8; 904/904 |
| Mean / maximum network calls | 10.105 / 13 | 11.308 / 18 |
| Worst pressure-field relative L2 | 4.85702e-06 | 1.42217e-05 |
| Median final pressure-field relative L2 | 3.69646e-07 | 4.3498e-07 |
| Cap hits / hard stops | 0 / 0 | 0 / 0 |

All native controls qualify. The new model uses 11.9% more calls and has 2.93× higher worst pressure-field deviation on these new final cases. It is not promoted. Pressure-field relative L2 is distinct from the unchanged 1e-6 equation-residual gate. The final population is now consumed for future work.

## Other trajectory tradeoffs

| Population | Control mean calls / worst pressure deviation | New mean calls / worst pressure deviation |
|---|---:|---:|
| Standard fresh 8 | 9.468 / 5.77329e-06 | 8.035 / 5.15368e-06 |
| Prior grids 7 | 10.482 / 4.65375e-07 | 8.571 / 8.06593e-07 |
| 512-step cases 3 | 9.292 / 6.13957e-06 | 7.738 / 4.08533e-06 |
| Consumed batch 0 | 10.082 / 1.98779e-06 | 8.719 / 3.94554e-06 |
| Consumed batch 1 | 9.583 / 5.40009e-06 | 10.482 / 2.49767e-05 |

The exploratory elongated population remains 3/4 qualified for both models. On 127×5×5, the new model misses 16/49 AI steps versus 13/49 for the control. Its observed residual ratios remain below one, with median 0.8486, so this is slow convergence within the 32-call cap rather than an observed runaway. Worst pressure deviation on the elongated population is 0.00184093 versus 0.00174176. These cases are explicitly outside selection eligibility.

## Timing, recovery and retained evidence

- Recorded training interval: 405.499s (6m 45.5s). Complete managed train/restore/validation: 1024.160s (17m 4.2s).
- Standard CPU qualification: 207.603s. Resumed development/final evaluation: 255.642s.
- Capture through final scientific evaluation: 1693.189s (28m 13.2s), including the reference-binding repair gap. Active controller durations sum to 1487.864s; overlapping job sums are not elapsed makespan.
- The initial postprocessor stopped on a mismatched historical reference checkpoint after successful training/standard qualification. The original failure, sequence bytes and wrong-reference drift are preserved. All three reference bindings were corrected by weight identity, scene and binary. Explicit development-only recovery re-sealed the same artifacts and submitted zero new managed Runs or optimizer updates. No final case had been used at the failure.
- All 100 native trajectory children exited zero and were reaped. Five diagnostic populations covering 428 states drained. No owned Python controllers remain. Local retained output is 112,853,577 bytes, below the 1 GiB budget.
- A separate mixed-odd three-grid bank passed 384 independent label checks, worst relative operator error 3.05752e-14, with legacy parity and RNG preservation. It is prepared, not trained. The prepared 32/64-real-parent cohorts have at most 16 cells per axis and do not fill this larger odd-domain gap.

Exact commands/environment and reproduction: [methods](reports/methods.md). Small retained receipts: [training comparison](reports/three-way-comparison.json), [decision](reports/decision.json), [full timing index](reports/timings.json), [lifecycle/custody index](reports/summary.json), [recovery](reports/postprocessing-recovery.json), [correction traces](reports/flow-feedback.json). The [checkpoint contract](model-contract.md) and [next steps](next-steps.md) separate current qualification from future work. Rheon deployment is unchanged; no weights or simulation arrays are committed.
