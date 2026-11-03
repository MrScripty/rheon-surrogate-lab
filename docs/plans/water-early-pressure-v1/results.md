# Earlier pressure corrections: findings

Both 5,000-update CUDA fits completed and restored on CPU. All 74 scenes qualified for each model (148 model trajectories, 17,046 AI steps), and all 466 sparse independent pressure-bound checks passed. Float32 neural arithmetic, cubic cells, physical gates and the pressure-bound threshold were unchanged.

The harder-data model reduced the recent population’s peak from 17 calls to 11 and the overall peak from 17 to 12. It improves every declared gate against the retained reference, but sacrifices some of the prior late model’s pressure accuracy. The first-pressure-loss variant also trades accuracy and work across populations. Neither satisfies the frozen no-regression rule against both baselines. No checkpoint was promoted; the eight independent scenes remain unconsumed.

## Complete consumed-population comparison

All pressure errors below compare independently advanced native and AI tracks. Sparse same-RHS exact errors are retained separately. Both baselines are learned models; the native algorithm supplies the reference trajectory for every row.

| Population | Learned model | Mean calls | Peak calls | Worst trajectory pressure relative L2 |
| --- | --- | ---: | ---: | ---: |
| original58 | Retained reference | 9.664 | 13 | 6.139568e-06 |
| original58 | Prior late model | 7.937 | 12 | 3.144795e-06 |
| original58 | Harder data, unchanged loss | 7.735 | 12 | 4.681886e-06 |
| original58 | Harder data + first pressure loss | 7.740 | 11 | 3.594327e-06 |
| late-amended8 | Retained reference | 10.461 | 18 | 3.201617e-06 |
| late-amended8 | Prior late model | 9.562 | 12 | 1.348955e-06 |
| late-amended8 | Harder data, unchanged loss | 8.845 | 11 | 1.262996e-06 |
| late-amended8 | Harder data + first pressure loss | 9.293 | 14 | 1.349483e-06 |
| minimum8-failed8 | Retained reference | 9.126 | 11 | 3.373359e-06 |
| minimum8-failed8 | Prior late model | 9.080 | 17 | 1.415167e-06 |
| minimum8-failed8 | Harder data, unchanged loss | 8.168 | 11 | 2.385126e-06 |
| minimum8-failed8 | Harder data + first pressure loss | 8.619 | 13 | 1.67379e-06 |

The original58 population rejects both candidates on worst pressure versus the prior late model. The harder-data model passes the amended8 population, but regresses prior-late pressure on the recent8. The first-pressure variant also regresses some peak-call gates. Full per-frame and per-case records preserve these tradeoffs.

On 15 identical consumed saved inputs, the first-pressure variant’s8-call worst mean-free pressure error is47.4% lower than its matched harder-data control and 32.4% lower than the prior late model. That benefit is not uniform: the prior late model is better at 4 and 12 calls. These fixed-input results do not substitute for autonomous trajectories.

## Timing and resources

| Fit | Optimizer seconds | Complete managed train + CPU restore seconds | Peak CUDA allocated bytes | Weight file bytes |
| --- | ---: | ---: | ---: | ---: |
| Harder data, unchanged loss | 416.986 | 889.303 | 770,565,632 | 1,485,452 |
| Harder data + first pressure loss | 453.771 | 969.403 | 770,565,632 | 1,485,452 |

The full CPU regression took 37.85min; serial timing took 5.13min. The measured interval from qualified preparation through final timing was85.93min, including preparation, managed setup and analysis gaps. The initial failed homogeneity attempt lacks a full elapsed receipt; subsequent documentation is excluded. These scopes are distinct.

Current serial uninstrumented timing compares the frozen prior late model and harder-data model, two repetitions in reversed order, one pinned CPU core/thread, first AI timing step excluded. Ordinary desktop apps remain. Positive reduction means faster.

| Counts XYZ | Prior late median ms | Harder-data median ms | Observed reduction |
| --- | ---: | ---: | ---: |
| 17×13×9 | 37.142 | 34.305 | +7.64% |
| 24×16×12 | 67.816 | 68.295 | -0.71% |
| 40×40×40 | 1458.118 | 1436.118 | +1.51% |

The long rectangular case is effectively unchanged/slightly slower; the 40³ observation has only14 steady samples per model and its small difference is within repetition variation. These three workloads do not establish a universal latency gain or an AI/native-solver crossover. Archived and concurrent timings are not substituted for current measurements.

All 237 native children and 261 recorded measurement command children were reaped. All 6 managed Runs completed and drained in the existing catalog. The maximum completed worker high-water mark was866,668,544 bytes (individual process, not simultaneous system memory). Raw retained output was498,760,027 bytes at seal. The native solver, neural architecture, refinement/fallback behavior and deployment defaults were unchanged.

The next refinement must recover the best prior trajectory pressure accuracy while retaining the harder-data model’s reduced peak work. No further fit, stopping-profile change or independent validation was started.

## Amplitude and frequency

The answer differs for isolated modes and mixtures. On three manufactured grids,
globally reducing a pure low spatial mode from 1Pa to 1e-8Pa preserved approximately
the same relative error. In a mixture, a stronger high mode created low-mode
error that overwhelmed a weak low signal. That is consistent with an effective
error floor caused by learned cross-band leakage; it does not establish that
float32 cannot represent weak modes.

For the frozen late model after 8 single evaluations on 9×11×33:

| Input | Low-mode coefficient error | Relative to the supplied low coefficient |
| --- | ---: | ---: |
| Pure low,1Pa | -0.001838Pa | 0.184% |
| Pure low,0.001Pa | -1.838e-6Pa | 0.184% |
| Pure low,1e-8Pa | -1.838e-11Pa | 0.184% |
| High only,1Pa | -0.018253Pa | Low input absent |
| Low0.001Pa + high1Pa | -0.018256Pa | 1826% |

The float32-rounded exact high-only pressure's low coefficient error was about
1.23e-10Pa, far below the learned leakage on this case. Physical pressure is
accumulated in float64 even though network computation is float32, so that
comparator is not a universal inference precision floor. Leakage declines with
additional learned corrections: high-only low error is3.066Pa after 1 call,
-0.3052Pa after 4, -0.01825Pa after 8 and-0.001096Pa after 12. This is a stationary
manufactured test, not an independently advanced fluid trajectory.

The temporal test replays16 consumed recorded teacher trajectories,49 states
apiece. Both signal and error use cell-wise demeaned periodic Hann32 windows,
hop8, identical gauge removal and PSD unitsPa²/Hz. At the lowest positive bin,
1.875Hz, late-model8-call median relative RMS error across weak/middle/strong
signal tertiles is3.10e-7 /4.99e-7 /2.71e-7. Those tertiles do not show monotonic
weak-signal degradation. Pooled windows/scenes are correlated; associations do
not identify a causal noise floor. These short windows cannot resolve slower
cycles. Temporal frequency and spatial cosine mode index are distinct.

A separate input-only quantization test (no neural computation) induces about
5.16e-8Pa of low error for the high-only long-grid field. That is much smaller
than the model leakage above. But a mixed1e-8Pa low signal is indeed overwhelmed
by this input-rounding contribution. Weak modes can be obscured by limited
dynamic range; the measured learning gap remains much larger on this case.
This single-input test is not a floor for iterative correction, which recomputes
physical residuals between calls. [Input rounding](reports/input-rounding.json).

No fitting examples or loss weights were selected from these observations.
The already admitted first-correction ablation remains unchanged.

[Methods](execution-ledger.md) · [Raw scalar probes](reports/amplitude-late.json) ·
[Temporal summaries](reports/amplitude-baseline-summary.json) · [Current status](reports/current.json).
