# Concurrent CPU execution and low-frequency feedback results

Completed the authorized sequence. Concurrent execution and measurement are accepted. The low-frequency challenger is retained as a rejected candidate: no promotion, and the eight unused whole-scene holdouts remain unused. Historical results and failed attempts are preserved.

Tuldok owns the actual Models, Studies, Trials and Runs in the original catalog `a7da4a8de53541a2a0e2aad08005bb71` at http://127.0.0.1:8091. Standalone CPU qualification is labelled external observation in the existing Project/Rheon preview. The final recorded pressure preview remains available; its publisher has stopped. It is not live velocity or tracer flow.

The scheduler permits independent CPU work alongside one CUDA fit, retaining process custody through publication and drainage, reserving memory and assigning disjoint affinity. Actual managed CPU/CUDA compute overlapped for 1.255602 seconds in lifecycle qualification. Ordinary CPU execution retains its 600-second bound; explicitly requested CUDA training may use 1,800 seconds. Tests cover cancellation, startup/GO races, unknown-state fences, lost database, shutdown and executor drainage. Scheduling changes are separate from model changes.

On the repeated, identical short workload of 16 consumed scenes, median complete times were 112.252 seconds with two workers, 67.379 with eight, and 54.373 with sixteen. Sixteen workers were **2.0645× faster than two**, with identical physical outputs and call sequences. The selected configuration uses sixteen independent one-thread workers, a 2 GiB accounting reservation each, an 8 GiB available-memory reserve and explicit desktop/CUDA/restore exclusions. Reservations are admission accounting, not an OS memory sandbox. The original eight-worker identity-comparison failure and corrected physical reanalysis remain retained.

Both matched models received 5,000 CUDA optimizer updates with identical actual initialization, inputs, normalization, runtime, sample order, Adam schedule and two training corrections. Only low-frequency feedback weight changed: zero versus 0.1. Feedback covers spatial DCT modes 0–3 per axis, excluding DC, at the first/final existing predictions, with a 0.1 full-signal RMS floor. It adds no network forward. Both fresh CPU restores exactly match their CUDA artifacts. Architecture remains 370,016 parameters and 1,485,452 bytes of weights.

| Training measurement | Control | Challenger |
|---|---:|---:|
| Optimizer seconds | 362.149 | 397.050 |
| Trainer seconds | 365.772 | 400.858 |
| Peak CUDA bytes | 770,565,632 | 837,691,904 |

Complete Study command times were 1,015.366 and 1,161.684 seconds; challenger time includes queueing behind control. These are observed phase times. The initial 600-second control timeout is retained with verified termination and drainage; the successful fresh fit started from initialization using the reviewed 1,800-second CUDA bound.

Each development campaign physically qualified **74 trajectories, 8,523 AI steps and 233 independent pressure checks**. All workers and native children drained. Control took 737.499 seconds and challenger 723.304 seconds at sixteen workers. This uncontrolled wall difference is not isolated inference speedup. Peak conservative summed RSS was 11,227,897,856 / 11,783,892,992 bytes; minimum available host memory was 46,977,703,936 / 47,515,836,416 bytes. Shared pages can be counted repeatedly.

The fresh control reproduced the retained `26b` weights exactly. Exact numerical/executable/runtime files and physical/stopping contracts, plus every fresh first-frame array, support explicitly labelled reuse of this turn’s complete control campaign. It is not another 74 executions. Challenger results are freshly executed.

| Population | Control mean/max calls | Challenger mean/max calls | Control worst trajectory pressure error | Challenger worst error | Prior accurate worst error |
|---|---:|---:|---:|---:|---:|
| original58 | 7.7351 / 12 | 7.7245 / 10 | 4.68189e-6 | 4.48189e-6 | 3.14479e-6 |
| late-amended8 | 8.8451 / 11 | 8.8197 / 10 | 1.26300e-6 | 1.53655e-6 | 1.34895e-6 |
| minimum8-failed8 | 8.1681 / 11 | 8.0973 / 12 | 2.38513e-6 | 1.57326e-6 | 1.41517e-6 |

Overall mean calls fell from 7.898745 to 7.880207; the maximum remains 12. Eligibility requires every unchanged gate in each population against all three retained comparators. Rejection follows worse pressure than the prior accurate model in all three populations, worse late-amended pressure than control, and 12 versus 11 maximum calls in the difficult minimum-eight population. Physical qualification alone does not imply model improvement.

Median same-RHS mean-free pressure error improved from 2.36459e-7 to 1.48805e-7, and median gradient error from 1.51305e-7 to 1.16380e-7; maxima regressed. These describe each model on its own autonomous states, not matched-RHS causal effects. Every pressure bound covered its independent oracle; maximum oracle defect remained below 1.646e-13.

Fixed-input manufactured tests separate amplitude response from cross-band leakage. At eight calls on 33×11×9, pure-low 1 Pa coefficient error improved 55.75× and high-only leakage 1.98×. Across all 21 samples, low-mode absolute error improved in 18 and worsened in three at eight calls; at four calls, ten improved and eleven worsened. All 84 rows remain retained. Weak modes have larger relative errors but far less absolute error energy than strong modes in the descriptive 233-state audit. These observations do not establish a temporal sampling or causal noise-floor explanation.

Serial timing used one core (22), one episode at a time after broad qualification drained, and two repetitions per model in control/challenger/challenger/control order. Each consumed scene has fifteen native warm-up steps then eight AI steps. The grids refine a fixed one-metre cube; voxels remain cubic. Measurements cover full paired simulation steps through the iterative adapter, including IPC and protocol trace logging. Startup is separate; no sparse oracle or capture runs in the timed episodes. This is not a pure network-forward or pressure-solver-only benchmark. Repeated physical frames and call sequences were exact.

| Grid | Control AI ms/step | Challenger AI ms/step | Native ms/step (control/challenger) | AI/native ratio (control/challenger) |
|---|---:|---:|---:|---:|
| 16³ | 68.248 | 70.280 | 13.948 / 14.162 | 4.89 / 4.96 |
| 24³ | 444.367 | 412.015 | 62.975 / 63.882 | 7.06 / 6.45 |
| 32³ | 619.130 | 621.436 | 188.507 / 183.590 | 3.28 / 3.38 |
| 40³ | 1622.334 | 1664.388 | 433.182 / 435.854 | 3.75 / 3.82 |

AI remains slower over this tested range; the ratio does not improve monotonically with resolution. Four complete timing batches took 73.038, 73.594, 72.585 and 73.632 seconds. Raw trace/callback hashes, campaign/config/source hashes, startup times, call counts, memory and child drainage are retained in the timing report. Desktop/cache conditions were not fully controlled; this is a small repeated measurement, not a scaling law.

A separate, unpromoted runtime-validation prototype was prepared on CPU during CUDA fitting. It preserves 26,818 path-containment pairs and identical full manifests. One uncontrolled reference/prototype pair took 9.875 / 6.747 seconds. Neither scientific fit used it. An isolated benchmark and lifecycle review must precede deployment.

Next, test smaller feedback weights **0.01 and 0.03**, keeping the same frozen data, seeds and budgets. Qualify each first frame, then all 74 development trajectories and unchanged population gates. If the tradeoff persists, introduce difficult late-correction states as a separate training-distribution experiment; mixed-amplitude low/high fields are a subsequent controlled ablation. Keep the unused eight untouched until eligibility passes. No further fits were started.

Source branches: lab `experiment/pressure-throughput-feedback-20261017`; Tuldok `performance/concurrent-cpu-cuda-20261017` at `92ef4393c327b9bdb06dc044267a4ce682eca1f9`, with tested deployed code `1688f7d9fc6f941d8a23d8b077df6af4213f683a`. Rheon is unchanged at `759d928448dec8b160f4c113e35bf2bf2bd1b9d3`. Existing PR89 is unchanged; these performance branches were not pushed or merged. Canonical numerical source was committed before qualification and its hashes checked throughout.

Commands, environment, lifecycle tests, clocks, memory, failures, equivalence and case records are in [execution-ledger.md](execution-ledger.md) and [reports](reports/). See the [qualification decision](reports/qualification-decision.json), [regression summary](reports/regression-summary.json), [serial timings](reports/serial-timing.json), [matched training](reports/matched-training.json) and [independent reviews](reports/final-review.json). Training uses Pumas-managed CUDA PyTorch; inference uses CPU-only, one-thread PyTorch. No runtime packages, weights, simulation arrays or credentials were committed. The unavailable legacy TinyCNN environment was excluded, not counted as passing.
