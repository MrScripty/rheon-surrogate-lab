# Smaller feedback weights: results

The two matched candidates completed 5,000 CUDA updates, fresh CPU restores and
all 74 development trajectories each. Both are rejected for promotion by the
unchanged population gates. Weight 0.03 is the useful development lead: fewer
calls and better worst trajectory pressure than the matched zero-weight control
in all three populations, but worse pressure than the prior accurate model in
two populations. Eight unused whole scenes remain untouched. Historical results,
failed attempts and the original Tuldok catalog are preserved.

The scientific change is only low-frequency feedback weight: 0.01 or 0.03 versus
zero. Actual initialization, data, normalization, runtime, sample order, Adam
schedule, two training corrections and 5,000-update budget match the retained
control. Feedback covers spatial DCT modes 0–3 per axis, excludes DC, and uses a
0.1 full-signal RMS floor at the first/final existing predictions. No additional
network forward is introduced. For both candidates, the five numerical artifacts match byte for byte across
CUDA training and fresh CPU restore.
Architecture remains 370,016 parameters and 1,485,452 bytes of weights. Training
uses Pumas-managed CUDA PyTorch, float32, no AMP or TF32; inference uses CPU-only,
one-thread PyTorch with the existing float64 physical pressure accumulation.

| Measurement | Control (retained) | Weight 0.01 | Weight 0.03 |
|---|---:|---:|---:|
| Optimizer seconds | 362.149 | 393.834 | 389.520 |
| Trainer seconds | 365.772 | 397.646 | 393.141 |
| Complete Study command seconds | 1,015.366 | 1,160.340 | 1,852.909 |
| Peak CUDA bytes | 770,565,632 | 837,691,904 | 837,691,904 |
| Complete 74-scene CPU campaign seconds | 737.499 (reused) | 752.907 | 710.910 |
| Conservative summed peak RSS bytes | 11,227,897,856 | 11,261,349,888 | 11,253,071,872 |
| Minimum available host memory bytes | 46,977,703,936 | 47,556,329,472 | 47,800,639,488 |

Complete Study clocks include preparation, queueing, CPU restore and readback;
they are not GPU execution durations. CPU campaign clocks include concurrent
startup, native controls and independent sparse oracles; their differences are
not isolated inference speedups. One bounded pool uses sixteen one-thread CPU
workers, with disjoint CUDA/restore/desktop affinity, 2 GiB accounting reservation
per worker and an 8 GiB available-memory reserve. Shared RSS pages can be counted
repeatedly. All campaign workers and native children exited and were reaped.

Each new campaign physically qualified **74/74 trajectories, 8,523 AI steps and
233 independent checks**. First-frame qualification preceded the full campaign:
both took six calls after fifteen native warm-up steps. The 0.01 / 0.03 first
frames took 8.526 / 8.605 episode seconds (9.116 / 9.126 outer command seconds).
No native fallback or refinement was admitted. Sealed full-water physics, cubic
spacing, valid previous-pressure history, 32-call ceiling and all residual,
pressure, divergence and Courant limits remain unchanged.

The control's full campaign is explicitly reused from the preceding experiment.
A fresh control frame and exact numerical/executable/runtime/contract and
all-array equivalence took 9.871 seconds. Reuse is not another 74 executions.

| Population | Control mean/max calls | 0.01 mean/max calls | 0.03 mean/max calls | Control worst trajectory pressure | 0.01 worst | 0.03 worst | Prior accurate worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| original58 | 7.7351 / 12 | 7.7577 / 11 | 7.3550 / 10 | 4.68189e-6 | 3.84876e-6 | 3.61038e-6 | 3.14479e-6 |
| late-amended8 | 8.8451 / 11 | 9.2290 / 11 | 8.2478 / 9 | 1.26300e-6 | 1.34723e-6 | 1.08574e-6 | 1.34895e-6 |
| minimum8-failed8 | 8.1681 / 11 | 9.2301 / 18 | 7.8385 / 11 | 2.38513e-6 | 1.45444e-6 | 1.68747e-6 | 1.41517e-6 |

Overall mean calls are 7.898745 / 8.069928 / 7.500997 for control / 0.01 / 0.03;
maximum calls are 12 / 18 / 11. Weight 0.03 reduces overall mean calls by 5.036%
versus control. It passes every comparison except worst-pressure against the
prior accurate model in original58 and minimum8-failed8. Weight 0.01 also
regresses call work and has an 18-call difficult step. The retained 0.1 candidate
previously averaged 7.880207 calls with maximum 12 and was also rejected. No
candidate meets every population gate against all three retained comparators;
physical qualification alone does not establish improvement. The preregistered
eligible-candidate ranking selects none, and no holdout or promotion occurs.

Same-RHS independent pressure errors are separate from trajectory drift. Median
mean-free pressure error is 2.36459e-7 / 2.03831e-7 / 1.69867e-7; median gradient
error is 1.51305e-7 / 1.52296e-7 / 1.23183e-7. Maxima do not uniformly improve.
Every pressure bound covers its independent oracle, whose maximum relative
defect remains below 1.638e-13. These evaluate each model on its own autonomous
states; the comparison is descriptive, not a matched-RHS causal ablation.

All 84 fixed-input manufactured rows per candidate are retained: three grids,
seven pure-low/mixed/high-only amplitude patterns and calls 1/4/8/12. At eight
calls each candidate improves absolute low-mode coefficient error in 14 of 21
samples and worsens seven; at four calls each improves ten and worsens eleven.
There is no uniform low-frequency improvement. Spatial modes are not temporal
Hz, and weak relative errors alone do not establish a causal noise floor.

A post hoc trace on consumed scene 73, step 27, exposes a possible next training
target. Control / prior 0.1 / 0.01 / 0.03 take 11 / 10 / 18 / 11 calls. Median
consecutive residual-L2 ratios from proposal four onward are 0.29864 / 0.29358 /
0.49273 / 0.31711. The selected 0.01 trace keeps contracting more slowly; it does
not plateau. Each model has a different autonomous input, so this is a hard-case
diagnostic, not a causal comparison. Training covers two corrections while
inference may need many more. Existing training already includes a scaled late
direction bank from an older frozen reference. The next distinct question is
current-policy late states at their actual small amplitudes; diagnose those on
training-only parents before a matched early/late supplemental comparison. See
[next proposal](next-proposal.md). Retain these gates and the untouched eight.
No subsequent fit has started.

CPU work overlapped CUDA fitting: a repeated, unpromoted full runtime-validation
prototype (reference/prototype/prototype/reference on core 23) had median
9.865 / 6.583 seconds, with identical complete asset verification. Warm-cache and
desktop/catalog load were uncontrolled; no deployment or pipeline speedup is
claimed. A 233-state frequency/amplitude audit of the already consumed prior
0.1 candidate took 2.877 seconds on sixteen one-thread workers. Weak low-mode
relative RMS error worsened while strong-mode error improved; weak absolute
error energy was far smaller. Autonomous-state correlations do not establish
noise-floor causation. Retained hard-case trace analysis advances no simulation.

Serial CPU timing ran on core22, one episode at a time after both broad pools
drained, in control/0.01/0.03/0.03/0.01/control order. Each fixed one-metre cube
uses fifteen native warm-up steps then eight AI steps; resolution16/24/32/40
refines cubic cells rather than stretching them. Full paired simulation step
time includes iterative correction, IPC and protocol trace logging. Startup is
separate, with no sparse oracle or capture in timed episodes. This is not a
pure network-forward or pressure-solver-only benchmark. Repeated physical frames
and call sequences match exactly for all three models.

| Grid | Control AI ms/step | 0.01 AI ms/step | 0.03 AI ms/step | Native ms/step (control/0.01/0.03) | AI/native ratio (control/0.01/0.03) |
|---|---:|---:|---:|---:|---:|
| 16³ | 68.862 | 69.892 | 72.723 | 13.974 / 14.044 / 14.754 | 4.93 / 4.98 / 4.93 |
| 24³ | 440.130 | 388.898 | 384.593 | 62.737 / 65.004 / 65.561 | 7.02 / 5.98 / 5.87 |
| 32³ | 618.662 | 703.356 | 591.921 | 185.660 / 188.557 / 185.509 | 3.33 / 3.73 / 3.19 |
| 40³ | 1646.684 | 1655.429 | 1526.754 | 444.494 / 446.576 / 434.345 | 3.70 / 3.71 / 3.52 |

The six complete batches took 73.482, 74.285, 71.883, 71.887, 73.741, 73.636 seconds.
Raw frame/callback, campaign/config/source hashes, startup, memory and drainage
are retained in [serial timing](reports/serial-timing.json). AI remains slower
than native across this tested range. Two repetitions under uncontrolled desktop
and cache conditions do not establish a scaling law or isolated solver speedup.

Original Tuldok Models/Studies/Runs remain in catalog
`a7da4a8de53541a2a0e2aad08005bb71` at http://127.0.0.1:8091. Final exact readback
confirms all four new CUDA/CPU Runs completed and drained. External source
observation uses the existing Project/Rheon paths and recorded pressure previews;
it is not live velocity or tracer animation.

The first Project import failed with HTTP409 because its root alias was not
configured, before source read or admission. The journal and rejection are
retained; a new explicit attempt used verified `current-lab`, without replay or
resume. The composite initial owner preflight took about 87 seconds; its phases
were not isolated, so it does not identify Run listing as the bottleneck. A
reporting path-dependence finding was fixed using required `--root`, negative
artifact-parity coverage and actual root/lab byte-exact replay. Eight focused
tests pass under the selected Pumas runtime; system Python's missing NumPy/Torch
was an unqualified discovery attempt, not a scientific failure.

Source: lab branch `experiment/pressure-feedback-weight-ablation-20261017`;
Tuldok unchanged at `92ef4393c327b9bdb06dc044267a4ce682eca1f9`, deployed code
`1688f7d9fc6f941d8a23d8b077df6af4213f683a`; Rheon unchanged at
`759d928448dec8b160f4c113e35bf2bf2bd1b9d3`. No branches were pushed or merged.
No weights, datasets, runtime packages or credentials are committed.

See [execution ledger](execution-ledger.md), [matched training](reports/matched-training.json),
[regression records](reports/regression-summary.json), [decision](reports/qualification-decision.json),
[diagnostics](reports/diagnostic-summary.json), and [recorded hard-case traces](reports/hardcase-trace-comparison.json).
