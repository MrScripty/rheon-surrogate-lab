# Smaller current-feedback weights: completed results

Neither new candidate met all unchanged population accuracy and inference-work
gates. Both completed their matched CUDA training, exact CPU checkpoint restores,
fixtures and all 74 physical regression trajectories. No model was promoted, and
the eight unused whole-scene holdouts remain untouched.

Only `current_feedback.weight` changed, to 0.01 or 0.03. The original
`band_feedback.weight` stayed 0.03. Each arm used the same frozen native late
bank02, initialization, stream sampling and 5,000 optimizer updates: 50,000 actual
network forwards and 425,703,026 forward cells. Numerical source identities and
five CUDA-to-CPU artifact pairs per arm match exactly. The model remains 370,016
parameters and its weights file is 1,485,452 bytes. Network training and CPU
inference use float32; existing physical arithmetic remains float64.

| Measurement | Retained current 0.1 | New current 0.01 | New current 0.03 |
|---|---:|---:|---:|
| Optimizer time, seconds | 631.954 | 512.601 | 543.642 |
| Trainer time, seconds | 643.475 | 516.627 | 547.501 |
| Full CPU campaign, seconds | 694.736 | 735.341 | 789.264 |
| Mean network calls per AI step | 7.371817 | 8.082013 | 7.720404 |
| Maximum network calls | 15 | 15 | 16 |

Retained campaign values are historical context, reused after fresh exact fixture
equivalence checks. They are not new measurements under identical machine load.
The new full managed Studies took 2,691.722 seconds (0.01, including queued CUDA
time) and 1,922.802 seconds (0.03). CPU restore trainers took 2.952 and 2.896
seconds. The interval from the first managed Study intent through final timing
and summary commands was 3,958.704 seconds, about 66 minutes; overlapping Study
durations must not be added as wall time. Preparation and final review are outside
that interval. See [clock scope](reports/campaign-clock-scope.json),
[matched training](reports/matched-training.json) and [serial timing](timing.md).

Each new full campaign passed 74 physical trajectories, 8,523 AI steps and 233
independent native-pressure checks, with all workers reaped. Across both arms,
that is 148 trajectories, 17,046 AI steps and 466 independent checks. The two
broad CPU campaigns ran sequentially with 16 single-thread workers on cores 5–20,
2 GiB accounting per worker and an 8 GiB available reserve. Peak summed worker
RSS was 10,849,009,664 / 10,932,510,720 bytes; minimum available system RAM was
49,426,640,896 / 49,679,126,528 bytes for 0.01 / 0.03. CUDA peak allocated memory
was 870,009,344 bytes per arm (about 830 MiB). These are separately scoped memory
measurements, not a sum of concurrent peaks.

| Population | Worst paired pressure deviation, current 0.1 | New 0.01 | New 0.03 |
|---|---:|---:|---:|
| Original 58 | 3.316357e-6 | 4.569538e-6 | 4.140443e-6 |
| Late amended 8 | 1.089718e-6 | 1.058658e-6 | 1.819289e-6 |
| Minimum-8 failed 8 | 1.657885e-6 | 1.739476e-6 | 1.377527e-6 |

Pressure values are the worst paired trajectory pressure relative-L2 deviation
between separately evolved AI and native trajectories. Independent same-RHS
oracle errors are separate metrics retained in the full report. Some populations
improved, but the gates apply independently to every population
against all three preregistered comparators. Current 0.01 regressed original and
hard-population pressure against the accurate comparator and inference work on
several populations. Current 0.03 regressed original/late pressure and hard-case
work. Physical qualification therefore passes while promotion is rejected. The
full comparator matrix, every case and exact failed gates are in
[regression results](reports/regression-summary.json) and the
[decision receipt](reports/qualification-decision.json).

The older no-current, band-0.03 model is additional historical context only: its
mean/max calls were 7.500997/11 and it used a different training work budget. It
must not be treated as a matched third arm of this experiment.

The shared train-RHS diagnostic exposes local improvements that did not transfer
to autonomous trajectories. The 0.01 candidate's early two-call mean-free pressure
error improved from median/max 0.010709/0.196360 to 0.007953/0.101925, while its
late worst error increased from 0.169402 to 0.235989. The 0.03 candidate had
early median/max 0.011458/0.278474 and late 0.023292/0.203451. All 144 pair hashes
and preparation bytes match; independent label/operator checks pass. See
[common-RHS results](reports/trained-common-rhs-summary.json).

The four-model spatial transfer report retains all 1,008 rows, with pressure,
gradient and equation errors, known-signal energy and high-to-low leakage. Pure
low-frequency relative errors barely change across amplitudes from 1 to 1e-8 Pa;
for example, the 0.01 one-call median is about 0.190177 at both endpoints. This
weakens an absolute amplitude-floor explanation on these manufactured fields.
Mixed signals still show leakage and relative dynamic-range effects. One-call
high-frequency error in 0.03 decreases under repeated corrections; it is
not proof of divergent autonomous motion. Spatial mode indices are not temporal
Hz, and none of these diagnostics establishes a causal noise floor. Full tails
remain in [transfer JSON](reports/transfer-summary.json), its adjacent CSV and
[trajectory spectra](reports/sparse-spectrum-summary.json).

All six serial timing batches completed after broad pools drained. Repeated
fields and calls match exactly. The AI branch includes an additional same-request
native diagnostic solve, capture, correction, IPC and logging. Its timing cannot
be presented as standalone AI inference or compared as a production solver
speedup by subtracting the native branch.

No Tuldok or Rheon source changed. Existing Models/Studies/Runs belong to the
original Tuldok catalog; the original Rheon desktop remains open. The recorded
pressure preview is an observation aid, not a live fluid tracer. Full weights,
datasets and runtime packages remain local under the run directory; source,
methods, small receipts, hashes, clocks and result tables are retained in Git.
Separate read-only Tuldok latency evidence is described in
[handoff preparation](handoff-performance-preparation.md). It did not change the
model or weaken artifact custody.
