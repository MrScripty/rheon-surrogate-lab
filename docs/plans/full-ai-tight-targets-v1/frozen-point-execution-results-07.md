# Frozen learned point execution: controlled results

Compiling the existing learned point projection into specialized code reduces single-core latency without changing the checked physical output bytes. Four-core results are mixed. The complete 1.2 ms maximum and 1e-5 own-history targets remain unmet.

## Mechanism and provenance

The generator reads the exact inherited managed CPU checkpoint through the existing uniform inference quantization and CSR ordering. It emits the 33 nonzero point coefficients as exact hexadecimal double literals, preserving term order. Carrier indices become compile-time constants. Query gates and mixing remain the original 48 managed-trained double parameters. Viscosity, flux, pressure, momentum, forcing and per-voxel arithmetic are unchanged. No analytical stencil or new weight is assigned, and no learning occurs.

The compiled kernel checks all incoming point CSR rows, channels, offsets and weights against its generated constants before use. Separate controls reject a changed member of each array and an undeclared three-worker configuration. The generated header, source checkpoint provenance, strict compiler flags, binary and all registrations are retained.

## Correctness

One and four performance-core configurations each passed 336 boundary, extreme and subnormal donor-address controls plus 64 complete four-step state comparisons against the declared tensor graph. All complete fields were also byte-identical to the previous serial implementation. Every output in 1,280 alternating timed complete blocks matched the baseline bytes. No independent scene reservation was consumed. Long-history accuracy was not rerun for this execution-only change; the previous measured accuracy failures remain active.

## Paired complete latency at 32 cubed

| Cores | Steps/pass | Baseline median ms/step | Specialized median ms/step | Baseline/specialized | Specialized p95 | Specialized maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 8.184 | 7.393 | 1.107 | 8.431 | 8.581 |
| 1 | 2 | 8.158 | 7.404 | 1.102 | 8.357 | 8.467 |
| 1 | 3 | 8.018 | 7.241 | 1.107 | 8.301 | 8.545 |
| 1 | 4 | 8.385 | 7.655 | 1.095 | 8.508 | 8.516 |
| 4 | 1 | 4.669 | 4.253 | 1.098 | 6.155 | 6.302 |
| 4 | 2 | 5.024 | 4.777 | 1.052 | 9.075 | 10.815 |
| 4 | 3 | 4.601 | 4.428 | 1.039 | 7.058 | 9.788 |
| 4 | 4 | 4.940 | 5.020 | 0.984 | 7.263 | 7.627 |

The one-core comparator is the faster serial v5 implementation on CPU 6. The four-core comparator is v6 on CPUs 6/4/5/7. Each pair shares the same learned model and pressure-plan owners, with alternating execution order and five repeats per scene/start/horizon. Input checks, forcing, every model stage, complete intermediate outputs and synchronization inside inference count. Output hashing occurs outside the clock. All raw observations, including timing tails, remain available.

Single-core median latency falls about 9–10% at 32 cubed. Four-core K1–3 medians improve modestly, while K4 is about 1.6% slower in this run. Keep the specialization as a single-core optimization and retain the preceding parallel path for comparison. These are development timings and do not establish independent timing qualification or a maximum-latency guarantee. Compare paired ratios rather than interpreting different absolute timings across sessions as additional optimization gains.

## Profile informing this experiment

| CPU allocation | Median complete profiled ms | Transport | Learned viscosity | Learned pressure | Forcing |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.216 | 5.398 | 1.049 | 0.567 | 0.454 |
| 4 | 3.614 | 1.631 | 0.324 | 0.554 | 0.461 |

These profiles describe the preceding serial/parallel implementations, with instrumentation overhead; they are diagnostic costs, not the specialized model benchmark. Transport remains the largest measured stage. Even removing one stage would leave substantial complete work, so no extrapolated target pass is claimed.

## Next work

Retain single-core efficiency as a primary metric and preserve the two/four-core deployment alternatives. Further speed work must reduce complete work rather than assume additional cores scale linearly. The remaining 32-cubed own-history equation difference also needs resolution before the reference target can pass. Freeze a selected successor and execute independent confirmation before opening higher-density or 600-step stages.

[Machine-readable evidence](frozen-point-execution-summary-07.json) · [CPU resource policy](cpu-resource-policy-04.md) · [Previous complete prototype and CPU results](recurrent-sampler-followup-results-02.md)
