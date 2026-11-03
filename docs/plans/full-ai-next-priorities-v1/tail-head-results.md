# Learned final pressure block: development result

This records the earlier plain compiled tail. The [later precise execution independently confirms the target](pressure-confirmation-v2/results.md).

`a-tail8` improves pressure accuracy but misses the 4 ms CPU target. These banks are consumed development populations; a new independent confirmation remains outstanding.

All 94,956 parent parameters remain frozen. A 33,982-parameter geometry-conditioned neural head completed 10,000 managed CUDA updates and an exact independent CPU restore. Inference adds one learned eight-mode global block after the two existing fine corrections. Analytical inverse labels remain offline. Larger grids retain the parent prediction bitwise.

| Population | Small-grid requests | Worst equation residual | Worst pressure field error |
| --- | ---: | ---: | ---: |
| historical | 128 | 0.000225506 | 7.93642e-05 |
| initial_fresh | 768 | 0.000474528 | 7.30338e-05 |
| confirmation | 768 | 0.000556144 | 8.99337e-05 |

Complete CPU timings use five balanced serial repeats, one thread, batch one and CPU 23. Each variant warms the entire 192-request population before every repeat. Timing excludes loading and trace setup. The recent-bank timing subset has 48 small-grid and 16 separate 32³ requests; accuracy above uses every request.

| Model | Execution | Historical median ms | Recent small-grid median ms | Recent 32³ median ms |
| --- | --- | ---: | ---: | ---: |
| a-head-calibration-s2-routed | original | 5.635 | 6.081 | 106.784 |
| a-head-calibration-s2-routed | whole_graph | 4.292 | 4.624 | 113.497 |
| a-tail8 | original | 6.549 | 6.873 | 109.344 |
| a-tail8 | whole_graph | 4.370 | 4.802 | 108.476 |

The compiled tail median is 4.802 ms on the recent small-grid mix; repeat medians span 4.666–5.123 ms. Its parent measures 4.624 ms in the same run. Earlier parent measurements of 3.908 ms came from a separate session; use paired measurements when assessing the incremental cost. Warm latency is not cold application throughput.

The complete compiled tail passed all 2,176 consumed requests plus input, gauge, output ownership, label independence and cache invalidation checks. Maximum relative output difference from eager is 1.43597e-09; this is numerical equivalence within the registered tolerance, not bitwise equality.

The original training queue stopped on an evaluation path bookkeeping error after successful training and CPU restore. Corrected evaluation resumed without retraining; the failed attempt and logs remain retained.

Next: reduce complete inference cost while retaining the accuracy margin, run the conditional continuation seeds for a selected recipe, and freeze before a new pressure confirmation. Native application acceptance and temporal rollout accuracy remain separate requirements.

[Evidence and hashes](tail-head-results.json) · [Combined multi-step budgets and architecture options](multi-step-generation-options.md)

The offline temporal component check also passes all 384 true-future-RHS queries: worst small-grid development pressure error is 1.0677e-4 and equation error is 5.4045e-4. This tests the learned pressure component only. It is not causal temporal prediction; future reference RHS remains forbidden in candidate inputs.
