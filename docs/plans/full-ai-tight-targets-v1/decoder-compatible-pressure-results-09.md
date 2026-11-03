# Decoder-compatible pressure refit: independent results

Independent accuracy confirmation completed; one or more 1e-5 criteria remain unmet. The combined 1.2 ms / 1e-5 goal remains unachieved; higher-density and 600-step stages have not started.

## Managed training and candidate scope

One managed CUDA refinement used 80 LBFGS optimizer calls on the same 144 native pressure/RHS pairs from 16 original training scenes. The 38,673 double learned pressure parameters started from the previous exact managed checkpoint. A separate managed CPU Run restored all five artifacts bitwise and reproduced the double physical-pressure probe. Both owners completed and drained. The complete generator retains 40,017 learned parameters and predicts pressure, three staggered velocities and tracer.

Only the training label coordinates and fixed input projection changed, followed by additional managed optimization. Pressure labels were encoded using the inverse of the small fixed cosine-coordinate matrices offline. This forms no physical pressure inverse, coefficient ratio or analytical gain target. Candidate inference remains the existing learned modal multiplier, learned local operators and learned sampling head. No native pressure solver or numerical correction enters the candidate.

This experiment changes both label coordinates and continued optimization; it does not separately establish their causal contributions. CPU/CUDA objective gradients and typed serialization qualified without updates before the fit. The existing 2 MiB normalization declarations passed the actual publication verifier before admission.

## Independent complete-state accuracy

The checkpoint and inference sources were frozen after all K1–4 development rollouts passed. Sixteen previously unused whole scenes were then captured and scored. They cover the four supported geometries and densities 8/16 cells per metre, with new seeds and forcing parameters. This establishes scene generalization within those geometries, not unseen-shape or higher-density generalization.

| Interval | Domain group | Worst equation | Worst pressure | Worst velocity | Worst tracer |
| --- | --- | ---: | ---: | ---: | ---: |
| Short K1–4 blocks | small | 9.10185e-14 | 2.58771e-15 | 0 | 0 |
| Short K1–4 blocks | 32cubed | 8.76015e-12 | 5.31896e-14 | 1.10503e-12 | 0 |
| 240 own-history steps, K1–4 | small | 2.35746e-05 | 7.75223e-07 | 1.24404e-07 | 6.56559e-08 |
| 240 own-history steps, K1–4 | 32cubed | 6.75336e-05 | 4.89046e-07 | 1.2307e-07 | 5.05683e-08 |

Every actual intermediate state was typed, checked for finiteness, hashed and scored without teacher resets. The equation criterion includes accumulated trajectory differences against the independent native RHS. The accuracy execution uses the frozen contiguous v3 implementation and single-thread FFTW pressure on CPU 23. Timing variants have separately verified complete output equivalence; the final deployment still needs its complete own-history verification under its final timing policy.

## Complete CPU costs on independent current-state inputs

Performance-core allocations are CPU 6, CPUs 6/4 and CPUs 6/4/5/7. The one-core baseline is serial v5; parallel v6 is the baseline for two/four cores. Each is paired with frozen-point v7, sharing identical learned weights and pressure-plan owners. Costs include validation, conditioning, every neural stage and every intermediate physical output. Output hashing is outside the clock. Full-block and raw observations are retained; per-step values are full-block time divided by K.

| Cores | Steps/pass | Baseline median ms/step | Frozen-point median | Frozen-point p95 | Frozen-point maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 8.762 | 7.863 | 11.967 | 12.039 |
| 1 | 2 | 8.551 | 7.804 | 10.801 | 11.820 |
| 1 | 3 | 8.500 | 7.641 | 9.805 | 10.477 |
| 1 | 4 | 8.433 | 7.586 | 10.125 | 10.137 |
| 2 | 1 | 5.564 | 5.034 | 6.312 | 6.596 |
| 2 | 2 | 5.388 | 4.945 | 6.102 | 6.195 |
| 2 | 3 | 5.362 | 4.860 | 5.815 | 5.984 |
| 2 | 4 | 5.395 | 5.018 | 5.920 | 6.028 |
| 4 | 1 | 4.535 | 4.149 | 6.740 | 7.470 |
| 4 | 2 | 5.669 | 5.192 | 9.280 | 9.947 |
| 4 | 3 | 5.058 | 4.779 | 6.532 | 8.824 |
| 4 | 4 | 4.715 | 4.391 | 6.069 | 6.601 |

All 3,840 timed blocks across the three allocations were byte-identical within their paired execution comparisons. Single-core performance remains a primary efficiency metric. More cores provide an optional latency tradeoff, without a linear-scaling assumption. No timing outliers were removed.

## Disposition

Retain this managed checkpoint and its full independent evidence. The same-request pressure component, all development trajectories and all three preregistered K1 pressure-execution controls are recorded separately from independent confirmation. Multi-step execution still uses internal recurrence and exposes every intermediate state; no endpoint-only shortcut is claimed.

Next work is to reduce complete execution cost while preserving this checkpoint and its numerical behavior, then verify the selected final CPU deployment under the full accuracy and latency contract. Higher-density training remains behind all K1–4 joint gates, followed by the 600-step 60/30 Hz objective. These later objectives remain active and incomplete.

[Machine-readable evidence](decoder-compatible-pressure-summary-09.json) · [Frozen selection](decoder-modal-frozen-selection-09.json) · [CPU resource policy](cpu-resource-policy-04.md)

Independent accuracy failures must also be resolved. These 16 scenes are now consumed confirmation evidence; any learned successor requires new scene reservations before it can receive independent confirmation.
