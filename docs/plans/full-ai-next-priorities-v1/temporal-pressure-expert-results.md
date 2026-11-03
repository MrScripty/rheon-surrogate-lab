# Causal full-state pressure-expert follow-up

One bounded K2 endpoint fit reuses the frozen learned pressure expert and managed request initialization. It retains every pressure, velocity and tracer horizon. The control shares data, objective, sampling, schedule and feedback; architecture and initialization differ. Inherited request and pressure training are additional, not matched optimizer work. This result does not isolate a single architectural cause.

| Grids | Interval | Field | Direct increment | Pressure expert |
| --- | --- | --- | ---: | ---: |
| small | single_block | pressure | 0.935781 | 0.256432 |
| small | single_block | equation | 1.15383 | 0.472663 |
| small | single_block | velocity | 0.822376 | 0.391067 |
| small | single_block | tracer | 0.0187646 | 0.0148124 |
| small | own_history12 | pressure | 0.857419 | 0.360348 |
| small | own_history12 | equation | 0.832194 | 0.476824 |
| small | own_history12 | velocity | 0.822376 | 0.391067 |
| small | own_history12 | tracer | 0.00843826 | 0.00669798 |
| small | own_history48 | pressure | 0.896438 | 0.360348 |
| small | own_history48 | equation | 0.840644 | 2.51582 |
| small | own_history48 | velocity | 0.822376 | 0.391067 |
| small | own_history48 | tracer | 0.156324 | 0.126168 |
| small | own_history240 | pressure | 1.79329 | 1.23039e+12 |
| small | own_history240 | equation | 1.45802 | 1.38799e+12 |
| small | own_history240 | velocity | 1.00629 | 6.98315e+09 |
| small | own_history240 | tracer | 2.19563 | 9.54372e+29 |
| 32cubed | single_block | pressure | 0.820142 | 0.291533 |
| 32cubed | single_block | equation | 0.70033 | 0.361652 |
| 32cubed | single_block | velocity | 0.614465 | 0.427267 |
| 32cubed | single_block | tracer | 0.00429224 | 0.00584662 |
| 32cubed | own_history12 | pressure | 0.818576 | 0.337841 |
| 32cubed | own_history12 | equation | 0.620164 | 0.40224 |
| 32cubed | own_history12 | velocity | 0.649699 | 0.427267 |
| 32cubed | own_history12 | tracer | 0.00586416 | 0.00164909 |
| 32cubed | own_history48 | pressure | 0.826749 | 0.337841 |
| 32cubed | own_history48 | equation | 0.656571 | 0.40224 |
| 32cubed | own_history48 | velocity | 0.658192 | 0.427267 |
| 32cubed | own_history48 | tracer | 0.0298767 | 0.0160197 |
| 32cubed | own_history240 | pressure | 0.880531 | 1237.17 |
| 32cubed | own_history240 | equation | 1.10547 | 297872 |
| 32cubed | own_history240 | velocity | 0.83086 | 182.376 |
| 32cubed | own_history240 | tracer | 0.582019 | 2.43801e+09 |

Small-grid full-state accuracy-qualified: False.
The exact CPU restore and declared comparison-control audit pass. Frozen learned coefficients remain unchanged. CPU latency remains unmeasured for these new weights, and previous component/model timings cannot establish a complete-block claim. Reserved scenes remain unopened. The original full-state and own-history objectives remain active.

Predeclared train-only controls separately replace anchor pressure with native truth, shift its arbitrary gauge, and probe zero current tracer. These controls make zero updates, never enter candidate inference and are excluded from frontiers. Their complete metrics and pressure-readout lead-time alignments are included in the machine-readable result.

[All errors, provenance and controls](temporal-pressure-expert-results.json)
