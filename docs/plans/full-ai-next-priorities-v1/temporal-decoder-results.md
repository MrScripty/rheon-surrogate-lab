# Lead-time, gauge and passive-tracer decoder comparison

One bounded K2 endpoint fit changes pressure gauge/lead-time conditioning and the passive-tracer readout. The control has identical inherited pressure/request assets, shared seeded fluid/request tensors, data, objective, sampled stream, schedule and feedback. Tracer parameters and its initial function differ; arithmetic work is not FLOP-matched. This combined representation comparison does not isolate a single cause.

| Grids | Interval | Field | Expert control | Decoder |
| --- | --- | --- | ---: | ---: |
| small | single_block | pressure | 0.256432 | 0.268811 |
| small | single_block | equation | 0.472663 | 0.617791 |
| small | single_block | velocity | 0.391067 | 0.178924 |
| small | single_block | tracer | 0.0148124 | 0.00951832 |
| small | own_history12 | pressure | 0.360348 | 0.230786 |
| small | own_history12 | equation | 0.476824 | 0.271865 |
| small | own_history12 | velocity | 0.391067 | 0.178924 |
| small | own_history12 | tracer | 0.00669798 | 0.00754312 |
| small | own_history48 | pressure | 0.360348 | 0.230786 |
| small | own_history48 | equation | 2.51582 | 0.271865 |
| small | own_history48 | velocity | 0.391067 | 0.178924 |
| small | own_history48 | tracer | 0.126168 | 0.0738111 |
| small | own_history240 | pressure | 1.23039e+12 | 2.43253 |
| small | own_history240 | equation | 1.38799e+12 | 18.5637 |
| small | own_history240 | velocity | 6.98315e+09 | 0.408793 |
| small | own_history240 | tracer | 9.54372e+29 | 0.695806 |
| 32cubed | single_block | pressure | 0.291533 | 0.214717 |
| 32cubed | single_block | equation | 0.361652 | 0.334131 |
| 32cubed | single_block | velocity | 0.427267 | 0.20593 |
| 32cubed | single_block | tracer | 0.00584662 | 0.00605245 |
| 32cubed | own_history12 | pressure | 0.337841 | 0.289687 |
| 32cubed | own_history12 | equation | 0.40224 | 0.367406 |
| 32cubed | own_history12 | velocity | 0.427267 | 0.20593 |
| 32cubed | own_history12 | tracer | 0.00164909 | 0.00226576 |
| 32cubed | own_history48 | pressure | 0.337841 | 0.304458 |
| 32cubed | own_history48 | equation | 0.40224 | 0.431742 |
| 32cubed | own_history48 | velocity | 0.427267 | 0.20593 |
| 32cubed | own_history48 | tracer | 0.0160197 | 0.018845 |
| 32cubed | own_history240 | pressure | 1237.17 | 0.411584 |
| 32cubed | own_history240 | equation | 297872 | 53.2016 |
| 32cubed | own_history240 | velocity | 182.376 | 0.20593 |
| 32cubed | own_history240 | tracer | 2.43801e+09 | 0.802766 |

Small-grid full-state accuracy-qualified: False.
The exact CPU restore and declared comparison-control audit pass. Frozen learned coefficients remain unchanged. CPU latency remains unmeasured for these new weights, and previous component/model timings cannot establish a complete-block claim. Reserved scenes remain unopened. The original full-state and own-history objectives remain active.

Predeclared train-only controls separately replace anchor pressure with native truth, shift its arbitrary gauge, and probe zero current tracer. These controls make zero updates, never enter candidate inference and are excluded from frontiers. Their complete metrics and pressure-readout lead-time alignments are included in the machine-readable result.

[All errors, provenance and controls](temporal-decoder-results.json)
