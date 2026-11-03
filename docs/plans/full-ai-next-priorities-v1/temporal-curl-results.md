# Generative curl velocity readout comparison

Both models share initial tensors, training data, sampled windows, optimizer schedule, increment-MSE loss and two-block own-history curriculum. The candidate interprets three learned output channels as edge potentials and reconstructs velocity increments with a fixed linear curl readout. It adds no trainable parameters. Extra readout arithmetic is counted as model work; FLOPs are not matched. Pressure, tracer and all intermediate states remain required.

| Grids | Interval | Field | Direct increment | Curl increment |
| --- | --- | --- | ---: | ---: |
| small | single_block | pressure | 0.935781 | 1.05832 |
| small | single_block | equation | 1.15383 | 1.20885 |
| small | single_block | velocity | 0.822376 | 1.24667 |
| small | single_block | tracer | 0.0187646 | 0.0176969 |
| small | own_history12 | pressure | 0.857419 | 1.04861 |
| small | own_history12 | equation | 0.832194 | 0.892081 |
| small | own_history12 | velocity | 0.822376 | 1.26586 |
| small | own_history12 | tracer | 0.00843826 | 0.0116759 |
| small | own_history48 | pressure | 0.896438 | 1.04861 |
| small | own_history48 | equation | 0.840644 | 0.892081 |
| small | own_history48 | velocity | 0.822376 | 1.26586 |
| small | own_history48 | tracer | 0.156324 | 0.145455 |
| small | own_history240 | pressure | 1.79329 | 1.08979 |
| small | own_history240 | equation | 1.45802 | 1.66831 |
| small | own_history240 | velocity | 1.00629 | 1.64711 |
| small | own_history240 | tracer | 2.19563 | 2.39153 |
| 32cubed | single_block | pressure | 0.820142 | 0.962517 |
| 32cubed | single_block | equation | 0.70033 | 0.909388 |
| 32cubed | single_block | velocity | 0.614465 | 1.77142 |
| 32cubed | single_block | tracer | 0.00429224 | 0.00486248 |
| 32cubed | own_history12 | pressure | 0.818576 | 0.936108 |
| 32cubed | own_history12 | equation | 0.620164 | 0.772914 |
| 32cubed | own_history12 | velocity | 0.649699 | 1.78862 |
| 32cubed | own_history12 | tracer | 0.00586416 | 0.0120207 |
| 32cubed | own_history48 | pressure | 0.826749 | 0.936108 |
| 32cubed | own_history48 | equation | 0.656571 | 0.82161 |
| 32cubed | own_history48 | velocity | 0.658192 | 1.78862 |
| 32cubed | own_history48 | tracer | 0.0298767 | 0.0500785 |
| 32cubed | own_history240 | pressure | 0.880531 | 1.10794 |
| 32cubed | own_history240 | equation | 1.10547 | 6.78875 |
| 32cubed | own_history240 | velocity | 0.83086 | 3.60182 |
| 32cubed | own_history240 | tracer | 0.582019 | 1.40381 |

Small-grid full-state accuracy-qualified: False.
The representation constrains increments, not pre-existing input divergence. Float32 interchange adds roundoff. Constraint satisfaction alone is not prediction accuracy. CPU latency remains unmeasured for the new weights; no reserved scenes were consumed and no model is promoted.

[Complete metrics, physical diagnostics and control audit](temporal-curl-results.json)
