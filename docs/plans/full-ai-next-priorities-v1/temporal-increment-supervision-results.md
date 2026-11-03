# Matched full-state and increment-weighted MSE supervision

All three recipes use the same step-scaled K2 endpoint network, initial tensors, data, sampled windows, schedule and own-history curriculum. The two MSE arms differ only in velocity/tracer field-error normalization; both use the same pressure, equation and physical terms. The earlier log objective is shown separately. Evaluation targets and error floors are unchanged.

| Grids | Interval | Field | Log control | Full-state MSE | Increment MSE |
| --- | --- | --- | ---: | ---: | ---: |
| small | single_block | pressure | 1.23846 | 1.04348 | 0.935781 |
| small | single_block | equation | 0.809676 | 0.77314 | 1.15383 |
| small | single_block | velocity | 0.99992 | 1.00445 | 0.822376 |
| small | single_block | tracer | 0.019605 | 0.019604 | 0.0187646 |
| small | own_history12 | pressure | 1.18777 | 0.995251 | 0.857419 |
| small | own_history12 | equation | 0.798359 | 0.658131 | 0.832194 |
| small | own_history12 | velocity | 0.999972 | 1.03435 | 0.822376 |
| small | own_history12 | tracer | 0.0100811 | 0.0101347 | 0.00843826 |
| small | own_history48 | pressure | 1.20108 | 1.04571 | 0.896438 |
| small | own_history48 | equation | 0.798359 | 0.658131 | 0.840644 |
| small | own_history48 | velocity | 1.00005 | 1.04835 | 0.822376 |
| small | own_history48 | tracer | 0.157565 | 0.157394 | 0.156324 |
| small | own_history240 | pressure | 1.39647 | 1.366 | 1.79329 |
| small | own_history240 | equation | 1.19494 | 1.15242 | 1.45802 |
| small | own_history240 | velocity | 1.00005 | 1.04835 | 1.00629 |
| small | own_history240 | tracer | 2.58878 | 2.55099 | 2.19563 |
| 32cubed | single_block | pressure | 0.455976 | 0.578031 | 0.820142 |
| 32cubed | single_block | equation | 0.518283 | 0.542423 | 0.70033 |
| 32cubed | single_block | velocity | 0.999521 | 1.08886 | 0.614465 |
| 32cubed | single_block | tracer | 0.00542816 | 0.00541376 | 0.00429224 |
| 32cubed | own_history12 | pressure | 0.425594 | 0.549764 | 0.818576 |
| 32cubed | own_history12 | equation | 0.475239 | 0.51018 | 0.620164 |
| 32cubed | own_history12 | velocity | 0.999623 | 1.20573 | 0.649699 |
| 32cubed | own_history12 | tracer | 0.00217273 | 0.00358804 | 0.00586416 |
| 32cubed | own_history48 | pressure | 0.425594 | 0.549764 | 0.826749 |
| 32cubed | own_history48 | equation | 0.494854 | 0.531831 | 0.656571 |
| 32cubed | own_history48 | velocity | 0.99966 | 1.20573 | 0.658192 |
| 32cubed | own_history48 | tracer | 0.0268941 | 0.0277652 | 0.0298767 |
| 32cubed | own_history240 | pressure | 0.482363 | 0.790852 | 0.880531 |
| 32cubed | own_history240 | equation | 1.02428 | 0.898684 | 1.10547 |
| 32cubed | own_history240 | velocity | 0.99966 | 1.20573 | 0.83086 |
| 32cubed | own_history240 | tracer | 0.641234 | 0.639664 | 0.582019 |

Small-grid accuracy-qualified new recipes: none.
Full CPU latency remains unmeasured for these new fits. Every intermediate state and240-step own-history remain required; no reserved scenes were consumed and no model is promoted.

[All evidence, fitting diagnostics and control audits](temporal-increment-supervision-results.json)
