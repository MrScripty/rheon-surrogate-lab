# Temporal step-unit conditioning comparison

Only fixed aggregate train-only forcing and predicted-increment units change. Paired fits verify identical initial parameter tensors, parameter counts, data, physical error normalization, sample stream, objective, curriculum and counted training work. All values are worst relative errors across all declared scenes and horizons. No reserved scenes were consumed.

| Architecture | Grids | Interval | Field | Original | Step-scaled |
| --- | --- | --- | --- | ---: | ---: |
| endpoint | small | single_block | pressure | 1.63989 | 1.23846 |
| endpoint | small | single_block | equation | 1.19849 | 0.809676 |
| endpoint | small | single_block | velocity | 0.999585 | 0.99992 |
| endpoint | small | single_block | tracer | 0.0189023 | 0.019605 |
| endpoint | small | own_history12 | pressure | 1.55082 | 1.18777 |
| endpoint | small | own_history12 | equation | 0.693105 | 0.798359 |
| endpoint | small | own_history12 | velocity | 0.999585 | 0.999972 |
| endpoint | small | own_history12 | tracer | 0.00960183 | 0.0100811 |
| endpoint | small | own_history48 | pressure | 1.5736 | 1.20108 |
| endpoint | small | own_history48 | equation | 0.705704 | 0.798359 |
| endpoint | small | own_history48 | velocity | 0.999585 | 1.00005 |
| endpoint | small | own_history48 | tracer | 0.15467 | 0.157565 |
| endpoint | small | own_history240 | pressure | 1.83554 | 1.39647 |
| endpoint | small | own_history240 | equation | 1.09106 | 1.19494 |
| endpoint | small | own_history240 | velocity | 0.999585 | 1.00005 |
| endpoint | small | own_history240 | tracer | 2.51743 | 2.58878 |
| endpoint | 32cubed | single_block | pressure | 1.17716 | 0.455976 |
| endpoint | 32cubed | single_block | equation | 1.19864 | 0.518283 |
| endpoint | 32cubed | single_block | velocity | 0.998478 | 0.999521 |
| endpoint | 32cubed | single_block | tracer | 0.00495946 | 0.00542816 |
| endpoint | 32cubed | own_history12 | pressure | 1.18379 | 0.425594 |
| endpoint | 32cubed | own_history12 | equation | 1.02983 | 0.475239 |
| endpoint | 32cubed | own_history12 | velocity | 0.998543 | 0.999623 |
| endpoint | 32cubed | own_history12 | tracer | 0.0085964 | 0.00217273 |
| endpoint | 32cubed | own_history48 | pressure | 1.18379 | 0.425594 |
| endpoint | 32cubed | own_history48 | equation | 1.02983 | 0.494854 |
| endpoint | 32cubed | own_history48 | velocity | 0.998681 | 0.99966 |
| endpoint | 32cubed | own_history48 | tracer | 0.0401073 | 0.0268941 |
| endpoint | 32cubed | own_history240 | pressure | 1.18379 | 0.482363 |
| endpoint | 32cubed | own_history240 | equation | 1.02983 | 1.02428 |
| endpoint | 32cubed | own_history240 | velocity | 0.998681 | 0.99966 |
| endpoint | 32cubed | own_history240 | tracer | 0.671634 | 0.641234 |
| reuse | small | single_block | pressure | 1.74979 | 1.03753 |
| reuse | small | single_block | equation | 1.0992 | 0.826395 |
| reuse | small | single_block | velocity | 0.985982 | 0.999921 |
| reuse | small | single_block | tracer | 0.0186923 | 0.019607 |
| reuse | small | own_history12 | pressure | 1.54487 | 0.995134 |
| reuse | small | own_history12 | equation | 0.825517 | 0.765811 |
| reuse | small | own_history12 | velocity | 0.985982 | 0.999921 |
| reuse | small | own_history12 | tracer | 0.00776989 | 0.0100823 |
| reuse | small | own_history48 | pressure | 1.56997 | 1.00191 |
| reuse | small | own_history48 | equation | 0.825517 | 0.824758 |
| reuse | small | own_history48 | velocity | 0.985982 | 0.999924 |
| reuse | small | own_history48 | tracer | 0.149353 | 0.157471 |
| reuse | small | own_history240 | pressure | 1.86403 | 1.15325 |
| reuse | small | own_history240 | equation | 1.19227 | 1.14635 |
| reuse | small | own_history240 | velocity | 0.985982 | 0.999924 |
| reuse | small | own_history240 | tracer | 2.43611 | 2.58751 |
| reuse | 32cubed | single_block | pressure | 1.25247 | 0.422108 |
| reuse | 32cubed | single_block | equation | 1.53961 | 0.499284 |
| reuse | 32cubed | single_block | velocity | 0.979753 | 0.999688 |
| reuse | 32cubed | single_block | tracer | 0.00481542 | 0.0054243 |
| reuse | 32cubed | own_history12 | pressure | 1.1117 | 0.404126 |
| reuse | 32cubed | own_history12 | equation | 0.964474 | 0.460943 |
| reuse | 32cubed | own_history12 | velocity | 0.979753 | 0.999688 |
| reuse | 32cubed | own_history12 | tracer | 0.00722421 | 0.0021104 |
| reuse | 32cubed | own_history48 | pressure | 1.14598 | 0.405129 |
| reuse | 32cubed | own_history48 | equation | 0.964474 | 0.486827 |
| reuse | 32cubed | own_history48 | velocity | 0.979753 | 0.999688 |
| reuse | 32cubed | own_history48 | tracer | 0.0363458 | 0.0266742 |
| reuse | 32cubed | own_history240 | pressure | 1.22399 | 0.464017 |
| reuse | 32cubed | own_history240 | equation | 0.993221 | 1.05736 |
| reuse | 32cubed | own_history240 | velocity | 0.979753 | 0.999688 |
| reuse | 32cubed | own_history240 | tracer | 0.646141 | 0.639982 |

Small-grid accuracy-qualified recipes: none.
CPU latency remains unmeasured for these new fits; accuracy alone does not establish the combined target. Fixed-window train/development and cold/warm fitting diagnostics remain in the evidence.

[Full comparison and audit evidence](temporal-step-scaled-results.json)
