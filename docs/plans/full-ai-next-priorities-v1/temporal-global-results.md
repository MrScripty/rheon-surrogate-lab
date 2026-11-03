# Shared-global temporal capacity comparison

The same data, sampled work, objective and training loop were verified. Capacity changed; the initial prediction function matches the original persistence baseline.
All values below are worst relative errors across development scenes and output steps. No reserved scenes were consumed.

| Regime | Interval | Field | Original log loss | Shared global |
| --- | --- | --- | ---: | ---: |
| small | single_block | pressure | 1.11999 | 1.00265 |
| small | single_block | equation | 1.03072 | 0.994576 |
| small | single_block | velocity | 0.79546 | 0.712315 |
| small | single_block | tracer | 0.00978446 | 0.007997 |
| small | own_history12 | pressure | 1.50538 | 1.0295 |
| small | own_history12 | equation | 1.94808 | 1.32804 |
| small | own_history12 | velocity | 0.839196 | 0.714361 |
| small | own_history12 | tracer | 0.010088 | 0.0109264 |
| small | own_history48 | pressure | 4.56501 | 1.82537 |
| small | own_history48 | equation | 8.26201 | 3.51851 |
| small | own_history48 | velocity | 1.09759 | 0.739463 |
| small | own_history48 | tracer | 0.157481 | 0.115305 |
| small | own_history240 | pressure | 21.0239 | 11.6265 |
| small | own_history240 | equation | 39.9824 | 20.0335 |
| small | own_history240 | velocity | 2.69855 | 1.21702 |
| small | own_history240 | tracer | 2.57151 | 2.35008 |
| 32cubed | single_block | pressure | 1.04858 | 1.00389 |
| 32cubed | single_block | equation | 0.982137 | 0.998395 |
| 32cubed | single_block | velocity | 0.869869 | 0.665813 |
| 32cubed | single_block | tracer | 0.00270199 | 0.00167565 |
| 32cubed | own_history12 | pressure | 3.01122 | 1.13306 |
| 32cubed | own_history12 | equation | 1.73664 | 1.01843 |
| 32cubed | own_history12 | velocity | 1.05838 | 0.775206 |
| 32cubed | own_history12 | tracer | 0.00211727 | 0.004936 |
| 32cubed | own_history48 | pressure | 8.31405 | 1.69874 |
| 32cubed | own_history48 | equation | 7.12153 | 2.72554 |
| 32cubed | own_history48 | velocity | 1.49279 | 0.828974 |
| 32cubed | own_history48 | tracer | 0.0270547 | 0.0247984 |
| 32cubed | own_history240 | pressure | 37.5485 | 5.66322 |
| 32cubed | own_history240 | equation | 40.0511 | 16.8466 |
| 32cubed | own_history240 | velocity | 2.81171 | 1.29259 |
| 32cubed | own_history240 | tracer | 0.679206 | 0.557361 |

## Fixed-window fitting diagnosis

Four predeclared starting times per scene; this is a fitting diagnostic, not the all-window criterion.

| Split | Phase | Field | Original median | Global median | Original worst | Global worst |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train | cold | pressure | 0.97022 | 0.998952 | 1.01766 | 1.00298 |
| train | cold | equation | 0.977387 | 0.992413 | 1.00434 | 1.00109 |
| train | cold | velocity | 0.65939 | 0.469077 | 0.741027 | 0.689072 |
| train | cold | tracer | 5.40829e-05 | 0.00061255 | 8.89149e-05 | 0.0011873 |
| train | warm | pressure | 0.0806367 | 0.0997881 | 0.159034 | 0.258363 |
| train | warm | equation | 0.119141 | 0.120625 | 0.326902 | 0.339444 |
| train | warm | velocity | 0.00436225 | 0.00279962 | 0.0657158 | 0.0491814 |
| train | warm | tracer | 0.00152139 | 0.000590152 | 0.00859351 | 0.00345538 |
| development | cold | pressure | 0.973695 | 0.999092 | 1.11999 | 1.00389 |
| development | cold | equation | 0.981506 | 0.991608 | 1.03072 | 0.998395 |
| development | cold | velocity | 0.711673 | 0.665176 | 0.869869 | 0.712315 |
| development | cold | tracer | 4.16023e-05 | 0.00049886 | 8.2726e-05 | 0.00103365 |
| development | warm | pressure | 0.0946224 | 0.0893175 | 0.29643 | 0.17053 |
| development | warm | equation | 0.174052 | 0.157583 | 0.335387 | 0.238207 |
| development | warm | velocity | 0.00477222 | 0.00529678 | 0.0791751 | 0.0760538 |
| development | warm | tracer | 0.00128148 | 0.000960746 | 0.00978446 | 0.007997 |

Accuracy still insufficient; assess fitting and startup coverage before bundled replications.
Untrained feasibility profiles are not trained-model timings. No new latency target claim is inferred for this one-step diagnostic.

[Comparison controls](../../../runs/full-ai-next-priorities-v1/temporal-global-comparison-audit-01.json) · [All evidence hashes](temporal-global-results.json)
