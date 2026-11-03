# Controlled temporal loss comparison

The same model, exact initialization, sampled training work and static inputs were verified. Only the registered objective changed.
All values below are worst relative errors across development scenes and output steps. No reserved scenes were consumed.

| Regime | Interval | Field | Original log loss | Relative MSE |
| --- | --- | --- | ---: | ---: |
| small | single_block | pressure | 1.11999 | 1.18009 |
| small | single_block | equation | 1.03072 | 1.06879 |
| small | single_block | velocity | 0.79546 | 0.893832 |
| small | single_block | tracer | 0.00978446 | 0.00978725 |
| small | own_history12 | pressure | 1.50538 | 2.37709 |
| small | own_history12 | equation | 1.94808 | 2.39548 |
| small | own_history12 | velocity | 0.839196 | 0.936234 |
| small | own_history12 | tracer | 0.010088 | 0.0101999 |
| small | own_history48 | pressure | 4.56501 | 11.1107 |
| small | own_history48 | equation | 8.26201 | 18.9374 |
| small | own_history48 | velocity | 1.09759 | 0.936234 |
| small | own_history48 | tracer | 0.157481 | 0.157552 |
| small | own_history240 | pressure | 21.0239 | 28.1892 |
| small | own_history240 | equation | 39.9824 | 90.1375 |
| small | own_history240 | velocity | 2.69855 | 0.936234 |
| small | own_history240 | tracer | 2.57151 | 2.58043 |
| 32cubed | single_block | pressure | 1.04858 | 1.02791 |
| 32cubed | single_block | equation | 0.982137 | 0.998926 |
| 32cubed | single_block | velocity | 0.869869 | 0.837428 |
| 32cubed | single_block | tracer | 0.00270199 | 0.0027009 |
| 32cubed | own_history12 | pressure | 3.01122 | 2.80296 |
| 32cubed | own_history12 | equation | 1.73664 | 2.38893 |
| 32cubed | own_history12 | velocity | 1.05838 | 0.890474 |
| 32cubed | own_history12 | tracer | 0.00211727 | 0.00216408 |
| 32cubed | own_history48 | pressure | 8.31405 | 5.50224 |
| 32cubed | own_history48 | equation | 7.12153 | 8.14289 |
| 32cubed | own_history48 | velocity | 1.49279 | 0.89109 |
| 32cubed | own_history48 | tracer | 0.0270547 | 0.0268423 |
| 32cubed | own_history240 | pressure | 37.5485 | 20.737 |
| 32cubed | own_history240 | equation | 40.0511 | 66.7939 |
| 32cubed | own_history240 | velocity | 2.81171 | 1.01375 |
| 32cubed | own_history240 | tracer | 0.679206 | 0.639253 |

## Fixed-window fitting diagnosis

Four predeclared starting times per scene; this is a fitting diagnostic, not the all-window criterion.

| Split | Phase | Field | Original median | MSE median | Original worst | MSE worst |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train | cold | pressure | 0.97022 | 0.955072 | 1.01766 | 1.07768 |
| train | cold | equation | 0.977387 | 0.982474 | 1.00434 | 1.02416 |
| train | cold | velocity | 0.65939 | 0.680219 | 0.741027 | 0.787739 |
| train | cold | tracer | 5.40829e-05 | 5.22188e-05 | 8.89149e-05 | 0.000148972 |
| train | warm | pressure | 0.0806367 | 0.0703053 | 0.159034 | 0.152973 |
| train | warm | equation | 0.119141 | 0.1035 | 0.326902 | 0.319651 |
| train | warm | velocity | 0.00436225 | 0.00888018 | 0.0657158 | 0.106896 |
| train | warm | tracer | 0.00152139 | 0.00152878 | 0.00859351 | 0.00861706 |
| development | cold | pressure | 0.973695 | 0.976984 | 1.11999 | 1.18009 |
| development | cold | equation | 0.981506 | 0.977972 | 1.03072 | 1.06879 |
| development | cold | velocity | 0.711673 | 0.793954 | 0.869869 | 0.893832 |
| development | cold | tracer | 4.16023e-05 | 6.77582e-05 | 8.2726e-05 | 0.000128578 |
| development | warm | pressure | 0.0946224 | 0.158397 | 0.29643 | 0.280139 |
| development | warm | equation | 0.174052 | 0.259439 | 0.335387 | 0.453738 |
| development | warm | velocity | 0.00477222 | 0.0107085 | 0.0791751 | 0.126845 |
| development | warm | tracer | 0.00128148 | 0.00128635 | 0.00978446 | 0.00978725 |

Accuracy still insufficient; investigate shared spatial/global capacity before more temporal replicas.
Architecture identity does not constitute a new measured latency result. The completed primary CPU timings remain separately identified.

[Comparison controls](../../../runs/full-ai-next-priorities-v1/temporal-mse-comparison-audit-01.json) · [All evidence hashes](temporal-loss-followup-results.json)
