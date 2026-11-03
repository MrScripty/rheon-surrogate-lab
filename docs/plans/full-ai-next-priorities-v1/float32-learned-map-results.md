# Float32 learned pressure-map execution

The learned neural tensors are unchanged. The derived neural unit-response map and requests use float32 matrix arithmetic on the three small geometries;32³ retains its original component. This changes rounding and receives a complete new accuracy evaluation. No inverse, solver, correction or parameter fitting is added.

| Grid group | Interval | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- |
| small | single_block | 0.0002358896 / 0.001667243 / 0.0001327896 / 1.969804e-05 | True |
| small | rollout_12 | 0.0001009982 / 0.0002907651 / 0.0001327896 / 8.708265e-07 | True |
| small | rollout_48 | 0.0001009982 / 0.0002907651 / 0.0001327896 / 5.907204e-05 | True |
| small | rollout_240 | 0.0007930647 / 0.003395266 / 0.0003792424 / 0.001325998 | True |
| 32cubed | single_block | 0.0006200704 / 0.0006411642 / 0.0003365973 / 7.681109e-07 | True |
| 32cubed | rollout_12 | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491027e-07 | True |
| 32cubed | rollout_48 | 0.00026237 / 0.0003978269 / 0.0003365973 / 7.016604e-07 | True |
| 32cubed | rollout_240 | 0.00026237 / 0.0008968199 / 0.0003365973 / 9.515068e-05 | True |

No CPU speed claim is made before a separate balanced comparison. The original float64 preparation maps remain allocated, so halving the map arithmetic does not establish halved process memory. Every field, horizon, own240 step and original near-zero criterion remains required.

[Actual errors and source evidence](float32-learned-map-results.json)
