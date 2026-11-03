# Neural viscosity repair: full-state accuracy

Only243 viscosity coefficients change. They come from a completed managed neural fit and exact CPU restore. All other displacement, encoder, pressure and flux tensors are byte-identical to the parent. The assembled model calls no native viscosity primitive during inference.

| Regime | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- | --- |
| small | single_block | True | 0.0002359589 / 0.001667559 / 0.0001327859 / 1.969804e-05 | True |
| small | own_history12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.708265e-07 | True |
| small | own_history48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.907202e-05 | True |
| small | own_history240 | True | 0.0007931372 / 0.003394093 / 0.0003792245 / 0.001326029 | True |
| 32cubed | single_block | True | 0.0006200704 / 0.0006411642 / 0.0003365973 / 7.681109e-07 | True |
| 32cubed | own_history12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491027e-07 | True |
| 32cubed | own_history48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 7.016604e-07 | True |
| 32cubed | own_history240 | True | 0.00026237 / 0.0008968199 / 0.0003365973 / 9.515068e-05 | True |

Accuracy uses every intermediate field on all eight development scenes and through240own-history steps. The original normalization, near-zero rules and0.004gate remain unchanged. CPU speed, K3/K4, seed replication and reserved trajectories remain separate gates; this accuracy report alone does not qualify the full objective.

[Complete evaluation and provenance](neural-viscosity-results.json) · [Managed neural component](viscosity-calibration-results.md) · [Original plan](plan.md)
