# Frozen temporal models: original reserved-scene accuracy

All ten frozen checkpoints/policies are evaluated, retaining every additional seed and all requested horizons. Eight whole scenes were reserved before capture. Each reference contains the initial state and240consecutive native steps; all archives and native child custody are verified. No model or primary seed is reselected.

| Model | Seed | Grid group | Interval | Complete | Worst error | Accuracy pass |
| --- | ---: | --- | --- | --- | ---: | --- |
| b1-fixed-viscosity-fast-float64 | 1 | small | single_block | True | 0.001791904 | True |
| b1-fixed-viscosity-fast-float64 | 1 | small | rollout_12 | True | 0.000310126 | True |
| b1-fixed-viscosity-fast-float64 | 1 | small | rollout_48 | True | 0.000310126 | True |
| b1-fixed-viscosity-fast-float64 | 1 | small | rollout_240 | True | 0.004102316 | False |
| b1-fixed-viscosity-fast-float64 | 1 | 32cubed | single_block | True | 0.000466084 | True |
| b1-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b1-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b1-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_240 | True | 0.001650292 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | small | single_block | True | 0.002077092 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | small | rollout_12 | True | 0.0003101175 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | small | rollout_48 | True | 0.0003101175 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | small | rollout_240 | True | 0.003188158 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | 32cubed | single_block | True | 0.0004660364 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b2-fixed-viscosity-fast-float32-map | 1 | 32cubed | rollout_240 | True | 0.001223344 | True |
| b4-fixed-viscosity-fast-float64 | 1 | small | single_block | True | 0.001936429 | True |
| b4-fixed-viscosity-fast-float64 | 1 | small | rollout_12 | True | 0.000310126 | True |
| b4-fixed-viscosity-fast-float64 | 1 | small | rollout_48 | True | 0.000310126 | True |
| b4-fixed-viscosity-fast-float64 | 1 | small | rollout_240 | True | 0.002819311 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 32cubed | single_block | True | 0.0004660613 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_240 | True | 0.001848883 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | small | single_block | True | 0.002309551 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_12 | True | 0.000310126 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_48 | True | 0.000310126 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_240 | True | 0.00309342 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | single_block | True | 0.0004660741 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_240 | True | 0.0009134984 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | small | single_block | True | 0.001970008 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | small | rollout_12 | True | 0.0003101175 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | small | rollout_48 | True | 0.0003101175 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | small | rollout_240 | True | 0.003793254 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 32cubed | single_block | True | 0.0004660363 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 32cubed | rollout_240 | True | 0.001863401 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | small | single_block | True | 0.002175506 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_12 | True | 0.000310126 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_48 | True | 0.000310126 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | small | rollout_240 | True | 0.002854654 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | single_block | True | 0.0004660669 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 32cubed | rollout_240 | True | 0.001480316 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | small | single_block | True | 0.001936444 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_12 | True | 0.000310126 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_48 | True | 0.000310126 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_240 | True | 0.003824814 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | single_block | True | 0.0004660374 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_240 | True | 0.001084011 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | small | single_block | True | 0.001783729 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | small | rollout_12 | True | 0.0003101175 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | small | rollout_48 | True | 0.0003101175 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | small | rollout_240 | True | 0.003569937 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 32cubed | single_block | True | 0.0004660684 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 32cubed | rollout_240 | True | 0.001067103 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | small | single_block | True | 0.002183664 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_12 | True | 0.000310126 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_48 | True | 0.000310126 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | small | rollout_240 | True | 0.004039953 | False |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | single_block | True | 0.0004660732 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 32cubed | rollout_240 | True | 0.001235165 | True |
| b3-fixed-viscosity-fast-float64 | 1 | small | single_block | True | 0.002127937 | True |
| b3-fixed-viscosity-fast-float64 | 1 | small | rollout_12 | True | 0.000310126 | True |
| b3-fixed-viscosity-fast-float64 | 1 | small | rollout_48 | True | 0.000310126 | True |
| b3-fixed-viscosity-fast-float64 | 1 | small | rollout_240 | True | 0.003041221 | True |
| b3-fixed-viscosity-fast-float64 | 1 | 32cubed | single_block | True | 0.0004660514 | True |
| b3-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_12 | True | 0.000319539 | True |
| b3-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_48 | True | 0.000319539 | True |
| b3-fixed-viscosity-fast-float64 | 1 | 32cubed | rollout_240 | True | 0.001059171 | True |

Every equation/pressure/velocity/tracer output uses the original near-zero and4e-3criteria. This report establishes actual reserved accuracy; fresh CPU and sustained application measurements remain separate.

[All models, fields, intervals and source evidence](temporal-reserved-accuracy-results.json)
