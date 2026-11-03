# Frozen temporal models: reserved accuracy and fresh CPU costs

All primary horizons and added seeds are retained. Every policy has an actual complete reserved accuracy evaluation. Five balanced serial repeats use one CPU thread, core23 and batch one, with complete/tensor scopes and matched sequential B1controls. Original primary seed remains fixed.

| Model | Seed ordinal | Block ms | ms/state | Worst own240 error | Combined median pass |
| --- | ---: | ---: | ---: | ---: | --- |
| b1-fixed-viscosity-fast-float64 | 1 | 2.4521 | 2.4521 | 0.004102316 | Control |
| b2-fixed-viscosity-fast-float32-map | 1 | 3.5390 | 1.7695 | 0.003188158 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 7.0980 | 1.7745 | 0.002819311 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 2.3612 | 2.3612 | 0.00309342 | Control |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 3.4434 | 1.7217 | 0.003793254 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 6.8182 | 1.7046 | 0.002854654 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 2.3545 | 2.3545 | 0.003824814 | Control |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 3.6783 | 1.8392 | 0.003569937 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 6.9093 | 1.7273 | 0.004039953 | False |
| b3-fixed-viscosity-fast-float64 | 1 | 5.5266 | 1.8422 | 0.003041221 | True |

These are warm medians on the original reserved small-grid mix. Per-grid/tails,32³, setup/RSS and sustained application/event throughput are separate. Seeds vary the temporal cell and sampling, with shared frozen component pretraining.

[Every frozen model, fresh costs, reserved accuracy and sources](temporal-reserved-cost-results.json)
