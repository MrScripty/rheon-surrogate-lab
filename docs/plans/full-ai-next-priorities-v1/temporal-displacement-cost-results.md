# Learned displacement CPU cost and accuracy

Five balanced serial repeats time exact restored weights on CPU23, one thread and batch size one. The full scope includes state encoding, prescribed forcing, all internal model work and every physical output; tensor-only timings are retained separately. Each worker uses 48 small-grid and 16 separate 32³ windows.

| Model | Small-grid block ms | Per generated step ms | Block p95 ms | Worst single-block error | Combined pass |
| --- | ---: | ---: | ---: | ---: | --- |
| b2-displacement | 6.3501 | 3.1751 | 21.6392 | 0.0164246063164532 | False |
| b2-coupled | 5.4599 | 2.7300 | 21.0104 | 0.4291320443029117 | False |
| b2-provisional-aux | 5.6138 | 2.8069 | 21.9744 | 0.4468192543381807 | False |

The aggregate median spans different grids. Per-grid values, tails, individual repeats, setup and RSS remain in the evidence. Setup includes model/map construction, case preparation and warmup after numerical imports; it is separate from warm inference. These fixed-window model costs are not measured sustained own-history or application throughput. The same costs accompany long-interval accuracy rows as a comparison, with incomplete rollouts retaining unavailable complete worst errors.

[All timings, accuracy joins and source evidence](temporal-displacement-cost-results.json) · [Full-state accuracy and custody](temporal-displacement-results.md) · [Original objectives](plan.md)
