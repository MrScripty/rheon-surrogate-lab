# Float32 learned-map CPU cost and actual accuracy

Five balanced serial CPU repeats use exact restored/composed neural weights, one thread, batch size one and core23. Full costs include physical encoding, known forcing, all intermediate predictions and physical field reconstruction. The float64 baseline retains its qualified accuracy; float32 map arithmetic receives its own complete accuracy evaluation.

| Model | Small-grid block ms | Per generated step ms | Block p95 ms | Worst own240 error | Combined own240 pass |
| --- | ---: | ---: | ---: | ---: | --- |
| b2-displacement-viscosity-fast | 4.1750 | 2.0875 | 18.5433 | 0.003394093 | False |
| b2-displacement-viscosity-fast-f32map | 3.7069 | 1.8534 | 13.4924 | 0.003395266 | True |

The median mixes three small geometries and is not a worst-call or per-grid guarantee. 32³, per-grid latency, tails, preparation and peak RSS are reported separately in the evidence. Preparation includes model/maps, case preparation and warmup after imports. These fixed-window costs do not establish sustained own-history/application throughput. K3/K4, replicas and reserved scenes remain separate requirements.

The first publisher expected its own parent to have completed. Its failed read-only attempt is retained; this dependent publisher validates the original timing responses and worker reaping without repeating numerical work.

[Raw timings, exact accuracy joins and sources](float32-map-cost-results.json) · [Recomputed float32 accuracy](float32-learned-map-results.md)
