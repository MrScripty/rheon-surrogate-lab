# Qualified forcing and state preparation CPU costs

Five balanced serial CPU repeats use exact restored/composed neural weights, one thread, batch size one and core23. Full costs include physical encoding, known forcing, all intermediate predictions and physical field reconstruction. The fused implementation matches all3,856physical outputs byte for byte, so it inherits the same accuracy.

| Model | Small-grid block ms | Per generated step ms | Block p95 ms | Worst own240 error | Combined own240 pass |
| --- | ---: | ---: | ---: | ---: | --- |
| b2-displacement-viscosity-fused | 4.4722 | 2.2361 | 17.7108 | 0.003394093 | False |
| b2-displacement-viscosity-fast | 3.9522 | 1.9761 | 16.9643 | 0.003394093 | True |

The median mixes three small geometries and is not a worst-call or per-grid guarantee. 32³, per-grid latency, tails, preparation and peak RSS are reported separately in the evidence. Preparation includes model/maps, case preparation and warmup after imports. These fixed-window costs do not establish sustained own-history/application throughput. K3/K4, replicas and reserved scenes remain separate requirements.

[Raw timings, exact accuracy joins and sources](fast-preparation-cost-results.json) · [Repaired full-state accuracy](neural-viscosity-results.md)
