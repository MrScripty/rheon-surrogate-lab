# Corrected neural family: measured CPU cost and accuracy

Five balanced serial CPU repeats use one thread, batch size one and core23. Complete costs include current-state encoding, prescribed forcing, every intermediate prediction and physical reconstruction. Each fast model preserves every evaluated physical output byte for byte against its exact managed CPU restore.

| Model | Small-grid block ms | Per step ms | Block p95 ms | Worst own240 error | Combined target |
| --- | ---: | ---: | ---: | ---: | --- |
| b1-fixed-viscosity-fast | 2.4301 | 2.4301 | 9.4519 | 0.003721229 | Control |
| b2-fixed-viscosity-fast | 4.1240 | 2.0620 | 18.0276 | 0.003396646 | False |
| b3-fixed-viscosity-fast | 5.5742 | 1.8581 | 26.4835 | 0.003531768 | True |
| b4-fixed-viscosity-fast | 7.4335 | 1.8584 | 34.5601 | 0.003367723 | True |

Sequential B1 comparisons use complete2/3/4-call blocks from reference current state followed by predicted inputs; no future reference state enters the control. Those latency controls do not inherit single-step reference-history accuracy. The full raw controls and bundled speed ratios are reported separately.

The median mixes three small geometries; repeat spread, latency tails, each grid,32³, worker preparation and peak RSS remain separately visible. These fixed-window costs do not establish sustained application or interaction throughput. Replica and reserved-scene gates remain open.

[Raw CPU costs and exact accuracy joins](fixed-viscosity-cost-results.json) · [Every-field accuracy and work audit](fixed-viscosity-family-results.md)
