# Compensated interpolation: diagnostic GPU timings

RTX 5090 Laptop GPU; 32×32×32 cells; density 100 cells per metre; eight internal advances per physical output. Complete calls include CPU↔GPU transfers and every output. Median and maximum below come from 20 calls per configuration.

| Steps per call | Parent median ms/step | Candidate median ms/step | Candidate max ms/step | Candidate median block ms |
|---:|---:|---:|---:|---:|
| 1 | 2.281 | 2.298 | 3.989 | 2.298 |
| 2 | 2.182 | 2.147 | 2.772 | 4.294 |
| 3 | 2.172 | 2.103 | 2.791 | 6.309 |
| 4 | 2.130 | 2.070 | 2.412 | 8.280 |

The full balanced study measured 3,840 calls across 24 development scenes, four starting windows, K=1–4 and five repeats. All 9,600 timed outputs and 11,520 warmup outputs matched their previously scored bytes. Shape and density groups remain separate in the JSON report.

Accuracy remains failed: 1 of 1,018 native outputs exceeded the unchanged 1e-5 limit, with worst equation error 1.0918954836401302e-5. Full 240-step candidate histories were not qualified. These timings were requested by the user despite that failure; they do not admit the model or meet the 1.2 ms goal.

Detailed comparisons: `paired-interpolator-results-02.json`. Raw observations: `runs/full-ai-data-diversity-v1/paired-interpolator-diagnostic-benchmark-01/results.json`.
