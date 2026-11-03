# Coupled-model CPU cost and accuracy

Five balanced serial repeats measure actual restored weights, one CPU thread on core23 and batch size one. Each worker times 48 small-grid and 16 separate32³ requests per scope. Accuracy uses every output horizon and every original field on all eight development scenes.

| Model | Small-grid full block ms | Per generated step ms | Block p95 ms | Worst single-block equation/field error | Combined target |
| --- | ---: | ---: | ---: | ---: | --- |
| b2-coupled | 5.7534 | 2.8767 | 22.2314 | 0.429132 | False |
| b2-provisional-control | 5.5914 | 2.7957 | 21.7468 | 0.4287571 | False |
| b2-provisional-aux | 5.5867 | 2.7933 | 21.8025 | 0.4468193 | False |

The median spans heterogeneous small grids and is not a per-grid or worst-call guarantee. These costs quantify architecture headroom; accuracy failures remain failures. The same reference-history block costs accompany longer-interval error rows as a model-cost comparison, not a measured own-history/application throughput result.

| Model | Grid XYZ | Full block ms | Per generated step ms |
| --- | --- | ---: | ---: |
| b2-coupled | 9×11×11 | 5.7406 | 2.8703 |
| b2-coupled | 15×6×8 | 4.2947 | 2.1473 |
| b2-coupled | 16×16×16 | 20.8931 | 10.4466 |
| b2-coupled | 32×32×32 | 248.4514 | 124.2257 |
| b2-provisional-control | 9×11×11 | 5.5712 | 2.7856 |
| b2-provisional-control | 15×6×8 | 4.2643 | 2.1321 |
| b2-provisional-control | 16×16×16 | 19.9839 | 9.9920 |
| b2-provisional-control | 32×32×32 | 302.4842 | 151.2421 |
| b2-provisional-aux | 9×11×11 | 5.5867 | 2.7933 |
| b2-provisional-aux | 15×6×8 | 4.2618 | 2.1309 |
| b2-provisional-aux | 16×16×16 | 20.0581 | 10.0291 |
| b2-provisional-aux | 32×32×32 | 250.6332 | 125.3166 |

| Worker | Preparation after imports s | Peak benchmark RSS GiB |
| --- | ---: | ---: |
| b2-coupled | 77.282 | 0.875 |
| b2-provisional-control | 98.322 | 0.876 |
| b2-provisional-aux | 77.473 | 0.874 |

Preparation includes model loading, dense-map construction, case preparation and contract warmup; RSS includes the benchmark cases. Dense maps remain unsuitable for production-scale domains. The first attempt recorded zero repeats and reaped its worker after constructor stdout disrupted JSON replies. The fresh wrapper changes logging only; all fits, accuracy evaluations, prediction math, populations and scopes remain unchanged.

[Complete timing, accuracy and source evidence](temporal-provisional-cost-results.json) · [Matched fit outcomes](temporal-provisional-results.md) · [Original goal](plan.md)
