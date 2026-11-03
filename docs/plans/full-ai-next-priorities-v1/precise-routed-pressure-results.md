# Precise learned-matrix pressure execution

The [new independent confirmation](pressure-confirmation-v2/results.md) is complete and passes for all three seeds. The consumed development comparison below remains separate.

The registered precise routed execution reaches the requested accuracy and warm median CPU targets on consumed development banks. Independent confirmation and the two continuation seeds remain outstanding.

| Population | Complete CNN median ms | Precise routed median ms | Full-population worst equation error |
| --- | ---: | ---: | ---: |
| historical_small | 4.161 | 1.909 | 0.000225505 |
| fresh_small | 4.586 | 2.075 | 0.00055613 |
| fresh_32cubed_extension | 84.237 | 82.997 | 0.000547369 |

Accuracy uses every historical/consumed-bank request; timing uses 128 historical, 48 recent small-grid and 16 separate 32³ requests in five balanced serial repeats. One CPU thread, batch one and CPU 23. All 2,176 requests pass numerical qualification, preserving target classification. Maximum pressure relative difference is 5.29e-8 and maximum equation-error difference is 3.22e-8. This is not bitwise equality.

| Recent small grid | CNN median ms | Precise routed median ms |
| --- | ---: | ---: |
| 16×16×16 | 8.026 | 7.925 |
| 9×11×11 | 4.586 | 2.075 |
| 15×6×8 | 3.300 | 1.564 |

The 2.075 ms small-grid median spans heterogeneous grids. The 16³ cube still takes approximately 7.93 ms. The 32³ extension is approximately 83 ms and uses the unchanged parent path. These timings do not establish performance at production grid sizes.

The frozen spatial CNN is bias free and linear. For at most 2,048 voxels, materialize its learned coefficients using float64 unit responses and use a float64 matrix product, casting the fine output back to float32. Larger small grids retain the original qualified compiled CNN. Every global block, contraction gate and physical reconstruction remains. There is no numerical pressure inverse, solver, corrective projection or reference label in candidate inference. Trained parameter count stays 128,938; the matrix is a geometry-specific execution cache.

Cold setup and memory are substantial. In the paired 64-geometry worker, setup takes 179.43 seconds, routed preparation totals 127.87 seconds, and learned unit-response construction accounts for 103.70 seconds. The dense coefficient cache is 439,569,488 bytes (approximately 419 MiB). Peak RSS is 7.30 GB for that worker holding both execution paths and many layouts; this is not a standalone three-grid memory measurement.

For the recent three geometries, observed preparation costs are approximately 0.58 seconds for 16³, 3.09 seconds for 9×11×11 and 1.45 seconds for 15×6×8. Python/Torch import and checkpoint loading are extra. For 9×11×11, amortizing its setup over 240 requests gives approximately 14.94 ms per request, despite the 2.075 ms warm median. Warm target success is not cold application throughput. Per-grid 12/48/240 arithmetic amortization is in the JSON.

The failed float32 and mixed-storage attempts remain retained. The successful version uses the same declared numerical tolerances and unchanged physical targets. The conditional two-seed continuation batch is now running. Select and freeze before a new independent pressure bank; keep the eight reserved temporal scenes untouched.

[Evidence and cold-cost accounting](precise-routed-pressure-results.json) · [Execution investigation](materialized-pressure-execution-notes.md) · [Original plan](plan.md)
