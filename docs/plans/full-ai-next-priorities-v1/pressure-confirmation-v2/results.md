# Independent learned-tail pressure confirmation

The fixed primary passes the requested C107-successor prepared-pressure target: **2.133 ms median CPU inference and 4.6963e-4 worst equation residual**. Both additional continuation seeds pass too. The source, weights, primary choice and 16 new scenes were frozen before capture.

| Model | Small-grid median ms | Worst residual across 768 requests | Passing requests | Native complete episodes |
| --- | ---: | ---: | ---: | ---: |
| C107 | 1.642 | 0.001562049 | 761/768 | 0/16 |
| a-tail8 | 2.133 | 0.0004696256 | 768/768 | 0/16 |
| a-tail8-s2 | 2.182 | 0.0004696051 | 768/768 | 0/16 |
| a-tail8-s3 | 2.119 | 0.0004696025 | 768/768 | 0/16 |

C107 receives the same execution optimization. It is faster but misses accuracy. The three tail seeds are calibrations from a fixed parent, not independent pretraining. All use verified managed CUDA training and exact CPU restores. There was no seed reselection after confirmation.

Five balanced serial repeats time 48 fixed small-grid requests, with full validation, every learned block and reconstruction. Accuracy uses all 768 small-grid requests. The independent 256-request 32³ extension remains separate. The primary repeat medians span 2.053–2.216 ms; p95 is 8.713 ms. The aggregate median is not a per-grid or worst-latency guarantee.

| Primary grid | Median ms |
| --- | ---: |
| 16×16×16 | 7.707 |
| 9×11×11 | 2.133 |
| 15×6×8 | 1.596 |
| 32×32×32 | 83.759 |

The primary has approximately 7.08 seconds of numerical worker setup after module imports and peaks at approximately 0.98 GB RSS with three small-grid layouts. Python/Torch import time is extra. These standalone three-grid measurements differ from the earlier 64-geometry paired worker, which peaked at 7.30 GB. Cold setup can outweigh warm gains in short episodes.

All four models reject on the first step of every native scene under the unchanged stricter acceptance gates. There is no accepted-trajectory or application-throughput claim. The next priority is the full-state multi-step track; its accuracy and 240-step own-history targets remain unmet. The eight reserved temporal scenes remain unconsumed.

The timing command first failed because the control interpreter lacks NumPy. The frozen timing source then ran successfully under the approved numerical runtime. No weights, evaluation populations or timing algorithm changed, and no training or capture was repeated. Both attempts remain retained.

[Complete error, timing, memory and custody evidence](results.json) · [Original plan](../plan.md) · [Multi-step component plan](../causal-pressure-request-component-plan.md)
