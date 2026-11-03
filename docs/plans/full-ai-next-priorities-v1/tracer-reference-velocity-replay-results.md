# Scalar transport replay under reference velocity

The first-order displacement control advances its own tracer for 128 steps while receiving native reference velocity at every step. It covers all 16 training scenes and 2,048 outputs. The offline center-velocity interpolation matches 64 independent native sampler cases exactly.

| Grid group | Interval | Worst relative tracer error |
| --- | ---: | ---: |
| small | 12 | 1.7705555e-06 |
| small | 48 | 0.0001174828 |
| small | 128 | 0.0012068056 |
| 32cubed | 12 | 1.0030246e-07 |
| 32cubed | 48 | 3.5854952e-06 |
| 32cubed | 128 | 7.4542143e-05 |

This is a scalar transport control, with supplied reference fluid history and no learned displacement predictor. It does not meet the full-state or 240-step forecasting objective, establish CPU model latency, or trigger reserved capture. It supports testing a compact learned displacement decoder with convex readout; an actual model must use its own predicted fluid history and pass every original gate.

[Source-bound results](tracer-reference-velocity-replay-results.json) · [Single-step representation checks](tracer-representation-results.md) · [Original scope](plan.md)
