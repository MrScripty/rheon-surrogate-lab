# Frozen learned pressure component: batched differentiable port

The existing learned pressure component now passes CPU and CUDA qualification.
This is component preparation, with zero optimizer updates. It is not a causal
full-state temporal model and supplies no new inference-latency claim.

| Device | Native requests | Gradient cases | Maximum parent pressure difference | Maximum equation-error difference |
| --- | ---: | ---: | ---: | ---: |
| cpu | 128 | 32 | 1.23995e-08 | 2.07715e-08 |
| cuda | 128 | 32 | 8.26776e-09 | 2.31964e-08 |

Four grids, both cells-per-metre values, batch sizes1–4 and deterministic
input gradients qualify. Learned coefficients remain unchanged. Native future
requests enter only this offline component check; reserved scenes remain unopened.
The unchanged independently confirmed pressure implementation is the reference.

Precise dense CNN maps are constructed from learned unit responses at setup,
without physical inverses or reference labels. They occupy13,634,568bytes;
setup times are recorded separately and are not a warm-latency benchmark.
The first float32 numerical failure and subsequent CUDA loader-order preflight
failure are retained with their source snapshots. Numerical tolerances did not change.

Next: integrate the pressure expert with causal request, velocity and tracer
prediction, then qualify the deployed full-state feedback path before fitting.
Keep endpoint/intermediate and retained-feature comparisons and complete4/6/8ms
budgets. Every physical output and own-history accuracy criterion remains open.

[Machine-readable evidence](temporal-pressure-core-port-results.json) · [Next implementation plan](temporal-pressure-expert-reuse-plan.md)
