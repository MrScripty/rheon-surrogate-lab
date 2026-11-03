# Controlled supervision of simulation-step changes

The scaled endpoint fit improves pressure but still barely advances velocity.
On fixed training windows, its cold-start velocity median error remains0.9991;
its warm-start velocity median remains close to persistence. This is a training
fit problem as well as a development failure. The current retained-feature
conditioning fit will finish unchanged before the next experiment is admitted.

Compare two10,000-update fits of the same scaled K2 endpoint model:

| Arm | Pressure and equation | Velocity and tracer | Physical terms |
| --- | --- | --- | --- |
| Full-state MSE | Existing relative-MSE terms | Existing full-state relative-MSE terms | Same divergence, mass, energy and bounds terms |
| Increment MSE | Identical to control | Prediction MSE divided by fixed train-only per-horizon increment RMS squared | Identical to control |

The earlier log-loss fit remains a third comparison. The two new MSE arms
separate the normalization question from changing the robust loss form. Both
have163,068 parameters, identical initial tensors, data, sample stream, learning
rate schedule and two-block own-history curriculum. Every absolute future
physical state remains the target, including when the input is the model's own
prediction. The loss does not subtract away accumulated state error.

CPU qualification exercises the actual network and objective constructors on
all four grid groups, cold and warm starts and two predicted-history blocks.
It verifies deterministic active gradients, zero loss for identical states and
bitwise agreement of the full-state MSE arm with the established MSE objective.
The captured Projects match those sources and the fixed training statistics.
The actual constructors and objectives now also pass CUDA qualification on
those cases. The completed-evidence gate admits both managed fit/restore pairs;
the full-state MSE control is fitting first, with the increment arm queued.

Admission is conditional on the completed conditioning comparison: no recipe
may already qualify, and the scaled endpoint's fixed cold-training velocity
median must remain above0.5. This diagnoses severe underfitting; it does not
change the actual accuracy target. The external fit controller checks that
admission before performing any mutation.

All equation, pressure, velocity and tracer errors must still be at most0.004,
at every horizon and through240 own-history steps. The K2 complete-block budget
remains below4ms. No new latency claim, reserved evaluation or model promotion
is justified by this preparation. Any promising K2 result still needs measured
CPU blocks, K3/K4 extensions, replication and frozen reserved evaluation before
the full original goal is complete.

[Frozen training registration](temporal-increment-supervision-registration-01.json) · [Evaluation protocol](temporal-increment-supervision-evaluation-protocol-01.json)

## First control outcome

The full-state MSE control completes 10,000 updates, exact CPU restoration and
all development/240-step checks. It still fails: worst small-grid single-block
pressure error is 1.04348, equation error 0.77314 and velocity error 1.00445.
Training-window velocity improves modestly but remains inaccurate. The
increment-weighted arm is fitting next; the full matched interpretation awaits
that result. These control outcomes do not establish a temporal target pass.

## Completed matched result

Both fits and exact CPU restores complete. All comparison-control audits pass.
Increment weighting improves cold training velocity median from 0.8418 to
0.3664 and warm training tracer median from 0.00190 to 0.000677, but worst
small-grid development velocity is still 0.8224. Equation error worsens from
0.7731 to 1.1538, and240-step tracer error remains 2.1956. No recipe qualifies.
See [all matched results](temporal-increment-supervision-results.md) and the
[next generative velocity representation trial](temporal-curl-plan.md).
