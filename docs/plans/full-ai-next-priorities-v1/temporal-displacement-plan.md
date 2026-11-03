# Learned displacement full-state comparison

The latest coupled models miss both full-state accuracy and the two-state
4 ms CPU budget. A matched auxiliary-loss continuation did not resolve the
failure. Native controls identify provisional transport and tracer decoding as
useful representation changes. This comparison replaces both with neural
displacements and fixed convex field readouts, while preserving all requested
physical outputs and the original evaluation gates.

The representation prerequisites are complete. Native-qualified face/cell
labels and readouts pass 80 cases per device. A 972-coefficient point layer
completes 1,000 ordinary managed CUDA updates from zero, exact CPU restoration,
and 24 native geometry/density checks of its published float32 coefficients.
Worst native point error is 2.51e-8. These are component results; they supply
no full-state forecast or CPU timing claim.

## Implemented inference

Encode current velocity and geometry once per block. For each requested
physical horizon, update a small retained hidden state from its prior features
and entering predicted velocity. One spatial projection combines the frozen
learned point coefficients with trainable local velocity features.

Predict 12 displacement channels: three axes at each velocity-face and
tracer-cell destination. In normalized units, the displacement is
`alpha * learned_point_velocity + alpha^2 * learned_curvature`, with
`alpha = dt * velocity_scale / h`. The curvature decoder uses bilinear,
upwind-sensitive and cubic local features conditioned on retained hidden
features. Its initial output is zero. Displacements have bounded subcell
support and feed a fixed eight-donor convex tensor readout of the four fields.

Transport uses entering velocity before current-step forcing. Apply known
force and a learned viscosity readout, then the same pretrained learned
flux, pressure map and tied momentum readout as the prior coupled model.
Return pressure, staggered velocity and tracer at every horizon. Reconstruct
and re-encode each predicted physical state before the next internal update
and across blocks. Numerical pressure solves, native backtracing and reference
future fields remain absent from candidate inference.

The shared pressure representation retains its measured setup, memory and
larger-grid limitations. Dense small-grid learned maps do not establish a
scalable implementation for production domains.

## Actual training and qualification

The implemented trainer retains the original training scenes, normalization,
step-unit physical objective and sampling stream. It adds displacement
supervision at weight one, in alpha-squared units so the small curvature signal
is visible. Labels use the actual entering predicted velocity and are computed
after each prediction, without gradients or future-state inference inputs.
They also cover the model's own history in the second block.

The intended bounded fit uses 10,000 managed updates: 2,000 single-block and
8,000 two-block updates, Adam at 0.001 then 0.0003 after update 8,000. These
settings are captured in the four K configurations. Model and objective changes
are joint changes; this is not a single-factor attribution experiment.

Before admission, the actual constructors must pass CPU/CUDA checks for K=1,
2, 3 and 4, all four geometries, both densities and cold/warm states. Checks
exercise repeatable two-block gradients, frozen coefficient custody, active
trainable paths, device parity, one encoder per block, every internal readout,
passive tracer behavior and physical staging. The training label function is
disabled during inference checks. Qualification makes zero optimizer updates.

Then capture the managed full-state Project, freeze its sources and objective,
fit a bounded K2 comparison and restore its exact CPU artifacts. Evaluate every
development horizon and own-history 12/48/240 before selecting further work.
Test all requested K values; retain the K1 sequential control for confirmed
sharing comparisons. Replicate and freeze qualifying finalists before opening
reserved trajectories.

The complete budgets remain below 4/6/8 ms for K=2/3/4, without individual-stage
caps. Every equation/P/V/T output must satisfy 0.004 over its declared interval.
Measure actual full-block CPU costs, setup and memory; component accuracy and
old timings cannot qualify this model.

[Component evidence](point-sampling-results.md) ·
[Constructor qualification protocol](temporal-displacement-qualification-protocol-01.json) ·
[Original immutable objectives](plan.md)
