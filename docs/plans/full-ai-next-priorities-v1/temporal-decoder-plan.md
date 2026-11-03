# Lead-time, gauge and passive-tracer decoder comparison

The completed pressure-expert K2 fit improves single-block accuracy but fails
every target and becomes severely unstable on240-step own-history rollout.
Train-only controls show that exact anchor pressure does not fix cold velocity
decoding. Its learned pressure response needs different gains at the two lead
times, a pressure gauge shift changes velocity, and zero tracer input creates
nonzero tracer. Preserve that [failed fit and all controls](temporal-pressure-expert-results.md).

Test a combined decoder representation change. This is not a single-factor
attribution. Keep the same frozen pressure expert, managed request initialization,
fluid/request seed tensors, training scenes, normalization, sampling, objective,
10,000-update schedule and 2,000 single/8,000 two-block feedback curriculum.
Tracer parameters and its initial function change. The inherited request and
pressure training are identical to the immediate expert control, and additional
relative to the original models trained without those components.

## Actual model

- Center predicted physical pressure before every fluid decoder. Parameterize
  its learned convolution with zero spatial DC response. This supplies gauge
  invariance without an analytical pressure-gradient stencil or projection.
- Multiply the pressure-conditioned velocity readout by the physical lead time
  `horizon*dt`, retaining learned coefficients and all original velocity heads.
  This is an inductive bias to test, not a proof of correct accumulated dynamics.
- Predict passive-tracer weights from fluid features only. A neural softmax
  mixes27 current tracer donors, optionally with27 predicted endpoint/previous
  tracer donors. No native backtrace or analytical advection supplies weights.
  The resulting map is linear in tracer for fixed fluid, preserves zero and
  constants, and stays within donor range up to float32 interchange roundoff.
  Float64 weighting avoids unnecessary loss of these readout properties.

Both endpoint/intermediate and retained-feature paths return every P/V/T state
for K=2/3/4. Each block retains one fluid encoder and one frozen pressure-expert
call. The intermediate tracer decoder consumes the predicted endpoint tracer;
true future states remain offline labels only. Trainable tensors are float32;
the pressure component and tracer weighting use mixed precision as disclosed
in their source and registration. All work belongs in complete block timing.

## Qualification and first bounded fit

Before training, exercise the actual managed constructors and deployed two-block
feedback on CPU and CUDA. Cover all four grids, both 8/16 cells-per-metre values,
cold/warm starts and all six architecture/horizon combinations:96cases per
device. Require deterministic active trainable gradients, unchanged frozen
coefficients, CPU/device parity, fluid independence from tracer/previous pressure,
forcing dependence, gauge invariance, zero/constant tracer preservation,
tracer linearity/range and actual predicted-endpoint tracer use. Probe assignments
activate diagnostic copies only; qualification makes zero optimizer updates.

Admit one K=2 endpoint fit after both qualifications pass. Require its exact
managed CPU restore, every development window and12/48/240-step own-history
evaluation. Compare against the completed expert control with matched data,
sampled stream, loss and feedback, while reporting changed parameter count and
tracer initialization. Predeclared train-only oracle/gauge/zero-tracer diagnostics
remain excluded from candidate frontiers.

The original accuracy criterion remains every equation/P/V/T error at or below
0.004. Complete CPU budgets remain below 4/6/8 ms for K=2/3/4, with no internal-stage
2ms cap. Measure new complete-block CPU latency from accuracy-qualified weights;
never add old component timings. Keep all eight reserved scenes unopened until
a full-state finalist qualifies, then replicate and freeze before reserved
capture. Structural guarantees alone do not qualify prediction accuracy.

[Frozen registration](temporal-decoder-registration-01.json) · [Original immutable plan](plan.md)

## Current execution

All 96 actual-constructor cases pass on each of CPU and CUDA, including both
spatial densities. Probe pressure-gauge changes produce zero observed decoder
change; maximum CPU/device output difference is 2.39e-7. All trainable tensors
have active deterministic two-block gradients. The K=2 endpoint model stores
107,523 parameter elements, of which 14,882 are trainable, versus 111,650/19,009
for its immediate expert control. Fixed cached learned pressure-head responses
also carry parent information. Parameter count alone supplies no latency claim.

The first controller attempt stopped at a wrong prior-report path before owner
API setup, pair creation or any training submission. That source and failure
are retained. Correcting only the controller guard admits a fresh owned
`temporal-decoder-batch-02` attempt with unchanged model/trainer/Project sources.
Its bounded managed fit, exact CPU restore and dependent evaluations are active.
No new prediction accuracy or CPU latency is established yet.

[CPU qualification](../../../runs/full-ai-next-priorities-v1/temporal-decoder-trainer-cpu-qualification-01.json) ·
[CUDA qualification](../../../runs/full-ai-next-priorities-v1/temporal-decoder-trainer-cuda-qualification-01.json) ·
[Admission](temporal-decoder-admission-01.json)

## Completed outcome

The managed 10,000-update fit, exact CPU restore, every development window and
240-step rollout complete, with the declared control audit passed. Worst
small-grid single-block velocity improves to 0.1789 and tracer to 0.009518, while
pressure/equation worsen to 0.2688/0.6178. All240-step outputs remain finite;
worst P/equation/V/T are2.433/18.56/0.4088/0.6958. The full accuracy target
still fails. Trained gauge and zero-tracer probes both have zero observed change.
CPU latency remains unmeasured; no model is promoted or reserved scene consumed.

[Complete result and controls](temporal-decoder-results.md) ·
[Accumulated-pressure evidence](tied-momentum-results.md) ·
[Next full-state architecture](temporal-coupled-dynamics-plan.md)
