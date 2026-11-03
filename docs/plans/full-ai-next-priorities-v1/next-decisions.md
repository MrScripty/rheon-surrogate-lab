# Next full AI priorities after independent pressure confirmation

The [original plan](plan.md), accuracy criteria and combined block budgets remain
unchanged. The pressure benchmark objective is independently confirmed; the
full-state multi-step objectives remain open. No application model is promoted.

## Current decision after the qualified execution comparison

Both first benchmarks now pass on the declared small-grid mixture. Track A
reaches 2.133 ms median pressure inference and 4.6963e-4 worst equation residual.
Track B K2 reaches 3.9522 ms median per complete two-state block (1.9761 ms/state)
and 0.0033941 worst equation/pressure/velocity/tracer error through own240.
The K2 timing margin is small: repeat medians straddle the 4 ms limit, so retain
the full spread and confirm repeatability. Individual calls and production grids
remain slower; no application model is promoted.

Fused prescribed forcing and lean state encoding preserve all 3,856 physical
outputs byte for byte, with unchanged learned tensors. The force primitive also
matches every float32 byte over 7,632 development cases across K1/2/3/4. Every
required output, forcing construction and field reconstruction is counted in
the complete block. Tensor-only results are separate.

Both requested architecture ideas use the combined 4/6/8 ms budgets: an accurate
endpoint plus a smaller learned intermediate decoder, and retained first-state
features with cheaper subsequent updates. Their earlier fits failed accuracy;
the current successful representation uses shared encoding and internal updates,
with learned pressure and displacement components. Preserve those comparisons.

The finite corrected-family sequence now starts actual CPU/CUDA qualification
for K1/2/3/4 with neural viscosity fixed from initialization. Only complete
constructor/gradient passes admit the four managed fits and exact CPU restores.
Each fit supervises 36,000 physical states; different optimizer/update counts
are not equal FLOPs. The original 1,800-second CUDA wall limit is unchanged.
Evaluate every field/horizon and own240; then replicate qualifying recipes and
the one-step control. Keep the eight reserved scenes untouched until recipe
and execution are frozen. Application/event timing must count invalidated
blocks and discarded outputs after unannounced interactions.

[Qualified full CPU costs](fast-preparation-cost-results.md) ·
[All-field accuracy](neural-viscosity-results.md) ·
[Corrected-family work and gates](fixed-viscosity-family-plan.md)

## Retain the confirmed pressure result

The fixed `a-tail8` primary reaches **2.133 ms median CPU inference and 4.6963e-4
worst equation residual** on 768 new small-grid requests. Both additional
continuation seeds pass too. The consistently optimized C107 control measures
1.642 ms but misses accuracy at 1.5620e-3. Source, weights, primary choice and
scene identities froze before capture. Preserve the [independent evidence](pressure-confirmation-v2/results.md)
and all earlier failed fits and execution variants.

These are warm heterogeneous small-grid medians. Individual grids are slower;
32³ remains approximately 84 ms. Standalone primary setup takes about 7.08 seconds
after module imports and peaks near 0.98 GB RSS with three layouts. Every model
still rejects the first native step on all 16 scenes under unchanged stricter
gates. Cold application throughput, native acceptance and production grid
performance are separate limitations. Further pressure sweeps are lower priority
than making temporal prediction accurate.

## Fit the missing causal pressure-request component

The completed generic K=1,2,3,4 models and the loss/capacity follow-ups fail
accuracy. Diagnostics show the larger model mostly copies previous pressure and
barely responds to forcing. Native controls confirm that previous pressure is
not needed to advance the qualified closed-grid reference contract.

The improved learned pressure component passes all 384 offline future-RHS
oracle queries, but that oracle is not causal prediction. The first causal
request fit fails both train and development queries. The factored follow-up
learns and freezes a linear flux response, then trains nonlinear transport from
velocity alone. It passes cold starts and substantially improves median and
worst pressure errors, but still misses the all-window threshold. Preserve both
[component outcomes](factored-pressure-request-results.md); neither includes
velocity, tracer or own-history rollout. Further component-only sweeps are lower
priority than the explicit requested full-state architecture comparison.

## Preserve the completed full-state sharing comparison

Both requested architectures complete K=2,3,4 fits, exact CPU restores,
every-horizon development evaluation,240-step own-history rollout and five
balanced CPU timing repeats. The [full control audit and results](temporal-sharing-results.md)
verify common data, sampling, loss, curriculum and paired encoder initialization.
All intermediate pressure, staggered velocity and tracer states remain included.

Small-grid median complete-block costs are2.86–3.64ms across the six models,
or0.85–1.47ms per generated state. All pass the registered median speed budgets,
but individual latency tails are slower. None passes accuracy; pressure,
velocity and own-history tracer errors are large. No qualifying recipe is
selected for replication or reserved evaluation. The32³ extension remains
outside the speed budget. Preserve these negative results.

The combined limits remain **4/6/8ms for2/3/4states**, with no separate internal
stage cap. Every equation/pressure/velocity/tracer error must stay at or below
4e-3, including240-step own-history. Keep the eight reserved temporal scenes
unopened until a full-state finalist qualifies, then replicate and freeze its
recipe before reserved evaluation.

## Preserve the conditioning result and test actual timestep supervision

The [two conditioning fits](temporal-step-scaled-results.md) and exact CPU
restores complete with matched controls. Pressure improves, but velocity remains
near persistence even on training scenes, and both fail full development and
240-step own-history criteria. Unit conditioning alone is insufficient.

The [next matched loss experiment](temporal-increment-supervision-plan.md) uses
the fixed K2 endpoint architecture, not a new capacity or data sweep. Compare an
established full-state relative-MSE control against an MSE arm whose velocity
and tracer errors use fixed per-horizon native-step scales. Pressure, equation,
physical terms, initial tensors, data, sampling, optimizer schedule and feedback
curriculum match. The older log-loss fit remains a separate reference, so a log
to MSE change cannot be mistaken for evidence about increment scaling alone.

Both matched MSE fits, CPU restores and complete evaluations now finish.
Increment weighting improves training velocity/tracer fits, but development
velocity remains far outside target and equation/long-rollout pressure errors
worsen. Preserve [all three loss outcomes](temporal-increment-supervision-results.md).
No new recipe qualifies for replication or reserved evaluation.

## Retain the failed generative curl comparison

The [curl-readout candidate](temporal-curl-plan.md) changes how neural velocity
outputs are represented, with no additional trainable parameters. Three learned
channels encode staggered edge potentials. A fixed signed linear curl readout
inside the model reconstructs face increments; it is not a pressure solve or a
projection of separately predicted velocity. The representation retains any
existing input divergence and float32 interchange adds roundoff.

Compare against the direct increment-MSE endpoint model using identical initial
tensors, training data, sampling, optimizer schedule and own-history curriculum.
Independent readout constraints and actual two-block CPU/CUDA gradients qualify.
The fit, exact CPU restore and complete evaluation now finish. The control audit
passes. Velocity divergence decreases by roughly five orders of magnitude, but
worst small-grid single-block velocity error worsens from0.8224 to1.2467 and
240-step tracer error rises to2.3915. Pressure and equation errors also fail.
Preserve [the negative result](temporal-curl-results.md). No new CPU latency is
measured and no extension, replica or reserved evaluation is triggered.

## Reuse the successful pressure expert in the next full-state comparison

The next model should reuse the accurate learned pressure component instead of
requiring the shared temporal encoder to learn pressure from scratch. First
qualify a batched, differentiable port at frozen weights against its unchanged
parent, including all four grids, both cells-per-metre values and active input
gradients. This [CPU/CUDA qualification now passes](temporal-pressure-core-port-results.md):
128 native requests and32 gradient cases on each device, with no optimizer
updates. Cached geometry gains come from the learned shape heads; no analytical
inverse replaces them. This is component preparation only.

The actual causal full-state architectures and deployed two-block feedback
qualify on CPU and CUDA for both sharing paths andK2/3/4. The bounded K2 endpoint
fit, exact CPU restore and full development evaluation now complete.
Its request predictor receives current state
and known forcing; future native pressure requests are supervision only. Keep
fluid prediction independent of tracer for the current physical contract.
Compare an accurate predicted endpoint with a small intermediate decoder against
retained first-prediction features and cheaper subsequent latent updates.
Evaluate pressure, velocity and tracer at every horizon, followed by240-step
own-history continuation. Component or oracle success cannot satisfy that test.

The total budgets remain4/6/8ms; expensive and cheap stages may divide them
unevenly. Report complete blocks and amortized time together. The successful
pressure-only2.133ms result cannot be added to an old temporal timing to claim
that a new full-state model passes. See [the implementation order](temporal-pressure-expert-reuse-plan.md).

## Repair measured decoder weaknesses before extending this family

The [full-state pressure-expert fit](temporal-pressure-expert-results.md)
improves small-grid worst single-block pressure from0.9358 to0.2564, equation
from1.1538 to0.4727, and velocity from0.8224 to0.3911. It still misses every
accuracy target and becomes severely unstable by240 steps. Its declared data,
sampling, objective and feedback controls pass; inherited training is additional,
so this comparison does not isolate a single cause. No new CPU timing, replica
or reserved evaluation is triggered.

Predeclared train-only controls identify decoder limitations independent of
the remaining request-prediction error. Replacing the anchor pressure with true
native pressure leaves cold median velocity error approximately0.242, compared
with0.236 normally. The pressure-conditioned velocity readout needs different
scalar alignments at horizons1 and2: cold medians0.780 and1.399, with cosines
above0.99. This supports testing explicit lead-time conditioning rather than
forcing one pressure-response gain to represent both accumulated intervals.
These are offline diagnostic scalars, never fitted into the candidate.

A constant pressure gauge shift changes predicted velocity by up to0.0641
relative to the reference norm. An identically zero current tracer produces
nonzero future tracer, up to2.43e-5 absolute concentration RMS. Neither effect
is evidence that the rollout instability has a single cause. They support a
next decoder representation comparison: pressure features/readouts invariant
to gauge, explicit physical lead time, and passive-tracer dependence that is
linear in tracer and exactly zero when tracer is zero. Retain neural learned
spatial coefficients and the full inference prohibition on numerical solves
and post-corrective projections.

Register the actual comparison and qualify its CPU/CUDA deployed two-block path
before fitting. Keep the current failed source/weights unchanged. Require cold
training accuracy and stable own-history development as well as every-horizon
accuracy before adding more seeds or reserving scenes. A one-step causal control
may help separate decoder stability from temporal sharing, but cannot substitute
for the requestedK2/3/4 full-state objectives. Component-only request sweeps are
not the immediate priority.

## Preserve the decoder improvement and couple the next full-state dynamics

The [combined decoder fit](temporal-decoder-results.md) completes10,000updates,
exact CPU restoration, all eight development scenes and240-step own-history.
Declared controls pass, including identical inherited assets, shared initial
fluid/request tensors, data, sampling, loss and feedback. Tracer architecture
and its initial function differ; the three representation changes are combined.

Small-grid worst single-block velocity improves from0.3911 to0.1789 and tracer
from0.01481 to0.009518. Pressure/equation worsen to0.2688/0.6178. Own240worst
P/equation/V/T are2.433/18.56/0.4088/0.6958, all finite, instead of the previous
explosive growth. The trained model has zero observed response to the gauge
probe and exactly zero output from zero tracer input. Lead-time readout scalar
alignments move near one. None of these improvements satisfies0.004; no new
CPU latency, replica, promotion or reserved evaluation is triggered.

The next architecture should represent accumulated pressure and couple its
learned transport, pressure request and velocity readout. New predeclared
[offline controls](tied-momentum-results.md) show why: with exact future pressure,
using twice the endpoint pressure gives0.09693worst cold velocity error, while
using the two-pressure sum gives4.96e-5. At four steps those controls are0.3310
versus3.38e-4. Future reference pressure appears only in these controls; they
are not temporal predictions or frontier points.

The negative spatial transpose of the already trained neural flux convolution
also matches the independent native momentum primitive within2.70e-8relative
on144comparisons, without new fitting. This supports sharing learned flux and
momentum coefficients rather than independently fitting an imperfect gradient
readout. A further [offline gain-bound probe](learned-pressure-reuse-note.md)
supports qualifying more reuse of the frozen linear neural pressure map.

Follow the [next coupled full-state plan](temporal-coupled-dynamics-plan.md).
Keep a shared fluid encoding, cheap internal latent updates and every physical
horizon. Do not replace the full-state objective with a pressure component,
reference-pressure oracle or repeated independent full spatial models. Numerical
pressure solves and external corrective projections remain excluded.

## Diagnose the completed coupled dynamics before further fitting

The coupled K2 fit, exact CPU restore, eight-scene evaluation and fixed-window
fitting diagnosis now complete. [The audited result](temporal-coupled-results.md)
retains the width change from 16 to 12 and the different internal arithmetic.
Small-grid single-block worst P/equation/V/T are
0.1041/0.4291/0.005251/0.01013. Velocity improves by approximately 34 times, but
own240 errors remain 3.143/13.53/0.5901/0.8088. Every scene stays finite.
No accuracy-qualified recipe, CPU latency claim, replica or reserved capture
is triggered.

On the fixed training windows, cold-start worst P/equation/V/T are
3.65e-4/5.76e-4/2.91e-4/4.80e-5. Warm training worsts are
0.08195/0.2428/0.002439/0.001758. Warm pressure fitting is still insufficient,
even where velocity and tracer pass. This supports diagnosing the learned
provisional velocity and tracer transport before another capacity sweep.

Use train-only counterfactual controls to separate transport, pressure readout
and tracer errors. Compare the unchanged fit against its saved initialization
and zero local transport/viscosity heads. An explicitly labeled offline native
provisional-velocity control may test the remaining readout limit, after its
reconstruction is independently qualified. Future reference data cannot become
an inference input or a frontier point. If direct provisional-velocity or
tracer-increment supervision is indicated, preregister it and its matched
control separately. Preserve the existing data, full-state outputs, every
horizon, own240 criteria and combined 4/6/8 ms budgets.

## Run the matched provisional-supervision continuation

The [train-only counterfactual controls](temporal-coupled-transport-results.md)
now complete. Replacing pressure alone leaves warm worst velocity error at
0.002496 versus 0.002439 normally. Replacing provisional velocity instead gives
2.825e-5, with worst pressure/equation at 3.526e-4/6.316e-4. These controls use
future reference fields offline and do not satisfy a forecasting objective.
Learned transport improves on zero transport but still has 33% median error in
native transport-increment units. Tracer barely changes under these fluid
controls and remains a separate learning/generalization issue.

Two matched 4,000-update continuations are registered and CPU/CUDA qualified.
Both start from the same completed 10,000-update coupled checkpoint, with fresh
Adam, identical sampling, schedule and two-block feedback. The control retains
the full-state increment-MSE objective. The other adds provisional-velocity
error in the same fixed native-step velocity units, at weight one. Both compute
the training-only teacher labels; no reference field enters prediction.

The labels reconstructed from the actual normalized float32 training targets
pass 128 independent native checks per device: maximum RHS error 5.79e-5 and
velocity round-trip error 9.78e-7. Actual constructors pass 32 repeated-gradient
cases per device. Inference architecture, pressure expert, 2/3/4-state budgets,
every field/horizon and own240 criteria remain unchanged. The finite batch owns
both fits, exact CPU restores and full eight-scene evaluations. No new accuracy
or latency is established until those complete.

[Frozen objective registration](temporal-provisional-registration-01.json) ·
[Finite batch](temporal-provisional-batch-specification-01.json)

## Qualify a compact learned tracer representation if the matched fits still fail

New [offline tracer controls](tracer-representation-results.md) show that a
convex displacement-based readout reproduces all 256 native primitive outputs
exactly at float32 precision when native displacements are supplied. This covers
captured, zero, constant and random fields on all 16 training scenes. The native
primitive also matches all 64 captured next-tracer states bitwise.

An offline first-order displacement has worst single-step tracer error 1.93e-5.
With supplied native velocities and its own tracer over 128 steps, its worst
small-grid error reaches 0.001207; 32³ reaches 7.45e-5. The
[scalar replay](tracer-reference-velocity-replay-results.md) uses reference fluid
history and is not a full-state forecast or a 240-step pass.

This supports a learned displacement decoder with a fixed convex tensor-product
readout instead of 27 unconstrained softmax weights. Its displacement must come
from causal neural features, with zero entering velocity preserving tracer and
zero/constant/donor-range behavior qualified. Native backtracing, reference
displacements and reference fluid history remain excluded from inference.
Qualify actual CPU/CUDA gradients and every field/horizon before a bounded
full-state comparison. Any coefficient calibration must have ordinary managed
training custody. Preserve the current matched-fit outcomes before selecting
the next architecture; no representation control replaces the original goal.

## Preserve the matched negative result and measure architecture cost

Both continuations now finish every update, exact CPU restore and full evaluation.
All 25 paired controls pass. Small-grid worst single-block P/equation/V/T are
0.10418/0.42876/0.005229/0.009963 for the control and
0.10289/0.44682/0.005072/0.010014 with auxiliary supervision. Own240 P/equation/V/T
are 1.339/4.972/0.3938/0.8775 versus 1.674/6.787/0.4428/0.8740.
The added loss does not qualify the model. Preserve
[the complete matched result](temporal-provisional-results.md).

The next implementation should target the learned transport representation,
including displacement-based tracer decoding; more repetitions of the same
loss are lower priority. Any new displacement must be predicted by the network.
Qualify native-label generation separately from inference and retain all P/V/T
outputs, shared encoding, cheap internal updates and original accuracy gates.

A separate registered CPU cost campaign measures the parent, control and
auxiliary checkpoints with five balanced serial repeats, full physical
encoding/forcing/output scopes and separate tensor measurements. It uses the
same 48 small-grid and 16 separate 32³ requests per worker. Accuracy failures
remain failures; the purpose is to quantify the compute available for changes.
The first setup attempt recorded zero repeats and reaped its worker after model
constructor logging disrupted JSON replies. A fresh wrapper directs constructor
messages to stderr. Models, populations, numeric inference and timing scopes
are unchanged; no training or accuracy evaluation is repeated.

## Change the transport representation and reduce CPU overhead

The [completed cost campaign](temporal-provisional-cost-results.md) measures
5.75/5.59/5.59 ms complete two-state blocks for the parent/control/auxiliary
models, or 2.88/2.80/2.79 ms per generated step. This exceeds the 4 ms block
budget. The 32³ costs are much higher and remain a separate extension. Every
model also fails accuracy. No replica or reserved capture is triggered.

The [stage profile](coupled-CPU-stage-profile-results.md) measures about 1.08 ms
in the tracer decoder on selected small-grid blocks, with substantial
input/forcing/output work outside the network. The preliminary denormal screen
offers no speed benefit and is not adopted. Instrumented profile values are
separate from the five-repeat benchmark.

The next bounded full-state comparison should use neural displacement prediction
and fixed convex field readouts, while retaining the frozen learned pressure
component, learned flux/momentum coefficients and every physical horizon.
First qualify native face/cell displacement labels and their training-only
tensor implementation. Any point-sampling coefficient calibration must use
ordinary managed training and exact CPU custody. Native backtrace and reference
fields must remain absent from deployed prediction.

Supervise the learned transport parameters in units that expose the small
transport change, alongside the original full-state objective and own-history
feedback. Keep one shared encoding and cheap internal updates. Fuse compatible
linear feature projections, field reconstruction and prescribed-force work where
qualification supports it. The CPU target leaves little room for simply
increasing convolution width. Register the actual architecture/objective and
qualify its CPU/CUDA gradients and numerical behavior before fitting, then test
all requested K values and retain the original every-field/own240 gates.

The native face/cell label and readout prerequisite now passes 80 cases on each
device, with exact native agreement and repeatable gradients. Preserve
[that offline qualification](displacement-transport-results.md). No learned
displacement predictor or speed is established by those zero-update checks.
Proceed to the registered learned coefficient/full-state comparison rather
than treating the representation check as a forecast result.

Continue to allocate compute across the whole block. An expensive endpoint and
small intermediate decoder, or reusable first-state features and cheap later
updates, are both valid under the combined 4/6/8 ms limits. Both initial
architecture families already passed speed and failed accuracy; their next
versions need a better dynamics representation, not a different interpretation
of the timing target. The separate pressure successor already meets its warm
small-grid benchmark.

The learned point prerequisite is now complete: 1,000 managed CUDA updates,
exact CPU restore, and 24 native float32 checks at 2.51e-8 worst error.
The actual displacement model/trainer is implemented and undergoing CPU/CUDA
constructor, gradient and staging qualification for all four K values.
Complete these checks before capturing and admitting the bounded full-state
fit. [Component custody](point-sampling-results.md) and the
[implemented model plan](temporal-displacement-plan.md) retain the exact scope.

Both actual CPU/CUDA qualifications now pass all 64 cases per device. The
bounded K2 full-state fit is registered, captured and admitted as managed Run
`52851a02f7cf4162b983a5be029f71a0`. Follow the same Run and finite batch through
training, exact CPU restore and every-output/own240 evaluation; do not resubmit
on an observation timeout. Assess those results before selecting architecture
extensions, replications or CPU execution changes. No new accuracy or speed
claim follows from constructor checks.

The fixed fitting diagnosis, source/custody audit and actual CPU comparison
are registered and queued behind the existing batch. Observe these bounded
controllers and their specific Run rather than restarting on a polling delay.
The CPU comparison uses restored displacement/coupled/auxiliary weights and
five balanced serial repeats; it submits no additional fits. Assess complete
every-horizon accuracy and own240 alongside those actual full-block costs
before selecting extensions or CPU execution changes.

The completed K2 training loop takes about 1,014 seconds plus 78 seconds of
setup, for 18,000 block forwards and 36,000 supervised physical states. This
is training cost, not CPU inference. The managed CUDA recipe wall limit is
1,800 seconds, so blindly reusing 10,000 updates for K4 risks a resource timeout.
Before admitting further horizons, use this measured cost to register finite
work that fits the limit. Horizon-dependent update counts can match physical
target work: K3 at 6,667 updates with 1,334 single-block updates and K4 at
5,000 updates with 1,000 single-block updates each produce 36,000 physical
state targets under the two-block curriculum. They differ in sampled windows
and optimizer calls and are not a FLOP-matched comparison. Alternatively use
explicit managed continuations. Do not alter the current fit or weaken any
output, accuracy, timing or own240 criterion to meet a training wall limit.

[Measured K2 training work](temporal-displacement-training-cost-01.json)

## Calibrate neural viscosity and fuse the fixed readout

The displacement K2 comparison now completes fitting, exact CPU restore,
all eight development scenes, own240, source/control audits and balanced CPU
cost measurement. Worst small-grid single-block equation/V/T improve to
0.01642/0.00014197/0.00001950; own240 equation/V/T remain
0.01992/0.007352/0.005568. Complete small-grid blocks take 6.350 ms, or
3.175 ms per generated step. Both combined gates fail. No replica or reserved
capture is triggered by this result.

The next bounded accuracy action is now specific. Train-only controls show
that replacing learned displacement alone barely changes fluid error.
Replacing the viscosity stage with the native primitive reduces warm worst
equation error from 0.01098 to 0.000659 and velocity from 0.0001390 to
0.00002825. On the cpm16 odd-grid 128-step training replay, equation error
drops from 0.01166 to 0.000566 and velocity from 0.003994 to 0.00003465.
These are offline controls, not a hybrid candidate or a 240-step pass.

Calibrate the neural viscosity coefficients against independently qualified
native labels using an ordinary managed fit and exact CPU restore. The public
native viscosity primitive and dyadic exact-label fixtures pass 56 cases.
Keep numerical viscosity out of candidate inference. Bind any composed
checkpoint to both managed sources, retain the existing trained displacement
weights, then evaluate every field/horizon and full own240 again. Do not
replace the reference dynamics or weaken its error gate.

The independent stage profile measures approximately 1.59 ms per small-grid
block in the fixed field readout, versus 0.75 ms in displacement projection and
0.94 ms in physical preparation/reconstruction. These inclusive profile times
are not additive or a replacement for the five-repeat benchmark. Qualify a
fused implementation of the fixed readout with unchanged neural weights,
including bounds, zero/constant behavior, field errors and rollout parity,
before measuring its actual complete-block CPU speed. Further forcing and
projection execution work should follow measured remaining costs.

[Full-state results](temporal-displacement-results.md) ·
[Measured CPU costs](temporal-displacement-cost-results.md) ·
[Native controls and profile](displacement-component-results.md)


## Neural viscosity accuracy pass and measured fused cost

The managed viscosity calibration and exact CPU restore are complete. Only
243 coefficients change; all other full-state tensors remain byte-identical.
The calibration uses manufactured native labels, with no native primitive in
candidate inference. This zero-update assembly is bound to its two managed
parents and is not represented as a new full-model training Run.

The repaired K2 model passes every field and horizon on all eight development
scenes, including all 240 own-history steps. Worst combined error is 0.0033941.
The fused fixed readout preserves all 3,856 physical outputs byte for byte.
Five balanced serial repeats measure 4.7793 ms per complete small-grid block,
or 2.3896 ms per generated step. Thus accuracy passes but the 4 ms block budget
still fails; tensor-only 3.7526 ms does not satisfy the complete timing scope.
32³ costs 183.0 ms per block and is reported separately.

The next execution comparison targets forcing construction and unused pressure
input conversions while preserving all physical outputs and learned weights.
Qualify it against the complete development/own240 population before timing.
Then assess remaining encoder/displacement computation reuse. K3/K4 in the
repaired family, replicated controls, reserved240 and application/event timing
remain open. Eight reserved temporal scenes are still untouched.

[Managed calibration](viscosity-calibration-results.md) ·
[All-field accuracy](neural-viscosity-results.md) ·
[Exact fused execution and CPU costs](neural-viscosity-cost-results.md)


## Corrected-family training and bounded float32 execution comparison

All64actual constructor/gradient cases now pass on CPU and CUDA for K1/2/3/4.
The four Projects and finite fit recipes are captured with viscosity frozen
from initialization. The first managed K2 Run is5183fc284792458a902ec6c8ce123d4c;
K3,K4and the matched-physical-target K1control follow serially. Fit completion,
exact CPU restore and every-output/own240 results remain the authoritative gates.

The dependent postprocessing campaign is registered and waiting for that same
controller. It audits all four fits and actual36000-state work, qualifies every
physical byte for fast execution, and runs55balanced CPU timing responses
(3520blocks), including sequential2/3/4-call B1controls. It submits no new fits.

A separate zero-update float32 learned-map execution now passes every field,
horizon and own240 criterion on all eight development scenes. Worst small-grid
combined error is0.00339527, against0.00339409for the float64 parent. This changes
rounding and uses a new full evaluation; it does not inherit exact output bytes.
No inverse, numerical pressure solve or parameter learning is introduced.

Its1280-block CPU comparison is registered and waiting for the entire original
family/CPU campaign to finish and reap. Do not start seed-replication numerical
work while either CPU comparison is active. Select recipes after both actual
cost reports are available, then qualify replications and reserve final weights
and execution before capturing the eight untouched reserved scenes.

[Corrected-family work and gates](fixed-viscosity-family-plan.md) ·
[Actual float32-map accuracy](float32-learned-map-results.md) ·
[Registered family postprocessing](fixed-viscosity-postprocessing-registration-01.json) ·
[Registered float32 costs](float32-map-cost-registration-01.json)


A publication-order defect was identified before the float32 CPU campaign runs:
the original reporter expects its parent controller to be complete while running
inside that controller. A separately registered dependent publisher now waits
for the original owner to reap, checks that both numerical commands succeeded,
and accepts only that exact retained read-only publication exception. It then
publishes the existing timing records without repeating training or timing.
The original registered sources and command list remain preserved.
See [the bounded publication repair](float32-map-publication-repair-registration-01.json).


## Corrected K2 managed fit and complete accuracy pass

K2 Run `5183fc284792458a902ec6c8ce123d4c` completes all10,000updates. CPU restore
`35c5d3f6de734f268644ae39b8373b33` makes zero updates and reproduces all five published
artifacts byte for byte. The actual training work is18,000block forwards and
36,000physical state targets; setup takes74.53seconds and the fitting loop
1,021.48seconds. The managed viscosity coefficients remain frozen from the start.

The complete development evaluation passes every equation/pressure/velocity/tracer
criterion, every intermediate output and all240own-history steps on all eight
scenes. Worst combined error is0.0033966464. New-weight CPU latency is not yet
measured; the previous composition's timing is not inherited as a new result.
K3,K4and the one-step control continue through the existing finite controller.
The registered comparison and float32 timing/publication dependencies remain
owned; do not duplicate fits or overlap new numerical work with CPU campaigns.

[Full K2 raw summary](../../../runs/full-ai-next-priorities-v1/fixed-viscosity-evaluation-01/b2-fixed-viscosity/summary.json) ·
[Exact managed fit and CPU restore](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-pair-01/receipt.json)


## K3 accuracy pass and completed application preparation

The independently trained corrected K3model completes its managed fit, exact
CPU restore and all eight development scenes. Every output/field and own240
criterion passes; worst combined error is0.00353177. K4is training in the same
existing sequence, with the one-step control following. New-weight CPU costs
remain unmeasured until the registered serial comparison.

Bounded native transport/restart and12neural/native mechanics cases now pass.
The latter use K2checkpoint diagnostic horizon views and do not substitute for
independently trained horizon results. All eight fixed48-step development event
references are captured; states/RHS through13remain bitwise unchanged and the
forcing change affects velocity14. No reserved scene is opened. The forecast
application loop takes only causal current state and active forcing, excludes
offline reference metrics from timing and forbids native numerical advance.

[Application preparation evidence](application-preparation-results.md) ·
[Remaining application measurements](application-evaluation-plan.md)


## All four primary accuracies pass; application/control interfaces qualify

The corrected K1fit and exact CPU restore complete all20,000updates and the
matched36,000physical targets. Its complete eight-scene, every-output/own240
worst error is0.00372123. The K2/K3/K4models also pass, at0.00339665,
0.00353177and0.00336772respectively. The full four-fit work/source/custody audit
passes. New-weight CPU speed remains pending the registered serial comparison.
Fast execution already matches2,888K1,3,856K2and4,824K3physical outputs byte
for byte; K4is the final exact-execution qualification before timing.

Actual K2/K3/K4deployment/evaluator interfaces pass12limited48-step cases
across float64/float32-map policies and steady/event scenarios. Six native
controls qualify with unchanged gates: PCG/spectral complete, and C107rejects
its first step without fallback. These checks do not establish whole-bank
precision accuracy or measured application speed. The application controller
and report implementation require concrete frozen model/policy evidence before
use. No new registry or model selection is inferred by the executable.

[Complete four-fit accuracy and controls](fixed-viscosity-family-results.md) ·
[Application interface qualifications](application-interface-results.md)


## All four exact execution proofs pass; isolated CPU comparison active

Fast execution now preserves all17,360physical outputs byte for byte across
all four actual trained models:2,888forK1,3,856forK2,4,824forK3and5,792forK4.
This includes every reference-history horizon and all own240steps on all eight
development scenes. All neural tensors remain unchanged. The existing finite
controller has entered its five-balanced-repeat CPU campaign, including the
sequential2/3/4-call one-step controls. Do not overlap new numerical work with
this campaign or the queued float32 comparison. No new-weight timing result
is claimed until the complete responses and worker reaping are verified.

Application generation, native numerical/C107 controls, deferred metrics,
qualified-model admission, the seven-method serial controller and its reporter
are implemented. Their interface qualifications have passed; the full application
campaign still requires concrete frozen checkpoint/precision choices after
seed confirmation. Generic float32 deployment admission requires its own full
actual eight-scene evaluation and cannot inherit the old composition probe or
the limited48-step interface results.


## Latest combined-budget results and next comparison

The2 ms target applies to total generation time divided by K; no individual
internal stage has a2 ms limit. The corrected family retains learned spatial
features after a shared encoder, with internal updates and every intermediate
physical readout included in these measured costs.

| Generated states | Complete small-grid median ms | Amortized ms/state | Worst own240 error | Combined median target |
| --- | ---: | ---: | ---: | --- |
| 2 | 4.1240 | 2.0620 | 0.00339665 | Misses speed |
| 3 | 5.5742 | 1.8581 | 0.00353177 | Passes |
| 4 | 7.4335 | 1.8584 | 0.00336772 | Passes |

All eight development scenes pass every original equation/pressure/velocity/
tracer criterion, every output and own-history intervals through240steps.
Latency is the warm median over three small geometries; per-grid/tails,32³,
setup/RSS, reserved generalization and sustained application costs are separate.
The C107 successor independently passes its prepared-pressure benchmark at
2.133 ms and4.6963e-4 worst residual; stricter native acceptance remains unmet.

The earlier-composition paired precision comparison completes: float64 measures
4.1750 ms/block and float32 learned-map arithmetic3.7069 ms/block, with actual
float32 own240 worst error0.00339527. Every numerical worker reaps; the retained
read-only publication-order exception is recovered without replaying measurements.

Next qualify float32 arithmetic on the actual corrected K2weights across all
eight scenes/own240 and time a new five-repeat paired comparison. That evaluation
is running under its own frozen protocol; old-composition or short interface
results cannot establish the new model's accuracy. Then select up to two
bundled recipes for two additional seeds, including a replicated one-step
control. Preserve the original primary seed and freeze weights/arithmetic
before reserved capture and application/event timing. Endpoint-plus-small-decoder
remains a separate architecture follow-up if a measured gap persists; its early
speed pass and accuracy failure remain reported.

[Actual family costs](fixed-viscosity-cost-results.md) ·
[Earlier-composition precision costs](float32-map-cost-results.md) ·
[New corrected K2 precision protocol](fixed-viscosity-k2-float32-evaluation-protocol-01.json)


## Corrected K2 precision pass; concrete seed confirmation begins

The actual corrected K2float32-map policy passes every original criterion on
all eight scenes, every intermediate state and own240, with worst error
0.0034031815. Its five-repeat complete small-grid median is3.5445 ms/block,
or1.7723 ms/generated step. The same-campaign float64 baseline is4.0838 ms.
K3/K4retain their qualified float64 development passes at5.5742/7.4335 ms/block.
Thus all requested K now pass development accuracy and the combined median
budget. Per-grid/tail/32³, setup/RSS, reserved evidence and application cost
remain separate; this is not a final whole-plan success or model promotion.

The first corrected-K2publisher failed before execution because its control
interpreter lacks NumPy. The registered read-only repair runs the unchanged
reporter under the qualified numerical runtime, validates the original records
and repeats no training or timing. The original timing and observer failures
remain retained. A fresh admission controller proceeds only after publication.

The concrete selection isK2float32-map andK4float64, plusB1float64 as matched
control. Add initialization seeds2026102046/2026102047 and sampling seeds
2026101923/2026101924. Each of the six managed fits keeps shared frozen learned
components, data, loss and36,000physical-target work. These are independent
cell initializations with shared component pretraining. Qualify actual CPU/CUDA
constructors and two-block gradients before fitting. The ordinary managed CUDA
fits, typed exact CPU restores and every-output/own240 evaluations are one
finite serial command list; no failed/unknown Run is automatically resubmitted.
The original primary seed is retained. K3remains a measured primary even though
its seed coverage is smaller. Reserve final weights/arithmetic before capturing
the eight untouched reserved temporal scenes and measuring application events.

[Actual K2 precision result](fixed-viscosity-k2-precision-results.md) ·
[Concrete seed selection](fixed-viscosity-replication-selection-01.json) ·
[Publication repair](fixed-viscosity-k2-precision-publication-repair-01.json) ·
[Finite admission continuation](fixed-viscosity-replication-admission-batch-specification-02.json)


## Added-seed fits admitted; dependent confirmation owners registered

All96actual CPU and96actual CUDA constructor/gradient cases pass with zero
updates. The finite six-fit controller now owns the added temporal-cell seeds
forK1/K2/K4. The first managed training Run is
`ae4f4d36db0b4b40ad31698109220822` (K1seed2,20,000updates). Its live progress
is provisional; only drained artifacts and the exact typed CPU restore establish
completion. Original component pretraining remains shared and frozen.

A dependent postprocessing controller waits for this same existing owner,
without resubmission. It will bind nine actual primary/replica deployments,
prove float64 physical-byte equivalence or recompute float32 accuracy for each
new seed, then measure five balanced serial CPU repeats and matched sequential
B1controls. All seeds, including unfavorable outcomes, remain in the results.
K2float32 versusB1float64 compares complete deployment choices, including both
precision and computation sharing; it does not isolate the encoder contribution.
K4andB1both use float64.

A separately registered final confirmation owner waits for all of those workers
to finish and reap. It will freeze all four primary horizons and six added
seeds before opening any reserved data, capture all eight originally reserved
240-step scenes, evaluate every output/own240 for all ten checkpoints, measure
fresh reserved CPU costs, and finally run560steady/interaction application
measurements over four actual primary horizons and three native controls.
Native numerical gates remain unchanged, model-only budgets remain4/6/8ms,
and application acceptance limits remain diagnostic. No native numerical advance
or corrective projection enters a forecast candidate. Captures, evaluations and
CPU timing remain serial. Final target attainment is unproven until those
actual results and the original29requirements are audited.

The new orchestration modules parse and import under the qualified numerical
runtime. Original plan/scope, qualified inference code and managed training
source closures remain unchanged. This prepares and owns the remaining work;
it does not claim seed confirmation, reserved success or application speed.

[Replica postprocessing](fixed-viscosity-replication-postprocessing-registration-01.json) ·
[Final confirmation registration](temporal-final-confirmation-registration-01.json) ·
[Finite final sequence](temporal-final-confirmation-batch-specification-01.json)


## First additional seed: exact restore and all-eight-scene accuracy pass

K1seed2managed CUDA Run `ae4f4d36db0b4b40ad31698109220822` completes/drains
with20,000updates and36,000physical targets. All six published artifact hashes
are verified. Independent CPU restore `25be3d3d19f54d1b901d0e8ca28a054c`
completes with all five declared artifact comparisons byte-identical.

All eight development scenes pass every original equation/pressure/velocity/
tracer criterion and every intermediate/own240output. An additional raw-record
check independently recomputes all criterion decisions, summary extrema and
output coverage for2,888physical outputs. Worst combined error is0.0033227321.
This is control-seed accuracy, not a selected-policy CPU or reserved result.
See the [raw audit](../../../runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s2-raw-accuracy-audit-01.json).

The same finite controller moves toK2seed2training Run
`38ac29a9de52464393a02213bd6fdd75`. No Run is restarted. The other five pairs,
selected execution/precision proofs, CPU confirmation, reserved scenes and
application measurements remain owned by the existing dependent controllers.
Original sources, plan, thresholds and combined budgets remain unchanged.


## Completion audit and guarded final publication

A separate current-state completion audit now checks all29original requirements.
It verifies source/weight bindings, actual managed artifact parity, the original
reserved scene split, raw per-output criteria and drift crossings, fresh CPU
budgets and every application episode's causal/process/discard accounting.
Missing results stay pending; completing an experiment cannot substitute for
passing the target. No goal status is updated by the auditor.

The report publisher now requires a current passing scientific audit for
`--final`; its guard is verified to reject the current incomplete state before
writing any report. Normal publication remains interim. The first audit has
14passed scopes and15pending, with no failed completed gates. Reserved and
application evidence is still absent because the existing controllers are
waiting for the finite managed fit sequence. No extra numerical job or fit is
started by this audit. The currentK2seed2Run remains the existing
`38ac29a9de52464393a02213bd6fdd75`.

[Current-state completion audit](completion-audit.md) ·
[Auditor source](../../../experiments/full-ai-next-priorities-v1/audit_completion.py)


## Second completed pair: K2seed2 passes every development criterion

K2seed2CUDA Run `38ac29a9de52464393a02213bd6fdd75` completes/drains all10,000
updates,18,000block forwards and36,000physical targets. All six published
artifact hashes are verified. CPU restore `d2b17d21dead470cb8264d9da98cae42`
passes all five byte comparisons and its fresh CPU probe, with zero updates.

All eight development scenes, every intermediate output and own240criteria
pass. Worst combined equation/field error is0.0035599317. The independent
raw-record check verifies3,856physical outputs, each original near-zero/error
criterion and every summary count/extremum. See the
[raw K2audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s2-raw-accuracy-audit-01.json).
This is the original restored float64 prediction policy; the selected float32
execution still requires its separately queued complete evaluation and timing.

Two of the six added-seed pairs now have exact restores and complete accuracy
passes. The same finite owner moves toK4seed2Run
`276729eada3b43caa4a376208b1de1c8`, without resubmission. Four pairs remain.
No reserved scene is opened and no selected-policy speed or whole-goal success
is inferred from these new accuracy results. Original sources, thresholds,
training budgets and pre-reserved primary choice remain unchanged.


## Supported boundary change: fresh prediction beyond the stop guard

The originalB4scope requires unannounced boundary changes to invalidate unused
outputs and trigger a new prediction. The existing unsupported-boundary guard
stops correctly, but that alone does not demonstrate supported continuation.
A separate registered functional comparison now covers a physical closed-box
rescale at fixed logical cell counts, switching8↔16cells per metre after step13.
This changes physical boundary positions and spatial sampling density; fluid
mass density, logical domain resolution and the closed boundary type stay fixed.
Concentration/velocity arrays are copied without resampling. The externally
imposed volume/mass mapping is explicit in the offline physical diagnostics.

Mechanics smoke tests pass allK1/2/3/4with a stub predictor: every stale state
is discarded, the next model call uses the new geometry at13, all48states are
consumed and all native children reap. A separate native-reference smoke test
checks actual pressure-request spacing metadata in both8→16and16→8directions,
with unchanged numerical gates and no numerical step during restore. Neither
smoke establishes trained-model accuracy or CPU speed.

`application-domain-change-batch-01` now waits for the existing final application
owner to complete/reap. It then captures eight consumed-development reference
episodes and tests all four actual frozen primary models:32episodes and1,536
physical output diagnostics. Native pressure/RHS and errors use the geometry
active at each timestep; every call's spacing and every stale-state discard are
checked. Actual fields and geometry, rather than a future reference endpoint,
enter each fresh prediction. No new fit or numerical candidate correction is
introduced. Accuracy remains an explicit diagnostic, with no arbitrary-boundary
or new CPU-performance claim. Unsupported boundary types retain the separate
stop guard. The completion auditor now requires this actual-model continuation
in addition to the original560steady/forcing application episodes.

The original training, timing and final-confirmation registrations remain
unchanged. No reserved scene is opened by this preparation. K4seed2continues in
the same owned managed Run; there is no resubmission.

[Supported-boundary registration](application-domain-change-registration-01.json) ·
[Finite dependent comparison](application-domain-change-batch-specification-01.json) ·
[Current completion audit](completion-audit.md)


## All second-seed fits pass; third-seed control begins

K4seed2CUDA Run `276729eada3b43caa4a376208b1de1c8` completes/drains5,000updates,
9,000block forwards and36,000physical targets. All six published artifact
hashes verify. CPU restore `184fc0d32eef4a689c4562af4106c159` passes all five
actual byte comparisons and the independent CPU probe with zero updates.
All eight development scenes, intermediate outputs and own240criteria pass.
The raw audit independently verifies5,792physical outputs and every criterion,
count and summary extremum: worst combined error0.0033281110.

All three ordinal2models now have exact managed restores and full accuracy
passes: B1worst0.0033227321, K2worst0.0035599317andK4worst0.0033281110.
These are actual original-policy accuracies, with new-weight execution/precision
qualifications and CPU measurements still separately queued. No seed is selected
as a replacement primary. See the
[raw K4audit](../../../runs/full-ai-next-priorities-v1/b4-fixed-viscosity-s2-raw-accuracy-audit-01.json).

The same finite controller advances toB1seed3Run
`f2a41e2042624421bfbde13f02a1de0c`. Three of six added-seed pairs remain.
The supported physical-box boundary comparison is registered and its dependent
owner is live; both native spacing directions and all four discard/regeneration
mechanics smoke cases pass. Its actual32trained-model episodes still wait for
the existing complete application campaign. The original source registrations,
thresholds, primary and reserved scenes remain unchanged. The completion audit
also distinguishes an interim report awaiting live-owner progress from altered
completed scientific evidence; immutable result checks remain strict.


## Third-seed control completes; late equation criterion fails and is retained

B1seed3CUDA Run `f2a41e2042624421bfbde13f02a1de0c` completes/drains20,000updates
and36,000physical targets. All six training artifact hashes verify. Independent
CPU restore `e0552a87be8042c5a6e40c39fc6e7950` completes/drains with all five
actual artifact byte comparisons, typed origin/closure and CPU probe verified.

The complete eight-scene evaluation remains finite through240steps, but the
original accuracy criterion fails: worst equation error0.0043489455. Six raw
own-history records fail on `temporal-g2-cpm16-r2`, a15×6×8grid at16cells/metre,
at steps207–212. The other seven scenes, all single blocks and own12/48intervals
pass; pressure/velocity/tracer field criteria pass throughout. Global field
maxima are pressure0.00093235, velocity0.00039357andtracer0.00135467.
The independent raw audit verifies all2,888outputs, counts/extrema and both
passing and failing decisions against the unchanged limits. See the
[retained negative raw audit](../../../runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-raw-accuracy-audit-01.json).

Four of six added-seed pairs now have exact restores and complete evaluations:
three pass accuracy and this control seed fails. The original primary is not
replaced by a better seed. This is variation of the temporal-cell initialization
and sampled windows with common frozen component pretraining; it does not
establish a failure of the still-qualified original bundled primaries. Both
unfavorable control accuracy and its future measured cost remain in the reports.

The same finite owner startsK2seed3Run
`bff7ba1f5e484b569a39c1deee04ad5a`; K4seed3follows. Their exact restores,
all-field accuracy, selected execution/precision proofs, repeated CPU costs,
reserved scenes and application/boundary episodes remain pending. No new fit,
threshold change, retry or primary reselection is introduced in response to
this negative result. The whole-plan goal remains active.

## Conditional follow-on: tighter targets, higher density, then long drift

After the current targets are reached and fully verified, the user authorizes
the ordered [follow-on program](deferred-target-program-01.json):

1. All complete-state K=1,2,3,4 generators: maximum 1.2 ms per generated step and
   worst error at most 1e-5. Block budgets are 1.2/2.4/3.6/4.8 ms. Proposed timing
   uses the observed maximum over the registered population, with median/p95
   also reported. Every output and autonomous history retain all-field checks.
2. After those passes, add 50 and 100 cells per axis per metre with random
   logical dimensions below, above and mixed relative to those density counts.
   Qualify references and resource cost before fixing captures. Train/test all
   four K values at 1e-5; larger-grid per-step budgets grow at most as voxel count,
   1.2 ms times V/V_reference. Seek better measured scaling than this ceiling.
3. After the larger-size objectives pass, minimize drift over 600 actual steps
   at 60 Hz and 30 Hz, interpreted as 10 and 20 physical seconds. Qualify both
   timestep contracts and report best measured drift without a global claim.

The reference-grid clarification is pending: the user gave 32 cubed as an
example; current primary timing uses smaller grids. Freeze the choice before
new fits/scaling claims; do not imply an achieved 32-cubed 1.2 ms baseline.
Original plan and registrations remain unchanged. No follow-on fit or capture
starts before current confirmation succeeds.

## Fifth additional seed passes original-policy accuracy

K2seed3 managed CUDA Run `bff7ba1f5e484b569a39c1deee04ad5a` and independent
CPU restore `f9e4a99c97784ce9a107bbd42f09da9f` complete and drain. All published
artifact hashes, five actual byte comparisons, typed origin/closure and fresh
CPU probe verify. The fit performs 10,000 updates, 18,000 block forwards and
36,000 supervised physical states.

All eight development scenes pass every output and own240 criterion. The
[independent raw audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s3-raw-accuracy-audit-01.json)
verifies 3,856 physical outputs, unchanged near-zero criteria, interval extrema
and diagnostic drift crossings. Worst equation error is 0.0039455145, with
pressure 0.0008139454, velocity 0.0003963127 and tracer 0.0011256434. This is
close to the original 0.004 equation limit. It is original-policy accuracy;
the selected float32-map evaluation and actual CPU costs remain pending.

Five of six additional pairs now complete: four accuracy passes and one retained
B1seed3 failure. The same finite owner starts final K4seed3 Run
`bee6f205d1c247e5b01fc833fd1981d3`. No primary reselection or target change.

Final publication is separately registered and owned behind all existing
confirmations. It admits no numerical work, retains actual failed outcomes, and
requires all original scientific gates before writing final status. A second
audit checks every requirement after the final render; independent agent review
still controls the thread goal. See the
[publication registration](final-publication-registration-01.json).

## All six added-seed fits and accuracy evaluations complete

Final K4seed3 CUDA Run `bee6f205d1c247e5b01fc833fd1981d3` completes/drains
5,000 updates, 9,000 block forwards and 36,000 physical targets. Independent
CPU restore `d36eeed79ee54dbfb22db793aa6a0f54` completes/drains with all six
terminal artifact hashes, five actual byte comparisons, typed origin/closure
and independent CPU probe verified.

All eight development scenes and every original criterion pass for this seed.
The [raw audit](../../../runs/full-ai-next-priorities-v1/b4-fixed-viscosity-s3-raw-accuracy-audit-01.json)
checks 5,792 physical outputs, all intermediate horizons, own240, interval
extrema, near-zero decisions and drift crossings. Worst errors are equation
0.0030208889, pressure 0.0007165861, velocity 0.0003365973 and tracer
0.0018260075. This does not establish replica CPU speed or reserved accuracy.

The complete [six-seed report](fixed-viscosity-replication-results.json) now
verifies every managed fit, control and physical-target count. All 140 evidence
bindings verify independently. Five seeds pass accuracy; B1seed3 retains its
six late equation failures. Both original-policy K2 seeds pass, with the
selected float32-map accuracy qualification still separate. Neither thresholds
nor the original primary seeds change.

The training and admission owners complete and reap. Their existing dependent
postprocessor starts actual new-checkpoint exact execution/F32-policy checks.
Its nine-model CPU campaign, original reserved population, seven-method
application episodes and supported-boundary supplement remain queued in order.
The completion audit advances to **19 passed, 10 pending, zero failed**.
The current goal remains active; the user's tighter targets stay deferred.

## First execution proof and actual replica float32 policy pass

The existing serial postprocessor completes its first two of six checks:

- B1seed2 fast-float64: all 2,888 physical outputs are byte-identical to its
  original execution across all eight scenes, every horizon and own240. The
  proof's exact weights, source evidence and teacher isolation verify. This
  inherits that checkpoint's accuracy, with no new latency claim.
- K2seed2 fast-float32-map: actual new-policy evaluation passes all eight scenes,
  every output and own240. The independent
  [raw audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s2-F32-raw-accuracy-audit-01.json)
  verifies 3,856 outputs, all unchanged criteria, interval extrema and drift
  crossings. Worst equation error is 0.0035592525; pressure 0.0013684782,
  velocity 0.0004135846 and tracer 0.0013482553. The recorded policy and weights
  match this seed's actual managed CPU restore.

The directory-explicit raw auditor is a separate version; the original auditor
and its completed source identities remain unchanged. Four additional checks
remain, starting with K4seed2 exact execution. The nine-model CPU measurements
still wait behind all qualifications. No new fit, primary selection, reserved
capture or speed claim is introduced by these results.

K4seed2 exact execution now passes all 5,792 physical outputs across all eight
scenes, every intermediate horizon and own240. The complete proof, concrete
checkpoint, source/protocol hashes and zero-update teacher isolation verify.
Three of six execution/policy checks are complete. The same owner proceeds to
B1seed3, retaining its original late equation failure; byte equivalence cannot
turn a failed original accuracy record into a pass. Actual repeated CPU costs,
the original reserved scenes and application studies remain pending.

## Final two-step float32 seed passes; one execution proof remains

B1seed3 fast-float64 matches all 2,888 physical output bytes on eight scenes.
The exact proof and its source bindings verify; the original six late equation
failures at steps207–212 remain negative. This is execution equivalence, not an
accuracy improvement.

K2seed3's actual fast-float32-map evaluation passes all eight scenes and every
intermediate output/own240 criterion. Its independent
[raw audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s3-F32-raw-accuracy-audit-01.json)
checks all 3,856 physical outputs, original near-zero decisions, interval extrema
and drift crossings. Worst equation error is 0.0039463236, pressure
0.0008143239, velocity 0.0003962547 and tracer 0.0011256184. The exact managed
weights, selected policy and every audit source binding verify. The equation
margin below 0.004 is small; neither the limit nor the primary changes.

Five of six actual new-checkpoint execution/policy checks are complete. The
existing finite owner starts final K4seed3 exact execution. All selected K2
seeds now have their own actual-policy accuracy evidence; actual replica CPU
latency, original reserved scenes and application outcomes remain unmeasured.

## All six new-checkpoint qualifications pass; CPU campaign admitted

Final K4seed3 fast-float64 matches all 5,792 physical outputs on all eight
scenes, every horizon and own240. Its concrete managed checkpoint and all
source bindings verify. The six-command execution owner completes and reaps:
four byte-exact F64 proofs (B1/K4, both additional seeds) and two actual F32
K2 evaluations. The unfavorable B1seed3 accuracy remains retained.

The existing postprocessor registers the nine actual primary/additional-seed
deployments with their own policy evidence and starts the five-repeat serial
CPU campaign: 24 full/tensor/sequential variants, 120 responses and 7,680 timed
blocks. CPU setup and numerical work are now owned by that campaign. No new
numerical audit, plotting job or fit is started alongside it. These are planned
measurement counts, not completed timing results.

An independent standard-library timing auditor is prepared to recompute window
coverage, actual CPU context, raw response medians/p95/maxima, repeat summaries
and generated-step divisors after workers finish/reap. It is not yet executed
and makes no timing or accuracy claim. The original reserved/application and
boundary owners remain queued behind this campaign; tighter user targets stay
deferred until complete current verification.

## Three-seed CPU confirmation passes; original reserved data captured

The nine-model campaign completes all 120 responses / 7,680 timed blocks and
all nine numerical workers exit successfully and reap. The independent
[raw timing audit](../../../runs/full-ai-next-priorities-v1/fixed-viscosity-replication-timing-raw-audit-01.json)
recomputes all 48 campaign summary rows from actual samples: physical window
coverage, one-thread/core23 context, repeat medians, p95/maxima and generated-step
divisors all agree. All 154 cost-report evidence bindings verify.

Every selected K2/K4 seed passes both development accuracy and the original warm
small-grid median budget. Results are complete-block latency divided by K:

| Horizon | Original primary block / state ms | Three-seed ms/state range | Worst error across three seeds |
| --- | ---: | ---: | ---: |
| K2 F32 | 3.666263 / 1.833132 | 1.789396–1.833132 | 0.0039463236 |
| K4 F64 | 6.898085 / 1.724521 | 1.724521–1.746983 | 0.0033677226 |

These are mixed-small-grid warm medians, not maximum latency, 32-cubed speed or
application throughput. Per-grid/tails, setup/RSS and all sequential controls
remain in the [full cost report](fixed-viscosity-replication-cost-results.json).
K2 F32 versus B1 F64 is a compound deployment comparison; the measurements do
not isolate encoder reuse alone. The B1seed3 accuracy failure stays negative.

The postprocessor completes/reaps. The existing final-confirmation owner freezes
all ten models (four original primaries plus all six added seeds), without
primary reselection, before capturing the original eight reserved scenes. Native
capture now completes all 1,928 state archives (241 per scene), with unchanged
reservation/selection bindings and every native child successful/gracefully
reaped. All receipt identities verify. The ten-model every-output/own240 held-out
evaluation is active. No new fit or future-target data is introduced. Completion
remains 19 passed / 10 pending / zero failed until that confirmation and the
remaining reserved timing/application/boundary studies finish.

## Reporting layout repaired without changing scientific values

The expanded temporal legends overlapped and clipped identifiers in the interim
plots. A separate `build_report_v2.py` uses one shared two-column legend and
reserves sufficient figure space. Visual inspection confirms readable complete
labels. Every top-level AST node outside `plots` is identical to the frozen
original renderer; all 336 scientific points and the full accuracy arrays are
unchanged. Original reporting and scientific sources remain exact.

The [visual qualification](../../../runs/full-ai-next-priorities-v1/report-layout-repair-01/visual-qualification.json)
records those checks. A separately registered
[presentation finalization owner](report-layout-finalization-registration-02.json)
waits for the original final publication to become terminal/reaped, preserves
its exact report/figure snapshot, and renders the corrected layout afterward.
Final status still requires the complete original scientific audit; unfavorable
studies remain interim. Neither owner updates the thread goal. Completion review
must wait for this second publication owner and inspect its final figures.

## First reserved primaries: K2 and K4 pass; B1 control fails late equation checks

The first three frozen models complete every output and own240 on all eight
original reserved scenes. Independent raw audits verify 12,536 physical outputs
in total, unchanged criteria, field/near-zero decisions, interval extrema and
diagnostic drift crossings:

| Frozen primary | Physical outputs | Worst P / equation / velocity / tracer | All accuracy criteria |
| --- | ---: | --- | --- |
| B1 F64 | 2,888 | 0.00218817 / 0.00410232 / 0.00072188 / 0.00124099 | False |
| K2 F32 | 3,856 | 0.00144138 / 0.00318816 / 0.00049540 / 0.00094943 | True |
| K4 F64 | 5,792 | 0.00153230 / 0.00281931 / 0.00053308 / 0.00100045 | True |

B1's eight failures are equation-only own-history steps221–228 on
`temporal-g2-cpm16-r3` (15×6×8,16cells/metre). All its pressure, velocity and
tracer criteria pass, and every trajectory remains finite/full length. The
unfavorable control is retained; it is not replaced with a better seed.
K2/K4 original primaries pass every intermediate field and full own240 research
criterion. K3 and the six additional-seed reserved evaluations remain pending.
Fresh reserved CPU timing, full application and boundary episodes are still
required; development speed is not substituted for those measurements.

## All second-seed reserved accuracies pass; six of ten models complete

The three ordinal2 checkpoints finish all eight original reserved scenes and
full own240 with every criterion passing. Independent raw audits verify all
12,536 additional physical outputs, unchanged near-zero decisions, all interval
extrema and diagnostic crossings:

| Seed ordinal2 | Physical outputs | Worst P / equation / velocity / tracer |
| --- | ---: | --- |
| B1 F64 | 2,888 | 0.00148786 / 0.00309342 / 0.00053441 / 0.00168226 |
| K2 F32 | 3,856 | 0.00130346 / 0.00379325 / 0.00054240 / 0.00162939 |
| K4 F64 | 5,792 | 0.00124023 / 0.00285465 / 0.00052562 / 0.00124196 |

Completed evaluations match their pre-capture frozen weights, selected policies
and the exact original reserved bank. Six models / 25,072 physical outputs now
have full held-out evaluations and independent raw audits: five accuracy passes
and the retained original B1 control failure. The same live serial owner is
evaluating ordinal3 B1, followed by ordinal3 K2/K4 and original K3. No weight,
threshold or primary changes occur after reserved data are opened. Fresh
reserved timing, application and boundary tests remain required; goal active.

## Three two-step seeds pass held-out accuracy; eight of ten models complete

Third-seed B1 F64 passes all eight reserved scenes and own240: worst errors
P0.0020877044 / equation0.0038248141 / velocity0.0006727435 /
tracer0.0017314413. Its independent raw audit checks all 2,888 physical outputs,
criteria, interval summaries and drift crossings. Its six earlier development
equation failures remain negative; the new reserved pass does not overwrite
that evidence.

Third-seed K2 F32 also passes all eight reserved scenes and own240. The
[independent raw audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s3-F32-reserved-raw-accuracy-audit-01.json)
verifies all 3,856 physical outputs and unchanged criteria/summary/crossing
decisions, with exact pre-capture weights and policy bindings. Worst errors are
P0.0027350353 / equation0.0035699368 / velocity0.0007814332 /
tracer0.0013142709.

All three selected K2 F32 seeds now pass actual development and original reserved
accuracy. These continuations share frozen component pretraining; no independent
pretraining or primary reselection is claimed. Eight models / 31,816 physical
outputs are now fully evaluated and independently audited on the reserved bank:
seven passes and the original B1 control's retained late equation failure.
Final K4seed3 and original K3 remain in the existing serial queue. Fresh reserved
CPU costs, application and boundary studies remain pending, so the overall goal
stays active and the user's tighter targets remain deferred.

## All ten held-out evaluations complete; original K2/K3/K4 primaries pass

Final K4seed3 remains finite/full240 but fails the unchanged equation criterion:
worst0.0040399529. Five own-history records fail at steps223–227 on
`temporal-g2-cpm16-r3` (15×6×8,16cells/metre). Pressure0.0025598648,
velocity0.0007445593 and tracer0.0010999136 all pass. The independent
[raw audit](../../../runs/full-ai-next-priorities-v1/b4-fixed-viscosity-s3-reserved-raw-accuracy-audit-01.json)
verifies all5,792outputs, positive/negative decisions, interval extrema, drift
crossings and exact frozen weights/policy. Its development pass stays separate;
no threshold change or replacement seed is introduced.

Original K3 completes all eight reserved scenes with all horizons/fullown240
criteria passing in the registered report. Worst errors are P0.0015020930 /
equation0.0030412214 / velocity0.0004948772 / tracer0.0009880362.
Its independent raw check is deferred during CPU timing; the queued completion
auditor must verify every raw K3 record after numerical work finishes.

The [complete reserved report](temporal-reserved-accuracy-results.json) verifies
all1,928native archives and all ten actual frozen policies. Eight models pass;
original B1 and K4seed3 fail late equation checks. All original requested
K2/K3/K4 primaries pass all-field/own240 accuracy without reselection. All three
K2 seeds pass; two of three K4 seeds pass. Nine models /37,608physical outputs
have independent per-model raw audits; K3 adds4,824outputs for the complete audit.

The evaluation owner completes/reaps and the existing final controller starts
fresh reserved CPU timing for all ten models, including failures: planned
27variants,135responses and8,640blocks across five balanced serial repeats.
Numerical audits and plotting stay idle during timing. Fresh speed, full
application and supported-boundary episodes remain pending. The goal stays
active, with all original thresholds/primaries preserved and tighter targets
deferred.

## Fresh reserved CPU targets pass for all original requested primaries

The reserved campaign completes all135responses /8,640timed blocks and all ten
numerical workers successfully exit/reap. The published
[fresh cost report](temporal-reserved-cost-results.json) joins actual frozen
policies, every-output/own240 reserved accuracy and five balanced CPU repeats:

| Original primary | Full block ms | ms/generated state | Worst reserved error | Combined median target |
| --- | ---: | ---: | ---: | --- |
| K2 F32 | 3.539019 | 1.769510 | 0.0031881576 | True |
| K3 F64 | 5.526574 | 1.842191 | 0.0030412214 | True |
| K4 F64 | 7.097961 | 1.774490 | 0.0028193108 | True |

These are original mixed-small-grid warm medians, not maximum latency,32cubed
speed or application throughput. Every added seed remains reported: all three
K2seeds pass combined reserved targets; K4ordinals1/2pass and ordinal3fails
accuracy. Original B1 control also fails reserved accuracy. There is no
post-capture primary reselection or tolerance change.

The existing controller immediately starts the seven-method steady/forcing
application campaign. Further numerical audits and plotting remain idle during
its timing windows. The independent reserved raw-timing check and raw K3 record
verification are still required after CPU work; the queued full completion
auditor must verify every original requirement, including application and
supported-boundary continuation. Reported model benchmarks meet the original
metric targets, while complete goal verification and the tighter follow-on
program remain pending.


## Original goal complete; tighter follow-on activated

The original plan is complete and the thread goal is marked complete after an
independent review of all29original requirement texts, every final evidence and
publication hash, raw timing statistics,42432reserved physical outputs,
560timed application episodes and32supported-boundary episodes. The final
figures are visually inspected and readable. The path-bookkeeping failure is
preserved: version2reuses its one validated case and runs only31remaining
cases. No training, reference capture or CPU timing is replayed. The repaired
publisher snapshots the prior interim report and publishes the complete proof
using the stronger audit and readable renderer. All unfavorable model outcomes
and the native C107 rejection remain reported.

[Independent completion review](goal-completion-review-01.json) ·
[Final requirement audit](completion-audit.md) · [Final results](results.html).

The user-authorized follow-on is now active in
[the tighter-target activation](../full-ai-tight-targets-v1/activation-01.json):
full-state K1/2/3/4at maximum1.2ms/generatedstep and worst1e-5, provisionally
using32cubedas reference from the user's example and retaining smaller-grid
regressions. These new targets are not yet achieved. Higher-density50/100data
and600-step drift optimization remain behind their original ordered gates.
